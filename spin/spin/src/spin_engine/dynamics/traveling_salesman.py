import tensorflow as tf
from .base import Dynamics
from typing import Optional, Tuple, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.dynamics.tracker import Tracker


class TravelingSalesmanDynamics(Dynamics):
    """
    Dynamics specifically designed for TSP optimization.

    Instead of flipping single spins (which breaks TSP constraints),
    this class implements a 'Swap Move' (2-Opt equivalent).

    It selects two random columns (time steps) and swaps them.
    This preserves the constraints:
    - Sum of rows = 1 (Each city visited once)
    - Sum of cols = 1 (One city per time step)
    """

    def __init__(self, system: 'BaseSpinSystem') -> None:
        super().__init__(system)
        # We track current energy to avoid re-computing it every step
        self.current_energy = tf.Variable(
            system.compute_energy(), trainable=False, name="current_energy"
        )

    def reverse_segment(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs a 2-Opt move (Segment Reversal) for each replica.

        Mechanism:
        1. Pick two random indices i and j.
        2. Reverse the order of visits between time step i and time step j.

        Returns:
            Tuple[New Spin State, New Energy]
        """
        L = self.system.lattice_length
        R = self.system.lattice_replicas
        Q = self.system.quenched_replicas

        # --- 1. Identify Segment Bounds ---
        # Generate random noise and pick top 2 indices to ensure they are distinct
        rand_noise = tf.random.uniform((Q, R, L), dtype=tf.float32)
        # Shape: (Q, R, 2)
        random_indices = tf.argsort(rand_noise, axis=2)[:, :, :2]

        # Sort them so we strictly have start < end
        start = tf.reduce_min(random_indices, axis=2)  # Shape: (Q, R)
        end = tf.reduce_max(random_indices, axis=2)   # Shape: (Q, R)

        # --- 2. Build Permutation Map ---
        # We need to create a map where indices inside [start, end] are reversed.
        # Logic: if index k is in [start, end], new_index = start + end - k

        # Create a grid of indices [0, 1, ..., L-1] for every replica
        # Shape: (1, 1, L) -> broadcastable to (Q, R, L)
        indices = tf.range(L, dtype=tf.int32)[None, None, :]

        # Create Boolean Mask for the segment
        # Shape: (Q, R, L)
        # mask is True if index is inside the chosen segment
        mask = (indices >= start[:, :, None]) & (indices <= end[:, :, None])

        # Calculate the reversed indices
        # The math: (start + end) - current_index flips the range [start, end]
        sum_bounds = start + end
        reversed_indices = sum_bounds[:, :, None] - indices

        # Select: If in mask, use reversed index; otherwise keep original index
        final_indices = tf.where(mask, reversed_indices, indices)

        # --- 3. Create New State ---
        # Use gather to apply the permutation to the columns (time steps)
        # axis=3 is columns
        new_state = tf.gather(
            self.system.spin_state,
            final_indices,
            axis=3,
            batch_dims=2
        )

        # --- 4. Compute Energy of New State ---
        new_energy = self.system.compute_energy(new_state)

        return new_state, new_energy

    def swap_stops(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs a swap of two random stops (columns) for each replica.

        Returns:
            Tuple[New Spin State, New Energy]
        """
        L = self.system.lattice_length
        R = self.system.lattice_replicas
        Q = self.system.quenched_replicas

        # --- 1. Identify Columns to Swap ---
        # We need two distinct random indices (columns) per replica.
        # Generating random noise and sorting it gives us a shuffled list of indices.
        rand_noise = tf.random.uniform((Q, R, L), dtype=tf.float32)
        shuffled_indices = tf.argsort(rand_noise, axis=2)

        # Pick the first two indices as the columns to swap
        col_a = shuffled_indices[:, :, 0]  # Shape: (Q, R)
        col_b = shuffled_indices[:, :, 1]  # Shape: (Q, R)

        # --- 2. Build Permutation Map ---
        # We start with identity mapping: [0, 1, 2, ..., L-1]
        base_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(L, dtype=tf.int32), 0), 0),
            [Q, R, 1]
        )  # Shape: (Q, R, L)

        q_range = tf.range(Q, dtype=tf.int32)
        r_range = tf.range(R, dtype=tf.int32)
        qq, rr = tf.meshgrid(q_range, r_range, indexing='ij')

        coords_a = tf.stack([tf.reshape(qq, [-1]), tf.reshape(rr, [-1]), tf.reshape(tf.cast(col_a, tf.int32), [-1])], axis=1)
        coords_b = tf.stack([tf.reshape(qq, [-1]), tf.reshape(rr, [-1]), tf.reshape(tf.cast(col_b, tf.int32), [-1])], axis=1)

        scatter_indices = tf.concat([coords_a, coords_b], axis=0)  # Shape: (2*Q*R, 3)

        updates = tf.concat([
            tf.reshape(tf.cast(col_b, dtype=tf.int32), [-1]),
            tf.reshape(tf.cast(col_a, dtype=tf.int32), [-1])
        ], axis=0)

        # Apply the swap to the permutation map
        perm_map = tf.tensor_scatter_nd_update(
            base_indices, scatter_indices, updates)

        # --- 3. Create New State ---
        # Use gather to permute the columns of the existing state
        # axis=3 is columns (Time steps)
        # batch_dims=2 allows different permutations per replica
        new_state = tf.gather(self.system.spin_state,
                              perm_map, axis=3, batch_dims=2)

        # --- 4. Compute Energy of New State ---
        new_energy = self.system.compute_energy(new_state)

        return new_state, new_energy

    def _disturb_state(
        self,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Internal wrapper to standardize the Dynamics interface.
        For TSP, we ignore 'num_disturbances' > 1 complexity for now 
        and perform exactly one 2-column swap per step.
        """
        # Note: We ignore theta_max as this is a discrete system.
        # return self.swap_stops()
        return self.reverse_segment()

    # @tf.function
    def step(
        self,
        beta: float,
        num_disturbances: tf.Tensor = cast(tf.Tensor, 1),
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        """
        Metropolis-Hastings Step tailored for TSP.
        1. Propose a swap of two stops.
        2. Calculate Energy Delta (dH).
        3. Accept if dH < 0 or with prob exp(-beta * dH).
        """

        updated, updated_energy = self._disturb_state(
            num_disturbances, theta_max)

        # Calculate energy difference
        energy_delta = tf.math.subtract(updated_energy, self.current_energy)

        # Metropolis Acceptance Probability
        prob_accept = tf.exp(-tf.multiply(beta, energy_delta))

        random_vals = tf.random.uniform(
            shape=(self.system.quenched_replicas, self.system.lattice_replicas), dtype=tf.float32
        )

        # Accept if Improvement (delta < 0) OR Random < Probability
        accept = tf.logical_or(
            tf.less(energy_delta, 0.0),
            random_vals < prob_accept
        )

        # Create the new state tensor based on acceptance
        # Shape broadcasting is required for the mask
        accept_mask = tf.reshape(accept, (self.system.quenched_replicas, self.system.lattice_replicas, 1, 1))  # (Q, R, 1, 1)

        new_spin_state = tf.where(
            accept_mask,
            updated,
            self.system.spin_state
        )

        # Update System State
        self.system.update_state(new_spin_state)

        # Update Local Energy Variable
        new_energy_vals = tf.where(
            accept,
            updated_energy,
            self.current_energy
        )
        self.current_energy.assign(new_energy_vals)

        return None

    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        sweep_length: int,
        num_disturbances: tf.Tensor = cast(tf.Tensor, 1),
        theta_max: Optional[tf.Tensor] = None,
    ) -> None:
        """
        Orchestrator for multiple steps (Standard TensorFlow loop).
        """
        # Initialize tracking (Energy, Magnetization, etc.)
        tracking_arrays = tracker.init_run(cast(tf.Tensor, sweep_length))

        # Track initial state (step 0)
        tracking_arrays = tracker.track(
            cast(tf.Tensor, 0), self.system, tracking_arrays
        )

        def body(i, tracking_arrays):
            # Perform one MC step
            _ = self.step(beta, num_disturbances, theta_max)

            # Track new state
            current_step = i + 1
            new_arrays = tracker.track(
                current_step, self.system, tracking_arrays
            )
            return i + 1, new_arrays

        # Run the loop
        i0 = tf.constant(0, dtype=tf.int32)
        loop_result = tf.while_loop(
            cond=lambda i, _: i < sweep_length,
            body=body,
            loop_vars=[i0, tracking_arrays]
        )

        # Finalize tracking
        final_arrays = cast(Tuple, loop_result)[1]
        tracker.finalize(final_arrays)

        return None
