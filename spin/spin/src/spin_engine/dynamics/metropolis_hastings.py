import tensorflow as tf
from .base import Dynamics

from typing import Optional, TYPE_CHECKING, Tuple, cast, Dict

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.models.ising import IsingSystem
    from spin_engine.dynamics.tracker import Tracker


class MetropolisHastings(Dynamics):
    """
    Metropolis-Hastings dynamics for the Spin System
    """

    def __init__(self, system: 'BaseSpinSystem') -> None:
        super().__init__(system)
        self.current_energy = tf.Variable(
            system.compute_energy(), trainable=False, name="current_energy")

    def flip_spins(self, num_flips: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Propose spin flips without creating a new full state tensor.
        Returns: scatter_indices, updates, original_spins, updated_energy
        """
        Q = self.system.quenched_replicas
        R = self.system.lattice_replicas

        spin_flat = tf.reshape(self.system.spin_state,
                               (Q, R, -1))

        idx = tf.random.uniform(
            shape=(Q, R, num_flips),
            maxval=tf.cast(self.system.number_spins, tf.int32),
            dtype=tf.int32
        )
        
        q_idx = tf.repeat(tf.range(Q)[:, None, None], R, axis=1)
        q_idx = tf.repeat(q_idx, num_flips, axis=2)

        r_idx = tf.repeat(tf.range(R)[None, :, None], Q, axis=0)
        r_idx = tf.repeat(r_idx, num_flips, axis=2)

        scatter_indices = tf.stack([q_idx, r_idx, idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 3))

        original_spins = tf.gather_nd(spin_flat, scatter_indices)
        updates = tf.reshape(-original_spins, [-1])

        # compute_delta_energy only needs idx
        delta_energy = self.system.compute_delta_energy(
            self.system.spin_state, self.system.spin_state, idx)
        updated_energy = self.current_energy + delta_energy

        return scatter_indices, updates, original_spins, updated_energy

    def _disturb_state(self, num_disturbances: tf.Tensor, theta_max: Optional[tf.Tensor]) -> Tuple[tf.Variable | tf.Tensor, tf.Variable | tf.Tensor]:
        if theta_max is None:
            scatter_indices, updates, _, updated_energy = self.flip_spins(num_disturbances)
            spin_flat = tf.reshape(self.system.spin_state, (self.system.quenched_replicas, self.system.lattice_replicas, -1))
            updated = tf.tensor_scatter_nd_update(spin_flat, scatter_indices, updates)
            updated = tf.reshape(updated, self.system.spin_state.shape)
            return updated, updated_energy
        else:
            if self.system.__class__.__name__ == 'IsingSystem':
                raise TypeError(
                    "Can't perform rotations on Ising Spins. Remove theta_max or use Spherical System")
            return self.system.spin_state, self.current_energy

    # @tf.function
    def step(
        self,
        beta: float,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        if theta_max is not None:
            updated, updated_energy = self._disturb_state(
                num_disturbances=num_disturbances, theta_max=theta_max)
            energy_delta = tf.math.subtract(updated_energy, self.current_energy)
            prob_accept = tf.exp(-tf.multiply(beta, energy_delta))
            random_vals = tf.random.uniform(
                shape=(self.system.quenched_replicas, self.system.lattice_replicas), dtype=tf.float32)
            accept = tf.logical_or(
                tf.less(energy_delta, 0.0),
                random_vals < prob_accept
            )
            
            accept_reshaped = accept
            for _ in range(len(self.system.spin_state.shape) - 2):
                accept_reshaped = tf.expand_dims(accept_reshaped, -1)
                
            new_spin_state = tf.where(
                accept_reshaped,
                updated,
                self.system.spin_state
            )
            self.system.update_state(new_spin_state)
            new_energy = tf.where(accept, updated_energy, self.current_energy)
            self.current_energy.assign(new_energy)
            return None

        # Optimized O(1) memory update for Ising Systems
        scatter_indices, updates, original_spins, updated_energy = self.flip_spins(num_disturbances)

        energy_delta = tf.math.subtract(updated_energy, self.current_energy)

        prob_accept = tf.exp(-tf.multiply(beta, energy_delta))

        random_vals = tf.random.uniform(
            shape=(self.system.quenched_replicas, self.system.lattice_replicas), dtype=tf.float32)

        accept = tf.logical_or(
            tf.less(energy_delta, 0.0),
            random_vals < prob_accept
        )

        accept_expanded = tf.repeat(tf.reshape(accept, [-1]), num_disturbances)
        
        # Select updates only if accepted, otherwise keep original
        final_updates = tf.where(accept_expanded, updates, tf.reshape(original_spins, [-1]))

        spin_flat = tf.reshape(self.system.spin_state, (self.system.quenched_replicas, self.system.lattice_replicas, -1))
        new_spin_state = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, final_updates)
        new_spin_state = tf.reshape(new_spin_state, self.system.spin_state.shape)

        self.system.update_state(new_spin_state)

        new_energy = tf.where(
            accept,
            updated_energy,
            self.current_energy
        )
        self.current_energy.assign(new_energy)
        return None

    # TODO: Fix typing errors here...

    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        sweep_length: int,
        num_disturbances: Optional[tf.Tensor] = None,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        if num_disturbances is None:
            num_disturbances = tf.cast(
                self.system.number_spins, dtype=tf.int32)

        # Call the compiled inner loop that now manages the TensorArrays internally
        final_stacked_tensors = self._run_sweep_loop(
            tracker, beta, sweep_length, num_disturbances, theta_max
        )

        tracker.finalize(final_stacked_tensors)

    @tf.function(jit_compile=True)
    def _run_sweep_loop(
        self,
        tracker: 'Tracker',
        beta: float,
        sweep_length: int,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor]
    ) -> Tuple[tf.Tensor, ...]:
        
        tracking_arrays = tracker.init_run(tf.cast(sweep_length, tf.float32))

        tracking_arrays = tracker.track_initial(self.system, tracking_arrays)

        i0 = tf.constant(1)

        def condition(i, arrays):
            return i <= sweep_length

        def body(i, arrays):
            self.step(beta=beta, num_disturbances=num_disturbances,
                      theta_max=theta_max)
            arrays = tracker.track(i, self.system, arrays)
            return i + 1, arrays

        _, final_arrays = tf.while_loop(
            condition,
            body,
            loop_vars=[i0, tracking_arrays]
        )

        return tuple(arr.stack() for arr in final_arrays)
