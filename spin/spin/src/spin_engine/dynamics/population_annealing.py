import tensorflow as tf
from .base import Dynamics

from typing import Optional, TYPE_CHECKING, Tuple, cast, Dict, List

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.models.ising import IsingSystem
    from spin_engine.dynamics.tracker import Tracker


class PopulationAnnealing(Dynamics):
    """
    Population Annealing dynamics for Spin Systems.
    
    Population annealing is a sequential Monte Carlo method that maintains
    a population of replicas and gradually anneals the inverse temperature.
    At each temperature step, replicas are resampled based on their statistical
    weights to maintain the equilibrium distribution.
    
    Key features:
    - Maintains a population of R lattice replicas
    - Anneals beta from beta_min to beta_max in discrete steps
    - Uses multinomial resampling based on Boltzmann weights
    - Tracks the partition function estimate via the cumulative weight factor
    """

    def __init__(self, system: 'BaseSpinSystem', population_size: Optional[int] = None) -> None:
        """
        Initialize Population Annealing dynamics.
        
        Args:
            system: The spin system to simulate
            population_size: Number of replicas in the population. If None, uses
                           the system's lattice_replicas count.
        """
        super().__init__(system)
        self.population_size = population_size if population_size is not None else system.lattice_replicas
        
        # Track the current energy for each replica
        self.current_energies = tf.Variable(
            system.compute_energy(), trainable=False, name="current_energies",
            shape=(system.quenched_replicas, self.population_size)
        )
        
        # Track the cumulative weight factor (for partition function estimation)
        self.cumulative_weight_factor = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="cumulative_weight_factor"
        )
        
        # Track the current beta
        self.current_beta = tf.Variable(
            0.0, trainable=False, dtype=tf.float32, name="current_beta"
        )

    def _compute_weights(self, beta_old: tf.Tensor, beta_new: tf.Tensor) -> tf.Tensor:
        """
        Compute the statistical weights for resampling.
        
        The weight for each replica is proportional to:
        w = exp(-(beta_new - beta_old) * E)
        
        Args:
            beta_old: Previous inverse temperature
            beta_new: New inverse temperature
            
        Returns:
            Tensor of weights with shape (quenched_replicas, population_size)
        """
        delta_beta = beta_new - beta_old
        # weights = exp(-delta_beta * E) for each replica
        weights = tf.exp(-delta_beta * self.current_energies)
        return weights

    def _normalize_weights(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Normalize weights to form a probability distribution.
        
        Args:
            weights: Unnormalized weights
            
        Returns:
            Normalized probabilities that sum to 1
        """
        # Sum over all replicas (both quenched and lattice)
        total_weight = tf.reduce_sum(weights, keepdims=True)
        return weights / total_weight

    def _resample_replicas(self, probabilities: tf.Tensor) -> tf.Tensor:
        """
        Resample replicas based on their probabilities using multinomial resampling.
        
        Args:
            probabilities: Normalized probabilities for each replica
            
        Returns:
            Indices of selected replicas with shape (quenched_replicas, population_size)
        """
        Q = self.system.quenched_replicas
        R = self.population_size
        
        # Flatten probabilities for sampling
        probs_flat = tf.reshape(probabilities, (Q, -1))
        
        # Sample indices for each quenched configuration independently
        sampled_indices_flat = tf.random.categorical(
            tf.math.log(probs_flat + 1e-10),  # Add small value for numerical stability
            num_samples=R,
            dtype=tf.int32
        )
        
        # Convert flat indices back to (quenched, lattice) indices
        # sampled_indices_flat has shape (Q, R)
        # We need to convert these to indices into the original (Q, R) structure
        
        # For each quenched replica, we sample R lattice replicas
        # The flat index i corresponds to lattice replica i within that quenched replica
        return sampled_indices_flat

    def _perform_resampling(self, beta_old: tf.Tensor, beta_new: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform the full resampling step of population annealing.
        
        Args:
            beta_old: Previous inverse temperature
            beta_new: New inverse temperature
            
        Returns:
            Tuple of (new_spin_state, new_energies) after resampling
        """
        # Compute weights based on energy change
        weights = self._compute_weights(beta_old, beta_new)
        
        # Update cumulative weight factor
        # The normalization constant contributes to the partition function estimate
        Q = tf.cast(self.system.quenched_replicas, tf.float32)
        R = tf.cast(self.population_size, tf.float32)
        mean_weight = tf.reduce_mean(weights)
        self.cumulative_weight_factor.assign(
            self.cumulative_weight_factor * mean_weight * (Q * R)
        )
        
        # Normalize to get probabilities
        probabilities = self._normalize_weights(weights)
        
        # Resample replicas
        sampled_indices = self._resample_replicas(probabilities)
        
        # Gather the selected replicas
        # Create indices for gather_nd
        Q = self.system.quenched_replicas
        q_idx = tf.repeat(tf.range(Q)[:, None], self.population_size, axis=1)
        gather_indices = tf.stack([q_idx, sampled_indices], axis=-1)
        
        # Reshape spin state for gathering: (Q, R, ...) -> we gather along axis 1
        # Use tf.gather with batch_dims=1 to gather along the lattice replica dimension
        spin_flat = tf.reshape(
            self.system.spin_state, 
            (Q, self.population_size, -1)
        )
        
        # Gather selected replicas for each quenched configuration
        new_spin_flat = tf.gather(spin_flat, sampled_indices, batch_dims=1)
        new_spin_state = tf.reshape(new_spin_flat, self.system.spin_state.shape)
        
        # Gather energies for selected replicas
        new_energies = tf.gather(self.current_energies, sampled_indices, batch_dims=1)
        
        return new_spin_state, new_energies

    def _metropolis_step(
        self,
        beta: tf.Tensor,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        """
        Perform a single Metropolis-Hastings step on all replicas.
        
        This is used within population annealing to equilibrate at each temperature.
        
        Args:
            beta: Current inverse temperature
            num_disturbances: Number of spin flip attempts per replica
            theta_max: Maximum rotation angle (for continuous spins, optional)
        """
        Q = self.system.quenched_replicas
        R = self.population_size
        
        # Generate random spin indices to flip
        spin_flat = tf.reshape(self.system.spin_state, (Q, R, -1))
        N = tf.cast(self.system.number_spins, tf.int32)
        
        idx = tf.random.uniform(
            shape=(Q, R, num_disturbances),
            maxval=N,
            dtype=tf.int32
        )
        
        # Build scatter indices
        q_idx = tf.repeat(tf.range(Q)[:, None, None], R, axis=1)
        q_idx = tf.repeat(q_idx, num_disturbances, axis=2)
        
        r_idx = tf.repeat(tf.range(R)[None, :, None], Q, axis=0)
        r_idx = tf.repeat(r_idx, num_disturbances, axis=2)
        
        scatter_indices = tf.stack([q_idx, r_idx, idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 3))
        
        # Get original spins and compute proposed updates
        original_spins = tf.gather_nd(spin_flat, scatter_indices)
        updates = tf.reshape(-original_spins, [-1])
        
        # Compute energy change
        delta_energy = self.system.compute_delta_energy(
            self.system.spin_state, self.system.spin_state, idx
        )
        updated_energy = self.current_energies + delta_energy
        
        # Metropolis acceptance criterion
        energy_delta = updated_energy - self.current_energies
        prob_accept = tf.exp(-beta * energy_delta)
        
        random_vals = tf.random.uniform(
            shape=(Q, R), dtype=tf.float32
        )
        
        accept = tf.logical_or(
            tf.less(energy_delta, 0.0),
            random_vals < prob_accept
        )
        
        # Apply accepted updates
        accept_expanded = tf.repeat(tf.reshape(accept, [-1]), num_disturbances)
        final_updates = tf.where(accept_expanded, updates, tf.reshape(original_spins, [-1]))
        
        spin_flat = tf.reshape(self.system.spin_state, (Q, R, -1))
        new_spin_state = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, final_updates
        )
        new_spin_state = tf.reshape(new_spin_state, self.system.spin_state.shape)
        
        # Update system state and energies
        self.system.update_state(new_spin_state)
        
        new_energies = tf.where(accept, updated_energy, self.current_energies)
        self.current_energies.assign(new_energies)

    @tf.function(jit_compile=True)
    def anneal_step(
        self,
        beta_old: tf.Tensor,
        beta_new: tf.Tensor,
        equilibration_steps: tf.Tensor,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        """
        Perform one step of the population annealing algorithm.
        
        This consists of:
        1. Resampling replicas based on their statistical weights
        2. Equilibrating the population at the new temperature via Metropolis steps
        
        Args:
            beta_old: Previous inverse temperature
            beta_new: New inverse temperature
            equilibration_steps: Number of Metropolis steps to perform after resampling
            num_disturbances: Number of spin flips per Metropolis step
            theta_max: Maximum rotation angle (optional, for continuous spins)
        """
        # Step 1: Resample based on weights
        new_spin_state, new_energies = self._perform_resampling(beta_old, beta_new)
        self.system.update_state(new_spin_state)
        self.current_energies.assign(new_energies)
        
        # Update current beta
        self.current_beta.assign(beta_new)
        
        # Step 2: Equilibrate at new temperature
        def step_body(i, _):
            self._metropolis_step(beta_new, num_disturbances, theta_max)
            return i + 1, tf.constant(0.0)
        
        _, _ = tf.while_loop(
            lambda i, _: i < equilibration_steps,
            step_body,
            (tf.constant(0), tf.constant(0.0))
        )

    @tf.function(jit_compile=True)
    def sweep(
        self,
        tracker: 'Tracker',
        beta_schedule: tf.Tensor,
        equilibration_steps: tf.Tensor,
        num_disturbances: Optional[tf.Tensor] = None,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        """
        Perform a full population annealing sweep across a temperature schedule.
        
        Args:
            tracker: Tracker object to record measurements
            beta_schedule: Array of inverse temperatures to anneal through
            equilibration_steps: Number of Metropolis steps between resampling
            num_disturbances: Number of spin flips per Metropolis step. If None,
                            defaults to the number of spins in the system.
            theta_max: Maximum rotation angle (optional, for continuous spins)
        """
        if num_disturbances is None:
            num_disturbances = tf.cast(self.system.number_spins, tf.int32)
        
        num_beta_steps = tf.shape(beta_schedule)[0]
        
        # Initialize tracking arrays
        tracking_arrays = tracker.init_run(tf.cast(num_beta_steps, tf.float32))
        
        # Track initial state (at beta = 0 or first beta value)
        tracking_arrays = tracker.track_initial(self.system, tracking_arrays)
        
        # Anneal through the temperature schedule
        def anneal_body(i, arrays):
            beta_old = beta_schedule[i]
            beta_new = beta_schedule[i + 1]
            
            self.anneal_step(
                beta_old=beta_old,
                beta_new=beta_new,
                equilibration_steps=equilibration_steps,
                num_disturbances=num_disturbances,
                theta_max=theta_max
            )
            
            arrays = tracker.track(i + 1, self.system, arrays)
            return i + 1, arrays
        
        _, final_arrays = tf.while_loop(
            lambda i, _: i < num_beta_steps - 1,
            anneal_body,
            (tf.constant(0), tracking_arrays)
        )
        
        tracker.finalize(final_arrays)

    def get_partition_function_estimate(self) -> tf.Tensor:
        """
        Get the estimate of the partition function from the cumulative weight factor.
        
        In population annealing, the product of mean weights at each step gives
        an estimate of the partition function ratio Z(beta_final) / Z(beta_initial).
        
        Returns:
            Estimate of the partition function ratio
        """
        return self.cumulative_weight_factor

    def reset(self) -> None:
        """
        Reset the population annealing state.
        """
        self.current_energies.assign(self.system.compute_energy())
        self.cumulative_weight_factor.assign(1.0)
        self.current_beta.assign(0.0)
