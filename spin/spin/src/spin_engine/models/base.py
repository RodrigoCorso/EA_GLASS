import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable, cast
import abc


class BaseSpinSystem(tf.Module, abc.ABC):
    """
    Abstract base class for all spin systems.
    Encapsulates the lattice state, shape, and common utilities.
    """

    def __init__(
        self,
        lattice_dim: int,
        lattice_length: int,
        lattice_replicas: int,
        quenched_replicas: int = 1,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        super().__init__()
        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.lattice_replicas = lattice_replicas
        self.quenched_replicas = quenched_replicas

        # Derived properties
        self.shape = [lattice_length] * lattice_dim
        self.number_spins = tf.cast(lattice_length ** lattice_dim, tf.float32)

        # Initialize or validate spin state
        self.spin_state = self._initialize_or_validate_state(
            initial_spin_state)

    @abc.abstractmethod
    def initialize_state(self) -> tf.Tensor:
        """
        Generates the initial spin state configuration.
        Must be implemented by subclasses.
        Returns:
            tf.Tensor of shape (quenched_replicas, lattice_replicas, *shape, [components])
        """
        pass

    @abc.abstractmethod
    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the energy of the system for each replica and quenched configuration.
        Args:
            spin_state: Optional tensor to compute energy for. Uses self.spin_state if None.
        Returns:
            tf.Tensor of shape (quenched_replicas, lattice_replicas) containing energy values.
        """
        pass

    def compute_delta_energy(
        self,
        spin_state: tf.Tensor,
        updated_spin_state: tf.Tensor,
        changed_indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Computes the energy change ΔE from a local spin update.

        Uses the general formula:
            ΔE = -Σ_{j∈D} Δσ_j h_j  -  ½ Σ_{i,j∈D} J_ij Δσ_i Δσ_j
        where h_j = Σ_i J_ij σ_i is the local field at site j.

        This default implementation uses the dense interaction matrix and works
        for any model that stores self.interaction_matrix. Models without one
        (e.g., WegnerSystem) fall back to full energy recomputation.

        Args:
            spin_state: Current spin state, shape (quenched, replicas, L, ..., L).
            updated_spin_state: Proposed spin state after flipping.
            changed_indices: Flat indices of flipped sites, shape (quenched, replicas, num_flips).

        Returns:
            tf.Tensor of shape (quenched, replicas) — the energy difference E_new - E_old.
        """
        # Fallback for models without an interaction matrix (e.g., Wegner, TSP)
        if not hasattr(self, 'interaction_matrix'):
            return self.compute_energy(updated_spin_state) - self.compute_energy(spin_state)

        N = tf.cast(self.number_spins, tf.int32)
        num_flips = tf.shape(changed_indices)[2]
        Q = self.quenched_replicas
        R = self.lattice_replicas

        # Flatten states: (Q, R, N)
        spin_flat = tf.reshape(spin_state, (Q, R, N))
        updated_flat = tf.reshape(updated_spin_state, (Q, R, N))

        # Δσ at changed sites: (Q, R, num_flips)
        delta_sigma = tf.gather(updated_flat, changed_indices, batch_dims=2) \
                    - tf.gather(spin_flat, changed_indices, batch_dims=2)

        # Flatten J: (Q, N, N)
        J_flat = tf.reshape(self.interaction_matrix, (Q, N, N))

        # Gather J rows for all flipped sites across Q and R
        flat_idx = tf.reshape(changed_indices, (Q, R * num_flips)) # (Q, R * num_flips)
        J_rows = tf.gather(J_flat, flat_idx, batch_dims=1)         # (Q, R * num_flips, N)
        J_rows = tf.reshape(J_rows, (Q, R, num_flips, N))

        # Local fields h_j = Σ_i J_ji σ_i for each changed site j
        # spin_flat: (Q, R, N) → (Q, R, 1, N)
        h_local = tf.reduce_sum(
            J_rows * spin_flat[:, :, tf.newaxis, :], axis=-1
        )  # (Q, R, num_flips)

        # Term 1: -Σ_j Δσ_j h_j
        term1 = -tf.reduce_sum(delta_sigma * h_local, axis=-1)  # (Q, R)

        # Term 2: -½ Σ_{i,j∈D} J_ij Δσ_i Δσ_j (self-interaction correction)
        # For single-site flips with J_nn = 0 this vanishes, but we compute it
        # for correctness with multi-site flips.
        # Gather the sub-matrix J[D, D]: (Q, R, num_flips, num_flips)
        # Build column gather from J_rows at the changed indices
        J_sub = tf.gather(J_rows, changed_indices, batch_dims=2, axis=3)  # (Q, R, num_flips, num_flips)
        # Δσ: (Q, R, num_flips) → (Q, R, num_flips, 1) and (Q, R, 1, num_flips)
        term2 = -0.5 * tf.reduce_sum(
            delta_sigma[:, :, :, tf.newaxis] * J_sub * delta_sigma[:, :, tf.newaxis, :],
            axis=[-2, -1]
        )  # (Q, R)

        return tf.cast(term1 + term2, tf.float32)

    def update_state(self, updated_spin_state: tf.Tensor) -> None:
        """
        Take a new spin configuration and update the current state.
        Args:
           updated_spin_state: New spin configuration
        Returns:
          None
        """
        self._validate_tensor_shape(updated_spin_state, cast(
            tuple, self.spin_state.shape), "Updated Spin State")
        self.spin_state.assign(updated_spin_state)

    def _initialize_or_validate_state(
        self,
        initial_state: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]]
    ) -> tf.Variable:
        """
        Helper to handle state initialization logic.
        """
        if initial_state is None:
            initial_value = self.initialize_state()
        elif callable(initial_state):
            initial_value = initial_state()
        else:
            initial_value = tf.convert_to_tensor(
                initial_state)

        return tf.Variable(initial_value, trainable=True, name="spin_state")

    # TODO: Review _validate_tensor_shape method
    def _validate_tensor_shape(
        self,
        tensor: Optional[Union[tf.Tensor, np.ndarray]],
        expected_shape: tuple[int, ...],
        name: str,
        allow_none: bool = False,
        default: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Convert input to tf.Tensor and validate its shape.
        """
        if tensor is None:
            if allow_none:
                if callable(default):
                    return default()
                elif default is not None:
                    return default
                # else:
                #     return None
            else:
                raise ValueError(f"{name} cannot be None")

        tensor = tf.convert_to_tensor(tensor)
        if tensor.shape != expected_shape:
            # Check for compatibility (e.g. broadcasting) if strict equality fails?
            # For now strict check as in legacy_core
            raise ValueError(
                f"{name} must be shape {expected_shape}, got {tensor.shape}"
            )
        return tensor
