import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable
from .base import BaseSpinSystem


class EdwardsAndersonSystem(BaseSpinSystem):
    """
    Edwards-Anderson Model for Spin Glass Systems.

    This model implements discrete Ising spins ({-1, 1}) on a d-dimensional lattice
    with quenched random couplings (J_ij). The energy computation handles an arbitrary 
    interaction matrix, which represents the disorder and frustration inherent in 
    spin glass systems.

    Args:
        lattice_length (int): The size of each dimension in the lattice.
        lattice_replicas (int): The number of independent replicas to simulate in parallel.
        interaction_matrix (Union[tf.Tensor, np.ndarray]): The specific coupling matrix (J_ij)
            dictating the interaction strength between spins.
        initial_magnetization (float, optional): Sets the probability of spins initializing to +1.
            Defaults to 0.5 (random initialization).
        lattice_dim (int, optional): Number of spatial dimensions. Defaults to 2.
        initial_spin_state (Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]], optional):
            Pre-defined spin states to initialize with. Defaults to None.
    """

    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        quenched_variable_replicas: int = 1,
        initial_magnetization: float = 0.5,
        lattice_dim: int = 2,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        self.initial_magnetization = initial_magnetization

        super().__init__(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            quenched_replicas=quenched_variable_replicas,
            initial_spin_state=initial_spin_state
        )

        self.interaction_matrix = self._validate_tensor_shape(
            interaction_matrix,
            expected_shape=(self.quenched_replicas,) + tuple(self.shape + self.shape),
            name="Interaction matrix",
        )

    def initialize_state(self) -> tf.Tensor:
        """
        Discrete spins {-1, 1} based on initial magnetization.
        """
        p_up = 0.5 + 0.5 * tf.tanh(self.initial_magnetization)

        full_shape = [self.quenched_replicas, self.lattice_replicas] + self.shape
        rand_vals = tf.random.uniform(full_shape, dtype=tf.float32)

        spin_state = tf.cast(rand_vals < p_up, tf.int8)
        spin_state = 2* spin_state - 1

        return spin_state

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state

        Q = self.quenched_replicas
        N = self.number_spins
        R = self.lattice_replicas

        # Flatten spins: (Q, R, N)
        spin_state_flat = tf.cast(tf.reshape(spin_state, (Q, R, -1)), tf.int32)
        interaction_matrix_flat = tf.cast(tf.reshape(self.interaction_matrix, (Q, N, N)), tf.int32)
        h_local = tf.einsum('qrn,qnm->qrm', spin_state_flat, interaction_matrix_flat)

        pairwise = -0.5 * tf.cast(tf.reduce_sum(spin_state_flat * h_local, axis=2), tf.float32)

        return pairwise

    def compute_delta_energy(
        self,
        spin_state: tf.Tensor,
        updated_spin_state: tf.Tensor,
        changed_indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Efficient ΔE for Edwards-Anderson spin flips.

        Same formula as Ising but without external field:
            ΔE = 2 Σ_k σ_{n_k} h_{n_k}
                 - 2 Σ_{i,j∈D} J_{ij} σ_i σ_j   (cross-term for multi-flip)

        For single-site flips (num_flips=1), the cross-term vanishes (J_nn=0).

        Args:
            spin_state: Current spin state, shape (replicas, L, ..., L).
            updated_spin_state: Proposed spin state (unused, indices suffice).
            changed_indices: Flat indices of flipped sites, shape (replicas, num_flips).

        Returns:
            tf.Tensor of shape (replicas,) — the energy difference E_new - E_old.
        """
        N = tf.cast(self.number_spins, tf.int32)
        num_flips = tf.shape(changed_indices)[2]
        Q = self.quenched_replicas
        R = self.lattice_replicas

        # Flatten spins: (Q, R, N)
        spin_flat = tf.reshape(spin_state, (Q, R, -1))

        # Flatten J: (Q, N, N)
        J_flat = tf.reshape(self.interaction_matrix, (Q, N, N))

        # Gather J rows for flipped sites
        flat_idx = tf.reshape(changed_indices, (Q, R * num_flips))
        J_rows = tf.gather(J_flat, flat_idx, batch_dims=1)
        J_rows = tf.cast(tf.reshape(J_rows, (Q, R, num_flips, N)),tf.int8)

        # Local fields h_n = Σ_j J_{nj} σ_j
        h_local = tf.reduce_sum(
            J_rows * spin_flat[:, :, tf.newaxis, :], axis=-1
        )  # (Q, R, num_flips)

        # Spin values at flipped sites
        s_flipped = tf.gather(spin_flat, changed_indices, batch_dims=2)

        # Term 1: Σ_k 2 * σ_{n_k} * h_{n_k}
        term1 = tf.reduce_sum(
            2 * s_flipped * h_local, axis=-1
        )  # (Q, R)

        # Term 2: Cross-term for multi-site flips
        J_sub = tf.gather(J_rows, changed_indices, batch_dims=2, axis=3)
        term2 = -2 * tf.reduce_sum(
            s_flipped[:, :, :, tf.newaxis] * J_sub * s_flipped[:, :, tf.newaxis, :],
            axis=[-2, -1]
        )  # (Q, R)

        return tf.cast(term1 + term2, tf.float32)
