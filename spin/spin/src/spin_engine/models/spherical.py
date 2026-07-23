import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable, cast
from .base import BaseSpinSystem


class SphericalSystem(BaseSpinSystem):
    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        initial_magnetization: float = 0.0,
        spherical_constraint: bool = False,
        lattice_dim: int = 2,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        self.initial_magnetization = initial_magnetization
        self.spherical_constraint = spherical_constraint

        super().__init__(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            quenched_replicas=1,
            initial_spin_state=initial_spin_state
        )

        self.interaction_matrix = self._validate_tensor_shape(
            interaction_matrix,
            expected_shape=tuple(self.shape + self.shape),
            name="Interaction matrix",
        )

        self.external_field = self._validate_tensor_shape(
            external_field,
            expected_shape=tuple(self.shape),
            name="External field",
            allow_none=True,
            default=tf.zeros(self.shape, dtype=tf.float32),
        )

    def initialize_state(self) -> tf.Tensor:
        """
        Continuous spins normally distributed.
        """
        full_shape = [self.quenched_replicas, self.lattice_replicas] + self.shape
        spin_state = tf.random.normal(
            full_shape, mean=self.initial_magnetization, stddev=1.0
        )

        if self.spherical_constraint:
            spin_state = cast(tf.Tensor, self._apply_spherical_constraint(
                spin_state))  # Casting because I know this is a Tensor

        return spin_state

    @tf.function
    def _apply_spherical_constraint(self, spin_state: tf.Tensor) -> tf.Tensor:
        """
        Applies the spherical constraint independently to each replica.
        Sum(s_i^2) = N
        """
        original_shape = tf.shape(spin_state)
        spin_state_flat_replicas = tf.reshape(
            spin_state, (self.quenched_replicas, self.lattice_replicas, -1)
        )

        normalized_flat = tf.math.l2_normalize(
            spin_state_flat_replicas, axis=2
        )

        normalized_spins = tf.reshape(normalized_flat, original_shape)

        return tf.sqrt(self.number_spins) * normalized_spins

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state.value()
        
        Q = self.quenched_replicas
        R = self.lattice_replicas
        
        # Flatten spins: (Q, R, N)
        spin_state_flat = tf.reshape(
            spin_state, (Q, R, -1))

        # Flatten interaction matrix: (Q, N, N)
        interaction_matrix_flat = tf.reshape(
            self.interaction_matrix, (Q, self.number_spins, self.number_spins)
        )

        # Flatten field
        external_field_flat = tf.reshape(self.external_field, (1, 1, -1))

        h_local = tf.einsum('qrn,qnm->qrm', spin_state_flat, interaction_matrix_flat)

        pairwise = -0.5 * tf.reduce_sum(spin_state_flat * h_local, axis=2)
        field_term = -tf.reduce_sum(spin_state_flat *
                                    external_field_flat, axis=2)

        return pairwise + field_term

    def compute_delta_energy(
        self,
        spin_state: tf.Tensor,
        updated_spin_state: tf.Tensor,
        changed_indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        ΔE for continuous (spherical) spin perturbations with external field.

        Uses the general formula:
            ΔE = -Σ_{j∈D} Δσ_j (h_j + h^ext_j)  -  ½ Σ_{i,j∈D} J_ij Δσ_i Δσ_j
        where h_j = Σ_i J_ij σ_i is the local field and Δσ_j is continuous.

        Args:
            spin_state: Current spin state, shape (replicas, L, ..., L).
            updated_spin_state: Proposed spin state after perturbation.
            changed_indices: Flat indices of perturbed sites, shape (replicas, num_flips).

        Returns:
            tf.Tensor of shape (replicas,) — the energy difference E_new - E_old.
        """
        N = tf.cast(self.number_spins, tf.int32)
        num_flips = tf.shape(changed_indices)[2]
        Q = self.quenched_replicas
        R = self.lattice_replicas

        # Flatten states: (Q, R, N)
        spin_flat = tf.reshape(spin_state, (Q, R, -1))
        updated_flat = tf.reshape(updated_spin_state, (Q, R, -1))

        # Δσ at changed sites: (Q, R, num_flips)
        delta_sigma = tf.gather(updated_flat, changed_indices, batch_dims=2) \
                    - tf.gather(spin_flat, changed_indices, batch_dims=2)

        # Flatten J: (Q, N, N)
        J_flat = tf.reshape(self.interaction_matrix, (Q, N, N))

        # Gather J rows for perturbed sites
        flat_idx = tf.reshape(changed_indices, (Q, R * num_flips))
        J_rows = tf.gather(J_flat, flat_idx, batch_dims=1)
        J_rows = tf.reshape(J_rows, (Q, R, num_flips, N))

        # Local fields h_j = Σ_i J_ji σ_i
        h_local = tf.reduce_sum(
            J_rows * spin_flat[:, :, tf.newaxis, :], axis=-1
        )  # (Q, R, num_flips)

        # External field at perturbed sites
        field_flat = tf.reshape(self.external_field, [-1])
        h_ext = tf.gather(field_flat, changed_indices)

        # Term 1: -Σ_j Δσ_j (h_j + h^ext_j)
        term1 = -tf.reduce_sum(delta_sigma * (h_local + h_ext), axis=-1)

        # Term 2: -½ Σ_{i,j∈D} J_ij Δσ_i Δσ_j
        J_sub = tf.gather(J_rows, changed_indices, batch_dims=2, axis=3)
        term2 = -0.5 * tf.reduce_sum(
            delta_sigma[:, :, :, tf.newaxis] * J_sub * delta_sigma[:, :, tf.newaxis, :],
            axis=[-2, -1]
        )

        return term1 + term2
