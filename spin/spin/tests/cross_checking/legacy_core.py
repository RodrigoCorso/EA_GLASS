# type: ignore

import tensorflow as tf
import numpy as np
from typing import Union, Optional, Callable, Tuple, Literal, Dict


class SpinSystem(tf.Module):
    def __init__(
        self,
        lattice_dim: int,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        initial_spin_state: Optional[Union[tf.Tensor, np.ndarray]] = None,
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        model: Literal["ising", "spherical", "z2_gauge"] = "ising",
        spherical_constraint: bool = False,
        initial_magnetization: float = 0.5,
    ) -> None:

        if model not in ("ising", "spherical", "z2_gauge"):
            raise ValueError(
                f"Invalid model: {model}. Please select one of 'ising', 'spherical', or 'z2_gauge'."
            )

        if model != "spherical" and spherical_constraint:
            raise ValueError(
                "Spherical constraint can only be applied to spherical model.")

        if model == "z2_gauge" and lattice_dim != 2:
            raise ValueError(
                "Lattice dimension must be 2 for Z2 gauge model."
            )

        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.lattice_replicas = lattice_replicas
        self.shape = [lattice_length] * lattice_dim
        self.number_spins = tf.cast(lattice_length ** lattice_dim, tf.float32)
        self.model = model
        self.spherical_constraint = spherical_constraint
        self.initial_magnetization = initial_magnetization

        self.interaction_matrix = self._validate_tensor_shape(
            interaction_matrix,
            expected_shape=(
                3, *self.shape, *self.shape) if self.model == "z2_gauge" else self.shape + self.shape,
            name="Interaction matrix",
        )

        self.external_field = self._validate_tensor_shape(
            external_field,
            expected_shape=self.shape,
            name="External field",
            allow_none=True,
            default=tf.zeros(self.shape, dtype=tf.float32),
        )

        self.spin_state = self._validate_tensor_shape(
            initial_spin_state,
            expected_shape=(self.lattice_replicas, *self.shape,
                            2) if self.model == "z2_gauge" else (self.lattice_replicas, *self.shape),
            name="Initial spin state",
            allow_none=True,
            default=lambda: tf.Variable(
                self._initialize_spins_state(), trainable=True
            ),
        )

        if self.model == "z2_gauge":
            self.plaquette = tf.Variable(
                self.compute_plaquette(), trainable=False)
            self.energy = tf.Variable(tf.reduce_sum(
                self.plaquette, axis=[1, 2]), trainable=False)
        else:
            self.energy = tf.Variable(self.compute_pairwise_energies())

    def _validate_tensor_shape(
        self,
        tensor: Optional[tf.Tensor],
        expected_shape: tuple[int, ...],
        name: str,
        allow_none: bool = False,
        default: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Convert input to tf.Tensor and validate its shape.
        If None is allowed, returns a default if provided.
        """
        if tensor is None:
            if allow_none:
                if callable(default):
                    return default()
                elif default is not None:
                    return default
                else:
                    return None
            else:
                raise ValueError(f"{name} cannot be None")

        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must be shape {expected_shape}, got {tensor.shape}"
            )
        return tensor

    @tf.function
    def _apply_spherical_constraint(self, spin_state: tf.Tensor) -> tf.Tensor:
        """
        Applies the spherical constraint independently to each replica.
        """
        original_shape = tf.shape(spin_state)
        spin_state_flat_replicas = tf.reshape(
            spin_state, (self.lattice_replicas, -1))

        normalized_flat = tf.math.l2_normalize(
            spin_state_flat_replicas, axis=1)

        normalized_spins = tf.reshape(normalized_flat, original_shape)

        return tf.sqrt(self.number_spins) * normalized_spins

    def _initialize_spins_state(self) -> tf.Tensor:
        if self.model == "ising":
            p_up = 0.5 + 0.5 * tf.tanh(self.initial_magnetization)
            spin_state = tf.cast(tf.random.uniform(
                self.shape) < p_up, tf.float32)
            spin_state = 2 * spin_state - 1
        elif self.model == "spherical":
            # TODO: We could implement an argument that enables a different initial distribution for spins
            spin_state = tf.random.normal(
                self.shape, mean=self.initial_magnetization, stddev=1.0)
        elif self.model == "z2_gauge":
            p_up = 0.5 + 0.5 * tf.tanh(self.initial_magnetization)
            spin_state_horizontal = tf.cast(tf.random.uniform(
                self.shape) < p_up, tf.float32)
            spin_state_vertical = tf.cast(tf.random.uniform(
                self.shape) < p_up, tf.float32)

            spin_state_horizontal = 2 * spin_state_horizontal - 1
            spin_state_vertical = 2 * spin_state_vertical - 1

            spin_state = tf.stack(
                [spin_state_horizontal, spin_state_vertical], axis=-1)

        expanded_spin_state = tf.expand_dims(spin_state, axis=0)

        multiples = [self.lattice_replicas] + [1] * \
            (len(expanded_spin_state.shape) - 1)

        spin_state = tf.tile(expanded_spin_state, multiples)

        if self.spherical_constraint:
            spin_state = self._apply_spherical_constraint(spin_state)

        return spin_state

    @tf.function
    def compute_pairwise_energies(
        self,
        spin_state: Optional[tf.Tensor] = None,
        interaction_matrix: Optional[tf.Tensor] = None,
        external_field: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:

        if self.model == "z2_gauge":
            raise ValueError(
                "Cannot compute pairwise energy for Z2 gauge model")

        if spin_state is None:
            spin_state = self.spin_state
        if interaction_matrix is None:
            interaction_matrix = self.interaction_matrix
        if external_field is None:
            external_field = self.external_field

        spin_state_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))

        interaction_matrix_flat = tf.reshape(
            interaction_matrix, (self.number_spins, self.number_spins))

        external_field_flat = tf.reshape(
            external_field, (1, -1))
        h = spin_state_flat @ interaction_matrix_flat
        pairwise = -0.5 * tf.reduce_sum(spin_state_flat * h, axis=1)
        field_term = -tf.reduce_sum(spin_state_flat *
                                    external_field_flat, axis=1)

        return pairwise + field_term

    @tf.function
    def _compute_pairwise_energy_deltas(
        self,
        spin_flat: tf.Tensor,
        updated_spin_flat: tf.Tensor,
        disturbed_idx: tf.Tensor
    ) -> tf.Tensor:

        replicas = tf.shape(spin_flat)[0]
        num_flips = tf.shape(disturbed_idx)[1]
        num_spins = tf.cast(self.number_spins, tf.int32)

        batch_idx = tf.repeat(tf.range(replicas)[:, None], num_flips, axis=1)
        gather_indices = tf.stack([batch_idx, disturbed_idx], axis=-1)
        delta_sigma = tf.gather_nd(
            updated_spin_flat - spin_flat, gather_indices)

        interaction_matrix_flat = tf.reshape(
            self.interaction_matrix, (num_spins, num_spins))

        h = tf.linalg.matmul(
            spin_flat, interaction_matrix_flat)

        h_j = tf.gather(h, disturbed_idx, batch_dims=1)
        term1 = -tf.reduce_sum(delta_sigma * h_j, axis=1)

        J_rows = tf.gather(interaction_matrix_flat, disturbed_idx, axis=0)

        J_sub = tf.gather(J_rows, disturbed_idx, axis=2, batch_dims=1)

        delta_sigma_expanded = tf.expand_dims(
            delta_sigma, axis=1)
        quad_form = tf.matmul(delta_sigma_expanded, tf.matmul(
            J_sub, tf.expand_dims(delta_sigma, axis=-1)))
        term2 = -0.5 * tf.squeeze(quad_form, axis=[1, 2])

        field_flat = tf.reshape(self.external_field, [-1])

        disturbed_fields = tf.gather(field_flat, disturbed_idx)

        field_term = -tf.reduce_sum(delta_sigma * disturbed_fields, axis=1)

        return term1 + term2 + field_term

    @tf.function
    def compute_plaquette(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute the plaquette for Z2 gauge model:
        σ_h = horizontal spins
        σ_v = vertical spins
        J_h, J_v, J_hv = interaction matrices
        Returns: plaquette of shape (replicas, L, L)
        """

        if self.model != "z2_gauge":
            raise ValueError(
                "Cannot compute plaquettes for non Z2 gauge model")

        if spin_state is None:
            spin_state = self.spin_state

        sigma_h = spin_state[..., 0]
        sigma_v = spin_state[..., 1]

        J_h = self.interaction_matrix[0, ...]
        J_v = self.interaction_matrix[1, ...]
        J_hv = self.interaction_matrix[2, ...]

        # flatten spins
        sigma_h_flat = tf.reshape(
            sigma_h, (self.lattice_replicas, self.number_spins, 1))
        sigma_v_flat = tf.reshape(
            sigma_v, (self.lattice_replicas, self.number_spins, 1))

        # flatten interactions
        Jh_flat = tf.reshape(J_h, (self.number_spins, self.number_spins))
        Jv_flat = tf.reshape(J_v, (self.number_spins, self.number_spins))
        Jhv_flat = tf.reshape(J_hv, (self.number_spins, self.number_spins))

        plaquette = tf.reshape(
            (sigma_h_flat * (Jh_flat @ sigma_h_flat)) *
            (Jhv_flat @ (sigma_v_flat * (Jv_flat @ sigma_v_flat))),
            (self.lattice_replicas, self.lattice_length, self.lattice_length)
        )

        return plaquette

    @tf.function
    def compute_magnetizations(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state
        return tf.reduce_mean(tf.reshape(spin_state, (self.lattice_replicas, -1)), axis=1)

    @tf.function
    def compute_magnetization(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state
        return tf.reduce_mean(spin_state)

    @tf.function
    def compute_magnetic_susceptibility(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state
        return tf.math.reduce_variance(spin_state)

    @tf.function
    def compute_overlap_matrix(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute the overlap matrix between all replicas.

        Q_ab = (1/N) * sum_i s_i^a * s_i^b
        Returns: Tensor of shape (replicas, replicas)
        """
        if spin_state is None:
            spin_state = self.spin_state

        spin_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))

        overlap = tf.matmul(spin_flat, spin_flat, transpose_b=True)
        overlap /= self.number_spins

        return overlap

    def compute_wilson_loop(
        self,
        loop_size: int = 1,
        spin_state: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Compute Wilson loops of size loop_size x loop_size for all replicas.

        Args:
            loop_size: Size of the square loop.
            spin_state: Optional spin state tensor (replica, L, L, 2). 
                        Defaults to self.spin_state.

        Returns:
            Tensor of shape (replicas, L - loop_size, L - loop_size)
            with Wilson loop value at each starting site.
        """
        if self.model != "z2_gauge":
            raise ValueError("Wilson loops only defined for z2_gauge model.")

        if spin_state is None:
            spin_state = self.spin_state

        L = self.lattice_length

        sigma_h = spin_state[..., 0]
        sigma_v = spin_state[..., 1]

        wilson_loops = []
        for i in range(L - loop_size):
            row_loops = []
            for j in range(L - loop_size):
                top = tf.reduce_prod(sigma_h[:, i, j:j+loop_size], axis=-1)
                right = tf.reduce_prod(
                    sigma_v[:, i:i+loop_size, j+loop_size], axis=-1)
                bottom = tf.reduce_prod(
                    sigma_h[:, i+loop_size, j:j+loop_size], axis=-1)
                left = tf.reduce_prod(sigma_v[:, i:i+loop_size, j], axis=-1)

                loop_val = top * right * bottom * left
                row_loops.append(loop_val)
            wilson_loops.append(tf.stack(row_loops, axis=-1))

        wilson_loops = tf.stack(wilson_loops, axis=-2)
        return wilson_loops

    @tf.function
    def flip_spins(self, num_flips: tf.Tensor, spin_state: Optional[tf.Tensor] = None, spin_flat: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Flip `num_flips` random spins in replicas of Ising model."""
        both_none = (spin_state is None) and (spin_flat is None)
        exactly_one = (spin_state is not None) ^ (spin_flat is not None)

        tf.debugging.assert_equal(both_none or exactly_one, True,
                                  message="Either provide no arguments or exactly one of spin_state or spin_flat")

        if spin_state is not None:
            spin_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))
        elif spin_flat is None:
            spin_flat = tf.reshape(
                self.spin_state, (self.lattice_replicas, -1))

        idx = tf.random.uniform(
            shape=(self.lattice_replicas, num_flips),
            maxval=tf.cast(self.number_spins, tf.int32),
            dtype=tf.int32
        )
        replica_idx = tf.repeat(tf.range(self.lattice_replicas)[
                                :, None], num_flips, axis=1)
        scatter_indices = tf.stack([replica_idx, idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 2))

        updates = tf.reshape(
            -tf.gather_nd(spin_flat, scatter_indices),
            (self.lattice_replicas * num_flips,)
        )

        updated = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, updates)

        energy_delta = self._compute_pairwise_energy_deltas(
            spin_flat, updated, idx)

        return updated, energy_delta

    @tf.function
    def rotate_spins(self, num_pairs: tf.Tensor, theta_max: tf.Tensor,
                     spin_state: Optional[tf.Tensor] = None, spin_flat: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Rotate `num_pairs` pairs of spins by random angles in [-theta_max, theta_max] for each replica."""

        both_none = (spin_state is None) and (spin_flat is None)
        exactly_one = (spin_state is not None) ^ (spin_flat is not None)

        tf.debugging.assert_equal(both_none or exactly_one, True,
                                  message="Either provide no arguments or exactly one of spin_state or spin_flat")

        if spin_state is not None:
            spin_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))
        elif spin_flat is None:
            spin_flat = tf.reshape(
                self.spin_state, (self.lattice_replicas, -1))

        num_spins = tf.cast(self.number_spins, tf.int32)

        all_indices = tf.stack([
            tf.random.shuffle(tf.range(num_spins))[:2 * num_pairs]
            for _ in range(self.lattice_replicas)
        ], axis=0)

        idx1 = all_indices[:, :num_pairs]
        idx2 = all_indices[:, num_pairs:]

        replica_idx = tf.repeat(tf.range(self.lattice_replicas)[
                                :, None], num_pairs, axis=1)

        gather_indices_i = tf.stack([replica_idx, idx1], axis=-1)
        gather_indices_j = tf.stack([replica_idx, idx2], axis=-1)

        sigma_i = tf.gather_nd(spin_flat, gather_indices_i)
        sigma_j = tf.gather_nd(spin_flat, gather_indices_j)

        theta = tf.random.uniform(
            [self.lattice_replicas, num_pairs], -theta_max, theta_max)
        cos_t, sin_t = tf.cos(theta), tf.sin(theta)

        new_i = cos_t * sigma_i - sin_t * sigma_j
        new_j = sin_t * sigma_i + cos_t * sigma_j

        updates = tf.concat([new_i, new_j], axis=1)
        disturbed_idx = tf.concat([idx1, idx2], axis=1)

        scatter_replica_idx = tf.repeat(tf.range(self.lattice_replicas)[
                                        :, None], 2 * num_pairs, axis=1)
        scatter_indices = tf.stack(
            [scatter_replica_idx, disturbed_idx], axis=-1)

        updated_flat = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, updates)

        energy_delta = self._compute_pairwise_energy_deltas(
            spin_flat, updated_flat, disturbed_idx)

        return updated_flat, energy_delta

    @tf.function
    def flip_links(self, num_flips: tf.Tensor, spin_state: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Randomly flip num_flips gauge links per replica for Z2 gauge model."""
        if spin_state is None:
            spin_state = self.spin_state

        L = tf.cast(self.lattice_length, tf.int32)
        num_sites = tf.cast(self.number_spins, tf.int32)

        idx_sites = tf.stack([
            tf.random.shuffle(tf.range(num_sites))[:num_flips]
            for _ in range(self.lattice_replicas)
        ], axis=0)

        i_coords = idx_sites // L
        j_coords = idx_sites % L

        sublattice_idx = tf.random.uniform(
            shape=(self.lattice_replicas, num_flips),
            minval=0, maxval=2, dtype=tf.int32
        )

        replica_idx = tf.repeat(tf.range(self.lattice_replicas)[
                                :, None], num_flips, axis=1)
        gather_indices = tf.stack(
            [replica_idx, i_coords, j_coords, sublattice_idx], axis=-1)

        gathered = tf.gather_nd(spin_state, gather_indices)
        flipped = -gathered

        new_spin_state = tf.tensor_scatter_nd_update(
            spin_state, gather_indices, flipped)

        new_plaquette = self.compute_plaquette(new_spin_state)
        new_energy = tf.reduce_sum(
            new_plaquette, axis=(1, 2))

        energy_delta = new_energy - self.energy

        return new_spin_state, new_plaquette, energy_delta

    @tf.function
    def _disturb_state(self, num_disturb: Optional[tf.Tensor], theta_max: Optional[tf.Tensor]):
        if self.model == "ising":
            new_spin_flat, energy_delta = self.flip_spins(num_disturb)
            next_spin_state = tf.reshape(new_spin_flat, self.spin_state.shape)
            return next_spin_state, None, energy_delta

        elif self.model == "spherical":
            new_spin_flat, energy_delta = self.rotate_spins(
                num_disturb, theta_max)
            next_spin_state = tf.reshape(new_spin_flat, self.spin_state.shape)

            if self.spherical_constraint:
                next_spin_state = self._apply_spherical_constraint(
                    next_spin_state)

            return next_spin_state, None, energy_delta

        elif self.model == "z2_gauge":
            next_spin_state, next_plaquette, energy_delta = self.flip_links(
                num_disturb)
            return next_spin_state, next_plaquette, energy_delta

    @tf.function
    def metropolis_step(
        self,
        beta: float,
        num_disturb: Optional[tf.Tensor] = 1,
        theta_max: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Perform a single Metropolis update step independently for each replica.
        """
        next_spin_state, next_plaquette, energy_delta = self._disturb_state(
            num_disturb=num_disturb,
            theta_max=theta_max
        )

        prob_accept = tf.exp(-beta * energy_delta)
        random_vals = tf.random.uniform(
            shape=(self.lattice_replicas,), dtype=tf.float32)

        accept = tf.logical_or(
            energy_delta < 0.0,
            random_vals < prob_accept
        )

        if tf.reduce_any(accept):
            # Update spins for accepted replicas
            new_spin_state = tf.where(
                tf.reshape(
                    accept, (-1,) + (1,) * (self.lattice_dim + 1)
                ) if self.model == "z2_gauge"
                else tf.reshape(accept, (-1,) + (1,) * self.lattice_dim),
                next_spin_state,
                self.spin_state
            )
            self.spin_state.assign(new_spin_state)

            # Update energies
            new_energy = tf.where(accept, self.energy +
                                  energy_delta, self.energy)
            self.energy.assign(new_energy)

            # Update plaquette only for z2_gauge
            if self.model == "z2_gauge" and next_plaquette is not None:
                new_plaquette = tf.where(
                    tf.reshape(accept, (-1,) + (1,) * 2),  # (replica, 1, 1)
                    next_plaquette,
                    self.plaquette
                )
                self.plaquette.assign(new_plaquette)

        return self.spin_state

    @tf.function
    def metropolis_sweep(
        self,
        beta: float,
        num_disturb: Optional[tf.Tensor] = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: int = None,
        measurement_granularity: int = 100,
        track_spins: bool = True,
        track_energy: bool = True,
        track_magnetization: bool = True,
        track_overlap: bool = True,
        track_plaquette: bool = False
    ) -> Dict:

        if self.model == "spherical" and theta_max is None:
            raise ValueError(
                "For the spherical model, theta_max must be provided.")

        if self.model != "z2_gauge" and track_plaquette is True:
            raise ValueError(
                "You can only track plaquettes for the z2 gauge model."
            )

        def make_array(track, size):
            # Code smell
            return tf.TensorArray(dtype=tf.float32, size=size) if track else tf.TensorArray(dtype=tf.float32, size=0)

        if sweep_length == None:
            sweep_length = self.number_spins * measurement_granularity

        spin_evolution = make_array(
            track_spins, int(round(sweep_length/measurement_granularity)) + 1)
        energy_evolution = make_array(
            track_energy, int(round(sweep_length/measurement_granularity)) + 1)
        magnetization_evolution = make_array(
            track_magnetization, int(round(sweep_length/measurement_granularity)) + 1)
        overlap_evolution = make_array(
            track_overlap, int(round(sweep_length/measurement_granularity)) + 1)
        plaquette_evolution = make_array(
            track_plaquette, int(round(sweep_length/measurement_granularity)) + 1)

        if track_spins:
            spin_evolution = spin_evolution.write(0, self.spin_state)
        if track_energy:
            energy_evolution = energy_evolution.write(
                0, self.energy)
        if track_magnetization:
            magnetization_evolution = magnetization_evolution.write(
                0, self.compute_magnetizations())
        if track_overlap:
            overlap_evolution = overlap_evolution.write(
                0, self.compute_overlap_matrix())
        if track_plaquette:
            plaquette_evolution = plaquette_evolution.write(0, self.plaquette)

        def body(i, spin_evolution, energy_evolution, magnetization_evolution, overlap_evolution, plaquette_evolution):
            _ = self.metropolis_step(beta, num_disturb, theta_max)

            if i % measurement_granularity == 0:
                j = int(i/measurement_granularity)
                if track_spins:
                    spin_evolution = spin_evolution.write(
                        j + 1, self.spin_state)
                if track_energy:
                    energy_evolution = energy_evolution.write(
                        j + 1, self.energy)
                if track_magnetization:
                    magnetization_evolution = magnetization_evolution.write(
                        j + 1, self.compute_magnetizations())
                if track_overlap:
                    overlap_evolution = overlap_evolution.write(
                        j + 1, self.compute_overlap_matrix())
                if track_plaquette:
                    plaquette_evolution = plaquette_evolution.write(
                        j + 1, self.plaquette
                    )

            return i + 1, spin_evolution, energy_evolution, magnetization_evolution, overlap_evolution, plaquette_evolution

        i = tf.constant(0)
        _, spin_evolution, energy_evolution, magnetization_evolution, overlap_evolution, plaquette_evolution = tf.while_loop(
            lambda i, *_: i < sweep_length,
            body,
            loop_vars=[i, spin_evolution,
                       energy_evolution, magnetization_evolution, overlap_evolution, plaquette_evolution],
        )

        result = {}
        if track_spins:
            result["spin_evolution"] = spin_evolution.stack()
        if track_energy:
            result["energy_evolution"] = energy_evolution.stack()
        if track_magnetization:
            result["magnetization_evolution"] = magnetization_evolution.stack()
        if track_overlap:
            result["overlap_evolution"] = overlap_evolution.stack()
        if track_plaquette:
            result["plaquette_evolution"] = plaquette_evolution.stack()

        return result

    @tf.function
    def multi_temperature_sweep(
        self,
        betas: Union[tf.Tensor, np.ndarray],
        num_disturb: Optional[tf.Tensor] = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: int = 100,
        restore_initial_state: bool = True,
        track_spins: bool = False,
        track_energy: bool = False,
        track_magnetization: bool = True,
        track_overlap: bool = True,
        track_plaquette: bool = False
    ):
        """
        Perform independent Metropolis sweeps for multiple inverse temperatures (betas).
        Returns a dict with evolution of tracked quantities stacked along the first dimension
        corresponding to beta index.
        """
        betas = tf.convert_to_tensor(betas, dtype=tf.float32)
        n_temps = tf.shape(betas)[0]

        def make_array(track, size):
            return tf.TensorArray(dtype=tf.float32, size=size) if track else tf.TensorArray(dtype=tf.float32, size=0)

        spin_array = make_array(track_spins, n_temps)
        energy_array = make_array(track_energy, n_temps)
        mag_array = make_array(track_magnetization, n_temps)
        overlap_array = make_array(track_overlap, n_temps)
        plaquette_array = make_array(track_plaquette, n_temps)

        def body(i, spin_array, energy_array, mag_array, overlap_array, plaquette_array):
            beta = betas[i]

            if restore_initial_state:
                original_spin_state = tf.identity(self.spin_state)

            # Run full sweep
            results = self.metropolis_sweep(
                beta=beta,
                num_disturb=num_disturb,
                theta_max=theta_max,
                sweep_length=sweep_length,
                track_spins=track_spins,
                track_energy=track_energy,
                track_magnetization=track_magnetization,
                track_overlap=track_overlap,
                track_plaquette=track_plaquette,
            )

            if track_spins:
                spin_array = spin_array.write(i, results["spin_evolution"])
            if track_energy:
                energy_array = energy_array.write(
                    i, results["energy_evolution"])
            if track_magnetization:
                mag_array = mag_array.write(
                    i, results["magnetization_evolution"])
            if track_overlap:
                overlap_array = overlap_array.write(
                    i, results["overlap_evolution"])
            if track_plaquette:
                plaquette_array = plaquette_array.write(
                    i, results["plaquette_evolution"])

            if restore_initial_state:
                self.spin_state.assign(original_spin_state)

            return i + 1, spin_array, energy_array, mag_array, overlap_array, plaquette_array

        i0 = tf.constant(0)
        _, spin_array, energy_array, mag_array, overlap_array, plaquette_array = tf.while_loop(
            lambda i, *_: i < n_temps,
            body,
            loop_vars=[i0, spin_array, energy_array,
                       mag_array, overlap_array, plaquette_array],
        )

        result = {"betas": betas}
        if track_spins:
            result["spin_evolution"] = spin_array.stack()
        if track_energy:
            result["energy_evolution"] = energy_array.stack()
        if track_magnetization:
            result["magnetization_evolution"] = mag_array.stack()
        if track_overlap:
            result["overlap_evolution"] = overlap_array.stack()
        if track_plaquette:
            result["plaquette_evolution"] = plaquette_array.stack()

        return result
