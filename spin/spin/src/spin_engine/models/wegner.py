# type: ignore
import tensorflow as tf
import itertools
from typing import Optional, Union, Callable, List
from .base import BaseSpinSystem


class WegnerSystem(BaseSpinSystem):
    """
    Implements the Standard Wegner Model (Z2 Lattice Gauge Theory).

    The energy is defined by the Wilson Action: sum over all elementary 
    4-link plaquettes in every plane.

    Hamiltonian:
        H = -J * sum_{p} ( prod_{l in p} sigma_l )

    Tensor Shape:
        (Replicas, L_1, ..., L_D, D)
        The last dimension stores the D links originating from the site r in direction mu.
        - spin_state[..., mu] corresponds to the link U_mu(r)
    """

    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        lattice_dim: int = 2,
        interaction_strength: float = 1.0,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
        **kwargs
    ):
        # Base class calculates shape as [L]*D.
        super().__init__(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            quenched_replicas=1,
            initial_spin_state=initial_spin_state
        )
        self.interaction_strength = tf.constant(
            interaction_strength, dtype=tf.float32)

    def initialize_state(self) -> tf.Tensor:
        """
        Initializes random Z2 links {-1, 1}.
        """
        # Full Shape: [Quenched, Replicas, L, ..., L, D]
        # We append lattice_dim to the spatial shape
        full_shape = [self.quenched_replicas, self.lattice_replicas] + self.shape + [self.lattice_dim]

        rand = tf.random.uniform(full_shape, dtype=tf.float32)
        spin_state = tf.where(rand > 0.5, 1.0, -1.0)

        return spin_state

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the total energy (Wilson Action).
        Sum over all plaquettes in all planes (mu, nu).
        """
        if spin_state is None:
            spin_state = self.spin_state

        plaquettes = self.compute_all_plaquettes(spin_state)

        # Sum over all sites (spatial dims 2..D+1) and all planes (last dim)
        # Result shape: (Q, R)
        # Note: plaquettes tensor has shape (Q, R, L..., L, Num_Planes)
        spatial_axes = list(range(2, self.lattice_dim + 2))
        # Sum over spatial dimensions first
        sum_spatial = tf.reduce_sum(plaquettes, axis=spatial_axes)
        # Sum over the planes (mu-nu pairs)
        total_plaquette_sum = tf.reduce_sum(sum_spatial, axis=-1)

        return tf.multiply(self.interaction_strength, -total_plaquette_sum)

    @tf.function
    def compute_all_plaquettes(self, spin_state: tf.Tensor) -> tf.Tensor:
        """
        Calculates the value of every plaquette in the lattice.
        Returns tensor of shape (Replicas, L, ..., L, Num_Planes)
        where Num_Planes = D*(D-1)/2.
        """
        # We iterate over all unique pairs of dimensions (mu, nu) with mu < nu
        # E.g., in 3D: (0,1) [xy], (0,2) [xz], (1,2) [yz]

        plaquette_terms = []

        for mu, nu in itertools.combinations(range(self.lattice_dim), 2):
            # The tensor axes for spatial dimensions are shifted by 2 (index 0 is Q, 1 is R)
            # axis_mu corresponds to the spatial dimension mu
            axis_mu = mu + 2
            axis_nu = nu + 2

            # --- Link 1: U_mu(r) ---
            # Standard link at r pointing in mu
            U_mu = spin_state[..., mu]

            # --- Link 2: U_nu(r + mu) ---
            U_nu_shift_mu = tf.roll(
                spin_state[..., nu], shift=-1, axis=axis_mu)

            # --- Link 3: U_mu(r + nu) ---
            U_mu_shift_nu = tf.roll(
                spin_state[..., mu], shift=-1, axis=axis_nu)

            # --- Link 4: U_nu(r) ---
            # Standard link at r pointing in nu
            U_nu = spin_state[..., nu]

            # Plaquette = U_mu(r) * U_nu(r+mu) * U_mu(r+nu) * U_nu(r)
            p_mu_nu = U_mu * U_nu_shift_mu * U_mu_shift_nu * U_nu

            plaquette_terms.append(p_mu_nu)

        if not plaquette_terms:
            return tf.zeros(tf.shape(spin_state)[:-1] + [1])

        return tf.stack(plaquette_terms, axis=-1)
