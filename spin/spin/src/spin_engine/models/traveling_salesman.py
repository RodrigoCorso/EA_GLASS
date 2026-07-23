import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable, cast
from .base import BaseSpinSystem


class TravelingSalesmanSystem(BaseSpinSystem):
    def __init__(
        self,
        cost_matrix: Union[tf.Tensor, np.ndarray],
        lattice_replicas: int,
        constraint_strength: float = 10.0,
        distance_strength: float = 1.0,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        """
        Args:
            cost_matrix (LxL): Matrix defining distances between nodes.
            lattice_replicas: Number of parallel simulations.
            constraint_strength (A): Penalty for invalid paths (must be > max(cost)).
            distance_strength (B): Multiplier for the distance cost.
        """
        # Validate cost matrix shape
        self.cost_matrix = tf.convert_to_tensor(cost_matrix, dtype=tf.float32)
        L = self.cost_matrix.shape[0]
        L = cast(int, L)

        # Initialize Base.
        # Note: We treat the L x L grid as a 2D lattice of dimensions L, L
        super().__init__(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=lattice_replicas,
            quenched_replicas=1,
            initial_spin_state=initial_spin_state
        )

        self.A = tf.constant(constraint_strength, dtype=tf.float32)
        self.B = tf.constant(distance_strength, dtype=tf.float32)

    # def initialize_state_legacy(self) -> tf.Tensor:
    #     """
    #     Initializes random spins.
    #     """
    #     # Start with random -1/+1 spins
    #     full_shape = [self.lattice_replicas,
    #                   self.lattice_length, self.lattice_length]
    #     rand = tf.random.uniform(full_shape)
    #     return tf.where(rand > 0.5, 1.0, -1.0)

    def initialize_state(self) -> tf.Tensor:
        """
        Initializes valid Permutation Matrices for TSP.

        Guarantees:
        1. Each row has exactly one +1.0 (Each city visited once).
        2. Each column has exactly one +1.0 (One city per time step).
        3. All other values are -1.0.
        """
        noise = tf.random.uniform(
            (self.quenched_replicas, self.lattice_replicas, self.lattice_length),
            dtype=tf.float32
        )
        random_permutation_indices = tf.argsort(noise, axis=2)

        spin_state = tf.one_hot(
            random_permutation_indices,
            depth=self.lattice_length,
            dtype=tf.float32
        )

        spin_state = 2.0 * spin_state - 1.0

        return spin_state

    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        """
        Computes H = H_constraints + H_distance


        --- H_A: Constraints ---
        Sum over columns (axis 2: time steps) -> Should be 1
        Sum over rows (axis 1: cities) -> Should be 1
        A topology penalty could be added if the interaction matrix between cities is not fully connected.
            However, the if added a distance cost great enought, we would not need it.
        --- H_B: Distance Cost ---
        The distance cost is the sum of all distances between nodes
        """
        if spin_state is None:
            spin_state = self.spin_state

        spin_state = tf.divide(tf.add(spin_state, 1.0), 2.0)

        row_sums = tf.reduce_sum(spin_state, axis=3)

        col_sums = tf.reduce_sum(spin_state, axis=2)

        row_penalty = tf.reduce_sum(tf.square(1.0 - row_sums), axis=2)
        col_penalty = tf.reduce_sum(tf.square(1.0 - col_sums), axis=2)

        term_A = self.A * (row_penalty + col_penalty)

        spin_state_next = tf.roll(spin_state, shift=-1, axis=3)

        step_correlation = tf.matmul(
            spin_state, spin_state_next, transpose_b=True)

        dist_cost = tf.reduce_sum(
            step_correlation * self.cost_matrix, axis=[2, 3])

        term_B = self.B * dist_cost

        return term_A + term_B
