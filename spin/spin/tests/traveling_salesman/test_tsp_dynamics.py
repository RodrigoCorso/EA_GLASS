from spin_engine.dynamics.traveling_salesman import TravelingSalesmanDynamics
from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
import pytest
import tensorflow as tf
import numpy as np


class TestTSPDynamics:

    @pytest.fixture
    def tsp_graph_5(self):
        """
        Fixture with a simple geometry of 5 nodes.
        Nodes at (0,0), (0, 1), (0.5, 1.5), (1, 1), (1, 0).
        Optimal path length is ~4.414.
        """
        coords = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.5],
            [1.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        return tf.convert_to_tensor(dist_matrix, dtype=tf.float32)

    def test_dynamics_evolution(self, tsp_graph_5):
        """
        Evolve for a set of steps and check constraints and energy.
        """
        tf.random.set_seed(42)

        replicas = 2

        system = TravelingSalesmanSystem(
            cost_matrix=tsp_graph_5,
            lattice_replicas=replicas,
            distance_strength=1.0,
            constraint_strength=10.0
        )
        system.initialize_state()

        dynamics = TravelingSalesmanDynamics(system)

        initial_energy = system.compute_energy()

        beta = 10.0
        steps = 100

        for _ in range(steps):
            dynamics.step(beta=beta)

        final_energy = system.compute_energy()

        spin_state = system.spin_state
        binary_state = tf.divide(tf.add(spin_state, 1.0), 2.0)
        row_sums = tf.reduce_sum(binary_state, axis=3)
        col_sums = tf.reduce_sum(binary_state, axis=2)

        tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-5)
        tf.debugging.assert_near(col_sums, tf.ones_like(col_sums), atol=1e-5)

        avg_init = tf.reduce_mean(initial_energy)
        avg_final = tf.reduce_mean(final_energy)

        assert avg_final <= avg_init

        optimal_energy = 4.414
        min_final = tf.reduce_min(final_energy)

        assert min_final >= optimal_energy - 1e-3
