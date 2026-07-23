from spin_engine.models.traveling_salesman import TravelingSalesmanSystem

import pytest
import tensorflow as tf
from typing import cast


class TestTSPModel:

    @pytest.fixture
    def tsp_system(self):
        """
        Creates a basic TSP system for testing.
        3 Nodes (L=3), 2 Replicas.
        """
        L = 3
        replicas = 2
        cost_matrix = tf.constant([
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0]
        ], dtype=tf.float32)

        return TravelingSalesmanSystem(
            cost_matrix=cast(tf.Tensor, cost_matrix),
            lattice_replicas=replicas
        )

    def test_initialization_constraints(self, tsp_system):
        """
        Test that the initialized state satisfies TSP constraints:
        1. Values are strictly -1 or 1.
        2. When mapped to 0/1 (binary), row sums = 1 (each city visited once).
        3. Column sums = 1 (one city per time step).
        """
        spin_state = tsp_system.initialize_state()

        assert spin_state.shape == (1, 2, 3, 3)

        tf.debugging.assert_near(
            tf.abs(spin_state), tf.ones_like(spin_state), atol=1e-5)

        binary_state = (spin_state + 1.0) / 2.0

        row_sums = tf.reduce_sum(binary_state, axis=3)
        tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-5)

        col_sums = tf.reduce_sum(binary_state, axis=2)
        tf.debugging.assert_near(col_sums, tf.ones_like(col_sums), atol=1e-5)

    def test_matrix_compatibility(self):
        """
        Test that the system correctly infers lattice length from cost matrix
        and validates compatibility.
        """
        L = 4
        cost_matrix = tf.random.uniform((L, L))

        system = TravelingSalesmanSystem(
            cost_matrix=cost_matrix,
            lattice_replicas=5
        )

        assert system.lattice_length == L

        state = system.initialize_state()
        assert state.shape == (1, 5, L, L)
