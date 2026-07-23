from spin_engine.models.traveling_salesman import TravelingSalesmanSystem

import pytest
import tensorflow as tf
import numpy as np


class TestTSPMeasurements:

    @pytest.fixture
    def tsp_graph_5(self):
        """
        Fixture with a simple geometry of 5 nodes.
        Nodes at (0,0), (0, 1), (0.5, 1.5), (1, 1), (1, 0).
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

    def test_energy_computation(self, tsp_graph_5):
        """
        Initialize the fixture for the given graph with the 42 seed and compute the energy.
        """
        tf.random.set_seed(42)

        system = TravelingSalesmanSystem(
            cost_matrix=tsp_graph_5,
            lattice_replicas=1,
            distance_strength=1.0,
            constraint_strength=10.0
        )

        system.initialize_state()

        energy = system.compute_energy()

        assert energy.shape == (1, 1)

        assert energy[0, 0] > 0.0  # type: ignore

        assert tf.math.is_finite(energy[0, 0])  # type: ignore
