from spin_engine.interactions import (
    DecayingInteraction,
    PeriodicNearestNeighborInteraction,
    CurieWeissInteraction,
    GaussianInteraction
)
import tensorflow as tf
import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


class TestInteractions:
    def test_decaying_interaction(self):
        D = 1
        L = 4
        # alpha=0 means J = J0 (except diagonal)
        interaction = DecayingInteraction(J0=1.0, alpha=0.0)
        J = interaction.generate(D, L)

        # Check shape
        # (L,)*D*2 => (4, 4) in 1D
        assert J.shape == (4, 4)

        # Diagonal should be 0
        np.testing.assert_equal(np.diag(J), np.zeros(4))

        # Off-diagonal should be J0 (1.0) because alpha=0
        J_no_diag = J.copy()
        np.fill_diagonal(J_no_diag, 1.0)
        np.testing.assert_allclose(J_no_diag, np.ones((4, 4)))

    def test_periodic_nn_interaction_1d(self):
        D = 1
        L = 4
        interaction = PeriodicNearestNeighborInteraction()
        J = interaction.generate(D, L)  # Shape (4, 4)

        # In 1D with L=4:
        # 0 is connected to 1 and 3
        # 1 is connected to 0 and 2
        # 2 is connected to 1 and 3
        # 3 is connected to 2 and 0

        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)

        np.testing.assert_array_equal(J, expected)

    def test_periodic_nn_interaction_2d(self):
        D = 2
        L = 3
        interaction = PeriodicNearestNeighborInteraction()
        J = interaction.generate(D, L)  # Shape (3, 3, 3, 3)

        # Check center point (1, 1). Neighbors: (0,1), (2,1), (1,0), (1,2)
        center_interactions = J[1, 1, :, :]

        expected = np.zeros((3, 3))
        expected[0, 1] = 1
        expected[2, 1] = 1
        expected[1, 0] = 1
        expected[1, 2] = 1

        np.testing.assert_array_equal(center_interactions, expected)

    def test_curie_weiss_interaction(self):
        D = 1
        L = 4
        N = L**D
        J0 = 2.0
        interaction = CurieWeissInteraction(J0=J0)
        J = interaction.generate(D, L)

        # Shape (4, 4)
        assert J.shape == (4, 4)

        # Diagonal is 0
        np.testing.assert_equal(np.diag(J), np.zeros(N))

        # Off-diagonal is J0/N = 2.0/4 = 0.5
        expected_val = J0 / N
        J_no_diag = J.copy()
        np.fill_diagonal(J_no_diag, expected_val)
        np.testing.assert_allclose(J_no_diag, np.ones((N, N)) * expected_val)

    def test_gaussian_interaction(self):
        D = 1
        L = 10
        interaction = GaussianInteraction(mean=0.0, std=1.0, seed=42)
        J1 = interaction.generate(D, L, quenched=1)
        J2 = interaction.generate(D, L, quenched=1)

        # Consistency with seed
        np.testing.assert_array_equal(J1, J2)

        # Symmetry J_ij = J_ji
        np.testing.assert_array_equal(J1, np.transpose(J1, (0, 2, 1)))

        # Zero diagonal
        np.testing.assert_equal(np.diag(J1[0]), np.zeros(10))
