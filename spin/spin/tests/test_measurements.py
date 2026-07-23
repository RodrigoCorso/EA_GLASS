from spin_engine.measurements import (
    Energy, Magnetization, MagneticSusceptibility, OverlapMatrix
)
from spin_engine.models.wegner import WegnerSystem
from spin_engine.models.spherical import SphericalSystem
from spin_engine.models.ising import IsingSystem
from spin_engine.interactions import (
    DecayingInteraction,
    PeriodicNearestNeighborInteraction,
    CurieWeissInteraction,
    GaussianInteraction
)
import tensorflow as tf
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


class TestManualMeasurements:

    @pytest.fixture
    def manual_ising(self):
        """
        Setup a small 2x2 Ising system with 2 replicas and known states.
        """
        L = 2
        N = 4  # L*L
        replicas = 2

        # Interaction matrix J: all ones (scaled by 0.5)
        # In legacy_core (and our models), J is passed as (L, L, L, L) or (N, N)
        # but internally it expects (L, L, L, L) for the constructor validator,
        # or (N, N) if we bypass.
        # Let's provide (L, L, L, L) all ones.
        j_val = 1.0
        J_flat = tf.ones((N, N)) * j_val
        J = tf.reshape(J_flat, (L, L, L, L))

        h = tf.ones((L, L))  # External field = 1 everywhere

        # Spin States:
        # Replica 0: All Up [1, 1, 1, 1]
        # Replica 1: Checkerboard [1, -1, 1, -1]
        s0 = tf.ones((L, L))
        s1 = tf.constant([[1.0, -1.0], [1.0, -1.0]])
        spins = tf.stack([s0, s1])  # Shape (2, 2, 2)

        system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            external_field=h,
            initial_spin_state=spins
        )
        return system

    def test_ising_magnetization(self, manual_ising):
        """
        Verify Magnetization against manual calc.
        Rep 0: All 1s -> Mean = 1.0
        Rep 1: Two 1s, two -1s -> Mean = 0.0
        """
        mag = Magnetization(manual_ising).compute()
        expected = tf.constant([1.0, 0.0])
        tf.debugging.assert_near(mag, expected, atol=1e-6)

    def test_ising_energy(self, manual_ising):
        """
        Verify Energy against manual calc.
        E = -0.5 * sum(S_i J_ij S_j) - sum(h_i S_i)

        J_ij = 1 everywhere.
        h_i = 1 everywhere.

        Replica 0 (All 1):
            sum(S_i S_j) = sum(1 * 1) for all 4*4 pairs = 16
            Term 1: -0.5 * 16 = -8
            sum(h_i S_i) = sum(1 * 1) * 4 = 4
            Term 2: -4
            Total E = -12

        Replica 1 (Checkerboard [1, -1, 1, -1]):
            S = [1, -1, 1, -1]
            sum S_i = 0
            sum(S_i S_j):
                S = [1, -1, 1, -1]
                Outer product S^T S:
                [[ 1, -1,  1, -1],
                 [-1,  1, -1,  1],
                 [ 1, -1,  1, -1],
                 [-1,  1, -1,  1]]
                Sum of all elements = 0 (rows sum to 0)
            Term 1: -0.5 * 0 = 0
            sum(h_i S_i) = 0
            Term 2: 0
            Total E = 0
        """
        energies = Energy(manual_ising).compute()
        expected = tf.constant([-12.0, 0.0])
        tf.debugging.assert_near(energies, expected, atol=1e-6)

    def test_overlap_matrix(self, manual_ising):
        """
        Verify Overlap Matrix Q_ab = (1/N) sum(S_i^a S_i^b)

        Replica 0: [1, 1, 1, 1]
        Replica 1: [1, -1, 1, -1]

        Q_00 = (1/4) * (1+1+1+1) = 1.0
        Q_11 = (1/4) * (1+1+1+1) = 1.0
        Q_01 = (1/4) * (1*1 + 1*-1 + 1*1 + 1*-1) = (1 - 1 + 1 - 1)/4 = 0.0
        """
        overlap = OverlapMatrix(manual_ising).compute()
        expected = tf.constant([[1.0, 0.0], [0.0, 1.0]])

    def test_interactions_non_zero_energy(self):
        """
        Verify that new interactions produce non-zero energy for random states.
        This ensures they are hooked up correctly to the system.
        """
        L = 4
        D = 2
        replicas = 2

        # 1. Periodic NN
        # For a ferromagnetic state (all 1), energy should be negative (ferromagnetic coupling).
        # E = -0.5 * sum J_ij S_i S_j.
        # NN: each spin has 2*D neighbors. Total pairs = N * D. J=1.
        # Sum = N * D * 2 (double counting in formula cancel with 0.5? No).
        # Energy = -0.5 * sum_{i!=j} J_ij S_i S_j
        # sum_{i!=j} J_ij = N * (2D)
        # E = -0.5 * N * 2D = - N * D = -16 * 2 = -32.

        inter_nn = PeriodicNearestNeighborInteraction()
        J_nn = inter_nn.generate(D, L)

        spins_all_up = tf.ones((replicas,) + (L,)*D)

        system_nn = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J_nn,
            lattice_dim=D,
            initial_spin_state=spins_all_up
        )

        e_nn = Energy(system_nn).compute()
        # Expect -32.0 for both replicas
        tf.debugging.assert_near(e_nn, tf.constant([-32.0, -32.0]), atol=1e-5)

    def test_curie_weiss_energy(self):
        """
        Curie-Weiss: J_ij = J0/N for all i!=j.
        State all up.
        Sum_{i!=j} J_ij = (N^2 - N) * (J0/N) = (N-1)*J0
        E = -0.5 * (N-1)*J0
        Let J0=1, N=16. E = -0.5 * 15 = -7.5
        """
        L = 4
        D = 2  # N=16
        replicas = 1
        J0 = 1.0

        inter_cw = CurieWeissInteraction(J0=J0)
        J_cw = inter_cw.generate(D, L)

        spins_all_up = tf.ones((replicas,) + (L,)*D)

        system_cw = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J_cw,
            lattice_dim=D,
            initial_spin_state=spins_all_up
        )

        e_cw = Energy(system_cw).compute()
        expected = -0.5 * ((L**D) - 1) * J0
        tf.debugging.assert_near(e_cw, tf.constant([expected]), atol=1e-5)
