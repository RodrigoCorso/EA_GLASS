import pytest
import tensorflow as tf
import numpy as np
from spin_engine.models.ising import IsingSystem
from spin_engine.models.spherical import SphericalSystem
from spin_engine.models.wegner import WegnerSystem
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


class TestModels:
    def test_ising_initialization(self):
        L = 10
        replicas = 5
        interaction = tf.reshape(
            tf.ones((L*L, L*L), dtype=tf.float32), (L, L, L, L))

        system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=interaction,
            lattice_dim=2
        )

        # Check shape
        assert system.spin_state.shape == (1, replicas, L, L)
        # Check values are -1 or 1
        unique_vals = np.unique(system.spin_state.value().numpy())
        assert np.all(np.isin(unique_vals, [-1.0, 1.0]))

    def test_ising_energy_shape(self):
        L = 4
        replicas = 3
        # Simple interaction matrix
        interaction = tf.reshape(tf.random.normal((L*L, L*L)), (L, L, L, L))

        system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=interaction
        )

        energy = system.compute_energy()
        assert energy.shape == (1, replicas)

    def test_spherical_initialization(self):
        L = 10
        replicas = 5
        interaction = tf.reshape(
            tf.ones((L*L, L*L), dtype=tf.float32), (L, L, L, L))

        system = SphericalSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=interaction,
            spherical_constraint=True
        )

        # Check shape
        assert system.spin_state.shape == (1, replicas, L, L)

        # Check constraint: sum(x^2) = N
        spin_flat = tf.reshape(system.spin_state, (system.quenched_replicas, replicas, -1))
        norm_sq = tf.reduce_sum(spin_flat**2, axis=2)
        expected_norm = float(L*L)

        assert np.allclose(norm_sq.numpy(), expected_norm, atol=1e-4)

    def test_wegner_system(self):
        # WegnerSystem is now implemented, test initialization and energy compute
        system = WegnerSystem(
            lattice_dim=2, lattice_length=4, lattice_replicas=2)

        state = system.spin_state
        assert state.shape == (1, 2, 4, 4, 2)  # Q, R, L, L, D
        
        energy = system.compute_energy()
        assert energy.shape == (1, 2)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_spherical_constraint_nd(self, dim):
        """
        Verifies that the spherical constraint holds for D=1, 2, 3.
        Sum(s_i^2) should be equal to N (number of spins) for each replica.
        """
        L = 5
        replicas = 4

        # Needs interaction matrix of correct shape
        # Shape is (L,)*dim
        # Interaction shape is (L,)*dim + (L,)*dim
        shape = (L,) * dim

        # We can use a dummy interaction since we only care about initialization here
        # Total spins N = L^dim
        N = int(L**dim)
        interaction_flat = tf.eye(N)  # Simple identity
        interaction = tf.reshape(interaction_flat, shape + shape)

        system = SphericalSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=interaction,
            lattice_dim=dim,
            spherical_constraint=True
        )

        # Check shape
        expected_shape = (1, replicas,) + shape
        assert system.spin_state.shape == expected_shape

        # Verify constraint
        # Flatten everything except replicas: (Q, R, N)
        spin_flat = tf.reshape(system.spin_state, (1, replicas, -1))

        # Calculate sum of squares
        norm_sq = tf.reduce_sum(spin_flat**2, axis=2)

        # Expected is N
        expected_norm = float(N)

        assert np.allclose(norm_sq.numpy(), expected_norm, atol=1e-4), \
            f"Spherical constraint failed for dim={dim}. Expected {expected_norm}, got {norm_sq.numpy()}"
