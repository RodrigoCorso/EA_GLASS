import pytest
import numpy as np
import tensorflow as tf

from spin_engine.models.edwards_anderson import EdwardsAndersonSystem
from spin_engine.interactions.standard import BinaryRandomInteraction


def test_binary_random_interaction():
    interaction = BinaryRandomInteraction(J=2.0, seed=42)
    J_tensor = interaction.generate(D=2, L=2, quenched=1)
    J_flat = J_tensor.reshape((4, 4))
    
    # Check symmetric
    np.testing.assert_array_equal(J_flat, J_flat.T)
    
    # Check zero diagonal
    np.testing.assert_array_equal(np.diag(J_flat), np.zeros(4))
    
    # Check values are only +/- J or 0 (on diagonal)
    unique_vals = np.unique(J_flat)
    for val in unique_vals:
        assert val in [-2.0, 0.0, 2.0]


def test_edwards_anderson_energy():
    # 2x2 lattice
    L = 2
    D = 2
    
    # Hand-calculated matrix for testing
    J_matrix_flat = np.array([
        [ 0,  1, -1,  1],
        [ 1,  0,  1, -1],
        [-1,  1,  0,  1],
        [ 1, -1,  1,  0]
    ], dtype=np.float32)
    
    J_tensor = J_matrix_flat.reshape((1, L, L, L, L))
    
    system = EdwardsAndersonSystem(
        lattice_length=L,
        lattice_dim=D,
        lattice_replicas=2,
        interaction_matrix=J_tensor
    )
    
    # Set explicit spin state
    # Replica 0: all +1
    # Replica 1: s=[1, -1, 1, -1]
    
    spins_0 = np.array([1, 1, 1, 1], dtype=np.float32)
    spins_1 = np.array([1, -1, 1, -1], dtype=np.float32)
    
    spin_state_flat = np.stack([spins_0, spins_1])
    spin_state = tf.convert_to_tensor(spin_state_flat.reshape((1, 2, L, L)))
    system.update_state(spin_state)
    
    energies = system.compute_energy()
    
    expected_energies = np.array([[-2.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(energies.numpy(), expected_energies)
