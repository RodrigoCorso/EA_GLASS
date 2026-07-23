import pytest
import numpy as np
import tensorflow as tf

from spin_engine.models.sk import SherringtonKirkpatrickSystem
from spin_engine.models.ising import IsingSystem


def test_sk_system_initialization():
    lattice_length = 100
    lattice_dim = 1
    num_spins = lattice_length ** lattice_dim
    lattice_replicas = 2
    J = 2.0
    
    system = SherringtonKirkpatrickSystem(
        lattice_length=lattice_length,
        lattice_dim=lattice_dim,
        lattice_replicas=lattice_replicas,
        J=J,
        seed=42
    )
    
    # Check inheritance
    assert isinstance(system, IsingSystem)
    
    # Check shape overrides
    assert system.lattice_dim == 1
    assert system.lattice_length == num_spins
    assert system.number_spins == num_spins
    
    # Check spin state shape
    assert system.spin_state.shape == (1, lattice_replicas, num_spins)
    
    # Check interaction matrix shape
    assert system.interaction_matrix.shape == (num_spins, num_spins)
    
    # Verify interaction matrix statistics (SK model std = J/sqrt(N))
    # It won't be exact because we force symmetric and 0 diagonal, but it should be close
    J_flat = system.interaction_matrix.numpy()
    
    # Check diagonal is zero
    np.testing.assert_array_equal(np.diag(J_flat), np.zeros(num_spins))
    
    # Check symmetric
    np.testing.assert_array_equal(J_flat, J_flat.T)
    
    # Check std is approximately J/sqrt(N)
    expected_std = J / np.sqrt(num_spins)
    actual_std = np.std(J_flat[np.triu_indices(num_spins, k=1)])
    
    np.testing.assert_allclose(actual_std, expected_std, rtol=0.1)

def test_sk_energy_computation():
    lattice_length = 10
    lattice_dim = 1
    system = SherringtonKirkpatrickSystem(
        lattice_length=lattice_length,
        lattice_dim=lattice_dim,
        lattice_replicas=4,
        J=1.0,
        seed=1
    )
    
    energies = system.compute_energy()
    assert energies.shape == (1, 4)
    
    # The energy shouldn't be nan or infinity
    assert not np.any(np.isnan(energies.numpy()))
    assert not np.any(np.isinf(energies.numpy()))
