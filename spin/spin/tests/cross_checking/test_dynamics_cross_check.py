import tensorflow as tf
import pytest
from spin_engine.models.ising import IsingSystem
from spin_engine.dynamics import MetropolisHastings
from legacy_core import SpinSystem as LegacySpinSystem


class TestDynamicsCrossCheck:

    @pytest.fixture
    def setup_data(self):
        L = 4
        replicas = 2
        dim = 2
        N = L**dim

        # Consistent random seed
        tf.random.set_seed(42)

        J_flat = tf.random.normal((N, N))
        J_flat = 0.5 * (J_flat + tf.transpose(J_flat))  # Symmetric
        J = tf.reshape(J_flat, (L, L, L, L))

        h = tf.random.normal((L, L))

        initial_spins = tf.where(
            tf.random.uniform((replicas, L, L)) > 0.5,
            1.0, -1.0
        )

        return {
            'L': L, 'replicas': replicas, 'dim': dim, 'N': N,
            'J': J, 'h': h,
            'spins': initial_spins
        }

    def test_flip_spins_cross_check(self, setup_data):
        L, replicas, J, h = setup_data['L'], setup_data['replicas'], setup_data['J'], setup_data['h']
        spins = setup_data['spins']

        # Setup Legacy
        legacy = LegacySpinSystem(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            initial_spin_state=spins,
            external_field=h,
            model="ising"
        )

        # Setup New
        new_system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            external_field=h,
            initial_spin_state=spins
        )
        dynamics = MetropolisHastings(new_system)

        # Force same initial energy (though should be guaranteed by init)
        legacy.energy.assign(legacy.compute_pairwise_energies())
        # MetropolisHastings inits energy in __init__

        num_flips = tf.constant(1)

        # Set seed before action to ensure same random numbers are drawn
        seed = 12345

        # Legacy action
        tf.random.set_seed(seed)
        legacy_updated_spins, legacy_energy_delta = legacy.flip_spins(
            num_flips)  # type: ignore

        # New action
        @tf.function
        def run_new_flip():
            scatter_indices, updates, _, new_total_energy = dynamics.flip_spins(num_flips)
            spin_flat = tf.reshape(new_system.spin_state, (new_system.quenched_replicas, new_system.lattice_replicas, -1))
            new_updated_spins_flat = tf.tensor_scatter_nd_update(spin_flat, scatter_indices, updates)
            new_updated_spins = tf.reshape(new_updated_spins_flat, new_system.spin_state.shape)
            return new_updated_spins, new_total_energy

        tf.random.set_seed(seed)
        new_updated_spins, new_total_energy = run_new_flip()
        new_energy_delta = new_total_energy - dynamics.current_energy

        # Legacy returns flat spins (replicas, N), New returns (1, replicas, L, L)
        legacy_updated_spins = tf.reshape(
            legacy_updated_spins, new_updated_spins.shape)

        # Compare states
        tf.debugging.assert_equal(legacy_updated_spins, new_updated_spins,
                                  message="Updated spin states do not match")

        # Compare energy deltas
        tf.debugging.assert_near(legacy_energy_delta, new_energy_delta, atol=5e-3,
                                 message="Energy deltas do not match")
