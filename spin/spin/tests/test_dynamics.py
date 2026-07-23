import pytest
import tensorflow as tf
from typing import cast, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings, Tracker
from spin_engine.measurements.base import Measurement


class TestDynamics:

    @pytest.fixture
    def setup_system(self):
        lattice_dim = 2
        lattice_length = 4
        lattice_replicas = 16

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )

        simulation = MetropolisHastings(ising_system)

        return {
            'system': ising_system,
            'simulation': simulation,
            'replicas': lattice_replicas,
            'num_spins': lattice_length ** lattice_dim
        }

    def test_flip_spins_changes_sign(self, setup_system):
        simulation = setup_system['simulation']
        system = setup_system['system']
        replicas = setup_system['replicas']

        initial_spins = tf.identity(system.spin_state)

        num_flips = cast(tf.Tensor, tf.constant(1))

        # The current API returns scatter_indices, updates, original_spins, updated_energy
        # but the step logic handles the state update internally. We just test the math.
        scatter_indices, updates, _, _ = simulation.flip_spins(num_flips=num_flips)
        proposed_spin_state = tf.tensor_scatter_nd_update(tf.reshape(initial_spins, (1, replicas, -1)), scatter_indices, updates)
        proposed_spin_state = tf.reshape(proposed_spin_state, initial_spins.shape)

        # Check that exactly num_flips * replicas spins changed
        # Since we use -1/1 spins, changed spins will have product -1, unchanged 1
        product = initial_spins * proposed_spin_state

        # Count differences
        diff = tf.reduce_sum(
            tf.cast(tf.abs(initial_spins - proposed_spin_state) > 1e-5, tf.int32))

        expected_diff = replicas * num_flips
        assert diff == expected_diff, f"Expected {expected_diff} flips, got {diff}"

        # Verify the changed ones are exactly negated
        # Where they differ, sum should be 0 (x + (-x) = 0)
        # Or diff should be 2.0 or -2.0

        changes = proposed_spin_state - initial_spins
        # Filter non-zero changes
        nonzero_changes = tf.boolean_mask(changes, tf.abs(changes) > 0)

        # Changes should be magnitude 2 (flip from 1 to -1 or -1 to 1)
        tf.debugging.assert_near(tf.abs(nonzero_changes), 2.0)

    def test_energy_delta_small_system(self):
        # Setup small system for manual verification
        lattice_dim = 2
        lattice_length = 4  # Small enough
        lattice_replicas = 1

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )

        simulation = MetropolisHastings(ising_system)

        initial_energy = simulation.current_energy
        num_flips = cast(tf.Tensor, tf.constant(1))

        scatter_indices, updates, _, _ = simulation.flip_spins(num_flips=num_flips)
        proposed_spin_state = tf.tensor_scatter_nd_update(tf.reshape(ising_system.spin_state, (1, lattice_replicas, -1)), scatter_indices, updates)
        proposed_spin_state = tf.reshape(proposed_spin_state, ising_system.spin_state.shape)
        
        idx_for_delta = tf.reshape(scatter_indices[:, 2], (1, lattice_replicas, num_flips))
        delta_energy = ising_system.compute_delta_energy(ising_system.spin_state, proposed_spin_state, idx_for_delta)
        new_energy = initial_energy + delta_energy

        # Calculate new energy manually from the proposed state
        new_energy_recomputed = ising_system.compute_energy(
            spin_state=proposed_spin_state)

        tf.debugging.assert_near(new_energy, new_energy_recomputed, atol=1e-5)

    def test_sweep_execution(self, setup_system):
        simulation = setup_system['simulation']

        # Mock tracker
        class MockMeasurement(Measurement):
            def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
                spin_state, _ = self._resolve(spin_state, system)
                return tf.reduce_mean(tf.cast(spin_state, tf.float32))

        tracker = Tracker([MockMeasurement()])

        # Run small sweep
        simulation.sweep(tracker, beta=0.1,
                         num_disturbances=1, sweep_length=10)

        # Check that history is populated
        assert "MockMeasurement" in tracker.history
        # 0 to 10 inclusive
        assert tracker.history["MockMeasurement"].numpy().shape[0] == 11

    def test_sweep_tracking_integration(self, setup_system):
        simulation = setup_system['simulation']
        granularity = 2

        class MockMeasurement(Measurement):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def compute(self, spin_state=None, system=None):
                return tf.constant(1.0)  # Dummy value

        tracker = Tracker([MockMeasurement("Dummy")], granularity=granularity)

        sweep_length = 10
        simulation.sweep(tracker, beta=0.1, num_disturbances=1,
                         sweep_length=sweep_length)

        expected_steps = (sweep_length // granularity) + 1
        assert tracker.history["Dummy"].numpy().shape[0] == expected_steps

    def test_high_temperature_convergence(self):
        # High Temp -> Low Beta -> Magnetization should be random (near 0 avg)
        lattice_dim = 2
        lattice_length = 8
        lattice_replicas = 5

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )
        simulation = MetropolisHastings(ising_system)

        class Magnetization(Measurement):
            def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
                spin_state, _ = self._resolve(spin_state, system)
                # Mean per replica
                return tf.reduce_mean(tf.cast(spin_state, tf.float32), axis=[2, 3])

        tracker = Tracker([Magnetization()])

        # Beta = 0.0 (Infinite Temperature)
        simulation.sweep(tracker, beta=0.0,
                         num_disturbances=10, sweep_length=200)

        final_mag = tracker.history["Magnetization"][-1]

        # Should be close to 0
        mean_abs_mag = tf.reduce_mean(tf.abs(final_mag))
        assert mean_abs_mag < 0.2, f"Magnetization at infinite temp should be low, got {mean_abs_mag}"

    def test_low_temperature_convergence(self):
        # Low Temp -> High Beta -> Magnetization should order (+1 or -1)
        lattice_dim = 2
        lattice_length = 4  # Keep small for fast ordering
        lattice_replicas = 20

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )
        simulation = MetropolisHastings(ising_system)

        class Magnetization(Measurement):
            def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
                spin_state, _ = self._resolve(spin_state, system)
                return tf.reduce_mean(tf.cast(spin_state, tf.float32), axis=[2, 3])

        tracker = Tracker([Magnetization()])

        # High Beta
        simulation.sweep(tracker, beta=1.0,
                         num_disturbances=1, sweep_length=5000)

        final_mag = tracker.history["Magnetization"][-1]

        # Should be close to 1 or -1
        mean_abs_mag = tf.reduce_mean(tf.abs(final_mag))
        assert mean_abs_mag > 0.8, f"Magnetization at low temp should be ordered, got {mean_abs_mag}"
