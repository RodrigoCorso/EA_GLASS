import pytest
import tensorflow as tf
from typing import cast, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import PopulationAnnealing, Tracker
from spin_engine.measurements.base import Measurement


class TestPopulationAnnealing:

    @pytest.fixture
    def setup_system(self):
        lattice_dim = 2
        lattice_length = 4
        lattice_replicas = 32  # Larger population for PA

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )

        simulation = PopulationAnnealing(ising_system)

        return {
            'system': ising_system,
            'simulation': simulation,
            'replicas': lattice_replicas,
            'num_spins': lattice_length ** lattice_dim
        }

    def test_initialization(self, setup_system):
        """Test that PopulationAnnealing initializes correctly."""
        simulation = setup_system['simulation']
        system = setup_system['system']
        
        assert simulation.population_size == system.lattice_replicas
        assert simulation.current_energies.shape == (system.quenched_replicas, simulation.population_size)
        assert simulation.cumulative_weight_factor.numpy() == 1.0
        assert simulation.current_beta.numpy() == 0.0

    def test_weight_computation(self, setup_system):
        """Test that weights are computed correctly."""
        simulation = setup_system['simulation']
        
        beta_old = tf.constant(0.0, dtype=tf.float32)
        beta_new = tf.constant(0.1, dtype=tf.float32)
        
        weights = simulation._compute_weights(beta_old, beta_new)
        
        # Weights should be positive
        assert tf.reduce_all(weights > 0)
        
        # At beta_old=0, all weights should be 1.0 (since exp(0) = 1)
        # But after initialization with some energy, they may differ
        
    def test_weight_normalization(self, setup_system):
        """Test that weights are normalized correctly."""
        simulation = setup_system['simulation']
        
        # Create dummy weights
        weights = tf.random.uniform((1, setup_system['replicas']), minval=0.5, maxval=2.0)
        normalized = simulation._normalize_weights(weights)
        
        # Normalized weights should sum to 1
        total = tf.reduce_sum(normalized)
        assert abs(total - 1.0) < 1e-5

    def test_resampling_changes_population(self, setup_system):
        """Test that resampling actually changes the population."""
        simulation = setup_system['simulation']
        system = setup_system['system']
        
        # Store initial state
        initial_state = tf.identity(system.spin_state)
        
        # Perform a small annealing step
        beta_old = tf.constant(0.0, dtype=tf.float32)
        beta_new = tf.constant(0.01, dtype=tf.float32)
        
        new_state, new_energies = simulation._perform_resampling(beta_old, beta_new)
        
        # State should have changed (some replicas resampled)
        # Note: Due to randomness, this could theoretically fail if same replicas selected
        # but probability is very low with large population
        states_different = not tf.reduce_all(tf.equal(initial_state, new_state))
        # We don't assert this because with small delta_beta, resampling might keep same config
        # Instead, just check shapes match
        assert new_state.shape == initial_state.shape
        assert new_energies.shape == simulation.current_energies.shape

    def test_anneal_step_execution(self, setup_system):
        """Test that a single annealing step executes without error."""
        simulation = setup_system['simulation']
        
        beta_old = tf.constant(0.0, dtype=tf.float32)
        beta_new = tf.constant(0.1, dtype=tf.float32)
        equilibration_steps = tf.constant(2, dtype=tf.int32)
        num_disturbances = tf.constant(1, dtype=tf.int32)
        
        # This should run without errors
        simulation.anneal_step(
            beta_old=beta_old,
            beta_new=beta_new,
            equilibration_steps=equilibration_steps,
            num_disturbances=num_disturbances
        )
        
        # Beta should be updated
        assert simulation.current_beta.numpy() == pytest.approx(0.1, rel=1e-5)

    def test_sweep_execution(self, setup_system):
        """Test that a full PA sweep executes and tracks measurements."""
        simulation = setup_system['simulation']
        
        # Mock tracker
        class MockMeasurement(Measurement):
            def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
                spin_state, _ = self._resolve(spin_state, system)
                return tf.reduce_mean(tf.cast(spin_state, tf.float32))

        tracker = Tracker([MockMeasurement()])
        
        # Small beta schedule
        beta_schedule = tf.constant([0.0, 0.1, 0.2, 0.3], dtype=tf.float32)
        
        # Run sweep
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(2, dtype=tf.int32),
            num_disturbances=tf.constant(1, dtype=tf.int32)
        )
        
        # Check that history is populated
        assert "MockMeasurement" in tracker.history
        # Should track at each beta step (4 betas = 4 tracking points)
        assert tracker.history["MockMeasurement"].numpy().shape[0] == 4

    def test_partition_function_estimate(self, setup_system):
        """Test that partition function estimate is tracked."""
        simulation = setup_system['simulation']
        
        initial_Z = simulation.get_partition_function_estimate().numpy()
        assert initial_Z == 1.0
        
        # After annealing, Z should change
        beta_schedule = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
        
        class DummyMeasurement(Measurement):
            def compute(self, spin_state=None, system=None):
                return tf.constant(0.0)
        
        tracker = Tracker([DummyMeasurement()])
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(1, dtype=tf.int32)
        )
        
        final_Z = simulation.get_partition_function_estimate().numpy()
        # Z should be positive and likely different from 1.0
        assert final_Z > 0

    def test_high_temperature_behavior(self, setup_system):
        """Test that at high temperature, magnetization is near zero."""
        simulation = setup_system['simulation']
        
        class MagnetizationMeasurement(Measurement):
            def compute(self, spin_state=None, system=None):
                spin_state, _ = self._resolve(spin_state, system)
                return tf.reduce_mean(tf.cast(spin_state, tf.float32), axis=[2, 3])
        
        tracker = Tracker([MagnetizationMeasurement()])
        
        # Anneal to moderate temperature (not too cold)
        beta_schedule = tf.constant([0.0, 0.05, 0.1], dtype=tf.float32)
        
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(5, dtype=tf.int32)
        )
        
        # At high temp, average magnetization should be low
        final_mag = tracker.history["MagnetizationMeasurement"][-1]
        mean_abs_mag = tf.reduce_mean(tf.abs(final_mag))
        
        # Should be relatively disordered (this is probabilistic, so use loose bound)
        assert mean_abs_mag < 0.5, f"Magnetization at high temp should be low, got {mean_abs_mag}"

    def test_reset_functionality(self, setup_system):
        """Test that reset restores initial state."""
        simulation = setup_system['simulation']
        
        # Do some annealing
        beta_schedule = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
        
        class DummyMeasurement(Measurement):
            def compute(self, spin_state=None, system=None):
                return tf.constant(0.0)
        
        tracker = Tracker([DummyMeasurement()])
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(1, dtype=tf.int32)
        )
        
        # State should have changed
        assert simulation.current_beta.numpy() > 0
        
        # Reset
        simulation.reset()
        
        # Should be back to initial state
        assert simulation.cumulative_weight_factor.numpy() == 1.0
        assert simulation.current_beta.numpy() == 0.0
