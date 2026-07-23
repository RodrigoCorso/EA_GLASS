import pytest
import tensorflow as tf
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.base import Measurement
from spin_engine.models.base import BaseSpinSystem
from typing import Optional, Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from spin_engine.measurements.base import Measurement
    from spin_engine.models.base import BaseSpinSystem


class MockMeasurement(Measurement):
    def __init__(self, name: str = "Mock"):
        super().__init__()
        self.name = name

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> Any:
        return tf.constant(1.0, dtype=tf.float32)


class TestTracker:
    def test_init_run(self):
        measurements = [MockMeasurement("M1"), MockMeasurement("M2")]
        tracker = Tracker(measurements, granularity=1)

        sweep_length = cast(tf.Tensor, 10)
        arrays = tracker.init_run(sweep_length)

        assert len(arrays) == 2
        # Size should be sweep_length // granularity + 1
        expected_size = cast(tf.Tensor, 11)
        for array in arrays:
            assert array.size() == expected_size

    def test_track_granularity(self):
        measurements = [MockMeasurement()]
        granularity = 5
        tracker = Tracker(measurements, granularity=granularity)

        sweep_length = cast(tf.Tensor, 20)
        arrays = tracker.init_run(sweep_length)

        # Create a mock system with spin_state
        class MockSystem(BaseSpinSystem):
            spin_state = tf.zeros((1, 1))

            def initialize_state(self) -> tf.Tensor:
                return tf.zeros((1, 1))

            def compute_energy(self, spin_state: tf.Variable | tf.Tensor | None = None) -> tf.Tensor:
                return tf.zeros((1))

        system = MockSystem(0, 0, 0)

        # Track step 0 (should write)
        arrays = tracker.track_initial(system, arrays)
        # Track step 1 (should NOT write)
        arrays = tracker.track(cast(tf.Tensor, 1), system, arrays)
        # Track step 5 (should write)
        arrays = tracker.track(cast(tf.Tensor, 5), system, arrays)

        # Finalize to check values
        tracker.finalize(arrays)

        history = tracker.history["Mock"].numpy()

        # We expect writes at index 0 and index 1 (corresponding to step 5)
        # Index 0 is step 0. Index 1 corresponds to step 5 (5 // 5 = 1).
        # We wrote 1.0 at these positions.
        # TensorArray defaults are not zero-initialized if not written, but we write sequentially usually.
        # Here we skipped steps.  However, the tracker logic maps step -> index.
        # step 0 -> index 0. step 5 -> index 1.

        assert history[0] == 1.0
        assert history[1] == 1.0
        # step 1 -> index 0 (1//5 = 0) but we only write if step % granularity == 0.
        # So step 1 didn't trigger a write.

    def test_finalize(self):
        measurements = [MockMeasurement()]
        tracker = Tracker(measurements, granularity=1)
        sweep_length = cast(tf.Tensor, tf.constant(2))

        arrays = tracker.init_run(sweep_length)

        class MockSystem(BaseSpinSystem):
            spin_state = tf.zeros((1, 1))

            def initialize_state(self) -> tf.Tensor:
                return tf.zeros((1, 1))

            def compute_energy(self, spin_state: tf.Variable | tf.Tensor | None = None) -> tf.Tensor:
                return tf.zeros((1))

        system = MockSystem(0, 0, 0)
        arrays = tracker.track_initial(system, arrays)
        arrays = tracker.track(cast(tf.Tensor, 1), system, arrays)
        arrays = tracker.track(cast(tf.Tensor, 2), system, arrays)

        tracker.finalize(arrays)

        assert "Mock" in tracker.history
        assert tracker.history["Mock"].shape == (3,)  # 0, 1, 2

    def test_track_measurement_counts(self):
        num_measurements = 5
        measurements = [MockMeasurement(f"M{i}")
                        for i in range(num_measurements)]
        tracker = Tracker(measurements)

        assert len(tracker.measurements) == num_measurements

        sweep_length = cast(tf.Tensor, 10)
        arrays = tracker.init_run(sweep_length)

        assert len(arrays) == num_measurements
