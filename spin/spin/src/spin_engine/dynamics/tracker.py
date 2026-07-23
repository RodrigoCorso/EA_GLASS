import tensorflow as tf
from typing import Sequence, Dict, Tuple, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.measurements.base import Measurement
    from spin_engine.models.base import BaseSpinSystem


class Tracker(tf.Module):
    def __init__(self, measurements: Sequence['Measurement'], granularity: int = 1):
        super().__init__()
        self.measurements = measurements
        self.granularity = granularity
        self.history = {}

    def init_run(self, sweep_length: tf.Tensor) -> Tuple[tf.TensorArray, ...]:
        """
        Initialize the tracking arrays for the run.
        """
        num_measurements = tf.cast(
            tf.divide(sweep_length, self.granularity), tf.int32)
        size = tf.add(num_measurements, 1)
        arrays = []
        for _ in self.measurements:
            arrays.append(tf.TensorArray(
                dtype=tf.float32, size=size, clear_after_read=True))

        return tuple(arrays)

    def track(
        self,
        step: tf.Tensor,
        system: 'BaseSpinSystem',
        tracking_arrays: Tuple[tf.TensorArray, ...]
    ) -> Tuple[tf.TensorArray, ...]:
        """
        Track measurements if step is a multiple of granularity.
        """
        step_int = tf.cast(step, tf.int32)
        condition = tf.equal(tf.math.floormod(step_int, self.granularity), 0)
        index = tf.math.floordiv(step_int, self.granularity)

        new_arrays = list(tracking_arrays)
        for i, measurement in enumerate(self.measurements):
            # Create isolated closures for the true/false branches
            def get_write_fn(arr, meas):
                state = system.spin_state
                return lambda: arr.write(index, meas.compute(state, system=system))
            
            def get_skip_fn(arr):
                return lambda: arr
            
            new_arrays[i] = tf.cond(
                condition,
                get_write_fn(tracking_arrays[i], measurement),
                get_skip_fn(tracking_arrays[i])
            )

        return tuple(new_arrays)

    def track_initial(
        self,
        system: 'BaseSpinSystem',
        tracking_arrays: Tuple[tf.TensorArray, ...]
    ) -> Tuple[tf.TensorArray, ...]:
        """
        Unconditionally track the initial state at index 0.
        This initializes the TensorArray shape for XLA compilation.
        """
        new_arrays = list(tracking_arrays)
        state = system.spin_state
        for i, measurement in enumerate(self.measurements):
            new_arrays[i] = tracking_arrays[i].write(0, measurement.compute(state, system=system))
        return tuple(new_arrays)

    def finalize(self, stacked_tensors: Tuple[tf.Tensor | tf.TensorArray, ...]):
        """
        Stack results and store in self.history.
        """
        for i, measurement in enumerate(self.measurements):
            name = getattr(measurement, 'name', measurement.__class__.__name__)
            val = stacked_tensors[i]
            if isinstance(val, tf.TensorArray):
                result = val.stack()
            else:
                result = val
            if name not in self.history:
                self.history[name] = tf.Variable(
                    result, validate_shape=False, trainable=False)
            else:
                self.history[name].assign(result)


