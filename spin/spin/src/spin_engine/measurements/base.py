from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Optional, TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class Measurement(ABC):
    """
    Abstract base class for all measurements.

    A Measurement is an observer that can compute a value (scalar or tensor)
    given a Spin System and its state.
    """

    def __init__(self, system: Optional['BaseSpinSystem'] = None) -> None:
        self.system = system

    def _resolve(
        self,
        spin_state: Optional[tf.Tensor | tf.Variable],
        system: Optional['BaseSpinSystem']
    ) -> Tuple[tf.Tensor, Optional['BaseSpinSystem']]:
        """
        Resolves the system and spin_state based on priority:
        1. system: arg > self.system
        2. spin_state: arg > resolved_system.spin_state
        """
        eff_sys = system if system is not None else self.system

        eff_state = spin_state
        if eff_state is None:
            if eff_sys is not None:
                eff_state = eff_sys.spin_state.value()
            else:
                raise ValueError(
                    f"Measurement '{self.__class__.__name__}' failed: "
                    "No spin_state provided and no system available to provide one."
                )

        return tf.convert_to_tensor(eff_state), eff_sys

    @abstractmethod
    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> Any:
        """
        Compute the measurement.

        Args:
            spin_state: Optional tensor representing the state to measure.
                        If None, uses self.system.spin_state.
            system: Optional BaseSpinSystem instance. If None, uses self.system.

        Returns:
            The computed measurement value.
        """
        pass

    def __call__(self, spin_state: Optional[tf.Variable | tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> Any:
        return self.compute(spin_state, system)
