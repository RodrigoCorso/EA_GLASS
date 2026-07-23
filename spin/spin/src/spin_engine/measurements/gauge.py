import tensorflow as tf
from typing import Optional
from .base import Measurement
from spin_engine.models.wegner import WegnerSystem
from spin_engine.models.base import BaseSpinSystem


class Plaquette(Measurement):
    """
    Computes the average plaquette value.
    Placeholder until WegnerSystem is implemented.
    """

    def __init__(self, system):
        super().__init__(system)
        if not isinstance(system, WegnerSystem):
            raise TypeError(
                "Plaquette measurement only valid for WegnerSystem")

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
        raise NotImplementedError("WegnerSystem is not implemented yet.")


class WilsonLoop(Measurement):
    """
    Computes Wilson loops of a given size.
    Placeholder until WegnerSystem is implemented.
    """

    def __init__(self, system, loop_size: int = 1):
        super().__init__(system)
        if not isinstance(system, WegnerSystem):
            raise TypeError(
                "WilsonLoop measurement only valid for WegnerSystem")
        self.loop_size = loop_size

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
        raise NotImplementedError("WegnerSystem is not implemented yet.")
