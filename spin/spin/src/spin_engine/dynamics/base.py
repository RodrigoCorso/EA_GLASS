import abc
import tensorflow as tf

from typing import Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.dynamics.tracker import Tracker


class Dynamics(abc.ABC):
    """
    Abstract base class for all dynamics

    The Dynamics dictates how the spin state evolves over time.
    """

    def __init__(self, system: 'BaseSpinSystem', ) -> None:
        self.system = system

    @abc.abstractmethod
    def step(
        self,
        beta: float,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> None:
        """
        How a step is taken inside our simulation. This method should be called in the main loop of the simulation.
        """
        pass

    # TODO: Add annealing inside the sweep to avoid retracing.

    @abc.abstractmethod
    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        sweep_length: int,
        num_disturbances: tf.Tensor = cast(tf.Tensor, 1),
        theta_max: Optional[tf.Tensor] = None,
    ) -> None:
        """
        The orchestrator of multiple steps of the simulation.
        """
        pass
