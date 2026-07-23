import tensorflow as tf
from typing import Optional, TYPE_CHECKING
from .base import Measurement

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class Energy(Measurement):
    """
    Computes the total energy of the system.
    Requires a system instance to access the Hamiltonian logic.
    Returns a tensor of shape (replicas,).
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        if sys is None:
            raise ValueError(
                "Energy computation requires a system (Hamiltonian logic).")

        energy = sys.compute_energy(state)
        if sys.quenched_replicas == 1:
            return tf.squeeze(energy, axis=0)
        return energy


class Magnetization(Measurement):
    """
    Computes the average magnetization per site for each replica.
    Can function with just a spin_state (inferring replicas from shape).
    Returns a tensor of shape (replicas,).
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        # Resolve replicas: System metadata > Tensor shape
        Q = sys.quenched_replicas if sys else state.shape[0]
        R = sys.lattice_replicas if sys else state.shape[1]

        flat_state = tf.reshape(state, (Q, R, -1))
        mag = tf.reduce_mean(flat_state, axis=2)
        
        if sys and sys.quenched_replicas == 1:
            return tf.squeeze(mag, axis=0)
        elif not sys and Q == 1:
            return tf.squeeze(mag, axis=0)
        return mag


class MagneticSusceptibility(Measurement):
    """
    Computes the variance of the spin state magnetizations across replicas.
    Can function with just a spin_state.
    Returns a scalar.
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        Q = sys.quenched_replicas if sys else state.shape[0]
        R = sys.lattice_replicas if sys else state.shape[1]

        flat_state = tf.reshape(state, (Q, R, -1))
        m_per_replica = tf.reduce_mean(flat_state, axis=2)
        suscep = tf.math.reduce_variance(m_per_replica, axis=1)
        
        if sys and sys.quenched_replicas == 1:
            return tf.squeeze(suscep, axis=0)
        elif not sys and Q == 1:
            return tf.squeeze(suscep, axis=0)
        return suscep


class SpinGlassOrderParameter(Measurement):
    """
    Computes the Edwards-Anderson spin glass order parameter: q_EA = [1/N Σ ⟨s_i⟩²]_avg.
    Requires stateful accumulation over a sweep to compute ⟨s_i⟩.
    Must be instantiated with a system to initialize the state tracker.
    Returns a scalar.
    """

    def __init__(self, system: Optional['BaseSpinSystem'] = None) -> None:
        super().__init__(system)
        if system is None:
            raise ValueError(
                "SpinGlassOrderParameter requires a system at initialization to shape the accumulation variables."
            )
        self.spin_sum = tf.Variable(
            tf.zeros_like(system.spin_state, dtype=tf.float32), trainable=False
        )
        self.count = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        self.spin_sum.assign_add(tf.cast(state, tf.float32))
        self.count.assign_add(1.0)

        s_avg = self.spin_sum / self.count
        
        axes_to_reduce = tf.range(2, tf.rank(s_avg))
        q_ea_per_replica = tf.reduce_mean(tf.square(s_avg), axis=axes_to_reduce)
        q_ea = tf.reduce_mean(q_ea_per_replica, axis=1)

        if sys and sys.quenched_replicas == 1:
            return tf.squeeze(q_ea, axis=0)
        return q_ea

    def reset(self):
        """Reset the accumulation variables to start a new measurement period."""
        self.spin_sum.assign(tf.zeros_like(self.spin_sum))
        self.count.assign(0.0)
