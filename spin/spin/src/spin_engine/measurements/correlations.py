import tensorflow as tf
from typing import Optional, TYPE_CHECKING
from .base import Measurement

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class OverlapMatrix(Measurement):
    """
    Computes the overlap matrix between all replicas.
    Q_ab = (1/N) * sum_i s_i^a * s_i^b

    Returns: Tensor of shape (replicas, replicas)
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        Q = sys.quenched_replicas if sys else state.shape[0]
        R = sys.lattice_replicas if sys else state.shape[1]

        if sys is not None:
            if hasattr(sys, 'number_spins'):
                n_spins = tf.cast(sys.number_spins, tf.float32)
            else:
                n_spins = tf.cast(tf.reduce_prod(sys.shape), tf.float32)
        else:
            n_spins = tf.cast(tf.reduce_prod(state.shape[2:]), tf.float32)

        spin_flat = tf.cast(tf.reshape(state, (Q, R, -1)), tf.float32)
        overlap = tf.cast(tf.matmul(spin_flat, spin_flat, transpose_b=True), tf.float32) / n_spins
        
        if sys and sys.quenched_replicas == 1:
            return tf.squeeze(overlap, axis=0)
        elif not sys and Q == 1:
            return tf.squeeze(overlap, axis=0)
        return overlap


class OverlapDistribution(OverlapMatrix):
    """
    Computes the distribution of off-diagonal overlaps P(q).
    Uses the OverlapMatrix to extract Q_ab for a != b.

    Returns: 1D Tensor of shape (replicas * (replicas - 1) / 2)
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        q_matrix = super().compute(spin_state, system)
        
        has_q_dim = len(q_matrix.shape) == 3
        replicas = q_matrix.shape[-1]

        # Create a boolean mask for the strictly upper triangular part
        mask = tf.linalg.band_part(tf.ones((replicas, replicas), dtype=tf.bool), 0, -1)
        mask = tf.linalg.set_diag(mask, tf.zeros(replicas, dtype=tf.bool))

        if has_q_dim:
            return tf.vectorized_map(lambda qm: tf.boolean_mask(qm, mask), q_matrix)
        else:
            return tf.boolean_mask(q_matrix, mask)


class ParisiOverlapParameter(OverlapDistribution):
    """
    Computes the Parisi overlap parameter: <q^2> - <|q|>^2
    This serves as an indicator of replica symmetry breaking (RSB).

    Returns: Scalar Tensor
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        q_values = super().compute(spin_state, system)

        has_q_dim = len(q_values.shape) == 2
        
        if has_q_dim:
            q_sq_mean = tf.reduce_mean(tf.square(q_values), axis=1)
            abs_q_mean = tf.reduce_mean(tf.abs(q_values), axis=1)
        else:
            q_sq_mean = tf.reduce_mean(tf.square(q_values))
            abs_q_mean = tf.reduce_mean(tf.abs(q_values))

        return q_sq_mean - tf.square(abs_q_mean)
