import pytest
import tensorflow as tf
from spin_engine.measurements.correlations import OverlapDistribution, ParisiOverlapParameter
from spin_engine.measurements.scalars import SpinGlassOrderParameter


class DummySystem:
    def __init__(self, state):
        self.spin_state = tf.Variable(state)
        self.shape = state.shape[2:]
        self.lattice_replicas = state.shape[1]
        self.quenched_replicas = state.shape[0]


def test_overlap_distribution():
    state = tf.constant([[
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0]
    ]])
    
    sys = DummySystem(state)
    dist = OverlapDistribution(sys)
    q_vals = dist.compute()
    
    assert q_vals.shape == (3,)
    
    q_vals_np = list(q_vals.numpy().flatten())
    assert q_vals_np.count(0.0) == 2
    assert q_vals_np.count(-1.0) == 1


def test_parisi_overlap_parameter():
    state = tf.constant([[
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0]
    ]])
    
    sys = DummySystem(state)
    parisi = ParisiOverlapParameter(sys)
    val = parisi.compute()
    
    val_np = val.numpy()
    assert pytest.approx(val_np, 0.001) == 2.0 / 9.0


def test_spin_glass_order_parameter():
    state = tf.constant([[
        [1.0, 1.0],
        [-1.0, 1.0]
    ]])
    sys = DummySystem(state)
    q_ea = SpinGlassOrderParameter(sys)
    
    val1 = q_ea.compute()
    assert pytest.approx(val1.numpy()) == 1.0
    
    state2 = tf.constant([[
        [-1.0, -1.0],
        [-1.0, 1.0]
    ]])
    val2 = q_ea.compute(spin_state=state2)
    assert pytest.approx(val2.numpy()) == 0.5
    
    q_ea.reset()
    assert q_ea.count.numpy() == 0.0
