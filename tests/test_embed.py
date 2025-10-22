import numpy as np

from src.edm import horizon_shift


def test_horizon_shift_delta_two():
    y = np.arange(10.0)
    shifted = horizon_shift(y, 2)
    assert shifted.shape[0] == 8
    assert shifted[0] == 2.0
