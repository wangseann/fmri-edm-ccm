import numpy as np

from src.edm import simplex


def test_simplex_prediction_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6))
    y = rng.normal(size=150)
    yhat = simplex(X, y, k=10)
    assert yhat.shape == y.shape
