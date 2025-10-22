import numpy as np

from src.edm import embed_multivariate


def test_embed_multivariate_shape():
    X = np.random.randn(100, 5)
    result = embed_multivariate(X, E=4, tau=1)
    assert result.shape == (97, 20)
