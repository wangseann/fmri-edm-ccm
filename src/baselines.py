from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .utils import ensure_same_length, set_seed


def ridge_forecast(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    set_seed(seed)
    X, y = ensure_same_length(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)
    return model.predict(X)
