"""ROI-space encoding utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ROIEncoder:
    """Fits a ridge regression model from semantic PCs to ROI activity."""

    def __init__(self, alpha: float = 1.0):
        """Configure ridge model and scalers for ROI prediction."""
        self.alpha = float(alpha)
        self._model = Ridge(alpha=self.alpha)
        self._x_scaler = StandardScaler(with_mean=True, with_std=True)
        self._y_scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted = False

    def fit(self, X_sem: np.ndarray, Y_roi: np.ndarray) -> "ROIEncoder":
        X_sem = np.asarray(X_sem, dtype=float)
        Y_roi = np.asarray(Y_roi, dtype=float)
        X_t = self._x_scaler.fit_transform(X_sem)
        Y_t = self._y_scaler.fit_transform(Y_roi)
        self._model.fit(X_t, Y_t)
        self._fitted = True
        return self

    def predict(self, X_sem: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ROIEncoder must be fit before calling predict")
        X_sem = np.asarray(X_sem, dtype=float)
        X_t = self._x_scaler.transform(X_sem)
        Y_t = self._model.predict(X_t)
        return self._y_scaler.inverse_transform(Y_t)
