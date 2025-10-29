from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import ensure_same_length, set_seed

_EPS = 1e-12


def horizon_shift(y: np.ndarray, delta: int) -> np.ndarray:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    series = np.asarray(y, dtype=float).ravel()
    if delta == 0:
        return series.copy()
    if series.size <= delta:
        raise ValueError("delta exceeds series length")
    return series[delta:]


def embed_multivariate(X: np.ndarray, E: int, tau: int) -> np.ndarray:
    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    max_lag = (E - 1) * tau
    if data.shape[0] <= max_lag:
        raise ValueError("Time series too short for embedding")
    lagged = [data[max_lag - j * tau : data.shape[0] - j * tau] for j in range(E)]
    return np.hstack(lagged)


def _candidate_neighbors(X: np.ndarray, k: int, theiler: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    n_samples = X.shape[0]
    if n_samples <= k:
        raise ValueError("k must be smaller than number of samples")
    buffer = max(5, 2 * theiler + 5)
    n_neighbors = min(n_samples, k + buffer)
    set_seed(0)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(X)
    return nbrs.kneighbors(X)


def _select_neighbors(dist_row: np.ndarray, idx_row: np.ndarray, t: int, k: int, theiler: int) -> tuple[np.ndarray, np.ndarray]:
    mask = idx_row != t
    if theiler > 0:
        mask &= np.abs(idx_row - t) > theiler
    idx_valid = idx_row[mask]
    dist_valid = dist_row[mask]
    if idx_valid.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    if idx_valid.size > k:
        idx_valid = idx_valid[:k]
        dist_valid = dist_valid[:k]
    return idx_valid, dist_valid


def simplex(X: np.ndarray, y: np.ndarray, k: int, theiler: int = 0) -> np.ndarray:
    X, y = ensure_same_length(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    dist_full, idx_full = _candidate_neighbors(X, k, theiler)
    preds = np.empty(X.shape[0])
    for t in range(X.shape[0]):
        neigh_idx, neigh_dist = _select_neighbors(dist_full[t], idx_full[t], t, k, theiler)
        if neigh_idx.size == 0:
            preds[t] = y[t]
            continue
        base = max(neigh_dist[0], _EPS)
        weights = np.exp(-(neigh_dist / base))
        weights /= weights.sum()
        preds[t] = float(np.dot(weights, y[neigh_idx]))
    return preds


def smap(X: np.ndarray, y: np.ndarray, k: int, theta: float, theiler: int = 0) -> np.ndarray:
    X, y = ensure_same_length(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    dist_full, idx_full = _candidate_neighbors(X, k, theiler)
    n_samples, n_features = X.shape
    preds = np.empty(n_samples)
    ridge = 1e-4
    for t in range(n_samples):
        neigh_idx, neigh_dist = _select_neighbors(dist_full[t], idx_full[t], t, k, theiler)
        if neigh_idx.size == 0:
            preds[t] = y[t]
            continue
        Xn = X[neigh_idx]
        yn = y[neigh_idx]
        scale = max(neigh_dist[0], _EPS)
        w = np.exp(-theta * neigh_dist / scale)
        w = w / np.sum(w)
        X_aug = np.c_[np.ones(Xn.shape[0]), Xn]
        WX = X_aug * w[:, None]
        A = WX.T @ X_aug + ridge * np.eye(X_aug.shape[1])
        b = WX.T @ yn
        coeff = np.linalg.solve(A, b)
        preds[t] = coeff[0] + X[t] @ coeff[1:]
    return preds


def corr_skill(yhat: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    if method.lower() != "pearson":
        raise ValueError("Only Pearson correlation is supported")
    yhat, y = ensure_same_length(np.asarray(yhat, dtype=float), np.asarray(y, dtype=float))
    if np.std(yhat) < _EPS or np.std(y) < _EPS:
        return 0.0
    return float(np.corrcoef(yhat, y)[0, 1])
