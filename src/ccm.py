from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression

from .edm import corr_skill, embed_multivariate

_EPS = 1e-12


def _shadow(series: np.ndarray, E: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    emb = embed_multivariate(series, E, tau)
    trimmed = np.asarray(series, dtype=float)
    offset = (E - 1) * tau
    return emb, trimmed[offset:]


def _simplex_from_library(manifold: np.ndarray, target: np.ndarray, L: int, theiler: int) -> float:
    L = min(L, manifold.shape[0])
    preds = np.empty(L)
    base_idx = np.arange(L)
    for i, idx in enumerate(base_idx):
        dists = np.linalg.norm(manifold[idx] - manifold[base_idx], axis=1)
        mask = (np.abs(base_idx - idx) > theiler) & (base_idx != idx)
        valid = base_idx[mask]
        if valid.size == 0:
            preds[i] = target[idx]
            continue
        order = np.argsort(dists[mask])
        valid = valid[order]
        neighbor_dists = dists[mask][order]
        k = min(len(valid), manifold.shape[1] + 1)
        valid = valid[:k]
        neighbor_dists = neighbor_dists[:k]
        d1 = max(neighbor_dists[0], _EPS)
        weights = np.exp(-neighbor_dists / d1)
        weights /= weights.sum()
        preds[i] = np.sum(weights * target[valid])
    return corr_skill(preds, target[:L])


def ccm_pair(
    X: np.ndarray,
    Y: np.ndarray,
    E: int,
    tau: int,
    theiler: int,
    L_grid: List[int],
) -> Dict[str, float | List[float]]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("Inputs must be 1-D")
    X, Y = np.asarray(X), np.asarray(Y)
    emb_X, Y_aligned = _shadow(X, E, tau)
    emb_Y, X_aligned = _shadow(Y, E, tau)
    Ls = [min(L, emb_X.shape[0]) for L in L_grid]
    skill_curve = [_simplex_from_library(emb_X, Y_aligned, L, theiler) for L in Ls]
    convergence = float(np.polyfit(Ls, skill_curve, 1)[0]) if len(Ls) > 1 else 0.0
    reverse_skill = _simplex_from_library(emb_Y, X_aligned, Ls[-1], theiler)
    poslag_asym = float(skill_curve[-1] - reverse_skill)
    return {"skill_curve": skill_curve, "convergence": convergence, "poslag_asym": poslag_asym}


def _residualize(signal: np.ndarray, drivers: np.ndarray) -> np.ndarray:
    if drivers.size == 0:
        return signal - np.mean(signal)
    model = LinearRegression(fit_intercept=True)
    model.fit(drivers, signal)
    return signal - model.predict(drivers)


def ccm_conditional_screen(
    X_mat: np.ndarray,
    y: np.ndarray,
    cond: List[np.ndarray],
    E: int,
    tau: int,
    theiler: int,
    L_grid: List[int],
) -> List[int]:
    X_mat = np.asarray(X_mat, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_mat.ndim != 2:
        raise ValueError("X_mat must be 2-D")
    min_len = min([X_mat.shape[0], y.shape[0]] + [np.asarray(c).shape[0] for c in cond])
    X_trim = X_mat[:min_len]
    y_trim = y[:min_len]
    drivers = np.column_stack([np.asarray(c, dtype=float)[:min_len] for c in cond]) if cond else np.empty((min_len, 0))
    y_res = _residualize(y_trim, drivers)
    scores = []
    for idx in range(X_trim.shape[1]):
        x_res = _residualize(X_trim[:, idx], drivers)
        result = ccm_pair(x_res, y_res, E, tau, max(theiler, E), L_grid)
        score = float(result["convergence"]) + 0.5 * float(result["poslag_asym"])
        scores.append((score, idx))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scores]
