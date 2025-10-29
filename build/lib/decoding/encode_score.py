"""Scoring utilities to compare predicted and candidate encodings."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cross_decomposition import CCA


def _prepare_pair(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    mask = np.isfinite(A) & np.isfinite(B)
    if mask.ndim == 2:
        row_mask = mask.all(axis=1)
        A = A[row_mask]
        B = B[row_mask]
    else:
        row_mask = mask
        A = A[row_mask]
        B = B[row_mask]
    return A, B


def mean_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the mean Pearson correlation across columns."""
    A, B = _prepare_pair(A, B)
    if A.size == 0 or A.shape[0] < 2:
        return float("nan")
    corrs = []
    for col_idx in range(A.shape[1]):
        a_col = A[:, col_idx]
        b_col = B[:, col_idx]
        if np.std(a_col) == 0 or np.std(b_col) == 0:
            continue
        a_norm = a_col - a_col.mean()
        b_norm = b_col - b_col.mean()
        denom = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
        if denom == 0:
            continue
        corrs.append(float(np.dot(a_norm, b_norm) / denom))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def cca_corr(A: np.ndarray, B: np.ndarray, n_comp: int) -> float:
    """Mean canonical correlation using up to ``n_comp`` components."""
    A, B = _prepare_pair(A, B)
    if A.shape[0] < 2:
        return float("nan")
    n_comp = int(max(1, min(n_comp, A.shape[1], B.shape[1], A.shape[0] - 1)))
    if n_comp <= 0:
        return float("nan")
    cca = CCA(n_components=n_comp)
    try:
        X_c, Y_c = cca.fit_transform(A, B)
    except Exception:
        return float("nan")
    corrs = []
    for idx in range(n_comp):
        x = X_c[:, idx]
        y = Y_c[:, idx]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        x_norm = x - x.mean()
        y_norm = y - y.mean()
        denom = np.linalg.norm(x_norm) * np.linalg.norm(y_norm)
        if denom == 0:
            continue
        corrs.append(float(np.dot(x_norm, y_norm) / denom))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Compute the mean cosine similarity across rows."""
    A, B = _prepare_pair(A, B)
    if A.size == 0 or A.shape[0] == 0:
        return float("nan")
    cosines = []
    for row_idx in range(A.shape[0]):
        a_row = A[row_idx]
        b_row = B[row_idx]
        denom = np.linalg.norm(a_row) * np.linalg.norm(b_row)
        if denom == 0:
            continue
        cosines.append(float(np.dot(a_row, b_row) / denom))
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def pc_encoding_score(
    pc_TR_pred: np.ndarray,
    pc_TR_cand: np.ndarray,
    method: str = "mean",
    cca_components: Optional[int] = None,
) -> float:
    """Score PC trajectories using the requested method."""
    method = method.lower()
    if method == "mean":
        return mean_corr(pc_TR_pred, pc_TR_cand)
    if method == "cca":
        n_comp = cca_components if cca_components is not None else min(pc_TR_pred.shape[1], 5)
        return cca_corr(pc_TR_pred, pc_TR_cand, n_comp)
    if method == "cosine":
        return mean_cosine(pc_TR_pred, pc_TR_cand)
    raise ValueError(f"Unknown PC encoding method: {method}")


def roi_encoding_score(
    roi_TR_pred: np.ndarray,
    roi_TR_cand: np.ndarray,
    method: str = "mean",
    cca_components: Optional[int] = None,
) -> float:
    """Score ROI trajectories similar to ``pc_encoding_score``."""
    return pc_encoding_score(roi_TR_pred, roi_TR_cand, method=method, cca_components=cca_components)
