from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class PreprocessResult:
    """Container for Huth-style temporal preprocessing outputs."""

    cleaned: np.ndarray
    kept_ranges: List[Tuple[int, int]]
    segment_means: List[np.ndarray]
    segment_stds: List[np.ndarray]
    win_len_tr: int
    trim_tr: int
    tr: float
    kept_indices: Optional[List[np.ndarray]] = None


def _validate_segments(total_len: int, segment_bounds: Optional[Sequence[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    if segment_bounds is None:
        return [(0, total_len)]
    bounds: List[Tuple[int, int]] = []
    for start, end in segment_bounds:
        if not (0 <= start < end <= total_len):
            raise ValueError(f"Invalid segment bounds {(start, end)} for length {total_len}.")
        bounds.append((int(start), int(end)))
    return bounds


def _compute_window_length_tr(window_s: float, tr: float, polyorder: int, seg_len: int) -> int:
    win_len_tr = int(round(window_s / tr))
    min_len = polyorder + 2
    if win_len_tr < min_len:
        raise ValueError(
            f"Savitzky–Golay window ({win_len_tr} TR) is too short for polyorder {polyorder}. "
            "Increase window_s or decrease polyorder."
        )
    if win_len_tr % 2 == 0:
        win_len_tr += 1
    if win_len_tr > seg_len:
        raise ValueError(
            f"Savitzky–Golay window ({win_len_tr} TR) longer than segment ({seg_len} TR). "
            "Reduce window_s or adjust segment definitions."
        )
    return win_len_tr


def preprocess_huth_style(
    ts: ArrayLike,
    *,
    tr: float = 2.0,
    window_s: float = 120.0,
    polyorder: int = 2,
    trim_s: float = 20.0,
    zscore: bool = True,
    segment_bounds: Optional[Sequence[Tuple[int, int]]] = None,
    return_tr_indices: bool = False,
    eps: float = 1e-8,
) -> PreprocessResult:
    """Apply Huth-style temporal cleaning to a time series."""

    arr = np.asarray(ts, dtype=float)
    original_ndim = arr.ndim
    if original_ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}.")

    n_tr, n_feat = arr.shape
    bounds = _validate_segments(n_tr, segment_bounds)
    trim_tr = int(round(trim_s / tr))
    if trim_tr < 0:
        raise ValueError("trim_s must be non-negative.")

    cleaned_segments: List[np.ndarray] = []
    kept_ranges: List[Tuple[int, int]] = []
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    kept_indices: List[np.ndarray] = []
    win_lengths: List[int] = []

    for start, end in bounds:
        seg = arr[start:end]
        seg_len = seg.shape[0]
        if trim_tr * 2 >= seg_len:
            raise ValueError(
                f"Segment [{start}, {end}) too short ({seg_len} TR) for trimming {trim_tr} TR at both ends."
            )

        win_len_tr = _compute_window_length_tr(window_s, tr, polyorder, seg_len)
        trend = savgol_filter(seg, window_length=win_len_tr, polyorder=polyorder, axis=0, mode="interp")
        detrended = seg - trend

        cleaned_seg = detrended[trim_tr: seg_len - trim_tr]
        kept_start = start + trim_tr
        kept_end = end - trim_tr

        if zscore:
            mean = cleaned_seg.mean(axis=0)
            std = cleaned_seg.std(axis=0)
            std = np.where(std < eps, 1.0, std)
            cleaned_seg = (cleaned_seg - mean) / std
        else:
            mean = np.zeros(n_feat, dtype=float)
            std = np.ones(n_feat, dtype=float)

        cleaned_segments.append(cleaned_seg)
        kept_ranges.append((kept_start, kept_end))
        means.append(mean)
        stds.append(std)
        if return_tr_indices:
            kept_indices.append(np.arange(kept_start, kept_end, dtype=int))
        win_lengths.append(win_len_tr)

    cleaned = np.vstack(cleaned_segments) if cleaned_segments else np.zeros((0, n_feat), dtype=float)
    if original_ndim == 1:
        cleaned = cleaned[:, 0]

    kept_idx_list = kept_indices if return_tr_indices else None

    return PreprocessResult(
        cleaned=cleaned,
        kept_ranges=kept_ranges,
        segment_means=means,
        segment_stds=stds,
        win_len_tr=win_lengths[0] if win_lengths else _compute_window_length_tr(window_s, tr, polyorder, n_tr),
        trim_tr=trim_tr,
        tr=tr,
        kept_indices=kept_idx_list,
    )


def preprocess_targets_huth_style(y: ArrayLike, **kwargs) -> PreprocessResult:
    """Convenience wrapper for 1-D semantic target series."""

    return preprocess_huth_style(y, **kwargs)


def quick_drift_diagnostics(
    raw_ts: np.ndarray,
    cleaned_ts: np.ndarray,
    *,
    tr: float,
    segment_bounds: Optional[Sequence[Tuple[int, int]]] = None,
    n_features: int = 3,
):
    """Plot raw vs. cleaned series for inspection."""

    import matplotlib.pyplot as plt

    raw = np.asarray(raw_ts, dtype=float)
    cleaned = np.asarray(cleaned_ts, dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    if cleaned.ndim == 1:
        cleaned = cleaned[:, None]

    n_feat = raw.shape[1]
    idxs = np.linspace(0, n_feat - 1, min(n_features, n_feat), dtype=int)
    time = np.arange(raw.shape[0]) * tr

    fig, axes = plt.subplots(len(idxs), 1, figsize=(10, 3 * len(idxs)), sharex=True)
    if len(idxs) == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, idxs):
        ax.plot(time, raw[:, feat_idx], label="Raw", alpha=0.5)
        ax.plot(np.arange(cleaned.shape[0]) * tr, cleaned[:, feat_idx], label="Cleaned", linewidth=1.5)
        ax.set_ylabel(f"Feature {feat_idx}")
        ax.grid(alpha=0.3)
        if segment_bounds:
            for start, end in segment_bounds:
                ax.axvline(start * tr, color="gray", linestyle="--", alpha=0.2)
                ax.axvline(end * tr, color="gray", linestyle="--", alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tr = 2.0

    seg1_len = 200
    seg2_len = 220
    t1 = np.arange(seg1_len) * tr
    t2 = np.arange(seg2_len) * tr

    seg1 = 0.2 * t1[:, None] + rng.normal(scale=0.5, size=(seg1_len, 3))
    seg2 = -0.15 * t2[:, None] + rng.normal(scale=0.6, size=(seg2_len, 3))

    raw = np.vstack([seg1, seg2])
    segments = [(0, seg1_len), (seg1_len, seg1_len + seg2_len)]

    def slope(arr):
        x = np.arange(arr.shape[0])
        return np.polyfit(x, arr, 1)[0]

    slopes_before = [slope(raw[:, i]) for i in range(raw.shape[1])]

    res = preprocess_huth_style(
        raw,
        tr=tr,
        segment_bounds=segments,
        return_tr_indices=True,
    )

    slopes_after = [slope(res.cleaned[:, i]) for i in range(res.cleaned.shape[1])]

    print("Raw slopes:", np.round(slopes_before, 4))
    print("Cleaned slopes:", np.round(slopes_after, 4))
    print("Original shape:", raw.shape, "Cleaned shape:", res.cleaned.shape)
    print("Kept ranges:", res.kept_ranges)

    target = np.sin(np.linspace(0, 6 * np.pi, raw.shape[0])) + 0.01 * np.arange(raw.shape[0])
    target_res = preprocess_targets_huth_style(
        target,
        tr=tr,
        segment_bounds=segments,
    )
    print("Target cleaned shape:", target_res.cleaned.shape)
