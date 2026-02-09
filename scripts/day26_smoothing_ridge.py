#!/usr/bin/env python3
"""Run Day26-style smoothing + Ridge baseline sweep from the command line."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

CURRENT_DIR = Path(__file__).resolve().parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
PROJECT_PARENT = CURRENT_DIR.parent
pyedm_src = PROJECT_PARENT / "pyEDM" / "src"
mde_src = PROJECT_PARENT / "MDE-main" / "src"
for extra in (pyedm_src, mde_src):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.append(str(extra))
    elif not extra.exists():
        warnings.warn(f"Optional dependency path not found: {extra}")

from src.utils import load_yaml
from src.day19_category_builder import (
    generate_category_time_series,
    build_smoothing_kernel,
    apply_smoothing_kernel,
)
from src.day24_subject_concat import (
    build_story_inventory,
    concatenate_subject_timeseries,
    load_subject_concat_manifest,
    save_subject_concat,
    DEFAULT_FEATURES_ROOT as DAY24_DEFAULT_FEATURES_ROOT,
    DEFAULT_OUTPUT_SUBDIR as DAY24_DEFAULT_OUTPUT_SUBDIR,
    _resolve_cache_dir as day24_resolve_cache_dir,
)
from src.day22_category_mde import (
    sanitize_name,
    make_lag_dict,
    make_splits,
)
from src.preprocessing.huth import preprocess_huth_style
from src import roi


# ---------------------------------------------------------------------
# Helper functions reused from your MDE script
# ---------------------------------------------------------------------


def update_span_metric_summary(
    base_dir: Path,
    config_name: str,
    category: str,
    metric_key: str,
    values: Dict[str, float],
    ylabel: str,
    title_prefix: Optional[str] = None,
) -> None:
    if not values:
        return

    metric_key = metric_key.lower()
    title_prefix = title_prefix or ylabel
    summary_csv = base_dir / f"{config_name}_{metric_key}_summary.csv"
    column_names = [f"{metric_key}_train", f"{metric_key}_validation", f"{metric_key}_test"]
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = df[df["category"] != category]
    else:
        df = pd.DataFrame(columns=["category", *column_names])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "category": category,
                        column_names[0]: values.get("train", np.nan),
                        column_names[1]: values.get("validation", values.get("val", np.nan)),
                        column_names[2]: values.get("test", np.nan),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df.sort_values("category", inplace=True)
    df.to_csv(summary_csv, index=False)

    if not df.empty:
        categories = df["category"].tolist()
        x = np.arange(len(categories))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), 5))
        ax.bar(x - width, df[column_names[0]], width, label="Train")
        ax.bar(x, df[column_names[1]], width, label="Validation")
        ax.bar(x + width, df[column_names[2]], width, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} by span – {config_name}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(base_dir / f"{config_name}_{metric_key}_summary.png", dpi=180)
        plt.close(fig)


def update_rmse_summary(base_dir: Path, config_name: str, category: str, rmse: Dict[str, float]) -> None:
    update_span_metric_summary(
        base_dir=base_dir,
        config_name=config_name,
        category=category,
        metric_key="rmse",
        values=rmse,
        ylabel="RMSE",
        title_prefix="RMSE",
    )


def build_segment_bounds(total_rows: int, manifest_data: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int]]:
    """
    Construct half-open segment boundaries for temporal preprocessing.
    """
    if total_rows <= 0:
        return []
    if not manifest_data:
        return [(0, total_rows)]

    bounds: List[Tuple[int, int]] = []
    resets_raw = manifest_data.get("lag_reset_indices") or []
    resets = sorted({int(r) for r in resets_raw if 0 < int(r) < total_rows})
    if resets:
        start = 0
        for idx in resets:
            bounds.append((start, idx))
            start = idx
        bounds.append((start, total_rows))
        return bounds

    boundaries_path = manifest_data.get("boundaries_path")
    if boundaries_path and Path(boundaries_path).exists():
        boundaries_df = pd.read_csv(boundaries_path)
        for _, row in boundaries_df.iterrows():
            start_idx = int(row.get("start_index", row.get("start", 0)))
            end_idx = int(row.get("end_index", row.get("end", total_rows - 1))) + 1
            start_idx = max(0, start_idx)
            end_idx = min(total_rows, end_idx)
            if start_idx < end_idx:
                bounds.append((start_idx, end_idx))
        if bounds:
            return bounds

    return [(0, total_rows)]


def fill_segmentwise_nans(matrix: np.ndarray, segments: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Fill NaNs segment-wise using per-column means (fallback to zero)."""

    arr = np.asarray(matrix, dtype=float)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze = True
    filled = arr.copy()
    for start, end in segments:
        seg = filled[start:end]
        if seg.size == 0:
            continue
        mask = np.isnan(seg)
        if not mask.any():
            continue
        col_means = np.nanmean(seg, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(mask)
        seg[inds] = col_means[inds[1]]
        filled[start:end] = seg
    return filled[:, 0] if squeeze else filled


def add_split_background(
    ax,
    time_axis: np.ndarray,
    splits: Optional[Dict[str, Sequence[int]]],
    *,
    alpha: float = 0.10,
    shown: Optional[set] = None,
) -> set:
    """Shade train/val/test spans on an axis using provided time axis."""

    if shown is None:
        shown = set()
    if splits is None:
        return shown

    spans = [
        ("Train", splits.get("train"), "#4CAF50"),
        ("Validation", splits.get("val"), "#FFC107"),
        ("Test", splits.get("test"), "#F44336"),
    ]
    n = len(time_axis)
    for label, span, color in spans:
        if span is None:
            continue
        if isinstance(span, (list, tuple)) and len(span) == 2:
            start_idx, end_idx = int(span[0]), int(span[1])
        else:
            continue
        start_idx = max(0, start_idx)
        end_idx = min(n, end_idx)
        if end_idx <= start_idx or start_idx >= n:
            continue
        start_val = time_axis[start_idx]
        end_val = time_axis[end_idx - 1]
        label_to_use = label if label not in shown else None
        ax.axvspan(start_val, end_val, color=color, alpha=alpha, label=label_to_use)
        shown.add(label)
    return shown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep smoothing configs and run Ridge baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="Path to YAML config.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. UTS01.")
    parser.add_argument("--story", required=True, help="Story ID, e.g. wheretheressmoke.")
    parser.add_argument("--target", default="cat_abstract", help="Category column to model.")
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-stop", type=float, default=1.25)
    parser.add_argument("--window-step", type=float, default=0.25)
    parser.add_argument("--windows", type=float, nargs="*", default=None, help="Explicit smoothing windows (seconds).")
    parser.add_argument("--methods", nargs="*", default=["gaussian", "moving_average"], help="Smoothing methods to include.")
    parser.add_argument("--ccm-samples", type=int, default=10, help="(Unused) kept for CLI compatibility.")
    parser.add_argument("--features-eval-base", type=Path, default=Path("features_day26_eval_cli_ridge"))
    parser.add_argument("--figs-base", type=Path, default=None, help="Override output figs root.")
    parser.add_argument("--top-n-plot", type=int, default=6, help="(Unused for ridge; kept for compatibility.)")
    parser.add_argument("--lib-sizes", type=int, nargs="*", default=None, help="(Unused for ridge.)")
    parser.add_argument("--tau-grid", type=int, nargs="*", default=None, help="Tau grid to define max lag for ROI-lag matrix.")
    parser.add_argument("--max-predictors", type=int, default=None, help="Maximum number of predictors (embedding dimension cap → lags).")
    parser.add_argument("--use-concat", action="store_true", help="Use Day24 concatenated subject-level series instead of a single story.")
    parser.add_argument("--concat-manifest", type=Path, default=None, help="Path to Day24 manifest JSON for the subject.")
    parser.add_argument("--concat-story-label", type=str, default="all_stories", help="Synthetic story label when using concatenated series.")
    parser.add_argument("--concat-features-root", type=Path, default=None, help="Root directory containing per-story category outputs for Day24.")
    parser.add_argument("--concat-output-subdir", type=Path, default=None, help="Relative subdirectory under features root to store Day24 outputs.")
    parser.add_argument("--concat-story-order", nargs="*", default=None, help="Optional explicit story order for concatenation.")
    parser.add_argument("--concat-force", action="store_true", help="Regenerate Day24 concatenation even if manifest exists.")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running Ridge.")
    parser.add_argument(
        "--huth-preproc",
        dest="huth_preproc",
        action="store_true",
        help="Apply Huth-style drift removal / trimming prior to modelling (default).",
    )
    parser.add_argument(
        "--no-huth-preproc",
        dest="huth_preproc",
        action="store_false",
        help="Disable Huth-style temporal preprocessing.",
    )
    parser.set_defaults(huth_preproc=True)
    parser.add_argument(
        "--preproc-window",
        type=float,
        default=120.0,
        help="Savitzky–Golay window (seconds) used when estimating low-frequency drift.",
    )
    parser.add_argument(
        "--preproc-trim",
        type=float,
        default=20.0,
        help="Seconds trimmed from the start and end of each segment after detrending.",
    )
    parser.add_argument(
        "--preproc-polyorder",
        type=int,
        default=2,
        help="Polynomial order for the Savitzky–Golay filter.",
    )
    parser.add_argument(
        "--preproc-zscore",
        dest="preproc_zscore",
        action="store_true",
        help="Z-score each feature after detrending/trimming (default).",
    )
    parser.add_argument(
        "--no-preproc-zscore",
        dest="preproc_zscore",
        action="store_false",
        help="Skip z-scoring after temporal preprocessing.",
    )
    parser.set_defaults(preproc_zscore=True)
    return parser.parse_args()


def resolve_paths(project_root: Path, paths_cfg: Dict[str, str]) -> Dict[str, str]:
    resolved = dict(paths_cfg)
    resolved.setdefault("project_root", str(project_root))
    for key in ("cache", "figs", "results", "data_root"):
        val = resolved.get(key)
        if val and not Path(val).is_absolute():
            resolved[key] = str((project_root / val).resolve())
    return resolved


def build_configs(
    start: float,
    stop: float,
    step: float,
    methods: List[str],
    seconds_bin_width: float,
    temporal_weighting: str,
    override_windows: List[float] | None,
) -> List[Dict[str, Any]]:
    start = max(0.0, start)
    stop = max(start, stop)
    step = max(step, 1e-6)

    if override_windows:
        windows = sorted({round(float(val), 6) for val in override_windows if float(val) >= 0.0})
    else:
        windows = np.round(np.arange(start, stop + 1e-9, step), 6)
    configs: List[Dict[str, Any]] = []

    # Always include "no smoothing" baseline
    configs.append(
        {
            "name": "none_0p00",
            "smoothing_seconds": 0.0,
            "method": "moving_average",
            "gaussian_sigma_seconds": None,
            "pad_mode": "edge",
            "seconds_bin_width": seconds_bin_width,
            "temporal_weighting": temporal_weighting,
        }
    )

    for seconds in windows:
        if math.isclose(seconds, 0.0):
            continue
        tag = f"{seconds:.2f}".replace(".", "p")
        for method in methods:
            method_lower = method.lower()
            if method_lower == "gaussian":
                configs.append(
                    {
                        "name": f"gauss_{tag}",
                        "smoothing_seconds": seconds,
                        "method": "gaussian",
                        "gaussian_sigma_seconds": seconds * 0.5,
                        "pad_mode": "reflect",
                        "seconds_bin_width": seconds_bin_width,
                        "temporal_weighting": temporal_weighting,
                    }
                )
            elif method_lower in {"moving_average", "movavg"}:
                configs.append(
                    {
                        "name": f"movavg_{tag}",
                        "smoothing_seconds": seconds,
                        "method": "moving_average",
                        "gaussian_sigma_seconds": None,
                        "pad_mode": "edge",
                        "seconds_bin_width": seconds_bin_width,
                        "temporal_weighting": temporal_weighting,
                    }
                )
            else:
                warnings.warn(f"Skipping unknown smoothing method '{method}'.")
    return configs


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_concat_manifest(
    subject: str,
    *,
    features_root: Path,
    output_subdir: Path,
    n_parcels: int,
    paths_cfg: Dict[str, str],
    cfg_base: Dict[str, Any],
    categories_cfg_base: Dict[str, Any],
    cluster_csv_path: str,
    temporal_weighting_default: str,
    prototype_power: float,
    seconds_bin_width_default: float,
    TR: float,
    story_order: Optional[Sequence[str]],
    force: bool,
) -> Path:
    features_root = features_root.resolve()
    output_subdir = Path(output_subdir)
    output_dir = features_root / "subjects" / subject / output_subdir
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"Using existing Day24 manifest at {manifest_path}")
        return manifest_path

    cache_root = day24_resolve_cache_dir(paths_cfg)
    subject_cache_dir = cache_root / subject
    if not subject_cache_dir.exists():
        raise FileNotFoundError(f"ROI cache directory not found for {subject}: {subject_cache_dir}")

    candidate_stories = sorted([p.name for p in subject_cache_dir.iterdir() if p.is_dir()])
    if not candidate_stories:
        raise ValueError(f"No story ROI caches discovered for {subject} under {subject_cache_dir}")

    if story_order:
        missing_from_cache = [story for story in story_order if story not in candidate_stories]
        if missing_from_cache:
            raise ValueError(f"Requested stories {missing_from_cache} missing Schaefer cache under {subject_cache_dir}")
        candidate_stories = [story for story in story_order if story in candidate_stories]

    subject_dir = features_root / "subjects" / subject
    for story in candidate_stories:
        ensure_dir(subject_dir / story)

    print(f"Building Day24 concatenation for {subject} → {output_dir}")
    inventory = build_story_inventory(
        subject,
        features_root=features_root,
        paths_cfg=paths_cfg,
        n_parcels=n_parcels,
    )
    target_story_set = set(candidate_stories)

    def _needs_generation(row) -> bool:
        return (row.status == "missing_category") and (row.story in target_story_set)

    missing_rows = [row for row in inventory.itertuples() if _needs_generation(row)]
    if missing_rows:
        print(f"Generating Day19 category series for {len(missing_rows)} stories: {[row.story for row in missing_rows]}")
        cluster_csv_use = cluster_csv_path if not cluster_csv_path or Path(cluster_csv_path).exists() else ""
        if cluster_csv_path and not cluster_csv_use:
            warnings.warn(f"Cluster CSV missing at {cluster_csv_path}; proceeding without clusters.")
        for row in missing_rows:
            story_name = row.story
            print(f" - Generating categories for {subject}/{story_name}")
            generate_category_time_series(
                subject,
                story_name,
                cfg_base=cfg_base,
                categories_cfg_base=categories_cfg_base,
                cluster_csv_path=cluster_csv_use,
                temporal_weighting=temporal_weighting_default,
                prototype_weight_power=prototype_power,
                smoothing_seconds=0.0,
                smoothing_method="moving_average",
                gaussian_sigma_seconds=None,
                smoothing_pad="edge",
                seconds_bin_width=seconds_bin_width_default,
                features_root=features_root,
                paths=paths_cfg,
                TR=TR,
                save_outputs=True,
            )

        inventory = build_story_inventory(
            subject,
            features_root=features_root,
            paths_cfg=paths_cfg,
            n_parcels=n_parcels,
        )

    ready_mask = inventory["status"] == "ready"
    ready_stories = set(inventory.loc[ready_mask, "story"].astype(str))
    if story_order:
        missing_ready = [story for story in story_order if story not in ready_stories]
        if missing_ready:
            raise ValueError(f"Stories {missing_ready} remain unavailable after category generation.")
        chosen_order = list(story_order)
    else:
        chosen_order = [story for story in candidate_stories if story in ready_stories]

    if not chosen_order:
        raise ValueError(f"No usable stories found for Day24 concatenation under {features_root}")

    result = concatenate_subject_timeseries(
        subject,
        inventory,
        story_order=chosen_order,
        features_root=features_root,
        paths_cfg=paths_cfg,
        n_parcels=n_parcels,
    )
    saved_paths = save_subject_concat(
        result,
        features_root=features_root,
        output_subdir=output_subdir,
        include_inventory=True,
    )
    manifest_path = Path(saved_paths["manifest_path"]).resolve()
    print(f"Day24 manifest written to {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------
# Ridge-specific helpers
# ---------------------------------------------------------------------


def _segment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if y_true.size < 2:
        rho = float("nan")
    elif np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        rho = float("nan")
    else:
        cov = np.corrcoef(y_true, y_pred)
        rho = float(cov[0, 1])
    cae = float(np.sum(np.abs(y_true - y_pred)))
    return rmse, rho, cae


def run_ridge_no_lag(
    *,
    target_series: np.ndarray,
    roi_matrix: np.ndarray,
    time_axis: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray, Dict[str, Tuple[int, int]], np.ndarray, np.ndarray, float,]:
    """
    Ridge baseline WITHOUT ROI lags.
    Uses the preprocessed ROI time series directly.
    """

    N, n_roi = roi_matrix.shape
    if N != target_series.shape[0] or N != time_axis.shape[0]:
        raise ValueError(f"Length mismatch: target={target_series.shape[0]}, " f"roi={N}, time={time_axis.shape[0]}")

    X = roi_matrix.astype(float)
    y = target_series.astype(float)
    time_trim = time_axis.astype(float)

    # Train/val/test splits on full (no-trim) axis
    splits = make_splits(len(time_trim), train_frac=train_frac, val_frac=val_frac)

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("Empty train/val/test segment for Ridge baseline (no-lag).")

    # Hyperparameter selection on validation
    # Wide log grid to avoid sticking on floor/ceiling
    alphas = np.array(sorted({*np.logspace(-2, 10, 25), 1e12, 1e13}))
    best_alpha = None
    best_val_rmse = float("inf")

    # Train-only standardization to keep feature scales comparable
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_std = (X_train - mu) / sigma
    X_val_std = (X_val - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    X_std = (X - mu) / sigma

    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_std, y_train)
        y_val_pred = model.predict(X_val_std)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            best_alpha = alpha

    model = Ridge(alpha=best_alpha if best_alpha is not None else 1.0, fit_intercept=True)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_std)

    # Metrics per split
    rmse_dict: Dict[str, float] = {}
    rho_dict: Dict[str, float] = {}
    cae_dict: Dict[str, float] = {}

    for label, (start_idx, end_idx) in splits.items():
        seg_true = y[start_idx:end_idx]
        seg_pred = y_pred[start_idx:end_idx]
        if seg_true.size == 0:
            continue
        rmse, rho, cae = _segment_metrics(seg_true, seg_pred)
        rmse_dict[label] = rmse
        if np.isfinite(rho):
            rho_dict[label] = rho
        if np.isfinite(cae):
            cae_dict[label] = cae

    target_trim = y  # full series here

    return rmse_dict, rho_dict, cae_dict, y_pred, splits, time_trim, target_trim, best_alpha


def run_ridge_on_roi_lags(
    *,
    target_series: np.ndarray,
    roi_matrix: np.ndarray,
    time_axis: np.ndarray,
    tau_grid: Sequence[int],
    E_cap: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray, Dict[str, Tuple[int, int]], np.ndarray, np.ndarray, float]:
    """Build ROI–lag matrix, fit Ridge with val-based alpha selection, return metrics and predictions."""

    N, n_roi = roi_matrix.shape
    if N != target_series.shape[0] or N != time_axis.shape[0]:
        raise ValueError(f"Length mismatch: target={target_series.shape[0]}, roi={N}, time={time_axis.shape[0]}")

    roi_cols = [f"roi_{i}" for i in range(n_roi)]
    base = pd.DataFrame(
        {
            "Time": np.arange(1, N + 1, dtype=int),
            "target": target_series.astype(float),
            **{col: roi_matrix[:, i] for i, col in enumerate(roi_cols)},
        }
    )

    max_tau = max(tau_grid)
    max_lag = max_tau * (E_cap - 1)
    if N <= max_lag + 5:
        raise ValueError(f"Not enough samples ({N}) for max lag {max_lag}.")

    lag_store: Dict[str, Dict[int, pd.Series]] = {"target": make_lag_dict(base["target"], max_lag)}
    for col in roi_cols:
        lag_store[col] = make_lag_dict(base[col], max_lag)

    time_trim = time_axis[max_lag:]
    target_trim = lag_store["target"][0].to_numpy(dtype=float)
    if time_trim.shape[0] != target_trim.shape[0]:
        raise RuntimeError("Lag trimming produced mismatched time/target lengths.")

    splits = make_splits(len(time_trim), train_frac=train_frac, val_frac=val_frac)

    lagged_features: Dict[str, np.ndarray] = {}
    for col in roi_cols:
        for lag in range(max_lag + 1):
            series = lag_store[col].get(lag)
            if series is None:
                continue
            lagged_features[f"{col}_lag{lag}"] = series.to_numpy(dtype=float)

    if not lagged_features:
        raise RuntimeError("No lagged predictors constructed for Ridge baseline.")

    X = np.column_stack(list(lagged_features.values()))
    y = target_trim

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("Empty train/val/test segment for Ridge baseline.")

    alphas = np.array(sorted({*np.logspace(-2, 10, 25), 1e12, 1e13}))
    best_alpha = None
    best_val_rmse = float("inf")

    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_std = (X_train - mu) / sigma
    X_val_std = (X_val - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    X_std = (X - mu) / sigma

    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_std, y_train)
        y_val_pred = model.predict(X_val_std)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            best_alpha = alpha

    model = Ridge(alpha=best_alpha if best_alpha is not None else 1.0, fit_intercept=True)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_std)

    rmse_dict: Dict[str, float] = {}
    rho_dict: Dict[str, float] = {}
    cae_dict: Dict[str, float] = {}

    for label, (start_idx, end_idx) in splits.items():
        seg_true = y[start_idx:end_idx]
        seg_pred = y_pred[start_idx:end_idx]
        if seg_true.size == 0:
            continue
        rmse, rho, cae = _segment_metrics(seg_true, seg_pred)
        rmse_dict[label] = rmse
        if np.isfinite(rho):
            rho_dict[label] = rho
        if np.isfinite(cae):
            cae_dict[label] = cae

    return rmse_dict, rho_dict, cae_dict, y_pred, splits, time_trim, target_trim, best_alpha


def refresh_ridge_overlay(
    parent_dir: Path,
    config_name: str,
    subject: str,
    story: str,
    mode_tag: str,
) -> None:
    """
    Build combined overlay (grid) of target vs Ridge predictions across
    categories for a given smoothing config + mode.

    Layout:
        parent_dir/
            <category_name>/
                <config_name>/
                    <mode_tag>/
                        ridge_<mode_tag>_<category>_<config>_prediction.csv
                        ridge_<mode_tag>_<category>_<config>_summary.json
    """
    records: List[Tuple[str, pd.DataFrame, Path, Dict[str, Any]]] = []

    for category_dir in sorted(parent_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        config_dir = category_dir / config_name / mode_tag
        if not config_dir.exists():
            continue

        # Prediction CSV
        prediction_files = list(config_dir.glob(f"ridge_{mode_tag}_*_prediction.csv"))
        if not prediction_files:
            continue

        try:
            df = pd.read_csv(prediction_files[0])
        except Exception:
            continue

        # Optional summary JSON with splits
        summary_files = list(config_dir.glob(f"ridge_{mode_tag}_*_summary.json"))
        summary_data: Dict[str, Any] = {}
        if summary_files:
            try:
                summary_data = json.loads(summary_files[0].read_text())
            except Exception:
                summary_data = {}

        records.append((category_dir.name, df, config_dir, summary_data))

    if not records:
        return

    cols = 3
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    # Turn off unused axes
    for ax in axes[len(records) :]:
        ax.axis("off")

    global_shown: set[str] = set()

    for ax, (category_name, df, config_dir, summary_data) in zip(axes, records):
        # Use time axis if present, else fall back to index
        if "time" in df.columns:
            x_vals = df["time"].to_numpy(dtype=float)
            x_label = "Time (s)"
        elif "trim_index" in df.columns:
            x_vals = df["trim_index"].to_numpy(dtype=float)
            x_label = "Trimmed index"
        else:
            x_vals = np.arange(len(df), dtype=float)
            x_label = "Index"

        alpha_val = summary_data.get("alpha") if isinstance(summary_data, dict) else None
        alpha_str = f", alpha={alpha_val:.3g}" if alpha_val not in (None, float("nan")) else ""

        ax.plot(x_vals, df["target"], label="Target", linewidth=1.1)
        ax.plot(x_vals, df["prediction"], label=f"Ridge ({mode_tag})", linewidth=1.1, alpha=0.85)

        # Add train/val/test shading if we have splits
        splits = summary_data.get("splits") if summary_data else None
        if isinstance(splits, dict):
            # convert back to tuple index spans
            span_dict = {k: (int(v[0]), int(v[1])) for k, v in splits.items()}
            global_shown = add_split_background(
                ax,
                x_vals,  # time axis
                span_dict,
                shown=global_shown,
            )

        ax.set_title(f"{category_name}{alpha_str}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

        # Also save a single-panel version for that category
        panel_dir = config_dir / "ridge_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        fig_single, ax_single = plt.subplots(figsize=(6, 3))
        ax_single.plot(x_vals, df["target"], label="Target", linewidth=1.2)
        ax_single.plot(x_vals, df["prediction"], label=f"Ridge ({mode_tag})", linewidth=1.2, alpha=0.85)
        if isinstance(splits, dict):
            add_split_background(ax_single, x_vals, span_dict)
        ax_single.set_title(f"{category_name}{alpha_str}")
        ax_single.set_xlabel(x_label)
        ax_single.set_ylabel("Value")
        ax_single.grid(alpha=0.3)
        ax_single.legend(loc="upper right")
        fig_single.tight_layout()
        fig_single.savefig(panel_dir / f"{category_name}_ridge_{mode_tag}.png", dpi=180)
        plt.close(fig_single)

    # Put legend once
    axes[0].legend(loc="upper right")
    fig.suptitle(f"{subject}/{story} – {config_name} (Ridge {mode_tag}) overlays", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(parent_dir / f"{config_name}_ridge_{mode_tag}_overlays.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    subject = args.subject.strip()
    story = args.story.strip()
    target_col = args.target.strip()
    target_safe = sanitize_name(target_col)

    paths_cfg = resolve_paths(project_root, cfg.get("paths", {}) or {})

    categories_cfg = cfg.get("categories", {}) or {}
    cluster_csv_rel = categories_cfg.get("cluster_csv_path", "")
    cluster_csv_path = str((project_root / cluster_csv_rel).resolve()) if cluster_csv_rel else ""

    TR = float(cfg.get("TR", 2.0))
    prototype_power = float(categories_cfg.get("prototype_weight_power", 1.0))
    n_parcels = int(cfg.get("n_parcels", 400))
    seconds_bin_width_default = float(categories_cfg.get("seconds_bin_width", 0.05))
    temporal_weighting_default = str(categories_cfg.get("temporal_weighting", "proportional")).lower()

    tau_grid = args.tau_grid if args.tau_grid else cfg.get("tau_grid") or [1, 2]
    if isinstance(tau_grid, (int, float)):
        tau_grid = [int(tau_grid)]
    tau_grid = [int(t) for t in tau_grid]

    max_predictors = int(args.max_predictors) if args.max_predictors is not None else int(cfg.get("E_mult", cfg.get("E_cap", 6)))

    requested_figs_base = args.figs_base.resolve() if args.figs_base else None
    features_eval_base = ensure_dir(args.features_eval_base.resolve())

    use_concat = args.use_concat
    concat_category_df: Optional[pd.DataFrame] = None
    roi_matrix_concat: Optional[np.ndarray] = None
    seconds_bin_width_concat: Optional[float] = None
    concat_segment_bounds: Optional[List[Tuple[int, int]]] = None
    story_concat_label = args.concat_story_label.strip() if args.concat_story_label else "all_stories"

    if use_concat:
        concat_features_root_arg = args.concat_features_root
        if concat_features_root_arg is None:
            cfg_features_root = paths_cfg.get("features_root")
            if cfg_features_root:
                concat_features_root_arg = Path(cfg_features_root)
            else:
                concat_features_root_arg = DAY24_DEFAULT_FEATURES_ROOT
        if not concat_features_root_arg.is_absolute():
            concat_features_root = (project_root / concat_features_root_arg).resolve()
        else:
            concat_features_root = concat_features_root_arg.resolve()

        concat_output_subdir = Path(args.concat_output_subdir) if args.concat_output_subdir else DAY24_DEFAULT_OUTPUT_SUBDIR
        if args.concat_manifest is not None:
            manifest_path = args.concat_manifest.resolve()
        else:
            manifest_path = ensure_concat_manifest(
                subject,
                features_root=concat_features_root,
                output_subdir=concat_output_subdir,
                n_parcels=int(cfg.get("n_parcels", 400)),
                paths_cfg=paths_cfg,
                cfg_base=cfg,
                categories_cfg_base=categories_cfg,
                cluster_csv_path=cluster_csv_path,
                temporal_weighting_default=temporal_weighting_default,
                prototype_power=prototype_power,
                seconds_bin_width_default=seconds_bin_width_default,
                TR=TR,
                story_order=args.concat_story_order,
                force=args.concat_force,
            )
        paths_cfg.setdefault("features_root", str(concat_features_root))
        manifest_data = json.loads(Path(manifest_path).read_text())
        if "categories_path" not in manifest_data or "roi_path" not in manifest_data:
            raise KeyError(f"Manifest at {manifest_path} missing required paths.")
        categories_concat_path = Path(manifest_data["categories_path"]).resolve()
        roi_concat_path = Path(manifest_data["roi_path"]).resolve()
        if not categories_concat_path.exists():
            raise FileNotFoundError(f"Concatenated category CSV not found at {categories_concat_path}")
        if not roi_concat_path.exists():
            raise FileNotFoundError(f"Concatenated ROI array not found at {roi_concat_path}")
        concat_category_df = pd.read_csv(categories_concat_path)
        roi_store = np.load(roi_concat_path, allow_pickle=False)
        if isinstance(roi_store, np.lib.npyio.NpzFile):
            roi_matrix_concat = roi_store["roi"] if "roi" in roi_store.files else roi_store[list(roi_store.files)[0]]
        else:
            roi_matrix_concat = roi_store
        roi_matrix_concat = np.asarray(roi_matrix_concat, dtype=float)
        concat_segment_bounds = build_segment_bounds(len(concat_category_df), manifest_data)
        if "start_sec" in concat_category_df.columns and "end_sec" in concat_category_df.columns and len(concat_category_df) > 1:
            seconds_bin_width_concat = float(concat_category_df["end_sec"].iloc[0] - concat_category_df["start_sec"].iloc[0])
            if not np.isfinite(seconds_bin_width_concat) or seconds_bin_width_concat <= 0:
                seconds_bin_width_concat = float(TR)
        else:
            seconds_bin_width_concat = float(TR)
    else:
        story_concat_label = story

    if use_concat:
        if roi_matrix_concat is None or concat_category_df is None:
            raise RuntimeError("Concatenated data not available after manifest load.")
        base_roi_matrix = np.asarray(roi_matrix_concat, dtype=float)
        base_segment_bounds = concat_segment_bounds or [(0, len(concat_category_df))]
    else:
        base_roi_matrix = np.asarray(
            roi.load_schaefer_timeseries_TR(subject, story, n_parcels, paths_cfg),
            dtype=float,
        )
        base_segment_bounds = [(0, base_roi_matrix.shape[0])]

    story_label_for_outputs = story_concat_label if use_concat else story
    if requested_figs_base is not None:
        figs_base = ensure_dir(requested_figs_base)
    else:
        figs_root = Path(paths_cfg.get("figs", project_root / "figs"))
        figs_base = ensure_dir(figs_root / subject / story_label_for_outputs / "day26_ridge_cli" / target_safe)

    configs = build_configs(
        args.window_start, args.window_stop, args.window_step, args.methods, seconds_bin_width_default, temporal_weighting_default, args.windows
    )
    print(f"Configured {len(configs)} smoothing settings for {subject}/{story} (Ridge baseline).")

    if args.dry_run:
        for cfg_item in configs:
            print(f" - {cfg_item['name']}: method={cfg_item['method']} window={cfg_item['smoothing_seconds']:.2f}s")
        return

    results_by_mode: Dict[str, List[Dict[str, Any]]] = {"nolag": [], "lag": []}

    for cfg_item in configs:
        cfg_name = cfg_item["name"]
        safe_name = sanitize_name(cfg_name)
        smoothing_seconds = float(cfg_item["smoothing_seconds"])
        smoothing_method = str(cfg_item["method"])
        gaussian_sigma = cfg_item.get("gaussian_sigma_seconds")
        pad_mode = cfg_item.get("pad_mode", "edge")
        seconds_bin_width = float(cfg_item.get("seconds_bin_width", seconds_bin_width_default))
        temporal_weighting = str(cfg_item.get("temporal_weighting", temporal_weighting_default))

        print(f"\n=== {cfg_name}: method={smoothing_method} window={smoothing_seconds:.2f}s (Ridge) ===")

        config_features_root = ensure_dir(features_eval_base / safe_name)
        cache_root = ensure_dir(config_features_root / "cache")
        segment_bounds = [tuple(b) for b in base_segment_bounds]
        roi_matrix_base_iter = np.asarray(base_roi_matrix, dtype=float).copy()
        roi_matrix_base_iter = fill_segmentwise_nans(roi_matrix_base_iter, segment_bounds)

        if use_concat:
            story_for_run = story_concat_label
            if concat_category_df is None or roi_matrix_concat is None:
                raise RuntimeError("Concatenated data not available. Run Day24 first or provide manifest.")
            category_df = concat_category_df.copy()
            if "trim_index" not in category_df.columns:
                category_df.insert(0, "trim_index", np.arange(len(category_df), dtype=int))
            category_cols = [c for c in category_df.columns if c.startswith("cat_")]
            if not category_cols:
                raise RuntimeError("No category columns found in concatenated data.")
            if target_col not in category_cols:
                raise RuntimeError(f"Target column {target_col} not present in concatenated data.")
            if smoothing_seconds > 0:
                causal_smoothing = True
                kernel = build_smoothing_kernel(
                    seconds_bin_width_concat or seconds_bin_width,
                    smoothing_seconds,
                    method=smoothing_method,
                    gaussian_sigma_seconds=gaussian_sigma,
                    causal=causal_smoothing,
                )
                smoothed_vals = apply_smoothing_kernel(
                    category_df[category_cols].to_numpy(dtype=float), kernel, pad_mode=pad_mode, causal=causal_smoothing
                )
                category_df.loc[:, category_cols] = smoothed_vals
            cat_vals = category_df[category_cols].to_numpy(dtype=float)
            cat_vals = fill_segmentwise_nans(cat_vals, segment_bounds)
            category_df.loc[:, category_cols] = cat_vals
            category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
            cache_subject_dir = ensure_dir(cache_root / subject / story_for_run)
            roi_output_path = cache_subject_dir / f"schaefer_{n_parcels}.npy"
            local_paths = dict(paths_cfg)
            local_paths["cache"] = str(cache_root)
        else:
            story_for_run = story
            cluster_csv_use = cluster_csv_path if not cluster_csv_path or Path(cluster_csv_path).exists() else ""
            if cluster_csv_path and not cluster_csv_use:
                warnings.warn(f"Cluster CSV missing at {cluster_csv_path}; proceeding without clusters.")
            result = generate_category_time_series(
                subject,
                story,
                cfg_base=cfg,
                categories_cfg_base=categories_cfg,
                cluster_csv_path=cluster_csv_use,
                temporal_weighting=temporal_weighting,
                prototype_weight_power=prototype_power,
                smoothing_seconds=smoothing_seconds,
                smoothing_method=smoothing_method,
                gaussian_sigma_seconds=gaussian_sigma,
                smoothing_pad=pad_mode,
                seconds_bin_width=seconds_bin_width,
                features_root=config_features_root,
                paths=paths_cfg,
                TR=TR,
                save_outputs=False,
            )
            category_df = result["category_df_selected"]
            category_cols = result["category_columns"]
            if not category_cols:
                raise RuntimeError("No category columns generated.")
            if target_col not in category_cols:
                raise RuntimeError(f"Target column {target_col} not found.")
            cat_vals = category_df[category_cols].to_numpy(dtype=float)
            cat_vals = fill_segmentwise_nans(cat_vals, segment_bounds)
            category_df.loc[:, category_cols] = cat_vals
            category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
            cache_subject_dir = ensure_dir(cache_root / subject / story_for_run)
            roi_output_path = cache_subject_dir / f"schaefer_{n_parcels}.npy"
            local_paths = dict(paths_cfg)
            local_paths["cache"] = str(cache_root)

        raw_length = len(category_df)

        if args.huth_preproc:
            cat_array = category_df[category_cols].to_numpy(dtype=float)
            cat_preproc = preprocess_huth_style(
                cat_array,
                tr=TR,
                window_s=args.preproc_window,
                polyorder=args.preproc_polyorder,
                trim_s=args.preproc_trim,
                zscore=args.preproc_zscore,
                segment_bounds=segment_bounds,
                return_tr_indices=True,
            )
            roi_preproc = preprocess_huth_style(
                roi_matrix_base_iter,
                tr=TR,
                window_s=args.preproc_window,
                polyorder=args.preproc_polyorder,
                trim_s=args.preproc_trim,
                zscore=args.preproc_zscore,
                segment_bounds=segment_bounds,
                return_tr_indices=True,
            )
            if cat_preproc.cleaned.shape[0] != roi_preproc.cleaned.shape[0]:
                raise RuntimeError(
                    f"Preprocessed category series ({cat_preproc.cleaned.shape[0]}) and ROI matrix ({roi_preproc.cleaned.shape[0]}) length mismatch."
                )
            if cat_preproc.kept_indices is None or roi_preproc.kept_indices is None:
                raise RuntimeError("Preprocessing must return kept indices; set return_tr_indices=True.")
            cat_indices = np.concatenate(cat_preproc.kept_indices)
            roi_indices = np.concatenate(roi_preproc.kept_indices)
            if not np.array_equal(cat_indices, roi_indices):
                raise RuntimeError("Category and ROI kept indices differ; segment parameters may be inconsistent.")
            category_df = category_df.iloc[cat_indices].reset_index(drop=True)
            category_df.loc[:, category_cols] = cat_preproc.cleaned
            roi_matrix_clean = roi_preproc.cleaned.astype(np.float32, copy=False)
        else:
            roi_matrix_clean = roi_matrix_base_iter.astype(np.float32, copy=False)
            cat_indices = np.arange(len(category_df), dtype=int)

        category_df = category_df.reset_index(drop=True)
        if "trim_index" in category_df.columns:
            category_df["trim_index"] = np.arange(len(category_df), dtype=int)

        clean_length = len(category_df)
        if clean_length == 0:
            raise RuntimeError("No samples remain after preprocessing; adjust trim/window parameters.")

        rows_trimmed = max(0, raw_length - clean_length)

        np.save(roi_output_path, roi_matrix_clean)
        meta_path = category_dir / "huth_preproc_meta.json"
        if args.huth_preproc:
            meta_payload = {
                "win_len_tr": int(cat_preproc.win_len_tr),
                "trim_tr": int(cat_preproc.trim_tr),
                "zscore": bool(args.preproc_zscore),
                "segment_bounds": [[int(s), int(e)] for s, e in segment_bounds],
                "kept_ranges": [[int(s), int(e)] for s, e in cat_preproc.kept_ranges],
                "original_rows": int(raw_length),
                "clean_rows": int(clean_length),
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2))
        elif meta_path.exists():
            meta_path.unlink()

        target_series = category_df[target_col].astype(float)
        target_std = float(target_series.std(ddof=1)) if len(target_series) > 1 else float("nan")
        target_range = float(target_series.max() - target_series.min()) if not category_df.empty else float("nan")
        target_diff = target_series.diff().dropna()
        target_diff_mean = float(target_diff.abs().mean()) if not target_diff.empty else 0.0

        top_cols = category_cols[:12]
        fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
        axes = axes.flatten()
        time_axis_full = category_df["start_sec"]
        for idx, col in enumerate(top_cols):
            ax = axes[idx]
            ax.plot(time_axis_full, category_df[col], linewidth=1.0)
            ax.set_title(col)
            ax.grid(alpha=0.3)
        for idx in range(len(top_cols), len(axes)):
            axes[idx].axis("off")
        fig.suptitle(f"{subject} / {story_for_run} – {cfg_name} smoothing (Ridge)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        plot_dir = ensure_dir(figs_base / safe_name)
        plot_path = plot_dir / "category_timeseries_overview.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved category plot to {plot_path}")

        category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
        category_df.to_csv(category_dir / "category_timeseries.csv", index=False)

        # ----------------- Ridge baselines: no-lag and ROI–lag -----------------
        for mode in ("nolag", "lag"):
            if mode == "nolag":
                rmse_info, rho_span, cae_span, y_pred, splits, time_trim, target_trim, best_alpha = run_ridge_no_lag(
                    target_series=category_df[target_col].astype(float).to_numpy(),
                    roi_matrix=roi_matrix_clean,
                    time_axis=category_df["start_sec"].to_numpy(dtype=float),
                    train_frac=0.5,
                    val_frac=0.25,
                )
            else:
                rmse_info, rho_span, cae_span, y_pred, splits, time_trim, target_trim, best_alpha = run_ridge_on_roi_lags(
                    target_series=category_df[target_col].astype(float).to_numpy(),
                    roi_matrix=roi_matrix_clean,
                    time_axis=category_df["start_sec"].to_numpy(dtype=float),
                    tau_grid=tau_grid,
                    E_cap=max_predictors,
                    train_frac=0.5,
                    val_frac=0.25,
                )

            mode_tag = mode  # "nolag" or "lag"
            ridge_dir = ensure_dir(plot_dir / mode_tag)

            # Prediction dataframe & overlay plots
            prediction_df = pd.DataFrame(
                {
                    "trim_index": np.arange(len(time_trim), dtype=int),
                    "time": time_trim,
                    "target": target_trim,
                    "prediction": y_pred.astype(float),
                }
            )
            prediction_csv_path = ridge_dir / f"ridge_{mode_tag}_{target_safe}_{safe_name}_prediction.csv"
            prediction_df.to_csv(prediction_csv_path, index=False)
            # Save a tiny summary JSON with splits (for overlay shading)
            summary_payload = {
                "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
                "alpha": float(best_alpha) if best_alpha is not None else None,
            }
            summary_json_path = ridge_dir / f"ridge_{mode_tag}_{target_safe}_{safe_name}_summary.json"
            summary_json_path.write_text(json.dumps(summary_payload, indent=2))

            fig_pred, ax_pred = plt.subplots(figsize=(12, 4.5))
            ax_pred.plot(time_trim, target_trim, label="Target", color="#1f77b4", linewidth=1.3)
            ax_pred.plot(time_trim, y_pred, label=f"Ridge ({mode_tag})", color="#ff7f0e", linewidth=1.3, alpha=0.9)
            add_split_background(ax_pred, time_trim, splits)
            ax_pred.set_xlabel("Time (s)")
            ax_pred.set_ylabel("Value")
            alpha_str = f"{best_alpha:.3g}" if best_alpha is not None else "n/a"
            ax_pred.set_title(f"{subject}/{story_for_run} – {cfg_name} (Ridge {mode_tag}, alpha={alpha_str})")
            ax_pred.legend(loc="upper right")
            ax_pred.grid(alpha=0.3)
            fig_pred.tight_layout()
            fig_pred.savefig(ridge_dir / f"ridge_{mode_tag}_{target_safe}_{safe_name}_prediction_overlay.png", dpi=200)
            plt.close(fig_pred)

            # Residual scatter
            both = np.concatenate([target_trim, y_pred])
            overall_min = float(np.nanmin(both))
            overall_max = float(np.nanmax(both))
            fig_sc, ax_sc = plt.subplots(figsize=(6, 6))
            colors = {"train": "#4CAF50", "validation": "#FFC107", "test": "#F44336"}
            for key, color in colors.items():
                span = splits.get(key)
                if span is None:
                    continue
                s, e = span
                ax_sc.scatter(
                    target_trim[s:e],
                    y_pred[s:e],
                    label=key.capitalize(),
                    alpha=0.65,
                    s=24,
                    color=color,
                )
            ax_sc.plot([overall_min, overall_max], [overall_min, overall_max], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_sc.set_xlabel("Observed target")
            ax_sc.set_ylabel(f"Predicted (Ridge {mode_tag})")
            ax_sc.set_title(f"{subject}/{story_for_run} – {cfg_name} (Ridge {mode_tag} scatter)")
            ax_sc.legend(loc="upper left")
            ax_sc.grid(alpha=0.3)
            fig_sc.tight_layout()
            fig_sc.savefig(ridge_dir / f"ridge_{mode_tag}_{target_safe}_{safe_name}_prediction_scatter.png", dpi=200)
            plt.close(fig_sc)

            # ----------------- Logging + per-mode summary -----------------
            results_by_mode[mode].append(
                {
                    "mode": mode_tag,
                    "config": cfg_name,
                    "story": story_for_run,
                    "safe_name": safe_name,
                    "method": smoothing_method,
                    "smoothing_seconds": smoothing_seconds,
                    "gaussian_sigma_seconds": gaussian_sigma,
                    "pad_mode": pad_mode,
                    "seconds_bin_width": seconds_bin_width,
                    "temporal_weighting": temporal_weighting,
                    "target_std": target_std,
                    "target_range": target_range,
                    "target_diff_abs_mean": target_diff_mean,
                    "rows_raw": raw_length,
                    "rows_clean": clean_length,
                    "rows_trimmed": rows_trimmed,
                    "alpha": float(best_alpha) if best_alpha is not None else np.nan,
                    "rmse_train": rmse_info.get("train", np.nan),
                    "rmse_validation": rmse_info.get("validation", rmse_info.get("val", np.nan)),
                    "rmse_test": rmse_info.get("test", np.nan),
                    "rho_train": rho_span.get("train", np.nan),
                    "rho_validation": rho_span.get("validation", rho_span.get("val", np.nan)),
                    "rho_test": rho_span.get("test", np.nan),
                    "cae_train": cae_span.get("train", np.nan),
                    "cae_validation": cae_span.get("validation", cae_span.get("val", np.nan)),
                    "cae_test": cae_span.get("test", np.nan),
                    "prediction_csv": str(prediction_csv_path),
                }
            )

            cfg_name_for_summary = f"{safe_name}_{mode_tag}"
            if rmse_info:
                update_rmse_summary(figs_base.parent, cfg_name_for_summary, target_col, rmse_info)
            if rho_span:
                update_span_metric_summary(
                    figs_base.parent,
                    cfg_name_for_summary,
                    target_col,
                    metric_key="rho",
                    values=rho_span,
                    ylabel="Pearson rho",
                    title_prefix="rho",
                )
            if cae_span:
                update_span_metric_summary(
                    figs_base.parent,
                    cfg_name_for_summary,
                    target_col,
                    metric_key="cae",
                    values=cae_span,
                    ylabel="CAE (Σ|residual|)",
                    title_prefix="CAE",
                )
            # Update the combined overlay for this config + mode across categories
            refresh_ridge_overlay(
                figs_base.parent,  # .../day26_ridge_cli
                safe_name,  # config directory name
                subject,
                story_label_for_outputs,
                mode_tag,
            )

    # ----------------- Global summary across configs (per mode) -----------------
    for mode, results in results_by_mode.items():
        if not results:
            continue

        mode_tag = mode  # "nolag" or "lag"
        print(f"\n=== Global summary for Ridge mode: {mode_tag} ===")
        mode_base = ensure_dir(figs_base / f"ridge_{mode_tag}")

        results_df = pd.DataFrame(results)
        summary_path = mode_base / f"day26_ridge_{mode_tag}_smoothing_summary.csv"
        if summary_path.exists():
            try:
                prev_df = pd.read_csv(summary_path)
                results_df = pd.concat([prev_df, results_df], ignore_index=True)
            except Exception:
                pass
        if not results_df.empty:
            dedup_cols = ["config", "story", "safe_name", "method", "smoothing_seconds"]
            existing_cols = [c for c in dedup_cols if c in results_df.columns]
            results_df = results_df.drop_duplicates(subset=existing_cols, keep="last")
            results_df = results_df.sort_values(by=["method", "smoothing_seconds", "story", "safe_name"]).reset_index(drop=True)
        results_df.to_csv(summary_path, index=False)
        print(f"Ridge summary ({mode_tag}) saved to {summary_path}")

        metrics = {
            "target_std": "Target std",
            "target_range": "Target range",
            "target_diff_abs_mean": "Mean |Δtarget|",
            "rmse_train": "RMSE train",
            "rmse_validation": "RMSE validation",
            "rmse_test": "RMSE test",
        }
        rho_split_cols = ["rho_train", "rho_validation", "rho_test"]
        if all(col in results_df.columns for col in rho_split_cols):
            if not results_df[rho_split_cols].isna().all().all():
                metrics.update(
                    {
                        "rho_train": "rho train",
                        "rho_validation": "rho validation",
                        "rho_test": "rho test",
                    }
                )
        cae_split_cols = ["cae_train", "cae_validation", "cae_test"]
        if all(col in results_df.columns for col in cae_split_cols):
            if not results_df[cae_split_cols].isna().all().all():
                metrics.update(
                    {
                        "cae_train": "CAE train",
                        "cae_validation": "CAE validation",
                        "cae_test": "CAE test",
                    }
                )

        pivot_dir = ensure_dir(mode_base / "matrices")
        for metric_key, metric_label in metrics.items():
            matrix = results_df.pivot(index="smoothing_seconds", columns="method", values=metric_key)
            matrix_path = pivot_dir / f"day26_ridge_{mode_tag}_{metric_key}_matrix.csv"
            matrix.to_csv(matrix_path)
            print(f"{metric_label} matrix ({mode_tag}) saved to {matrix_path}")


if __name__ == "__main__":
    main()
