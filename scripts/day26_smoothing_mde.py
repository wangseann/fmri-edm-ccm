#!/usr/bin/env python3
"""Run Day26-style smoothing + MDE sweep from the command line."""

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
    generate_category_time_series as generate_category_time_series_day19,
    build_smoothing_kernel,
    apply_smoothing_kernel,
)
from src.category_builder import (
    generate_category_time_series as generate_category_time_series_llm,
    get_embedding_backend,
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
from src.day22_category_mde import run_mde_for_pair, sanitize_name
from src.preprocessing.huth import preprocess_huth_style
from src import roi


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
                        column_names[1]: values.get("validation", np.nan),
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


def refresh_combined_overlay(parent_dir: Path, config_name: str, subject: str, story: str) -> None:
    records: List[Tuple[str, pd.DataFrame, Path, Dict[str, Any]]] = []
    for category_dir in sorted(parent_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        config_dir = category_dir / config_name
        if not config_dir.exists():
            continue
        prediction_files = list(config_dir.glob("**/mde_*_best_prediction.csv"))
        if not prediction_files:
            continue
        try:
            df = pd.read_csv(prediction_files[0])
        except Exception:
            continue
        summary_files = list(config_dir.glob("**/mde_*_summary.json"))
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
    for ax in axes[len(records) :]:
        ax.axis("off")

    global_shown: set[str] = set()

    for ax, (category_name, df, config_dir, summary_data) in zip(axes, records):
        if "time" in df.columns:
            x_vals = df["time"].to_numpy(dtype=float)
        elif "trim_index" in df.columns:
            x_vals = df["trim_index"].to_numpy(dtype=float)
        else:
            x_vals = np.arange(len(df), dtype=float)
        time_axis = x_vals
        ax.plot(time_axis, df["target"], label="Target", color="#1f77b4", linewidth=1.1)
        ax.plot(time_axis, df["prediction"], label="MDE best", color="#d62728", linewidth=1.1, alpha=0.85)
        splits = summary_data.get("splits") if summary_data else None
        if isinstance(splits, dict):
            global_shown = add_split_background(ax, time_axis, splits, shown=global_shown)
        ax.set_title(category_name)
        x_label = "Time (s)" if "time" in df.columns else "Trimmed index"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

        panel_dir = config_dir / "best_state_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        fig_single, ax_single = plt.subplots(figsize=(6, 3))
        ax_single.plot(time_axis, df["target"], label="Target", color="#1f77b4", linewidth=1.2)
        ax_single.plot(time_axis, df["prediction"], label="MDE best", color="#d62728", linewidth=1.2, alpha=0.85)
        if isinstance(splits, dict):
            add_split_background(ax_single, time_axis, splits)
        ax_single.set_title(category_name)
        ax_single.set_xlabel(x_label)
        ax_single.set_ylabel("Value")
        ax_single.grid(alpha=0.3)
        ax_single.legend(loc="upper right")
        fig_single.tight_layout()
        fig_single.savefig(panel_dir / f"{category_name}_best_state.png", dpi=180)
        plt.close(fig_single)

    axes[0].legend(loc="upper right")
    fig.suptitle(f"{subject}/{story} – {config_name} best-state overlays", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(parent_dir / f"{config_name}_best_state_overlays.png", dpi=200)
    plt.close(fig)


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
    parser = argparse.ArgumentParser(description="Sweep smoothing configs and run MDE.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="Path to YAML config.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. UTS01.")
    parser.add_argument("--story", required=True, help="Story ID, e.g. wheretheressmoke.")
    parser.add_argument(
        "--bold-run",
        type=str,
        default=None,
        help="Optional run token (e.g., run-2) for selecting a specific cached BOLD timeseries.",
    )
    parser.add_argument("--target", default="cat_abstract", help="Category column to model.")
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-stop", type=float, default=1.25)
    parser.add_argument("--window-step", type=float, default=0.25)
    parser.add_argument("--windows", type=float, nargs="*", default=None, help="Explicit smoothing windows (seconds).")
    parser.add_argument("--methods", nargs="*", default=["gaussian", "moving_average"], help="Smoothing methods to include.")
    parser.add_argument("--ccm-samples", type=int, default=10, help="Number of CCM bootstrap samples per library size (MDE sample argument).")
    parser.add_argument("--features-eval-base", type=Path, default=Path("features_day26_eval_cli"))
    parser.add_argument("--figs-base", type=Path, default=None, help="Override output figs root.")
    parser.add_argument("--top-n-plot", type=int, default=6)
    parser.add_argument("--lib-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--tau-grid", type=int, nargs="*", default=None)
    parser.add_argument("--max-predictors", type=int, default=None, help="Maximum number of predictors (embedding dimension cap).")
    parser.add_argument("--use-concat", action="store_true", help="Use Day24 concatenated subject-level series instead of a single story.")
    parser.add_argument("--concat-manifest", type=Path, default=None, help="Path to Day24 manifest JSON for the subject.")
    parser.add_argument("--concat-story-label", type=str, default="all_stories", help="Synthetic story label when using concatenated series.")
    parser.add_argument("--concat-features-root", type=Path, default=None, help="Root directory containing per-story category outputs for Day24.")
    parser.add_argument("--concat-output-subdir", type=Path, default=None, help="Relative subdirectory under features root to store Day24 outputs.")
    parser.add_argument("--concat-story-order", nargs="*", default=None, help="Optional explicit story order for concatenation.")
    parser.add_argument("--concat-force", action="store_true", help="Regenerate Day24 concatenation even if manifest exists.")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running MDE.")
    parser.add_argument(
        "--use-cae",
        action="store_true",
        help="Select the MDE step using minimum cumulative absolute error (CAE) instead of maximum rho.",
    )
    parser.add_argument(
        "--huth-preproc",
        dest="huth_preproc",
        action="store_true",
        help="Apply Huth-style drift removal / trimming prior to MDE (default).",
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
    parser.add_argument(
        "--category-embedding-backend",
        type=str,
        default="english1000",
        choices=["english1000", "english1000+word2vec", "llm"],
        help="Embedding backend for category featurization (default: english1000).",
    )
    parser.add_argument(
        "--lm-embedding-path",
        type=Path,
        default=None,
        help="Path to token→embedding file for the LLM backend (.npz/.npy/.json).",
    )
    parser.add_argument(
        "--lm-lowercase-tokens",
        dest="lm_lowercase_tokens",
        action="store_true",
        help="Lowercase tokens before lookup for the LLM backend (default).",
    )
    parser.add_argument(
        "--no-lm-lowercase-tokens",
        dest="lm_lowercase_tokens",
        action="store_false",
        help="Preserve token casing for the LLM backend.",
    )
    parser.set_defaults(lm_lowercase_tokens=True)
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

    for seconds in windows:
        # Allow explicit 0.0 to request the baseline; otherwise skip.
        if math.isclose(seconds, 0.0):
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
            try:
                generate_category_time_series_day19(
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
            except Exception as exc:
                raise RuntimeError(f"Failed to generate category series for {subject}/{story_name}: {exc}") from exc

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


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    subject = args.subject.strip()
    story = args.story.strip()
    bold_run = args.bold_run.strip() if args.bold_run else None
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
    category_backend = str(args.category_embedding_backend or "english1000").lower().strip()
    use_llm_backend = category_backend == "llm"
    if use_llm_backend and args.lm_embedding_path is None:
        raise ValueError("LLM backend selected but --lm-embedding-path was not provided.")
    lm_embedding_path = args.lm_embedding_path
    if lm_embedding_path and not lm_embedding_path.is_absolute():
        lm_embedding_path = (project_root / lm_embedding_path).resolve()
    embedding_backend_obj = None
    if use_llm_backend:
        embedding_backend_obj = get_embedding_backend(
            "llm",
            lm_embedding_path=lm_embedding_path,
            lm_lowercase_tokens=bool(args.lm_lowercase_tokens),
        )

    tau_grid = args.tau_grid if args.tau_grid else cfg.get("tau_grid") or [1, 2]
    if isinstance(tau_grid, (int, float)):
        tau_grid = [int(tau_grid)]
    tau_grid = [int(t) for t in tau_grid]

    max_predictors = int(args.max_predictors) if args.max_predictors is not None else int(cfg.get("E_mult", cfg.get("E_cap", 6)))

    lib_sizes = args.lib_sizes if args.lib_sizes else cfg.get("lib_sizes", [80, 120, 160])
    lib_sizes = [int(size) for size in lib_sizes]

    requested_figs_base = args.figs_base.resolve() if args.figs_base else None
    features_eval_base = ensure_dir(args.features_eval_base.resolve())

    use_concat = args.use_concat
    concat_category_df: Optional[pd.DataFrame] = None
    roi_matrix_concat: Optional[np.ndarray] = None
    seconds_bin_width_concat: Optional[float] = None
    concat_segment_bounds: Optional[List[Tuple[int, int]]] = None
    story_concat_label = args.concat_story_label.strip() if args.concat_story_label else "all_stories"
    if use_concat and bold_run:
        raise ValueError("--bold-run is only supported without --use-concat; concatenated inputs ignore individual runs.")
    if use_concat and use_llm_backend:
        raise ValueError("LLM embedding backend is not supported with --use-concat; run per-story generation instead.")
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
        manifest_path: Optional[Path]
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
            roi.load_schaefer_timeseries_TR(subject, story, n_parcels, paths_cfg, run=bold_run),
            dtype=float,
        )
        base_segment_bounds = [(0, base_roi_matrix.shape[0])]

    story_label_for_outputs = story_concat_label if use_concat else story
    if bold_run and not use_concat:
        story_label_for_outputs = f"{story_label_for_outputs}_{bold_run}"
    if requested_figs_base is not None:
        figs_base = ensure_dir(requested_figs_base)
    else:
        figs_root = Path(paths_cfg.get("figs", project_root / "figs"))
        figs_base = ensure_dir(figs_root / subject / story_label_for_outputs / "day26_smoothing_cli" / target_safe)

    run_suffix = f" ({bold_run})" if bold_run else ""
    configs = build_configs(
        args.window_start, args.window_stop, args.window_step, args.methods, seconds_bin_width_default, temporal_weighting_default, args.windows
    )
    print(f"Configured {len(configs)} smoothing settings for {subject}/{story}{run_suffix}.")

    if args.dry_run:
        for cfg_item in configs:
            print(f" - {cfg_item['name']}: method={cfg_item['method']} window={cfg_item['smoothing_seconds']:.2f}s")
        return

    results: List[Dict[str, Any]] = []

    for cfg_item in configs:
        cfg_name = cfg_item["name"]
        safe_name = sanitize_name(cfg_name)
        smoothing_seconds = float(cfg_item["smoothing_seconds"])
        smoothing_method = str(cfg_item["method"])
        gaussian_sigma = cfg_item.get("gaussian_sigma_seconds")
        pad_mode = cfg_item.get("pad_mode", "edge")
        seconds_bin_width = float(cfg_item.get("seconds_bin_width", seconds_bin_width_default))
        temporal_weighting = str(cfg_item.get("temporal_weighting", temporal_weighting_default))

        print(f"\n=== {cfg_name}: method={smoothing_method} window={smoothing_seconds:.2f}s ===")

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
            story_for_run = story_label_for_outputs
            cluster_csv_use = cluster_csv_path if not cluster_csv_path or Path(cluster_csv_path).exists() else ""
            if cluster_csv_path and not cluster_csv_use:
                warnings.warn(f"Cluster CSV missing at {cluster_csv_path}; proceeding without clusters.")
            if use_llm_backend:
                if embedding_backend_obj is None:
                    raise RuntimeError("LLM embedding backend was not initialized.")
                result = generate_category_time_series_llm(
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
                    embedding_backend=embedding_backend_obj,
                    save_outputs=False,
                )
            else:
                result = generate_category_time_series_day19(
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
        cat_preproc = None

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
        preproc_trim_tr = int(cat_preproc.trim_tr) if cat_preproc else 0
        preproc_win_tr = int(cat_preproc.win_len_tr) if cat_preproc else 0
        preproc_kept_ranges = [[int(s), int(e)] for s, e in cat_preproc.kept_ranges] if cat_preproc else [[int(s), int(e)] for s, e in segment_bounds]
        preproc_kept_ranges_json = json.dumps(preproc_kept_ranges)

        np.save(roi_output_path, roi_matrix_clean)
        meta_path = category_dir / "huth_preproc_meta.json"
        if args.huth_preproc and cat_preproc is not None:
            meta_payload = {
                "win_len_tr": preproc_win_tr,
                "trim_tr": preproc_trim_tr,
                "zscore": bool(args.preproc_zscore),
                "segment_bounds": [[int(s), int(e)] for s, e in segment_bounds],
                "kept_ranges": preproc_kept_ranges,
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
        time_axis = category_df["start_sec"]
        for idx, col in enumerate(top_cols):
            ax = axes[idx]
            ax.plot(time_axis, category_df[col], linewidth=1.0)
            ax.set_title(col)
            ax.grid(alpha=0.3)
        for idx in range(len(top_cols), len(axes)):
            axes[idx].axis("off")
        fig.suptitle(f"{subject} / {story_for_run} – {cfg_name} smoothing")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        plot_dir = ensure_dir(figs_base / safe_name)
        plot_path = plot_dir / "category_timeseries_overview.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved category plot to {plot_path}")

        category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
        category_df.to_csv(category_dir / "category_timeseries.csv", index=False)

        summary = run_mde_for_pair(
            subject,
            story_for_run,
            target_column=target_col,
            features_root=config_features_root,
            figs_root=plot_dir,
            paths_cfg=local_paths,
            n_parcels=int(cfg.get("n_parcels", 400)),
            tau_grid=tau_grid,
            E_cap=max_predictors,
            lib_sizes=lib_sizes,
            delta_default=int((cfg.get("delta") or [1])[0]),
            theiler_min=int(cfg.get("theiler_min", 3)),
            train_frac=0.5,
            val_frac=0.25,
            top_n_plot=args.top_n_plot,
            save_input_frame=True,
            save_scatter=True,
            sample_steps=args.ccm_samples,
            plt_module=plt,
            use_cae=args.use_cae,
        )

        selection_path = Path(summary["selection_csv"])
        mde_df = pd.read_csv(selection_path)
        rho_col = next((c for c in mde_df.columns if c.lower().startswith("rho")), None)
        best_rho = float(mde_df[rho_col].iloc[0]) if rho_col and not mde_df.empty else float("nan")
        rho_top5 = mde_df[rho_col].head(5) if rho_col else pd.Series(dtype=float)
        rho_mean_top5 = float(rho_top5.mean()) if not rho_top5.empty else float("nan")
        rho_median_top5 = float(rho_top5.median()) if not rho_top5.empty else float("nan")
        rho_std_top5 = float(rho_top5.std(ddof=1)) if len(rho_top5) >= 2 else float("nan")
        positive_rho_top5 = int((rho_top5 > 0).sum()) if not rho_top5.empty else 0

        best_var_col = "variables" if "variables" in mde_df.columns else "variable"
        best_variable = str(mde_df[best_var_col].iloc[0]) if not mde_df.empty else ""
        top5_variables = mde_df[best_var_col].head(5).astype(str) if best_var_col in mde_df.columns else pd.Series(dtype=str)
        unique_top5 = int(top5_variables.nunique()) if not top5_variables.empty else 0

        rmse_info = summary.get("rmse") or {}
        rho_span = summary.get("rho_by_span") or {}
        cae_span = summary.get("cae_by_span") or {}
        prediction_csv = summary.get("prediction_csv")
        preproc_trim_tr = int(cat_preproc.trim_tr) if args.huth_preproc and cat_preproc is not None else 0
        preproc_win_tr = int(cat_preproc.win_len_tr) if args.huth_preproc and cat_preproc is not None else 0
        kept_ranges_for_log = cat_preproc.kept_ranges if args.huth_preproc and cat_preproc is not None else segment_bounds
        preproc_kept_ranges_json = json.dumps([[int(s), int(e)] for s, e in kept_ranges_for_log])
        rows_trimmed = raw_length - clean_length

        results.append(
            {
                "config": cfg_name,
                "story": story_for_run,
                "safe_name": safe_name,
                "method": smoothing_method,
                "smoothing_seconds": smoothing_seconds,
                "gaussian_sigma_seconds": gaussian_sigma,
                "pad_mode": pad_mode,
                "seconds_bin_width": seconds_bin_width,
                "temporal_weighting": temporal_weighting,
                "top_variable": best_variable,
                "top_rho": best_rho,
                "top_cae": summary.get("best_cae", np.nan),
                "rho_mean_top5": rho_mean_top5,
                "rho_median_top5": rho_median_top5,
                "rho_std_top5": rho_std_top5,
                "positive_rho_top5": positive_rho_top5,
                "unique_top5_variables": unique_top5,
                "target_std": target_std,
                "target_range": target_range,
                "target_diff_abs_mean": target_diff_mean,
                "rows_raw": raw_length,
                "rows_clean": clean_length,
                "rows_trimmed": rows_trimmed,
                "preproc_applied": bool(args.huth_preproc),
                "preproc_trim_tr": preproc_trim_tr,
                "preproc_win_tr": preproc_win_tr,
                "preproc_kept_ranges": preproc_kept_ranges_json,
                "selection_csv": str(selection_path),
                "plot_dir": str(plot_dir),
                "mde_dir": str(plot_dir / "day22_category_mde"),
                "rmse_train": rmse_info.get("train", np.nan),
                "rmse_validation": rmse_info.get("validation", np.nan),
                "rmse_test": rmse_info.get("test", np.nan),
                "rho_train": rho_span.get("train", np.nan),
                "rho_validation": rho_span.get("validation", np.nan),
                "rho_test": rho_span.get("test", np.nan),
                "cae_train": cae_span.get("train", np.nan),
                "cae_validation": cae_span.get("validation", np.nan),
                "cae_test": cae_span.get("test", np.nan),
                "prediction_csv": prediction_csv,
                "selection_metric": summary.get("selection_metric", "rho"),
            }
        )

        if rmse_info:
            update_rmse_summary(figs_base.parent, safe_name, target_col, rmse_info)
        if rho_span:
            update_span_metric_summary(
                figs_base.parent,
                safe_name,
                target_col,
                metric_key="rho",
                values=rho_span,
                ylabel="Pearson rho",
                title_prefix="rho",
            )
        if cae_span:
            update_span_metric_summary(
                figs_base.parent,
                safe_name,
                target_col,
                metric_key="cae",
                values=cae_span,
                ylabel="CAE (Σ|residual|)",
                title_prefix="CAE",
            )
        if prediction_csv:
            refresh_combined_overlay(figs_base.parent, safe_name, subject, story_for_run)

    results_df = pd.DataFrame(results)
    # Merge with any existing summary so per-window runs accumulate.
    summary_path = figs_base / "day26_mde_smoothing_summary.csv"
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
    summary_path = figs_base / "day26_mde_smoothing_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    metrics = {
        "top_rho": "Top rho",
        "rho_mean_top5": "Mean rho (top5)",
        "rho_median_top5": "Median rho (top5)",
        "target_std": "Target std",
        "target_range": "Target range",
        "target_diff_abs_mean": "Mean |Δtarget|",
        "rmse_train": "RMSE train",
        "rmse_validation": "RMSE validation",
        "rmse_test": "RMSE test",
    }
    if "top_cae" in results_df.columns and not results_df["top_cae"].isna().all():
        metrics["top_cae"] = "Top CAE"
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
    pivot_dir = ensure_dir(figs_base / "matrices")
    for metric_key, metric_label in metrics.items():
        matrix = results_df.pivot(index="smoothing_seconds", columns="method", values=metric_key)
        matrix_path = pivot_dir / f"day26_{metric_key}_matrix.csv"
        matrix.to_csv(matrix_path)
        print(f"{metric_label} matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
