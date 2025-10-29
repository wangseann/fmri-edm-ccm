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
from src.day22_category_mde import run_mde_for_pair, sanitize_name


def update_rmse_summary(base_dir: Path, config_name: str, category: str, rmse: Dict[str, float]) -> None:
    if not rmse:
        return
    summary_csv = base_dir / f"{config_name}_rmse_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = df[df["category"] != category]
    else:
        df = pd.DataFrame(columns=["category", "rmse_train", "rmse_validation", "rmse_test"])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "category": category,
                        "rmse_train": rmse.get("train", np.nan),
                        "rmse_validation": rmse.get("validation", np.nan),
                        "rmse_test": rmse.get("test", np.nan),
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
        ax.bar(x - width, df["rmse_train"], width, label="Train")
        ax.bar(x, df["rmse_validation"], width, label="Validation")
        ax.bar(x + width, df["rmse_test"], width, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE by span – {config_name}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(base_dir / f"{config_name}_rmse_summary.png", dpi=180)
        plt.close(fig)


def refresh_combined_overlay(parent_dir: Path, config_name: str, subject: str, story: str) -> None:
    records: List[Tuple[str, pd.DataFrame, Path]] = []
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
        records.append((category_dir.name, df, config_dir))

    if not records:
        return

    cols = 3
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharex=False)
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(records) :]:
        ax.axis("off")

    for ax, (category_name, df, config_dir) in zip(axes, records):
        ax.plot(df["time"], df["target"], label="Target", color="#1f77b4", linewidth=1.1)
        ax.plot(df["time"], df["prediction"], label="MDE best", color="#d62728", linewidth=1.1, alpha=0.85)
        ax.set_title(category_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

        panel_dir = config_dir / "best_state_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        fig_single, ax_single = plt.subplots(figsize=(6, 3))
        ax_single.plot(df["time"], df["target"], label="Target", color="#1f77b4", linewidth=1.2)
        ax_single.plot(df["time"], df["prediction"], label="MDE best", color="#d62728", linewidth=1.2, alpha=0.85)
        ax_single.set_title(category_name)
        ax_single.set_xlabel("Time (s)")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep smoothing configs and run MDE.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="Path to YAML config.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. UTS01.")
    parser.add_argument("--story", required=True, help="Story ID, e.g. wheretheressmoke.")
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
            try:
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
    target_col = args.target.strip()
    target_safe = sanitize_name(target_col)

    paths_cfg = resolve_paths(project_root, cfg.get("paths", {}) or {})

    categories_cfg = cfg.get("categories", {}) or {}
    cluster_csv_rel = categories_cfg.get("cluster_csv_path", "")
    cluster_csv_path = str((project_root / cluster_csv_rel).resolve()) if cluster_csv_rel else ""

    TR = float(cfg.get("TR", 2.0))
    prototype_power = float(categories_cfg.get("prototype_weight_power", 1.0))
    seconds_bin_width_default = float(categories_cfg.get("seconds_bin_width", 0.05))
    temporal_weighting_default = str(categories_cfg.get("temporal_weighting", "proportional")).lower()

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
        if "start_sec" in concat_category_df.columns and "end_sec" in concat_category_df.columns and len(concat_category_df) > 1:
            seconds_bin_width_concat = float(concat_category_df["end_sec"].iloc[0] - concat_category_df["start_sec"].iloc[0])
            if not np.isfinite(seconds_bin_width_concat) or seconds_bin_width_concat <= 0:
                seconds_bin_width_concat = float(TR)
        else:
            seconds_bin_width_concat = float(TR)
    else:
        story_concat_label = story

    story_label_for_outputs = story_concat_label if use_concat else story
    if requested_figs_base is not None:
        figs_base = ensure_dir(requested_figs_base)
    else:
        figs_root = Path(paths_cfg.get("figs", project_root / "figs"))
        figs_base = ensure_dir(figs_root / subject / story_label_for_outputs / "day26_smoothing_cli" / target_safe)

    configs = build_configs(
        args.window_start, args.window_stop, args.window_step, args.methods, seconds_bin_width_default, temporal_weighting_default, args.windows
    )
    print(f"Configured {len(configs)} smoothing settings for {subject}/{story}.")

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
                kernel = build_smoothing_kernel(
                    seconds_bin_width_concat or seconds_bin_width, smoothing_seconds, method=smoothing_method, gaussian_sigma_seconds=gaussian_sigma
                )
                smoothed_vals = apply_smoothing_kernel(category_df[category_cols].to_numpy(dtype=float), kernel, pad_mode=pad_mode)
                category_df.loc[:, category_cols] = smoothed_vals
            category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
            category_df.to_csv(category_dir / "category_timeseries.csv", index=False)
            cache_root = ensure_dir(config_features_root / "cache")
            cache_subject_dir = ensure_dir(cache_root / subject / story_for_run)
            roi_output_path = cache_subject_dir / f"schaefer_{int(cfg.get('n_parcels', 400))}.npy"
            if not roi_output_path.exists():
                np.save(roi_output_path, roi_matrix_concat.astype(np.float32, copy=False))
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
            category_dir = ensure_dir(config_features_root / "subjects" / subject / story_for_run)
            category_df.to_csv(category_dir / "category_timeseries.csv", index=False)
            local_paths = paths_cfg

        target_series = category_df[target_col].astype(float)
        target_std = float(target_series.std(ddof=1))
        target_range = float(target_series.max() - target_series.min())
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
        prediction_csv = summary.get("prediction_csv")

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
                "rho_mean_top5": rho_mean_top5,
                "rho_median_top5": rho_median_top5,
                "rho_std_top5": rho_std_top5,
                "positive_rho_top5": positive_rho_top5,
                "unique_top5_variables": unique_top5,
                "target_std": target_std,
                "target_range": target_range,
                "target_diff_abs_mean": target_diff_mean,
                "selection_csv": str(selection_path),
                "plot_dir": str(plot_dir),
                "mde_dir": str(plot_dir / "day22_category_mde"),
                "rmse_train": rmse_info.get("train", np.nan),
                "rmse_validation": rmse_info.get("validation", np.nan),
                "rmse_test": rmse_info.get("test", np.nan),
                "prediction_csv": prediction_csv,
            }
        )

        if rmse_info:
            update_rmse_summary(figs_base.parent, safe_name, target_col, rmse_info)
        if prediction_csv:
            refresh_combined_overlay(figs_base.parent, safe_name, subject, story_for_run)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["method", "smoothing_seconds"]).reset_index(drop=True)
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
    pivot_dir = ensure_dir(figs_base / "matrices")
    for metric_key, metric_label in metrics.items():
        matrix = results_df.pivot(index="smoothing_seconds", columns="method", values=metric_key)
        matrix_path = pivot_dir / f"day26_{metric_key}_matrix.csv"
        matrix.to_csv(matrix_path)
        print(f"{metric_label} matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
