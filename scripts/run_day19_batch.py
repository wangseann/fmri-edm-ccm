#!/usr/bin/env python3
"""Batch regenerate Day19 category time series for every subject/story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.day19_category_builder import generate_category_time_series
from src.day24_subject_concat import _resolve_cache_dir as resolve_cache_dir
from src.utils import load_yaml


def resolve_paths(project_root: Path, paths_cfg: Dict[str, str]) -> Dict[str, str]:
    resolved = dict(paths_cfg)
    resolved.setdefault("project_root", str(project_root))
    for key in ("cache", "figs", "results", "data_root"):
        val = resolved.get(key)
        if val and not Path(val).is_absolute():
            resolved[key] = str((project_root / val).resolve())
    return resolved


def list_subjects(cache_root: Path, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [s.strip() for s in explicit if s.strip()]
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")
    subjects = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    if not subjects:
        raise ValueError(f"No subject directories found under {cache_root}")
    return subjects


def list_stories(cache_root: Path, subject: str, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [story.strip() for story in explicit if story.strip()]
    subject_dir = cache_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject cache directory missing: {subject_dir}")
    stories = sorted(p.name for p in subject_dir.iterdir() if p.is_dir())
    if not stories:
        raise ValueError(f"No story caches found for {subject} under {subject_dir}")
    return stories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Day19 category series for multiple stories.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="Path to YAML config.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subjects to process (defaults to all subjects with ROI caches).")
    parser.add_argument("--stories", nargs="*", default=None, help="Stories to process (defaults to all stories with ROI caches per subject).")
    parser.add_argument("--features-root", type=Path, default=Path("features_day26_concat_test"), help="Where to write Day19 features.")
    parser.add_argument("--seconds-bin-width", type=float, default=None, help="Override seconds bin width (defaults to config).")
    parser.add_argument("--temporal-weighting", type=str, default=None, help="Override temporal weighting (defaults to config).")
    parser.add_argument("--smoothing-seconds", type=float, default=0.0, help="Smoothing window (seconds).")
    parser.add_argument("--smoothing-method", type=str, default="moving_average", help="Smoothing kernel (moving_average|gaussian).")
    parser.add_argument("--gaussian-sigma", type=float, default=None, help="Sigma for Gaussian smoothing (defaults to 0.5 * window).")
    parser.add_argument("--prototype-weight-power", type=float, default=None, help="Override prototype weight power.")
    parser.add_argument("--dry-run", action="store_true", help="List planned work without executing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    categories_cfg = cfg.get("categories", {}) or {}
    paths_cfg = resolve_paths(project_root, cfg.get("paths", {}) or {})
    cache_root = Path(resolve_cache_dir(paths_cfg))
    features_root = args.features_root.resolve()
    features_root.mkdir(parents=True, exist_ok=True)

    seconds_bin_width_default = float(categories_cfg.get("seconds_bin_width", 0.05))
    temporal_weighting_default = str(categories_cfg.get("temporal_weighting", "proportional")).lower()
    prototype_power_default = float(categories_cfg.get("prototype_weight_power", 1.0))
    TR = float(cfg.get("TR", 2.0))

    cluster_csv_rel = categories_cfg.get("cluster_csv_path", "")
    cluster_csv_path = str((project_root / cluster_csv_rel).resolve()) if cluster_csv_rel else ""
    cluster_csv_use = cluster_csv_path if not cluster_csv_path or Path(cluster_csv_path).exists() else ""

    subjects = list_subjects(cache_root, args.subjects)
    todo: List[Dict[str, str]] = []
    for subject in subjects:
        stories = list_stories(cache_root, subject, args.stories)
        for story in stories:
            todo.append({"subject": subject, "story": story})

    print(f"[INFO] Located {len(todo)} subject/story pairs to process.")
    if args.dry_run:
        for item in todo:
            print(f" - {item['subject']}/{item['story']}")
        return

    for item in todo:
        subject = item["subject"]
        story = item["story"]
        print(f"\n=== Day19 batch build: {subject} / {story} ===")

        smoothing_seconds = float(args.smoothing_seconds)
        smoothing_method = args.smoothing_method
        gaussian_sigma = args.gaussian_sigma
        if gaussian_sigma is None and smoothing_method.lower() == "gaussian" and smoothing_seconds > 0:
            gaussian_sigma = 0.5 * smoothing_seconds

        temporal_weighting = (args.temporal_weighting or temporal_weighting_default).lower()
        seconds_bin_width = float(args.seconds_bin_width) if args.seconds_bin_width is not None else seconds_bin_width_default
        prototype_power = float(args.prototype_weight_power) if args.prototype_weight_power is not None else prototype_power_default

        try:
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
                smoothing_pad="edge",
                seconds_bin_width=seconds_bin_width,
                features_root=features_root,
                paths=paths_cfg,
                TR=TR,
                save_outputs=True,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process {subject}/{story}: {exc}")
            continue

        category_cols = result.get("category_columns", [])
        print(f"[INFO] Generated category series with columns: {json.dumps(category_cols)}")


if __name__ == "__main__":
    main()
