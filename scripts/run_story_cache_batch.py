#!/usr/bin/env python3
"""Batch create or refresh Schaefer ROI caches for multiple stories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from src.build_story_cache import build_story_cache
from src.utils import load_yaml


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
        raise ValueError(f"No story directories found for {subject} under {subject_dir}")
    return stories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build Schaefer ROI caches for multiple stories.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="YAML config describing data paths.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Explicit subjects to process (default: infer from cache root).")
    parser.add_argument("--stories", nargs="*", default=None, help="Explicit stories to process (default: infer per subject).")
    parser.add_argument("--cache-root", type=Path, default=None, help="Override cache root (default: paths.cache in config).")
    parser.add_argument("--dry-run", action="store_true", help="List planned subject/stories without generating caches.")
    parser.add_argument("--semantic-components", type=int, default=None, help="Override PCA components for semantic driver.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = cfg.get("paths", {}) or {}
    cache_root = args.cache_root or Path(paths_cfg.get("cache", "data_cache"))
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects(cache_root, args.subjects)
    schedule = []
    for subject in subjects:
        stories = list_stories(cache_root, subject, args.stories)
        for story in stories:
            schedule.append((subject, story))

    print(f"[INFO] Prepared {len(schedule)} subject/story pairs.")
    if args.dry_run:
        for subject, story in schedule:
            print(f" - {subject}/{story}")
        return

    for subject, story in schedule:
        print(f"\n=== Building cache for {subject} / {story} ===")
        try:
            build_story_cache(
                args.config.resolve(),
                subject=subject,
                story=story,
                semantic_components_override=args.semantic_components,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to build cache for {subject}/{story}: {exc}")
        else:
            print(f"[INFO] Cache refreshed for {subject}/{story}")


if __name__ == "__main__":
    main()
