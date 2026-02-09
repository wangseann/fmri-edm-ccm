#!/usr/bin/env python3
"""Batch create or refresh Schaefer ROI caches for multiple stories."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.build_story_cache import build_story_cache
from src.utils import load_yaml


def _normalize_subject(sub: str) -> str:
    sub = sub.strip()
    return sub if sub.startswith("sub-") else f"sub-{sub}"


def list_subjects(data_root: Path, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [s.strip() for s in explicit if s.strip()]
    pattern = re.compile(r"^sub-[A-Za-z0-9]+$")
    subjects = sorted(p.name for p in data_root.iterdir() if p.is_dir() and pattern.match(p.name))
    if not subjects:
        raise ValueError(f"No subject directories found under {data_root}")
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build Schaefer ROI caches for multiple stories.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="YAML config describing data paths.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Explicit subjects to process (default: infer from cache root).")
    parser.add_argument(
        "--stories",
        nargs="*",
        default=None,
        help=(
            "Explicit stories to process (default: discover from data_root per subject). "
            "To target a specific run, use story:run-2 syntax. If run is omitted, all runs for the story are built."
        ),
    )
    parser.add_argument("--cache-root", type=Path, default=None, help="Override cache root (default: paths.cache in config).")
    parser.add_argument("--dry-run", action="store_true", help="List planned subject/stories without generating caches.")
    parser.add_argument("--semantic-components", type=int, default=None, help="Override PCA components for semantic driver.")
    return parser.parse_args()


def _parse_story_spec(story_spec: str) -> Tuple[str, Optional[str]]:
    raw = story_spec.strip()
    if not raw:
        return "", None
    if ":" in raw:
        name, run_label = raw.split(":", 1)
        return name.strip(), run_label.strip() or None
    return raw, None


def _discover_story_runs(data_root: Path, subject: str, story_filter: Optional[Sequence[str]] = None) -> List[Tuple[str, Optional[str]]]:
    """Return (story, run_label) tuples discovered under the subject's func directories."""
    subject_dir = data_root / _normalize_subject(subject)
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    story_set = {s.lower().strip() for s in story_filter} if story_filter else None
    pattern = re.compile(r"_task-([A-Za-z0-9]+).*?(?:_run-([0-9]+))?.*?_bold\.nii\.gz$", re.IGNORECASE)
    found: Dict[str, set] = defaultdict(set)
    for bold_path in subject_dir.glob("**/*bold.nii.gz"):
        match = pattern.search(bold_path.name)
        if not match:
            continue
        story = match.group(1).lower()
        run_num = match.group(2)
        if story_set and story not in story_set:
            continue
        run_label = f"run-{run_num}" if run_num else None
        found[story].add(run_label)

    if story_filter and not found:
        missing = ", ".join(sorted(story_filter))
        raise ValueError(f"No BOLD runs found for stories [{missing}] under {subject_dir}")

    schedule: List[Tuple[str, Optional[str]]] = []
    for story in sorted(found.keys()):
        runs = sorted(found[story], key=lambda r: (r is None, r))
        for run_label in runs:
            schedule.append((story, run_label))
    return schedule


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = cfg.get("paths", {}) or {}
    cache_root = args.cache_root or Path(paths_cfg.get("cache", "data_cache"))
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(paths_cfg["data_root"]).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    subjects = list_subjects(data_root, args.subjects)
    schedule: List[Tuple[str, str, Optional[str]]] = []
    story_specs = [_parse_story_spec(story) for story in args.stories] if args.stories else None

    for subject in subjects:
        if story_specs:
            story_names = [s for s, _ in story_specs if s]
            discovered = _discover_story_runs(data_root, subject, story_filter=story_names)
            by_story: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
            for story, run_label in discovered:
                by_story[story].append((story, run_label))
            for story, run_label in story_specs:
                if not story:
                    continue
                if run_label:
                    schedule.append((subject, story, run_label))
                else:
                    if story in by_story:
                        schedule.extend((subject, s, r) for s, r in by_story[story])
                    else:
                        schedule.append((subject, story, None))
        else:
            discovered = _discover_story_runs(data_root, subject)
            schedule.extend((subject, story, run_label) for story, run_label in discovered)

    print(f"[INFO] Prepared {len(schedule)} subject/story pairs.")
    if args.dry_run:
        for subject, story, run_label in schedule:
            run_suffix = f" ({run_label})" if run_label else ""
            print(f" - {subject}/{story}{run_suffix}")
        return

    for subject, story, run_label in schedule:
        run_suffix = f" ({run_label})" if run_label else ""
        print(f"\n=== Building cache for {subject} / {story}{run_suffix} ===")
        try:
            build_story_cache(
                args.config.resolve(),
                subject=subject,
                story=story,
                semantic_components_override=args.semantic_components,
                bold_run=run_label,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to build cache for {subject}/{story}{run_suffix}: {exc}")
        else:
            print(f"[INFO] Cache refreshed for {subject}/{story}{run_suffix}")


if __name__ == "__main__":
    main()
