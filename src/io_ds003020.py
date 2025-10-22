from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List

DATA_ROOT = Path(os.environ.get("DS003020_ROOT", "/bucket/PaoU/seann/openneuro/ds003020"))


def _require_subject(sub: str) -> Path:
    path = DATA_ROOT / sub
    if not path.exists():
        raise FileNotFoundError(f"Subject directory not found: {path}")
    return path


def list_stories_for_subject(sub: str) -> List[str]:
    subject_dir = _require_subject(sub)
    stories: set[str] = set()
    pattern = re.compile(r"task-([A-Za-z0-9]+)")
    for bold_path in subject_dir.glob("ses-*/func/*_bold.nii.gz"):
        match = pattern.search(bold_path.name)
        if match:
            stories.add(match.group(1))
    return sorted(stories)


def _load_tsv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="	")
        rows: List[Dict[str, float]] = []
        for row in reader:
            rows.append({k: float(v) if v not in ("", None) else float("nan") for k, v in row.items()})
        return rows


def load_story_boundaries(sub: str, story: str) -> Dict:
    candidates = [
        DATA_ROOT / "derivatives" / "boundaries" / f"{sub}_{story}.tsv",
        DATA_ROOT / "stimuli" / f"{story}_boundaries.tsv",
    ]
    for path in candidates:
        if path.exists():
            return {
                "subject": sub,
                "story": story,
                "boundaries": _load_tsv(path),
                "source": str(path),
            }
    return {"subject": sub, "story": story, "boundaries": [], "source": None}
