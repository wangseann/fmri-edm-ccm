from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import roi

DEFAULT_FEATURES_ROOT = Path("features/single_story_missing_TextGrids/features_with_fallback")
DEFAULT_OUTPUT_SUBDIR = Path("all_stories") / "day24_subject_concat"


@dataclass
class SubjectConcatResult:
    """Container for the combined category/ROI payload."""

    subject: str
    stories: List[str]
    category_frame: pd.DataFrame
    roi_matrix: np.ndarray
    boundaries: pd.DataFrame
    inventory: pd.DataFrame
    category_columns: List[str]
    roi_columns: List[str]


def _resolve_story_category_path(story_dir: Path) -> Optional[Path]:
    """Return the preferred category CSV path for a story."""
    full_path = story_dir / "category_timeseries.csv"
    trimmed_path = story_dir / "category_timeseries_trimmed.csv"
    if full_path.exists():
        return full_path
    if trimmed_path.exists():
        return trimmed_path
    return None


def _resolve_cache_dir(paths_cfg: Dict[str, str]) -> Path:
    """Resolve the cache directory that stores per-story ROI arrays."""
    cache = paths_cfg.get("cache", "data_cache")
    cache_path = Path(cache)
    if not cache_path.is_absolute():
        root = Path(paths_cfg.get("project_root", "."))
        cache_path = root / cache_path
    return cache_path


def build_story_inventory(
    subject: str,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    paths_cfg: Optional[Dict[str, str]] = None,
    n_parcels: int,
) -> pd.DataFrame:
    """Summarise available story segments for a subject."""

    features_root = features_root.resolve()
    paths_cfg = paths_cfg or {}
    cache_root = _resolve_cache_dir(paths_cfg)

    subject_dir = features_root / "subjects" / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    records: List[Dict[str, object]] = []
    for story_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        story = story_dir.name
        cat_path = _resolve_story_category_path(story_dir)
        has_category = cat_path is not None
        category_len: Optional[int] = None
        category_error: Optional[str] = None
        if has_category and cat_path is not None:
            try:
                category_len = len(pd.read_csv(cat_path))
            except Exception as exc:  # pragma: no cover
                has_category = False
                category_error = str(exc)

        roi_path = cache_root / subject / story / f"schaefer_{n_parcels}.npy"
        has_roi = roi_path.exists()
        roi_len: Optional[int] = None
        roi_error: Optional[str] = None
        if has_roi:
            try:
                arr = np.load(roi_path, mmap_mode="r")  # type: ignore[attr-defined]
                roi_len = int(arr.shape[0])
            except Exception as exc:  # pragma: no cover
                has_roi = False
                roi_error = str(exc)
            else:
                try:
                    del arr
                except Exception:
                    pass

        usable_len = min(category_len or 0, roi_len or 0) if (has_category and has_roi) else 0
        if has_category and has_roi and usable_len >= 2:
            status = "ready"
            skip_reason = ""
        elif not has_category:
            status = "missing_category"
            skip_reason = category_error or "category file not found"
        elif not has_roi:
            status = "missing_roi"
            skip_reason = roi_error or "ROI cache not found"
        elif usable_len < 2:
            status = "too_short"
            skip_reason = f"usable length {usable_len} < 2"
        else:  # pragma: no cover
            status = "unusable"
            skip_reason = "unknown"

        records.append(
            {
                "subject": subject,
                "story": story,
                "category_path": str(cat_path) if cat_path else "",
                "category_len": category_len if category_len is not None else np.nan,
                "roi_path": str(roi_path),
                "roi_len": roi_len if roi_len is not None else np.nan,
                "usable_len": usable_len,
                "status": status,
                "skip_reason": skip_reason,
            }
        )

    inventory = pd.DataFrame.from_records(records)
    inventory["category_len"] = inventory["category_len"].astype("Int64")
    inventory["roi_len"] = inventory["roi_len"].astype("Int64")
    inventory["usable_len"] = inventory["usable_len"].astype("Int64")
    return inventory


def _load_category_frame(path: Path) -> pd.DataFrame:
    """Load a category DataFrame and ensure index columns exist."""
    df = pd.read_csv(path).copy()
    if "trim_index" not in df.columns:
        df.insert(0, "trim_index", np.arange(len(df), dtype=int))
    if "tr_index" not in df.columns:
        df.insert(0, "tr_index", np.arange(len(df), dtype=int))
    return df


def _ensure_numeric(df: pd.DataFrame, prefixes: Sequence[str]) -> pd.DataFrame:
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def concatenate_subject_timeseries(
    subject: str,
    inventory: pd.DataFrame,
    *,
    story_order: Optional[Sequence[str]] = None,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    paths_cfg: Optional[Dict[str, str]] = None,
    n_parcels: int,
) -> SubjectConcatResult:
    """Combine category and ROI segments for a subject following ``story_order``."""

    features_root = features_root.resolve()
    paths_cfg = paths_cfg or {}
    cache_root = _resolve_cache_dir(paths_cfg)

    ready_df = inventory[inventory["status"] == "ready"].copy()
    if ready_df.empty:
        raise ValueError("No usable stories found for concatenation.")

    if story_order is None:
        story_sequence = ready_df["story"].tolist()
    else:
        story_set = set(ready_df["story"])
        story_sequence = [story for story in story_order if story in story_set]
        missing = [story for story in story_order if story not in story_set]
        if missing:
            raise ValueError(f"Stories not available for concatenation: {missing}")
        if not story_sequence:
            raise ValueError("Provided story_order filtered out all usable stories.")

    all_category_frames: List[pd.DataFrame] = []
    roi_chunks: List[np.ndarray] = []
    boundary_rows: List[Dict[str, int]] = []
    category_columns: List[str] = []
    global_offset = 0

    for idx, story in enumerate(story_sequence):
        story_dir = features_root / "subjects" / subject / story
        category_path = _resolve_story_category_path(story_dir)
        if category_path is None:
            raise FileNotFoundError(f"Category time series missing for {subject}/{story}")

        category_df = _load_category_frame(category_path)
        category_df = _ensure_numeric(category_df, prefixes=("cat_",))

        roi_path = cache_root / subject / story / f"schaefer_{n_parcels}.npy"
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI cache missing for {subject}/{story}: {roi_path}")
        roi_matrix = np.load(roi_path)
        if roi_matrix.ndim != 2 or roi_matrix.shape[1] != n_parcels:
            raise ValueError(f"Unexpected ROI shape {roi_matrix.shape} for {subject}/{story}")

        usable_len = min(len(category_df), roi_matrix.shape[0])
        if usable_len < 2:
            raise ValueError(f"Story {story} too short after alignment ({usable_len} rows).")

        category_df = category_df.iloc[:usable_len].reset_index(drop=True)
        roi_matrix = roi_matrix[:usable_len, :]

        story_category_cols = [c for c in category_df.columns if c.startswith("cat_")]
        if not story_category_cols:
            raise ValueError(f"No category columns detected for {subject}/{story}")
        if not category_columns:
            category_columns = story_category_cols

        story_idx = idx + 1
        story_row = np.arange(usable_len, dtype=int)
        global_index = story_row + global_offset

        enriched = category_df.copy()
        enriched.insert(0, "subject", subject)
        enriched.insert(1, "story", story)
        enriched.insert(2, "story_idx", story_idx)
        enriched.insert(3, "story_row", story_row)
        enriched.insert(4, "global_index", global_index)
        enriched["is_story_start"] = story_row == 0
        enriched["is_story_end"] = story_row == (usable_len - 1)

        all_category_frames.append(enriched)
        roi_chunks.append(roi_matrix.astype(np.float32, copy=False))

        boundary_rows.append(
            {
                "subject": subject,
                "story": story,
                "story_idx": story_idx,
                "start_index": int(global_offset),
                "end_index": int(global_offset + usable_len - 1),
                "length": int(usable_len),
            }
        )

        global_offset += usable_len

    combined_category = pd.concat(all_category_frames, ignore_index=True)
    combined_roi = np.vstack(roi_chunks)
    roi_columns = [f"roi_{i}" for i in range(combined_roi.shape[1])]
    boundaries = pd.DataFrame(boundary_rows)

    return SubjectConcatResult(
        subject=subject,
        stories=story_sequence,
        category_frame=combined_category,
        roi_matrix=combined_roi,
        boundaries=boundaries,
        inventory=inventory.copy(),
        category_columns=category_columns,
        roi_columns=roi_columns,
    )


def save_subject_concat(
    result: SubjectConcatResult,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    output_subdir: Path = DEFAULT_OUTPUT_SUBDIR,
    include_inventory: bool = True,
) -> Dict[str, Path]:
    """Persist concatenated outputs + metadata to the workspace."""

    features_root = features_root.resolve()
    output_dir = features_root / "subjects" / result.subject / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    categories_path = output_dir / "combined_category_timeseries.csv"
    roi_path = output_dir / "combined_roi_timeseries.npz"
    boundaries_path = output_dir / "story_boundaries.csv"
    manifest_path = output_dir / "manifest.json"
    inventory_path = output_dir / "story_inventory.csv"

    result.category_frame.to_csv(categories_path, index=False)
    np.savez_compressed(roi_path, roi=result.roi_matrix)
    result.boundaries.to_csv(boundaries_path, index=False)
    if include_inventory:
        result.inventory.to_csv(inventory_path, index=False)

    lag_reset_indices = (result.boundaries["end_index"].iloc[:-1] + 1).astype(int).tolist() if len(result.boundaries) > 1 else []

    manifest = {
        "subject": result.subject,
        "stories": result.stories,
        "n_stories": len(result.stories),
        "total_rows": int(result.category_frame.shape[0]),
        "n_parcels": int(result.roi_matrix.shape[1]),
        "category_columns": result.category_columns,
        "roi_columns": result.roi_columns,
        "categories_path": str(categories_path),
        "roi_path": str(roi_path),
        "boundaries_path": str(boundaries_path),
        "inventory_path": str(inventory_path) if include_inventory else None,
        "lag_reset_indices": lag_reset_indices,
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "output_dir": output_dir,
        "categories_path": categories_path,
        "roi_path": roi_path,
        "boundaries_path": boundaries_path,
        "manifest_path": manifest_path,
        "inventory_path": inventory_path if include_inventory else None,
    }


def load_subject_concat_manifest(
    subject: str,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    output_subdir: Path = DEFAULT_OUTPUT_SUBDIR,
) -> Optional[Dict[str, object]]:
    """Load manifest metadata for a previously saved concatenation."""
    features_root = features_root.resolve()
    manifest_path = features_root / "subjects" / subject / output_subdir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def load_subject_boundaries(
    subject: str,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    output_subdir: Path = DEFAULT_OUTPUT_SUBDIR,
) -> Optional[pd.DataFrame]:
    """Load precomputed story boundary metadata if available."""
    features_root = features_root.resolve()
    boundaries_path = features_root / "subjects" / subject / output_subdir / "story_boundaries.csv"
    if not boundaries_path.exists():
        return None
    return pd.read_csv(boundaries_path)


def get_story_order_from_manifest(
    manifest: Dict[str, object],
) -> List[str]:
    """Return the story order saved in a manifest, fallback to empty list."""
    stories = manifest.get("stories")
    if isinstance(stories, list):
        return [str(item) for item in stories]
    return []
