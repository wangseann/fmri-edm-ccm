from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .day24_subject_concat import (
    DEFAULT_FEATURES_ROOT,
    SubjectConcatResult,
    save_subject_concat,
)

DEFAULT_DAY25_OUTPUT_SUBDIR = Path("all_stories") / "day25_bleed_correction"
_EPS = 1e-12


@dataclass
class NoiseConfig:
    mode: str = "flux"  # "flux" | "constant"
    scale: float = 1e-3
    random_seed: Optional[int] = None


def load_concat_result_from_manifest(manifest_path: Path) -> SubjectConcatResult:
    manifest = json.loads(Path(manifest_path).read_text())
    categories_path = Path(manifest["categories_path"]).resolve()
    roi_path = Path(manifest["roi_path"]).resolve()
    boundaries_path = Path(manifest["boundaries_path"]).resolve()
    inventory_path = manifest.get("inventory_path")

    category_df = pd.read_csv(categories_path)
    roi_store = np.load(roi_path)
    roi_matrix = roi_store["roi"] if "roi" in roi_store else roi_store[list(roi_store.files)[0]]
    boundaries = pd.read_csv(boundaries_path)
    if inventory_path:
        inventory = pd.read_csv(Path(inventory_path))
    else:
        inventory = pd.DataFrame()

    category_columns = manifest.get("category_columns") or [c for c in category_df.columns if c.startswith("cat_")]
    roi_columns = manifest.get("roi_columns") or [f"roi_{i}" for i in range(roi_matrix.shape[1])]
    stories = manifest.get("stories") or sorted(boundaries["story"].unique().tolist())

    return SubjectConcatResult(
        subject=str(manifest.get("subject", "")),
        stories=stories,
        category_frame=category_df,
        roi_matrix=roi_matrix,
        boundaries=boundaries,
        inventory=inventory,
        category_columns=category_columns,
        roi_columns=roi_columns,
    )


def compute_bleed_mask(boundaries: pd.DataFrame, max_lag: int, total_rows: int) -> np.ndarray:
    mask = np.zeros(total_rows, dtype=bool)
    if max_lag <= 0:
        return mask
    for _, row in boundaries.iterrows():
        start = int(row.get("end_index", -1)) + 1
        if start < 0 or start >= total_rows:
            continue
        stop = min(total_rows, start + max_lag)
        mask[start:stop] = True
    return mask


def _ensure_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _column_scale(values: np.ndarray, scale: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return max(scale, _EPS)
    std = np.std(finite)
    if not np.isfinite(std) or std <= _EPS:
        std = max(np.abs(np.mean(finite)), 1.0)
    return max(std * scale, scale)


def apply_bleed_noise(
    result: SubjectConcatResult,
    bleed_mask: np.ndarray,
    *,
    category_config: NoiseConfig,
    roi_config: NoiseConfig,
) -> SubjectConcatResult:
    if bleed_mask.size != len(result.category_frame):
        raise ValueError("Bleed mask length must match category frame")
    if not bleed_mask.any():
        return result

    rng_cat = _ensure_rng(category_config.random_seed)
    rng_roi = _ensure_rng(roi_config.random_seed)

    cat_df = result.category_frame.copy()
    roi_matrix = result.roi_matrix.copy()

    cat_cols = result.category_columns
    for col in cat_cols:
        values = cat_df[col].to_numpy(dtype=float)
        scale = _column_scale(values, category_config.scale)
        idx = np.where(bleed_mask)[0]
        if category_config.mode == "constant":
            fill_value = scale
            values[idx] = fill_value
        else:
            values[idx] = rng_cat.normal(loc=0.0, scale=scale, size=len(idx))
        cat_df[col] = values

    idx = np.where(bleed_mask)[0]
    if idx.size > 0:
        scale_template = np.std(roi_matrix[~bleed_mask, :], axis=0)
        scale_template[~np.isfinite(scale_template)] = 1.0
        scale_template = np.maximum(scale_template * roi_config.scale, roi_config.scale)
        if roi_config.mode == "constant":
            roi_matrix[idx, :] = scale_template
        else:
            noise = rng_roi.normal(loc=0.0, scale=scale_template, size=(len(idx), roi_matrix.shape[1]))
            roi_matrix[idx, :] = noise

    return SubjectConcatResult(
        subject=result.subject,
        stories=result.stories,
        category_frame=cat_df,
        roi_matrix=roi_matrix,
        boundaries=result.boundaries.copy(),
        inventory=result.inventory.copy(),
        category_columns=list(result.category_columns),
        roi_columns=list(result.roi_columns),
    )


def save_corrected_result(
    result: SubjectConcatResult,
    *,
    features_root: Path = DEFAULT_FEATURES_ROOT,
    output_suffix: str,
    include_inventory: bool = True,
) -> Dict[str, Path]:
    subdir = DEFAULT_DAY25_OUTPUT_SUBDIR / output_suffix
    return save_subject_concat(
        result,
        features_root=features_root,
        output_subdir=subdir,
        include_inventory=include_inventory,
    )
