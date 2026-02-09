from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pickle

import numpy as np
from sklearn.decomposition import FastICA

from .utils import set_seed, zscore_per_column

ATLAS_TEMPLATE = "Schaefer2018_{n_parcels}Parcels_{variant}_order_FSLMNI152_{resolution}.nii.gz"


def resolve_schaefer_atlas_path(
    n_parcels: int,
    paths: dict,
    variant: str = "7Networks",
    resolution: str = "2mm",
) -> Path:
    """Return the atlas file that matches the requested Schaefer parcellation."""
    if paths is None:
        paths = {}
    if "atlas_file" in paths:
        atlas = Path(paths["atlas_file"])
        if not atlas.is_absolute():
            atlas = Path(paths.get("project_root", ".")) / atlas
        if atlas.exists():
            return atlas
    atlas_root = Path(paths.get("atlas_root", "parcellations/Parcellations/MNI"))
    atlas = atlas_root / ATLAS_TEMPLATE.format(n_parcels=n_parcels, variant=variant, resolution=resolution)
    if not atlas.exists():
        raise FileNotFoundError(f"Could not resolve Schaefer atlas at {atlas}")
    return atlas


def _story_cache(paths: dict, sub: str, story: str, run: Optional[str] = None) -> Path:
    base = Path(paths.get("cache", "data_cache")) / sub / story
    if run:
        return base / run
    return base


def _infer_parcels(paths: dict, sub: str, story: str) -> int:
    cache_dir = _story_cache(paths, sub, story)
    for file in cache_dir.glob("schaefer_*.npy"):
        suffix = file.stem.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
    raise FileNotFoundError(f"Could not infer parcel count in {cache_dir}")


def load_schaefer_timeseries_TR(sub: str, story: str, n_parcels: int, paths: dict, run: Optional[str] = None) -> np.ndarray:
    path = _story_cache(paths, sub, story, run) / f"schaefer_{n_parcels}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing Schaefer timeseries cache: {path}")
    data = np.load(path)
    if data.shape[1] != n_parcels:
        raise ValueError(f"Expected {n_parcels} parcels, found {data.shape[1]}")
    return zscore_per_column(data)


def _ica_store(paths: dict, sub: str, n_components: int) -> Path:
    cache_root = Path(paths.get("cache", "data_cache")) / sub / "ica"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"ica_{n_components}.pkl"


def _fit_ica(sub: str, stories: List[str], n_components: int, paths: dict) -> FastICA:
    parcels = _infer_parcels(paths, sub, stories[0])
    matrices = [load_schaefer_timeseries_TR(sub, story, parcels, paths) for story in stories]
    data = np.vstack(matrices)
    set_seed(0)
    ica = FastICA(n_components=n_components, random_state=0, whiten="unit-variance", max_iter=500)
    ica.fit(data)
    return ica


def load_or_fit_ica(sub: str, stories: List[str], n_components: int, paths: dict) -> np.ndarray:
    model_path = _ica_store(paths, sub, n_components)
    if model_path.exists():
        with model_path.open("rb") as fh:
            ica = pickle.load(fh)
    else:
        ica = _fit_ica(sub, stories, n_components, paths)
        with model_path.open("wb") as fh:
            pickle.dump(ica, fh)
    parcels = _infer_parcels(paths, sub, stories[0])
    data = load_schaefer_timeseries_TR(sub, stories[0], parcels, paths)
    sources = ica.transform(data)
    return zscore_per_column(sources)
