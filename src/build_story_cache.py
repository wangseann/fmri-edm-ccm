from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import nibabel as nib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("nibabel is required to build story caches. Install with `pip install nibabel`.") from exc

try:
    from nilearn.maskers import NiftiLabelsMasker  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("nilearn is required for parcel averaging. Install with `pip install nilearn`.") from exc

from .edm_ccm import English1000Loader
from .qc_viz import DriverSeries, load_driver_series
from .utils import ensure_same_length, set_seed

DEFAULT_ENGLISH1000 = "derivative/english1000sm.hf5"


@dataclass
class CacheArtifacts:
    subject: str
    story: str
    tr: float
    n_tr_audio: int
    n_tr_bold: int
    bold_path: str
    atlas_path: str
    wav_path: str
    textgrid_path: Optional[str]
    semantic_source: Optional[str]
    semantic_components: Optional[int]
    run_label: Optional[str] = None


def _normalize_subject(sub: str) -> str:
    return sub if sub.startswith("sub-") else f"sub-{sub}"


def _resolve_story_key(story: str) -> str:
    return story.lower().replace("-", "").replace("_", "")


def _find_bold_path(data_root: Path, sub: str, story: str, run_token: Optional[str] = None) -> Path:
    sub_dir = data_root / _normalize_subject(sub)
    if not sub_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
    slug = _resolve_story_key(story)
    slug_wo_digits = "".join(ch for ch in slug if not ch.isdigit())
    alt_tokens = {slug, slug_wo_digits, story.lower(), story.lower().replace("_", ""), story.lower().replace("-", "")}
    alt_tokens = {tok for tok in alt_tokens if tok}

    def _match(paths):
        for token in alt_tokens:
            filtered = [p for p in paths if token in p.name.lower()]
            if filtered:
                return filtered
        return []

    candidates = sorted(sub_dir.glob("**/*desc-preproc_bold.nii.gz"))
    matches = _match(candidates)
    if not matches:
        candidates = sorted(sub_dir.glob("**/*bold.nii.gz"))
        matches = _match(candidates)
    if run_token:
        run_token = run_token.lower()
        matches = [p for p in matches if run_token in p.name.lower()]
        if not matches:
            raise FileNotFoundError(f'Could not locate BOLD run "{run_token}" for story "{story}" under {sub_dir}')
    if not matches:
        raise FileNotFoundError(f'Could not locate BOLD run for story "{story}" under {sub_dir}')
    return matches[0]


def _resolve_audio_path(data_root: Path, story: str) -> Path:
    candidates = [
        data_root / "stimuli" / f"{story}.wav",
        data_root / "audio" / f"{story}.wav",
        data_root / "stimuli" / f"{story.lower()}.wav",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Audio WAV not found for story {story}")


def _resolve_textgrid_path(data_root: Path, story: str) -> Optional[Path]:
    candidates = [
        data_root / "stimuli" / f"{story}.TextGrid",
        data_root / "annotations" / f"{story}.TextGrid",
        data_root / "stimuli" / f"{story.lower()}.TextGrid",
        data_root / "derivative" / "TextGrids" / f"{story}.TextGrid",
        data_root / "derivative" / "TextGrids" / f"{story.lower()}.TextGrid",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_english_loader(data_root: Path, paths_cfg: dict) -> Optional[English1000Loader]:
    english_path = paths_cfg.get("english1000_h5")
    if english_path is None:
        english_path = data_root / DEFAULT_ENGLISH1000
    else:
        english_path = Path(english_path)
        if not english_path.is_absolute():
            english_path = data_root / english_path
    if not english_path.exists():
        return None
    return English1000Loader(english_path)


def _resolve_atlas_path(paths_cfg: dict, n_parcels: int) -> Path:
    atlas_file = paths_cfg.get("atlas_file")
    atlas_root = Path(paths_cfg.get("atlas_root", "parcellations/Parcellations/MNI"))
    if atlas_file:
        atlas = Path(atlas_file)
        if not atlas.is_absolute():
            atlas = Path(paths_cfg.get("project_root", ".")) / atlas
        if atlas.exists():
            return atlas
    template = f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    atlas = atlas_root / template
    if not atlas.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas}")
    return atlas


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def _save_meta(path: Path, meta: CacheArtifacts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def _extract_roi_timeseries(bold_path: Path, atlas_path: Path, tr: float, n_parcels: int) -> np.ndarray:
    masker = NiftiLabelsMasker(labels_img=str(atlas_path), standardize=True, t_r=tr)
    series = masker.fit_transform(str(bold_path))
    if series.shape[1] != n_parcels:
        raise ValueError(f"Atlas produced {series.shape[1]} parcels but expected {n_parcels}.")
    return series.astype(np.float32)


def build_story_cache(
    config_path: Path,
    subject: Optional[str],
    story: Optional[str],
    semantic_components_override: Optional[int] = None,
    *,
    story_cache_label: Optional[str] = None,
    bold_run: Optional[str] = None,
) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    subject = subject or cfg["subject"]
    story = story or cfg["story"]
    story_cache_label = (story_cache_label or story).strip()
    bold_run = bold_run.strip() if bold_run else None
    tr = float(cfg["TR"])
    n_parcels = int(cfg["n_parcels"])
    semantic_components = semantic_components_override if semantic_components_override is not None else cfg.get("pca_components")
    semantic_components_arg = None if not semantic_components else int(semantic_components)

    paths_cfg = dict(cfg.get("paths", {}))
    paths_cfg.setdefault("project_root", str(config_path.parents[1]))

    data_root = Path(paths_cfg["data_root"])
    cache_root = Path(paths_cfg.get("cache", "data_cache"))

    bold_path = _find_bold_path(data_root, subject, story, run_token=bold_run)
    wav_path = _resolve_audio_path(data_root, story)
    textgrid_path = _resolve_textgrid_path(data_root, story)
    english_loader = _resolve_english_loader(data_root, paths_cfg)

    atlas_path = _resolve_atlas_path(paths_cfg, n_parcels)

    driver_series: DriverSeries = load_driver_series(
        wav_path=wav_path,
        textgrid_path=textgrid_path,
        tr=tr,
        semantic_loader=english_loader,
        semantic_components=semantic_components_arg,
    )

    if driver_series.word_rate is None:
        raise RuntimeError("Word-rate could not be computed; check TextGrid path/dependencies.")
    if driver_series.semantic is None:
        raise RuntimeError("Semantic features unavailable; ensure English1000 HDF5 is accessible.")

    bold_img = nib.load(str(bold_path))
    n_tr_bold = bold_img.shape[-1]
    driver_len = driver_series.n_tr

    cache_dir_base = cache_root / subject / story_cache_label
    if bold_run:
        cache_dir = cache_dir_base / bold_run
    else:
        cache_dir = cache_dir_base

    if n_tr_bold != driver_len:
        min_len = min(n_tr_bold, driver_len)
        driver_series.envelope = driver_series.envelope[:min_len]
        driver_series.word_rate = driver_series.word_rate[:min_len]
        if driver_series.semantic is not None:
            driver_series.semantic = driver_series.semantic[:min_len]
        if driver_series.phoneme_rate is not None:
            driver_series.phoneme_rate = driver_series.phoneme_rate[:min_len]
        data = bold_img.get_fdata()[..., :min_len]
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_dir / "bold_trimmed.nii.gz"
        nib.Nifti1Image(data, bold_img.affine, bold_img.header).to_filename(str(temp_path))
        bold_path = temp_path
        n_tr_bold = min_len

    roi_series = _extract_roi_timeseries(bold_path, atlas_path, tr, n_parcels)
    env, word_rate, semantic = ensure_same_length(driver_series.envelope, driver_series.word_rate, driver_series.semantic)

    _save_npy(cache_dir / "envelope_TR.npy", env)
    _save_npy(cache_dir / "wordrate_TR.npy", word_rate)
    _save_npy(cache_dir / "english1000_TR.npy", semantic)

    pca_components_cfg = cfg.get("pca_components")
    if semantic_components_arg is None and pca_components_cfg:
        try:
            from sklearn.decomposition import PCA

            components = int(pca_components_cfg)
            if 0 < components < semantic.shape[1]:
                semantic_pca = PCA(n_components=components, random_state=0).fit_transform(np.nan_to_num(semantic))
                _save_npy(cache_dir / f"semantic_pca{components}.npy", semantic_pca)
        except Exception:
            pass

    if driver_series.semantic_labels and cfg.get("pca_components"):
        labels_path = cache_dir / "semantic_labels.json"
        labels_path.write_text(json.dumps(driver_series.semantic_labels[: cfg["pca_components"]]), encoding="utf-8")

    _save_npy(cache_dir / f"schaefer_{n_parcels}.npy", roi_series)

    meta = CacheArtifacts(
        subject=subject,
        story=story_cache_label,
        tr=tr,
        n_tr_audio=driver_series.n_tr,
        n_tr_bold=n_tr_bold,
        bold_path=str(bold_path),
        atlas_path=str(atlas_path),
        wav_path=str(wav_path),
        textgrid_path=str(textgrid_path) if textgrid_path else None,
        semantic_source=str(english_loader.h5_path) if english_loader else None,
        semantic_components=int(semantic.shape[1]),
        run_label=bold_run,
    )
    _save_meta(cache_dir / "cache_meta.json", meta)
    return cache_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TR-aligned caches for one story.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config.")
    parser.add_argument("--subject", help="Override subject in config (optional).")
    parser.add_argument("--story", help="Override story in config (optional).")
    parser.add_argument("--run-label", help="Optional run token (e.g., run-2) to select a specific BOLD file.")
    parser.add_argument(
        "--story-cache-label",
        help="Optional directory label for the cache (defaults to story; run label, if provided, is appended as a subdirectory).",
    )
    parser.add_argument("--semantic-components", type=int, help="Override semantic components (0 for full embedding).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    cache_dir = build_story_cache(
        Path(args.config),
        args.subject,
        args.story,
        args.semantic_components,
        story_cache_label=args.story_cache_label,
        bold_run=args.run_label,
    )
    print(f"Cache written to {cache_dir}")


if __name__ == "__main__":
    main()
