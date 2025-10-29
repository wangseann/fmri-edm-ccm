"""Utility functions supporting EDM/CCM diagnostics for ds003020 drivers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - optional dep
    h5py = None  # type: ignore

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - optional dep
    PCA = None  # type: ignore

from .qc_viz import DriverSeries, SemanticLoader, load_driver_series


@dataclass
class StoryDriver:
    subject: str
    story_id: str
    session: Optional[str]
    run: Optional[str]
    drivers: DriverSeries
    wav_path: Path
    textgrid_path: Optional[Path]

    @property
    def n_tr(self) -> int:
        return self.drivers.n_tr


class English1000Loader(SemanticLoader):
    """Loads English1000 embeddings and projects them to TR-level series."""

    def __init__(self, h5_path: Path):
        """Load English1000 embeddings and prepare lookup tables."""
        if h5py is None:
            raise RuntimeError("h5py is required to load English1000 embeddings.")
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as hf:  # type: ignore[call-arg]
            data = hf["data"][:].astype(np.float32)
            vocab_raw = hf["vocab"][:]
        self.embeddings = data.T  # shape (vocab, dim)
        self.embed_dim = self.embeddings.shape[1]
        self.vocab = [token.decode("utf-8").strip().lower() for token in vocab_raw]
        self.lookup: Dict[str, np.ndarray] = {word: self.embeddings[idx] for idx, word in enumerate(self.vocab) if word}

    def tr_semantic_series(
        self,
        word_intervals: List[Tuple[str, float, float]],
        n_tr: int,
        tr: float,
        components: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        if not word_intervals:
            return None, None

        accum = np.zeros((n_tr, self.embed_dim), dtype=np.float32)
        counts = np.zeros(n_tr, dtype=np.float32)

        for word, start, _ in word_intervals:
            idx = int(np.floor(start / tr))
            if idx < 0 or idx >= n_tr:
                continue
            vec = self.lookup.get(word.lower())
            if vec is None:
                continue
            accum[idx] += vec
            counts[idx] += 1.0

        valid_mask = counts > 0
        if not np.any(valid_mask):
            return None, None

        accum[valid_mask] /= counts[valid_mask, None]

        series = accum
        labels = [f"sem_dim{i+1}" for i in range(self.embed_dim)]

        if components is not None and 0 < components < self.embed_dim:
            if PCA is None:
                raise RuntimeError("scikit-learn is required for PCA on semantic features.")
            pca = PCA(n_components=components)
            reduced = pca.fit_transform(series[valid_mask])
            series_reduced = np.zeros((n_tr, components), dtype=np.float32)
            series_reduced[valid_mask] = reduced
            series = series_reduced
            labels = [f"sem_pc{i+1}" for i in range(components)]

        return series, labels


def autocorrelation(series: Sequence[float], max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(series, dtype=float)
    arr = arr - np.nanmean(arr)
    if arr.size == 0:
        return np.arange(0), np.zeros(0)
    denom = np.dot(arr, arr)
    lags = np.arange(0, max_lag + 1)
    acf = np.zeros_like(lags, dtype=float)
    for idx, lag in enumerate(lags):
        if lag == 0:
            acf[idx] = 1.0
        else:
            acf[idx] = np.dot(arr[:-lag], arr[lag:]) / denom if denom != 0 else np.nan
    return lags, acf


def first_below_threshold(ac_values: Sequence[float], threshold: float) -> Optional[int]:
    for lag, value in enumerate(ac_values):
        if lag == 0:
            continue
        if value <= threshold:
            return lag
    return None


def multi_segment_library_lengths(lengths: Sequence[int], exclusion: int = 0) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=int)
    usable = np.maximum(lengths - exclusion, 0)
    return usable


def aggregate_driver_summary(stories: Iterable[StoryDriver]) -> List[dict]:
    rows: List[dict] = []
    for story in stories:
        env = story.drivers.envelope
        rate = story.drivers.word_rate
        phone = story.drivers.phoneme_rate
        row = {
            "subject": story.subject,
            "story_id": story.story_id,
            "session": story.session,
            "run": story.run,
            "n_tr": story.n_tr,
            "env_mean": float(env.mean()) if story.n_tr else np.nan,
            "env_sd": float(env.std()) if story.n_tr else np.nan,
            "env_ac1": np.nan,
            "env_e_fold": np.nan,
            "wordrate_mean": float(rate.mean()) if rate is not None else np.nan,
            "wordrate_sd": float(rate.std()) if rate is not None else np.nan,
            "word_count": float(rate.sum()) if rate is not None else np.nan,
            "phonerate_mean": float(phone.mean()) if phone is not None else np.nan,
            "phonerate_sd": float(phone.std()) if phone is not None else np.nan,
            "phoneme_count": float(phone.sum()) if phone is not None else np.nan,
            "semantic_dims": story.drivers.semantic.shape[1] if story.drivers.semantic is not None else 0,
            "semantic_active_tr": int(np.count_nonzero(np.linalg.norm(story.drivers.semantic, axis=1))) if story.drivers.semantic is not None else 0,
        }

        if story.n_tr:
            lags, acf_env = autocorrelation(env, min(60, story.n_tr - 1))
            row["env_ac1"] = float(acf_env[1]) if acf_env.size > 1 else np.nan
            e_fold = first_below_threshold(acf_env, np.exp(-1)) if acf_env.size else None
            row["env_e_fold"] = float(e_fold) if e_fold is not None else np.nan
            if rate is not None:
                _, acf_rate = autocorrelation(rate, min(60, story.n_tr - 1))
                row["wordrate_ac1"] = float(acf_rate[1]) if acf_rate.size > 1 else np.nan
                e_fold_rate = first_below_threshold(acf_rate, np.exp(-1)) if acf_rate.size else None
                row["wordrate_e_fold"] = float(e_fold_rate) if e_fold_rate is not None else np.nan
            if phone is not None:
                _, acf_phone = autocorrelation(phone, min(60, story.n_tr - 1))
                row["phonerate_ac1"] = float(acf_phone[1]) if acf_phone.size > 1 else np.nan
                e_fold_phone = first_below_threshold(acf_phone, np.exp(-1)) if acf_phone.size else None
                row["phonerate_e_fold"] = float(e_fold_phone) if e_fold_phone is not None else np.nan
        rows.append(row)
    return rows


def load_subject_stories(
    records: Sequence[dict],
    tr: float,
    semantic_loader: Optional[SemanticLoader] = None,
    semantic_components: Optional[int] = None,
) -> List[StoryDriver]:
    drivers: List[StoryDriver] = []
    for rec in records:
        wav = Path(rec["wav"]) if rec.get("wav") else None
        if wav is None or not wav.exists():
            continue
        tg_path = Path(rec["textgrid"]) if rec.get("textgrid") else None
        try:
            series = load_driver_series(
                wav,
                tg_path,
                tr=tr,
                semantic_loader=semantic_loader,
                semantic_components=semantic_components,
            )
        except Exception:
            continue
        drivers.append(
            StoryDriver(
                subject=rec.get("subject", ""),
                story_id=rec.get("story_id", ""),
                session=rec.get("session"),
                run=rec.get("run"),
                drivers=series,
                wav_path=wav,
                textgrid_path=tg_path,
            )
        )
    return drivers
