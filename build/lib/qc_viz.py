"""Helper utilities for QC visualizations on ds003020 stories."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf  # type: ignore
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional deps
    sf = None  # type: ignore
    librosa = None  # type: ignore

try:
    import textgrid  # type: ignore
except Exception:  # pragma: no cover - optional deps
    textgrid = None  # type: ignore

HAVE_AUDIO = sf is not None and librosa is not None
HAVE_TEXTGRID = textgrid is not None


@dataclass
class DriverSeries:
    envelope: np.ndarray
    word_rate: Optional[np.ndarray]
    phoneme_rate: Optional[np.ndarray]
    semantic: Optional[np.ndarray]
    semantic_labels: Optional[List[str]]
    tr: float

    @property
    def n_tr(self) -> int:
        return int(len(self.envelope))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_envelope_tr(wav_path: Path, tr: float, sr_target: int = 44100) -> np.ndarray:
    if not HAVE_AUDIO:
        raise RuntimeError("Audio dependencies (soundfile, librosa) are required for envelope computation.")

    data, sr = sf.read(str(wav_path))  # type: ignore[arg-type]
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sr_target:
        data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)  # type: ignore[operator]
        sr = sr_target

    # Frame-wise RMS then bin into TR windows
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)[0]  # type: ignore[index]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)  # type: ignore[arg-type]

    if len(times) == 0:
        return np.zeros(0, dtype=float)

    n_tr = int(np.ceil(times[-1] / tr))
    envelope = np.zeros(n_tr, dtype=float)
    for idx in range(n_tr):
        mask = (times >= idx * tr) & (times < (idx + 1) * tr)
        envelope[idx] = rms[mask].mean() if mask.any() else 0.0
    return envelope


def _find_tier(tg: "textgrid.TextGrid", keywords: Sequence[str]) -> Optional[object]:
    lower_keywords = [kw.lower() for kw in keywords]
    for tier in tg:
        name = tier.name.lower()
        if any(kw in name for kw in lower_keywords):
            return tier
    return None


def _extract_intervals(tier: object) -> List[Tuple[str, float, float]]:
    intervals: List[Tuple[str, float, float]] = []
    for item in tier:  # type: ignore[assignment]
        mark = getattr(item, "mark", "")
        if mark and mark.strip():
            intervals.append((mark.strip(), float(item.minTime), float(item.maxTime)))
    return intervals


def word_rate_from_textgrid(
    tg_path: Path,
    tr: float,
    n_tr: Optional[int] = None,
    tg_obj: Optional["textgrid.TextGrid"] = None,
) -> Tuple[np.ndarray, int, List[Tuple[str, float, float]]]:
    if not HAVE_TEXTGRID:
        raise RuntimeError("textgrid library is required for word-rate computation.")

    tg = tg_obj or textgrid.TextGrid.fromFile(str(tg_path))  # type: ignore[call-arg]
    word_tier = _find_tier(tg, ["word"])
    if word_tier is None:
        raise ValueError(f"No word tier found in TextGrid: {tg_path}")

    intervals = _extract_intervals(word_tier)

    if n_tr is None:
        max_time = max((end for _, _, end in intervals), default=0.0)
        n_tr = int(np.ceil(max_time / tr))

    counts = np.zeros(n_tr, dtype=float)
    for _, start, end in intervals:
        first = int(np.floor(start / tr))
        last = int(np.floor((end - 1e-6) / tr))
        counts[first : last + 1] += 1.0
    return counts, len(intervals), intervals


def phoneme_rate_from_textgrid(
    tg_path: Path,
    tr: float,
    n_tr: int,
    tg_obj: Optional["textgrid.TextGrid"] = None,
) -> Optional[np.ndarray]:
    if not HAVE_TEXTGRID:
        return None

    tg = tg_obj or textgrid.TextGrid.fromFile(str(tg_path))  # type: ignore[call-arg]
    phone_tier = _find_tier(tg, ["phoneme", "phone", "segment"])
    if phone_tier is None:
        return None

    intervals = _extract_intervals(phone_tier)
    if not intervals:
        return None

    counts = np.zeros(n_tr, dtype=float)
    for _, start, end in intervals:
        first = int(np.floor(start / tr))
        last = int(np.floor((end - 1e-6) / tr))
        counts[first : last + 1] += 1.0
    return counts


class SemanticLoader:
    """Callable interface added at runtime (see edm_ccm)."""

    def tr_semantic_series(
        self,
        word_intervals: List[Tuple[str, float, float]],
        n_tr: int,
        tr: float,
        components: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        raise NotImplementedError


def load_driver_series(
    wav_path: Path,
    textgrid_path: Optional[Path],
    tr: float,
    semantic_loader: Optional[SemanticLoader] = None,
    semantic_components: Optional[int] = None,
) -> DriverSeries:
    envelope = compute_envelope_tr(wav_path, tr=tr)
    word_rate = None
    phoneme_rate = None
    semantic = None
    semantic_labels: Optional[List[str]] = None

    tg_obj = None
    word_intervals: List[Tuple[str, float, float]] = []
    if textgrid_path is not None and textgrid_path.exists() and HAVE_TEXTGRID:
        tg_obj = textgrid.TextGrid.fromFile(str(textgrid_path))  # type: ignore[call-arg]
        word_rate, _, word_intervals = word_rate_from_textgrid(textgrid_path, tr=tr, n_tr=len(envelope), tg_obj=tg_obj)
        phoneme_rate = phoneme_rate_from_textgrid(textgrid_path, tr=tr, n_tr=len(envelope), tg_obj=tg_obj)
        if semantic_loader is not None:
            semantic, semantic_labels = semantic_loader.tr_semantic_series(
                word_intervals=word_intervals,
                n_tr=len(envelope),
                tr=tr,
                components=semantic_components,
            )

    return DriverSeries(
        envelope=envelope,
        word_rate=word_rate,
        phoneme_rate=phoneme_rate,
        semantic=semantic,
        semantic_labels=semantic_labels,
        tr=tr,
    )


def normalize(arr: np.ndarray) -> np.ndarray:
    max_val = np.nanmax(arr)
    if max_val <= 0:
        return arr.copy()
    return arr / max_val


def summarize_story(rec: Dict[str, str], drivers: DriverSeries) -> Dict[str, float]:
    n_tr = drivers.n_tr
    envelope = drivers.envelope
    word_rate = drivers.word_rate
    phoneme_rate = drivers.phoneme_rate
    word_count = float(word_rate.sum()) if word_rate is not None else np.nan
    words_per_tr = float(word_rate.mean()) if word_rate is not None else np.nan
    phoneme_count = float(phoneme_rate.sum()) if phoneme_rate is not None else np.nan
    phonemes_per_tr = float(phoneme_rate.mean()) if phoneme_rate is not None else np.nan

    summary = {
        "subject": rec.get("subject"),
        "story_id": rec.get("story_id"),
        "session": rec.get("session"),
        "run": rec.get("run"),
        "n_tr": n_tr,
        "env_mean": float(np.mean(envelope)) if n_tr else np.nan,
        "env_sd": float(np.std(envelope)) if n_tr else np.nan,
        "env_max": float(np.max(envelope)) if n_tr else np.nan,
        "word_count": word_count,
        "words_per_tr": words_per_tr,
        "wordrate_mean": float(word_rate.mean()) if word_rate is not None else np.nan,
        "wordrate_sd": float(word_rate.std()) if word_rate is not None else np.nan,
        "phoneme_count": phoneme_count,
        "phonemes_per_tr": phonemes_per_tr,
        "phonerate_mean": float(phoneme_rate.mean()) if phoneme_rate is not None else np.nan,
        "phonerate_sd": float(phoneme_rate.std()) if phoneme_rate is not None else np.nan,
    }
    return summary


def select_story_samples(summary_df, n: int = 3) -> List[int]:
    if len(summary_df) == 0:
        return []
    n = min(n, len(summary_df))
    df_sorted = summary_df.sort_values("n_tr").reset_index(drop=True)
    if n == 1:
        return [int(df_sorted.index[len(df_sorted) // 2])]
    indices: List[int] = []
    indices.append(int(df_sorted.index[0]))
    if n > 2:
        indices.append(int(df_sorted.index[len(df_sorted) // 2]))
    if n >= 2:
        indices.append(int(df_sorted.index[len(df_sorted) - 1]))
    return indices[:n]


def pearsonr_safe(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size or x_arr.size < 3:
        return float("nan")
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def story_cross_correlation(series_a: np.ndarray, series_b: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if series_a.size != series_b.size or series_a.size < max_lag * 2 + 3:
        raise ValueError("Series must have equal length and be longer than 2*max_lag for cross-correlation.")

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros_like(lags, dtype=float)
    for idx, lag in enumerate(lags):
        if lag < 0:
            x = series_a[:lag]
            y = series_b[-lag:]
        elif lag > 0:
            x = series_a[lag:]
            y = series_b[:-lag]
        else:
            x = series_a
            y = series_b
        if x.size < 3:
            corrs[idx] = np.nan
        else:
            corrs[idx] = pearsonr_safe(x, y)
    return lags, corrs


def make_alignment_table(story_rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for row in story_rows:
        records.append(
            {
                "subject": row.get("subject"),
                "story_id": row.get("story_id"),
                "driver_n_tr": row.get("n_tr"),
                "brain_n_tr": row.get("brain_n_tr"),
            }
        )
    return records
