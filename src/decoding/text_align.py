"""Utilities for aligning transcripts with TR windows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import csv
import json

try:
    import textgrid  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    textgrid = None  # type: ignore

WordEvent = Tuple[str, float, float]


def _candidate_paths(paths: dict, sub: str, story: str) -> List[Path]:
    """Return plausible transcript file locations."""
    candidates: List[Path] = []
    variants = {
        story,
        story.replace(" ", ""),
        story.replace("-", ""),
        story.replace("_", ""),
        story.lower(),
        story.lower().replace(" ", ""),
        story.lower().replace("-", ""),
        story.lower().replace("_", ""),
    }
    story_variants = sorted({variant for variant in variants if variant})
    base_names = []
    for variant in story_variants:
        base_names.extend(
            [
                variant,
                f"{variant}_words",
                f"{variant}_transcript",
                f"{sub}_{variant}_words",
            ]
        )
    base_names = list(dict.fromkeys(base_names))

    roots: List[Path] = []
    derivative_suffixes = [
        Path("derivative") / "TextGrids",
        Path("derivatives") / "TextGrids",
    ]
    for key in ("transcripts", "transcript_root", "cache", "data_root"):
        val = paths.get(key)
        if not val:
            continue
        base = Path(val)
        roots.extend(
            [
                base,
                base / sub,
            ]
        )
        for variant in story_variants:
            roots.append(base / sub / variant)
        for suffix in derivative_suffixes:
            roots.append(base / suffix)
    roots.extend(
        [
            Path("transcripts") / sub,
            Path("transcripts"),
        ]
    )
    for variant in story_variants:
        roots.append(Path("transcripts") / variant)

    for root in _dedupe(roots):
        for base in base_names:
            for ext in (".csv", ".tsv", ".json", ".TextGrid"):
                candidates.append(root / f"{base}{ext}")
    return candidates


def _dedupe(seq: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _events_from_textgrid(path: Path) -> Optional[List[WordEvent]]:
    if textgrid is None:
        return None
    tg = textgrid.TextGrid.fromFile(str(path))  # type: ignore[attr-defined]
    events: List[WordEvent] = []
    for tier in tg:  # type: ignore[assignment]
        name = getattr(tier, "name", "").lower()
        if "word" not in name:
            continue
        for item in tier:  # type: ignore[assignment]
            mark = getattr(item, "mark", "").strip()
            if not mark:
                continue
            start = float(getattr(item, "minTime", 0.0))
            end = float(getattr(item, "maxTime", start))
            events.append((mark, start, end))
    return events if events else None


def load_transcript_words(paths: dict, sub: str, story: str) -> List[WordEvent]:
    """Load word-level transcript with onset/offset timestamps."""
    candidates = _dedupe(_candidate_paths(paths, sub, story))
    for cand in candidates:
        if not cand.exists():
            continue
        suffix = cand.suffix.lower()
        if suffix == ".json":
            data = json.loads(cand.read_text())
            events = [
                (
                    str(entry["word"]),
                    float(entry["onset"]),
                    float(entry.get("offset", entry["onset"])),
                )
                for entry in data
            ]
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with cand.open("r", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                events = [
                    (
                        str(row["word"]),
                        float(row["onset"]),
                        float(row.get("offset", row["onset"])),
                    )
                    for row in reader
                ]
        elif suffix == ".textgrid":
            events = _events_from_textgrid(cand)
            if events is None:
                continue
        else:
            continue
        events.sort(key=lambda item: item[1])
        return events
    raise FileNotFoundError(
        "No transcript file found. Provide `paths['transcripts']` or place a " "transcript CSV/TSV/JSON/TextGrid with word/onset/offset information."
    )


def build_ngram_contexts(word_events: Sequence[WordEvent], ngram: int, use: str) -> Tuple[List[float], List[str]]:
    """Construct rolling n-gram contexts anchored on each token (previous-only)."""
    if ngram < 1:
        raise ValueError("ngram must be >= 1")
    use_mode = (use or "previous").lower()
    if use_mode != "previous":
        raise ValueError(f"Unsupported ngram context mode: {use}")
    if not word_events:
        return [], []
    times: List[float] = []
    contexts: List[str] = []
    window: List[str] = []
    for word, onset, offset in word_events:
        token = str(word).strip()
        if not token:
            continue
        window.append(token)
        times.append(float(onset))
        start_idx = max(0, len(window) - ngram)
        # Use preceding-only context (previous ngram words), excluding the current token beyond window length.
        contexts.append(" ".join(window[start_idx:-1]))
    return times, contexts


def make_tr_windows(
    n_tr: int,
    tr_s: float,
    window_len_tr: int,
    stride_tr: int,
    hrf_shift_tr: int = 0,
) -> List[Tuple[int, int]]:
    """Generate contiguous TR windows respecting the HRF shift."""
    if n_tr <= 0:
        return []
    start = max(0, hrf_shift_tr)
    windows: List[Tuple[int, int]] = []
    while start + window_len_tr <= n_tr:
        end = start + window_len_tr
        windows.append((start, end))
        start += stride_tr
    return windows


def reference_text_windows(
    word_list: Sequence[WordEvent],
    windows_sec: Sequence[Tuple[float, float]],
) -> List[str]:
    """Build reference texts for each time window."""
    references: List[str] = []
    for win_start, win_end in windows_sec:
        tokens: List[str] = []
        for word, onset, offset in word_list:
            if offset <= win_start:
                continue
            if onset >= win_end:
                break
            tokens.append(word)
        references.append(" ".join(tokens))
    return references
