#!/usr/bin/env python3
"""
Align a fallback embedding space to English1000 and report transcript coverage.

This script learns an orthogonal map that projects vectors from a fallback
embedding model (e.g., Word2Vec, fastText) into the English1000 space used by
this project. It also measures token-level coverage for every available
subject/story transcript when combining both vocabularies.

Example:
    python scripts/align_fallback_embeddings.py \
        --config configs/demo.yaml \
        --fallback-path /path/to/GoogleNews-vectors-negative300.bin \
        --fallback-format word2vec \
        --fallback-binary \
        --output-transform misc/fallback_to_english1000.npz \
        --coverage-report misc/embedding_coverage.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from gensim.models import KeyedVectors
except Exception:  # pragma: no cover - optional dependency
    KeyedVectors = None  # type: ignore

from src.decoding import load_transcript_words
from src.edm_ccm import English1000Loader
from src.utils import load_yaml


def safe_divide(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a fallback embedding space to English1000 and compute combined coverage.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo.yaml"),
        help="Path to the YAML config that defines dataset paths (default: configs/demo.yaml).",
    )
    parser.add_argument(
        "--fallback-path",
        type=Path,
        required=True,
        help="Path to the fallback embedding model (Word2Vec/fastText format).",
    )
    parser.add_argument(
        "--fallback-format",
        type=str,
        choices={"word2vec"},
        default="word2vec",
        help="Format of the fallback embeddings (currently only word2vec-compatible files are supported).",
    )
    parser.add_argument(
        "--fallback-binary",
        action="store_true",
        help="Indicate that the fallback embedding file is in binary word2vec format.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1000,
        help="Minimum number of shared tokens required to compute the alignment (default: 1000).",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=0,
        help="Optional cap on the number of overlapping tokens used for alignment (0 uses all).",
    )
    parser.add_argument(
        "--output-transform",
        type=Path,
        default=Path("misc/fallback_to_english1000.npz"),
        help="Where to write the learned transform (NumPy .npz).",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path("misc/embedding_coverage.csv"),
        help="Where to store the per-story coverage report (CSV).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of subjects to include when measuring coverage.",
    )
    parser.add_argument(
        "--stories",
        nargs="*",
        default=None,
        help="Optional subset of stories to include when measuring coverage.",
    )
    return parser.parse_args()


def load_fallback_model(path: Path, fmt: str, binary: bool) -> "KeyedVectors":
    if fmt != "word2vec":
        raise ValueError(f"Unsupported fallback format: {fmt}")
    if KeyedVectors is None:
        raise RuntimeError("gensim is required to load word2vec-format embeddings. Install gensim and retry.")
    print(f"[INFO] Loading fallback embeddings from {path} ...")
    model = KeyedVectors.load_word2vec_format(str(path), binary=binary)
    print(f"[INFO] Loaded fallback embeddings (tokens={len(model)}, dim={model.vector_size}).")
    return model


def normalize_token(token: str) -> str:
    return token.strip().lower()


def compute_alignment(
    english_lookup: Dict[str, np.ndarray],
    fallback_model: "KeyedVectors",
    *,
    min_overlap: int,
    max_overlap: int = 0,
) -> Dict[str, np.ndarray]:
    english_keys = set(english_lookup.keys())
    fallback_keys = set(fallback_model.key_to_index.keys())
    overlap_tokens = sorted(english_keys & fallback_keys)
    if len(overlap_tokens) < min_overlap:
        raise RuntimeError(f"Insufficient overlap between vocabularies: found {len(overlap_tokens)}, " f"but --min-overlap is {min_overlap}.")
    if max_overlap and len(overlap_tokens) > max_overlap:
        rng = np.random.default_rng(seed=0)
        overlap_tokens = sorted(rng.choice(overlap_tokens, size=max_overlap, replace=False).tolist())

    english_dim = len(next(iter(english_lookup.values())))
    fallback_dim = fallback_model.vector_size
    english_matrix = np.empty((len(overlap_tokens), english_dim), dtype=np.float64)
    fallback_matrix = np.empty((len(overlap_tokens), fallback_dim), dtype=np.float64)

    for idx, token in enumerate(overlap_tokens):
        english_matrix[idx] = english_lookup[token]
        fallback_matrix[idx] = fallback_model.get_vector(token)

    english_mean = english_matrix.mean(axis=0)
    fallback_mean = fallback_matrix.mean(axis=0)
    english_centered = english_matrix - english_mean
    fallback_centered = fallback_matrix - fallback_mean

    cross_cov = fallback_centered.T @ english_centered
    U, _, Vt = np.linalg.svd(cross_cov, full_matrices=False)
    rotation = U @ Vt

    return {
        "rotation": rotation.astype(np.float32),
        "fallback_mean": fallback_mean.astype(np.float32),
        "english_mean": english_mean.astype(np.float32),
        "tokens_used": np.array(overlap_tokens),
    }


def discover_subject_story_pairs(
    paths: Dict[str, str],
    *,
    subjects: Optional[Sequence[str]] = None,
    stories: Optional[Sequence[str]] = None,
    default_subject: str = "UTS01",
    default_story: str = "wheretheressmoke",
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    combos: set[Tuple[str, str]] = set()
    failures: List[Dict[str, str]] = []

    def _canon(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    transcript_variants: set[str] = set()

    candidate_transcript_roots: List[Path] = []
    transcripts_root_cfg = paths.get("transcripts")
    if transcripts_root_cfg:
        candidate_transcript_roots.append(Path(transcripts_root_cfg))
    data_root_cfg = paths.get("data_root")
    if data_root_cfg:
        dr_cfg = Path(data_root_cfg)
        candidate_transcript_roots.append(dr_cfg / "derivative" / "TextGrids")
        candidate_transcript_roots.append(dr_cfg / "derivatives" / "TextGrids")

    for root in candidate_transcript_roots:
        if not root.exists():
            continue
        for path in root.glob("**/*"):
            if path.is_file():
                transcript_variants.add(_canon(path.stem))

    def _norm(seq: Optional[Sequence[str]]) -> List[str]:
        return [str(item).strip() for item in (seq or []) if str(item).strip()]

    subjects_override = _norm(subjects)
    stories_override = _norm(stories)

    cache_root = Path(paths.get("cache", "data_cache"))
    if cache_root.exists():
        for subj_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
            for story_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
                combos.add((subj_dir.name, story_dir.name))

    transcripts_root = paths.get("transcripts")
    if transcripts_root:
        tr_root = Path(transcripts_root)
        if tr_root.exists():
            for subj_dir in sorted(p for p in tr_root.iterdir() if p.is_dir()):
                story_dirs = [d for d in subj_dir.iterdir() if d.is_dir()]
                if story_dirs:
                    for story_dir in story_dirs:
                        combos.add((subj_dir.name, story_dir.name))
                        transcript_variants.add(_canon(story_dir.name))
                else:
                    for file in subj_dir.glob("*.*"):
                        if file.is_file():
                            combos.add((subj_dir.name, file.stem))
                            transcript_variants.add(_canon(file.stem))
            for file in tr_root.glob("*.*"):
                if file.is_file():
                    transcript_variants.add(_canon(file.stem))
                if file.is_file() and "_" in file.stem:
                    sub_name, story_name = file.stem.split("_", 1)
                    combos.add((sub_name, story_name))
                    transcript_variants.add(_canon(story_name))

    if data_root_cfg:
        dr_root = Path(data_root_cfg)
        if dr_root.exists():
            for subj_dir in sorted(dr_root.glob("sub-*")):
                subject_name = subj_dir.name.replace("sub-", "", 1)
                for func_dir in sorted(subj_dir.glob("ses-*/func")):
                    for bold_file in func_dir.glob("*task-*_bold.nii.gz"):
                        task_part = bold_file.name.split("task-")[-1]
                        task_part = task_part.split("_", 1)[0]
                        if not task_part:
                            continue
                        canon = _canon(task_part)
                        if transcript_variants and canon not in transcript_variants:
                            continue
                        combos.add((subject_name, task_part))

    if subjects_override and stories_override:
        combos.update((sub, story) for sub in subjects_override for story in stories_override)
    elif subjects_override:
        fallback_stories = {story for _, story in combos} or {default_story}
        combos.update((sub, story) for sub in subjects_override for story in fallback_stories)
    elif stories_override:
        fallback_subjects = {sub for sub, _ in combos} or {default_subject}
        combos.update((sub, story) for sub in fallback_subjects for story in stories_override)

    combos.add((default_subject, default_story))

    valid_pairs: List[Tuple[str, str]] = []
    for subject, story in sorted(combos):
        try:
            load_transcript_words(paths, subject, story)
        except FileNotFoundError:
            failures.append({"subject": subject, "story": story, "error": "transcript not found"})
            continue
        except Exception as exc:
            failures.append({"subject": subject, "story": story, "error": str(exc)})
            continue
        valid_pairs.append((subject, story))

    return valid_pairs, failures


def collect_tokens(paths: Dict[str, str], subject: str, story: str) -> List[str]:
    events = load_transcript_words(paths, subject, story)
    tokens = []
    for word, _, _ in events:
        token = normalize_token(word)
        if token:
            tokens.append(token)
    return tokens


def measure_coverage(
    paths: Dict[str, str],
    english_lookup: Dict[str, np.ndarray],
    fallback_model: "KeyedVectors",
    subject_story_pairs: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    english_vocab = set(english_lookup.keys())
    fallback_vocab = set(fallback_model.key_to_index.keys())

    seen_stories: set[str] = set()
    rows = []

    for subject, story in subject_story_pairs:
        canon_story = story.lower()
        if canon_story in seen_stories:
            continue
        seen_stories.add(canon_story)
        try:
            tokens = collect_tokens(paths, subject, story)
        except Exception as exc:
            print(f"[WARN] Skipping coverage for {subject} / {story}: {exc}", file=sys.stderr)
            continue

        total_tokens = len(tokens)
        english_hits = sum(1 for tok in tokens if tok in english_vocab)
        fallback_hits = sum(1 for tok in tokens if tok in fallback_vocab)
        combined_hits = sum(1 for tok in tokens if (tok in english_vocab) or (tok in fallback_vocab))

        unique_tokens = len(set(tokens))
        english_unique = len({tok for tok in set(tokens) if tok in english_vocab})
        fallback_unique = len({tok for tok in set(tokens) if tok in fallback_vocab})
        combined_unique = len({tok for tok in set(tokens) if (tok in english_vocab) or (tok in fallback_vocab)})

        rows.append(
            {
                "story": story,
                "subject": subject,
                "tokens_total": total_tokens,
                "tokens_english": english_hits,
                "tokens_fallback": fallback_hits,
                "tokens_combined": combined_hits,
                "pct_tokens_english": 100.0 * safe_divide(english_hits, total_tokens),
                "pct_tokens_fallback": 100.0 * safe_divide(fallback_hits, total_tokens),
                "pct_tokens_combined": 100.0 * safe_divide(combined_hits, total_tokens),
                "unique_tokens": unique_tokens,
                "unique_english": english_unique,
                "unique_fallback": fallback_unique,
                "unique_combined": combined_unique,
                "pct_unique_english": 100.0 * safe_divide(english_unique, unique_tokens),
                "pct_unique_fallback": 100.0 * safe_divide(fallback_unique, unique_tokens),
                "pct_unique_combined": 100.0 * safe_divide(combined_unique, unique_tokens),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})

    english1000_path = Path(paths.get("data_root", "")) / "derivative" / "english1000sm.hf5"
    if not english1000_path.exists():
        raise FileNotFoundError(f"English1000 embedding file not found at {english1000_path}")

    english_loader = English1000Loader(english1000_path)
    english_lookup = {token: np.asarray(vec, dtype=np.float32) for token, vec in english_loader.lookup.items()}
    english_dim = len(next(iter(english_lookup.values())))
    print(f"[INFO] Loaded English1000 embeddings ({len(english_lookup)} tokens, dim={english_dim}).")

    fallback_model = load_fallback_model(args.fallback_path, args.fallback_format, args.fallback_binary)

    transform = compute_alignment(
        english_lookup,
        fallback_model,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
    )

    args.output_transform.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_transform,
        rotation=transform["rotation"],
        fallback_mean=transform["fallback_mean"],
        english_mean=transform["english_mean"],
        tokens_used=transform["tokens_used"],
        meta=json.dumps(
            {
                "english_dim": english_dim,
                "fallback_dim": fallback_model.vector_size,
                "overlap_tokens": len(transform["tokens_used"]),
                "fallback_path": str(args.fallback_path),
            }
        ),
    )
    print(f"[INFO] Saved fallback->English1000 transform to {args.output_transform} " f"(overlap tokens={len(transform['tokens_used'])}).")

    subject_story_pairs, discovery_failures = discover_subject_story_pairs(
        paths,
        subjects=args.subjects,
        stories=args.stories,
        default_subject=cfg.get("subject", "UTS01"),
        default_story=cfg.get("story", "wheretheressmoke"),
    )

    if discovery_failures:
        print(f"[WARN] {len(discovery_failures)} transcript(s) could not be loaded during discovery.")

    print(f"[INFO] Measuring coverage across {len(subject_story_pairs)} subject/story combinations.")
    coverage_df = measure_coverage(paths, english_lookup, fallback_model, subject_story_pairs)

    args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.sort_values("pct_tokens_combined", ascending=True).to_csv(args.coverage_report, index=False)
    print(f"[INFO] Coverage report saved to {args.coverage_report}")

    if not coverage_df.empty:
        overall_total = coverage_df["tokens_total"].sum()
        overall_primary = coverage_df["tokens_english"].sum()
        overall_combined = coverage_df["tokens_combined"].sum()
        print(
            f"[INFO] Overall coverage: English1000={overall_primary}/{overall_total} "
            f"({100.0 * safe_divide(overall_primary, overall_total):.2f}%), "
            f"Combined={overall_combined}/{overall_total} "
            f"({100.0 * safe_divide(overall_combined, overall_total):.2f}%)."
        )
        print("[INFO] Lowest combined coverage stories:")
        print(
            coverage_df.sort_values("pct_tokens_combined", ascending=True)[["story", "subject", "pct_tokens_combined", "pct_unique_combined"]]
            .head(10)
            .to_string(index=False)
        )

    if discovery_failures:
        failures_path = args.coverage_report.with_suffix(".failures.json")
        with failures_path.open("w") as fh:
            json.dump(discovery_failures, fh, indent=2)
        print(f"[WARN] Discovery failures written to {failures_path}")


if __name__ == "__main__":
    main()
