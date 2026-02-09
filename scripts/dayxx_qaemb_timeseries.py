#!/usr/bin/env python
"""
Batch QA-Emb time-series generation for ds003020 transcripts.

This script refactors notebooks/DayXX_qaemb_timeseries.ipynb into an HPC-friendly
Python entrypoint that runs end-to-end without any interactive widgets.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Ensure project root is importable
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from src.utils import load_yaml  # noqa: E402
from src.decoding import build_ngram_contexts, load_transcript_words  # noqa: E402
from src.day19_category_builder import apply_smoothing_kernel, build_smoothing_kernel, build_tr_edges, tr_token_overlap  # noqa: E402

logger = logging.getLogger(__name__)

def load_QAEmb_without_imodelsx_init():
    """
    Load imodelsx.qaemb.QAEmb without executing imodelsx/__init__.py.
    Also loads imodelsx.llm so QAEmb can call imodelsx.llm.get_llm().
    """
    import sys
    import types
    from pathlib import Path
    import importlib.util

    # Find site-packages directory that contains imodelsx
    pkg_root = None
    for p in map(Path, sys.path):
        if (p / "imodelsx" / "qaemb" / "qaemb.py").exists():
            pkg_root = p / "imodelsx"
            break
    if pkg_root is None:
        raise RuntimeError("Could not locate imodelsx in sys.path")

    # Create a minimal package module for 'imodelsx' (no __init__.py execution)
    if "imodelsx" not in sys.modules:
        pkg = types.ModuleType("imodelsx")
        pkg.__path__ = [str(pkg_root)]
        sys.modules["imodelsx"] = pkg
    else:
        pkg = sys.modules["imodelsx"]

    # Manually load imodelsx.llm
    llm_path = pkg_root / "llm.py"
    llm_spec = importlib.util.spec_from_file_location("imodelsx.llm", str(llm_path))
    llm_mod = importlib.util.module_from_spec(llm_spec)
    sys.modules["imodelsx.llm"] = llm_mod
    llm_spec.loader.exec_module(llm_mod)
    pkg.llm = llm_mod  # <-- this is what QAEmb expects

    # Manually load imodelsx.qaemb.qaemb
    qa_path = pkg_root / "qaemb" / "qaemb.py"
    qa_spec = importlib.util.spec_from_file_location("imodelsx.qaemb.qaemb", str(qa_path))
    qa_mod = importlib.util.module_from_spec(qa_spec)
    sys.modules["imodelsx.qaemb.qaemb"] = qa_mod
    qa_spec.loader.exec_module(qa_mod)

    return qa_mod.QAEmb

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA-Emb time series from transcript tokens.")
    parser.add_argument(
        "--config",
        default="configs/demo.yaml",
        help="Path to YAML config (default: configs/demo.yaml).",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject ID (overrides config).",
    )
    parser.add_argument(
        "--story",
        default=None,
        help="Story ID (overrides config).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Override project root (defaults to repository root inferred from this file).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--plot-preview",
        action="store_true",
        help="Save a small preview plot for the first few QA questions.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run lightweight validation checks without changing outputs.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def resolve_project_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return Path(arg_root).expanduser().resolve()
    return DEFAULT_PROJECT_ROOT


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_yaml(config_path)
    logger.info("Loaded config from %s", config_path)
    return cfg


def abbreviate_question(q: str, max_words: int = 3) -> str:
    q = str(q).strip()
    if q.endswith("?"):
        q = q[:-1]
    words = q.split()
    return " ".join(words[:max_words])


def build_token_buckets(
    edges: np.ndarray,
    event_records: Sequence[Dict],
    mode: str = "proportional",
) -> List[List[Dict]]:
    """Assign token events to edge-aligned buckets with optional proportional weighting."""
    if edges.size < 2:
        return []
    buckets: List[List[Dict]] = [[] for _ in range(len(edges) - 1)]
    for rec in event_records:
        start = rec["start"]
        end = rec["end"]
        if end <= edges[0] or start >= edges[-1]:
            continue
        start_idx = max(0, int(np.searchsorted(edges, start, side="right")) - 1)
        end_idx = max(0, int(np.searchsorted(edges, end, side="left")))
        end_idx = min(end_idx, len(buckets) - 1)
        for idx in range(start_idx, end_idx + 1):
            bucket_start = edges[idx]
            bucket_end = edges[idx + 1]
            if mode == "none":
                overlap = 1.0 if not (end <= bucket_start or start >= bucket_end) else 0.0
            else:
                overlap = tr_token_overlap(start, end, bucket_start, bucket_end, "proportional")
            if overlap <= 0:
                continue
            item = {
                "word": rec["word"],
                "overlap": overlap,
                "token_start": start,
                "token_end": end,
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
            }
            if "qa_vec" in rec:
                item["qa_vec"] = rec["qa_vec"]
            buckets[idx].append(item)
    return buckets


def score_qa_time_series(
    edges: np.ndarray,
    buckets: Sequence[Sequence[Dict]],
    n_questions: int,
    *,
    index_name: str = "bin_index",
    prefix: str = "qa_q",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Average QA vectors within each edge bin using overlap weighting."""
    n_bins = len(buckets)
    qa_ts = np.full((n_bins, n_questions), np.nan, dtype=float)
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        num = np.zeros(n_questions, dtype=float)
        denom = 0.0
        for item in bucket:
            qa_vec = item.get("qa_vec")
            if qa_vec is None:
                continue
            w = float(item.get("overlap", 1.0))
            num += qa_vec * w
            denom += w
        if denom > 0:
            qa_ts[i] = num / denom
    data = {
        index_name: np.arange(n_bins, dtype=int),
        "start_sec": edges[:-1],
        "end_sec": edges[1:],
    }
    cols: List[str] = []
    for j in range(n_questions):
        col = f"{prefix}{j:03d}"
        data[col] = qa_ts[:, j]
        cols.append(col)
    df = pd.DataFrame(data)
    return df, qa_ts, cols


def save_qaemb_tokens(qa_matrix: np.ndarray, questions: Sequence[str], qa_file: Path, questions_out: Path) -> None:
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(qa_file, qa_matrix)
    with questions_out.open("w") as fh:
        json.dump(list(questions), fh, indent=2)
    logger.info("Saved token-level QA embeddings to %s", qa_file)
    logger.info("Saved QA question list to %s", questions_out)


def maybe_plot_preview(
    canonical_df: pd.DataFrame,
    tr_df: pd.DataFrame,
    canonical_edges: np.ndarray,
    tr_edges: np.ndarray,
    qa_columns: Sequence[str],
    subject: str,
    story: str,
    smoothing_method: str,
    output_root: Path,
) -> Optional[Path]:
    if not qa_columns:
        logger.warning("Skipping preview plot: no QA columns found.")
        return None
    output_root.mkdir(parents=True, exist_ok=True)
    canonical_time = 0.5 * (canonical_edges[:-1] + canonical_edges[1:])
    tr_time = tr_edges[:-1]
    selected_cols = qa_columns[: min(3, len(qa_columns))]
    plt.figure(figsize=(12, 4))
    for col in selected_cols:
        plt.plot(canonical_time, canonical_df[col], label=f"{col} canonical", linewidth=1.4)
        plt.plot(tr_time, tr_df[col], label=f"{col} TR", linestyle="--", marker=".", markersize=3)
    plt.xlabel("Time (s)")
    plt.ylabel("QA score")
    plt.title(f"{subject} / {story} | smoothing={smoothing_method}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plot_path = output_root / "qaemb_preview.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logger.info("Saved preview plot to %s", plot_path)
    return plot_path


def run_qaemb_timeseries(
    *,
    config_path: Path,
    subject: Optional[str],
    story: Optional[str],
    project_root: Path,
    plot_preview: bool,
    validate: bool = False,
) -> Dict[str, Path]:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {}) or {}
    TR = float(cfg.get("TR", 2.0))
    subject = subject or cfg.get("subject")
    story = story or cfg.get("story")
    if not subject or not story:
        raise ValueError("Subject and story must be provided via arguments or config.")

    qa_cfg = cfg.get("qa_emb", {}) or {}
    qa_questions_rel = qa_cfg.get("questions_path", "configs/qaemb_questions.json")
    qa_checkpoint = qa_cfg.get("checkpoint", "meta-llama/Meta-Llama-3-8B-Instruct")
    context_mode = str(qa_cfg.get("context_mode", "ngram")).lower()
    context_ngram = int(qa_cfg.get("ngram", 10))
    time_anchor = str(qa_cfg.get("time_anchor", "onset")).lower()
    qa_use_cache = bool(qa_cfg.get("use_cache", True))
    seconds_bin_width = float(qa_cfg.get("seconds_bin_width", 0.05))
    smoothing_seconds = float(qa_cfg.get("smoothing_seconds", 1.0))
    smoothing_method = qa_cfg.get("smoothing_method", "moving_average")
    gaussian_sigma_seconds = float(qa_cfg.get("gaussian_sigma_seconds", 0.5 * smoothing_seconds))
    smoothing_pad_mode = qa_cfg.get("smoothing_pad_mode", "reflect")
    save_outputs_flag = bool(qa_cfg.get("save_outputs", True))
    batch_size = int(qa_cfg.get("batch_size", 128))
    hf_token = qa_cfg.get("hf_token")
    if context_mode not in {"ngram"}:
        raise ValueError(f"Unsupported qa_emb.context_mode: {context_mode} (only 'ngram' is supported)")
    if context_ngram < 1:
        raise ValueError("qa_emb.ngram must be >= 1")
    if time_anchor not in {"onset", "midpoint"}:
        raise ValueError(f"Unsupported qa_emb.time_anchor: {time_anchor}")

    questions_path = Path(qa_questions_rel)
    if not questions_path.is_absolute():
        questions_path = project_root / questions_path
    if not questions_path.exists():
        raise FileNotFoundError(f"QA question file not found at {questions_path}")
    QA_QUESTIONS: List[str] = json.loads(questions_path.read_text())
    if not isinstance(QA_QUESTIONS, list) or not all(isinstance(q, str) for q in QA_QUESTIONS):
        raise ValueError("QA questions JSON must be a list of strings.")
    qa_abbrevs = [abbreviate_question(q) for q in QA_QUESTIONS]

    features_root = Path(paths.get("featurestest", "featurestest"))
    if not features_root.is_absolute():
        features_root = project_root / features_root
    features_root = features_root / "qaemb"
    features_root.mkdir(parents=True, exist_ok=True)
    logger.info("Using features root: %s", features_root)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info("CUDA device: %s", device_name)
        except Exception:
            logger.info("CUDA device: <unavailable>")

    story_events = load_transcript_words(paths, subject, story)
    if not story_events:
        raise ValueError(f"No transcript events found for {subject} {story}.")
    token_df = pd.DataFrame(story_events, columns=["word", "start", "end"])
    token_df["word"] = token_df["word"].astype(str).str.strip()
    token_df["midpoint"] = 0.5 * (token_df["start"] + token_df["end"])
    token_df["token_index"] = np.arange(len(token_df))
    if context_mode == "ngram":
        token_df = token_df[token_df["word"] != ""].reset_index(drop=True)
        if token_df.empty:
            raise ValueError("No transcript tokens available after filtering empty tokens for ngram mode.")
    logger.info("Loaded %s transcript tokens for %s / %s", len(token_df), subject, story)
    word_events = [(row.word, float(row.start), float(row.end)) for row in token_df.itertuples(index=False, name="TokenRow")]

    if hf_token:
        os.environ.setdefault("HF_TOKEN", str(hf_token))
    elif "HF_TOKEN" not in os.environ:
        logger.warning("HF_TOKEN is not set; gated checkpoints may fail. Set it before rerunning.")

    qa_root = features_root / "tokens" / subject
    qa_root.mkdir(parents=True, exist_ok=True)
    qa_file = qa_root / f"{story}_qaemb_tokens.npy"
    qa_questions_out = qa_root / f"{story}_qaemb_questions.json"

    qa_matrix = None
    if qa_file.exists():
        cached = np.load(qa_file)
        if cached.shape == (len(token_df), len(QA_QUESTIONS)):
            qa_matrix = cached
            logger.info("Loaded cached QA embeddings from %s", qa_file)
        else:
            warnings.warn(
                f"Cached QA embeddings had shape {cached.shape}; expected {(len(token_df), len(QA_QUESTIONS))}. Recomputing."
            )

    if qa_matrix is None:
        try:
            QAEmb = load_QAEmb_without_imodelsx_init()
        except ImportError as exc:  # pragma: no cover - dependency managed externally
            logger.error("imodelsx is required for QAEmb encoding. Install it in the cluster environment.")
            raise
        embedder = QAEmb(
            questions=QA_QUESTIONS,
            checkpoint=qa_checkpoint,
            use_cache=qa_use_cache,
        )
        # --- Force ultra-short deterministic outputs to avoid CUDA OOM ---
        import functools, inspect

        # imodelsx calls a HF text-generation pipeline under the hood.
        # We hard-cap generation to 1 token ("Yes"/"No") to slash memory.
        llm_sig = inspect.signature(embedder.llm.__call__)
        forced = {}
        for k, v in [
            ("max_new_tokens", 1),
            ("do_sample", False),
            ("temperature", 0.0),
            ("top_p", 1.0),
            ("num_return_sequences", 1),
            ("return_full_text", False),
        ]:
            if k in llm_sig.parameters:
                forced[k] = v
        if forced:
            embedder.llm = functools.partial(embedder.llm, **forced)
            logger.info("Forced LLM kwargs: %s", forced)

        # Make sure we don't accidentally batch inside the HF pipeline
        try:
            pipe_sig = inspect.signature(embedder.llm.pipeline_.__call__)
            if "batch_size" in pipe_sig.parameters:
                embedder.llm.pipeline_.batch_size = 1
        except Exception:
            pass

        expected_tokens = len(token_df)
        _, examples = build_ngram_contexts(word_events, context_ngram, use="previous")
        logger.info(
            "Built ngram contexts: %s tokens, ngram=%s, time_anchor=%s",
            len(examples),
            context_ngram,
            time_anchor,
        )
        if len(examples) != expected_tokens:
            raise ValueError(
                f"Context/example count mismatch: got {len(examples)} examples, expected {expected_tokens} tokens."
            )

        qa_rows = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            emb = embedder(batch)
            qa_rows.append(np.asarray(emb, dtype=float))
        qa_matrix = np.vstack(qa_rows) if qa_rows else np.empty((0, len(QA_QUESTIONS)), dtype=float)
        save_qaemb_tokens(qa_matrix, QA_QUESTIONS, qa_file, qa_questions_out)
    else:
        if not qa_questions_out.exists():
            qa_questions_out.parent.mkdir(parents=True, exist_ok=True)
            qa_questions_out.write_text(json.dumps(QA_QUESTIONS, indent=2))
    assert qa_matrix.shape == (len(token_df), len(QA_QUESTIONS)), "QA matrix shape mismatch."
    logger.info("qa_matrix shape: %s", qa_matrix.shape)

    tr_edges = build_tr_edges(story_events, TR)
    if not np.all(np.diff(tr_edges) > 0):
        raise ValueError("Non-monotone TR edges.")
    logger.info("TR bins: %s", len(tr_edges) - 1)

    anchor_times: List[float] = []
    event_records: List[Dict] = []
    for i, row in token_df.iterrows():
        start_val = float(row["start"])
        end_val = float(row["end"])
        anchor_time = start_val if time_anchor == "onset" else float(row["midpoint"])
        anchor_times.append(anchor_time)
        event_records.append(
            {
                "word": row["word"],
                "start": anchor_time,
                "end": anchor_time,
                "qa_vec": qa_matrix[i].astype(float),
            }
        )

    token_times = np.asarray(anchor_times, dtype=float)
    qa_columns = [f"qa_q{j:03d}" for j in range(len(QA_QUESTIONS))]

    # Resample irregular word-time QA to TR grid directly.
    tr_values_raw = _resample_irregular_to_edges(token_times, qa_matrix, tr_edges)
    smoothing_kernel = build_smoothing_kernel(
        TR,
        smoothing_seconds,
        method=smoothing_method,
        gaussian_sigma_seconds=gaussian_sigma_seconds,
    )
    smoothing_applied = smoothing_kernel.size > 1
    if tr_values_raw.size and smoothing_applied:
        tr_values_smoothed = apply_smoothing_kernel(tr_values_raw, smoothing_kernel, pad_mode=smoothing_pad_mode)
    else:
        tr_values_smoothed = tr_values_raw.copy()

    # Canonical seconds output now mirrors the TR grid to avoid sparse micro-bins.
    canonical_edges = tr_edges
    canonical_df_raw = pd.DataFrame(
        {
            "bin_index": np.arange(len(canonical_edges) - 1, dtype=int),
            "start_sec": canonical_edges[:-1],
            "end_sec": canonical_edges[1:],
        }
    )
    canonical_df_smoothed = canonical_df_raw.copy()
    for j, col in enumerate(qa_columns):
        canonical_df_raw[col] = tr_values_raw[:, j]
        canonical_df_smoothed[col] = tr_values_smoothed[:, j]
    canonical_df_selected = canonical_df_smoothed if smoothing_applied else canonical_df_raw

    base_df = pd.DataFrame({"tr_index": np.arange(len(tr_edges) - 1, dtype=int), "start_sec": tr_edges[:-1], "end_sec": tr_edges[1:]})
    tr_df_raw = base_df.copy()
    tr_df_smoothed = base_df.copy()
    for j, col in enumerate(qa_columns):
        tr_df_raw[col] = tr_values_raw[:, j]
        tr_df_smoothed[col] = tr_values_smoothed[:, j]
    tr_df_selected = tr_df_smoothed if smoothing_applied else tr_df_raw

    logger.info(
        "canonical_matrix shape: %s, tr_matrix shape: %s",
        canonical_df_selected[qa_columns].to_numpy().shape,
        tr_df_selected[qa_columns].to_numpy().shape,
    )
    canonical_buckets = [None] * (len(canonical_edges) - 1)
    tr_buckets = [None] * (len(tr_edges) - 1)
    if validate:
        canonical_check = canonical_df_selected[qa_columns].to_numpy() if qa_columns else np.zeros((len(canonical_df_selected), 0))
        tr_check = tr_df_selected[qa_columns].to_numpy() if qa_columns else np.zeros((len(tr_df_selected), 0))
        _validate_outputs(
            n_questions=len(QA_QUESTIONS),
            canonical_matrix=canonical_check,
            tr_matrix=tr_check,
            canonical_buckets=canonical_buckets,
            tr_buckets=tr_buckets,
            allow_nans=False,
            mode=context_mode,
        )

    output_root = features_root / "subjects" / subject / story
    canonical_root = features_root / "stories" / story
    plot_root = features_root / "plots" / subject / story
    outputs: Dict[str, Path] = {}
    plot_path: Optional[Path] = None

    if save_outputs_flag:
        output_root.mkdir(parents=True, exist_ok=True)
        canonical_root.mkdir(parents=True, exist_ok=True)

        canonical_csv = canonical_root / "qaemb_timeseries_seconds.csv"
        canonical_df_selected.to_csv(canonical_csv, index=False)
        outputs["canonical_csv"] = canonical_csv
        if smoothing_applied:
            canonical_raw_csv = canonical_root / "qaemb_timeseries_seconds_raw.csv"
            canonical_df_raw.to_csv(canonical_raw_csv, index=False)
            outputs["canonical_csv_raw"] = canonical_raw_csv

        tr_csv = output_root / "qaemb_timeseries.csv"
        tr_df_selected.to_csv(tr_csv, index=False)
        outputs["tr_csv"] = tr_csv
        if smoothing_applied:
            tr_raw_csv = output_root / "qaemb_timeseries_raw.csv"
            tr_df_raw.to_csv(tr_raw_csv, index=False)
            outputs["tr_csv_raw"] = tr_raw_csv

        meta = {
            "subject": subject,
            "story": story,
            "tr_seconds": TR,
            "seconds_bin_width": seconds_bin_width,
            "smoothing_seconds": smoothing_seconds,
            "smoothing_method": smoothing_method,
            "gaussian_sigma_seconds": gaussian_sigma_seconds,
            "smoothing_pad_mode": smoothing_pad_mode,
            "questions_path": str(questions_path),
            "checkpoint": qa_checkpoint,
            "n_questions": len(QA_QUESTIONS),
            "context_mode": context_mode,
            "context_ngram": context_ngram,
            "time_anchor": time_anchor,
            "n_tokens": len(token_df),
            "qa_abbreviations": qa_abbrevs,
        }
        meta_path = output_root / "qaemb_metadata.json"
        with meta_path.open("w") as fh:
            json.dump(meta, fh, indent=2)
        outputs["metadata"] = meta_path
        logger.info("Saved canonical QA series to %s", canonical_csv)
        logger.info("Saved TR QA series to %s", tr_csv)
        logger.info("Saved metadata to %s", meta_path)
    else:
        logger.info("Skipping file outputs (save_outputs is False).")

    if plot_preview:
        plot_path = maybe_plot_preview(
            canonical_df_selected,
            tr_df_selected,
            canonical_edges,
            tr_edges,
            qa_columns,
            subject,
            story,
            smoothing_method,
            plot_root,
        )
    if plot_path:
        outputs["preview_plot"] = plot_path

    return outputs


def _validate_outputs(
    *,
    n_questions: int,
    canonical_matrix: np.ndarray,
    tr_matrix: np.ndarray,
    canonical_buckets: Sequence[Sequence[Dict]],
    tr_buckets: Sequence[Sequence[Dict]],
    allow_nans: bool,
    mode: str,
) -> None:
    """Lightweight validation to catch shape/count mismatches during development."""
    if canonical_matrix.shape[1] != n_questions:
        raise ValueError(f"Validation failed ({mode}): canonical questions {canonical_matrix.shape[1]} != {n_questions}")
    if tr_matrix.shape[1] != n_questions:
        raise ValueError(f"Validation failed ({mode}): TR questions {tr_matrix.shape[1]} != {n_questions}")
    if canonical_matrix.shape[0] != len(canonical_buckets):
        raise ValueError(
            f"Validation failed ({mode}): canonical bins {canonical_matrix.shape[0]} != bucket count {len(canonical_buckets)}"
        )
    if tr_matrix.shape[0] != len(tr_buckets):
        raise ValueError(f"Validation failed ({mode}): TR bins {tr_matrix.shape[0]} != bucket count {len(tr_buckets)}")
    has_nan = np.isnan(canonical_matrix).any() or np.isnan(tr_matrix).any()
    if has_nan and not allow_nans:
        raise ValueError(f"Validation failed ({mode}): NaNs detected in aggregated QAEmb values.")


def _resample_irregular_to_edges(token_times: np.ndarray, token_values: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Resample irregular QAEmb values to target bin edges using linear interpolation on bin centers."""
    if token_values.size == 0:
        return np.empty((len(target_edges) - 1, 0), dtype=float)
    if token_times.ndim != 1:
        raise ValueError("token_times must be 1D")
    if token_values.shape[0] != token_times.shape[0]:
        raise ValueError("token_times and token_values length mismatch")
    if token_times.size == 0:
        raise ValueError("token_times is empty")
    order = np.argsort(token_times)
    times_sorted = token_times[order]
    values_sorted = token_values[order]
    centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    out = np.empty((centers.size, token_values.shape[1]), dtype=float)
    for j in range(token_values.shape[1]):
        out[:, j] = np.interp(centers, times_sorted, values_sorted[:, j], left=values_sorted[0, j], right=values_sorted[-1, j])
    return out


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    project_root = resolve_project_root(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    outputs = run_qaemb_timeseries(
        config_path=config_path,
        subject=args.subject,
        story=args.story,
        project_root=project_root,
        plot_preview=args.plot_preview,
        validate=args.validate,
    )
    for label, path in outputs.items():
        logger.info("Output [%s]: %s", label, path)


if __name__ == "__main__":
    main()
