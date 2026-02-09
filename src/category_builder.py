from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.day19_category_builder import (
    EPS,
    aggregate_seconds_to_edges,
    apply_smoothing_kernel,
    build_smoothing_kernel,
    build_states_from_csv,
    build_token_buckets,
    build_tr_edges,
    ensure_serializable,
    expand_category,
    load_clusters_from_csv,
    load_story_words,
    make_category_prototype,
    score_time_series,
)


class EmbeddingBackend:
    """Minimal embedding backend interface."""

    name: str
    lowercase_tokens: bool
    embedding_dim: int

    def get_vector(self, token: str) -> Optional[np.ndarray]:
        raise NotImplementedError

    def __contains__(self, token: str) -> bool:  # pragma: no cover - convenience
        return self.get_vector(token) is not None


@dataclass
class LLMEmbeddingBackend(EmbeddingBackend):
    """
    Loads a precomputed token -> embedding map from disk (npz/npy/json).
    The map is kept in-memory for fast lookup.
    """

    path: Path
    lowercase_tokens: bool = True
    name: str = "llm"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._token_to_vec = self._load_embeddings(self.path)
        if not self._token_to_vec:
            raise ValueError(f"No embeddings loaded from {self.path}")
        first_vec = next(iter(self._token_to_vec.values()))
        self.embedding_dim = int(np.asarray(first_vec).shape[0])

    def _load_embeddings(self, path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(path.read_text())
            return self._normalize_dict(raw)
        if suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            if "tokens" in data.files and "embeddings" in data.files:
                tokens = data["tokens"]
                vectors = data["embeddings"]
                return self._from_parallel_arrays(tokens, vectors)
            if data.files:
                arr = data[data.files[0]]
                return self._coerce_emb_map(arr)
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            return self._coerce_emb_map(arr)
        raise ValueError(f"Unsupported embedding format for {path}")

    def _from_parallel_arrays(self, tokens: Iterable[Any], vectors: np.ndarray) -> Dict[str, np.ndarray]:
        if vectors.ndim != 2:
            raise ValueError("embeddings array must be 2D (n_tokens x dim)")
        if len(tokens) != vectors.shape[0]:
            raise ValueError("tokens and embeddings length mismatch")
        out: Dict[str, np.ndarray] = {}
        for tok, vec in zip(tokens, vectors):
            key = self._norm_token(tok)
            if key:
                out[key] = np.asarray(vec, dtype=float)
        return out

    def _coerce_emb_map(self, obj: Any) -> Dict[str, np.ndarray]:
        if isinstance(obj, Mapping):
            return self._normalize_dict(obj)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            inner = obj.item()
            if isinstance(inner, Mapping):
                return self._normalize_dict(inner)
        raise ValueError("Embedding file must contain a dict-like token -> vector mapping or (tokens, embeddings).")

    def _normalize_dict(self, raw: Mapping[Any, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for tok, vec in raw.items():
            key = self._norm_token(tok)
            if not key:
                continue
            arr = np.asarray(vec, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"Embedding for token '{tok}' is not 1D.")
            out[key] = arr
        return out

    def _norm_token(self, token: Any) -> str:
        if token is None:
            return ""
        text = str(token).strip()
        if self.lowercase_tokens:
            text = text.lower()
        return text

    def get_vector(self, token: str) -> Optional[np.ndarray]:
        key = self._norm_token(token)
        if not key:
            return None
        vec = self._token_to_vec.get(key)
        if vec is None:
            return None
        return np.asarray(vec, dtype=float)

    def __contains__(self, token: str) -> bool:
        key = self._norm_token(token)
        return key in self._token_to_vec

    @property
    def vocab(self) -> List[str]:
        return list(self._token_to_vec.keys())

    @property
    def matrix(self) -> np.ndarray:
        if not hasattr(self, "_matrix_cache"):
            self._matrix_cache = np.stack(list(self._token_to_vec.values()), axis=0)
        return self._matrix_cache


def get_embedding_backend(
    backend_name: str,
    *,
    lm_embedding_path: Optional[Path] = None,
    lm_lowercase_tokens: bool = True,
) -> EmbeddingBackend:
    backend = backend_name.lower().strip()
    if backend == "llm":
        if lm_embedding_path is None:
            raise ValueError("LLM backend selected but lm_embedding_path is missing.")
        return LLMEmbeddingBackend(path=lm_embedding_path, lowercase_tokens=lm_lowercase_tokens, name="llm")
    raise ValueError(f"Unsupported embedding backend '{backend}'. Only 'llm' is implemented in this module.")


def _build_category_states(
    categories_cfg: Dict[str, Any],
    *,
    prototype_weight_power: float,
    category_score_method: str,
    exp_enabled: bool,
    exp_top_k: int,
    exp_min_sim: float,
    token_lookup: MutableMapping[str, np.ndarray],
    vocab_matrix: Optional[np.ndarray],
    vocab_words: Sequence[str],
    cluster_csv_path: str,
) -> tuple[Dict[str, Dict], Dict[str, Dict], List[str], List[str]]:
    category_states: Dict[str, Dict] = {}
    category_definitions: Dict[str, Dict] = {}
    category_sets = categories_cfg.get("sets", {})
    category_set_name = categories_cfg.get("category_set") or (sorted(category_sets.keys())[0] if category_sets else None)

    if cluster_csv_path:
        csv_clusters = load_clusters_from_csv(cluster_csv_path)
        category_states, category_definitions = build_states_from_csv(
            csv_clusters,
            token_lookup,
            None,
            weight_power=prototype_weight_power,
        )
        category_names = sorted(category_states.keys())
        category_columns = [f"cat_{name}" for name in category_names]
        zero_norm = [k for k, v in category_states.items() if v.get("prototype") is not None and (v.get("prototype_norm") or 0.0) < EPS]
        if zero_norm:
            warnings.warn(f"Zero-norm prototypes (check OOV/weights): {zero_norm}")
        return category_states, category_definitions, category_names, category_columns

    selected_set_spec = category_sets.get(category_set_name, {}) if category_sets else {}
    category_names: List[str] = []
    category_columns: List[str] = []
    seed_oov_counter: Dict[str, int] = {}
    for cat_name, cat_spec in selected_set_spec.items():
        seeds = cat_spec.get("seeds", [])
        explicit_words = cat_spec.get("words", [])
        prototype = None
        found_seeds: List[str] = []
        missing_seeds: List[str] = []
        if seeds:
            prototype, found_seeds, missing_seeds = make_category_prototype(
                seeds, token_lookup, None, allow_single=bool(categories_cfg.get("allow_single_seed", False))
            )
            seed_oov_counter[cat_name] = len(missing_seeds)
            if prototype is None and category_score_method == "similarity":
                warnings.warn(f"Category '{cat_name}' has no usable prototype; TR scores will be NaN.")
        elif category_score_method == "similarity":
            warnings.warn(f"Category {cat_name} has no seeds; similarity method will yield NaNs.")
        lexicon = {word.lower(): 1.0 for word in explicit_words}
        for seed in found_seeds:
            lexicon.setdefault(seed.lower(), 1.0)
        prototype_norm = None
        expanded_words = {}
        if prototype is not None:
            prototype_norm = float(np.linalg.norm(prototype))
            if exp_enabled and vocab_matrix is not None:
                expanded_words = expand_category(prototype, vocab_matrix, vocab_words, exp_top_k, exp_min_sim)
                for word, weight in expanded_words.items():
                    lexicon.setdefault(word.lower(), float(weight))
        if not lexicon and category_score_method == "count":
            warnings.warn(f"Category {cat_name} lexicon is empty; counts will be NaN.")
        category_states[cat_name] = {
            "name": cat_name,
            "seeds": seeds,
            "found_seeds": found_seeds,
            "missing_seeds": missing_seeds,
            "prototype": prototype,
            "prototype_norm": prototype_norm,
            "lexicon": lexicon,
            "expanded_count": len(expanded_words),
            "expansion_params": {
                "enabled": exp_enabled,
                "top_k": exp_top_k,
                "min_sim": exp_min_sim,
            },
        }
        category_definitions[cat_name] = {
            "seeds": seeds,
            "found_seeds": found_seeds,
            "missing_seeds": missing_seeds,
            "prototype_dim": int(prototype.shape[0]) if isinstance(prototype, np.ndarray) else 0,
            "prototype_norm": prototype_norm,
            "expanded_neighbors": ensure_serializable(expanded_words),
            "lexicon": {word: float(weight) for word, weight in sorted(category_states[cat_name]["lexicon"].items())},
        }
        category_names.append(cat_name)
        category_columns.append(f"cat_{cat_name}")
    if seed_oov_counter:
        print("Category seeds missing counts:", dict(seed_oov_counter))
    print(f"Prepared {len(category_names)} categories: {category_names}")
    return category_states, category_definitions, category_names, category_columns


def generate_category_time_series(
    subject: str,
    story: str,
    *,
    cfg_base: Dict[str, Any],
    categories_cfg_base: Dict[str, Any],
    cluster_csv_path: str,
    temporal_weighting: str,
    prototype_weight_power: float,
    smoothing_seconds: float,
    smoothing_method: str,
    gaussian_sigma_seconds: Optional[float],
    smoothing_pad: str,
    seconds_bin_width: float,
    features_root: Path,
    paths: Dict[str, Any],
    TR: float,
    embedding_backend: EmbeddingBackend,
    save_outputs: bool = True,
    word_events: Optional[Sequence[tuple[str, float, float]]] = None,
) -> Dict[str, Any]:
    """
    Generate category time series using a pluggable embedding backend (e.g., LLM).
    This mirrors Day19 behavior but swaps the embedding source.
    """
    if not subject or not story:
        raise ValueError("Subject and story must be provided.")
    if embedding_backend is None:
        raise ValueError("An embedding backend must be provided.")

    print(f"=== Category build (backend={embedding_backend.name}) for {subject} / {story} ===")

    categories_cfg = json.loads(json.dumps(categories_cfg_base or {}))
    categories_cfg["seconds_bin_width"] = float(seconds_bin_width)
    category_sets = categories_cfg.get("sets", {})
    available_sets = sorted(category_sets.keys())
    category_set_name = categories_cfg.get("category_set") or (available_sets[0] if available_sets else None)
    if cluster_csv_path:
        if not category_set_name:
            category_set_name = "csv_clusters"
        categories_cfg["category_set"] = category_set_name
        categories_cfg["category_score_method"] = "similarity"
        categories_cfg["allow_single_seed"] = True
        categories_cfg["expansion"] = {"enabled": False}
    category_score_method = str(categories_cfg.get("category_score_method", "similarity")).lower()
    overlap_mode = str(categories_cfg.get("overlap_weighting", "proportional")).lower()
    expansion_cfg = categories_cfg.get("expansion", {})
    allow_single = bool(categories_cfg.get("allow_single_seed", False))
    exp_enabled = bool(expansion_cfg.get("enabled", True))
    exp_top_k = int(expansion_cfg.get("top_k", 2000)) if exp_enabled else 0
    exp_min_sim = float(expansion_cfg.get("min_sim", 0.35)) if exp_enabled else 0.0

    output_root = features_root / "subjects" / subject / story
    canonical_root = features_root / "stories" / story
    if save_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        canonical_root.mkdir(parents=True, exist_ok=True)

    if word_events is not None:
        story_events = [(str(w).strip(), float(s), float(e)) for w, s, e in word_events]
    else:
        story_events = load_story_words(paths, subject, story)
    if not story_events:
        raise ValueError(f"No transcript events found for {subject} {story}.")

    print(f"Loaded {len(story_events)} transcript events.")
    tr_edges = build_tr_edges(story_events, TR)
    n_tr = len(tr_edges) - 1
    print(f"TR edges: {len(tr_edges)} (n_tr={n_tr}) spanning {tr_edges[-1]:.2f} seconds.")

    token_lookup: Dict[str, np.ndarray] = {}
    if isinstance(embedding_backend, LLMEmbeddingBackend):
        token_lookup = embedding_backend._token_to_vec  # type: ignore[attr-defined]
    else:
        token_lookup = {}

    vocab_words = list(token_lookup.keys())
    vocab_matrix = np.stack(list(token_lookup.values()), axis=0) if token_lookup else None

    category_states, category_definitions, category_names, category_columns = _build_category_states(
        categories_cfg,
        prototype_weight_power=prototype_weight_power,
        category_score_method=category_score_method,
        exp_enabled=exp_enabled,
        exp_top_k=exp_top_k,
        exp_min_sim=exp_min_sim,
        token_lookup=token_lookup,
        vocab_matrix=vocab_matrix,
        vocab_words=vocab_words,
        cluster_csv_path=cluster_csv_path,
    )

    tw_mode = str(temporal_weighting or "proportional").lower()
    if tw_mode not in {"proportional", "none", "midpoint"}:
        raise ValueError(f"Unsupported temporal weighting: {tw_mode}")

    seconds_bin_width = float(seconds_bin_width)
    if seconds_bin_width <= 0:
        raise ValueError("seconds_bin_width must be positive.")
    smoothing_method = str(smoothing_method or "moving_average").lower()
    gaussian_sigma_seconds = gaussian_sigma_seconds if gaussian_sigma_seconds not in (None, "") else None
    smoothing_pad = str(smoothing_pad or "edge").lower()
    if smoothing_pad not in {"edge", "reflect"}:
        smoothing_pad = "edge"

    embedding_cache: Dict[str, Optional[np.ndarray]] = {}
    event_records: List[Dict[str, Any]] = []
    tokens_with_embeddings = 0
    for word, onset, offset in story_events:
        token = word.strip()
        if not token:
            continue
        key = token.lower() if embedding_backend.lowercase_tokens else token
        if key not in embedding_cache:
            embedding_cache[key] = embedding_backend.get_vector(token)
        emb = embedding_cache[key]
        emb_norm = float(np.linalg.norm(emb)) if emb is not None else None
        if emb is not None:
            tokens_with_embeddings += 1
        event_records.append(
            {
                "word": token,
                "start": float(onset),
                "end": float(offset),
                "embedding": emb,
                "embedding_norm": emb_norm,
            }
        )

    total_tokens = len(event_records)
    print(
        f"Tokens with embeddings: {tokens_with_embeddings}/{total_tokens} (OOV rate={(total_tokens - tokens_with_embeddings) / max(total_tokens, 1):.2%})."
    )
    if not event_records:
        raise ValueError("No token events available for category featurization.")

    max_end_time = max(rec["end"] for rec in event_records)
    canonical_edges = np.arange(0.0, max_end_time + seconds_bin_width, seconds_bin_width, dtype=float)
    if canonical_edges[-1] < max_end_time:
        canonical_edges = np.append(canonical_edges, canonical_edges[-1] + seconds_bin_width)
    if canonical_edges[-1] < max_end_time - 1e-9:
        canonical_edges = np.append(canonical_edges, canonical_edges[-1] + seconds_bin_width)
    assert np.all(np.diff(canonical_edges) > 0), "Non-monotone canonical edges."

    canonical_buckets = build_token_buckets(canonical_edges, event_records, tw_mode)
    empty_canonical = sum(1 for bucket in canonical_buckets if not bucket)
    print(f"Canonical bins without tokens: {empty_canonical}/{len(canonical_buckets)}")

    canonical_df_raw, canonical_matrix = score_time_series(
        canonical_edges,
        canonical_buckets,
        category_states,
        category_names,
        category_columns,
        category_score_method,
        index_name="bin_index",
    )
    canonical_values_raw = canonical_matrix.copy()
    smoothing_kernel = build_smoothing_kernel(
        seconds_bin_width,
        smoothing_seconds,
        method=smoothing_method,
        gaussian_sigma_seconds=gaussian_sigma_seconds,
        causal=True,
    )
    smoothing_applied = smoothing_kernel.size > 1
    if canonical_values_raw.size and smoothing_applied:
        canonical_values_smoothed = apply_smoothing_kernel(canonical_values_raw, smoothing_kernel, pad_mode=smoothing_pad, causal=True)
    else:
        canonical_values_smoothed = canonical_values_raw.copy()

    canonical_df_smoothed = canonical_df_raw.copy()
    if category_columns:
        canonical_df_smoothed.loc[:, category_columns] = canonical_values_smoothed
    canonical_df_selected = canonical_df_smoothed if smoothing_applied else canonical_df_raw

    if save_outputs:
        canonical_root.mkdir(parents=True, exist_ok=True)
        canonical_csv_path = canonical_root / "category_timeseries_seconds.csv"
        canonical_df_selected.to_csv(canonical_csv_path, index=False)
        if smoothing_applied:
            canonical_df_raw.to_csv(canonical_root / "category_timeseries_seconds_raw.csv", index=False)
        canonical_definition_path = canonical_root / "category_definition.json"
        with canonical_definition_path.open("w") as fh:
            definition_payload = dict(ensure_serializable(category_definitions))
            definition_payload["_embedding_backend"] = {
                "name": embedding_backend.name,
                "embedding_dim": int(getattr(embedding_backend, "embedding_dim", 0)),
                "lowercase_tokens": bool(getattr(embedding_backend, "lowercase_tokens", False)),
                "lm_embedding_path": str(getattr(embedding_backend, "path", "")),
            }
            json.dump(definition_payload, fh, indent=2)
        print(f"Saved canonical story series to {canonical_csv_path}")

    tr_buckets = build_token_buckets(tr_edges, event_records, tw_mode)
    empty_tr = sum(1 for bucket in tr_buckets if not bucket)
    print(f"TRs without tokens: {empty_tr}/{len(tr_buckets)}")

    if category_columns:
        tr_values_raw = aggregate_seconds_to_edges(canonical_edges, canonical_values_raw, tr_edges)
        tr_values_smoothed = aggregate_seconds_to_edges(canonical_edges, canonical_values_smoothed, tr_edges)
    else:
        tr_values_raw = np.empty((len(tr_edges) - 1, 0), dtype=float)
        tr_values_smoothed = tr_values_raw

    base_index = np.arange(len(tr_edges) - 1, dtype=int)
    base_df = pd.DataFrame({"tr_index": base_index, "start_sec": tr_edges[:-1], "end_sec": tr_edges[1:]})
    category_df_raw = base_df.copy()
    category_df_smoothed = base_df.copy()
    if category_columns:
        category_df_raw.loc[:, category_columns] = tr_values_raw
        category_df_smoothed.loc[:, category_columns] = tr_values_smoothed
    category_df = category_df_smoothed if smoothing_applied else category_df_raw
    print(category_df.head())

    if category_score_method == "similarity" and category_columns:
        finite_vals = category_df[category_columns].to_numpy(dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size:
            assert np.nanmin(finite_vals) >= -1.0001 and np.nanmax(finite_vals) <= 1.0001, "Similarity scores out of bounds."
    else:
        if category_columns:
            assert (category_df[category_columns].fillna(0.0) >= -1e-9).all().all(), "Count scores must be non-negative."

    if save_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        category_csv_path = output_root / "category_timeseries.csv"
        category_df.to_csv(category_csv_path, index=False)
        if smoothing_applied:
            category_df_raw.to_csv(output_root / "category_timeseries_raw.csv", index=False)
        definition_path = output_root / "category_definition.json"
        with definition_path.open("w") as fh:
            definition_payload = dict(ensure_serializable(category_definitions))
            definition_payload["_embedding_backend"] = {
                "name": embedding_backend.name,
                "embedding_dim": int(getattr(embedding_backend, "embedding_dim", 0)),
                "lowercase_tokens": bool(getattr(embedding_backend, "lowercase_tokens", False)),
                "lm_embedding_path": str(getattr(embedding_backend, "path", "")),
            }
            json.dump(definition_payload, fh, indent=2)
        print(f"Saved category time series to {category_csv_path}")

    trimmed_path = Path(paths.get("figs", "figs")) / subject / story / "day16_decoding" / "semantic_pcs_trimmed.csv"
    max_lag_primary = 0
    trimmed_df = None
    if trimmed_path.exists():
        day16_trim = pd.read_csv(trimmed_path)
        expected_len = len(day16_trim)
        if len(day16_trim) > len(category_df):
            raise ValueError("Day16 trimmed series longer than category series; regenerate Day16 or rerun Day17.")
        max_lag_primary = max(0, len(category_df) - expected_len)
        trimmed_df = category_df.iloc[max_lag_primary:].reset_index(drop=True)
        if save_outputs:
            trimmed_out = trimmed_df.copy()
            trimmed_out.insert(0, "trim_index", np.arange(len(trimmed_out), dtype=int))
            trimmed_out.drop(columns=["tr_index"], inplace=True, errors="ignore")
            trimmed_out.to_csv(output_root / "category_timeseries_trimmed.csv", index=False)
            print(f'Saved trimmed category series to {output_root / "category_timeseries_trimmed.csv"}')
    else:
        warnings.warn("Day16 trimmed PCs not found; skipping auto-alignment.")

    smoothing_meta = {
        "applied": bool(smoothing_applied),
        "seconds": smoothing_seconds,
        "method": smoothing_method,
        "gaussian_sigma_seconds": float(gaussian_sigma_seconds) if gaussian_sigma_seconds is not None else None,
        "kernel_size": int(smoothing_kernel.size),
        "pad_mode": smoothing_pad,
        "bin_width_seconds": seconds_bin_width,
    }

    return {
        "subject": subject,
        "story": story,
        "temporal_weighting": tw_mode,
        "category_columns": category_columns,
        "category_states": category_states,
        "category_definitions": category_definitions,
        "category_score_method": category_score_method,
        "event_records": event_records,
        "canonical_buckets": canonical_buckets,
        "tr_buckets": tr_buckets,
        "canonical_df_raw": canonical_df_raw,
        "canonical_df_smoothed": canonical_df_smoothed,
        "canonical_df_selected": canonical_df_selected,
        "category_df_raw": category_df_raw,
        "category_df_smoothed": category_df_smoothed,
        "category_df_selected": category_df,
        "canonical_edges": canonical_edges,
        "tr_edges": tr_edges,
        "smoothing": smoothing_meta,
        "output_root": output_root,
        "canonical_root": canonical_root,
        "trimmed_df": trimmed_df,
        "max_lag_primary": max_lag_primary,
    }
