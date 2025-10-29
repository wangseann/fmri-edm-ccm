from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .decoding import load_transcript_words
from .edm_ccm import English1000Loader

EPS = 1e-12

# Helper functions


def load_story_words(paths: Dict, subject: str, story: str) -> List[Tuple[str, float, float]]:
    events = load_transcript_words(paths, subject, story)
    if not events:
        raise ValueError(f"No transcript events found for {subject} {story}.")
    return [(str(word).strip(), float(start), float(end)) for word, start, end in events]


def load_clusters_from_csv(csv_path: str) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    from pathlib import Path

    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(f"Cluster CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    for needed in ("category", "word"):
        assert needed in cols, f"CSV must contain '{needed}' column."
    cat_col = cols["category"]
    word_col = cols["word"]
    weight_col = cols.get("weight")
    if weight_col is None:
        df["_weight"] = 1.0
        weight_col = "_weight"
    df = df[[cat_col, word_col, weight_col]].copy()
    df[word_col] = df[word_col].astype(str).str.strip().str.lower()
    df[cat_col] = df[cat_col].astype(str).str.strip().str.lower()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    clusters: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for cat, sub in df.groupby(cat_col):
        bucket: Dict[str, float] = {}
        for w, wt in zip(sub[word_col].tolist(), sub[weight_col].tolist()):
            if not w:
                continue
            bucket[w] = float(wt)
        pairs = sorted(bucket.items())
        if pairs:
            clusters[cat] = {"words": pairs}
    if not clusters:
        raise ValueError("No clusters parsed from CSV.")
    return clusters


def build_states_from_csv(
    clusters: Dict[str, Dict[str, List[Tuple[str, float]]]], primary_lookup: Dict[str, np.ndarray], fallback=None, weight_power: float = 1.0
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    category_states: Dict[str, Dict] = {}
    category_definitions: Dict[str, Dict] = {}
    oov_counts: Dict[str, int] = {}
    for cat, spec in clusters.items():
        pairs = spec.get("words", [])
        vecs: List[np.ndarray] = []
        weights: List[float] = []
        found_words: List[str] = []
        missing_words: List[str] = []
        for word, wt in pairs:
            vec = lookup_embedding(word, primary_lookup, fallback)
            if vec is None:
                missing_words.append(word)
                continue
            vecs.append(vec.astype(float))
            weights.append(float(max(0.0, wt)) ** float(weight_power))
            found_words.append(word)
        if not vecs:
            warnings.warn(f"[{cat}] no usable representative embeddings; prototype will be None.")
            prototype = None
            prototype_norm = None
        else:
            W = np.array(weights, dtype=float)
            W = W / (W.sum() + 1e-12)
            M = np.stack(vecs, axis=0)
            prototype = (W[:, None] * M).sum(axis=0)
            prototype_norm = float(np.linalg.norm(prototype))
            if prototype_norm < EPS:
                prototype = None
                prototype_norm = None
        rep_lex = {word: float(wt) for word, wt in pairs}
        category_states[cat] = {
            "name": cat,
            "seeds": [],
            "found_seeds": found_words,
            "missing_seeds": missing_words,
            "prototype": prototype,
            "prototype_norm": prototype_norm,
            "lexicon": rep_lex,
            "expanded_count": 0,
            "expansion_params": {"enabled": False, "top_k": 0, "min_sim": 0.0},
        }
        category_definitions[cat] = {
            "from": "csv",
            "seeds": [],
            "found_seeds": found_words,
            "missing_seeds": missing_words,
            "prototype_dim": int(prototype.shape[0]) if isinstance(prototype, np.ndarray) else 0,
            "prototype_norm": prototype_norm,
            "representative_words": rep_lex,
            "lexicon": rep_lex,
            "expanded_neighbors": {},
        }
        oov_counts[cat] = len(missing_words)
    if any(oov_counts.values()):
        warnings.warn(f"OOV representative words: {oov_counts}")
    return category_states, category_definitions


def build_tr_edges(word_events: Sequence[Tuple[str, float, float]], tr_s: float) -> np.ndarray:
    if not word_events:
        return np.arange(0, tr_s, tr_s)
    max_end = max(end for _, _, end in word_events)
    n_tr = max(1, int(math.ceil(max_end / tr_s)))
    edges = np.arange(0.0, (n_tr + 1) * tr_s, tr_s, dtype=float)
    if edges[-1] < max_end:
        edges = np.append(edges, edges[-1] + tr_s)
    if edges[-1] < max_end - 1e-9:
        edges = np.append(edges, edges[-1] + tr_s)
    return edges


class FallbackEmbeddingAdapter:
    """Wrap a fallback embedding model with optional linear transform into English1000 space."""

    def __init__(
        self,
        model,
        *,
        rotation: Optional[np.ndarray] = None,
        fallback_mean: Optional[np.ndarray] = None,
        english_mean: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.rotation = rotation
        self.fallback_mean = fallback_mean
        self.english_mean = english_mean

    def __contains__(self, key: str) -> bool:
        try:
            if hasattr(self.model, "__contains__"):
                return key in self.model
            if hasattr(self.model, "key_to_index"):
                return key in self.model.key_to_index
        except Exception:
            return False
        return False

    def get_vector(self, key: str) -> np.ndarray:
        vec = None
        if hasattr(self.model, "get_vector"):
            vec = self.model.get_vector(key)
        elif hasattr(self.model, "__getitem__"):
            vec = self.model[key]
        if vec is None:
            raise KeyError(key)
        vec = np.asarray(vec, dtype=float)
        if self.rotation is not None and self.fallback_mean is not None and self.english_mean is not None:
            vec = np.asarray(vec, dtype=float)
            transformed = (vec - self.fallback_mean) @ self.rotation
            return transformed + self.english_mean
        return vec


def lookup_embedding(
    token: str, primary_lookup: Dict[str, np.ndarray], fallback: Optional[Union[FallbackEmbeddingAdapter, Any]] = None
) -> Optional[np.ndarray]:
    key = token.lower().strip()
    if not key:
        return None
    vec = primary_lookup.get(key) if primary_lookup else None
    if vec is not None:
        return np.asarray(vec, dtype=float)
    if fallback is not None:
        try:
            if hasattr(fallback, "get_vector") and key in fallback:
                return np.asarray(fallback.get_vector(key), dtype=float)
            if hasattr(fallback, "__contains__") and key in fallback:
                return np.asarray(fallback[key], dtype=float)
        except Exception:
            return None
    return None


def make_category_prototype(
    seeds: Sequence[str], primary_lookup: Dict[str, np.ndarray], fallback=None, allow_single: bool = False
) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    found_vectors = []
    found_words = []
    missing_words = []
    for seed in seeds:
        vec = lookup_embedding(seed, primary_lookup, fallback)
        if vec is None:
            missing_words.append(seed)
            continue
        found_vectors.append(vec)
        found_words.append(seed)
    if not found_vectors:
        return None, found_words, missing_words
    if len(found_vectors) < 2 and not allow_single:
        warnings.warn(f"Only {len(found_vectors)} usable seed(s); enable allow_single_seed to accept singleton prototypes.")
        if not allow_single:
            return None, found_words, missing_words
    prototype = np.mean(found_vectors, axis=0)
    return prototype, found_words, missing_words


def expand_category(prototype: np.ndarray, vocab_embeddings: np.ndarray, vocab_words: Sequence[str], top_k: int, min_sim: float) -> Dict[str, float]:
    if prototype is None or vocab_embeddings is None or vocab_words is None:
        return {}
    proto = np.asarray(prototype, dtype=float)
    proto_norm = np.linalg.norm(proto)
    if proto_norm == 0:
        return {}
    proto_unit = proto / proto_norm
    vocab_norms = np.linalg.norm(vocab_embeddings, axis=1)
    valid_mask = vocab_norms > 0
    sims = np.full(vocab_embeddings.shape[0], -1.0, dtype=float)
    sims[valid_mask] = (vocab_embeddings[valid_mask] @ proto_unit) / vocab_norms[valid_mask]
    top_k_eff = min(top_k, len(sims))
    if top_k_eff <= 0:
        return {}
    candidate_idx = np.argpartition(-sims, top_k_eff - 1)[:top_k_eff]
    out = {}
    for idx in candidate_idx:
        score = float(sims[idx])
        if score < min_sim:
            continue
        out[vocab_words[idx]] = score
    return out


def tr_token_overlap(token_start: float, token_end: float, tr_start: float, tr_end: float, mode: str = "proportional") -> float:
    token_start = float(token_start)
    token_end = float(token_end)
    if token_end <= token_start:
        token_end = token_start + 1e-3
    if mode == "midpoint":
        midpoint = 0.5 * (token_start + token_end)
        return 1.0 if tr_start <= midpoint < tr_end else 0.0
    overlap = max(0.0, min(token_end, tr_end) - max(token_start, tr_start))
    duration = token_end - token_start
    if duration <= 0:
        return 1.0 if overlap > 0 else 0.0
    return max(0.0, min(1.0, overlap / duration))


def score_tr(
    token_payload: Sequence[Dict],
    method: str,
    *,
    lexicon: Optional[Dict[str, float]] = None,
    prototype: Optional[np.ndarray] = None,
    prototype_norm: Optional[float] = None,
) -> float:
    if not token_payload:
        return float("nan")
    method = method.lower()
    if method == "count":
        if not lexicon:
            return float("nan")
        total = 0.0
        for item in token_payload:
            weight = lexicon.get(item["word"].lower())
            if weight is None:
                continue
            total += weight * item["overlap"]
        return float(total)
    if method == "similarity":
        if prototype is None or prototype_norm is None or prototype_norm < EPS:
            return float("nan")
        num = 0.0
        denom = 0.0
        for item in token_payload:
            emb = item.get("embedding")
            if emb is None:
                continue
            emb_norm = item.get("embedding_norm")
            if emb_norm is None or emb_norm < EPS:
                continue
            sim = float(np.dot(emb, prototype) / (emb_norm * prototype_norm))
            num += sim * item["overlap"]
            denom += item["overlap"]
        if denom == 0:
            return float("nan")
        value = num / denom
        return float(np.clip(value, -1.0, 1.0))
    raise ValueError(f"Unknown scoring method: {method}")


def ensure_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_serializable(v) for v in obj]
    return obj


def build_token_buckets(edges: np.ndarray, event_records: Sequence[Dict], mode: str = "proportional") -> List[List[Dict]]:
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
            buckets[idx].append(
                {
                    "word": rec["word"],
                    "overlap": overlap,
                    "embedding": rec["embedding"],
                    "embedding_norm": rec["embedding_norm"],
                    "token_start": rec["start"],
                    "token_end": rec["end"],
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                }
            )
    return buckets


def score_time_series(
    edges: np.ndarray,
    buckets: Sequence[Sequence[Dict]],
    category_states: Dict[str, Dict],
    category_names: Sequence[str],
    category_columns: Sequence[str],
    method: str,
    index_name: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    n_bins = len(buckets)
    score_matrix = np.full((n_bins, len(category_names)), np.nan, dtype=float)
    for col_idx, cat_name in enumerate(category_names):
        state = category_states[cat_name]
        lexicon = state.get("lexicon")
        prototype = state.get("prototype")
        prototype_norm = state.get("prototype_norm")
        for bin_idx, bucket in enumerate(buckets):
            score_matrix[bin_idx, col_idx] = score_tr(bucket, method, lexicon=lexicon, prototype=prototype, prototype_norm=prototype_norm)
    data = {
        index_name: np.arange(n_bins, dtype=int),
        "start_sec": edges[:-1],
        "end_sec": edges[1:],
    }
    for col_idx, col in enumerate(category_columns):
        data[col] = score_matrix[:, col_idx]
    df = pd.DataFrame(data)
    return df, score_matrix


def build_smoothing_kernel(
    seconds_bin_width: float, smoothing_seconds: float, *, method: str = "moving_average", gaussian_sigma_seconds: Optional[float] = None
) -> np.ndarray:
    if smoothing_seconds <= 0:
        return np.array([1.0], dtype=float)
    method = str(method or "moving_average").lower()
    if method == "moving_average":
        window_samples = max(1, int(round(smoothing_seconds / seconds_bin_width)))
        if window_samples % 2 == 0:
            window_samples += 1
        kernel = np.ones(window_samples, dtype=float)
    elif method == "gaussian":
        sigma_seconds = float(gaussian_sigma_seconds) if gaussian_sigma_seconds not in (None, "") else max(smoothing_seconds / 2.0, seconds_bin_width)
        sigma_samples = max(sigma_seconds / seconds_bin_width, 1e-6)
        half_width = max(1, int(round(3.0 * sigma_samples)))
        grid = np.arange(-half_width, half_width + 1, dtype=float)
        kernel = np.exp(-0.5 * (grid / sigma_samples) ** 2)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def apply_smoothing_kernel(values: np.ndarray, kernel: np.ndarray, *, pad_mode: str = "edge", eps: float = 1e-8) -> np.ndarray:
    if values.size == 0 or kernel.size <= 1:
        return values.copy()
    pad_mode = pad_mode if pad_mode in {"edge", "reflect"} else "edge"
    half = kernel.size // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode=pad_mode)
    mask = np.isfinite(padded).astype(float)
    filled = np.where(mask, padded, 0.0)
    smoothed = np.empty((values.shape[0], values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        numerator = np.convolve(filled[:, col], kernel, mode="valid")
        denominator = np.convolve(mask[:, col], kernel, mode="valid")
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed_col = numerator / np.maximum(denominator, eps)
        smoothed_col[denominator < eps] = np.nan
        smoothed[:, col] = smoothed_col
    return smoothed


def aggregate_seconds_to_edges(canonical_edges: np.ndarray, canonical_values: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    if canonical_values.size == 0:
        return np.empty((len(target_edges) - 1, 0), dtype=float)
    midpoints = 0.5 * (canonical_edges[:-1] + canonical_edges[1:])
    bin_ids = np.digitize(midpoints, target_edges) - 1
    if bin_ids.size:
        bin_ids = np.clip(bin_ids, 0, len(target_edges) - 2)
    out = np.full((len(target_edges) - 1, canonical_values.shape[1]), np.nan, dtype=float)
    for idx in range(out.shape[0]):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        values = canonical_values[mask]
        if values.ndim == 1:
            values = values[:, None]
        finite_any = np.isfinite(values).any(axis=0)
        if not finite_any.any():
            continue
        col_means = np.full(values.shape[1], np.nan, dtype=float)
        col_means[finite_any] = np.nanmean(values[:, finite_any], axis=0)
        out[idx] = col_means
    return out


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
    save_outputs: bool = True,
) -> Dict[str, Any]:
    if not subject or not story:
        raise ValueError("Subject and story must be provided.")
    print(f"=== Day19 category build for {subject} / {story} ===")

    categories_cfg = json.loads(json.dumps(categories_cfg_base or {}))
    fallback_cfg = {}
    if isinstance(cfg_base, dict):
        fallback_cfg = cfg_base.get("fallback") or {}
    if not fallback_cfg and isinstance(categories_cfg, dict):
        fallback_cfg = categories_cfg.get("fallback") or {}
    if fallback_cfg and fallback_cfg.get("enabled", False):
        model_path = fallback_cfg.get("model_path", "")
        if model_path and not categories_cfg.get("word2vec_path"):
            categories_cfg["word2vec_path"] = model_path
        if categories_cfg.get("embedding_source", "english1000").lower() == "english1000":
            categories_cfg["embedding_source"] = "both"
        transform_path = fallback_cfg.get("transform_path")
        if transform_path and not categories_cfg.get("fallback_transform_path"):
            categories_cfg["fallback_transform_path"] = transform_path
        label = fallback_cfg.get("label")
        if label and not categories_cfg.get("word2vec_label"):
            categories_cfg["word2vec_label"] = label
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

    selected_set_spec = category_sets.get(category_set_name, {}) if category_sets else {}

    output_root = features_root / "subjects" / subject / story
    canonical_root = features_root / "stories" / story
    if save_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        canonical_root.mkdir(parents=True, exist_ok=True)

    story_events = load_story_words(paths, subject, story)
    print(f"Loaded {len(story_events)} transcript events.")
    tr_edges = build_tr_edges(story_events, TR)
    n_tr = len(tr_edges) - 1
    print(f"TR edges: {len(tr_edges)} (n_tr={n_tr}) spanning {tr_edges[-1]:.2f} seconds.")

    embedding_source = str(categories_cfg.get("embedding_source", "english1000")).lower()
    english_loader = None
    english_lookup: Dict[str, np.ndarray] = {}
    english_vocab: List[str] = []
    english_matrix = None
    if embedding_source in {"english1000", "both"}:
        english1000_path = Path(paths.get("data_root", "")) / "derivative" / "english1000sm.hf5"
        if english1000_path.exists():
            english_loader = English1000Loader(english1000_path)
            english_lookup = english_loader.lookup
            english_vocab = english_loader.vocab
            english_matrix = english_loader.embeddings
            print(f"Loaded English1000 embeddings from {english1000_path} (vocab={len(english_vocab)}).")
        else:
            raise FileNotFoundError(f"English1000 embeddings not found at {english1000_path}")
    else:
        print("English1000 disabled by configuration.")

    word2vec_model = None
    fallback_transform: Optional[Dict[str, np.ndarray]] = None
    if embedding_source in {"word2vec", "both"}:
        w2v_path = categories_cfg.get("word2vec_path")
        if w2v_path:
            w2v_path = Path(w2v_path)
            if w2v_path.exists():
                try:
                    from gensim.models import KeyedVectors

                    binary = w2v_path.suffix.lower() in {".bin", ".gz"}
                    word2vec_raw = KeyedVectors.load_word2vec_format(w2v_path, binary=binary)
                    print(f"Loaded Word2Vec fallback from {w2v_path}.")
                    transform_path_cfg = categories_cfg.get("fallback_transform_path")
                    if transform_path_cfg:
                        transform_path = Path(transform_path_cfg)
                        if transform_path.exists():
                            data = np.load(transform_path)
                            rotation = data.get("rotation")
                            fallback_mean = data.get("fallback_mean")
                            english_mean = data.get("english_mean")
                            if rotation is None or fallback_mean is None or english_mean is None:
                                warnings.warn(f"Fallback transform file {transform_path} missing required arrays; using raw vectors.")
                                word2vec_model = word2vec_raw
                            else:
                                rotation = np.asarray(rotation, dtype=float)
                                fallback_mean = np.asarray(fallback_mean, dtype=float)
                                english_mean = np.asarray(english_mean, dtype=float)
                                if rotation.shape[0] != word2vec_raw.vector_size:
                                    warnings.warn(
                                        f"Fallback rotation shape {rotation.shape} incompatible with vector size {word2vec_raw.vector_size}; using raw vectors."
                                    )
                                    word2vec_model = word2vec_raw
                                else:
                                    word2vec_model = FallbackEmbeddingAdapter(
                                        word2vec_raw,
                                        rotation=rotation,
                                        fallback_mean=fallback_mean,
                                        english_mean=english_mean,
                                    )
                                    print(f"Applied fallback transform from {transform_path}.")
                        else:
                            warnings.warn(f"Fallback transform path {transform_path_cfg} does not exist; using raw vectors.")
                            word2vec_model = word2vec_raw
                    else:
                        word2vec_model = word2vec_raw
                except Exception as exc:
                    warnings.warn(f"Failed to load Word2Vec fallback: {exc}")
            else:
                warnings.warn(f"Word2Vec path does not exist: {w2v_path}")
        else:
            warnings.warn("Word2Vec fallback requested but no path provided.")
    else:
        print("Word2Vec fallback disabled.")

    if cluster_csv_path:
        csv_clusters = load_clusters_from_csv(cluster_csv_path)
        category_states, category_definitions = build_states_from_csv(
            csv_clusters,
            english_lookup,
            word2vec_model,
            weight_power=prototype_weight_power,
        )
        category_names = sorted(category_states.keys())
        category_columns = [f"cat_{name}" for name in category_names]
        print(f"Loaded {len(category_names)} CSV-driven categories from {cluster_csv_path}: {category_names}")
        zero_norm = [k for k, v in category_states.items() if v.get("prototype") is not None and (v.get("prototype_norm") or 0.0) < EPS]
        if zero_norm:
            warnings.warn(f"Zero-norm prototypes (check OOV/weights): {zero_norm}")
    else:
        category_states = {}
        category_definitions = {}
        seed_oov_counter = Counter()
        for cat_name, cat_spec in selected_set_spec.items():
            seeds = cat_spec.get("seeds", [])
            explicit_words = cat_spec.get("words", [])
            prototype = None
            found_seeds: List[str] = []
            missing_seeds: List[str] = []
            if seeds:
                prototype, found_seeds, missing_seeds = make_category_prototype(seeds, english_lookup, word2vec_model, allow_single)
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
                if exp_enabled and english_matrix is not None:
                    expanded_words = expand_category(prototype, english_matrix, english_vocab, exp_top_k, exp_min_sim)
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
        print("Category seeds missing counts:", dict(seed_oov_counter))
        category_names = sorted(category_states.keys())
        category_columns = [f"cat_{name}" for name in category_names]
        print(f"Prepared {len(category_names)} categories: {category_names}")

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
    event_records: List[Dict] = []
    tokens_with_embeddings = 0
    for word, onset, offset in story_events:
        token = word.strip()
        if not token:
            continue
        key = token.lower()
        if key not in embedding_cache:
            embedding_cache[key] = lookup_embedding(token, english_lookup, word2vec_model)
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
    )
    smoothing_applied = smoothing_kernel.size > 1
    if canonical_values_raw.size and smoothing_applied:
        canonical_values_smoothed = apply_smoothing_kernel(canonical_values_raw, smoothing_kernel, pad_mode=smoothing_pad)
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
            json.dump(ensure_serializable(category_definitions), fh, indent=2)
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
            json.dump(ensure_serializable(category_definitions), fh, indent=2)
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
