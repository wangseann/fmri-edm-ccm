from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .utils import set_seed


def _cache_file(paths: dict, sub: str, story: str, stem: str) -> Path:
    cache_root = Path(paths.get("cache", "data_cache"))
    return cache_root / sub / story / f"{stem}.npy"


def _load_array(paths: dict, sub: str, story: str, stem: str) -> np.ndarray:
    path = _cache_file(paths, sub, story, stem)
    if path.exists():
        return np.load(path)
    raise FileNotFoundError(f"Missing cached array: {path}")


def load_english1000_TR(sub: str, story: str, paths: dict) -> np.ndarray:
    data = _load_array(paths, sub, story, "english1000_TR")
    if data.shape[1] != 985:
        raise ValueError(f"Expected 985 features, found {data.shape[1]}")
    return data


def load_envelope_TR(sub: str, story: str, paths: dict) -> np.ndarray:
    return _load_array(paths, sub, story, "envelope_TR").astype(float).ravel()


def load_wordrate_TR(sub: str, story: str, paths: dict) -> np.ndarray:
    return _load_array(paths, sub, story, "wordrate_TR").astype(float).ravel()


def pca_fit_transform(X: np.ndarray, n_components: int, seed: int = 0) -> Tuple[np.ndarray, Any]:
    set_seed(seed)
    X = np.asarray(X, dtype=float)
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
    Z = pca.fit_transform(X)
    return Z, pca


def make_lag_stack(X: np.ndarray, E: int, tau: int) -> np.ndarray:
    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    max_lag = (E - 1) * tau
    if data.shape[0] <= max_lag:
        raise ValueError("Time series too short for requested embedding")
    indices = [slice(max_lag - j * tau, data.shape[0] - j * tau) for j in range(E)]
    lags = [data[idx] for idx in indices]
    return np.hstack(lags)


try:
    from .edm_ccm import English1000Loader
except Exception:  # pragma: no cover - optional dependency
    English1000Loader = None  # type: ignore


def embed_tokens(tokens: Sequence[str], loader: "English1000Loader", lowercase: bool = True) -> np.ndarray:
    if loader is None:
        raise ValueError("An English1000Loader instance is required for embedding tokens.")
    seq = [tok.lower() if lowercase else tok for tok in tokens if tok.strip()]
    embeds = [loader.lookup.get(tok, None) for tok in seq]
    embeds = [vec for vec in embeds if vec is not None]
    if not embeds:
        dim = int(getattr(loader, "embed_dim", 0))
        return np.zeros((0, dim), dtype=float)
    return np.vstack(embeds)


def embed_text(text: str, loader: "English1000Loader", lowercase: bool = True) -> np.ndarray:
    tokens = text.strip().split()
    return embed_tokens(tokens, loader=loader, lowercase=lowercase)
