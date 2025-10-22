"""Projection utilities for decoding in PCA space."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class PCProjector:
    """Helper to map embeddings into the PCA space used by the model."""

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        """Store PCA components and mean for projection."""
        self.components = np.asarray(components, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        if self.components.ndim != 2:
            raise ValueError("components must be 2-D")
        if self.mean.ndim != 1:
            raise ValueError("mean must be 1-D")
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError("components and mean dimensionality mismatch")

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def word_to_pc(self, word_embed: np.ndarray) -> np.ndarray:
        """Project a word embedding (or stack) into PCA space."""
        vec = np.asarray(word_embed, dtype=float)
        if vec.ndim == 1:
            centered = vec - self.mean
            return centered @ self.components.T
        if vec.ndim == 2:
            centered = vec - self.mean
            return centered @ self.components.T
        raise ValueError("word_embed must be 1-D or 2-D array")

    def aggregate_to_TR(
        self,
        pc_words: np.ndarray,
        word_times_s: Sequence[Tuple[float, float]],
        tr_edges_s: Sequence[float],
    ) -> np.ndarray:
        """Average word-level PCs into TR bins using overlap weighting."""
        if len(pc_words) != len(word_times_s):
            raise ValueError("pc_words and word_times_s must align")
        edges = np.asarray(tr_edges_s, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("tr_edges_s must be a 1-D array of edges")
        n_tr = edges.size - 1
        n_comp = pc_words.shape[1]
        agg = np.zeros((n_tr, n_comp), dtype=float)
        weights = np.zeros(n_tr, dtype=float)
        for vec, (start, end) in zip(pc_words, word_times_s):
            if end <= start:
                continue
            for tr_idx in range(n_tr):
                tr_start = edges[tr_idx]
                tr_end = edges[tr_idx + 1]
                overlap = min(end, tr_end) - max(start, tr_start)
                if overlap <= 0:
                    continue
                agg[tr_idx] += vec * overlap
                weights[tr_idx] += overlap
        mask = weights > 0
        if np.any(mask):
            agg[mask] /= weights[mask, None]
        if np.any(~mask):
            agg[~mask] = np.nan
        return agg
