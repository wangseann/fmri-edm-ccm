from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "zscore_per_column",
    "ensure_same_length",
    "save_csv",
    "load_yaml",
    "set_seed",
]


def zscore_per_column(X: np.ndarray) -> np.ndarray:
    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    mean = np.nanmean(data, axis=0, keepdims=True)
    std = np.nanstd(data, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def ensure_same_length(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    if not arrays:
        return tuple()
    arrays_np = [np.asarray(arr) for arr in arrays]
    min_len = min(arr.shape[0] for arr in arrays_np)
    return tuple(arr[:min_len] for arr in arrays_np)


def save_csv(arr: np.ndarray | pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, pd.DataFrame):
        arr.to_csv(target, index=False)
    else:
        np.savetxt(target, np.asarray(arr), delimiter=",")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
