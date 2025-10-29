from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _prepare_outpath(outpath: str | Path) -> Path:
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def forecast_bars(skills: Dict[str, float], outpath: str) -> None:
    path = _prepare_outpath(outpath)
    labels = list(skills.keys())
    values = [skills[label] for label in labels]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color="#4c72b0")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Skill (ρ)")
    ax.set_ylim(min(0.0, min(values) - 0.05), max(values) + 0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ccm_curve(L: List[int], skill: List[float], outpath: str) -> None:
    path = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(L, skill, marker="o")
    if len(L) > 1:
        slope = (skill[-1] - skill[0]) / (L[-1] - L[0])
        ax.set_title(f"CCM convergence slope = {slope:.3f}")
    ax.set_xlabel("Library size (L)")
    ax.set_ylabel("Skill (ρ)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def theta_sweep(thetas: List[float], skills: List[float], outpath: str) -> None:
    path = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(thetas, skills, marker="o")
    if skills:
        best_idx = max(range(len(skills)), key=skills.__getitem__)
        ax.scatter([thetas[best_idx]], [skills[best_idx]], color="red", zorder=5)
        ax.set_title(f"Peak θ = {thetas[best_idx]:.2f}")
    ax.set_xlabel("θ")
    ax.set_ylabel("Skill (ρ)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attractor_3d(X: np.ndarray, outpath: str, color: np.ndarray | None = None) -> None:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("X must have shape (n_samples, >=3)")
    coords = arr[:, :3]
    path = _prepare_outpath(outpath)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    if color is not None:
        color_arr = np.asarray(color)
        if color_arr.shape[0] != coords.shape[0]:
            raise ValueError("color must match the number of samples")
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color_arr, cmap="viridis", s=6)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="index")
    else:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
