from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from MDE import MDE
from src import roi
import warnings

try:
    from nilearn import plotting as nilearn_plotting
except Exception:
    nilearn_plotting = None

try:
    from IPython.display import display
except Exception:
    display = None


def sanitize_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in name)


def make_lag_dict(series: pd.Series, max_lag: int) -> Dict[int, pd.Series]:
    return {lag: series.shift(lag).iloc[max_lag:].reset_index(drop=True) for lag in range(max_lag + 1)}


def make_splits(n_samples: int, *, train_frac: float, val_frac: float) -> Dict[str, Tuple[int, int]]:
    train_end = max(1, int(np.floor(n_samples * train_frac)))
    val_span = max(1, int(np.floor(n_samples * val_frac)))
    val_end = min(n_samples - 1, train_end + val_span)
    if val_end <= train_end:
        val_end = min(n_samples - 1, train_end + 1)
    test_end = n_samples
    if test_end <= val_end:
        test_end = min(n_samples, val_end + 1)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def parse_variable_name(name: str) -> Tuple[str, int]:
    if "_lag" not in name:
        return name, 0
    base, lag_str = name.rsplit("_lag", 1)
    try:
        lag = int(lag_str)
    except ValueError:
        lag = 0
    return base, lag


def collect_top_series(
    mde_output: pd.DataFrame,
    lag_store: Dict[str, Dict[int, pd.Series]],
    target_len: int,
    max_items: Optional[int] = None,
) -> List[Tuple[str, float, np.ndarray]]:
    if mde_output.empty:
        return []
    var_col = "variables" if "variables" in mde_output.columns else "variable"
    rho_candidates = [c for c in mde_output.columns if "rho" in c.lower()]
    rho_col = rho_candidates[0] if rho_candidates else None

    rows = mde_output
    if max_items is not None:
        rows = rows.iloc[:max_items]

    series_list: List[Tuple[str, float, np.ndarray]] = []
    for _, row in rows.iterrows():
        var_name = str(row.get(var_col, "")).strip()
        if not var_name:
            continue
        base, lag = parse_variable_name(var_name)
        series = lag_store.get(base, {}).get(lag)
        if series is None or len(series) != target_len:
            continue
        rho_val = float(row.get(rho_col, np.nan)) if rho_col else float("nan")
        series_list.append((var_name, rho_val, series.to_numpy(dtype=float)))
    return series_list


def zscore(array: np.ndarray) -> np.ndarray:
    mu = np.nanmean(array)
    sigma = np.nanstd(array)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return array - mu
    return (array - mu) / sigma


def clamp_span(span: Sequence[int], max_len: int) -> List[int]:
    start, end = span
    start = min(max(1, start), max_len)
    end = min(max(1, end), max_len)
    if end < start:
        end = start
    return [start, end]


def plot_time_series_and_ranking(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    time_axis: np.ndarray,
    target_series: np.ndarray,
    top_series: List[Tuple[str, float, np.ndarray]],
    target_name: str,
    top_n_plot: int,
    save_scatter: bool,
    plt_module=None,
) -> None:
    if plt_module is None or not top_series:
        return

    top_series = top_series[:top_n_plot]
    safe_target = sanitize_name(target_name)

    main_count = min(3, len(top_series))
    main_series = top_series[:main_count]
    extra_series = top_series[main_count:]

    cmap = plt_module.cm.get_cmap("tab10", max(3, len(top_series)))
    color_map: Dict[str, Any] = {}

    fig, ax = plt_module.subplots(figsize=(12, 5))
    ax.plot(
        time_axis,
        zscore(target_series),
        label=f"{target_name} (target)",
        color="black",
        linewidth=2.2,
    )

    for idx, (name, rho_val, series) in enumerate(main_series):
        color = cmap(idx)
        color_map[name] = color
        label = f"{name} (rho={rho_val:.3f})"
        ax.plot(
            time_axis,
            zscore(series),
            label=label,
            color=color,
            linewidth=1.8,
        )

    for name, rho_val, series in extra_series:
        color_map[name] = "#999999"
        ax.plot(
            time_axis,
            zscore(series),
            label=f"{name} (rho={rho_val:.3f})",
            color="#999999",
            linewidth=1.0,
            alpha=0.45,
        )

    ax.set_title(f"MDE – {subject} / {story}: Target vs ROI predictors")
    ax.set_xlabel("Trimmed index")
    ax.set_ylabel("Z-scored value")
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    overlay_path = output_dir / f"mde_{safe_target}_time_series.png"
    fig.savefig(overlay_path, dpi=200)
    plt_module.close(fig)

    if extra_series:
        rows = len(extra_series)
        fig_extra, axes = plt_module.subplots(
            rows,
            1,
            figsize=(12, max(2.4 * rows, 3.0)),
            sharex=True,
        )
        if rows == 1:
            axes = [axes]
        target_line = zscore(target_series)
        for ax_extra, (name, rho_val, series) in zip(axes, extra_series):
            ax_extra.plot(time_axis, target_line, color="black", linewidth=1.0, alpha=0.25)
            ax_extra.plot(
                time_axis,
                zscore(series),
                color="#606060",
                linewidth=1.4,
                label=f"{name} (rho={rho_val:.3f})",
            )
            ax_extra.set_ylabel("z")
            ax_extra.legend(loc="upper right", frameon=False, fontsize=9)
            ax_extra.grid(alpha=0.25)
        axes[-1].set_xlabel("Trimmed index")
        fig_extra.tight_layout()
        detail_path = output_dir / f"mde_{safe_target}_time_series_detail.png"
        fig_extra.savefig(detail_path, dpi=200)
        plt_module.close(fig_extra)

    names = [item[0] for item in top_series]
    rho_vals = [item[1] for item in top_series]
    colors = [color_map.get(name, "#999999") for name in names]

    fig, ax = plt_module.subplots(figsize=(8.5, 0.45 * len(names) + 2.5))
    ax.barh(range(len(names)), rho_vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("rho value")
    ax.set_title(f"MDE-selected ROI variables ({subject} / {story}, target={target_name})")
    for idx, val in enumerate(rho_vals):
        ax.text(val + 0.01, idx, f"{val:.3f}", va="center", fontsize=8)
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    ranking_path = output_dir / f"mde_{safe_target}_variable_ranking.png"
    fig.savefig(ranking_path, dpi=200)
    plt_module.close(fig)

    if save_scatter:
        cols = min(3, max(1, len(top_series)))
        rows = int(np.ceil(len(top_series) / cols))
        fig, axes = plt_module.subplots(
            rows,
            cols,
            figsize=(cols * 4.0, rows * 3.2),
            squeeze=False,
        )
        for idx, (name, rho_val, series) in enumerate(top_series):
            r, c = divmod(idx, cols)
            ax_scatter = axes[r][c]
            ax_scatter.scatter(
                series,
                target_series,
                s=14,
                alpha=0.7,
                edgecolor="none",
                color=color_map.get(name, "#4C72B0"),
            )
            title = name
            if not math.isnan(rho_val):
                title += f" (rho={rho_val:.3f})"
            ax_scatter.set_title(title, fontsize=9)
            ax_scatter.set_xlabel(name)
            ax_scatter.set_ylabel(target_name)
            ax_scatter.grid(alpha=0.2)
            ax_scatter.set_facecolor("#f7f7f7")
        for idx in range(len(top_series), rows * cols):
            r, c = divmod(idx, cols)
            fig.delaxes(axes[r][c])
        fig.tight_layout()
        scatter_path = output_dir / f"mde_{safe_target}_scatter.png"
        fig.savefig(scatter_path, dpi=200)
        plt_module.close(fig)


def plot_lag_heatmap(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    mde_output: pd.DataFrame,
    target_name: str,
    plt_module=None,
) -> None:
    if plt_module is None or mde_output.empty:
        return

    var_col = "variables" if "variables" in mde_output.columns else "variable"
    if var_col not in mde_output.columns:
        return
    rho_col = next((c for c in mde_output.columns if "rho" in c.lower()), None)

    records: List[Tuple[str, int, float]] = []
    for _, row in mde_output.iterrows():
        var_name = str(row.get(var_col, "")).strip()
        if not var_name:
            continue
        base, lag = parse_variable_name(var_name)
        rho_val = float(row.get(rho_col, np.nan)) if rho_col else np.nan
        if not np.isfinite(rho_val):
            continue
        records.append((base, lag, rho_val))

    if not records:
        return

    df = pd.DataFrame(records, columns=["roi", "lag", "rho"])
    pivot = df.pivot_table(index="roi", columns="lag", values="rho", aggfunc="max")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig_width = max(6.0, min(14.0, pivot.shape[1] * 0.6))
    fig_height = max(6.0, min(18.0, pivot.shape[0] * 0.3 + 2.0))
    fig, ax = plt_module.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ROI")
    ax.set_title(f"MDE ROI-lag importance ({subject}/{story}, target={target_name})")
    fig.colorbar(im, ax=ax, label="rho")
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_lag_heatmap.png", dpi=200)
    plt_module.close(fig)


def plot_cumulative_rho(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    rho_sequence: Sequence[float],
    target_name: str,
    plt_module=None,
    highlight_step: Optional[int] = None,
) -> None:
    if plt_module is None or not rho_sequence:
        return

    steps = np.arange(1, len(rho_sequence) + 1, dtype=int)
    rho_vals = np.array(rho_sequence, dtype=float)
    if rho_vals.size == 0:
        return
    cum_max = np.maximum.accumulate(rho_vals)

    fig, ax = plt_module.subplots(figsize=(8, 4))
    ax.plot(steps, rho_vals, marker="o", label="Step rho", linewidth=1.5)
    ax.plot(steps, cum_max, marker="s", linestyle="--", label="Cumulative max rho", linewidth=1.5)
    ax.set_xlabel("MDE step")
    ax.set_ylabel("rho value")
    ax.set_title(f"MDE rho progression ({subject}/{story}, target={target_name})")
    ax.set_xlim(1, steps[-1])
    if highlight_step is not None and steps.size and 1 <= highlight_step <= steps[-1]:
        ax.axvline(
            highlight_step,
            color="#d62728",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"Selected step ({highlight_step})",
        )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_rho_progression.png", dpi=200)
    plt_module.close(fig)


def plot_cae_progression(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    cae_sequence: Sequence[float],
    target_name: str,
    plt_module=None,
    highlight_step: Optional[int] = None,
) -> None:
    if plt_module is None or not cae_sequence:
        return

    cae_vals = np.array(cae_sequence, dtype=float)
    if cae_vals.size == 0 or not np.isfinite(cae_vals).any():
        return

    steps = np.arange(1, len(cae_vals) + 1, dtype=int)

    fig, ax = plt_module.subplots(figsize=(8, 4))
    ax.plot(steps, cae_vals, marker="o", linewidth=1.5, color="#ff7f0e", label="CAE")
    ax.set_xlabel("MDE step")
    ax.set_ylabel("CAE")
    ax.set_title(f"MDE CAE progression ({subject}/{story}, target={target_name})")
    ax.set_xlim(1, steps[-1])
    if highlight_step is not None and steps.size and 1 <= highlight_step <= steps[-1]:
        ax.axvline(
            highlight_step,
            color="#1f77b4",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"Selected step ({highlight_step})",
        )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_cae_progression.png", dpi=200)
    plt_module.close(fig)


def plot_residuals(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    time_axis: np.ndarray,
    target_series: np.ndarray,
    splits: Dict[str, Tuple[int, int]],
    mde_columns: Sequence[str],
    lag_store: Dict[str, Dict[int, pd.Series]],
    target_name: str,
    plt_module=None,
) -> Tuple[Dict[str, Dict[str, float]], Optional[np.ndarray]]:
    if plt_module is None or not mde_columns:
        return {"rmse": {}, "rho": {}, "cae": {}}, None

    series_list = []
    valid_columns = []
    for col_name in mde_columns:
        base, lag = parse_variable_name(col_name)
        series = lag_store.get(base, {}).get(lag)
        if series is None:
            continue
        series_list.append(series.to_numpy(dtype=float))
        valid_columns.append(col_name)

    if not series_list:
        return {"rmse": {}, "rho": {}, "cae": {}}, None

    X = np.column_stack(series_list)
    y = target_series

    train_start, train_end = splits["train"]
    if train_end <= train_start:
        return {"rmse": {}, "rho": {}, "cae": {}}, None

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    if X_train.size == 0:
        return {"rmse": {}, "rho": {}, "cae": {}}, None

    coefs, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred = X @ coefs
    residuals = y - y_pred

    fig, ax = plt_module.subplots(figsize=(12, 4.5))
    ax.plot(time_axis, residuals, color="#7B1FA2", linewidth=1.4, label="Residual (target - fit)")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    spans = [
        ("Train", splits["train"], "#4CAF50"),
        ("Validation", splits["val"], "#FFC107"),
        ("Test", splits["test"], "#F44336"),
    ]
    shown_labels = set()
    for label, (start_idx, end_idx), color in spans:
        if end_idx <= start_idx:
            continue
        start_time = time_axis[start_idx]
        end_time = time_axis[min(end_idx - 1, len(time_axis) - 1)]
        span_label = label if label not in shown_labels else None
        ax.axvspan(start_time, end_time, color=color, alpha=0.12, label=span_label)
        shown_labels.add(label)

    def seg_rmse(start_idx: int, end_idx: int) -> Optional[float]:
        if end_idx <= start_idx:
            return None
        seg = residuals[start_idx:end_idx]
        if seg.size == 0:
            return None
        return float(np.sqrt(np.nanmean(seg**2)))

    def seg_rho(start_idx: int, end_idx: int) -> Optional[float]:
        if end_idx <= start_idx:
            return None
        true_seg = target_series[start_idx:end_idx]
        pred_seg = y_pred[start_idx:end_idx]
        if true_seg.size < 2:
            return None
        if np.allclose(true_seg, true_seg[0]) or np.allclose(pred_seg, pred_seg[0]):
            return None
        if not np.isfinite(true_seg).all() or not np.isfinite(pred_seg).all():
            return None
        cov = np.corrcoef(true_seg, pred_seg)
        if cov.shape != (2, 2):
            return None
        rho_val = cov[0, 1]
        if not np.isfinite(rho_val):
            return None
        return float(rho_val)

    def seg_cae(start_idx: int, end_idx: int) -> Optional[float]:
        if end_idx <= start_idx:
            return None
        seg = residuals[start_idx:end_idx]
        if seg.size == 0:
            return None
        cae_val = float(np.nansum(np.abs(seg)))
        if not np.isfinite(cae_val):
            return None
        return cae_val

    rmse_dict: Dict[str, float] = {}
    rho_dict: Dict[str, float] = {}
    cae_dict: Dict[str, float] = {}
    rmse_lines = []
    for label, (start_idx, end_idx), _ in spans:
        rmse = seg_rmse(start_idx, end_idx)
        if rmse is None or not np.isfinite(rmse):
            continue
        rmse_dict[label.lower()] = rmse
        rmse_lines.append(f"{label}: {rmse:.3f}")
        rho_val = seg_rho(start_idx, end_idx)
        if rho_val is not None and np.isfinite(rho_val):
            rho_dict[label.lower()] = rho_val
        cae_val = seg_cae(start_idx, end_idx)
        if cae_val is not None and np.isfinite(cae_val):
            cae_dict[label.lower()] = cae_val
    if rmse_lines:
        ax.text(
            0.01,
            0.98,
            " | ".join(rmse_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xlabel("Trimmed index")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals vs time ({subject}/{story}, target={target_name})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_residuals.png", dpi=200)
    plt_module.close(fig)

    return {"rmse": rmse_dict, "rho": rho_dict, "cae": cae_dict}, y_pred


def plot_prediction_overlay(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    time_axis: np.ndarray,
    target_series: np.ndarray,
    prediction_series: np.ndarray,
    splits: Dict[str, Tuple[int, int]],
    target_name: str,
    rmse_info: Dict[str, float],
    plt_module=None,
) -> None:
    if plt_module is None:
        return

    fig, ax = plt_module.subplots(figsize=(12, 4.5))
    ax.plot(time_axis, target_series, label="Target", color="#1f77b4", linewidth=1.4)
    ax.plot(time_axis, prediction_series, label="Prediction", color="#d62728", linewidth=1.4, alpha=0.85)

    spans = [
        ("Train", splits.get("train"), "#4CAF50"),
        ("Validation", splits.get("val"), "#FFC107"),
        ("Test", splits.get("test"), "#F44336"),
    ]
    shown: set[str] = set()
    for label, span, color in spans:
        if span is None:
            continue
        start_idx, end_idx = span
        if end_idx <= start_idx:
            continue
        start_time = time_axis[start_idx]
        end_time = time_axis[min(end_idx - 1, len(time_axis) - 1)]
        ax.axvspan(start_time, end_time, color=color, alpha=0.10, label=label if label not in shown else None)
        shown.add(label)

    if rmse_info:
        rmse_lines = [f"{key}: {val:.3f}" for key, val in rmse_info.items() if np.isfinite(val)]
        if rmse_lines:
            ax.text(
                0.01,
                0.95,
                " | ".join(rmse_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

    ax.set_xlabel("Trimmed index")
    ax.set_ylabel("Value")
    ax.set_title(f"Target vs prediction ({subject}/{story}, target={target_name})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_prediction_overlay.png", dpi=200)
    plt_module.close(fig)


def plot_prediction_scatter(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    target_series: np.ndarray,
    prediction_series: np.ndarray,
    splits: Dict[str, Tuple[int, int]],
    target_name: str,
    plt_module=None,
) -> None:
    if plt_module is None:
        return

    colors = {"train": "#4CAF50", "val": "#FFC107", "test": "#F44336"}
    fig, ax = plt_module.subplots(figsize=(6, 6))

    for key, color in colors.items():
        span = splits.get(key)
        if span is None:
            continue
        start_idx, end_idx = span
        if end_idx <= start_idx:
            continue
        y_true = target_series[start_idx:end_idx]
        y_pred = prediction_series[start_idx:end_idx]
        if y_true.size == 0:
            continue
        ax.scatter(y_true, y_pred, label=key.capitalize(), alpha=0.65, s=24, color=color)

    both = np.concatenate([target_series, prediction_series])
    overall_min = float(np.nanmin(both))
    overall_max = float(np.nanmax(both))
    ax.plot([overall_min, overall_max], [overall_min, overall_max], color="#000000", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Observed target")
    ax.set_ylabel("Predicted target")
    ax.set_title(f"Prediction scatter ({subject}/{story}, target={target_name})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"mde_{sanitize_name(target_name)}_prediction_scatter.png", dpi=200)
    plt_module.close(fig)


def resolve_centroid_path(paths_cfg: Dict[str, str], n_parcels: int) -> Optional[Path]:
    root = paths_cfg.get("atlas_root") or "parcellations/Parcellations/MNI"
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = Path.cwd() / root_path
    centroid_dir = root_path / "Centroid_coordinates"
    candidates = [
        centroid_dir / f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
        centroid_dir / f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def save_candidate_roi_view(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    target_name: str,
    roi_variables: Sequence[str],
    paths_cfg: Dict[str, str],
    n_parcels: int,
    plotting_module=None,
) -> Optional[Path]:
    if plotting_module is None:
        plotting_module = nilearn_plotting
    if plotting_module is None:
        warnings.warn("Skipping candidate ROI HTML: nilearn is not available in the environment.")
        return None

    roi_ids = []
    seen = set()
    for var in roi_variables:
        parts = var.split("_")
        if len(parts) >= 2 and parts[0] == "roi" and parts[1].isdigit():
            roi_id = int(parts[1])
            if roi_id not in seen:
                seen.add(roi_id)
                roi_ids.append(roi_id)
    roi_ids.sort()
    if not roi_ids:
        return None

    centroid_path = resolve_centroid_path(paths_cfg, n_parcels)
    if centroid_path is None or not centroid_path.exists():
        warnings.warn(f"Skipping candidate ROI HTML: centroid file not found for n_parcels={n_parcels}.")
        return None

    cent = pd.read_csv(centroid_path)
    subset = cent[cent["ROI Label"].isin(roi_ids)].copy()
    if subset.empty:
        return None

    coords = subset[["R", "A", "S"]].to_numpy(dtype=float)
    labels = [f"ROI {roi_label}: {name}" for roi_label, name in zip(subset["ROI Label"], subset["ROI Name"])]
    colors = ["#ff7f0e"] * len(labels)
    sizes = [12] * len(labels)

    view = plotting_module.view_markers(coords, marker_color=colors, marker_size=sizes, marker_labels=labels)
    html_path = output_dir / f"mde_{sanitize_name(target_name)}_candidate_rois.html"
    view.save_as_html(str(html_path))
    if "display" in globals() and display is not None:
        try:
            display(view)
        except Exception:
            pass
    return html_path


def save_combined_roi_view(
    *,
    output_dir: Path,
    subject: str,
    story: str,
    summaries: Sequence[Dict[str, Any]],
    paths_cfg: Dict[str, str],
    n_parcels: int,
    plotting_module=None,
) -> Optional[Path]:
    if plotting_module is None:
        plotting_module = nilearn_plotting
    if plotting_module is None or display is None or not summaries:
        return None

    centroid_path = resolve_centroid_path(paths_cfg, n_parcels)
    if centroid_path is None or not centroid_path.exists():
        return None

    cent = pd.read_csv(centroid_path)
    coords_list: List[List[float]] = []
    colors: List[str] = []
    labels: List[str] = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, summary in enumerate(summaries):
        roi_vars = summary.get("roi_variables") or []
        roi_ids: List[int] = []
        seen: set[int] = set()
        for var in roi_vars:
            parts = str(var).split("_")
            if len(parts) >= 2 and parts[0] == "roi" and parts[1].isdigit():
                roi_id = int(parts[1])
                if roi_id not in seen:
                    seen.add(roi_id)
                    roi_ids.append(roi_id)
        if not roi_ids:
            continue
        subset = cent[cent["ROI Label"].isin(roi_ids)].copy()
        if subset.empty:
            continue
        color = palette[idx % len(palette)]
        category = summary.get("target_column", f"category_{idx}")
        for _, row in subset.iterrows():
            coords_list.append([float(row["R"]), float(row["A"]), float(row["S"])])
            colors.append(color)
            labels.append("{}: ROI {} {}".format(category, int(row["ROI Label"]), row["ROI Name"]))

    if not coords_list:
        return None

    coords = np.array(coords_list, dtype=float)
    marker_sizes = [12] * len(coords_list)
    view = plotting_module.view_markers(coords, marker_color=colors, marker_size=marker_sizes, marker_labels=labels)
    html_path = output_dir / "combined_category_rois.html"
    view.save_as_html(str(html_path))
    try:
        display(view)
    except Exception:
        pass
    return html_path


def run_mde_for_pair(
    subject: str,
    story: str,
    *,
    target_column: str,
    features_root: Path,
    figs_root: Path,
    paths_cfg: Dict[str, str],
    n_parcels: int,
    tau_grid: Sequence[int],
    E_cap: int,
    lib_sizes: Sequence[int],
    delta_default: int,
    theiler_min: int,
    train_frac: float,
    val_frac: float,
    top_n_plot: int,
    save_input_frame: bool,
    save_scatter: bool,
    sample_steps: int = 5,
    plt_module=None,
    use_cae: bool = False,
) -> Dict[str, Any]:
    subject = subject.strip()
    story = story.strip()

    category_dir = features_root / "subjects" / subject / story
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    full_path = category_dir / "category_timeseries.csv"
    trimmed_path = category_dir / "category_timeseries_trimmed.csv"
    if full_path.exists():
        category_df = pd.read_csv(full_path)
        source_name = full_path.name
    elif trimmed_path.exists():
        category_df = pd.read_csv(trimmed_path)
        source_name = trimmed_path.name
    else:
        raise FileNotFoundError(f"No category time series found under {category_dir}")

    category_df = category_df.copy()
    if "trim_index" not in category_df.columns:
        category_df.insert(0, "trim_index", np.arange(len(category_df), dtype=int))

    category_cols = [c for c in category_df.columns if c.startswith("cat_")]
    if not category_cols:
        raise ValueError("No category feature columns detected.")

    for col in category_cols:
        category_df[col] = pd.to_numeric(category_df[col], errors="coerce")

    if target_column not in category_cols:
        raise ValueError(f"{target_column!r} is not present in category dataframe.")

    roi_matrix = roi.load_schaefer_timeseries_TR(subject, story, n_parcels, paths_cfg)
    if roi_matrix.size == 0:
        raise ValueError(f"ROI time series empty for {subject}/{story}.")

    common_len = min(len(category_df), roi_matrix.shape[0])
    if common_len <= 2:
        raise ValueError(f"Insufficient samples ({common_len}) for {subject}/{story}.")

    category_df = category_df.iloc[:common_len].reset_index(drop=True)
    roi_matrix = roi_matrix[:common_len, :]

    target_series = pd.to_numeric(category_df[target_column], errors="coerce")
    target_series = target_series.ffill().bfill()
    if target_series.isna().any():
        raise ValueError(f"Target {target_column} contains NaNs after filling.")

    roi_cols = [f"roi_{idx}" for idx in range(roi_matrix.shape[1])]
    data_dict: Dict[str, np.ndarray] = {
        "Time": np.arange(1, common_len + 1, dtype=int),
        "target": target_series.astype(float).to_numpy(),
    }
    data_dict.update({col: roi_matrix[:, idx] for idx, col in enumerate(roi_cols)})
    base = pd.DataFrame(data_dict)

    max_tau = max(tau_grid)
    max_lag = max_tau * (E_cap - 1)
    if common_len <= max_lag + 5:
        raise ValueError(f"Not enough samples ({common_len}) for max lag {max_lag}.")

    lag_store: Dict[str, Dict[int, pd.Series]] = {"target": make_lag_dict(base["target"], max_lag)}
    for col in roi_cols:
        lag_store[col] = make_lag_dict(base[col], max_lag)

    time_trim = base["Time"].iloc[max_lag:].reset_index(drop=True)
    target_trim = lag_store["target"][0]
    if target_trim.empty:
        raise ValueError("Empty target after lag trimming.")

    splits = make_splits(len(time_trim), train_frac=train_frac, val_frac=val_frac)

    lagged_features: Dict[str, np.ndarray] = {}
    for col in roi_cols:
        for lag in range(max_lag + 1):
            series = lag_store[col].get(lag)
            if series is None:
                continue
            lagged_features[f"{col}_lag{lag}"] = series.to_numpy(dtype=float)

    if not lagged_features:
        raise ValueError("No ROI predictors available.")

    data_dict = {"target": target_trim.to_numpy(dtype=float)}
    data_dict.update(lagged_features)
    mde_df = pd.DataFrame(data_dict)

    N_trim = len(mde_df)
    lib_span = clamp_span([splits["train"][0] + 1, splits["train"][1]], N_trim)
    pred_span = clamp_span([splits["val"][0] + 1, splits["val"][1]], N_trim)

    p_lib_sizes = sorted({max(1, min(99, int(round(size / N_trim * 100)))) for size in lib_sizes if size < N_trim})
    if not p_lib_sizes:
        p_lib_sizes = [10, 25, 50, 75]

    sample_steps = max(1, int(sample_steps))

    mde = MDE(
        dataFrame=mde_df,
        target="target",
        removeColumns=["target"],
        D=E_cap,
        lib=lib_span,
        pred=pred_span,
        Tp=delta_default,
        tau=-1,
        exclusionRadius=theiler_min,
        sample=sample_steps,
        pLibSizes=p_lib_sizes,
        ccmSlope=0.0,
        crossMapRhoMin=0.0,
        embedDimRhoMin=0.0,
        cores=1,
        noTime=True,
        verbose=False,
        consoleOut=False,
    )
    mde.Run()

    mde_output = mde.MDEOut.copy()
    mde_output.insert(0, "step", range(1, len(mde_output) + 1))
    mde_output["cae"] = np.nan

    cae_array: Optional[np.ndarray] = None
    target_array = target_trim.to_numpy(dtype=float)
    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]

    if len(mde.MDEcolumns) > 0:
        if train_end <= train_start:
            if use_cae:
                warnings.warn("CAE selection requested, but training span is empty. Falling back to rho.")
        else:
            cae_values: List[float] = []
            for idx, _ in enumerate(mde.MDEcolumns):
                current_cols = list(mde.MDEcolumns[: idx + 1])
                try:
                    predictors = []
                    for col_name in current_cols:
                        base, lag = parse_variable_name(col_name)
                        series = lag_store.get(base, {}).get(lag)
                        if series is None:
                            raise KeyError(f"Missing lagged series for {col_name}")
                        arr = series.to_numpy(dtype=float)
                        if arr.shape[0] != target_array.shape[0]:
                            raise ValueError(f"Predictor length mismatch for {col_name}.")
                        predictors.append(arr)
                    if not predictors:
                        raise ValueError("No predictors available for CAE computation.")
                    X = np.column_stack(predictors)
                    X_train = X[train_start:train_end]
                    y_train = target_array[train_start:train_end]
                    if X_train.size == 0:
                        raise ValueError("Empty design matrix for CAE computation.")
                    if val_end <= val_start:
                        raise ValueError("Validation span is empty; cannot compute validation CAE.")

                    # Fit on TRAIN only
                    coefs, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                    y_pred_full = X @ coefs

                    # 🔑 CAE on VALIDATION ONLY
                    resid_val = target_array[val_start:val_end] - y_pred_full[val_start:val_end]
                    cae_val = float(np.nansum(np.abs(resid_val)))
                except (np.linalg.LinAlgError, ValueError, KeyError) as exc:
                    warnings.warn(f"Failed to compute CAE at step {idx + 1}: {exc}")
                    cae_val = float("nan")
                cae_values.append(cae_val)

            if cae_values:
                cae_array = np.asarray(cae_values, dtype=float)
                if len(mde_output) and cae_array.size > len(mde_output):
                    cae_array = cae_array[: len(mde_output)]
                mde_output.loc[mde_output.index[: len(cae_array)], "cae"] = cae_array

    rho_array = np.asarray(mde.MDErho, dtype=float)
    if len(mde_output) and rho_array.size > len(mde_output):
        rho_array = rho_array[: len(mde_output)]
    if rho_array.size:
        best_idx = int(np.nanargmax(rho_array))
    else:
        best_idx = len(mde_output) - 1
    best_idx = max(0, best_idx)
    best_cae = float("nan")
    selection_metric = "rho"

    if use_cae and cae_array is not None and np.isfinite(cae_array).any():
        new_idx = int(np.nanargmin(cae_array))
        best_idx = new_idx
        best_cae = float(cae_array[new_idx])
        selection_metric = "cae"
    elif cae_array is not None and 0 <= best_idx < len(cae_array):
        best_cae = float(cae_array[best_idx])

    best_output = mde_output.iloc[: best_idx + 1].copy()
    best_variables = list(mde.MDEcolumns[: best_idx + 1])
    best_rho = float(rho_array[best_idx]) if rho_array.size else float("nan")

    var_col = "variables" if "variables" in best_output.columns else "variable"
    roi_vars_from_output = [
        str(val) for val in best_output.get(var_col, pd.Series(dtype=str)).tolist() if isinstance(val, str) and val.startswith("roi_")
    ]

    output_dir = figs_root / subject / story / "day22_category_mde"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_target = sanitize_name(target_column)

    plots_dir = output_dir / "plots" / safe_target
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"mde_{safe_target}_selection.csv"
    mde_output.to_csv(output_csv, index=False)

    if save_input_frame:
        input_csv = output_dir / f"mde_{safe_target}_input_frame.csv"
        mde_df.to_csv(input_csv, index=False)

    summary_path = output_dir / f"mde_{safe_target}_summary.json"

    top_series = collect_top_series(best_output, lag_store, len(target_trim), max_items=top_n_plot)
    roi_variables = [var for var in best_variables if isinstance(var, str) and var.startswith("roi_")]
    plot_time_series_and_ranking(
        output_dir=plots_dir,
        subject=subject,
        story=story,
        time_axis=time_trim.to_numpy(dtype=float),
        target_series=target_trim.to_numpy(dtype=float),
        top_series=top_series,
        target_name=target_column,
        top_n_plot=top_n_plot,
        save_scatter=save_scatter,
        plt_module=plt_module,
    )
    plot_lag_heatmap(
        output_dir=plots_dir,
        subject=subject,
        story=story,
        mde_output=best_output,
        target_name=target_column,
        plt_module=plt_module,
    )
    full_rho_sequence = rho_array.tolist()
    plot_cumulative_rho(
        output_dir=plots_dir,
        subject=subject,
        story=story,
        rho_sequence=full_rho_sequence,
        target_name=target_column,
        plt_module=plt_module,
        highlight_step=(best_idx + 1) if full_rho_sequence else None,
    )
    if cae_array is not None and plt_module is not None:
        cae_sequence = cae_array.tolist()
        plot_cae_progression(
            output_dir=plots_dir,
            subject=subject,
            story=story,
            cae_sequence=cae_sequence,
            target_name=target_column,
            plt_module=plt_module,
            highlight_step=(best_idx + 1) if cae_sequence else None,
        )
    metrics_info, y_pred = plot_residuals(
        output_dir=plots_dir,
        subject=subject,
        story=story,
        time_axis=time_trim.to_numpy(dtype=float),
        target_series=target_trim.to_numpy(dtype=float),
        splits=splits,
        mde_columns=best_variables,
        lag_store=lag_store,
        target_name=target_column,
        plt_module=plt_module,
    )

    rmse_info_linear = metrics_info.get("rmse", {})
    rho_split_linear = metrics_info.get("rho", {})
    cae_split_linear = metrics_info.get("cae", {})

    # Prefer the span-specific linear diagnostics for reporting; fall back to the
    # overall best metric if the split metrics are missing/invalid.
    selection_rho_by_span = {
        key: val
        for key, val in (rho_split_linear or {}).items()
        if val is not None and np.isfinite(val)
    }
    if not selection_rho_by_span and np.isfinite(best_rho):
        selection_rho_by_span = {key: best_rho for key in ("train", "validation", "test")}

    selection_cae_by_span = {
        key: val
        for key, val in (cae_split_linear or {}).items()
        if val is not None and np.isfinite(val)
    }
    if not selection_cae_by_span and np.isfinite(best_cae):
        selection_cae_by_span = {key: best_cae for key in ("train", "validation", "test")}

    if y_pred is not None and plt_module is not None:
        plot_prediction_overlay(
            output_dir=plots_dir,
            subject=subject,
            story=story,
            time_axis=time_trim.to_numpy(dtype=float),
            target_series=target_trim.to_numpy(dtype=float),
            prediction_series=y_pred.astype(float),
            splits=splits,
            target_name=target_column,
            rmse_info=rmse_info_linear,
            plt_module=plt_module,
        )
        plot_prediction_scatter(
            output_dir=plots_dir,
            subject=subject,
            story=story,
            target_series=target_trim.to_numpy(dtype=float),
            prediction_series=y_pred.astype(float),
            splits=splits,
            target_name=target_column,
            plt_module=plt_module,
        )

    prediction_csv: Optional[Path] = None
    if y_pred is not None:
        start_sec_trim: Optional[np.ndarray] = None
        if "start_sec" in category_df.columns:
            start_sec_trim = category_df["start_sec"].iloc[max_lag:].to_numpy(dtype=float)
        prediction_df = pd.DataFrame(
            {
                "trim_index": np.arange(len(time_trim), dtype=int),
                "time": time_trim.to_numpy(dtype=float),
                "target": target_trim.to_numpy(dtype=float),
                "prediction": y_pred.astype(float),
            }
        )
        if start_sec_trim is not None and start_sec_trim.shape[0] == len(prediction_df):
            prediction_df.insert(1, "start_sec", start_sec_trim)
        prediction_csv = output_dir / f"mde_{safe_target}_best_prediction.csv"
        prediction_df.to_csv(prediction_csv, index=False)

    candidate_html = save_candidate_roi_view(
        output_dir=output_dir,
        subject=subject,
        story=story,
        target_name=target_column,
        roi_variables=roi_variables,
        paths_cfg=paths_cfg,
        n_parcels=n_parcels,
        plotting_module=nilearn_plotting,
    )

    summary = {
        "subject": subject,
        "story": story,
        "target_column": target_column,
        "source_file": source_name,
        "samples_trimmed": int(N_trim),
        "lib_span": lib_span,
        "pred_span": pred_span,
        "pLibSizes": p_lib_sizes,
        "best_step": int(best_idx + 1),
        "best_rho": best_rho,
        "best_cae": best_cae,
        "selection_metric": selection_metric,
        "top_variables": [item[0] for item in top_series],
        "selected_variables": best_variables,
        "roi_variables": roi_variables,
        "plots_dir": str(plots_dir),
        "candidate_roi_html": str(candidate_html) if candidate_html else None,
        "selection_csv": str(output_csv),
        "input_csv": str(input_csv) if save_input_frame else None,
        # Selection-metric view for rho/cae; RMSE remains linear (no MDE RMSE metric)
        "rmse": rmse_info_linear,
        "rho_by_span": selection_rho_by_span,
        "cae_by_span": selection_cae_by_span,
        # Linear diagnostics retained for transparency
        "rho_by_span_linear": rho_split_linear,
        "cae_by_span_linear": cae_split_linear,
        "prediction_csv": str(prediction_csv) if prediction_csv else None,
        "splits": {
            key: [int(span[0]), int(span[1])] if span is not None else None
            for key, span in splits.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary
