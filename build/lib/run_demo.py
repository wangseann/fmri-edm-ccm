from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import baselines, ccm, edm, features, plots, roi
from .utils import load_yaml, save_csv, set_seed, zscore_per_column


def _align_forecasting(cfg: dict, R: np.ndarray, Z: np.ndarray, env: np.ndarray, wr: np.ndarray, shortlist: list[int]):
    def _residualize(signal: np.ndarray, drivers: np.ndarray) -> np.ndarray:
        if drivers.size == 0:
            return signal - np.mean(signal)
        model = LinearRegression(fit_intercept=True)
        model.fit(drivers, signal)
        return signal - model.predict(drivers)

    E = cfg.get("E_mult", cfg["E_grid"][1])
    tau = cfg["tau"]
    delta = cfg["delta"][0]
    k = cfg.get("simplex_k", cfg["k_grid"][1])
    theta = cfg.get("smap_theta", cfg["theta_grid"][2])
    lag_embed = (E - 1) * tau
    baseline_E, baseline_tau = 3, 1
    lag_drivers = (baseline_E - 1) * baseline_tau
    theiler = max(cfg["theiler_min"], E)

    y_future = edm.horizon_shift(Z[:, 0], delta)
    t_start = max(lag_embed, lag_drivers)
    n_samples = y_future.shape[0] - t_start
    if n_samples <= 0:
        raise ValueError("Not enough samples for requested horizon")

    X_rois_full = edm.embed_multivariate(R[:, shortlist], E, tau)
    X_target_full = edm.embed_multivariate(Z[:, 0], E, tau)
    start_emb = t_start - lag_embed
    X_rois = zscore_per_column(X_rois_full[start_emb : start_emb + n_samples])
    X_target = zscore_per_column(X_target_full[start_emb : start_emb + n_samples])
    Xemb = np.hstack([X_target, X_rois])
    y_target = y_future[t_start:]

    drivers = np.column_stack([env, wr])
    Xb_full = features.make_lag_stack(drivers, E=baseline_E, tau=baseline_tau)
    start_base = t_start - lag_drivers
    Xb = Xb_full[start_base : start_base + n_samples]

    yhat_simplex = edm.simplex(Xemb, y_target, k, theiler=theiler)
    yhat_smap = edm.smap(Xemb, y_target, k, theta, theiler=theiler)
    yhat_baseline = baselines.ridge_forecast(Xb, y_target)

    rho_simplex = edm.corr_skill(yhat_simplex, y_target)
    rho_smap = edm.corr_skill(yhat_smap, y_target)
    rho_baseline = edm.corr_skill(yhat_baseline, y_target)

    drivers_trim = drivers[: y_target.shape[0]]
    y_target_resid = _residualize(y_target, drivers_trim)

    theta_scores = [edm.corr_skill(edm.smap(Xemb, y_target, k, th, theiler=theiler), y_target) for th in cfg["theta_grid"]]
    theta_pref = "smap" if max(theta_scores) > rho_simplex else "simplex"

    return {
        "delta": delta,
        "Xemb": Xemb,
        "y_target": y_target,
        "y_target_resid": y_target_resid,
        "simplex": yhat_simplex,
        "smap": yhat_smap,
        "baseline": yhat_baseline,
        "rho_simplex": rho_simplex,
        "rho_smap": rho_smap,
        "rho_baseline": rho_baseline,
        "theta_scores": theta_scores,
        "theta_pref": theta_pref,
        "n_samples": int(n_samples),
        "lag_embed": lag_embed,
        "lag_drivers": lag_drivers,
        "k": k,
        "theta": theta,
        "theiler": theiler,
        "t_start": int(t_start),
    }


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(0)
    sub, story = cfg["subject"], cfg["story"]
    paths = cfg["paths"]

    X = features.load_english1000_TR(sub, story, paths)
    Z, _ = features.pca_fit_transform(X, cfg["pca_components"])
    env = features.load_envelope_TR(sub, story, paths)
    wr = features.load_wordrate_TR(sub, story, paths)
    R = roi.load_schaefer_timeseries_TR(sub, story, cfg["n_parcels"], paths)

    shortlist = ccm.ccm_conditional_screen(
        R,
        Z[:, 0],
        [env, wr],
        cfg.get("E_univ", cfg["E_grid"][0]),
        cfg["tau"],
        max(cfg["theiler_min"], cfg.get("E_univ", cfg["E_grid"][0])),
        cfg["lib_sizes"],
    )[: cfg["shortlist_topk"]]

    aligned = _align_forecasting(cfg, R, Z, env, wr, shortlist)

    out_root = Path(paths["results"]) / sub / story
    fig_root = Path(paths["figs"]) / sub / story
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    save_csv(np.array(shortlist), out_root / "shortlist.csv")
    forecast_df = pd.DataFrame(
        {
            "target": aligned["y_target"],
            "drivers_only": aligned["baseline"],
            "simplex": aligned["simplex"],
            "smap": aligned["smap"],
        }
    )
    forecast_df.to_csv(out_root / "forecast.csv", index=False)

    skills = {
        "drivers-only": aligned["rho_baseline"],
        "simplex": aligned["rho_simplex"],
        "smap": aligned["rho_smap"],
    }
    pd.Series(skills).to_csv(out_root / "skills.csv", header=False)

    plots.forecast_bars(skills, str(fig_root / "forecast_bars.png"))

    roi_idx = int(shortlist[0]) if shortlist else 0
    delta = cfg["delta"][0]
    ccm_theiler = max(cfg["theiler_min"], cfg.get("E_univ", cfg["E_grid"][0]))
    target_shift = edm.horizon_shift(Z[:, 0], delta)
    roi_series = R[:, roi_idx]
    if delta > 0:
        roi_series = roi_series[:-delta]
        target_shift = target_shift
    drivers_ccm = np.column_stack([env[: roi_series.shape[0]], wr[: roi_series.shape[0]]])
    if drivers_ccm.size:
        model = LinearRegression(fit_intercept=True)
        model.fit(drivers_ccm, roi_series)
        roi_resid = roi_series - model.predict(drivers_ccm)
        model.fit(drivers_ccm, target_shift)
        target_resid = target_shift - model.predict(drivers_ccm)
    else:
        roi_resid = roi_series - roi_series.mean()
        target_resid = target_shift - target_shift.mean()
    ccm_res = ccm.ccm_pair(
        roi_resid,
        target_resid,
        cfg.get("E_univ", cfg["E_grid"][0]),
        cfg["tau"],
        ccm_theiler,
        cfg["lib_sizes"],
    )
    plots.ccm_curve(cfg["lib_sizes"], ccm_res["skill_curve"], str(fig_root / "ccm_curve.png"))
    plots.theta_sweep(cfg["theta_grid"], aligned["theta_scores"], str(fig_root / "theta_sweep.png"))
    plots.attractor_3d(
        aligned["Xemb"],
        str(fig_root / "attractor_3d.png"),
        color=np.linspace(0, 1, aligned["Xemb"].shape[0]),
    )

    meta = {
        "subject": sub,
        "story": story,
        "delta": aligned["delta"],
        "E_mult": cfg.get("E_mult", cfg["E_grid"][1]),
        "tau": cfg["tau"],
        "k": aligned["k"],
        "theta": cfg.get("smap_theta", cfg["theta_grid"][2]),
        "theiler": aligned["theiler"],
        "n_samples": aligned["n_samples"],
        "shortlist_size": len(shortlist),
        "smap_minus_baseline": aligned["rho_smap"] - aligned["rho_baseline"],
        "ccm_monotone": all(a <= b for a, b in zip(ccm_res["skill_curve"], ccm_res["skill_curve"][1:])),
        "theta_preference": aligned["theta_pref"],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Shortlist size: {len(shortlist)}")
    print(f"S-Map vs drivers-only: {aligned['rho_smap'] - aligned['rho_baseline']:.3f}")
    print(f"CCM convergence slope: {ccm_res['convergence']:.3f}")
    print(f"Preferred forecaster: {aligned['theta_pref']}")


def main():
    parser = argparse.ArgumentParser(description="Run the one-subject EDM/CCM demo")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
