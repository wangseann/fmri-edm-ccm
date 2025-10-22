# fMRI EDM/CCM Demo

One-subject prototype for embedding delay maps (EDM) and convergent cross mapping (CCM) on the ds003020 language dataset. The codebase now exposes a clean API plus a reproducible notebook/CLI for Subject `UTS01` and `Moth_Story_001`.

## Environment
- Dataset root: `/bucket/PaoU/seann/openneuro/ds003020`
- Parcellations: `parcellations/` (Schaefer atlases, etc.)
- Python ≥ 3.10
- Key deps: NumPy, pandas, scikit-learn, Matplotlib, PyYAML

Install locally:
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Workflow
1. Edit `configs/demo.yaml` to switch subjects, stories, or grids.
2. Run the tiny notebook `notebooks/Day9_One_Subject_Demo.ipynb` — each cell calls into `src/` modules.
3. Optional CLI mirroring the notebook:
```
python -m src.run_demo --config configs/demo.yaml
```
   Outputs land under `results/{sub}/{story}` (CSV + meta.json) and `figs/{sub}/{story}` (PNG plots).

## Module contracts
- `src/features.py`: cached driver loading, PCA, lag stacks
- `src.roi.py`: Schaefer ROI loading with per-story z-scoring, optional ICA path
- `src.edm.py`: horizon shift, embeddings, simplex/S-Map forecasters, correlation skill metric
- `src.ccm.py`: CCM pair analysis plus conditional screening for shortlist selection
- `src.baselines.py`: ridge regression forecaster for driver-only baseline
- `src.plots.py`: forecast and CCM summary plots
- `src.utils.py`: YAML I/O, seeding, CSV helpers, column-wise z-scoring

## Acceptance checks
The notebook/CLI prints:
- S-Map skill minus drivers-only at Δ = 1 TR (should be ≥ 0.05)
- CCM convergence slope and monotonic curve check
- Preferred forecaster from θ sweep (S-Map if any θ improves over simplex)

## Tests
Four smoke tests cover shapes and basic API contracts:
```
pytest tests
```

## Outputs
- Cached arrays: `data_cache/{sub}/{story}/`
- Results: `results/{sub}/{story}/` (`shortlist.csv`, `forecast.csv`, `skills.csv`, `meta.json`)
- Plots: `figs/{sub}/{story}/` (`forecast_bars.png`, `ccm_curve.png`, `theta_sweep.png`)
