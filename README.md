fMRI Language Decoding with EDM/CCM

This repo scaffolds a PhD project on forecasting semantic trajectories from fMRI (ds003020 dataset).

Today's scope:
- Create repo scaffold with CI and pre-commit hooks.
- Run one Day-1 notebook to explore dataset (subject/story list, driver previews).

Data root: /bucket/PaoU/seann/openneuro/ds003020, TR=2.0 s.

How to run Day 1 exploration
1. Install minimal deps: numpy, pandas, matplotlib. For driver previews also install librosa, soundfile, textgrid.
2. Open notebooks/Day1_Exploration.ipynb and run cells.

Development setup

Install pre-commit and enable hooks:
```
pip install pre-commit
pre-commit install
```
