# MPME — Multi-Pathway Multi-Echo MRI

Simulation and parameter-mapping tools for the MPME sequence:

> Cheng CC, Preiswerk F, Hoge WS, Kuo TH, Madore B. *Multipathway multi-echo (MPME)
> imaging: all main MR parameters mapped based on a single 3D scan.*
> Magn Reson Med. 2019;81:1699–1713. https://doi.org/10.1002/mrm.27525

## Layout

```
src/mpme/     Python package (signal models, simulation, fitting)
tests/        Unit tests (pytest)
scripts/      Command-line entry points / experiments
notebooks/    Exploratory Jupyter notebooks
docs/         Notes, derivations, LaTeX
references/   Papers and supporting material (not tracked)
data/         Input data (not tracked)
results/      Generated outputs (not tracked)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
