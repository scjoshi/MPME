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

## Scripts

Run everything from the repository root. Scripts import the package from `src/`, so use
`PYTHONPATH=src python3 scripts/<name>.py` (not needed after `pip install -e .`). Scripts that
import another script are run with `PYTHONPATH=src:scripts` in the Makefile. Only the scripts
listed with flags parse arguments: all others ignore `--help` and run the full computation.

Outputs in `results/` are not tracked, but figures and `.tex` tables written to `docs/` are, so
rerunning a script changes the working tree. Numbers quoted by hand in the prose of the notes are
not regenerated; compare and update them after a rerun. Times are for an 8-thread laptop CPU.

### Technical note — `make technote` (= `technote-results` + `technote-pdf`)

The five analysis scripts are independent and deterministic; each takes seconds to ~2 min.

| Script | Purpose | Writes |
|---|---|---|
| `crlb_analysis.py` | Cramér–Rao bounds and Fisher conditioning, 15°/330° vs 15°/30°; flip-angle, B1 and T1 sweeps | `docs/figures/crlb_*.png`; Markdown tables to stdout (copied by hand into `04_information.tex`) |
| `single_scan_family.py` | Worked examples of the exact one-scan solution family (Ernst-angle form) | `docs/technote/tables/family_{one,two}.tex`, `docs/technote/figures/ernst_logtan.png` |
| `shared_b1_check.py` | Whether voxels sharing one B1 rescue a single scan | `docs/technote/tables/shared_b1.tex`, `results/shared_b1_check.json` |
| `beyond_ideal_model.py` | Whether physics outside the ideal model gives scan 1 any B1 information | `docs/technote/tables/beyond_model.tex`, `results/beyond_ideal_model.json` |
| `start_count.py` | Whether one maximum-likelihood start is enough (2000 voxels, SNR 300/1000) | `results/start_count.json` (→ `\Start*` macros) |

The simulation experiments run next, in this order:

```bash
PYTHONPATH=src python3 scripts/phantom_experiment.py --snr 1000 300   # ~9 min per SNR
PYTHONPATH=src:scripts python3 scripts/brainweb_experiment.py --case pv_clean   # also: pv, crisp; ~3 min each
PYTHONPATH=src python3 scripts/technote_figures.py
```

- `phantom_experiment.py`: digital brain phantom, analytic inverse vs complex ML vs magnitude
  LS (Scenarios A and B). Writes `results/phantom_snr{1000,300}.pt` (~77 MB each) and
  `results/phantom_summary.json`. Only SNRs 1000 and 300 are supported downstream.
- `brainweb_experiment.py --case {pv_clean,pv,crisp}`: partial-volume test on one BrainWeb
  slice; needs the data in `data/brainweb/`. Run all three cases (not in parallel; they share
  `results/brainweb_summary.json`).
- `technote_figures.py`: all phantom and BrainWeb figures, tables and `macros.tex` for the
  technote. Needs the outputs of the two experiments and `start_count.py`.

### Estimator Monte Carlo — `make technote-reference` (slow)

| Script | Usage |
|---|---|
| `estimator_efficiency.py` | Efficiency of each estimator against the CRLB (1000 noise draws). Flags: `--jacobian {fd,implicit}` (default `fd`), `--float32`, `--n-iso 256`, `--n-iter 40`. Writes `results/estimator_efficiency{suffix}.json` and `_raw.pt`. Reference run ~25 min. |
| `magnitude_comparison.py` | Magnitude vs complex estimation on the same data. Same flags; run `estimator_efficiency.py` with the same flags first. |
| `make_mc_table.py [json]` | Writes `docs/latex/mc_table.tex` from the efficiency results (< 1 s). |
| `make_mag_table.py [json]` | Writes `docs/latex/mag_table.tex` and `mag_counts.tex` (< 1 s). |
| `fit_options.py` | Helper module for the shared flags (not run directly). Non-default settings add a filename suffix, so reference results are never overwritten. |
| `benchmark_fit.py [N]` | Timing of the voxelwise fitters on N voxels (default 4096, ~3 min); prints only. Not in the Makefile. |

### Flip-angle design note — `make flipnote`

1. `flip_design.py` — exhaustive scan of flip-angle pairs (1° grid to 540°, B1 0.7–1.3 plus
   the exact 360° crossings, five tissues): `results/flip_design.npz` (~15 min, >10 GB RAM peak).
2. `flip_design_report.py` — shortlist, closed-form bounds, alias intervals; figures, tables and
   macros in `docs/flipnote/` (1–2 min). Rerun alone to redraw without repeating step 1.
3. `PYTHONPATH=src:scripts python3 scripts/flip_design_robustness.py` — bias of the shortlisted
   designs under unmodelled physics: `docs/flipnote/tables/robustness.tex` (~6 min).

### B1/T1 limits companion — `make limits`

`limits_figures.py` writes `docs/figures/limits_{lever,crossing}.png`,
`docs/latex/limits_budget.tex` and `results/limits_numbers.json` (< 1 min), then the Makefile
builds `docs/latex/mpme_b1_t1_limits.pdf`.

### Other

`two_scan_design.py` (~1–3 min) shows why one scan cannot separate B1 and T1 and evaluates second-scan
designs: `results/two_scan_design.json`. It is not part of any Makefile target.
