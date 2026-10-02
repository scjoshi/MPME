# Plan: Neural-network reconstruction for MPME

## Goal

Replace or augment the analytic MPME pipeline (Cheng et al. 2019) with neural networks that
go from MPME data to quantitative maps (T1, T2, T2*, B0, B1+, M0). The networks should be
more robust to noise and allow faster (undersampled) scans.

There are three levels of "reconstruction". We build them in this order, each on top of the last:

| Level | Input → Output | Why |
|---|---|---|
| **A. Voxel-wise mapping** | MPME signals per voxel → parameters | Easiest; direct replacement for Eqs. 1, 15–17 |
| **B. Spatial mapping** | MPME images → parameter maps | Uses spatial context to fight noise. The 2019 paper only recommends ROI-level use because of noise |
| **C. Model-based k-space recon** | Undersampled multi-coil k-space → parameter maps | Acceleration; physics stays in the loop |

Guiding principle: **physics-informed and simulation-first.** A differentiable MPME forward
model is the backbone. It generates training data, gives a data-consistency term for the
networks, and enables self-supervised training on real scans where no ground truth exists.

---

## Phase 1 — Differentiable MPME forward model *(foundation)*

- `src/mpme/sequence.py`: protocol definition (TR, α per scan, gradient moments, pathway
  set e.g. [1,0,−1], echo times, 2-scan design).
- `src/mpme/epg.py`: extended phase graph (EPG) simulator in PyTorch, batched over voxels
  and differentiable. Steady state for each pathway F_k, multi-echo T2* decay, off-resonance
  phase, B1+ scaling. Add diffusion attenuation of the higher-order pathways, which matters
  for unbalanced gradients.
- `src/mpme/analytic.py`: reimplementation of the 2019 sequential analytic reconstruction,
  as the **baseline**.
- **Checks:** EPG matches closed-form steady states (FISP/PSIF/DESS limits); the analytic
  recon recovers ground truth from noiseless simulations; unit tests in `tests/`.

## Phase 2 — Synthetic data engine

- Parameter maps: BrainWeb / other digital brain phantoms with tissue T1/T2/T2*/PD, plus
  smooth random B0 and B1+ fields and randomised tissue values to avoid overfitting to
  textbook values. Optionally add lesion-like inclusions.
- `src/mpme/forward.py`: maps → EPG → images (pathways × echoes × 2 scans) → coil
  sensitivities → FFT → sampling mask (Cartesian / variable density / PROPELLER-like
  ky–kz, low-res 2nd scan) → complex Gaussian noise at realistic SNR.
- Store as HDF5 shards in `data/`.

## Phase 3 — Level A: voxel-wise network

- MLP (or small 1D conv) mapping complex/magnitude MPME signals → (T1, T2, T2*, B0, B1+, M0),
  trained on EPG simulations with noise. Use a heteroscedastic (Gaussian NLL) loss so each
  output comes with an uncertainty.
- Compare against: the analytic baseline, dictionary matching, and the Cramér–Rao lower
  bound (CRLB; tells us what is achievable per parameter).
- Research questions: can the NN reduce or remove the need for the high-α 2nd scan? Which
  pathways and echoes carry the most information (ablations)?
- **Go/no-go:** the NN beats the analytic baseline in RMSE at matched SNR and gets close to the CRLB.

## Phase 4 — Level B: spatial network

- U-Net / ResNet taking all MPME contrast channels and outputting parameter maps
  (+ uncertainty).
- Loss = supervised (sim ground truth) + **physics consistency**: re-simulate MPME images
  from the predicted maps through the differentiable EPG and compare to the input. The
  physics-consistency term alone allows self-supervised fine-tuning on real data.
- Handle the 2-scan resolution mismatch and possible inter-scan motion (register, or feed
  the low-res scan as a separate input branch).

## Phase 5 — Level C: model-based k-space reconstruction

- Step 1: joint multi-contrast image recon from undersampled k-space using an unrolled
  network (VarNet/MoDL style) with a SENSE data-consistency operator. Contrasts are
  stacked as channels to exploit shared anatomy.
- Step 2: end-to-end model-based recon. The network iterates on the *parameter maps*;
  data consistency goes through EPG → coils → FFT → mask. This gives quantitative maps
  straight from k-space.
- Self-supervised variants (SSDU-style k-space splitting) for real undersampled data.
- Evaluate quality vs acceleration factor (R = 2…8) against the fully sampled analytic baseline.

## Phase 6 — Real data and validation

- Data sources (to be decided): raw MPME k-space from the original authors / collaborators,
  or our own implementation in **Pulseq** to acquire phantoms and volunteers.
- References: NIST/ISMRM system phantom; IR-SE (T1), multi-echo SE (T2), multi-echo GRE
  (T2*), B1+ map (AFI/DREAM), B0 field map.
- Metrics: bias and limits of agreement (Bland–Altman, as in the paper), NRMSE/SSIM on
  maps, ROI means for WM/GM/thalamus, test–retest repeatability, robustness to
  out-of-distribution tissue and pathology.
- Close the sim-to-real gap: fine-tune with the physics-consistency loss; model the effects
  missing from the sim (diffusion, magnetisation transfer, slab profile, B1+ calibration —
  the paper needed β ≈ 1.24).

## Phase 7 — Packaging

- CLI: `scripts/simulate.py`, `scripts/train.py`, `scripts/reconstruct.py`.
- Config files per experiment, seeded runs, results in `results/`.
- Write-up in `docs/`.

---

## Proposed code layout

```
src/mpme/
  sequence.py    protocol definition
  epg.py         differentiable EPG simulator
  analytic.py    2019 baseline reconstruction
  phantom.py     parameter-map generation
  forward.py     images → coils → k-space → mask → noise
  crlb.py        Cramér–Rao bounds
  models/        mlp.py, unet.py, unrolled.py
  losses.py      supervised, NLL, physics consistency
  train.py, eval.py
```

## Main risks

| Risk | Mitigation |
|---|---|
| No access to raw MPME data | Simulation-first design; Pulseq implementation; contact the Madore lab |
| Sim-to-real gap (diffusion, MT, B1 calibration) | Richer forward model, physics-consistency fine-tuning, phantom calibration |
| Parameter degeneracy (T1 vs α from one scan) | CRLB analysis up front; keep the 2-scan design until shown unnecessary |
| Network hallucination in undersampled recon | Data-consistency layers, uncertainty outputs, reference-scan validation |

## Open decisions

1. What data is available: raw multi-coil k-space, reconstructed images, or none yet?
2. Primary target: better maps from fully sampled data (Levels A/B) or acceleration (Level C)?
3. Compute: local GPU / Apple MPS / cluster?

---

## Findings so far

### B1⁺/T1 are poorly conditioned with the placeholder protocol (2026-10-02)

Default protocol (`sequence.default_protocol`: α = 3°/30°, TR = 22 ms, pathways [1, 0, −1],
3 echoes each, both scans treated as full resolution), white-matter-like voxel
(T1 850, T2 65 ms, R2′ 0.02/ms).

- The analytic baseline is exact on noise-free data (relative error ≤ 1e-7).
- T2, T2* and B0 are reasonably robust to noise, much more so when the decay fit also
  uses the high-flip scan (`decay_scans=(0, 1)`), whose echo pathway is ~20× stronger.
- B1⁺ and T1 are not: the Cramér–Rao bound for log T1 is ≈ 2800·σ/M0 even when the full
  model is fitted jointly to every pathway and echo (≈ 3600·σ/M0 for the FID/echo-only
  analytic step). 10% T1 precision would need SNR(M0) of order 10⁴.
- Larger flip angles help a lot (α = 10°/100°: ≈ 375·σ/M0 joint), and including the
  p = +1 pathway helps by 1.3–4×.

Implications: the real protocol values matter a great deal and should be confirmed from
the paper; protocol optimisation by CRLB belongs early in Phase 3; a neural network cannot
beat the CRLB, so its gains must come from spatial priors (Level B) or better protocols.
The low-resolution second scan (16× larger voxels) has higher SNR per voxel, which is
not yet modelled here.
