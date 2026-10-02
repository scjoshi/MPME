# Technical note plan (definitive version)

Audience: internal / technical. Full proofs. Supersedes `docs/latex/mpme_b1_t1_limits`
(kept as the review-stage record).

## Structure

0. Abstract, summary of findings, how to read
1. Notation and conventions (symbols, phase conventions, units, log parameters)
2. Sequence and forward model: timing and pathway refocusing (figure), configuration
   states, exact isochromat steady state, off-resonance = phase, echo evolution with
   Lorentzian R2′, Theorem 1 (two invariants per scan), closed forms, imaging and noise,
   validation
3. The published inverse (Cheng et al. 2019): steps 1–5 (flowchart), consistency with the
   model (proved), Eq. 16 reading, branch rule, β, noise-free exactness, failure modes
4. Information analysis: Fisher information, diagnostics, log parameters, one-scan
   singularity, small-angle scaling, how 330° breaks it, conditioning, global aliases,
   protocol dependence, magnitude (Rician) information
5. Maximum likelihood: algorithm, constraints, starts, failure check, Monte Carlo vs CRLB,
   one start vs four starts at low SNR
6. Least squares on magnitudes: Rician floor, why the branch test is needed, FID-sign
   constraint, comparison on identical data
7. Implicit Jacobians: derivation, LM reuse, verification, benchmark, implications
   (whole-brain feasibility, uncertainty maps, protocol optimisation, training data,
   EM E-step, float32/GPU)
8. 2D digital phantom: phantom, simulated MPME images, Scenario A (both scans at full
   resolution: analytic inverse, complex ML, magnitude LS + FID sign), Scenario B
   (published acquisition: low-resolution scan 2; paper pipeline vs two-stage ML),
   ROI statistics, error maps, run times, uncertainty-map calibration
9. Discussion: verdict table, limitations; future work: model mismatch, Rician ML,
   hierarchical EM, neural networks
A. Proofs; B. Algorithms (pseudocode); C. Reproducibility (`make technote`)

## Code additions

- `phantom.py`: 2D digital brain phantom (tissue classes, lesion, smooth B1⁺/B0/coil fields)
- simulation of MPME images incl. low-resolution scan 2 (k-space truncation)
- fixed-parameter (e.g. B1 given) ML fits; per-voxel Fisher uncertainty at the estimate
- B1 cap at 540/330 for the paper protocol (excludes the complex alias by construction)
- scripts for phantom experiments and figures; Makefile target
