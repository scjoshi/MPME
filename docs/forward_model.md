# MPME forward signal model

Derived from standard steady-state MR physics (extended phase graphs, EPG). It is
consistent with the 2019 paper's figure captions (B0 from multi-echo phase, T2/T2* from
echo decay, B1+/T1 from two flip angles). The paper's Methods section was not available
when this was written, so its notation, sign conventions and equation numbers may differ.

```
θ(r) = {C·M0, T1, T2, R2′, Δω, B1⁺, (D)}      tissue + field parameters per voxel
   │  Layer 1: EPG steady state (RF, relaxation, gradient dephasing)     src/mpme/epg.py
   ▼
a_p(α_i, T1, T2, TR_i, D)                       pathway amplitudes right after the RF pulse
   │  Layer 2: evolution to each echo time (T2, R2′, Δω)
   ▼
S_ijp(r)                                        voxel signal: scan i, echo j, pathway p
   │  Layer 3: imaging (coils, Fourier, sampling, noise)
   ▼
y_ijpc(k)                                       measured multi-coil k-space
```

## 1. Sequence and notation

- 3D gradient echo, **no RF spoiling** (constant RF phase φ), so transverse magnetisation
  carries over between TRs and forms multiple pathways.
- Every TR has the same **unbalanced** readout-gradient moment A, dephasing many cycles
  across a voxel ("fully dephased" assumption).
- Two scans i = 1, 2: (α₁, TR₁) at full resolution, and (α₂ ≈ 10·α₁, TR₂) covering only the
  k-space centre (1/16 matrix). Actual flip angle α_i = B1⁺(r)·α_nom,i.
- Configuration states: F_k (transverse, k turns of gradient twist), Z_k (longitudinal).
- With m(t) the gradient moment since the last RF (units of A), the state F_p created at the
  RF pulse is refocused when **m(t) = −p**.
  - [1, 0, −1] scheme: Gx goes negative to refocus p = 1, passes zero for p = 0 (FID,
    FISP-like), and reaches +1 for p = −1 (echo, PSIF-like).
  - [0, −1, −2] scheme: Gx reaches m = 2.

### Conventions used in the code

- Transverse magnetisation M⁺ = Mx + iMy. Intravoxel gradient phase ψ ∈ [0, 2π) and
  M⁺(ψ) = Σ_k F_k e^{ikψ}, Mz(ψ) = Σ_k Z_k e^{ikψ} (so Z_{−k} = Z_k*).
- One TR of gradient multiplies by e^{iψ}: F_k → F_{k+1}. Off-resonance uses the same
  sense: free precession for time t multiplies M⁺ by e^{iΔω t}.
- RF pulse with flip α and phase φ = rotation Rz(φ)·Rx(α)·Rz(−φ). From equilibrium
  it gives F_0 = −i e^{iφ} sin α.
- Times in ms, rates in 1/ms, D in mm²/s, gradient twist q in rad/mm.

## 2. Layer 1: EPG steady state

State x = {F_k, Z_k}, |k| ≤ K. Per TR:

**RF rotation T(α, φ)** (Weigel 2015 form):
```
F_k'  = cos²(α/2) F_k + e^{2iφ} sin²(α/2) F_{-k}* − i e^{iφ} sin α Z_k
Z_k'  = −(i/2) e^{-iφ} sin α F_k + (i/2) e^{iφ} sin α F_{-k}* + cos α Z_k
```

**Relaxation over TR:** F_k → E2·F_k, Z_k → E1·Z_k (k ≠ 0), Z_0 → E1·Z_0 + M0(1−E1),
with E1 = e^{−TR/T1}, E2 = e^{−TR/T2}.

**Gradient shift:** F_k → F_{k+1}.

**Diffusion (optional):** for a constant gradient over the TR, with twist q per TR and
τ = TR,
```
F_k: × exp(−D q² τ (k² + k + 1/3))      Z_k: × exp(−D q² k² τ)
```
p = 0 is barely affected; the echo pathways are attenuated (as in DESS).

**Steady state:** a_p = F_p just after the RF pulse, per unit M0.

### Two solvers

| Solver | How | Use |
|---|---|---|
| `steady_state_isochromat` | Exact 3×3 steady state for N isochromats spread over ψ, then FFT over ψ to get F_p | Default: fast, exact for the fully dephased case, works for long T2 |
| `simulate_epg` | Runs the EPG operators TR by TR (truncated at K states) | Diffusion, approach to steady state, cross-check |

Truncation and aliasing: both need the number of significant orders (≈ a few × T2/TR) to be
below K (EPG) or N/2 (isochromats). Long-T2 tissue such as CSF needs large N or K.

### Off-resonance only changes the phase

The gradient spreads ψ uniformly over [0, 2π). An off-resonance Δω adds θ = Δω·TR per TR,
which shifts the steady state: M_θ(ψ) = M_0(ψ + θ). F_p is the p-th Fourier coefficient
over ψ, so it is multiplied by e^{ipθ}:

```
|a_p| does not depend on Δω    (unlike balanced SSFP, which has banding)
```

Closed forms for p = 0 (FISP) and p = −1 (PSIF) are used as unit tests.

## 3. Layer 2: evolution to each echo

Pathway p, echo j of scan i is read at t = t_ijp after the RF pulse. One isochromat has
off-resonance phase Δω·(t + p·TR_i). Averaging over a Lorentzian intravoxel frequency
spread (half-width R2′):

```
S_ijp = C·M0 · a_p(B1⁺α_nom,i, T1, T2, TR_i, D)
             · e^{−t/T2}
             · e^{−R2′·|t + p·TR_i|}
             · e^{i(φ0 + φ_p + Δω·(t + p·TR_i))}
```

- p = 0: decays at R2 + R2′ = R2*.
- p = −1: |t − TR|, the reversible dephasing refocuses at t = TR; before that the log-slope
  is −(R2 − R2′).
- p = +1: t + TR, a full extra TR of T2′ decay.
- φ_p: constant phase per pathway. Echo pathways contain conjugated terms, so the
  transmit/receive phase enters them with the opposite sign.

| Parameter | Where it shows up |
|---|---|
| R2, R2′ | With log-slopes s₀ = −(R2+R2′), s₋₁ = −(R2−R2′): R2 = −(s₀+s₋₁)/2, R2′ = (s₋₁−s₀)/2. Needs ≥ 2 pathways × ≥ 2 echoes. T2* = 1/(R2+R2′). |
| Δω (B0) | Phase linear in echo time, same rate for every pathway. |
| B1⁺, T1 | Only through ratios of the a_p (the "mixing factor"). One scan cannot separate α from T1; the second, 10× flip-angle scan resolves both. T2 also enters, so it is estimated first. |
| C·M0 | Overall scale; receive sensitivity and M0 are inseparable (paper's "C×M0"). |

## 4. Layer 3: imaging operator

```
y_ijpc = M_i · F · (C_c ⊙ S_ijp) + n_ijpc
```

C_c coil sensitivities, F the 3D Fourier transform, M_i the sampling mask (scan 1:
PROPELLER-like ky–kz with 10× oversampled centre and 40% undersampled periphery; scan 2:
centre only), n complex Gaussian noise correlated across coils.

Second-order effects: T2* and Δω during the readout (blur and shift along x; opposite
shifts for alternating readout directions), leakage between pathways when A is not much
larger than the readout kx extent, motion between the two scans.

## 5. What the model leaves out

| Effect | Impact | Plan |
|---|---|---|
| Diffusion | Biases T2/T1 from the echo pathways | Included in `simulate_epg` |
| Slab profile | α varies across the 3D slab | Multiply by B1⁺ map, or simulate the profile |
| Non-Lorentzian lineshape | exp(−R2′\|τ\|) is approximate | Sensitivity tests; Gaussian option |
| Magnetisation transfer, finite RF pulses | T1 bias (the paper used a calibration β ≈ 1.24; MT as a cause is a guess) | Optional MT term; phantom calibration |
| Approach to steady state, flow, motion | Corrupts echo pathways | Assume steady state; robustness tests |
