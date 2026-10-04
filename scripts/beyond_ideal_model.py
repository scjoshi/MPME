"""Does physics outside the ideal model give one scan (scan 1, 15°) information about B1?

The ideal model (instantaneous rectangular pulses, one flip angle per voxel, no diffusion) makes
scan 1 exactly degenerate in B1 (Theorem 1). Each effect below replaces the pathway amplitudes
a_p of mpme.signal by a richer model, keeps the echo model, and gives the scan-1 CRLB on log B1
with every other parameter (log M0, log T1, log T2, log T2*, Δω, φ0, plus any extra) free.
Conventions of mpme.crlb: σ = 1 per real/imaginary part, M0 = 1, so bounds are per unit σ/M0.

  1. ideal model: two-scan reference (15°/330°) and scan 1 alone
  2. intra-voxel flip-angle distribution of known shape, each position its own steady state:
     Gaussian slice (|z| ≤ 3σ), Hann-windowed sinc slice (TBW 4, small-tip profile over
     |f| ≤ 1.5 TBW), slab-edge ramp (flip fraction 0.9…1)
  3. the same with the profile width unknown
  4. width mismatch: data with true width w = γ^{±1} × nominal (γ = 1.02, 1.05, 1.10), scan-1
     fit (all parameters free, started at the truth) with the nominal width
  5. relaxation during a finite on-resonance hard pulse (τ = 1, 2 ms)
  6. off-resonance and relaxation during a 1 ms hard pulse (Δω/2π = 8, 48, 160 Hz)
  7. diffusion in the unbalanced gradient (EPG), D known and D free

    PYTHONPATH=src python3 scripts/beyond_ideal_model.py
    → results/beyond_ideal_model.json, docs/technote/tables/beyond_model.tex
"""

import json
import math
import time
from pathlib import Path

import torch

from mpme.crlb import crlb, fisher_information
from mpme.epg import epg_pathways, simulate_epg, steady_state_isochromat
from mpme.sequence import Protocol, paper_protocol
from mpme.signal import mpme_signal

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "docs" / "technote" / "tables"
F64 = torch.float64
PR = paper_protocol()
SCAN1 = Protocol(PR.pathways, (PR.scans[0],))
TISSUES = {"WM": dict(M0=1.0, B1=1.0, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3),
           "GM": dict(M0=1.0, B1=1.0, T1=1350.0, T2=90.0, T2star=60.0, dw=0.05, phi0=0.3)}
N_ISO = 256               # isochromats per steady state
N_Z = 64                  # positions across the voxel for a flip-angle profile
EPG_TR, EPG_K = 1000, 64  # TRs to steady state, highest EPG order
DEGENERATE = 1e-9         # relative B1-column residual below which the information is rounding
GAMMAS = (1.02, 1.05, 1.10)
TAUS = (1.0, 2.0)                 # ms
OFFRES_HZ = (8.0, 48.0, 160.0)
D_TISSUE = 0.7e-3                 # mm²/s
VOXELS_MM = (1.0, 0.5)            # gradient twist q = 2π / voxel size per TR


# ------------------------------------------------------------------ signal model and bound
def theta(t, *extra):
    """(log M0, log B1, log T1, log T2, log T2*, Δω, φ0, *extra)."""
    return torch.tensor([math.log(t[k]) for k in ("M0", "B1", "T1", "T2", "T2star")]
                        + [t["dw"], t["phi0"], *extra], dtype=F64)


def signal(protocol, th, amp):
    """Echo signals [n_scans, P, J] of mpme_signal with pathway amplitudes from ``amp``.

    amp(alpha, T1, T2, TR, dw, extra) returns a_p per unit M0 without the factor e^{i p Δω TR},
    which is applied here with the echo decay and phase, exactly as in mpme_signal.
    """
    M0, B1, T1, T2, T2s = th[:5].exp()
    dw, phi0, extra = th[5], th[6], th[7:]
    p = torch.tensor(protocol.pathways, dtype=F64)[:, None]
    out = []
    for i, scan in enumerate(protocol.scans):
        a = amp(B1 * scan.alpha, T1, T2, scan.TR, dw, extra)
        t = protocol.echo_times(i)
        tau = t + p * scan.TR
        out.append(M0 * a[:, None] * torch.exp(-t / T2 - (1 / T2s - 1 / T2) * tau.abs()
                                               + 1j * (phi0 + dw * tau)))
    return torch.stack(out)


def real_signal(protocol, amp):
    return lambda th: torch.view_as_real(signal(protocol, th, amp)).flatten()


def jacobian(protocol, th, amp):
    """∂(Re, Im of every echo)/∂θ, [2 · n_echo_values, n_params]."""
    return torch.autograd.functional.jacobian(real_signal(protocol, amp), th, vectorize=True)


def b1_bound(J, drop=()):
    """CRLB on log B1 (column 1), all other columns free except ``drop`` (treated as known).

    r = B1 column minus its projection onto the free columns; the profiled information is |r|²,
    so the bound is 1/|r|, and |r|/|J_B1| = 0 means exact degeneracy.
    """
    others = [i for i in range(J.shape[1]) if i != 1 and i not in drop]
    Q, _ = torch.linalg.qr(J[:, others])
    r = J[:, 1] - Q @ (Q.T @ J[:, 1])
    return {"crlb": 1 / r.norm().item(), "rel_residual": (r.norm() / J[:, 1].norm()).item()}


def change(th, amp):
    """max |S − e^{iφ} S_ideal| / max |S_ideal| on scan 1 at the same parameters (size of the
    effect), φ the best common phase: φ0 absorbs it, and for a finite off-resonant pulse it is
    ≈ Δω τ/2, set only by timing the echoes from the end of the pulse."""
    S, S0 = signal(SCAN1, th, amp), signal(SCAN1, th[:7], ideal)
    c = (S0.conj() * S).sum()
    return ((S - S0 * c / c.abs()).abs().max() / S0.abs().max()).item()


def shape_gap(th, amp):
    """|a_−1/a_0 − (r − E)/(1 − E r)| with r = a_1/a_0 and E = e^{−TR/T2}, on scan 1.

    Zero when the pathway amplitudes have the one-parameter shape (r, given T2) of any
    instantaneous or finite pulse about x on resonance; then scan 1 fixes only r and the scale,
    so it cannot separate B1 from T1 and M0.
    """
    s = SCAN1.scans[0]
    a1, a0, am1 = amp(th[1].exp() * s.alpha, th[2].exp(), th[3].exp(), s.TR, th[5], th[7:])
    r, E = a1 / a0, torch.exp(-s.TR / th[3].exp())
    return (am1 / a0 - (r - E) / (1 - E * r)).abs().item()


# ------------------------------------------------------------------ pathway-amplitude models
def ideal(alpha, T1, T2, TR, dw, extra):
    return steady_state_isochromat(alpha, T1, T2, TR, PR.pathways, n_iso=N_ISO)


_T = torch.linspace(-0.5, 0.5, 2001, dtype=F64)                          # time / pulse length
_PULSE = torch.sinc(4 * _T) * (0.5 + 0.5 * torch.cos(2 * math.pi * _T))  # Hann-sinc, TBW 4


def hann_sinc(f):
    """Small-tip profile of the Hann-windowed sinc, f in units of 1/pulse length; P(0) = 1."""
    return (_PULSE * torch.cos(2 * math.pi * f[:, None] * _T)).sum(-1) / _PULSE.sum()


PROFILES = {  # name: (flip fraction p(z), voxel z range), z uniform over the voxel
    "gaussian": (lambda z: torch.exp(-z**2 / 2), (-3.0, 3.0)),
    "hann_sinc": (hann_sinc, (-6.0, 6.0)),
    "ramp": (lambda z: 1 - 0.1 * z, (0.0, 1.0)),
}


def profile(name, width=1.0, n_z=N_Z):
    """Voxel mean (midpoint rule over z) of the ideal amplitudes at flip alpha · p(z / w), every
    position its own steady state. w = ``width``, or exp(extra[0]) when an extra is passed."""
    shape, (lo, hi) = PROFILES[name]
    z = lo + (hi - lo) * (torch.arange(n_z, dtype=F64) + 0.5) / n_z

    def amp(alpha, T1, T2, TR, dw, extra):
        w = extra[0].exp() if len(extra) else width
        return steady_state_isochromat(alpha * shape(z / w), T1, T2, TR, PR.pathways,
                                       n_iso=N_ISO).mean(0)
    return amp


def hard_pulse(tau, off_resonance):
    """Rectangular pulse of duration ``tau`` (ms): exact Bloch solution with T1/T2 relaxation and,
    if ``off_resonance``, precession at Δω during the pulse; dephasing, relaxation and precession
    over the remaining TR − tau. Amplitudes just after the pulse (all at Δω = 0 otherwise).
    Echo times are taken from the end of the pulse."""
    def amp(alpha, T1, T2, TR, dw, extra):
        w = dw if off_resonance else torch.zeros((), dtype=F64)
        w1, z = alpha / tau, torch.zeros((), dtype=F64)
        L = torch.stack([torch.stack([-1 / T2, -w, z, z]),           # d/dt (Mx, My, Mz, 1)
                         torch.stack([w, -1 / T2, -w1, z]),
                         torch.stack([z, w1, -1 / T1, 1 / T1]),
                         torch.stack([z, z, z, z])])
        Tf = TR - tau
        E1, E2 = torch.exp(-Tf / T1), torch.exp(-Tf / T2)
        psi = 2 * math.pi * torch.arange(N_ISO, dtype=F64) / N_ISO + w * Tf
        c, s = E2 * torch.cos(psi), E2 * torch.sin(psi)
        o, n = torch.ones_like(psi), torch.zeros_like(psi)
        free = torch.stack([torch.stack([c, -s, n, n], -1), torch.stack([s, c, n, n], -1),
                            torch.stack([n, n, E1 * o, (1 - E1) * o], -1),
                            torch.stack([n, n, n, o], -1)], -2)          # [N, 4, 4]
        C = torch.linalg.matrix_exp(tau * L) @ free                      # one TR, ends after pulse
        M = torch.linalg.solve(torch.eye(3, dtype=F64) - C[:, :3, :3], C[:, :3, 3])
        F = torch.fft.fft(torch.complex(M[:, 0], M[:, 1])) / N_ISO
        k = torch.tensor(PR.pathways, dtype=F64)
        return F[[q % N_ISO for q in PR.pathways]] * torch.exp(-1j * k * w * TR)
    return amp


def diffusion(q, n_tr=EPG_TR, K=EPG_K):
    """EPG with diffusion under a gradient twist q (rad/mm) per TR; D = D_TISSUE, or exp(extra[0])
    (mm²/s) when an extra is passed."""
    def amp(alpha, T1, T2, TR, dw, extra):
        D = extra[0].exp() if len(extra) else D_TISSUE
        return epg_pathways(simulate_epg(alpha, T1, T2, TR, n_tr, K, D=D, q=q)[0], PR.pathways)
    return amp


# ------------------------------------------------------------------ mismatch fit
# Scan 1 constrains log B1 a million times more weakly than the other parameters, so a plain
# Levenberg–Marquardt fit crawls along the curved valley. The fit below separates the two:
# Gauss–Newton in the other parameters at fixed B1, and Gauss–Newton steps in log B1 on the
# profiled residual, each new B1 started on the ideal single-scan family (ξ, M0 ζ fixed).
FREE = [0, 2, 3, 4, 5, 6]          # all but log B1
B1_FLOOR = 0.01                    # below this the fit is reported as running to B1 → 0


def jac_fwd(f, th):
    """Forward-mode Jacobian: twice as fast here (fails inside simulate_epg, so not used there)."""
    return torch.autograd.functional.jacobian(f, th, vectorize=True, strategy="forward-mode")


def family_point(th, b):
    """θ moved to log B1 = b along the ideal scan-1 family: ζ' = tan(B1' α/2)/ξ, M0' = M0 ζ/ζ'."""
    a, TR = SCAN1.scans[0].alpha, SCAN1.scans[0].TR
    M0, B1, T1 = th[:3].exp()
    zeta = torch.tanh(TR / (2 * T1)).sqrt()
    zeta_new = math.tan(math.exp(b) * a / 2) * zeta / torch.tan(B1 * a / 2)
    return torch.cat([torch.stack([(M0 * zeta / zeta_new).log(), torch.tensor(b, dtype=F64),
                                   (TR / (2 * torch.atanh(zeta_new**2))).log()]), th[3:]])


def fit_fixed_b1(f, yr, th, n_iter=20):
    """Gauss–Newton with step halving in all parameters but log B1."""
    r = yr - f(th)
    for _ in range(n_iter):
        step, t = torch.linalg.lstsq(jac_fwd(f, th)[:, FREE], r).solution, 1.0
        while t > 1e-3:
            new = th.clone()
            new[FREE] += t * step
            r_new = yr - f(new)
            if r_new.norm() <= r.norm():
                break
            t /= 2
        else:
            break
        th, r = new, r_new
        if t * step.norm() < 1e-10:
            break
    return th, r


def fit(y, th, amp, n_outer=40, max_step=0.25):
    """Least-squares fit of all parameters to noise-free scan-1 data ``y``, started at ``th``.

    Returns the estimate and the relative residual |y − S(θ̂)| / |y|."""
    f = real_signal(SCAN1, amp)
    yr = torch.view_as_real(y).flatten()
    th, r = fit_fixed_b1(f, yr, th)
    for _ in range(n_outer):
        J = jac_fwd(f, th)
        Q, _ = torch.linalg.qr(J[:, FREE])
        jb = J[:, 1] - Q @ (Q.T @ J[:, 1])                  # d(profiled residual)/d log B1
        db = max(-max_step, min(max_step, (jb @ r / (jb @ jb)).item()))
        while abs(db) > 1e-10:
            th_new, r_new = fit_fixed_b1(f, yr, family_point(th, th[1].item() + db))
            if r_new.norm() < r.norm():
                break
            db /= 4
        else:
            break
        th, r = th_new, r_new
        if abs(db) < 1e-8 or r.norm() < 1e-13 * yr.norm() or th[1].exp() < B1_FLOOR:
            break
    return th, (r.norm() / yr.norm()).item()


# ------------------------------------------------------------------ experiments
def run():
    th = {n: theta(t) for n, t in TISSUES.items()}
    chk, out = {}, {}

    # 1. Ideal model, and the generic machinery against mpme.signal and mpme.crlb
    out["ideal"] = {}
    for n, t in TISSUES.items():
        S = mpme_signal(PR, t["M0"], t["T1"], t["T2"], 1 / t["T2star"] - 1 / t["T2"], t["dw"],
                        t["B1"], t["phi0"], n_iso=N_ISO)
        chk[f"signal_vs_mpme_signal_{n}"] = ((signal(PR, th[n], ideal) - S).abs().max()
                                             / S.abs().max()).item()
        two = b1_bound(jacobian(PR, th[n], ideal))["crlb"]
        ref = crlb(fisher_information(PR, t, n_iso=N_ISO))["B1"]
        chk[f"two_scan_vs_mpme_crlb_{n}"] = abs(two / ref - 1)
        out["ideal"][n] = {"two_scan": two, "scan1": b1_bound(jacobian(SCAN1, th[n], ideal))}

    # 2, 3. Flip-angle profile, width known / free (one Jacobian with a log-width column)
    out["profile_known"], out["profile_width_free"] = {}, {}
    for name in PROFILES:
        known, free = {}, {}
        for n, t in TISSUES.items():
            J = jacobian(SCAN1, theta(t, 0.0), profile(name))
            known[n] = {**b1_bound(J, drop=(7,)), "signal_change": change(th[n], profile(name))}
            free[n] = b1_bound(J)
        out["profile_known"][name], out["profile_width_free"][name] = known, free
        J = jacobian(SCAN1, th["WM"], profile(name, n_z=2 * N_Z))
        chk[f"profile_{name}_WM_crlb_n_z_{2 * N_Z}"] = b1_bound(J)["crlb"]

    # 4. Width mismatch: true width γ^{±1} × nominal, scan-1 fit with the nominal width
    out["width_mismatch"] = {}
    for name in PROFILES:
        rows = []
        for n in TISSUES:
            for w in sorted([1 / g for g in GAMMAS] + list(GAMMAS)):
                est, rel = fit(signal(SCAN1, th[n], profile(name, width=w)), th[n], profile(name))
                rows.append({"tissue": n, "width": w, "B1": est[1].exp().item(),
                             "T1": est[2].exp().item(), "M0": est[0].exp().item(),
                             "rel_residual": rel, "to_zero": est[1].exp().item() < B1_FLOOR})
        out["width_mismatch"][name] = rows

    # 5, 6. Finite hard pulse: relaxation on resonance; off-resonance with relaxation
    chk["pulse_tau_1e-6_vs_ideal_WM"] = change(th["WM"], hard_pulse(1e-6, True))
    th48 = theta({**TISSUES["WM"], "dw": 2 * math.pi * 48 / 1000})
    chk["shape_gap_WM"] = {"ideal": shape_gap(th["WM"], ideal),
                           "pulse 2 ms on resonance": shape_gap(th["WM"], hard_pulse(2.0, False)),
                           "pulse 1 ms at 48 Hz": shape_gap(th48, hard_pulse(1.0, True))}
    out["pulse_relaxation"] = {
        f"tau={tau:g} ms": {n: {**b1_bound(jacobian(SCAN1, th[n], hard_pulse(tau, False))),
                                "signal_change": change(th[n], hard_pulse(tau, False))}
                            for n in TISSUES} for tau in TAUS}
    out["pulse_off_resonance"] = {}
    for hz in OFFRES_HZ:
        thw = {n: theta({**t, "dw": 2 * math.pi * hz / 1000}) for n, t in TISSUES.items()}
        out["pulse_off_resonance"][f"{hz:g} Hz"] = {
            n: {**b1_bound(jacobian(SCAN1, thw[n], hard_pulse(1.0, True))),
                "signal_change": change(thw[n], hard_pulse(1.0, True))} for n in TISSUES}

    # 7. Diffusion (one Jacobian with a log-D column; D known = column dropped)
    s, t = PR.scans[0], TISSUES["WM"]
    epg = epg_pathways(simulate_epg(s.alpha, t["T1"], t["T2"], s.TR, EPG_TR, EPG_K)[0],
                       PR.pathways)
    iso = steady_state_isochromat(s.alpha, t["T1"], t["T2"], s.TR, PR.pathways, n_iso=N_ISO)
    chk["epg_D0_vs_isochromat_WM"] = ((epg - iso).abs().max() / iso.abs().max()).item()
    J = jacobian(SCAN1, th["WM"], diffusion(2 * math.pi / VOXELS_MM[-1], 2 * EPG_TR, 2 * EPG_K))
    chk[f"diffusion_WM_{VOXELS_MM[-1]:g}mm_crlb_epg_{2 * EPG_TR}TR_K{2 * EPG_K}"] = \
        b1_bound(J)["crlb"]
    out["diffusion"] = {}
    for dx in VOXELS_MM:
        amp = diffusion(2 * math.pi / dx)
        res = {}
        for n, t in TISSUES.items():
            J = jacobian(SCAN1, theta(t, math.log(D_TISSUE)), amp)
            res[n] = {"D_known": b1_bound(J, drop=(7,)), "D_free": b1_bound(J),
                      "signal_change": change(th[n], amp)}
        out["diffusion"][f"voxel {dx:g} mm"] = res
    out["checks"] = chk
    return out


# ------------------------------------------------------------------ outputs
def cell(b):
    if b["rel_residual"] < DEGENERATE:
        return "exactly degenerate"
    m, e = f"{b['crlb']:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def row(effect, assumption, d):
    return f"{effect} & {assumption} & {cell(d['WM'])} & {cell(d['GM'])}\\\\"


def table(r):
    ideal = r["ideal"]
    rows = ["Ideal model & reference: both scans & "
            f"{ideal['WM']['two_scan']:.2f} & {ideal['GM']['two_scan']:.2f}\\\\",
            row("", "scan 1", {n: ideal[n]["scan1"] for n in TISSUES}), "\\midrule"]
    labels = {"gaussian": "Gaussian slice profile, $|z|\\le3\\sigma$",
              "hann_sinc": "Hann-sinc slice profile, TBW 4",
              "ramp": "Slab-edge ramp, $p\\in[0.9,1]$"}
    for name, label in labels.items():
        rows += [row(label, "shape known", r["profile_known"][name]),
                 row("", "width unknown", r["profile_width_free"][name])]
    rows.append("\\midrule")
    for i, tau in enumerate(TAUS):
        rows.append(row("Relaxation during the pulse, on resonance" * (i == 0),
                        f"$\\tau={tau:g}$\\,ms",
                        r["pulse_relaxation"][f"tau={tau:g} ms"]))
    for i, hz in enumerate(OFFRES_HZ):
        rows.append(row("Off-resonance during the pulse, $\\tau=1$\\,ms" * (i == 0),
                        f"$\\dw/2\\pi={hz:g}$\\,Hz", r["pulse_off_resonance"][f"{hz:g} Hz"]))
    rows.append("\\midrule")
    for i, dx in enumerate(VOXELS_MM):
        d = r["diffusion"][f"voxel {dx:g} mm"]
        for j, which in enumerate(("known", "free")):
            rows.append(row("Diffusion, $D=0.7\\times10^{-3}$\\,mm$^2$/s" * (i == j == 0),
                            f"$q=2\\pi/{dx:g}$\\,mm, $D$ {which}",
                            {n: d[n][f"D_{which}"] for n in TISSUES}))
    return rows


def report(r):
    print("\nchecks:", json.dumps(r["checks"], indent=1))
    print("\nWidth mismatch: true width = w × nominal, scan-1 fit with the nominal width (truth "
          f"B1 = 1, M0 = 1).\n'→ 0': the fit runs to B1 → 0 (stopped below B1 = {B1_FLOOR}); the "
          "residual is the floor reached there")
    print(f"{'profile':10s} {'tissue':6s} {'w':>6s} {'B1':>7s} {'T1 (ms)':>8s} {'M0':>6s} "
          f"{'rel. residual':>14s}")
    for name, rows in r["width_mismatch"].items():
        for v in rows:
            est = (f"{'→ 0':>7s} {'→ ∞':>8s} {'':>6s}" if v["to_zero"] else
                   f"{v['B1']:7.3f} {v['T1']:8.0f} {v['M0']:6.3f}")
            print(f"{name:10s} {v['tissue']:6s} {v['width']:6.4f} {est} {v['rel_residual']:14.1e}")
    print("\nSignal change vs the ideal model at the true parameters, best common phase removed "
          "(max |ΔS| / max |S|), WM / GM")
    for key in ("profile_known", "pulse_relaxation", "pulse_off_resonance", "diffusion"):
        for k, d in r[key].items():
            print(f"  {key:20s} {k:12s} {d['WM']['signal_change']:.1e} / "
                  f"{d['GM']['signal_change']:.1e}")


def main():
    t0 = time.time()
    r = {"settings": dict(n_iso=N_ISO, n_z=N_Z, epg_tr=EPG_TR, epg_K=EPG_K, D=D_TISSUE,
                          degenerate_below=DEGENERATE, tissues=TISSUES), **run()}
    r["runtime_s"] = time.time() - t0
    RES.mkdir(exist_ok=True)
    (RES / "beyond_ideal_model.json").write_text(json.dumps(r, indent=1))
    rows = table(r)
    (TAB / "beyond_model.tex").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    report(r)
    print(f"\nruntime {r['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
