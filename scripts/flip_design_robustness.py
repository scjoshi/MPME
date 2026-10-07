"""Flip-angle design note, step 5: bias of the shortlisted designs under model error.

Noise-free data are generated with an effect the fitting model ignores and fitted with the
ideal two-scan model (mpme.mle.joint_fit, complex, all seven parameters free). Each fit is run
from ten starts (the truth, and B1 ∈ {0.75, 1, 1.25} × T1 ∈ {0.5, 1, 2} × truth) and the lowest
cost with B1 inside the fitting range [0.6, 1.4] is kept. Effects:

  ratio     scan-2 flip angle 1% larger than nominal relative to scan 1 (pulse-ratio error)
  ramp      slab-edge flip-angle spread, flip fraction 0.9…1 across the voxel
  pulse     hard pulses at a fixed peak B1 (duration α_nom/330° ms, so 330° lasts 1 ms) with
            relaxation during the pulse, on resonance; echo times measured from the pulse centre
  pulse_dw  the same with Δω/2π = 50 Hz (precession during the pulse)

Tissues WM, GM, CSF; B1 = 0.7, 0.8, …, 1.3, plus each design's B1 where the scan-2 angle is
330° and 345° (where finite-pulse and spread effects are largest), if inside [0.7, 1.3]. Reported: T1 bias 100 (T̂1/T1 − 1) %, the T1 bias
relative to the T1 SD at SNR(M0) = 300 (CRLB / 300), and the probability that a χ² test (99.9%,
29 degrees of freedom) flags the misfit at SNR 300 (noncentral χ², noncentrality = residual/σ²).
Also a sweep of the pulse duration (a 330° pulse lasting 0.125-4 ms at the same peak B1) for
two designs, WM at B1 = 1.

    PYTHONPATH=src:scripts python3 scripts/flip_design_robustness.py
    → results/flip_design_robustness.json, docs/flipnote/tables/robustness.tex
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from scipy.stats import chi2, ncx2

import beyond_ideal_model as bim
from mpme.design import bounds, ratio_bias, single_scan_fisher
from mpme.mle import joint_fit
from mpme.phantom import TISSUES
from mpme.sequence import Protocol, Scan, paper_protocol

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "docs" / "flipnote" / "tables"
PR = paper_protocol()
TR, ECHOES, PATHS = PR.scans[0].TR, PR.scans[0].echo_times, PR.pathways
TISSUE = {n: dict(M0=1.0, T1=t1, T2=t2, T2star=t2s, dw=0.05, phi0=0.3)
          for n, t1, t2, t2s, _ in TISSUES.values() if n in ("WM", "GM", "CSF")}
B1S = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
EFFECTS = ("ratio", "ramp", "pulse", "pulse_dw")
EPS = 0.01
SNR = 300.0
BOX = (0.6, 1.4)
DW_PULSE = 2 * math.pi * 50 / 1000          # rad/ms
THRESHOLD = chi2.ppf(0.999, 36 - 7)


def protocol(a1, a2, shift=(0.0, 0.0)):
    """Two-scan protocol; scan i's echo times are moved earlier by shift[i] (ms)."""
    sc = [Scan(a, TR, tuple(tuple(t - s for t in ts) for ts in ECHOES)) for a, s in zip((a1, a2), shift)]
    return Protocol(PATHS, tuple(sc))


def data(effect, a1, a2, tissue, b1, tau330=1.0):
    """Noise-free two-scan data [1, 2, P, J] with ``effect``."""
    t = {**tissue, "B1": b1}
    if effect == "pulse_dw":
        t["dw"] = DW_PULSE
    th = bim.theta(t)
    if effect == "ratio":
        return bim.signal(protocol(a1, a2 * (1 + EPS)), th, bim.ideal)[None]
    if effect == "ramp":
        return bim.signal(protocol(a1, a2), th, bim.profile("ramp"))[None]
    scans = []
    for i, a in enumerate((a1, a2)):           # duration set by the nominal angle (fixed peak B1)
        tau = tau330 * a / 330.0
        shift = [0.0, 0.0]
        shift[i] = tau / 2                     # amplitudes are at the end of the pulse; TE from its centre
        amp = bim.hard_pulse(tau, off_resonance=(effect == "pulse_dw"))
        p = protocol(a1, a2, tuple(shift))
        scans.append(bim.signal(Protocol(PATHS, (p.scans[i],)), th, amp)[0])
    return torch.stack(scans)[None]


def fit(S, a1, a2, tissue, b1, dw):
    """Best of ten starts with B1 inside BOX; flags whether the best fit sits on a safeguard."""
    starts = [(b1, 1.0)] + [(b, f) for b in (0.75, 1.0, 1.25) for f in (0.5, 1.0, 2.0)]
    n = len(starts)
    init = {"M0": torch.full((n,), tissue["M0"]),
            "B1": torch.tensor([b for b, _ in starts]),
            "T1": torch.tensor([tissue["T1"] * f for _, f in starts]),
            "T2": torch.full((n,), tissue["T2"]), "T2star": torch.full((n,), tissue["T2star"]),
            "dw": torch.full((n,), dw)}
    run = lambda sel: joint_fit(S.expand(len(sel), -1, -1, -1).contiguous(), protocol(a1, a2),
                                {q: v[sel] for q, v in init.items()}, n_iter=200, n_iso=256,
                                jacobian="implicit", b1_max=BOX[1])
    try:
        out = run(list(range(n)))
    except torch.linalg.LinAlgError:           # a start that drives the damped system singular:
        parts = []                             # fit the starts one by one and drop the failures
        for i in range(n):
            try:
                parts.append(run([i]))
            except torch.linalg.LinAlgError:
                pass
        out = {q: torch.cat([o[q] for o in parts]) for q in parts[0]}
    cost = torch.where((out["B1"] >= BOX[0]) & (out["B1"] <= BOX[1]), out["cost"],
                       torch.full_like(out["cost"], math.inf))
    k = int(torch.argmin(cost))
    return {**{q: float(out[q][k]) for q in ("T1", "B1", "M0", "T2", "cost")},
            "on_bound": bool(out["at_box"][k]) or abs(float(out["B1"][k]) - BOX[1]) < 1e-9,
            "r2p_zero": bool(out["at_r2p_bound"][k]),
            "in_range": bool(torch.isfinite(cost[k]))}


def crlb_t1(a1, a2, tissue, b1):
    F = single_scan_fisher([b1 * a1, b1 * a2], tissue, TR, ECHOES, PATHS)
    return float(bounds(F.sum(0))[2]), F


def b1_grid(a2):
    extra = [x / a2 for x in (330.0, 345.0) if B1S[0] <= x / a2 <= B1S[-1]]
    return sorted({*B1S, *(round(b, 4) for b in extra)})


def run(designs):
    out = {}
    for name, (a1, a2) in designs.items():
        rows = []
        for tn, tissue in TISSUE.items():
            for b1 in b1_grid(a2):
                sd, F = crlb_t1(a1, a2, tissue, b1)
                lin = ratio_bias(F.sum(0), F[1]) * EPS
                row = {"tissue": tn, "B1": b1, "crlb_T1": sd,
                       "ratio_linear": {"lnT1": float(lin[2]), "lnB1": float(lin[1]), "lnM0": float(lin[0])}}
                for effect in EFFECTS:
                    dw = DW_PULSE if effect == "pulse_dw" else tissue["dw"]
                    f = fit(data(effect, a1, a2, tissue, b1), a1, a2, tissue, b1, dw)
                    bias = f["T1"] / tissue["T1"] - 1
                    lam = f["cost"] * SNR**2
                    row[effect] = {"T1_pct": 100 * bias, "B1_pct": 100 * (f["B1"] / b1 - 1),
                                   "lnT1_over_sd": math.log(1 + bias) / (sd / SNR),
                                   "residual_sigma2": lam, "detect": float(ncx2.sf(THRESHOLD, 29, lam)),
                                   "on_bound": f["on_bound"], "r2p_zero": f["r2p_zero"],
                                   "in_range": f["in_range"]}
                rows.append(row)
        out[name] = {"angles": [a1, a2], "rows": rows}
        print(f"  {name} ({a1:g}/{a2:g}) done", flush=True)
    return out


def pulse_sweep(designs, taus=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0)):
    """T1 bias (%) of finite on-resonance pulses against the duration of a 330° pulse."""
    t = TISSUE["WM"]
    return {name: [{"tau330_ms": tau, "T1_pct": 100 * (fit(data("pulse", a1, a2, t, 1.0, tau), a1, a2,
                                                           t, 1.0, t["dw"])["T1"] / t["T1"] - 1)}
                   for tau in taus] for name, (a1, a2) in designs.items()}


def fmt(x):
    return "0.0" if abs(x) < 0.05 else f"{x:+.1f}"


def table(out):
    """Rows: design | effect | T1 bias (%) WM, GM, CSF at B1 = 1 | largest |T1 bias| (%) over the
    sampled cases | largest |ln T1 bias| / SD | detection probability of the largest-bias case."""
    labels = {"ratio": "pulse ratio $+1\\%$", "ramp": "slab-edge ramp", "pulse": "finite pulses",
              "pulse_dw": "finite pulses, 50\\,Hz"}
    lines = []
    for name, d in out.items():
        a1, a2 = d["angles"]
        for n, (eff, lab) in enumerate(labels.items()):
            at1 = {r["tissue"]: r[eff]["T1_pct"] for r in d["rows"] if r["B1"] == 1.0}
            worst = max(d["rows"], key=lambda r: abs(r[eff]["T1_pct"]))
            mz = max(abs(r[eff]["lnT1_over_sd"]) for r in d["rows"])
            head = f"{a1:g}/{a2:g}" if n == 0 else ""
            mark = "$^*$" if any(r[eff]["on_bound"] or not r[eff]["in_range"] for r in d["rows"]) else ""
            lines.append(f"{head} & {lab}{mark} & ${fmt(at1['WM'])}$ & ${fmt(at1['GM'])}$ & "
                         f"${fmt(at1['CSF'])}$ & {abs(worst[eff]['T1_pct']):.1f} & {mz:.1f} & "
                         f"{worst[eff]['detect']:.2f}\\\\")
        lines.append("\\midrule")
    return lines[:-1]


def main():
    t0 = time.time()
    designs = json.loads((RES / "flip_design.json").read_text())["shortlist"]
    angles = {k: tuple(v["angles"]) for k, v in designs.items()}
    out = run(angles)
    out["pulse_sweep"] = pulse_sweep({k: angles[k] for k in ("published", "margin")})
    out["runtime_s"] = time.time() - t0
    (RES / "flip_design_robustness.json").write_text(json.dumps(out, indent=1))
    TAB.mkdir(parents=True, exist_ok=True)
    rows = table({k: v for k, v in out.items() if k not in ("runtime_s", "pulse_sweep")})
    (TAB / "robustness.tex").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print("pulse sweep:", json.dumps(out["pulse_sweep"], indent=1))
    print(f"runtime {out['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
