"""Monte Carlo: each estimator's error distribution vs the Cramér–Rao bound.

Estimators: the paper's analytic inverse (B1 solved, and B1 given), the model-fit baseline,
and the joint maximum-likelihood fit (projected LM, four starts, lowest cost kept).
Paper reference tissue, B1 = 1, the paper protocol (15°/330°) and a 15°/30° control.

Per estimator and parameter, on log θ̂ − log θ: standard deviation, mean (bias), median,
RMSE, robust spread (IQR/1.349) — all over voxels with a valid estimate — plus the fraction
of undefined estimates and of estimates on a constraint/bound. For ML, also the number of
voxels whose final cost exceeds the cost at the true parameters (optimiser failure; a
diagnostic only available in simulation).

    python scripts/estimator_efficiency.py
    → results/estimator_efficiency.json, results/estimator_efficiency_raw.pt
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from mpme.analytic import analytic_reconstruction
from mpme.crlb import crlb, fisher_information
from mpme.mle import _model, joint_fit
from mpme.paper import paper_reconstruction
from mpme.sequence import Protocol, Scan, paper_protocol
from mpme.signal import add_noise, mpme_signal

RES = Path(__file__).resolve().parents[1] / "results"
TIS = dict(M0=1.0, B1=1.0, T1=1500.0, T2=70.0, T2star=60.0, dw=0.05, phi0=0.3)
N = 1000
KEYS = ("T1", "B1", "T2", "M0")

# Model-fit search limits (analytic.estimate_b1_t1 defaults, ±0.5 in log) and T2 clamp.
FIT_LIMITS = {"B1": (0.4 * math.exp(-0.5), 1.6 * math.exp(0.5)),
              "T1": (100.0 * math.exp(-0.5), 6000.0 * math.exp(0.5)),
              "T2": (1.0, 5000.0)}


def at_limits(est, limits):
    hit = torch.zeros(N, dtype=torch.bool)
    for k, (lo, hi) in limits.items():
        v = est[k]
        hit |= torch.isfinite(v) & ((v <= lo * (1 + 1e-6)) | (v >= hi * (1 - 1e-6)))
    return hit


def stats(est, key, boundary=None):
    v = est[key]
    ok = torch.isfinite(v) & (v > 0)
    e = (v[ok] / TIS[key]).log()
    q1, q3 = e.quantile(0.25).item(), e.quantile(0.75).item()
    return {"sd": e.std().item(), "mean": e.mean().item(), "median": e.median().item(),
            "rmse": e.pow(2).mean().sqrt().item(), "robust_sd": (q3 - q1) / 1.349,
            "undefined_frac": 1 - ok.float().mean().item(),
            "boundary_frac": None if boundary is None else boundary.float().mean().item()}


def ml_multistart(S, p, starts):
    fits = [joint_fit(S, p, s) for s in starts]
    cost = torch.stack([f["cost"] for f in fits])
    pick = cost.argmin(0)
    return {k: torch.stack([f[k] for f in fits]).gather(0, pick[None])[0] for k in fits[0]}


def main():
    pr = paper_protocol()
    s1, s2 = pr.scans
    pr30 = Protocol(pr.pathways, (s1, Scan(30.0, s2.TR, s2.echo_times)))
    results, raw = {}, {}
    th_true = torch.tensor([0.0, 0.0, math.log(TIS["T1"]), math.log(TIS["T2"]),
                            math.log(TIS["T2star"]), TIS["dw"], TIS["phi0"]], dtype=torch.float64)
    for pname, p, snrs in (("15/330", pr, (300, 1000, 3000)), ("15/30", pr30, (1000, 3000))):
        F2 = fisher_information(p, TIS)
        F1 = fisher_information(p, TIS, scan_sigma=(1.0, 1e8))
        bounds = {"all unknown": crlb(F2), "B1 known": crlb(F2, known=("B1",)),
                  "scan 1, B1 known": crlb(F1, known=("B1",))}
        S = mpme_signal(p, TIS["M0"], TIS["T1"], TIS["T2"], 1 / TIS["T2star"] - 1 / TIS["T2"],
                        TIS["dw"], TIS["B1"], TIS["phi0"]).expand(N, 2, 3, 3).clone()
        for snr in snrs:
            case = f"{pname} SNR {snr}"
            Sn = add_noise(S, 1 / snr, torch.Generator().manual_seed(snr))
            y = torch.view_as_real(Sn).flatten(1)
            cost_true = ((_model(th_true.expand(N, 7), p, 256) - y) ** 2).sum(1)
            row = {"CRLB": {name: {k: b[k] / snr for k in KEYS if k in b}
                            for name, b in bounds.items()}}

            fit = analytic_reconstruction(Sn, p)
            row["model-fit"] = {k: stats(fit, k, at_limits(fit, FIT_LIMITS)) for k in KEYS}
            starts = [fit]
            if pname == "15/330":
                pap = paper_reconstruction(Sn, p)
                row["paper"] = {k: stats(pap, k, at_limits(pap, {"T2": (1.0, 5000.0)}))
                                for k in KEYS}
                pk = paper_reconstruction(Sn, p, B1=torch.tensor(1.0, dtype=torch.float64))
                row["paper, B1 known"] = {k: stats(pk, k) for k in ("T1", "M0")}
                starts.append({k: torch.where(torch.isfinite(pap[k]) & (pap[k] > 0), pap[k], fit[k])
                               for k in fit})
                raw[case] = {"paper": pap, "paper, B1 known": pk}
            else:
                starts.append({**fit, "B1": fit["B1"] * 1.1, "T1": fit["T1"] / 1.21,
                               "M0": fit["M0"] / 1.1})
                raw[case] = {}
            for t1 in (700.0, 2000.0):
                starts.append({**fit, "T1": torch.full_like(fit["T1"], t1),
                               "T2star": torch.minimum(fit["T2star"], 0.85 * fit["T2"])})
            ml = ml_multistart(Sn, p, starts)
            bnd = ml["at_box"] | ml["at_r2p_bound"]
            row["joint ML"] = {k: stats(ml, k, bnd) for k in KEYS}
            row["joint ML diagnostics"] = {
                "r2p_bound_frac": ml["at_r2p_bound"].float().mean().item(),
                "box_frac": ml["at_box"].float().mean().item(),
                "cost_above_truth": int((ml["cost"] > cost_true + 1e-12).sum())}
            raw[case].update({"model-fit": fit, "joint ML": ml, "cost_true": cost_true})
            results[case] = row

            print(f"\n{case}   CRLB sd: " + "  ".join(
                f"{k} {row['CRLB']['all unknown'][k]:.4f}" for k in KEYS))
            for est in ("joint ML", "model-fit", "paper", "paper, B1 known"):
                if est not in row:
                    continue
                d = row[est]
                print(f"  {est:16s}" + "  ".join(
                    f"{k}: sd {d[k]['sd']:.4f} rob {d[k]['robust_sd']:.4f} mean {d[k]['mean']:+.4f}"
                    for k in ("T1", "B1") if k in d) +
                    f"  undef {d['T1']['undefined_frac']:.3f}"
                    + (f" bound {d['T1']['boundary_frac']:.3f}" if d['T1']['boundary_frac'] is not None else ""))
            print("  ML diagnostics:", row["joint ML diagnostics"])

    RES.mkdir(exist_ok=True)
    (RES / "estimator_efficiency.json").write_text(json.dumps(results, indent=1))
    torch.save(raw, RES / "estimator_efficiency_raw.pt")
    print(f"\nwritten {RES / 'estimator_efficiency.json'}")


if __name__ == "__main__":
    main()
