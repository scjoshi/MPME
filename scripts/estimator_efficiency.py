"""Monte Carlo: spread of each estimator vs the Cramér–Rao bound.

Estimators: the paper's analytic chain (B1 solved, and B1 known), the model-fit baseline,
and the joint maximum-likelihood fit (best of two starts). Paper reference tissue,
B1 = 1, both the paper protocol (15°/330°) and a 15°/30° control.

    python scripts/estimator_efficiency.py      # writes results/estimator_efficiency.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from mpme.analytic import analytic_reconstruction
from mpme.crlb import crlb, fisher_information
from mpme.mle import joint_fit
from mpme.paper import paper_reconstruction
from mpme.sequence import Protocol, Scan, paper_protocol
from mpme.signal import add_noise, mpme_signal

OUT = Path(__file__).resolve().parents[1] / "results" / "estimator_efficiency.json"
TIS = dict(M0=1.0, B1=1.0, T1=1500.0, T2=70.0, T2star=60.0, dw=0.05, phi0=0.3)
N = 1000
KEYS = ("T1", "B1", "T2", "M0")


def stats(est: dict, key: str) -> dict:
    v = est[key]
    ok = torch.isfinite(v) & (v > 0)
    lr = (v[ok] / TIS[key]).log()
    q1, q3 = lr.quantile(0.25).item(), lr.quantile(0.75).item()
    return {"robust_sd": (q3 - q1) / 1.349, "sd": lr.std().item(), "bias": lr.median().item(),
            "nan_frac": 1 - ok.float().mean().item()}


def best_of(S, pr, starts):
    fits = [joint_fit(S, pr, s) for s in starts]
    cost = torch.stack([f["cost"] for f in fits])
    pick = cost.argmin(0)
    return {k: torch.stack([f[k] for f in fits]).gather(0, pick[None])[0] for k in fits[0]}


def main():
    pr = paper_protocol()
    s1, s2 = pr.scans
    pr30 = Protocol(pr.pathways, (s1, Scan(30.0, s2.TR, s2.echo_times)))
    results = {}
    for pname, p, snrs in (("15/330", pr, (300, 1000, 3000)), ("15/30", pr30, (1000, 3000))):
        bound = crlb(fisher_information(p, TIS))
        S = mpme_signal(p, TIS["M0"], TIS["T1"], TIS["T2"], 1 / TIS["T2star"] - 1 / TIS["T2"],
                        TIS["dw"], TIS["B1"], TIS["phi0"]).expand(N, 2, 3, 3).clone()
        for snr in snrs:
            Sn = add_noise(S, 1 / snr, torch.Generator().manual_seed(snr))
            row = {"CRLB": {k: {"robust_sd": bound[k] / snr} for k in KEYS}}
            fit = analytic_reconstruction(Sn, p)
            row["model-fit"] = {k: stats(fit, k) for k in KEYS}
            starts = [fit]
            if pname == "15/330":
                pap = paper_reconstruction(Sn, p)
                row["paper"] = {k: stats(pap, k) for k in KEYS}
                pk = paper_reconstruction(Sn, p, B1=torch.tensor(1.0, dtype=torch.float64))
                row["paper, B1 known"] = {k: stats(pk, k) for k in ("T1", "M0")}
                alt = {k: torch.where(torch.isfinite(pap[k]) & (pap[k] > 0), pap[k], fit[k])
                       for k in fit}
                starts.append(alt)
            else:
                starts.append({**fit, "B1": fit["B1"] * 1.1, "T1": fit["T1"] / 1.21,
                               "M0": fit["M0"] / 1.1})
            ml = best_of(Sn, p, starts)
            row["joint ML"] = {k: stats(ml, k) for k in KEYS}
            results[f"{pname} SNR {snr}"] = row
            print(f"\n{pname}, SNR(M0) = {snr}")
            for est, d in row.items():
                print(f"  {est:16s} " + "  ".join(
                    f"{k} {100 * d[k]['robust_sd']:7.2f}%" for k in KEYS if k in d) +
                    (f"   T1 NaN {100 * d['T1']['nan_frac']:.1f}%" if "nan_frac" in d.get("T1", {}) else ""))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
