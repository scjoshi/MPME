"""Paper's analytic inverse vs Gaussian least squares on magnitude images (with and without
the FID-sign branch constraint).

Uses exactly the noise realisations of scripts/estimator_efficiency.py (same seeds) and its
saved results for the analytic inverse and complex ML, so all estimators see the same data.
Bounds: complex CRLB and the exact Rician (magnitude-data) CRLB.

    python scripts/magnitude_comparison.py   → results/magnitude_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mpme.analytic import analytic_reconstruction
from mpme.crlb import MAG_PARAMS, crlb, fisher_information, fisher_information_magnitude
from mpme.mle import fid_phase_sign, magnitude_fit
from mpme.paper import paper_reconstruction
from mpme.sequence import paper_protocol
from mpme.signal import add_noise, mpme_signal

RES = Path(__file__).resolve().parents[1] / "results"
TIS = dict(M0=1.0, B1=1.0, T1=1500.0, T2=70.0, T2star=60.0, dw=0.05, phi0=0.3)
N = 1000
KEYS = ("T1", "B1", "T2", "M0")


def stats(est, key):
    v = est[key]
    ok = torch.isfinite(v) & (v > 0)
    e = (v[ok] / TIS[key]).log()
    q1, q3 = e.quantile(0.25).item(), e.quantile(0.75).item()
    return {"sd": e.std().item(), "mean": e.mean().item(), "rmse": e.pow(2).mean().sqrt().item(),
            "robust_sd": (q3 - q1) / 1.349, "undefined_frac": 1 - ok.float().mean().item()}


def main():
    pr = paper_protocol()
    saved = json.loads((RES / "estimator_efficiency.json").read_text())
    S = mpme_signal(pr, TIS["M0"], TIS["T1"], TIS["T2"], 1 / TIS["T2star"] - 1 / TIS["T2"],
                    TIS["dw"], TIS["B1"], TIS["phi0"]).expand(N, 2, 3, 3).clone()
    A = S[0].abs()
    out = {}
    for snr in (300, 1000, 3000):
        Sn = add_noise(S, 1 / snr, torch.Generator().manual_seed(snr))     # same as before
        M = Sn.abs()
        # Rician floor: mean magnitude minus true amplitude, relative, per measurement
        floor = ((M.mean(0) - A) / A)
        fit = analytic_reconstruction(Sn, pr)
        pap = paper_reconstruction(Sn, pr)
        starts = [fit, {k: torch.where(torch.isfinite(pap[k]) & (pap[k] > 0), pap[k], fit[k])
                        for k in fit}]
        for t1 in (700.0, 2000.0):
            starts.append({**fit, "T1": torch.full_like(fit["T1"], t1),
                           "T2star": torch.minimum(fit["T2star"], 0.85 * fit["T2"])})
        def best(fid_sign=None):
            fits = [magnitude_fit(M, pr, s, fid_sign=fid_sign) for s in starts]
            cost = torch.stack([f["cost"] for f in fits])
            pick = cost.argmin(0)
            return {k: torch.stack([f[k] for f in fits]).gather(0, pick[None])[0] for k in fits[0]}

        mag = best()
        magb = best(fid_phase_sign(Sn, pr))

        bc = crlb(fisher_information(pr, TIS))
        br = crlb(fisher_information_magnitude(pr, TIS, snr), MAG_PARAMS)
        row = {
            "CRLB complex": {k: bc[k] / snr for k in KEYS},
            "CRLB magnitude (Rician)": {k: br[k] for k in KEYS},
            "paper": saved[f"15/330 SNR {snr}"]["paper"],
            "joint ML (complex)": saved[f"15/330 SNR {snr}"]["joint ML"],
            "magnitude LS": {k: stats(mag, k) for k in KEYS},
            "magnitude LS diagnostics": {
                "r2p_bound_frac": mag["at_r2p_bound"].float().mean().item(),
                "box_frac": mag["at_box"].float().mean().item(),
                "wrong_branch": int(((mag["B1"] * 330 - 360) * (TIS["B1"] * 330 - 360) < 0).sum())},
            "magnitude LS + FID sign": {k: stats(magb, k) for k in KEYS},
            "magnitude LS + FID sign diagnostics": {
                "r2p_bound_frac": magb["at_r2p_bound"].float().mean().item(),
                "branch_bound_frac": magb["at_branch_bound"].float().mean().item(),
                "wrong_branch": int(((magb["B1"] * 330 - 360) * (TIS["B1"] * 330 - 360) < 0).sum())},
            "rician floor max rel": floor.max().item(),
        }
        out[f"SNR {snr}"] = row
        print(f"\nSNR(M0) = {snr}   max Rician floor (E|y|-A)/A = {floor.max():.3f}")
        print(f"  CRLB complex / Rician: " + "  ".join(
            f"{k} {row['CRLB complex'][k]:.4f}/{row['CRLB magnitude (Rician)'][k]:.4f}" for k in KEYS))
        for est in ("paper", "magnitude LS", "magnitude LS + FID sign", "joint ML (complex)"):
            d = row[est]
            print(f"  {est:20s}" + "  ".join(
                f"{k}: sd {d[k]['sd']:.4f} mean {d[k]['mean']:+.4f} rmse {d[k]['rmse']:.4f}"
                for k in ("T1", "B1", "T2")))
        print("  magnitude LS diagnostics:", row["magnitude LS diagnostics"])
        print("  + FID sign diagnostics:  ", row["magnitude LS + FID sign diagnostics"])
    (RES / "magnitude_comparison.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten {RES / 'magnitude_comparison.json'}")


if __name__ == "__main__":
    main()
