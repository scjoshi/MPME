"""Partial volume: the BrainWeb phantom reconstructed with the unchanged estimators.

Cases (both scans at full resolution, the paper protocol):
  pv_clean  partial volume, noise-free     → model-error bias and mismatch residuals
  pv        partial volume, SNR(M0) = 1000 → bias + noise, uncertainty-map coverage
  crisp     same anatomy, each voxel assigned to its dominant tissue, SNR 1000 (control)

Estimators: analytic inverse, complex ML (4 starts + spatial restart), magnitude LS with the
FID sign (4 starts + spatial restart). Truth for a mixed voxel: the PD-weighted geometric
mean of the compartment values, exp(Σ w_c log θ_c), with w_c = f_c PD_c / Σ f PD.

    python scripts/brainweb_experiment.py --case pv_clean|pv|crisp
    → results/brainweb_<case>.pt, results/brainweb_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from scipy.stats import chi2

from mpme.analytic import analytic_reconstruction
from mpme.brainweb import make_brainweb_phantom, simulate_pv
from mpme.crlb import voxel_uncertainty
from mpme.mle import fid_phase_sign, joint_fit, magnitude_fit
from mpme.paper import paper_reconstruction
from mpme.phantom import polyfit_2d
from mpme.sequence import paper_protocol

from phantom_experiment import (B1_MAX, CHUNK, FIT, best_of, chunked, sanitize, spatial_restart,
                                starts_from, sub)

RES = Path(__file__).resolve().parents[1] / "results"
SNR = 1000.0
CATEGORIES = ("pure CSF", "pure GM", "pure WM", "WM/GM mixed",
              "CSF 5-25%", "CSF 25-50%", "CSF 50-95%")


def categories(F):
    """Voxel categories from fractions F [3, V] (CSF, GM, WM)."""
    csf, gm, wm = F
    mx = F.max(0).values
    pure = mx >= 0.95
    return {"pure CSF": pure & (csf >= 0.95), "pure GM": pure & (gm >= 0.95),
            "pure WM": pure & (wm >= 0.95),
            "WM/GM mixed": ~pure & (csf < 0.05),
            "CSF 5-25%": (csf >= 0.05) & (csf < 0.25), "CSF 25-50%": (csf >= 0.25) & (csf < 0.5),
            "CSF 50-95%": (csf >= 0.5) & (csf < 0.95)}


def fit_all(Sv, pr, V, mask, shape):
    """Analytic inverse, complex ML and magnitude LS (+FID), each with spatial restart."""
    times = {}
    t = time.time(); pap = paper_reconstruction(Sv, pr); times["analytic inverse"] = time.time() - t
    init = sanitize(analytic_reconstruction(Sv, pr))
    st = starts_from(init, pap)
    t = time.time()
    ml = best_of([chunked(lambda sl, s=s: joint_fit(Sv[sl], pr, sub(s, sl), b1_max=B1_MAX, **FIT), V)
                  for s in st])
    B1map = torch.full(shape, float("nan"), dtype=torch.float64); B1map[mask] = ml["B1"]
    B1s = polyfit_2d(B1map, mask)[mask].clamp(0.5, B1_MAX)
    rs = spatial_restart(ml, B1s)
    ml2 = best_of([chunked(lambda sl: joint_fit(Sv[sl], pr, sub(rs, sl), b1_max=B1_MAX, **FIT), V)])
    imp = ml2["cost"] < ml["cost"] * (1 - 1e-6)
    ml = {k: torch.where(imp, ml2[k], v) if v.shape == imp.shape else v for k, v in ml.items()}
    times["complex ML"] = time.time() - t
    t = time.time()
    Mv, sign = Sv.abs(), fid_phase_sign(Sv, pr)
    mag = best_of([chunked(lambda sl, s=s: magnitude_fit(Mv[sl], pr, sub(s, sl), fid_sign=sign[sl],
                                                          b1_max=B1_MAX, **FIT), V) for s in st])
    rsm = spatial_restart(mag, B1s)
    mag2 = best_of([chunked(lambda sl: magnitude_fit(Mv[sl], pr, sub(rsm, sl), fid_sign=sign[sl],
                                                      b1_max=B1_MAX, **FIT), V)])
    imp = mag2["cost"] < mag["cost"] * (1 - 1e-6)
    mag = {k: torch.where(imp, mag2[k], v) if v.shape == imp.shape else v for k, v in mag.items()}
    times["magnitude LS + FID"] = time.time() - t
    return {"analytic inverse": pap, "complex ML": ml, "magnitude LS + FID": mag}, times


def stats(est, truth, cats):
    out = {}
    for cname, sel in cats.items():
        row = {"voxels": int(sel.sum())}
        for k in ("T1", "T2", "T2star", "B1"):
            v, t = est[k][sel].double(), truth[k][sel]
            ok = torch.isfinite(v) & (v > 0)
            e = (v[ok] / t[ok]).log()
            row[k] = {"median": e.median().item() if ok.any() else float("nan"),
                      "sd": e.std().item() if ok.sum() > 1 else float("nan"),
                      "undefined": 1 - ok.float().mean().item()}
        out[cname] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("pv_clean", "pv", "crisp"), required=True)
    args = ap.parse_args()
    pr = paper_protocol()
    ph = make_brainweb_phantom()
    m = ph.mask
    if args.case == "crisp":                         # dominant tissue only, same anatomy
        dom = ph.fractions.argmax(0)
        ph.fractions = torch.stack([(dom == c).double() for c in range(3)])
    sigma = 0.0 if args.case == "pv_clean" else 1.0 / SNR
    S0, Sn = simulate_pv(ph, pr, sigma)
    Sv = Sn[m]
    V = Sv.shape[0]
    F = ph.fractions[:, m]
    truth = {"T1": ph.effective("T1")[m], "T2": ph.effective("T2")[m],
             "T2star": ph.effective("T2star")[m], "B1": ph.B1[m]}
    cats = categories(ph.fractions[:, m] if args.case != "crisp" else
                      make_brainweb_phantom().fractions[:, m])   # categorise by true anatomy
    print(f"case {args.case}: {V} voxels, σ = {sigma:g}", flush=True)
    est, times = fit_all(Sv, pr, V, m, m.shape)
    for k, v in times.items():
        print(f"  {k:22s} {v:6.1f} s", flush=True)
    summary = {"times_s": times, "stats": {n: stats(e, truth, cats) for n, e in est.items()},
               "voxels": V}
    # mismatch residual and χ² detectability (complex ML)
    cost = est["complex ML"]["cost"].double()
    dof = 36 - 7
    if args.case == "pv_clean":
        thr = chi2.ppf(0.999, dof)
        from scipy.stats import ncx2
        det = {}
        for snr in (300.0, 1000.0, 3000.0):
            lam = (cost * snr**2).numpy()
            p = ncx2.sf(thr, dof, lam)
            det[f"SNR {snr:g}"] = {c: float(p[sel.numpy()].mean()) for c, sel in cats.items()}
        summary["detection_power_chi2_99.9"] = det
        summary["mismatch_rms_over_sigma1000"] = {
            c: float((cost[sel] / 36).sqrt().median() / 1e-3) for c, sel in cats.items()}
    else:
        unc = chunked(lambda sl: voxel_uncertainty(sub(est["complex ML"], sl), pr, sigma), V)
        cov = {}
        for c, sel in cats.items():
            z = (est["complex ML"]["T1"][sel].double() / truth["T1"][sel]).log() / unc["T1"][sel]
            z = z[torch.isfinite(z)]
            cov[c] = {"coverage95": (z.abs() < 1.96).float().mean().item(), "sd_z": z.std().item()}
        summary["coverage_T1"] = cov
        thr = chi2.ppf(0.999, dof) * sigma**2
        summary["flagged_chi2_99.9"] = {c: float((cost[sel] > thr).float().mean()) for c, sel in cats.items()}
        est["complex ML"]["sd"] = unc
    for n in est:
        s = summary["stats"][n]
        print(f"  {n:20s} T1 median: " + "  ".join(f"{c} {s[c]['T1']['median']:+.3f}" for c in CATEGORIES))
    print("  extra:", {k: v for k, v in summary.items() if k not in ("stats", "times_s")})
    path = RES / "brainweb_summary.json"
    allsum = json.loads(path.read_text()) if path.exists() else {}
    allsum[args.case] = summary
    path.write_text(json.dumps(allsum, indent=1))
    torch.save({"phantom": ph, "S0": S0, "Sn": Sn, "est": est, "truth": truth, "sigma": sigma},
               RES / f"brainweb_{args.case}.pt")
    print(f"written {path}")


if __name__ == "__main__":
    main()
