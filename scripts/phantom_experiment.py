"""2D digital phantom: simulated MPME acquisition and reconstructions.

Scenario A — both scans at full resolution (the per-voxel setting of the analysis):
  analytic inverse (Cheng et al. 2019), complex ML, magnitude least squares + FID sign.
Scenario B — the published acquisition, scan 2 at low resolution (central 25 % of ky):
  paper pipeline: B1 from low-resolution data (analytic) → polynomial smoothing →
                  T1, T2, M0 from full-resolution scan 1 (Eqs. 1, 16, 17);
  ML pipeline:    B1 from low-resolution data (complex ML) → polynomial smoothing →
                  complex ML on full-resolution scan 1 with B1 fixed.
Also per-voxel uncertainty maps (local CRLB at the estimate) and their calibration.

    python scripts/phantom_experiment.py [--snr 1000 300]
    → results/phantom_snr<snr>.pt, results/phantom_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mpme.analytic import analytic_reconstruction
from mpme.crlb import voxel_uncertainty
from mpme.mle import fid_phase_sign, joint_fit, magnitude_fit
from mpme.paper import paper_reconstruction
from mpme.phantom import TISSUES, lowres_phase_encode, make_phantom, polyfit_2d, simulate
from mpme.sequence import Protocol, paper_protocol

RES = Path(__file__).resolve().parents[1] / "results"
B1_MAX = 540 / 330
FIT = dict(jacobian="implicit", n_iso=128, n_iter=40)
PARAMS = ("T1", "T2", "T2star", "B1", "M0")


CHUNK = 4096


def chunked(fn, n, chunk=CHUNK):
    """Apply fn(slice) over voxel chunks and concatenate the returned dicts (bounds memory)."""
    outs = [fn(slice(a, a + chunk)) for a in range(0, n, chunk)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def sub(d, sl):
    return {k: v[sl] for k, v in d.items()}


def best_of(fits):
    cost = torch.stack([f["cost"] for f in fits])
    pick = cost.argmin(0)
    return {k: torch.stack([f[k] for f in fits]).gather(0, pick[None])[0] for k in fits[0]}


def starts_from(fit, pap):
    starts = [fit]
    if pap is not None:
        starts.append({k: torch.where(torch.isfinite(pap[k]) & (pap[k] > 0), pap[k], fit[k])
                       for k in fit})
    for t1 in (700.0, 2000.0):
        starts.append({**fit, "T1": torch.full_like(fit["T1"], t1),
                       "T2star": torch.minimum(fit["T2star"], 0.85 * fit["T2"])})
    return starts


def sanitize(est):
    """Replace non-physical/NaN starting values by neutral ones (for use as ML starts)."""
    out = dict(est)
    for k, v in (("B1", 1.0), ("T1", 1000.0), ("T2", 70.0), ("M0", 1.0)):
        bad = ~torch.isfinite(out[k]) | (out[k] <= 0)
        out[k] = torch.where(bad, torch.full_like(out[k], v), out[k])
    out["T2star"] = torch.minimum(torch.where(torch.isfinite(out["T2star"]), out["T2star"],
                                              0.8 * out["T2"]), 0.95 * out["T2"])
    out["dw"] = torch.nan_to_num(out["dw"])
    return out


def roi_stats(est, truth, labels_vox):
    out = {}
    for lab, (name, *_) in TISSUES.items():
        sel = labels_vox == lab
        row = {}
        for k in PARAMS:
            if k not in est:
                continue
            v, t = est[k][sel].double(), truth[k][sel]
            ok = torch.isfinite(v) & (v > 0)
            e = (v[ok] / t[ok]).log()
            row[k] = {"median": e.median().item() if ok.any() else float("nan"),
                      "mean": e.mean().item() if ok.any() else float("nan"),
                      "sd": e.std().item() if ok.sum() > 1 else float("nan"),
                      "undefined": 1 - ok.float().mean().item()}
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[1000.0, 300.0])
    args = ap.parse_args()
    pr = paper_protocol()
    scan1 = Protocol(pr.pathways, (pr.scans[0],))
    ph = make_phantom()
    m = ph.mask
    truth = {"T1": ph.T1[m], "T2": ph.T2[m], "T2star": ph.T2star[m], "B1": ph.B1[m],
             "M0": ph.M0[m], "dw": ph.dw[m]}
    labels = ph.labels[m]
    summary = json.loads((RES / "phantom_summary.json").read_text()) \
        if (RES / "phantom_summary.json").exists() else {}

    for snr in args.snr:
        sigma = 1.0 / snr
        print(f"\n=== SNR(M0) = {snr:g}  (σ = {sigma:.4g}) ===")
        S0, Sn = simulate(ph, pr, sigma)
        Sv = Sn[m]                                           # [V, 2, 3, 3]
        times, recon = {}, {}

        # ---------------- Scenario A: full resolution
        t = time.time(); pap = paper_reconstruction(Sv, pr); times["A: analytic inverse"] = time.time() - t
        t = time.time(); init = sanitize(analytic_reconstruction(Sv, pr)); times["init (model fit)"] = time.time() - t
        st = starts_from(init, pap)
        t = time.time()
        V = Sv.shape[0]
        ml = best_of([chunked(lambda sl, s=s: joint_fit(Sv[sl], pr, sub(s, sl), b1_max=B1_MAX, **FIT), V)
                      for s in st])
        times["A: complex ML (4 starts)"] = time.time() - t
        sign = fid_phase_sign(Sv, pr)
        t = time.time()
        Mv = Sv.abs()
        mag = best_of([chunked(lambda sl, s=s: magnitude_fit(Mv[sl], pr, sub(s, sl), fid_sign=sign[sl],
                                                              b1_max=B1_MAX, **FIT), V)
                       for s in st])
        times["A: magnitude LS + FID sign (4 starts)"] = time.time() - t
        recon["A_paper"], recon["A_ml"], recon["A_mag"] = pap, ml, mag
        t = time.time()
        unc = chunked(lambda sl: voxel_uncertainty(sub(ml, sl), pr, sigma), V)
        times["A: uncertainty maps"] = time.time() - t
        recon["A_ml_sd"] = unc

        # ---------------- Scenario B: low-resolution scan 2
        S_low = Sn.clone()
        S_low[..., 0, :, :] = lowres_phase_encode(Sn[..., 0, :, :])
        S_low[..., 1, :, :] = lowres_phase_encode(Sn[..., 1, :, :])
        S_B = torch.stack([Sn[..., 0, :, :], S_low[..., 1, :, :]], -3)   # full scan 1, low scan 2
        Lv, Bv = S_low[m], S_B[m]
        # paper pipeline
        t = time.time()
        pap_low = paper_reconstruction(Lv, pr)
        B1_low = torch.full_like(ph.B1, float("nan")); B1_low[m] = pap_low["B1"]
        B1_s = polyfit_2d(B1_low, m)[m]
        papB = paper_reconstruction(Bv, pr, B1=B1_s)
        times["B: paper pipeline"] = time.time() - t
        papB["B1_low"] = pap_low["B1"]
        # ML pipeline
        t = time.time()
        init_low = sanitize(analytic_reconstruction(Lv, pr))
        ml_low = best_of([chunked(lambda sl, s=s: joint_fit(Lv[sl], pr, sub(s, sl), b1_max=B1_MAX, **FIT), V)
                          for s in starts_from(init_low, pap_low)])
        B1_lowML = torch.full_like(ph.B1, float("nan")); B1_lowML[m] = ml_low["B1"]
        B1_sML = polyfit_2d(B1_lowML, m)[m]
        S1 = Bv[:, :1].contiguous()
        st1 = [{**ml_low}] + [{**ml_low, "T1": torch.full_like(ml_low["T1"], t1)} for t1 in (700.0, 2000.0)]
        mlB = best_of([chunked(lambda sl, s=s: joint_fit(S1[sl], scan1, sub(s, sl),
                                                         fixed_B1=B1_sML[sl], **FIT), V)
                       for s in st1])
        times["B: ML pipeline"] = time.time() - t
        mlB["B1_low"] = ml_low["B1"]
        recon["B_paper"], recon["B_ml"] = papB, mlB
        recon["B_ml_sd"] = chunked(lambda sl: voxel_uncertainty(sub(mlB, sl), scan1, sigma,
                                                               fixed_B1=True), V)

        # ---------------- statistics
        stats = {name: roi_stats(est, truth, labels) for name, est in
                 (("A: analytic inverse", pap), ("A: complex ML", ml),
                  ("A: magnitude LS + FID sign", mag),
                  ("B: paper pipeline", papB), ("B: ML pipeline", mlB))}
        calib = {}
        for name, est, sd in (("A: complex ML", ml, unc), ("B: ML pipeline", mlB, recon["B_ml_sd"])):
            row = {}
            for k in ("T1", "T2"):
                z = (est[k].double() / truth[k]).log() / sd[k]
                z = z[torch.isfinite(z)]
                row[k] = {"sd_z": z.std().item(), "coverage95": (z.abs() < 1.96).float().mean().item()}
            calib[name] = row
        wrong_branch = {name: int(((est["B1"] * 330 - 360) * (truth["B1"] * 330 - 360) < 0).sum())
                        for name, est in (("A: complex ML", ml), ("A: magnitude LS + FID sign", mag),
                                          ("A: analytic inverse", pap))}
        summary[f"SNR {snr:g}"] = {"times_s": times, "roi": stats, "calibration": calib,
                                   "wrong_branch": wrong_branch, "voxels": int(m.sum())}
        for k, v in times.items():
            print(f"  {k:42s} {v:7.1f} s")
        for name in stats:
            wm, gm = stats[name]["WM"]["T1"], stats[name]["GM"]["T1"]
            print(f"  {name:28s} T1 WM median {wm['median']:+.3f} sd {wm['sd']:.3f} | "
                  f"GM median {gm['median']:+.3f} sd {gm['sd']:.3f} | undefined WM {wm['undefined']:.3f}")
        print("  calibration:", calib)
        print("  wrong branch:", wrong_branch)
        torch.save({"phantom": ph, "S0": S0, "Sn": Sn, "S_low": S_low, "recon": recon,
                    "sigma": sigma, "B1_smooth_paper": B1_s, "B1_smooth_ml": B1_sML},
                   RES / f"phantom_snr{snr:g}.pt")
    (RES / "phantom_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwritten {RES / 'phantom_summary.json'}")


if __name__ == "__main__":
    main()
