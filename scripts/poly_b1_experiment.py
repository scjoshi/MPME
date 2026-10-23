"""Joint polynomial-B1 ML vs two-stage smoothing on the digital phantom, 15°/85°.

Both scans at full resolution (2D phantom of phantom_experiment.py, same seeds), SNR(M0) 1000
and 300. Compared:
  - per-voxel complex ML (4 starts, as in phantom_experiment.py);
  - two-stage: robust degree-4 polynomial fit of the per-voxel B1 map (polyfit_2d), then
    complex ML with B1 fixed, on scan 1 only (paper style) and on both scans;
  - joint ML with log B1 a degree-4 polynomial shared by all voxels (mle.polynomial_b1_fit),
    started from the two-stage solution.

    PYTHONPATH=src:scripts python3 scripts/poly_b1_experiment.py
    → results/poly_b1.json, docs/technote/tables/poly_b1.tex
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from mpme.analytic import analytic_reconstruction
from mpme.mle import joint_fit, polynomial_b1_fit, polynomial_basis
from mpme.phantom import make_phantom, polyfit_2d, simulate
from mpme.sequence import Protocol, Scan, paper_protocol
from phantom_experiment import FIT, best_of, chunked, roi_stats, sanitize, starts_from, sub

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "docs" / "technote" / "tables"
ALPHA2 = 85.0
TISSUES_SHOWN = ("WM", "GM", "CSF", "lesion")


def summarise(est, B1, truth, labels):
    e = (B1 / truth["B1"]).log()
    r = roi_stats({**est, "B1": B1}, truth, labels)
    return {"B1_rms": e.pow(2).mean().sqrt().item(), "B1_max": e.abs().max().item(),
            "T1": {t: {"median": r[t]["T1"]["median"], "sd": r[t]["T1"]["sd"]} for t in TISSUES_SHOWN}}


def main():
    ph = make_phantom()
    m = ph.mask
    V, H = int(m.sum()), m.shape[0]
    truth = {"T1": ph.T1[m], "T2": ph.T2[m], "T2star": ph.T2star[m], "B1": ph.B1[m], "M0": ph.M0[m]}
    c = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) / (H / 2)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    Phi = polynomial_basis(torch.stack([xx[m], yy[m]], 1))
    base = paper_protocol()
    s1 = base.scans[0]
    pr = Protocol(base.pathways, (s1, Scan(ALPHA2, s1.TR, s1.echo_times)))
    scan1 = Protocol(pr.pathways, (s1,))
    out = {"alpha": [s1.flip_deg, ALPHA2], "voxels": V}
    for snr in (1000.0, 300.0):
        print(f"=== SNR {snr:g}", flush=True)
        _, Sn = simulate(ph, pr, 1 / snr)
        Sv = Sn[m]
        row, times = {}, {}
        t = time.time()
        init = sanitize(analytic_reconstruction(Sv, pr))
        vox = best_of([chunked(lambda sl, s=s: joint_fit(Sv[sl], pr, sub(s, sl), **FIT), V)
                       for s in starts_from(init, None)])
        times["per-voxel ML"] = time.time() - t
        row["per-voxel ML"] = summarise(vox, vox["B1"], truth, ph.labels[m])
        t = time.time()
        Bm = torch.full_like(ph.B1, float("nan"))
        Bm[m] = vox["B1"]
        Bs = polyfit_2d(Bm, m)[m]
        st = [{**vox}] + [{**vox, "T1": torch.full_like(vox["T1"], t1)} for t1 in (700.0, 2000.0)]
        S1 = Sv[:, :1].contiguous()
        two1 = best_of([chunked(lambda sl, s=s: joint_fit(S1[sl], scan1, sub(s, sl), fixed_B1=Bs[sl], **FIT), V)
                        for s in st])
        times["two-stage, scan 1"] = time.time() - t
        row["two-stage, scan 1"] = summarise(two1, Bs, truth, ph.labels[m])
        t = time.time()
        two2 = best_of([chunked(lambda sl, s=s: joint_fit(Sv[sl], pr, sub(s, sl), fixed_B1=Bs[sl], **FIT), V)
                        for s in st])
        times["two-stage, both scans"] = time.time() - t
        row["two-stage, both scans"] = summarise(two2, Bs, truth, ph.labels[m])
        Lm = torch.full_like(ph.B1, float("nan"))
        Lm[m] = vox["B1"].log()
        coef0 = torch.linalg.lstsq(Phi, polyfit_2d(Lm, m)[m][:, None]).solution[:, 0]
        k = vox["B1"] / (Phi @ coef0).exp()
        t = time.time()
        jt = polynomial_b1_fit(Sv, pr, Phi, {**vox, "T1": vox["T1"] * k**2, "M0": vox["M0"] * k}, coef0)
        times["joint polynomial-B1 ML"] = time.time() - t
        row["joint polynomial-B1 ML"] = summarise(jt, jt["B1"].double(), truth, ph.labels[m])
        row["joint_history"] = jt["history"]
        row["joint_downweighted"] = int((jt["weight"] < 1).sum())
        row["times_s"] = times
        out[f"SNR {snr:g}"] = row
        for k_, v in row.items():
            if isinstance(v, dict) and "B1_rms" in v:
                print(f"  {k_:<26} B1 rms {100 * v['B1_rms']:.3f}% max {100 * v['B1_max']:.3f}% | " +
                      "  ".join(f"{t_} {100 * v['T1'][t_]['median']:+.2f}/{100 * v['T1'][t_]['sd']:.2f}"
                                for t_ in TISSUES_SHOWN), flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "poly_b1.json").write_text(json.dumps(out, indent=1))
    methods = ("per-voxel ML", "two-stage, scan 1", "two-stage, both scans", "joint polynomial-B1 ML")
    lines = []
    for snr in (1000, 300):
        row = out[f"SNR {snr}"]
        lines.append(rf"\emph{{SNR$(\Mzero)={snr}$}} &&&&&&\\")
        for meth in methods:
            v = row[meth]
            t1 = " & ".join(f"${100 * v['T1'][t]['median']:+.1f}$ / {100 * v['T1'][t]['sd']:.1f}" for t in TISSUES_SHOWN)
            lines.append(f"{meth} & {100 * v['B1_rms']:.2f} & {100 * v['B1_max']:.2f} & {t1}\\\\")
        if snr == 1000:
            lines.append(r"\midrule")
    (TAB / "poly_b1.tex").write_text("\n".join(lines) + "\n")
    print("written", RES / "poly_b1.json", TAB / "poly_b1.tex")


if __name__ == "__main__":
    main()
