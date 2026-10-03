"""Why two scans: single-scan degeneracy demonstration and second-scan design bounds.

1. Single scan, smooth B1 fields: scan-1 data (noise-free, synthetic phantom) refitted with
   wrong B1 fields held fixed — residual, induced T1 error and d log T1 / d log B1.
2. Profiled Fisher information of polynomial B1 coefficients from one scan (should be ~0).
3. Second-scan design: B1 bound over B1 in [0.8, 1.2] vs alpha1, alpha2 and readout; stage-2
   T1 bound vs alpha1 (scan 1 alone, B1 known).

    python scripts/two_scan_design.py   → results/two_scan_design.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from mpme.crlb import crlb, fisher_information
from mpme.mle import joint_fit
from mpme.phantom import make_phantom, simulate
from mpme.sequence import Protocol, Scan, mpme_echo_times, paper_protocol

RES = Path(__file__).resolve().parents[1] / "results"


def degeneracy_demo():
    pr = paper_protocol()
    scan1 = Protocol(pr.pathways, (pr.scans[0],))
    ph = make_phantom()
    m = ph.mask
    idx = torch.nonzero(m.flatten())[::8, 0]
    S0, _ = simulate(ph, scan1, 0.0)
    S = S0.reshape(-1, 1, 3, 3)[idx].contiguous()
    H = m.shape[0]
    c = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) / (H / 2)
    y, x = torch.meshgrid(c, c, indexing="ij")
    tB1, tT1 = ph.B1.flatten()[idx], ph.T1.flatten()[idx]
    truth = dict(M0=ph.M0.flatten()[idx], T1=tT1, T2=ph.T2.flatten()[idx],
                 T2star=ph.T2star.flatten()[idx], dw=ph.dw.flatten()[idx])
    fields = {"true": tB1, "scaled 1.10": tB1 * 1.10,
              "tilt (1 + 0.15x - 0.10y^2)": tB1 * (1 + 0.15 * x - 0.10 * y**2).flatten()[idx]}
    sig = torch.view_as_real(S).flatten(1).norm(dim=1)
    out = {"voxels": int(len(idx))}
    for name, B1 in fields.items():
        init = {**truth, "B1": B1, "T1": tT1 * (tB1 / B1) ** 2, "M0": truth["M0"] * tB1 / B1}
        f = joint_fit(S, scan1, init, fixed_B1=B1, jacobian="implicit", n_iso=256, n_iter=60)
        eT1, eB = (f["T1"] / tT1).log(), (B1 / tB1).log()
        out[name] = {"max_rel_residual": float((f["cost"].sqrt() / sig).max()),
                     "logT1_median": float(eT1.median()), "logT1_min": float(eT1.min()),
                     "logT1_max": float(eT1.max()),
                     "slope": float((eT1 * eB).sum() / (eB * eB).sum()) if eB.abs().max() > 0 else None}
    return out


def profiled_information():
    """Smallest eigenvalue of the profiled information of polynomial B1 coefficients."""
    pr = paper_protocol()
    out = {}
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(60, 2))
    terms = [(i, j) for i in range(3) for j in range(3 - i)]          # degree-2 polynomial
    for label, protocol in (("one scan", Protocol(pr.pathways, (pr.scans[0],))), ("two scans", pr)):
        blocks, bcols = [], []
        for (px, py) in pts:
            tis = dict(M0=1.0, B1=1.0 + 0.1 * px, T1=900 + 400 * py, T2=70.0, T2star=55.0,
                       dw=0.05, phi0=0.3)
            F = fisher_information(protocol, tis).numpy()             # per-voxel 7×7
            blocks.append(F)
            bcols.append(np.array([px**i * py**j for i, j in terms]))
        # Profiled information of the shared coefficients: Σ_v b_v b_vᵀ · (F_BB − F_Br F_rr⁻¹ F_rB)
        P = np.zeros((len(terms), len(terms)))
        for F, b in zip(blocks, bcols):
            r = [0, 2, 3, 4, 5, 6]
            schur = F[1, 1] - F[1, r] @ np.linalg.solve(F[np.ix_(r, r)], F[r, 1])
            P += schur * np.outer(b, b)
        ev = np.linalg.eigvalsh(P)
        out[label] = {"min_eig": float(ev.min()), "max_eig": float(ev.max()),
                      "schur_median": float(np.median([
                          F[1, 1] - F[1, [0, 2, 3, 4, 5, 6]] @ np.linalg.solve(
                              F[np.ix_([0, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6])], F[[0, 2, 3, 4, 5, 6], 1])
                          for F in blocks]))}
    return out


def design():
    pr = paper_protocol()
    e = pr.scans[0].echo_times
    one = mpme_echo_times((1, 0, -1), 1, 2.0, 4.5)
    WM = dict(M0=1.0, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3)
    GM = dict(M0=1.0, T1=1350.0, T2=90.0, T2star=60.0, dw=0.05, phi0=0.3)
    B1s = np.linspace(0.8, 1.2, 9)
    res = {"b1": {}, "readout": {}, "t1_stage2": {}}
    for a1 in (15, 20):
        for a2 in (250, 290, 310, 320, 330, 340):
            v = [crlb(fisher_information(Protocol(pr.pathways, (Scan(a1, 25.0, e), Scan(a2, 25.0, e))),
                                         {**WM, "B1": b}))["B1"] for b in B1s]
            res["b1"][f"{a1}/{a2}"] = {"median": float(np.median(v)), "worst": float(max(v))}
    for a2 in (320, 330, 340):
        F1 = lambda b: fisher_information(Protocol(pr.pathways, (Scan(15, 25.0, e),)), {**WM, "B1": b})
        F2 = lambda b: fisher_information(Protocol(pr.pathways, (Scan(a2, 12.0, one),)), {**WM, "B1": b}) * (25 / 12)
        v = [crlb(F1(b) + F2(b))["B1"] for b in B1s]
        res["readout"][f"1 window TR 12, a2={a2}"] = {"median": float(np.median(v)), "worst": float(max(v))}
    for a1 in (10, 15, 18, 20, 22, 26):
        vals = {}
        for name, t in (("WM", WM), ("GM", GM)):
            F = fisher_information(Protocol(pr.pathways, (Scan(a1, 25.0, e),)), {**t, "B1": 1.0})
            vals[name] = crlb(F, known=("B1",))["T1"]
        res["t1_stage2"][str(a1)] = vals
    return res


def main():
    out = {"degeneracy": degeneracy_demo(), "profiled": profiled_information(), "design": design()}
    RES.mkdir(exist_ok=True)
    (RES / "two_scan_design.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
