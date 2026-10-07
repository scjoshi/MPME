"""Two-scan flip-angle design (flip-angle design note, steps 1-4): bounds over (α1, α2).

Both scans use the paper's TR (25 ms) and readout; only the nominal flip angles change. For
every pair α1 < α2 on a 1° grid in (0°, 540°], every transmit scale B1 in [0.7, 1.3] (0.01
steps) and every tissue of ``mpme.phantom.TISSUES`` (M0 = 1, so bounds are per unit σ/M0):

  * CRLB of log T1, log B1, log T2, log T2*, log M0 with all seven parameters free;
  * CRLB of log T1 with B1 known (both scans; and scan 1 alone, the two-stage pipeline);
  * first-order bias of log T1 and log B1 per unit relative error in the ratio α2/α1;
  * signal norm of each scan (for a signal floor).

All from F(α1, α2; B1) = F1(B1 α1) + F1(B1 α2) (mpme.design). Each metric is stored per tissue
as its maximum (|.| for biases) and mean over B1, and at B1 = 1. The bounds peak sharply (width
≈ 0.5% in B1) where a pulse crosses a multiple of 180°, so the maximum also includes each pair's
own crossing points B1 = 180° k / α in [0.7, 1.3], where the peak lies. Then the complex and magnitude
alias status of every pair (mpme.design.alias_roots, vectorised over B1).

    PYTHONPATH=src python3 scripts/flip_design.py   → results/flip_design.npz, flip_design.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mpme.design import alias_roots, bounds, ratio_bias, single_scan_fisher
from mpme.phantom import TISSUES
from mpme.sequence import paper_protocol

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PR = paper_protocol()
TR, ECHOES, PATHS = PR.scans[0].TR, PR.scans[0].echo_times, PR.pathways
TISSUE = {name: dict(M0=1.0, T1=t1, T2=t2, T2star=t2s) for name, t1, t2, t2s, _ in TISSUES.values()}
NAMES = list(TISSUE)                                  # CSF, GM, WM, deep GM, lesion
B1S = np.round(np.arange(0.70, 1.3001, 0.01), 4)
ANG = np.arange(1.0, 540.1, 1.0)                      # nominal angles, 1° grid
I, J = np.triu_indices(len(ANG), k=1)                 # pairs α1 = ANG[I] < α2 = ANG[J]
BOX = (0.6, 1.4)                                      # fitting range used for aliases
METRICS = ("T1", "B1", "T2", "T2star", "M0", "T1_knownB1", "bias_T1", "bias_B1")


def crossings():
    """(pair index, B1) for every pair and every B1 = 180° k / α in [0.7, 1.3] of either pulse."""
    pairs, b1 = [], []
    for k in range(1, int(1.3 * ANG[-1] / 180) + 1):
        for a_idx in (I, J):
            b = 180.0 * k / ANG[a_idx]
            sel = (b >= B1S[0]) & (b <= B1S[-1])
            pairs.append(np.nonzero(sel)[0])
            b1.append(b[sel])
    return np.concatenate(pairs), np.concatenate(b1)


def tables():
    """Per tissue: every metric aggregated over B1 → {metric: {"max", "mean", "at1"}: [n_tissue,
    n_pair]}; scan-1-only known-B1 T1 bound and single-scan signal norm [n_tissue, n_B1, n_angle]."""
    agg = {m: {k: np.empty((len(NAMES), len(I)), np.float32) for k in ("max", "mean", "at1")}
           for m in METRICS}
    t1_scan1 = np.empty((len(NAMES), len(B1S), len(ANG)))
    norm = np.empty_like(t1_scan1)
    It, Jt = torch.as_tensor(I), torch.as_tensor(J)
    b_one = int(np.argmin(np.abs(B1S - 1.0)))
    for t, name in enumerate(NAMES):
        Fs = single_scan_fisher(B1S[:, None] * ANG[None, :], TISSUE[name], TR, ECHOES, PATHS)
        norm[t] = Fs[..., 0, 0].sqrt().numpy()
        t1_scan1[t] = bounds(Fs, known=("B1",))[..., 2].numpy()
        run = {m: (np.full(len(I), -np.inf), np.zeros(len(I))) for m in METRICS}
        for b in range(len(B1S)):
            F = Fs[b, It] + Fs[b, Jt]
            free, known = bounds(F), bounds(F, known=("B1",))
            bias = ratio_bias(F, Fs[b, Jt])
            vals = {"M0": free[:, 0], "B1": free[:, 1], "T1": free[:, 2], "T2": free[:, 3],
                    "T2star": free[:, 4], "T1_knownB1": known[:, 2],
                    "bias_T1": bias[:, 2], "bias_B1": bias[:, 1]}
            for m, v in vals.items():
                v = v.numpy()
                mx, sm = run[m]
                np.maximum(mx, np.abs(v), out=mx)
                sm += v
                if b == b_one:
                    agg[m]["at1"][t] = v
        cp, cb = crossings()
        for s0 in range(0, len(cp), 100_000):
            p, b = cp[s0:s0 + 100_000], cb[s0:s0 + 100_000]
            F1 = single_scan_fisher(b * ANG[I[p]], TISSUE[name], TR, ECHOES, PATHS)
            F2 = single_scan_fisher(b * ANG[J[p]], TISSUE[name], TR, ECHOES, PATHS)
            F = F1 + F2
            free, known = bounds(F), bounds(F, known=("B1",))
            bias = ratio_bias(F, F2)
            vals = {"M0": free[:, 0], "B1": free[:, 1], "T1": free[:, 2], "T2": free[:, 3],
                    "T2star": free[:, 4], "T1_knownB1": known[:, 2],
                    "bias_T1": bias[:, 2], "bias_B1": bias[:, 1]}
            for m, v in vals.items():
                np.maximum.at(run[m][0], p, np.abs(v.numpy()))
        for m, (mx, sm) in run.items():
            agg[m]["max"][t], agg[m]["mean"][t] = mx, sm / len(B1S)
        print(f"  {name}: done", flush=True)
    return agg, t1_scan1, norm


def aliases(b_true=np.round(np.arange(0.70, 1.3001, 0.01), 4), box=BOX, n_grid=4001):
    """Alias status of every pair over true B1 in [0.7, 1.3]: 0 none, 1 magnitude alias only (the
    FID sign removes it), 2 complex alias. An alias is another b' in ``box`` with the same
    ξ₂/ξ₁ = tan(b α2/2)/tan(b α1/2) (|ξ₂/ξ₁| for magnitudes) that is admissible for at least one
    tissue (0 < tan(α_E'/2) = tan(α_E/2)·|r| < 1, r the ratio of the scan-1 half-angle tangents; a
    sign flip of both scans is absorbed by φ0 + π). Vectorised version of mpme.design.alias_roots."""
    bp = np.linspace(*box, n_grid)
    zeta_min = min(math.sqrt(math.tanh(TR / (2 * t["T1"]))) for t in TISSUE.values())
    near = np.abs(bp[None, :-1] - b_true[:, None]) < 3 * (bp[1] - bp[0])
    status = np.zeros(len(I), np.int8)
    for n, (a1, a2) in enumerate(zip(ANG[I], ANG[J])):
        f = np.tan(np.radians(bp * a2) / 2) / np.tan(np.radians(bp * a1) / 2)
        ft = np.tan(np.radians(b_true * a2) / 2) / np.tan(np.radians(b_true * a1) / 2)
        sing = np.zeros(n_grid - 1, bool)                       # intervals with a singularity of f
        for s in [(180 + 360 * m) / a2 for m in range(int(box[1] * a2 / 360) + 2)] + \
                 [360 * m / a1 for m in range(1, int(box[1] * a1 / 360) + 2)]:
            sing |= (bp[:-1] <= s) & (bp[1:] >= s)
        r = np.tan(np.radians(bp[None, :-1] * a1) / 2) / np.tan(np.radians(b_true[:, None] * a1) / 2)
        ok = ~sing[None, :] & ~near & (np.abs(r) * zeta_min < 1)
        for level, (g, gt) in ((2, (f, ft)), (1, (np.abs(f), np.abs(ft)))):
            d = g[None, :] - gt[:, None]
            if (ok & (np.sign(d[:, :-1]) * np.sign(d[:, 1:]) < 0)).any():
                status[n] = level
                break
    return status


def main():
    t0 = time.time()
    agg, t1_scan1, norm = tables()
    print(f"tables {time.time() - t0:.0f} s", flush=True)
    a_status = aliases()
    print(f"aliases {time.time() - t0:.0f} s", flush=True)
    RES.mkdir(exist_ok=True)
    np.savez_compressed(RES / "flip_design.npz", ang=ANG, I=I, J=J, b1=B1S, names=np.array(NAMES),
                        t1_scan1=t1_scan1, norm=norm, alias=a_status,
                        **{f"{m}_{k}": v for m, d in agg.items() for k, v in d.items()})
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
