"""Voxels that share B1 (technote sec:shared): joint information, profile and free joint fit.

Two voxel sets share one B1 (truth 1) and differ in every other parameter: WM + GM of
tab:sharedfamily, and 20 random tissues (seed 0). One complex noise realisation per set,
sigma = 1e-3 (seed 1), fitted from scan 1 alone and from both scans of the paper protocol.

1. Joint Fisher information (sigma = 1) of [log B1 | per voxel log M0, log T1, log T2, log T2*,
   dw, phi0]: smallest eigenvalue of the normalised matrix; its eigenvector, per unit d log B1,
   against the family tangent d log M0 = -x/sin x, d log T1 = -2 (x/sin x) sinh(y)/y
   (x = B1 alpha1, y = TR/T1); profiled information on log B1 relative to F_BB; CRLB SD of
   log B1 at sigma = 1e-3 (two scans).
2. Profile likelihood: total cost over a grid of fixed shared B1' in [0.6, 1.6] (per-voxel
   joint_fit, best of the family member and the true parameters as starts). One scan: its
   range. Two scans: location and curvature width of the minimum from local quadratic fits in
   log B1 (refined to +-1 CRLB SD), against the CRLB.
3. Free joint fit with a shared B1 (block Levenberg-Marquardt, exact Jacobian; joint_fit cannot
   share a parameter) from B1 starts STARTS, other parameters at a generic start (M0 0.8,
   T1 1000 ms, T2 80 ms, T2* 56 ms, dw = truth + 0.01 N(0,1), phi0 from the data). Fitting box
   B1 in [e^-1, 540/330]; two scans also without the cap (box [e^-1, e]), which admits the
   complex alias, the other root of tan(B1 a2/2)/tan(B1 a1/2) = f(B1hat) (eq:secondscan).

    python scripts/shared_b1_check.py   → results/shared_b1_check.json,
                                          docs/technote/tables/shared_b1.tex (fit 3, WM + GM,
                                          starts TABLE_STARTS)
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

from mpme.crlb import fisher_information, normalized_information
from mpme.fastjac import complex_model_and_jacobian
from mpme.mle import _b1_projector, _project, joint_fit
from mpme.sequence import Protocol, paper_protocol
from mpme.signal import add_noise, mpme_signal

torch.set_default_dtype(torch.float64)    # also for crlb.fisher_information's log-parameters

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "docs" / "technote" / "tables"
PROT2 = paper_protocol()
PROT1 = Protocol(PROT2.pathways, PROT2.scans[:1])
TR, ALPHA1, ALPHA2 = PROT2.scans[0].TR, PROT2.scans[0].alpha, PROT2.scans[1].alpha
SIGMA = 1e-3
N_ISO = 256                       # same isochromat count for data, information and fits
B1_MAX = 540 / 330
STARTS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.6)
TABLE_STARTS = (0.6, 0.8, 1.0, 1.3, 1.6)
LOCAL = [0, 2, 3, 4, 5, 6]        # per-voxel columns of θ = (log M0, log B1, log T1, ..., φ0)
WM = dict(M0=0.69, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3)
GM = dict(M0=0.82, T1=1350.0, T2=90.0, T2star=60.0, dw=-0.03, phi0=1.1)


def random_tissues(n=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T2 = rng.uniform(50, 120)
        out.append(dict(M0=rng.uniform(0.6, 1.0), T1=rng.uniform(600, 2000), T2=T2,
                        T2star=rng.uniform(0.6, 0.9) * T2, dw=rng.uniform(-0.1, 0.1),
                        phi0=rng.uniform(-math.pi, math.pi)))
    return out


def col(tissues, k):
    return torch.tensor([t[k] for t in tissues])


def family(t, B1p):
    """(M0', T1') of tissue t (true B1 = 1) at shared trial B1p (eq:sharedfamily)."""
    zeta = math.sqrt(math.tanh(TR / (2 * t["T1"])))
    xi = math.tan(ALPHA1 / 2) / zeta
    zeta_p = math.tan(B1p * ALPHA1 / 2) / xi
    return t["M0"] * zeta / zeta_p, TR / (2 * math.atanh(zeta_p**2))


def tangent(t):
    """(d log M0, d log T1) per d log B1 along the family, at the truth."""
    x, y = ALPHA1, TR / t["T1"]
    return -x / math.sin(x), -2 * (x / math.sin(x)) * math.sinh(y) / y


def noisy_data(tissues):
    S = mpme_signal(PROT2, col(tissues, "M0"), col(tissues, "T1"), col(tissues, "T2"),
                    1 / col(tissues, "T2star") - 1 / col(tissues, "T2"), col(tissues, "dw"),
                    1.0, col(tissues, "phi0"), n_iso=N_ISO)
    return add_noise(S, SIGMA, torch.Generator().manual_seed(1))


# ---- 1. Joint Fisher information ---------------------------------------------------------
def joint_fisher(protocol, tissues):
    """Information (σ = 1) on [log B1 | per voxel log M0, log T1, log T2, log T2*, Δω, φ0]."""
    N = len(tissues)
    F = torch.zeros(1 + 6 * N, 1 + 6 * N)
    for j, t in enumerate(tissues):
        idx = [1 + 6 * j, 0] + [2 + 6 * j + k for k in range(5)]          # crlb.PARAMS order
        F[np.ix_(idx, idx)] += fisher_information(protocol, {**t, "B1": 1.0}, n_iso=N_ISO)
    return F


def fisher_summary(protocol, tissues):
    F = joint_fisher(protocol, tissues)
    ev, U = torch.linalg.eigh(normalized_information(F))
    w = U[:, 0] / F.diagonal().sqrt()                                      # log coordinates
    w = w / w[0]                                                           # per unit d log B1
    t = torch.zeros_like(w)
    t[0] = 1.0
    for j, tis in enumerate(tissues):
        t[1 + 6 * j], t[2 + 6 * j] = tangent(tis)
    r = list(range(1, F.shape[0]))
    schur = (F[0, 0] - F[0, r] @ torch.linalg.solve(F[np.ix_(r, r)], F[r, 0])).item()
    return dict(min_norm_eig=ev[0].item(), second_norm_eig=ev[1].item(),
                max_dev_eigvec_from_tangent=(w - t).abs().max().item(),
                F_BB=F[0, 0].item(), profiled_info=schur, profiled_over_F_BB=schur / F[0, 0].item())


# ---- 2. Profile likelihood over a fixed shared B1' -----------------------------------------
def profile(S, protocol, tissues, grid):
    """Σ_voxels min ‖r‖² at each fixed shared B1' (best of two starts), and bound flags."""
    G, N = len(grid), len(tissues)
    B1 = torch.tensor(np.repeat(grid, N))
    best = torch.full((G * N,), math.inf)
    bound = torch.zeros(G * N, dtype=torch.bool)
    for start in ("family", "truth"):
        mt = torch.tensor([family(t, b) if start == "family" else (t["M0"], t["T1"])
                           for b in grid for t in tissues])
        init = dict(M0=mt[:, 0], T1=mt[:, 1], B1=B1,
                    **{k: col(tissues, k).repeat(G) for k in ("T2", "T2star", "dw")})
        out = joint_fit(S.repeat(G, 1, 1, 1), protocol, init, n_iter=100, n_iso=N_ISO,
                        jacobian="implicit", fixed_B1=B1)
        better = out["cost"] < best
        best = torch.where(better, out["cost"], best)
        bound = torch.where(better, out["at_box"] | out["at_r2p_bound"], bound)
    return best.reshape(G, N).sum(1).numpy(), bound.reshape(G, N).any(1).numpy()


def profile_minimum(S, tissues, centre, sd):
    """Minimum of the two-scan profile: grid ±0.05 in log B1, then quadratic fits over ±4 SD
    and twice over ±1 SD, each centred on the previous minimum.

    ±1 SD is the 68 % profile-likelihood interval (Δ cost = σ²), so the last fit gives the
    curvature that sets the interval; centring it on the minimum keeps the profile's cubic
    term out of that curvature. Returns B1 at the minimum, curvature, profile cost there.
    """
    x = centre + np.linspace(-0.05, 0.05, 21)
    c, _ = profile(S, PROT2, tissues, np.exp(x))
    centre = x[c.argmin()]
    for half in (4 * sd, sd, sd):
        x = np.linspace(-half, half, 9)
        c, _ = profile(S, PROT2, tissues, np.exp(centre + x))
        p = np.polyfit(x, c, 2)
        centre += -p[1] / (2 * p[0])
    b = math.exp(centre)
    return b, p[0], profile(S, PROT2, tissues, np.array([b]))[0][0]


def profile_summary(S, protocol, tissues, crlb_sd=None):
    grid = np.linspace(0.6, 1.6, 21)
    c, bound = profile(S, protocol, tissues, grid)
    out = dict(grid=grid.tolist(), cost_over_sigma2=(c / SIGMA**2).tolist(),
               bound_active=bound.tolist(), range_over_sigma2=float(np.ptp(c) / SIGMA**2),
               grid_argmin=float(grid[c.argmin()]))
    if crlb_sd is not None:
        b, c2, cmin = profile_minimum(S, tissues, math.log(grid[c.argmin()]), crlb_sd)
        out.update(B1_at_min=b, min_cost_over_sigma2=cmin / SIGMA**2, curvature=c2,
                   width_sd_logB1=SIGMA / math.sqrt(c2), crlb_sd_logB1=crlb_sd)
    return out


# ---- 3. Free joint fit with a shared B1 ------------------------------------------------------
def shared_b1_fit(S, protocol, theta, b1_max=None, n_iter=400):
    """Levenberg-Marquardt for voxels sharing B1, x = [log B1 | θ without log B1, per voxel].

    θ [N, 7] as in mle.joint_fit, with a common B1 column; same damping and box as joint_fit,
    B1 ≤ b1_max if given. While B1 sits on the upper bound and the gradient points outward,
    it is held there and only the per-voxel parameters are stepped (a plain projected step
    stalls on the bound). Returns the fitted θ and the cost ‖r‖².
    """
    N = S.shape[0]
    y = torch.view_as_real(S).flatten()
    project = _b1_projector(_project, b1_max, None)
    log_hi = math.log(b1_max or math.e)                                    # top of the B1 box

    def evaluate(th):
        r, J = complex_model_and_jacobian(th, protocol, N_ISO)            # [N, M], [N, M, 7]
        J = torch.cat([J[:, :, 1].reshape(-1, 1), torch.block_diag(*J[:, :, LOCAL])], 1)
        return r.flatten() - y, J

    theta = project(theta)
    res, J = evaluate(theta)
    cost, lam = res @ res, 1e-2
    for _ in range(n_iter):
        JtJ, g = J.T @ J, J.T @ res
        A = JtJ + lam * torch.diag(JtJ.diagonal() + 1e-12)
        k = int(theta[0, 1] >= log_hi - 1e-12 and g[0] < 0)   # B1 on its bound, pushed out: hold it
        dx = torch.zeros_like(g)
        dx[k:] = torch.linalg.solve(A[k:, k:], g[k:])
        trial = theta.clone()
        trial[:, 1] -= dx[0]
        trial[:, LOCAL] -= dx[1:].reshape(N, 6)
        trial = project(trial)
        r_new, J_new = evaluate(trial)
        c_new = r_new @ r_new
        if torch.isfinite(c_new) and c_new < cost:
            theta, res, J, cost, lam = trial, r_new, J_new, c_new, max(lam * 0.3, 1e-9)
        elif lam >= 1e9:                     # rejected at maximal damping: a fixed point, stop
            break
        else:
            lam = min(lam * 10, 1e9)
    return theta, cost.item()


def generic_start(S, protocol, dw, B1):
    """θ at M0 0.8, B1, T1 1000, T2 80, T2* 56 ms, Δω = dw; φ0 least-squares optimal."""
    N = len(dw)
    S0 = mpme_signal(protocol, 0.8, 1000.0, 80.0, 1 / 56 - 1 / 80, dw, B1, n_iso=N_ISO)
    phi0 = torch.angle((S * S0.conj()).flatten(1).sum(1))
    fixed = torch.tensor([0.8, B1, 1000.0, 80.0, 56.0]).log()
    return torch.cat([fixed.expand(N, 5), dw[:, None], phi0[:, None]], 1)


def complex_alias(B1):
    """The other root of f(b) = tan(b α2/2)/tan(b α1/2) = f(B1) in (1.8, 2.2)."""
    f = lambda b: math.tan(b * ALPHA2 / 2) / math.tan(b * ALPHA1 / 2)
    return brentq(lambda b: f(b) - f(B1), 1.8, 2.2)


def free_fits(S, tissues):
    rng = np.random.default_rng(3)
    dw = col(tissues, "dw") + 0.01 * torch.tensor(rng.standard_normal(len(tissues)))
    out = {}
    for name, protocol, Sd, b1_max in (("scan1", PROT1, S[:, :1], B1_MAX),
                                       ("scan1+2", PROT2, S, B1_MAX),
                                       ("scan1+2 uncapped", PROT2, S, None)):
        rows = []
        for b in STARTS:
            th, cost = shared_b1_fit(Sd, protocol, generic_start(Sd, protocol, dw, b), b1_max)
            rows.append(dict(start=b, B1=th[0, 1].exp().item(), cost_over_sigma2=cost / SIGMA**2,
                             T1_first2=th[:2, 2].exp().tolist(),
                             M0_first2=th[:2, 0].exp().tolist()))
        out[name] = rows
    best = min(out["scan1+2"], key=lambda r: r["cost_over_sigma2"])
    out["B1_ml"], out["predicted_alias"] = best["B1"], complex_alias(best["B1"])
    return out


def cost_tex(c):
    if c < 1e3:
        return f"{c:.4f}"
    m, e = f"{c:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def table_rows(fit):
    """start B1 | one scan: B1, cost/σ² | two scans (B1 ≤ 540/330): B1, cost/σ²."""
    return [f"{a['start']:.1f} & {a['B1']:.4f} & {cost_tex(a['cost_over_sigma2'])} & "
            f"{b['B1']:.4f} & {cost_tex(b['cost_over_sigma2'])}\\\\"
            for a, b in zip(fit["scan1"], fit["scan1+2"]) if a["start"] in TABLE_STARTS]


def report(name, r):
    f1, f2 = r["fisher"]["scan1"], r["fisher"]["scan1+2"]
    p1, p2 = r["profile"]["scan1"], r["profile"]["scan1+2"]
    print(f"\n=== {name} ===")
    for k, f in (("scan1", f1), ("scan1+2", f2)):
        print(f"Fisher {k:8s}: min norm eig {f['min_norm_eig']:.3e} "
              f"(next {f['second_norm_eig']:.3g}), "
              f"max |eigvec - tangent| {f['max_dev_eigvec_from_tangent']:.3g}, "
              f"profiled/F_BB {f['profiled_over_F_BB']:.3g}")
    print(f"CRLB SD log B1 (two scans, sigma {SIGMA:g}): {p2['crlb_sd_logB1']:.3e}")
    print(f"Profile scan1: range {p1['range_over_sigma2']:.2e} sigma^2 over B1' in [0.6, 1.6], "
          f"cost {p1['cost_over_sigma2'][0]:.6f} sigma^2, bound active {any(p1['bound_active'])}")
    print(f"Profile scan1+2: min at B1' = {p2['B1_at_min']:.5f}, "
          f"cost {p2['min_cost_over_sigma2']:.4f} sigma^2, "
          f"width SD log B1 {p2['width_sd_logB1']:.3e} (CRLB {p2['crlb_sd_logB1']:.3e}); "
          f"range {p2['range_over_sigma2']:.4g} sigma^2, bound active at "
          f"{[round(g, 2) for g, a in zip(p2['grid'], p2['bound_active']) if a]}")
    for k in ("scan1", "scan1+2", "scan1+2 uncapped"):
        print(f"Free fit {k:16s}: " + ", ".join(f"{x['start']:.1f}->{x['B1']:.4f} "
                                                 f"({x['cost_over_sigma2']:.4f})"
                                                 for x in r["free_fit"][k]))
    print(f"Predicted complex alias of B1 = {r['free_fit']['B1_ml']:.5f}: "
          f"{r['free_fit']['predicted_alias']:.5f}")


def main():
    t0 = time.time()
    out = {}
    for name, tissues in (("WM+GM", [WM, GM]), ("20 random", random_tissues())):
        S = noisy_data(tissues)
        fisher = {k: fisher_summary(p, tissues) for k, p in (("scan1", PROT1), ("scan1+2", PROT2))}
        sd = SIGMA / math.sqrt(fisher["scan1+2"]["profiled_info"])
        prof = {"scan1": profile_summary(S[:, :1], PROT1, tissues),
                "scan1+2": profile_summary(S, PROT2, tissues, sd)}
        out[name] = dict(fisher=fisher, profile=prof, free_fit=free_fits(S, tissues))
        report(name, out[name])
    rows = table_rows(out["WM+GM"]["free_fit"])
    out["runtime_s"] = time.time() - t0
    RES.mkdir(exist_ok=True)
    (RES / "shared_b1_check.json").write_text(json.dumps(out, indent=1))
    (TAB / "shared_b1.tex").write_text("\n".join(rows) + "\n")
    print("\n" + "\n".join(rows))
    print(f"\nruntime {out['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
