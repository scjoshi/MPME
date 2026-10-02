"""Faithful implementation of the Cheng et al. 2019 MPME reconstruction (sequential case).

Equation numbers refer to the paper (MRM 2019;81:1699–1713, author manuscript
nihms-986644). Steps:

1. Δω from the echo phase evolution of scan 1 (shared with ``analytic.estimate_b0``).
2. R2, R2′ from Eq. 1 fitted to scan 1:  S = F⁺ e^{−(R2 ± R2′)TE}, + for k ≥ 0, − for k < 0.
3. Real-valued F states (Eq. 18: +|F| for k ≥ 0, −|F| for k < 0): F⁺ at TE = 0 and
   F⇒ at TE = TR from Eq. 1; F⁻_k = F⇒_{k−1} (gradient shift); the remaining F⁻ via Eq. 7.
4. Mixing factor X and scale Ω (Eqs. 10–11) for both scans; the flip angle from Eq. 15 with
   c_i = cos α_i and α2 = β·(α2_nom/α1_nom)·α1 (Eqs. 8, 19; instantaneous RF, so the
   off-resonance nutation function ν(α, Δf) reduces to cos α). The branch α2 < 360° vs
   α2 > 360° is chosen from the relative phase of F⁺_0 in the two scans.
5. T1 from Eq. 16 and M0 from Eq. 17, using scan 1.

Two deviations: Eq. 16 is used as T1 = TR / ln[(Xc − 1)/(X − c)] (consistent with its own
derivation, Z⁺/Z⇒ = e^{TR/T1}); and B1 is solved per voxel, at full resolution, with no
low-resolution filtering or polynomial smoothing (those belong to the imaging layer).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .analytic import _TINY, estimate_b0
from .sequence import Protocol


def fit_eq1(S: Tensor, protocol: Protocol, scan: int = 0) -> tuple[Tensor, Tensor]:
    """R2, R2′ (1/ms) from Eq. 1, log-linear, weighted by |S| as in the paper."""
    mag = S[..., scan, :, :].abs().clamp_min(_TINY)
    P, J = mag.shape[-2:]
    t = protocol.echo_times(scan, mag.dtype)
    sgn = torch.tensor([1.0 if p >= 0 else -1.0 for p in protocol.pathways], dtype=mag.dtype)
    X = torch.cat([
        torch.eye(P, dtype=mag.dtype).repeat_interleave(J, 0),             # log F⁺_k
        -t.reshape(-1, 1),                                                  # R2
        -(sgn[:, None] * t).reshape(-1, 1),                                 # ±R2′
    ], dim=1)
    y = mag.log().flatten(-2)
    w = mag.flatten(-2)
    XtW = X.T * w[..., None, :]
    beta = torch.linalg.solve(XtW @ X, XtW @ y[..., None])[..., 0]
    return beta[..., P], beta[..., P + 1]


def f_states(S: Tensor, protocol: Protocol, R2: Tensor, R2p: Tensor):
    """Signed real F⁺ and F⇒ for every scan and pathway, each [..., n_scans, P] (Eqs. 1, 18)."""
    sgn = torch.tensor([1.0 if p >= 0 else -1.0 for p in protocol.pathways], dtype=R2.dtype)
    rate = R2[..., None] + sgn * R2p[..., None]                             # [..., P]
    Fp, Fd = [], []
    for i, scan in enumerate(protocol.scans):
        mag = S[..., i, :, :].abs().clamp_min(_TINY)
        t = protocol.echo_times(i, mag.dtype)
        logF = mag.log() + rate[..., None] * t                              # Eq. 1 at TE = 0
        w = mag
        F = sgn * torch.exp((w * logF).sum(-1) / w.sum(-1))
        Fp.append(F)
        Fd.append(F * torch.exp(-rate * scan.TR))                           # Eq. 1 at TE = TR
    return torch.stack(Fp, -2), torch.stack(Fd, -2)


def mixing_factor(Fp: Tensor, Fd: Tensor, protocol: Protocol, k: int) -> tuple[Tensor, Tensor]:
    """X and Ω (Eqs. 10–11) for every scan, using pathway setting k = +1 or −1."""
    idx = protocol.index
    Fm_1 = Fd[..., idx(0)]                                                  # F⁻_1  = F⇒_0
    if k == 1:   # [1, 0, −1] scheme
        Fp_k, Fm_k = Fp[..., idx(1)], Fm_1
        Fm_mk = Fm_1 - Fp[..., idx(1)] + Fp[..., idx(-1)]                   # Eq. 7
    elif k == -1:  # [0, −1, −2] scheme
        Fp_k, Fm_k = Fp[..., idx(-1)], Fd[..., idx(-2)]                     # F⁻_−1 = F⇒_−2
        Fm_mk = Fm_1
    else:
        raise ValueError("k must be +1 or −1")
    Om = Fm_k + Fm_mk                                                       # Eq. 10
    X = 1 - 2 * (Fm_k - Fp_k) / Om                                          # Eq. 11
    return X, Om


def _log_ratio(X: Tensor, c: Tensor) -> Tensor:
    """ln[(Xc − 1)/(X − c)] = TR/T1 (Eqs. 13, 12, 14, 2); NaN where not a valid ratio > 1."""
    r = (X * c - 1) / (X - c)
    return torch.where(r > 1, torch.log(r.clamp_min(_TINY)), torch.nan)


def solve_flip(
    X: Tensor,
    protocol: Protocol,
    branch_high: Tensor,
    *,
    beta: float = 1.0,
    n_grid: int = 4001,
    n_bisect: int = 40,
    chunk: int = 8192,
) -> Tensor:
    """Solve Eq. 15 for α1 (rad). ``X`` is [..., 2]; ``branch_high``: α2 > 360°."""
    s1, s2 = protocol.scans
    ratio = beta * s2.flip_deg / s1.flip_deg
    q = s2.TR / s1.TR
    lo = math.pi / ratio                                                    # α2 = 180°
    mid = 2 * math.pi / ratio                                               # α2 = 360°
    hi = min(3 * math.pi / ratio, math.pi - 1e-9)                           # α2 = 540°

    def h(a1, X1, X2):
        return q * _log_ratio(X1, torch.cos(a1)) - _log_ratio(X2, torch.cos(ratio * a1))

    shape = X.shape[:-1]
    X1, X2 = X[..., 0].reshape(-1), X[..., 1].reshape(-1)
    bh = branch_high.reshape(-1)
    out = torch.empty_like(X1)
    u = torch.linspace(0, 1, n_grid, dtype=X.dtype)[1:-1]
    for s in range(0, len(X1), chunk):
        sl = slice(s, s + chunk)
        a_lo = torch.where(bh[sl], mid, lo)[:, None]
        a_hi = torch.where(bh[sl], hi, mid)[:, None]
        grid = a_lo + (a_hi - a_lo) * u                                     # [n, G]
        hv = h(grid, X1[sl, None], X2[sl, None])
        # Bracket the sign change closest to the minimum of |h|.
        flip = (hv[:, :-1] * hv[:, 1:] <= 0) & torch.isfinite(hv[:, :-1] * hv[:, 1:])
        score = torch.where(flip, torch.minimum(hv[:, :-1].abs(), hv[:, 1:].abs()), torch.inf)
        g = score.argmin(-1)
        found = torch.isfinite(score.gather(1, g[:, None]))[:, 0]
        a = grid.gather(1, g[:, None])[:, 0]
        b = grid.gather(1, (g + 1)[:, None])[:, 0]
        ha = h(a, X1[sl], X2[sl])
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            hm = h(m, X1[sl], X2[sl])
            left = (ha * hm <= 0)
            b = torch.where(left, m, b)
            a = torch.where(left, a, m)
            ha = torch.where(left, ha, hm)
        fallback = grid.gather(1, torch.nan_to_num(hv.abs(), nan=torch.inf).argmin(-1, keepdim=True))[:, 0]
        out[sl] = torch.where(found, 0.5 * (a + b), fallback)
    return out.reshape(shape)


def paper_reconstruction(
    S: Tensor,
    protocol: Protocol,
    *,
    k: int | None = None,
    beta: float = 1.0,
    B1: Tensor | None = None,
    T2_range=(1.0, 5000.0),
    R2p_max: float = 1.0,
) -> dict[str, Tensor]:
    """Map MPME voxel signals S [..., 2, P, J] to parameters with the paper's equations.

    Args:
        k: pathway setting for Eqs. 10–11; default +1 if pathway +1 was acquired, else −1.
        beta: in-vivo calibration factor of Eq. 19 (the paper used 1.24 in vivo, 1 in phantom).
        B1: optional known/smoothed B1 map; skips Eq. 15. The paper computes B1 at low
            resolution and smooths it with a polynomial fit, so B1 is nearly noise-free when
            T1 is computed; passing it here mimics that.

    T1 is NaN where noise makes the Eq. 16 log-ratio invalid (ratio ≤ 1).

    Returns a dict with M0, T1, T2, T2star (ms), R2p (1/ms), dw (rad/ms) and B1.
    """
    if len(protocol.scans) != 2:
        raise ValueError("the paper's method needs exactly two scans")
    if k is None:
        k = 1 if 1 in protocol.pathways else -1
    with torch.no_grad():
        dw = estimate_b0(S, protocol)                                       # step 1
        R2, R2p = fit_eq1(S, protocol)                                      # step 2
        R2 = R2.nan_to_num(1 / T2_range[1]).clamp(1 / T2_range[1], 1 / T2_range[0])
        R2p = R2p.nan_to_num(0.0).clamp(0.0, R2p_max)
        Fp, Fd = f_states(S, protocol, R2, R2p)                             # step 3
        X, _ = mixing_factor(Fp, Fd, protocol, k)                           # step 4
        i0 = protocol.index(0)
        s1 = protocol.scans[0]
        if B1 is None:
            branch_high = (S[..., 0, i0, :] * S[..., 1, i0, :].conj()).sum(-1).real > 0
            a1 = solve_flip(X, protocol, branch_high, beta=beta)
        else:
            a1 = torch.as_tensor(B1, dtype=X.dtype).expand(X.shape[:-1]) * s1.alpha
        c1 = torch.cos(a1)
        T1 = s1.TR / _log_ratio(X[..., 0], c1)                              # Eq. 16 (step 5)
        E1 = torch.exp(-s1.TR / T1)
        F0p, F0m = Fp[..., 0, i0], Fd[..., 0, protocol.index(-1)]           # F⁻_0 = F⇒_−1
        M0 = ((F0p - c1 * F0m + (F0m - c1 * F0p) * E1).abs()
              / (torch.sqrt((1 - c1**2).clamp_min(_TINY)) * (1 - E1)))      # Eq. 17
    return {"M0": M0, "T1": T1, "T2": 1 / R2, "T2star": 1 / (R2 + R2p), "R2p": R2p,
            "dw": dw, "B1": a1 / s1.alpha}
