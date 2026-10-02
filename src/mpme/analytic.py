"""Analytic (non-learned) MPME parameter mapping: the baseline for the neural networks.

Follows the order of steps in Cheng et al. 2019 (Fig. 2a), implemented with the signal
model in docs/forward_model.md rather than the paper's own equations:

1. Δω (B0) from the phase change between consecutive echoes of scan 1.
2. R2 and R2′ (→ T2, T2*) from a weighted log-linear fit over all pathways and echoes of
   scan 1. Needs the FID (p = 0) and the echo (p = −1), which decay at R2 + R2′ and
   R2 − R2′ respectively.
3. Decay-free pathway amplitudes A_ip = M0·|a_p| for both scans, using T2, R2′ from step 2.
4. B1⁺ and T1 (and M0) from the FID and echo amplitudes of both scans, using closed-form
   fully dephased FISP/PSIF steady states. One scan alone cannot separate flip angle from
   T1; the second, high-flip scan resolves them. M0 is profiled out, leaving a 2-parameter
   fit per voxel: grid search, then Levenberg–Marquardt.

Unlike the paper (Eq. 8), Δω is not used in step 4: in the fully dephased model the
magnitudes do not depend on off-resonance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .sequence import Protocol

_TINY = 1e-30


def fid_echo_amplitudes(alpha, T1, T2, TR) -> tuple[Tensor, Tensor]:
    """Closed-form |F_0| and |F_−1| just after the RF pulse (unit M0, fully dephased).

    F_0 is the FISP steady state; |F_−1| is the PSIF echo signal (at t = TR) divided by E2.
    """
    E1, E2 = torch.exp(-TR / T1), torch.exp(-TR / T2)
    ca = torch.cos(alpha)
    p = 1 - E1 * ca - E2**2 * (E1 - ca)
    q = E2 * (1 - E1) * (1 + ca)
    r = torch.sqrt(p * p - q * q)
    t = torch.tan(alpha / 2)
    fid = t * (1 - (E1 - ca) * (1 - E2**2) / r)
    echo = t * (1 - (1 - E1 * ca) * (1 - E2**2) / r) / E2
    return fid, echo


def estimate_b0(S: Tensor, protocol: Protocol, scan: int = 0) -> Tensor:
    """Off-resonance Δω (rad/ms) from consecutive-echo phase differences.

    Unambiguous for |Δω| < π / (echo spacing).
    """
    s = S[..., scan, :, :]
    z = s[..., 1:] * torch.conj(s[..., :-1])                               # [..., P, J−1]
    dt = protocol.echo_times(scan, s.real.dtype).diff(dim=-1)               # [P, J−1]
    w = z.abs()
    return (w * torch.angle(z) / dt).sum((-2, -1)) / w.sum((-2, -1)).clamp_min(_TINY)


def _decay_design(protocol: Protocol, scan: int, dtype) -> tuple[Tensor, Tensor]:
    t = protocol.echo_times(scan, dtype)                                    # [P, J]
    p = torch.tensor(protocol.pathways, dtype=dtype)[:, None]
    return t, (t + p * protocol.scans[scan].TR).abs()


def estimate_decay(S: Tensor, protocol: Protocol, scans=(0,)) -> tuple[Tensor, Tensor]:
    """R2 and R2′ (1/ms) from log|S| = c_ip − R2·t − R2′·|t + p·TR_i|, weights |S|².

    Each (scan, pathway) gets its own intercept c_ip; R2 and R2′ are shared. The paper uses
    scan 1 only (``scans=(0,)``); adding the high-flip scan raises the echo-pathway SNR.
    """
    if not {0, -1} <= set(protocol.pathways) or protocol.n_echoes < 2:
        raise ValueError("decay fit needs pathways 0 and −1 with at least 2 echoes each")
    mag = S[..., list(scans), :, :].abs().clamp_min(_TINY)                 # [..., I, P, J]
    I, P, J = mag.shape[-3:]
    t, tau = zip(*(_decay_design(protocol, i, mag.dtype) for i in scans))
    t, tau = torch.stack(t), torch.stack(tau)                               # [I, P, J]
    X = torch.cat([
        torch.eye(I * P, dtype=mag.dtype).repeat_interleave(J, 0),         # c_ip
        -t.reshape(-1, 1),                                                  # R2
        -tau.reshape(-1, 1),                                                # R2′
    ], dim=1)                                                               # [I·P·J, I·P+2]
    y = mag.log().flatten(-3)
    w = mag.flatten(-3) ** 2
    XtW = X.T * w[..., None, :]
    beta = torch.linalg.solve(XtW @ X, (XtW @ y[..., None]))[..., 0]
    return beta[..., I * P], beta[..., I * P + 1]


def decay_free_amplitudes(S: Tensor, protocol: Protocol, R2: Tensor, R2p: Tensor) -> Tensor:
    """log(M0·|a_p|) for every scan and pathway, [..., n_scans, P] (weights |S|²)."""
    out = []
    for i in range(len(protocol.scans)):
        mag = S[..., i, :, :].abs().clamp_min(_TINY)
        t, tau = _decay_design(protocol, i, mag.dtype)
        logA = mag.log() + R2[..., None, None] * t + R2p[..., None, None] * tau
        w = mag**2
        out.append((w * logA).sum(-1) / w.sum(-1))
    return torch.stack(out, dim=-2)


def _log_amp_model(B1, T1, T2, protocol: Protocol) -> Tensor:
    """log|a_p| for p = 0, −1 in every scan: [..., n_scans·2]."""
    cols = []
    for scan in protocol.scans:
        fid, echo = fid_echo_amplitudes(B1 * scan.alpha, T1, T2, torch.as_tensor(scan.TR))
        cols += [fid.clamp_min(_TINY).log(), echo.clamp_min(_TINY).log()]
    return torch.stack(cols, -1)


def _profiled_residual(u: Tensor, T2: Tensor, logA: Tensor, protocol: Protocol) -> Tensor:
    """Residual with log M0 removed; u = (log B1, log T1)."""
    d = logA - _log_amp_model(u[..., 0].exp(), u[..., 1].exp(), T2, protocol)
    return d - d.mean(-1, keepdim=True)


def estimate_b1_t1(
    logA: Tensor,
    T2: Tensor,
    protocol: Protocol,
    *,
    b1_range=(0.4, 1.6),
    t1_range=(100.0, 6000.0),
    grid=(61, 81),
    n_iter: int = 30,
    chunk: int = 1024,
) -> tuple[Tensor, Tensor, Tensor]:
    """B1, T1 (ms) and M0 from FID/echo amplitudes ``logA`` [..., n_scans·2] (scan-major)."""
    shape = T2.shape
    logA, T2 = logA.reshape(-1, logA.shape[-1]), T2.reshape(-1)
    dtype = logA.dtype

    # Coarse grid search, chunked over voxels.
    gb = torch.linspace(math.log(b1_range[0]), math.log(b1_range[1]), grid[0], dtype=dtype)
    gt = torch.linspace(math.log(t1_range[0]), math.log(t1_range[1]), grid[1], dtype=dtype)
    G = torch.stack(torch.meshgrid(gb, gt, indexing="ij"), -1).reshape(-1, 2)  # [G, 2]
    u = torch.empty(len(T2), 2, dtype=dtype)
    for s in range(0, len(T2), chunk):
        sl = slice(s, s + chunk)
        r = _profiled_residual(G[None], T2[sl, None], logA[sl, None], protocol)
        u[sl] = G[(r**2).sum(-1).argmin(-1)]

    # Levenberg–Marquardt refinement (voxels are independent, so the batched gradient of
    # each residual component gives that component's Jacobian row for every voxel).
    lam = torch.full((len(T2),), 1e-3, dtype=dtype)
    lo = torch.tensor([gb[0], gt[0]], dtype=dtype) - 0.5
    hi = torch.tensor([gb[-1], gt[-1]], dtype=dtype) + 0.5
    for _ in range(n_iter):
        u_ = u.detach().requires_grad_(True)
        r = _profiled_residual(u_, T2, logA, protocol)
        J = torch.stack([torch.autograd.grad(r[:, m].sum(), u_, retain_graph=True)[0]
                         for m in range(r.shape[-1])], 1)                   # [N, M, 2]
        r = r.detach()
        JtJ, Jtr = J.transpose(1, 2) @ J, (J.transpose(1, 2) @ r[..., None])[..., 0]
        A = JtJ + lam[:, None, None] * torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2) + 1e-12)
        u_new = torch.minimum(torch.maximum(u - torch.linalg.solve(A, Jtr), lo), hi)
        with torch.no_grad():
            better = ((_profiled_residual(u_new, T2, logA, protocol) ** 2).sum(-1)
                      < (r**2).sum(-1))
        u = torch.where(better[:, None], u_new, u)
        lam = torch.where(better, lam * 0.3, lam * 10).clamp(1e-9, 1e9)

    B1, T1 = u[:, 0].exp(), u[:, 1].exp()
    with torch.no_grad():
        logM0 = (logA - _log_amp_model(B1, T1, T2, protocol)).mean(-1)
    return B1.reshape(shape), T1.reshape(shape), logM0.exp().reshape(shape)


def analytic_reconstruction(
    S: Tensor,
    protocol: Protocol,
    *,
    decay_scans=(0,),
    T2_range=(1.0, 5000.0),
    R2p_max: float = 1.0,
    **b1_t1_kwargs,
) -> dict[str, Tensor]:
    """Map MPME voxel signals S [..., n_scans, P, J] to parameters.

    Args:
        decay_scans: scans used for the R2/R2′ fit (paper: scan 1 only).
        T2_range, R2p_max: physical limits applied to the decay fit (ms, 1/ms).
        b1_t1_kwargs: passed to ``estimate_b1_t1`` (search ranges, iterations).

    Returns a dict with M0, T1, T2, T2star (ms), R2p (1/ms), dw (rad/ms) and B1.
    """
    with torch.no_grad():
        dw = estimate_b0(S, protocol)
        R2, R2p = estimate_decay(S, protocol, decay_scans)
        R2 = R2.nan_to_num(1 / T2_range[1]).clamp(1 / T2_range[1], 1 / T2_range[0])
        R2p = R2p.nan_to_num(0.0).clamp(0.0, R2p_max)
        logA = decay_free_amplitudes(S, protocol, R2, R2p)
        idx = [protocol.index(0), protocol.index(-1)]
        logA = logA[..., idx].flatten(-2)                                   # [..., n_scans·2]
    B1, T1, M0 = estimate_b1_t1(logA, 1 / R2, protocol, **b1_t1_kwargs)
    return {"M0": M0, "T1": T1, "T2": 1 / R2, "T2star": 1 / (R2 + R2p),
            "R2p": R2p, "dw": dw, "B1": B1}
