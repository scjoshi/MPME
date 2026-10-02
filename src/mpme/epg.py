"""Steady-state configuration (pathway) amplitudes for unbalanced, unspoiled gradient echo.

Layer 1 of the MPME forward model (see docs/forward_model.md). Two solvers:

- ``steady_state_isochromat``: exact steady state of N isochromats spread uniformly over
  the intravoxel gradient phase ψ, followed by an FFT over ψ. Fast and exact for the fully
  dephased case. Default choice.
- ``simulate_epg``: extended phase graph run TR by TR, truncated at |k| <= K. Supports
  diffusion and the approach to steady state.

Conventions: M⁺ = Mx + iMy, M⁺(ψ) = Σ_k F_k e^{ikψ}. One TR of gradient maps F_k → F_{k+1};
free precession for time t multiplies M⁺ by e^{iΔω t}. An RF pulse (flip α, phase φ) is the
rotation Rz(φ)·Rx(α)·Rz(−φ), giving F_0 = −i e^{iφ} sin α from equilibrium. Pathway p (the
state F_p just after the RF pulse) is refocused when the gradient moment since the pulse
equals −p TRs' worth.

Units: times in ms, off-resonance dw in rad/ms, D in mm²/s, gradient twist q in rad/mm.
All amplitudes are per unit M0. All parameters broadcast against each other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "rf_rotation",
    "steady_state_isochromat",
    "simulate_epg",
    "epg_pathways",
]


def _as_tensors(*xs, dtype=None, device=None) -> list[Tensor]:
    if dtype is None:
        dtype = next((x.dtype for x in xs if isinstance(x, Tensor) and x.is_floating_point()),
                     torch.float64)
    return [torch.as_tensor(x, dtype=dtype, device=device) for x in xs]


def _rot_x(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([
        torch.stack([o, z, z], -1),
        torch.stack([z, c, -s], -1),
        torch.stack([z, s, c], -1),
    ], -2)


def _rot_z(b: Tensor) -> Tensor:
    c, s = torch.cos(b), torch.sin(b)
    o, z = torch.ones_like(b), torch.zeros_like(b)
    return torch.stack([
        torch.stack([c, -s, z], -1),
        torch.stack([s, c, z], -1),
        torch.stack([z, z, o], -1),
    ], -2)


def rf_rotation(alpha, phi=0.0) -> Tensor:
    """Rotation matrix [..., 3, 3] acting on (Mx, My, Mz) for flip ``alpha``, phase ``phi`` (rad)."""
    alpha, phi = _as_tensors(alpha, phi)
    alpha, phi = torch.broadcast_tensors(alpha, phi)
    return _rot_z(phi) @ _rot_x(alpha) @ _rot_z(-phi)


def steady_state_isochromat(
    alpha,
    T1,
    T2,
    TR,
    pathways: Sequence[int] = (1, 0, -1),
    *,
    phi=0.0,
    dw=0.0,
    n_iso: int = 512,
) -> Tensor:
    """Steady-state pathway amplitudes F_p just after the RF pulse.

    Args:
        alpha: actual flip angle (rad), i.e. B1⁺ · nominal.
        T1, T2, TR: ms.
        pathways: configuration orders p to return.
        phi: RF phase (rad), constant every TR.
        dw: off-resonance (rad/ms). Only changes the phase: F_p · e^{i p dw TR}.
        n_iso: isochromats over ψ ∈ [0, 2π). Orders up to ~n_iso/2 must have decayed;
            increase for long T2/TR.

    Returns:
        Complex tensor [..., len(pathways)].
    """
    alpha, T1, T2, TR, phi, dw = _as_tensors(alpha, T1, T2, TR, phi, dw)
    alpha, T1, T2, TR, phi, dw = torch.broadcast_tensors(alpha, T1, T2, TR, phi, dw)
    if max(abs(p) for p in pathways) >= n_iso // 2:
        raise ValueError("n_iso too small for the requested pathway orders")

    E1 = torch.exp(-TR / T1)
    E2 = torch.exp(-TR / T2)
    R = rf_rotation(alpha, phi)[..., None, :, :]                       # [..., 1, 3, 3]

    psi = 2 * math.pi * torch.arange(n_iso, dtype=alpha.dtype, device=alpha.device) / n_iso
    beta = psi + (dw * TR)[..., None]                                  # [..., N]
    P = _rot_z(beta)                                                   # [..., N, 3, 3]
    E = torch.diag_embed(torch.stack([E2, E2, E1], -1))[..., None, :, :]

    eye = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
    A = eye - R @ P @ E
    rhs = R[..., 2] * (1 - E1)[..., None, None]                        # R · [0, 0, 1−E1]
    M = torch.linalg.solve(A, rhs.expand(A.shape[:-1]))                # [..., N, 3]

    Mxy = torch.complex(M[..., 0], M[..., 1])
    spec = torch.fft.fft(Mxy, dim=-1) / n_iso                          # spec[k] = F_k (k mod N)
    idx = torch.tensor([p % n_iso for p in pathways], device=alpha.device)
    return spec[..., idx]


def simulate_epg(
    alpha,
    T1,
    T2,
    TR,
    n_tr: int,
    K: int = 64,
    *,
    phi=0.0,
    D=0.0,
    q=0.0,
    return_history: bool = False,
):
    """Run the EPG for ``n_tr`` identical TRs from equilibrium.

    Args:
        alpha, T1, T2, TR, phi: as in ``steady_state_isochromat``.
        n_tr: number of TRs (≈ 5·T1/TR to reach steady state).
        K: highest configuration order kept; higher orders are discarded.
        D: diffusion coefficient (mm²/s). q: gradient twist per TR (rad/mm), assumed to be
            applied as a constant gradient over the whole TR.
        return_history: also return F after every RF pulse, [n_tr, ..., 2K+1].

    Returns:
        (F, Z): complex tensors [..., 2K+1] just after the last RF pulse, index k + K.
        With ``return_history``, (F, Z, F_history).
    """
    alpha, T1, T2, TR, phi, D, q = _as_tensors(alpha, T1, T2, TR, phi, D, q)
    alpha, T1, T2, TR, phi, D, q = torch.broadcast_tensors(alpha, T1, T2, TR, phi, D, q)
    shape = alpha.shape
    ctype = torch.complex128 if alpha.dtype == torch.float64 else torch.complex64

    k = torch.arange(-K, K + 1, dtype=alpha.dtype, device=alpha.device)
    tau = (TR / 1000.0)[..., None]                                     # s
    Dq2 = (D * q**2)[..., None]
    E1 = torch.exp(-TR / T1)[..., None]
    E2 = torch.exp(-TR / T2)[..., None]
    decay_F = E2 * torch.exp(-Dq2 * tau * (k**2 + k + 1.0 / 3.0))
    decay_Z = E1 * torch.exp(-Dq2 * tau * k**2)
    recovery = torch.zeros(*shape, 2 * K + 1, dtype=alpha.dtype, device=alpha.device)
    recovery[..., K] = (1 - E1[..., 0])

    a = alpha[..., None]
    c2 = torch.cos(a / 2) ** 2
    s2 = torch.sin(a / 2) ** 2
    sa, ca = torch.sin(a), torch.cos(a)
    eip = torch.exp(1j * phi[..., None].to(ctype))

    F = torch.zeros(*shape, 2 * K + 1, dtype=ctype, device=alpha.device)
    Z = torch.zeros_like(F)
    Z[..., K] = 1.0
    history = []

    for _ in range(n_tr):
        G = torch.conj(F.flip(-1))                                     # G_k = F_{-k}*
        F, Z = (
            c2 * F + eip**2 * s2 * G - 1j * eip * sa * Z,
            -0.5j * torch.conj(eip) * sa * F + 0.5j * eip * sa * G + ca * Z,
        )
        if return_history:
            history.append(F)
        F_post, Z_post = F, Z
        F = F * decay_F
        Z = Z * decay_Z + recovery
        F = torch.cat([torch.zeros_like(F[..., :1]), F[..., :-1]], -1)  # F_k → F_{k+1}

    if return_history:
        return F_post, Z_post, torch.stack(history)
    return F_post, Z_post


def epg_pathways(F: Tensor, pathways: Sequence[int]) -> Tensor:
    """Select orders ``pathways`` from an EPG state [..., 2K+1]."""
    K = (F.shape[-1] - 1) // 2
    if max(abs(p) for p in pathways) > K:
        raise ValueError("pathway order exceeds K")
    return F[..., [p + K for p in pathways]]
