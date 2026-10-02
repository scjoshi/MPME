"""Magnitude model with an exact Jacobian by implicit differentiation of the steady state.

For one isochromat (RF phase 0, intravoxel phase ψ, Δω irrelevant for magnitudes) the
post-pulse steady state solves

    A M = b,   A = I − R_x(α) R_z(ψ) diag(E2, E2, E1),   b = (1 − E1) R_x(α) ẑ,

so for any parameter x ∈ {α, E1, E2}

    ∂M/∂x = A⁻¹ (∂b/∂x − (∂A/∂x) M),

which reuses A⁻¹ (here the explicit 3×3 adjugate inverse, elementwise over the batch).
Pathway amplitudes are Fourier coefficients over ψ, so their derivatives follow by the same
FFT. The echo factor M0·exp(−t/T2 − R2′|t + kTR|) with R2′ = 1/T2* − 1/T2 is differentiated
analytically, and everything is chained to θ = (log M0, log B1, log T1, log T2, log T2*).

Cost: one forward solve plus three right-hand sides with the same inverse, instead of the
six full model evaluations of a forward-difference Jacobian. Works in float32 or float64.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .sequence import Protocol


def _inv3(A: Tensor) -> Tensor:
    """Batched explicit inverse of 3×3 matrices [..., 3, 3] via the adjugate."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    co = torch.stack([
        torch.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
        torch.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
        torch.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1),
    ], -2)
    det = a * co[..., 0, 0] + b * co[..., 1, 0] + c * co[..., 2, 0]
    return co / det[..., None, None]


def steady_state_with_derivatives(alpha: Tensor, E1: Tensor, E2: Tensor,
                                  pathways, n_iso: int):
    """Pathway amplitudes F [N, P] and ∂F/∂(α, E1, E2) [N, P, 3] (complex).

    ``alpha``, ``E1``, ``E2``: [N]. Matches ``epg.steady_state_isochromat`` with φ = Δω = 0.
    """
    dt = alpha.dtype
    N = alpha.shape[0]
    psi = 2 * math.pi * torch.arange(n_iso, dtype=dt) / n_iso
    cp, sp = torch.cos(psi), torch.sin(psi)                                 # [n]
    ca, sa = torch.cos(alpha)[:, None], torch.sin(alpha)[:, None]           # [N, 1]
    e1, e2 = E1[:, None], E2[:, None]
    z = torch.zeros(N, n_iso, dtype=dt)
    one = torch.ones_like(z)

    # Q = R_z(ψ) diag(E2, E2, E1); A = I − R_x(α) Q
    Q = torch.stack([
        torch.stack([e2 * cp + z, -e2 * sp + z, z], -1),
        torch.stack([e2 * sp + z, e2 * cp + z, z], -1),
        torch.stack([z, z, e1 + z], -1),
    ], -2)                                                                  # [N, n, 3, 3]
    R = torch.stack([
        torch.stack([one, z, z], -1),
        torch.stack([z, ca + z, -sa + z], -1),
        torch.stack([z, sa + z, ca + z], -1),
    ], -2)
    dR = torch.stack([                                                      # ∂R/∂α
        torch.stack([z, z, z], -1),
        torch.stack([z, -sa + z, -ca + z], -1),
        torch.stack([z, ca + z, -sa + z], -1),
    ], -2)
    RQ = R @ Q
    eye = torch.eye(3, dtype=dt)
    Ainv = _inv3(eye - RQ)
    Rz = R[..., :, 2]                                                       # R ẑ
    b = (1 - e1)[..., None] * Rz
    M = (Ainv @ b[..., None])[..., 0]                                       # [N, n, 3]

    # Right-hand sides: ∂b/∂x − (∂A/∂x) M, with ∂A/∂x = −∂(RQ)/∂x
    QM = (Q @ M[..., None])[..., 0]
    rhs_a = (1 - e1)[..., None] * dR[..., :, 2] + (dR @ QM[..., None])[..., 0]
    rhs_e1 = -Rz + R[..., :, 2] * M[..., 2:3]                               # R diag(0,0,1) M
    rot_xy = torch.stack([cp * M[..., 0] - sp * M[..., 1],
                          sp * M[..., 0] + cp * M[..., 1], z], -1)          # R_z(ψ) diag(1,1,0) M
    rhs_e2 = (R @ rot_xy[..., None])[..., 0]
    rhs = torch.stack([rhs_a, rhs_e1, rhs_e2], -1)                          # [N, n, 3, 3]
    dM = Ainv @ rhs                                                         # [N, n, 3(comp), 3(x)]

    Mxy = torch.complex(M[..., 0], M[..., 1])                               # [N, n]
    dMxy = torch.complex(dM[..., 0, :], dM[..., 1, :])                      # [N, n, 3]
    spec = torch.fft.fft(Mxy, dim=1) / n_iso
    dspec = torch.fft.fft(dMxy, dim=1) / n_iso
    idx = torch.tensor([p % n_iso for p in pathways])
    return spec[:, idx], dspec[:, idx, :]


def magnitude_model_and_jacobian(theta: Tensor, protocol: Protocol, n_iso: int = 128):
    """Magnitudes m [N, n_scans·P·J] and exact ∂m/∂θ [N, n_scans·P·J, 5].

    θ = (log M0, log B1, log T1, log T2, log T2*), same ordering as ``mle.magnitude_fit``.
    """
    dt = theta.dtype
    M0, B1, T1, T2, T2s = (theta[:, i].exp() for i in range(5))
    p = torch.tensor(protocol.pathways, dtype=dt)[:, None]                  # [P, 1]
    ms, Js = [], []
    for i, scan in enumerate(protocol.scans):
        TR = scan.TR
        alpha = B1 * scan.alpha
        E1, E2 = torch.exp(-TR / T1), torch.exp(-TR / T2)
        F, dF = steady_state_with_derivatives(alpha, E1, E2, protocol.pathways, n_iso)
        absF = F.abs()
        dabs = (F.conj()[..., None] * dF).real / absF[..., None]            # [N, P, 3]
        t = protocol.echo_times(i, dt)                                       # [P, J]
        tau = (t + p * TR).abs()
        R2p = (1 / T2s - 1 / T2)
        D = torch.exp(-t / T2[:, None, None] - R2p[:, None, None] * tau)    # [N, P, J]
        m = M0[:, None, None] * absF[..., None] * D
        base = (M0[:, None, None] * D)                                       # ∂m/∂|F|
        J = torch.stack([
            m,                                                               # log M0
            base * (dabs[..., 0] * alpha[:, None])[..., None],               # log B1
            base * (dabs[..., 1] * (E1 * TR / T1)[:, None])[..., None],      # log T1
            base * (dabs[..., 2] * (E2 * TR / T2)[:, None])[..., None]
            + m * (t - tau) / T2[:, None, None],                             # log T2
            m * tau / T2s[:, None, None],                                    # log T2*
        ], -1)                                                               # [N, P, J, 5]
        ms.append(m.flatten(1))
        Js.append(J.flatten(1, 2))
    return torch.cat(ms, 1), torch.cat(Js, 1)
