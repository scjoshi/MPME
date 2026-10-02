"""Layer 2 of the MPME forward model: voxel signal at every echo of every pathway.

    S_ijp = M0 · a_p(B1·α_i, T1, T2, TR_i) · e^{−t/T2} · e^{−R2′|t + p·TR_i|}
               · e^{i(φ0 + Δω (t + p·TR_i))}

with a_p the steady-state amplitude of pathway p just after the RF pulse (Layer 1, computed
at Δω = 0 since off-resonance only adds the phase written explicitly here), and the R2′ term
from a Lorentzian intravoxel frequency distribution. See docs/forward_model.md.

Units: ms for times, 1/ms for R2′, rad/ms for Δω.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .epg import _as_tensors, steady_state_isochromat
from .sequence import Protocol


def mpme_signal(
    protocol: Protocol,
    M0,
    T1,
    T2,
    R2p,
    dw,
    B1,
    phi0=0.0,
    *,
    n_iso: int = 512,
) -> Tensor:
    """Noise-free voxel signals.

    Args:
        protocol: pathways, flip angles, TRs and echo times.
        M0: proton density × receive sensitivity (C·M0).
        T1, T2: ms. R2p: reversible decay rate R2′ (1/ms). dw: off-resonance (rad/ms).
        B1: transmit scale, actual flip = B1 · nominal. phi0: global phase (rad).
        n_iso: isochromats for the steady-state solver.

    Returns:
        Complex tensor [..., n_scans, P, J], broadcast over the parameter shapes.
    """
    M0, T1, T2, R2p, dw, B1, phi0 = _as_tensors(M0, T1, T2, R2p, dw, B1, phi0)
    M0, T1, T2, R2p, dw, B1, phi0 = torch.broadcast_tensors(M0, T1, T2, R2p, dw, B1, phi0)
    p = torch.tensor(protocol.pathways, dtype=T1.dtype)[:, None]          # [P, 1]
    x = lambda v: v[..., None, None]                                       # → [..., 1, 1]

    out = []
    for i, scan in enumerate(protocol.scans):
        a = steady_state_isochromat(B1 * scan.alpha, T1, T2, scan.TR, protocol.pathways,
                                    n_iso=n_iso)                           # [..., P]
        t = protocol.echo_times(i, T1.dtype)                               # [P, J]
        tau = t + p * scan.TR                                              # time since dephasing origin
        mag = torch.exp(-t / x(T2) - x(R2p) * tau.abs())
        phase = torch.exp(1j * (x(phi0) + x(dw) * tau))
        out.append(x(M0) * a[..., None] * mag * phase)
    return torch.stack(out, dim=-3)


def add_noise(S: Tensor, sigma: float, generator: torch.Generator | None = None) -> Tensor:
    """Add circular complex Gaussian noise with standard deviation ``sigma`` per channel."""
    noise = torch.randn(S.shape, dtype=S.dtype, generator=generator)       # complex randn: var 1 total
    return S + sigma * (2 ** 0.5) * noise
