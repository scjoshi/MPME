"""Least-squares fits of the full MPME signal model.

``joint_fit``: maximum likelihood for complex data. Fits
θ = (log M0, log B1, log T1, log T2, log T2*, Δω, φ0) to all scans, pathways and echoes at
once (Gaussian noise ⇒ nonlinear least squares). Same parameterisation as ``crlb.py``, so its
spread can be compared directly with the Cramér–Rao bound.

``magnitude_fit``: Gaussian least squares on magnitude images, |y| ≈ |s(θ)|, with
θ = (log M0, log B1, log T1, log T2, log T2*) (Δω and φ0 do not affect magnitudes). This is
*not* maximum likelihood for magnitude data, whose noise is Rician; it ignores the noise
floor and is therefore biased when a signal is comparable to σ.

Both use voxel-wise projected Levenberg–Marquardt and need a starting point, e.g. the
output of ``analytic.analytic_reconstruction``. Constraints: the Lorentzian model needs
R2′ ≥ 0, i.e. T2* ≤ T2 (projection log T2* ≤ log T2); safeguard box B1 ∈ [e⁻¹, e],
T1 ∈ [10, 20000] ms, T2 ∈ [1, 5000] ms. Active constraints are reported per voxel, because
an estimate on a constraint is not covered by the interior, unbiased Cramér–Rao bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from .sequence import Protocol
from .signal import mpme_signal

_LOG = (0, 1, 2, 3, 4)  # indices of log-parameters in θ
_NAMES = ("M0", "B1", "T1", "T2", "T2star")
_BOX = {1: (-1.0, 1.0),                                  # log B1
        2: (math.log(10.0), math.log(20000.0)),          # log T1 (ms)
        3: (math.log(1.0), math.log(5000.0))}            # log T2 (ms)


def _project(th: Tensor) -> Tensor:
    th = th.clone()
    for i, (lo, hi) in _BOX.items():
        th[:, i] = th[:, i].clamp(lo, hi)
    th[:, 4] = torch.minimum(th[:, 4], th[:, 3])                            # T2* ≤ T2
    return th


def _at_box(th: Tensor) -> Tensor:
    hit = torch.zeros(th.shape[0], dtype=torch.bool)
    for i, (lo, hi) in _BOX.items():
        hit |= (th[:, i] <= lo + 1e-9) | (th[:, i] >= hi - 1e-9)
    return hit


def _signal(theta: Tensor, protocol: Protocol, n_iso: int) -> Tensor:
    M0, B1, T1, T2, T2s = (theta[:, i].exp() for i in _LOG)
    dw = theta[:, 5] if theta.shape[1] > 5 else 0.0
    phi0 = theta[:, 6] if theta.shape[1] > 6 else 0.0
    return mpme_signal(protocol, M0, T1, T2, 1 / T2s - 1 / T2, dw, B1, phi0, n_iso=n_iso)


def _model(theta: Tensor, protocol: Protocol, n_iso: int) -> Tensor:
    """Complex model, real and imaginary parts stacked: [N, 2·n_scans·P·J]."""
    return torch.view_as_real(_signal(theta, protocol, n_iso)).flatten(1)


def _levenberg_marquardt(th: Tensor, model: Callable[[Tensor], Tensor], y: Tensor,
                         n_iter: int, h: float):
    """Projected LM with Marquardt scaling and forward-difference Jacobian, batched."""
    N, n = th.shape
    th = _project(th)
    lam = torch.full((N,), 1e-2, dtype=y.dtype)
    r = model(th) - y
    cost = (r**2).sum(1)
    eye = torch.eye(n, dtype=y.dtype)
    for _ in range(n_iter):
        J = torch.stack([(model(th + h * eye[k]) - y - r) / h for k in range(n)], 2)
        JtJ = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r[..., None])[..., 0]
        A = JtJ + lam[:, None, None] * torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2) + 1e-12)
        th_new = _project(th - torch.linalg.solve(A, g))
        r_new = model(th_new) - y
        cost_new = (r_new**2).sum(1)
        ok = torch.isfinite(cost_new) & (cost_new < cost)
        th = torch.where(ok[:, None], th_new, th)
        r = torch.where(ok[:, None], r_new, r)
        cost = torch.where(ok, cost_new, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-9, 1e9)
    return th, cost


def _init_magnitudes(init: dict[str, Tensor]) -> Tensor:
    return torch.stack([init[k].log() for k in _NAMES], 1)


def _output(th: Tensor, cost: Tensor) -> dict[str, Tensor]:
    out = {k: th[:, i].exp() for k, i in zip(_NAMES, _LOG)}
    out.update(cost=cost, at_box=_at_box(th), at_r2p_bound=th[:, 4] >= th[:, 3] - 1e-12)
    if th.shape[1] > 5:
        out.update(dw=th[:, 5], phi0=th[:, 6])
    return out


def joint_fit(
    S: Tensor,
    protocol: Protocol,
    init: dict[str, Tensor],
    *,
    n_iter: int = 40,
    n_iso: int = 256,
    h: float = 1e-6,
) -> dict[str, Tensor]:
    """ML estimates for complex voxel signals S [N, n_scans, P, J].

    ``init`` needs M0, B1, T1, T2, T2star, dw; phi0 is initialised from the data.

    Returns estimates plus ``cost`` (final ‖r‖²) and boolean flags ``at_box`` (a safeguard
    bound is active) and ``at_r2p_bound`` (R2′ = 0).
    """
    N = S.shape[0]
    y = torch.view_as_real(S).flatten(1)
    th = torch.cat([_init_magnitudes(init), init["dw"][:, None],
                    torch.zeros(N, 1, dtype=y.dtype)], 1)
    th = _project(th)
    # Global phase: least-squares optimal for the initial magnitudes and Δω.
    S0 = torch.view_as_complex(_model(th, protocol, n_iso).reshape(*S.shape, 2))
    th[:, 6] = torch.angle((S * S0.conj()).flatten(1).sum(1))
    th, cost = _levenberg_marquardt(th, lambda t: _model(t, protocol, n_iso), y, n_iter, h)
    return _output(th, cost)


def magnitude_fit(
    M: Tensor,
    protocol: Protocol,
    init: dict[str, Tensor],
    *,
    n_iter: int = 40,
    n_iso: int = 256,
    h: float = 1e-6,
) -> dict[str, Tensor]:
    """Gaussian least squares on magnitudes M [N, n_scans, P, J] (real, ≥ 0).

    ``init`` needs M0, B1, T1, T2, T2star. Returns the same fields as ``joint_fit``
    except dw and phi0.
    """
    y = M.flatten(1)
    model = lambda t: _signal(t, protocol, n_iso).abs().flatten(1)
    th, cost = _levenberg_marquardt(_init_magnitudes(init), model, y, n_iter, h)
    return _output(th, cost)
