"""Joint maximum-likelihood fit of the full MPME signal model.

Fits θ = (log M0, log B1, log T1, log T2, log T2*, Δω, φ0) to all scans, pathways and
echoes of the complex data at once (Gaussian noise ⇒ nonlinear least squares), with
voxel-wise Levenberg–Marquardt. Same parameterisation as ``crlb.py``, so its spread can be
compared directly with the Cramér–Rao bound. Needs a starting point, e.g. the output of
``analytic.analytic_reconstruction``.

Constraints (projected Levenberg–Marquardt): the Lorentzian model needs R2′ ≥ 0, i.e.
T2* ≤ T2, enforced by projecting log T2* ≤ log T2 after every step. Safeguard box:
B1 ∈ [e⁻¹, e], T1 ∈ [10, 20000] ms, T2 ∈ [1, 5000] ms. Active constraints are reported per
voxel, because an estimate on a constraint is not covered by the interior, unbiased
Cramér–Rao bound.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .sequence import Protocol
from .signal import mpme_signal

_LOG = (0, 1, 2, 3, 4)  # indices of log-parameters in θ
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


def _model(theta: Tensor, protocol: Protocol, n_iso: int) -> Tensor:
    M0, B1, T1, T2, T2s = (theta[:, i].exp() for i in _LOG)
    S = mpme_signal(protocol, M0, T1, T2, 1 / T2s - 1 / T2, theta[:, 5], B1, theta[:, 6],
                    n_iso=n_iso)
    return torch.view_as_real(S).flatten(1)


def joint_fit(
    S: Tensor,
    protocol: Protocol,
    init: dict[str, Tensor],
    *,
    n_iter: int = 40,
    n_iso: int = 256,
    h: float = 1e-6,
) -> dict[str, Tensor]:
    """ML estimates for voxel signals S [N, n_scans, P, J] (complex).

    ``init`` needs M0, B1, T1, T2, T2star, dw; phi0 is initialised from the data.

    Returns estimates plus ``cost`` (final ‖r‖²) and boolean flags ``at_box`` (a safeguard
    bound is active) and ``at_r2p_bound`` (R2′ = 0).
    """
    N = S.shape[0]
    y = torch.view_as_real(S).flatten(1)
    th = torch.stack([init["M0"].log(), init["B1"].log(), init["T1"].log(),
                      init["T2"].log(), init["T2star"].log(), init["dw"],
                      torch.zeros(N, dtype=y.dtype)], 1)
    th = _project(th)
    # Global phase: least-squares optimal for the initial magnitudes and Δω.
    th[:, 6] = 0.0
    S0 = torch.view_as_complex(_model(th, protocol, n_iso).reshape(*S.shape, 2))
    th[:, 6] = torch.angle((S * S0.conj()).flatten(1).sum(1))

    lam = torch.full((N,), 1e-2, dtype=y.dtype)
    r = _model(th, protocol, n_iso) - y
    cost = (r**2).sum(1)
    eye = torch.eye(th.shape[1], dtype=y.dtype)
    for _ in range(n_iter):
        J = torch.stack([(_model(th + h * eye[k], protocol, n_iso) - y - r) / h
                         for k in range(th.shape[1])], 2)                    # [N, M, 7]
        JtJ = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r[..., None])[..., 0]
        A = JtJ + lam[:, None, None] * torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2) + 1e-12)
        th_new = _project(th - torch.linalg.solve(A, g))
        r_new = _model(th_new, protocol, n_iso) - y
        cost_new = (r_new**2).sum(1)
        ok = torch.isfinite(cost_new) & (cost_new < cost)
        th = torch.where(ok[:, None], th_new, th)
        r = torch.where(ok[:, None], r_new, r)
        cost = torch.where(ok, cost_new, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-9, 1e9)

    out = {k: th[:, i].exp() for k, i in zip(("M0", "B1", "T1", "T2", "T2star"), _LOG)}
    out.update(dw=th[:, 5], phi0=th[:, 6], cost=cost,
               at_box=_at_box(th),
               at_r2p_bound=th[:, 4] >= th[:, 3] - 1e-12)
    return out
