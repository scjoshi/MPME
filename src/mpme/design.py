"""Two-scan flip-angle design: Fisher information, Ernst-angle offsets, aliases, ratio bias.

Both scans share TR and readout; only the nominal flip angles α1, α2 differ. Three exact facts
make an exhaustive search over (α1, α2) cheap:

* the scans are independent, so the information of a design at transmit scale B1 is
  F(α1, α2; B1) = F₁(B1 α1) + F₁(B1 α2), with F₁(x) the information of ONE scan at actual flip
  angle x;
* the log-B1 column of a single-scan Jacobian is ∂/∂ln x, so F₁ depends on x alone;
* if the two pulses do not have the nominal ratio (scan 2 actual angle × (1 + ε)), the first-order
  bias of a fit that assumes the nominal ratio is ε · F⁻¹ F₁(B1 α2)[:, B1].

Parameters θ = (log M0, log B1, log T1, log T2, log T2*, Δω, φ0) as in ``mpme.crlb``; noise
σ = 1 per real/imaginary part, so bounds are per unit σ/M0 (with M0 = 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .fastjac import complex_model_and_jacobian
from .sequence import Protocol, Scan

PARAMS = ("M0", "B1", "T1", "T2", "T2star", "dw", "phi0")


def single_scan_fisher(angles_deg, tissue: Mapping[str, float], TR: float,
                       echo_times: tuple[tuple[float, ...], ...], pathways: Sequence[int],
                       n_iso: int = 256) -> Tensor:
    """Information of one scan at actual flip angles ``angles_deg`` (any shape) → [..., 7, 7].

    Evaluated with a 1° nominal pulse and B1 = x, so the log-B1 column is ∂/∂ln x. Δω and φ0 do
    not affect the information; they are set from ``tissue`` if present (default 0).
    """
    x = torch.as_tensor(angles_deg, dtype=torch.float64)
    shape = x.shape
    x = x.flatten()
    proto = Protocol(tuple(pathways), (Scan(1.0, TR, echo_times),))
    th = torch.empty(len(x), 7, dtype=torch.float64)
    th[:, 0] = math.log(tissue.get("M0", 1.0))
    th[:, 1] = x.log()
    th[:, 2] = math.log(tissue["T1"])
    th[:, 3] = math.log(tissue["T2"])
    th[:, 4] = math.log(tissue["T2star"])
    th[:, 5] = tissue.get("dw", 0.0)
    th[:, 6] = tissue.get("phi0", 0.0)
    _, J = complex_model_and_jacobian(th, proto, n_iso)
    return (J.transpose(1, 2) @ J).reshape(*shape, 7, 7)


def bounds(F: Tensor, known: Sequence[str] = ()) -> Tensor:
    """sqrt(diag F⁻¹) over the free parameters for a batch [..., 7, 7]; inf where singular.

    Returns [..., 7] with NaN in the columns of ``known`` parameters.
    """
    keep = [i for i, p in enumerate(PARAMS) if p not in known]
    Fk = F[..., keep, :][..., keep]
    scale = Fk.diagonal(dim1=-2, dim2=-1).clamp_min(1e-300).sqrt()
    Fn = Fk / (scale[..., :, None] * scale[..., None, :])            # unit diagonal
    C, info = torch.linalg.inv_ex(Fn)
    d = C.diagonal(dim1=-2, dim2=-1) / scale**2
    ok = (info == 0) & (d > 0).all(-1) & torch.isfinite(d).all(-1)
    d = torch.where(ok[..., None], d, torch.full_like(d, math.inf))
    out = torch.full((*F.shape[:-2], 7), math.nan, dtype=F.dtype)
    out[..., keep] = d.sqrt()
    return out


def ratio_bias(F: Tensor, F2: Tensor) -> Tensor:
    """First-order bias of θ̂ per unit relative error of the scan-2 flip angle: F⁻¹ F₂[:, B1]."""
    return torch.linalg.solve(F, F2[..., :, 1:2])[..., 0]


def ernst_offset(angles_deg, T1: float, TR: float):
    """Offset w = ln|tan(x/2)| − ln tan(α_E/2) of a sample from the Ernst angle, and the B1 lever
    c = d ln|tan(x/2)|/d ln B1 = x / sin x (x in radians)."""
    x = np.radians(np.asarray(angles_deg, dtype=float))
    zeta = math.sqrt(math.tanh(TR / (2 * T1)))
    return np.log(np.abs(np.tan(x / 2))) - math.log(zeta), x / np.sin(x)


def f_ratio(b, a1_deg: float, a2_deg: float):
    """ξ₂/ξ₁ = tan(b α2/2)/tan(b α1/2): the B1-only quantity two scans determine."""
    b = np.asarray(b, dtype=float)
    return np.tan(np.radians(b * a2_deg) / 2) / np.tan(np.radians(b * a1_deg) / 2)


def alias_roots(b_true: float, a1_deg: float, a2_deg: float, box=(0.5, 1.6),
                magnitude: bool = False, n_grid: int = 4001) -> list[float]:
    """Other transmit scales b′ in ``box`` with the same ξ₂/ξ₁ as ``b_true`` (complex data), or the
    same |ξ₂/ξ₁| (magnitude data). Admissibility (ζ′ < 1) is not checked here (see ``admissible``).

    f is continuous between its singularities b α2 = 180° + 360° m (poles) and b α1 = 360° m; the
    roots of f(b′) − f(b_true) are bracketed on each branch and refined by bisection.
    """
    target = f_ratio(b_true, a1_deg, a2_deg)
    g = (lambda b: np.abs(f_ratio(b, a1_deg, a2_deg)) - abs(target)) if magnitude else \
        (lambda b: f_ratio(b, a1_deg, a2_deg) - target)
    lo, hi = box
    sing = [(180 + 360 * m) / a2_deg for m in range(int(hi * a2_deg / 360) + 2)]
    sing += [360 * m / a1_deg for m in range(1, int(hi * a1_deg / 360) + 2)]
    edges = sorted({lo, hi, *[s for s in sing if lo < s < hi]})
    roots = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        pad = 1e-9 * (e1 - e0)
        b = np.linspace(e0 + pad, e1 - pad, max(int(n_grid * (e1 - e0) / (hi - lo)), 50))
        y = g(b)
        for i in np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]:
            u, v = b[i], b[i + 1]
            for _ in range(60):
                m = 0.5 * (u + v)
                if np.sign(g(m)) == np.sign(g(u)):
                    u = m
                else:
                    v = m
            r = 0.5 * (u + v)
            if abs(r - b_true) > 1e-6 and abs(g(r)) < 1e-6 * max(1.0, abs(target)):
                roots.append(float(r))
    return roots


def alias_t1(b_alias: float, b_true: float, a1_deg: float, T1: float, TR: float) -> float:
    """T1 of the alias at b_alias: tan(α_E′/2) = tan(α_E/2)·|tan(b′α1/2)/tan(b α1/2)| (a sign flip of
    both scans is absorbed by φ0 → φ0 + π); inf if inadmissible (tan(α_E′/2) ∉ (0, 1))."""
    zeta = math.sqrt(math.tanh(TR / (2 * T1)))
    zp = zeta * abs(math.tan(math.radians(b_alias * a1_deg) / 2) / math.tan(math.radians(b_true * a1_deg) / 2))
    if not 0 < zp < 1:
        return math.inf
    return TR / (2 * math.atanh(zp * zp))
