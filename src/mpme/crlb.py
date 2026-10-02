"""Cramér–Rao lower bounds and Fisher-information conditioning for MPME.

Noise model: every complex echo image gets circular Gaussian noise, standard deviation σ_i
per real/imaginary component in scan i. For real-valued parameters θ, the Fisher
information is

    F = Σ_i (1/σ_i²) · Re(J_iᴴ J_i),      J_i = ∂S_i/∂θ   (all pathways and echoes of scan i)

and any unbiased estimator has Cov(θ̂) ⪰ F⁻¹. Positive parameters (M0, B1, T1, T2, T2*)
are parameterised by their logarithm, so sqrt(diag F⁻¹) is directly a *relative* standard
deviation. Δω (rad/ms) and the global phase φ0 (rad) are linear nuisance parameters.

With σ expressed in units of M0 (σ = M0/SNR), every bound scales as 1/SNR, so the module
works with σ = 1 and results are "per unit σ/M0".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .sequence import Protocol
from .signal import mpme_signal

PARAMS = ("M0", "B1", "T1", "T2", "T2star", "dw", "phi0")
LOG_PARAMS = frozenset({"M0", "B1", "T1", "T2", "T2star"})


def _theta0(tissue: Mapping[str, float], params: Sequence[str]) -> Tensor:
    vals = [torch.log(torch.tensor(float(tissue[p]))) if p in LOG_PARAMS else
            torch.tensor(float(tissue.get(p, 0.0))) for p in params]
    return torch.stack(vals).to(torch.float64)


def jacobian(protocol: Protocol, tissue: Mapping[str, float], params: Sequence[str] = PARAMS,
             n_iso: int = 512) -> Tensor:
    """∂S/∂θ as a complex tensor [n_scans, P, J, n_params] (log-parameters where applicable)."""
    params = tuple(params)

    def signal(theta: Tensor) -> Tensor:
        v = dict(tissue)
        for p, t in zip(params, theta):
            v[p] = t.exp() if p in LOG_PARAMS else t
        R2p = 1 / v["T2star"] - 1 / v["T2"]
        S = mpme_signal(protocol, v["M0"], v["T1"], v["T2"], R2p, v.get("dw", 0.0), v["B1"],
                        v.get("phi0", 0.0), n_iso=n_iso)
        return torch.view_as_real(S)

    J = torch.autograd.functional.jacobian(signal, _theta0(tissue, params), vectorize=True)
    return torch.view_as_complex(J.movedim(-1, -2).contiguous())               # [I, P, Jt, n]


def fisher_information(
    protocol: Protocol,
    tissue: Mapping[str, float],
    params: Sequence[str] = PARAMS,
    scan_sigma: Sequence[float] | None = None,
    n_iso: int = 512,
) -> Tensor:
    """Fisher information [n_params, n_params] for one voxel, σ in units of M0.

    ``scan_sigma`` gives σ_i per scan (default 1 for all scans).
    """
    J = jacobian(protocol, tissue, params, n_iso)                          # [I, P, Jt, n]
    w = torch.ones(J.shape[0], dtype=torch.float64) if scan_sigma is None else \
        1.0 / torch.as_tensor(scan_sigma, dtype=torch.float64) ** 2
    Jf = J.flatten(1, 2)                                                    # [I, P·Jt, n]
    return torch.einsum("i,ikn,ikm->nm", w, Jf.conj(), Jf).real


def crlb(F: Tensor, params: Sequence[str] = PARAMS, known: Sequence[str] = ()) -> dict[str, float]:
    """sqrt(diag F⁻¹) for the unknown parameters, treating ``known`` ones as fixed."""
    keep = [i for i, p in enumerate(params) if p not in known]
    C = torch.linalg.inv(F[keep][:, keep])
    return {params[i]: C[n, n].sqrt().item() for n, i in enumerate(keep)}


def covariance(F: Tensor) -> Tensor:
    return torch.linalg.inv(F)


def correlation(C: Tensor) -> Tensor:
    """Correlation matrix of a covariance (here: the CRLB covariance F⁻¹)."""
    d = C.diagonal().sqrt()
    return C / d[:, None] / d[None, :]


def normalized_information(F: Tensor) -> Tensor:
    """D^{-1/2} F D^{-1/2} with D = diag F: unit diagonal, removes parameter scaling.

    Its eigenvalues lie in (0, n]; a near-zero eigenvalue means a combination of parameters
    that the data barely constrain, whatever the units.
    """
    d = F.diagonal().sqrt()
    return F / d[:, None] / d[None, :]


def conditioning(F: Tensor) -> dict[str, Tensor]:
    """Eigen-analysis of the normalised information matrix.

    Returns eigenvalues (ascending), eigenvectors (columns), the condition number, and the
    *variance inflation* per parameter: (F⁻¹)_ii · F_ii, i.e. how much the bound grows
    because the other parameters are unknown (1 = no confounding).
    """
    Fn = normalized_information(F)
    evals, evecs = torch.linalg.eigh(Fn)
    vif = torch.linalg.inv(F).diagonal() * F.diagonal()
    return {"eigenvalues": evals, "eigenvectors": evecs,
            "condition": evals[-1] / evals[0], "vif": vif}


def fixed_parameter_bias(F: Tensor, params: Sequence[str], fixed: str) -> dict[str, float]:
    """First-order bias of the other parameters per unit error in ``fixed``.

    If ``fixed`` is held at a wrong value (error δ) and the rest are fitted, the
    maximum-likelihood estimates shift by δθ_r = −F_rr⁻¹ F_r,fixed · δ. For log-parameters
    this is d log θ_r / d log θ_fixed.
    """
    j = params.index(fixed)
    r = [i for i in range(len(params)) if i != j]
    g = -torch.linalg.solve(F[r][:, r], F[r, j])
    return {params[i]: g[n].item() for n, i in enumerate(r)}


def rician_information_factor(a: Tensor, n_grid: int = 4000) -> Tensor:
    """g(a) = σ² × Fisher information about a Rician amplitude A, as a function of a = A/σ.

    For m ~ Rice(A, σ): I_A = (E[m² R(mA/σ²)²] − A²)/σ⁴ with R = I1/I0, i.e.
    g(a) = E[x² R(xa)²] − a² for x = m/σ. g → 1 for a → ∞ (Gaussian limit) and g → 0 for
    a → 0 (a signal buried in noise carries almost no information about its size).
    Computed by trapezoidal quadrature over a window of ±14σ around the amplitude, with
    exponentially scaled Bessel functions.
    """
    from scipy.special import i0e, i1e
    import numpy as np

    a_np = np.atleast_1d(a.detach().cpu().numpy().astype(np.float64))
    out = np.empty_like(a_np)
    for n, av in enumerate(a_np.ravel()):
        x = np.linspace(max(1e-9, av - 14.0), av + 14.0, n_grid)   # pdf ≈ N(a, 1) for large a
        z = x * av
        pdf = x * np.exp(-0.5 * (x - av) ** 2) * i0e(z)      # Rice pdf, I0 scaled by e^{-z}
        R = i1e(z) / i0e(z)
        out.ravel()[n] = np.trapezoid(x**2 * R**2 * pdf, x) - av**2
    return torch.as_tensor(out.reshape(a_np.shape), dtype=torch.float64)


MAG_PARAMS = ("M0", "B1", "T1", "T2", "T2star")


def fisher_information_magnitude(
    protocol: Protocol,
    tissue: Mapping[str, float],
    snr: float,
    params: Sequence[str] = MAG_PARAMS,
    gaussian: bool = False,
    n_iso: int = 512,
) -> Tensor:
    """Fisher information of magnitude data |S + n| at SNR(M0) = M0/σ.

    Exact Rician information (``gaussian=False``) or the high-SNR Gaussian approximation
    (g = 1). Unlike the complex case, the Rician information is not simply proportional to
    1/σ², so the SNR must be given. Δω and φ0 do not affect magnitudes and are excluded.
    """
    J = jacobian(protocol, tissue, params, n_iso)                          # complex [I,P,Jt,n]
    S = mpme_signal(protocol, tissue["M0"], tissue["T1"], tissue["T2"],
                    1 / tissue["T2star"] - 1 / tissue["T2"], tissue.get("dw", 0.0),
                    tissue["B1"], tissue.get("phi0", 0.0), n_iso=n_iso)
    A = S.abs().flatten()
    dA = (torch.conj(S)[..., None] * J).real.flatten(0, 2) / A[:, None]     # ∂|S|/∂θ
    sigma = tissue["M0"] / snr
    g = torch.ones_like(A) if gaussian else rician_information_factor(A / sigma)
    return (dA.T * g) @ dA / sigma**2
