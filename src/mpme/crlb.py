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
