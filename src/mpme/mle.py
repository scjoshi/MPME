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


def _fix_columns(J: Tensor, fixed: Tensor | None) -> Tensor:
    return J if fixed is None else J * (~fixed).to(J.dtype)


def _damped_system(J: Tensor, r: Tensor, lam: Tensor, fixed: Tensor | None):
    """(JᵀJ + λ diag JᵀJ) Δ = Jᵀr, with fixed parameters given a unit row (Δ = 0)."""
    JtJ = J.transpose(1, 2) @ J
    g = (J.transpose(1, 2) @ r[..., None])[..., 0]
    A = JtJ + lam[:, None, None] * torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2) + 1e-12)
    if fixed is not None:
        A = A + torch.diag(fixed.to(A.dtype))
    return A, g


def _levenberg_marquardt(th: Tensor, model: Callable[[Tensor], Tensor], y: Tensor,
                         n_iter: int, h: float,
                         project: Callable[[Tensor], Tensor] = _project,
                         fixed: Tensor | None = None):
    """Projected LM with Marquardt scaling and forward-difference Jacobian, batched.

    ``fixed``: boolean mask over parameters held at their initial values.
    """
    N, n = th.shape
    th = project(th)
    lam = torch.full((N,), 1e-2, dtype=y.dtype)
    r = model(th) - y
    cost = (r**2).sum(1)
    eye = torch.eye(n, dtype=y.dtype)
    for _ in range(n_iter):
        J = torch.stack([(model(th + h * eye[k]) - y - r) / h if fixed is None or not fixed[k]
                         else torch.zeros_like(r) for k in range(n)], 2)
        A, g = _damped_system(J, r, lam, fixed)
        th_new = project(th - torch.linalg.solve(A, g))
        r_new = model(th_new) - y
        cost_new = (r_new**2).sum(1)
        ok = torch.isfinite(cost_new) & (cost_new < cost)
        th = torch.where(ok[:, None], th_new, th)
        r = torch.where(ok[:, None], r_new, r)
        cost = torch.where(ok, cost_new, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-9, 1e9)
    return th, cost


def _levenberg_marquardt_exact(th: Tensor, model_jac: Callable[[Tensor], tuple[Tensor, Tensor]],
                               y: Tensor, n_iter: int,
                               project: Callable[[Tensor], Tensor] = _project,
                               fixed: Tensor | None = None):
    """Projected LM with an exact Jacobian supplied together with the model value.

    Each trial evaluation returns (value, Jacobian), so an accepted step already provides
    the Jacobian for the next iteration: one model+Jacobian evaluation per iteration.
    """
    N, n = th.shape
    th = project(th)
    lam = torch.full((N,), 1e-2, dtype=y.dtype)
    m, J = model_jac(th)
    J = _fix_columns(J, fixed)
    r = m - y
    cost = (r**2).sum(1)
    for _ in range(n_iter):
        A, g = _damped_system(J, r, lam, fixed)
        th_new = project(th - torch.linalg.solve(A, g))
        m_new, J_new = model_jac(th_new)
        J_new = _fix_columns(J_new, fixed)
        r_new = m_new - y
        cost_new = (r_new**2).sum(1)
        ok = torch.isfinite(cost_new) & (cost_new < cost)
        th = torch.where(ok[:, None], th_new, th)
        r = torch.where(ok[:, None], r_new, r)
        J = torch.where(ok[:, None, None], J_new, J)
        cost = torch.where(ok, cost_new, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-9, 1e9)
    return th, cost


def _b1_projector(base: Callable[[Tensor], Tensor], b1_max: float | None,
                  fixed_B1: Tensor | None) -> Callable[[Tensor], Tensor]:
    """Wrap a projection with an upper bound on B1 and/or a fixed (given) B1 map."""
    if b1_max is None and fixed_B1 is None:
        return base
    log_max = None if b1_max is None else math.log(b1_max)
    log_fix = None if fixed_B1 is None else fixed_B1.log()

    def project(t: Tensor) -> Tensor:
        t = base(t)
        if log_max is not None:
            t[:, 1] = t[:, 1].clamp(max=log_max)
        if log_fix is not None:
            t[:, 1] = log_fix.to(t.dtype)
        return t
    return project


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
    jacobian: str = "fd",
    dtype: torch.dtype | None = None,
    b1_max: float | None = None,
    fixed_B1: Tensor | None = None,
) -> dict[str, Tensor]:
    """ML estimates for complex voxel signals S [N, n_scans, P, J].

    ``init`` needs M0, B1, T1, T2, T2star, dw; phi0 is initialised from the data.
    ``jacobian``: "fd" (forward differences, step ``h``; needs float64) or "implicit" (exact
    derivatives, ``fastjac.py``; one model+Jacobian evaluation per iteration, works in
    float32). ``dtype`` overrides the working precision.
    ``b1_max``: upper bound on B1 (e.g. 540/330 for the paper protocol, which excludes the
    exact complex alias with α2 > 540°). ``fixed_B1``: hold B1 at a given map (two-stage
    reconstruction); its Jacobian column is removed, the other parameters are fitted.

    Returns estimates plus ``cost`` (final ‖r‖²) and boolean flags ``at_box`` (a safeguard
    bound is active) and ``at_r2p_bound`` (R2′ = 0).
    """
    if dtype is not None:
        S = S.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
        init = {k: v.to(dtype) for k, v in init.items()}
    if fixed_B1 is not None:
        init = {**init, "B1": fixed_B1.to(init["B1"].dtype)}
    project = _b1_projector(_project, b1_max, fixed_B1)
    fixed = None
    if fixed_B1 is not None:
        fixed = torch.zeros(7, dtype=torch.bool)
        fixed[1] = True
    N = S.shape[0]
    y = torch.view_as_real(S).flatten(1)
    th = torch.cat([_init_magnitudes(init), init["dw"][:, None],
                    torch.zeros(N, 1, dtype=y.dtype)], 1)
    th = project(th)
    # Global phase: least-squares optimal for the initial magnitudes and Δω.
    S0 = torch.view_as_complex(_model(th, protocol, n_iso).reshape(*S.shape, 2).contiguous())
    th[:, 6] = torch.angle((S * S0.conj()).flatten(1).sum(1))
    if jacobian == "implicit":
        from .fastjac import complex_model_and_jacobian
        th, cost = _levenberg_marquardt_exact(
            th, lambda t: complex_model_and_jacobian(t, protocol, n_iso), y, n_iter,
            project, fixed)
    elif jacobian == "fd":
        th, cost = _levenberg_marquardt(th, lambda t: _model(t, protocol, n_iso), y, n_iter, h,
                                        project, fixed)
    else:
        raise ValueError("jacobian must be 'fd' or 'implicit'")
    return _output(th, cost)


def fid_phase_sign(S: Tensor, protocol: Protocol) -> Tensor:
    """Sign of Re Σ_echoes F0_scan1 · conj(F0_scan2), per voxel (±1).

    F_0 is proportional to sin α (up to a common phase), so this equals
    sign(sin α1 · sin α2) for the actual flip angles: negative for the paper protocol when
    α2 = 330°·B1 < 360°, positive when it exceeds 360° (paper, text after Eq. 15). Needs the
    relative phase of the FID images of the two scans; only its sign is used.
    """
    i0 = protocol.index(0)
    s = torch.sign((S[:, 0, i0, :] * S[:, 1, i0, :].conj()).sum(-1).real)
    return torch.where(s == 0, torch.ones_like(s), s)


def _branch_zeros(protocol: Protocol, B1_max: float) -> Tensor:
    """B1 values (ascending) where sin(B1·α_i) = 0 for some scan, up to B1_max."""
    z = [n * math.pi / sc.alpha for sc in protocol.scans[:2]
         for n in range(1, int(B1_max * sc.alpha / math.pi) + 2)]
    return torch.tensor(sorted(set(z)), dtype=torch.float64)


def _branch_interval(B1: Tensor, zeros: Tensor) -> tuple[Tensor, Tensor]:
    """The interval (lo, hi) between consecutive zeros that contains each B1."""
    edges = torch.cat([torch.zeros(1, dtype=zeros.dtype), zeros,
                       torch.full((1,), math.inf, dtype=zeros.dtype)])
    i = torch.searchsorted(edges, B1.contiguous(), right=True)
    return edges[i - 1], edges[i]


def _branch_sign(B1: Tensor, protocol: Protocol) -> Tensor:
    a1, a2 = (sc.alpha for sc in protocol.scans[:2])
    return torch.sign(torch.sin(B1 * a1) * torch.sin(B1 * a2))


def _branch_feasible_start(B1: Tensor, sign: Tensor, protocol: Protocol,
                           zeros: Tensor) -> Tensor:
    """Reflect infeasible B1 across the nearest zero (|F| is nearly symmetric there)."""
    lo, hi = _branch_interval(B1, zeros)
    nearest = torch.where((B1 - lo) < (hi - B1), lo, hi)
    bad = _branch_sign(B1, protocol) != sign
    B1 = torch.where(bad, (2 * nearest - B1).clamp_min(1e-3), B1)
    still = _branch_sign(B1, protocol) != sign                    # e.g. reflected past 0
    lo, hi = _branch_interval(nearest * (1 + 1e-6), zeros)
    return torch.where(still, 0.5 * (lo + torch.where(torch.isfinite(hi), hi, 2 * lo)), B1)


def magnitude_fit(
    M: Tensor,
    protocol: Protocol,
    init: dict[str, Tensor],
    *,
    fid_sign: Tensor | None = None,
    n_iter: int = 40,
    n_iso: int = 256,
    h: float = 1e-6,
    jacobian: str = "fd",
    dtype: torch.dtype | None = None,
    b1_max: float | None = None,
) -> dict[str, Tensor]:
    """Gaussian least squares on magnitudes M [N, n_scans, P, J] (real, ≥ 0).

    ``jacobian``: "fd" (forward differences, step ``h``; needs float64) or "implicit" (exact
    derivatives of the steady state, ``fastjac.py``; one model+Jacobian evaluation per
    iteration, works in float32). ``dtype`` overrides the working precision. ``b1_max``: upper
    bound on B1 (see ``joint_fit``).

    ``init`` needs M0, B1, T1, T2, T2star. ``fid_sign`` (±1 per voxel, from
    ``fid_phase_sign`` on the complex images) restricts B1 to the flip-angle branch on which
    sign(sin B1α1 · sin B1α2) matches it: magnitudes alone cannot distinguish
    α2 = 360° − x from 360° + x. The branch is enforced by projecting B1 into the interval
    between the zeros of sin(B1α_i) that contains the (feasible) starting value.

    Returns the same fields as ``joint_fit`` except dw and phi0, plus ``at_branch_bound``
    when a branch constraint was given.
    """
    if dtype is not None:
        M = M.to(dtype)
        init = {k: v.to(dtype) for k, v in init.items()}
        if fid_sign is not None:
            fid_sign = fid_sign.to(dtype)
    y = M.flatten(1)
    model = lambda t: _signal(t, protocol, n_iso).abs().flatten(1)
    base = _b1_projector(_project, b1_max, None)
    th = base(_init_magnitudes(init))
    project = base
    if fid_sign is not None:
        zeros = _branch_zeros(protocol, math.exp(_BOX[1][1])).to(th.dtype)
        th[:, 1] = _branch_feasible_start(th[:, 1].exp(), fid_sign, protocol, zeros).log()
        lo, hi = _branch_interval(th[:, 1].exp(), zeros)
        log_lo = (lo * (1 + 1e-6)).clamp_min(1e-12).log()
        log_hi = torch.where(torch.isfinite(hi), (hi * (1 - 1e-6)).log(),
                             torch.full_like(hi, math.inf))

        def project(t: Tensor) -> Tensor:
            t = base(t)
            t[:, 1] = torch.minimum(torch.maximum(t[:, 1], log_lo), log_hi)
            return t

    if jacobian == "implicit":
        from .fastjac import magnitude_model_and_jacobian
        th, cost = _levenberg_marquardt_exact(
            th, lambda t: magnitude_model_and_jacobian(t, protocol, n_iso), y, n_iter, project)
    elif jacobian == "fd":
        th, cost = _levenberg_marquardt(th, model, y, n_iter, h, project)
    else:
        raise ValueError("jacobian must be 'fd' or 'implicit'")
    out = _output(th, cost)
    if fid_sign is not None:
        out["at_branch_bound"] = (th[:, 1] <= log_lo + 1e-9) | (th[:, 1] >= log_hi - 1e-9)
    return out


def polynomial_basis(coords: Tensor, degree: int = 4) -> Tensor:
    """Monomials xⁱyʲ (i + j ≤ degree) at normalised voxel coordinates ``coords`` [V, 2] = (x, y)."""
    x, y = coords[:, 0], coords[:, 1]
    return torch.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], 1)


def poly_b1_gn_step(y: Tensor, protocol: Protocol, Phi: Tensor, th: Tensor, *,
                    coef_weights: Tensor | None = None, info_cap: float | None = None,
                    n_iso: int = 128, chunk: int = 4096):
    """Gauss–Newton step for the coefficients of log B1 = Φ·c, voxel parameters eliminated.

    ``y``: data [V, M] (real/imaginary interleaved); ``th``: current θ [V, 7] with
    θ[:, 1] = Φ·c. The full normal equations in (θ_own of every voxel, c) have arrowhead
    form; eliminating each voxel's 6×6 block (Schur complement) leaves
    (Σ_v s_v φ_v φ_vᵀ) δ = Σ_v q_v φ_v with s_v = j_bᵀj_b − w_vᵀU_v⁻¹w_v and
    q_v = j_bᵀr − w_vᵀU_v⁻¹J_θᵀr (U_v = J_θᵀJ_θ, w_v = J_θᵀj_b, j_b = ∂r/∂log B1, r = model − y).
    The update is c ← c − δ. ``coef_weights`` (per voxel) and ``info_cap`` (cap on s_v as a
    multiple of its median) weight the voxels. Returns (δ, reduced matrix, s).
    """
    from .fastjac import complex_model_and_jacobian

    own = torch.tensor([0, 2, 3, 4, 5, 6])
    s_all, q_all = [], []
    for a in range(0, th.shape[0], chunk):
        m, J = complex_model_and_jacobian(th[a:a + chunk], protocol, n_iso)
        r = (m - y[a:a + chunk]).double()
        J = J.double()
        Jt, jb = J[:, :, own], J[:, :, 1]
        U = Jt.transpose(1, 2) @ Jt + 1e-12 * torch.eye(6, dtype=torch.float64)
        wv = (Jt.transpose(1, 2) @ jb[..., None])[..., 0]
        gt = (Jt.transpose(1, 2) @ r[..., None])[..., 0]
        Uw = torch.linalg.solve(U, wv)
        s_all.append((jb * jb).sum(1) - (wv * Uw).sum(1))
        q_all.append((jb * r).sum(1) - (Uw * gt).sum(1))
    s, q = torch.cat(s_all).clamp_min(0), torch.cat(q_all)
    scale = torch.ones_like(s) if coef_weights is None else coef_weights.double()
    if info_cap is not None:
        scale = scale * (info_cap * s.median() / s.clamp_min(1e-300)).clamp(max=1.0)
    red = (Phi * (scale * s)[:, None]).T @ Phi
    rhs = Phi.T @ (scale * q)
    step = torch.linalg.solve(red + 1e-10 * torch.diag(red.diagonal()), rhs)
    return step, red, s


def polynomial_b1_fit(
    S: Tensor,
    protocol: Protocol,
    basis: Tensor,
    init: dict[str, Tensor],
    coef: Tensor,
    *,
    n_outer: int = 10,
    n_inner: int = 5,
    n_iso: int = 128,
    dtype: torch.dtype = torch.float32,
    info_cap: float = 10.0,
    huber: float = 3.0,
    chunk: int = 4096,
) -> dict[str, Tensor]:
    """Joint ML with B1 shared through a polynomial: log B1(x, y) = basis · coef.

    Every voxel keeps its own (log M0, log T1, log T2, log T2*, Δω, φ0); the polynomial
    coefficients are common to all voxels. Gauss–Newton with variable projection, using the
    exact (implicit) Jacobian:

    1. With B1 fixed at the polynomial, the voxel parameters are refined by projected LM
       (``n_inner`` iterations, ``joint_fit(fixed_B1=...)``).
    2. The Gauss–Newton step for the coefficients comes from the arrowhead normal equations
       by a Schur complement: per voxel s_v = j_bᵀj_b − w_vᵀU_v⁻¹w_v (U_v = J_θᵀJ_θ,
       w_v = J_θᵀj_b, j_b = ∂r/∂log B1) is the profiled information about log B1, and
       (Σ_v s_v φ_v φ_vᵀ) δ = Σ_v (j_bᵀr − w_vᵀU_v⁻¹J_θᵀr) φ_v gives the step δ.
    3. The step is accepted if the (robustly weighted) total cost falls after step 1 is
       repeated at the new coefficients; otherwise it is halved (up to 4 times).

    Robustness: each voxel's information s_v is capped at ``info_cap`` × its median (no voxel
    can dominate, e.g. near the 360° crossing), and voxels whose cost exceeds ``huber``² ×
    the median cost are down-weighted (Huber weights on √cost).

    ``S``: complex voxel signals [V, n_scans, P, J]; ``basis``: [V, K] (``polynomial_basis``);
    ``init``: per-voxel starting values (M0, B1, T1, T2, T2star, dw[, phi0]); ``coef``: [K]
    starting coefficients for log B1. Returns per-voxel estimates (B1 = the polynomial),
    ``coef``, its covariance ``coef_cov`` (unscaled: multiply by σ²), ``weight`` and the
    total cost per outer iteration (``history``).
    """
    V = S.shape[0]
    Phi = basis.to(torch.float64)
    coef = coef.to(torch.float64).clone()
    S = S.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
    y = torch.view_as_real(S).flatten(1)

    def fit_voxels(c: Tensor, start: dict[str, Tensor]) -> dict[str, Tensor]:
        B1 = (Phi @ c).exp().to(dtype)
        outs = [joint_fit(S[a:a + chunk], protocol, {k: v[a:a + chunk] for k, v in start.items()},
                          n_iter=n_inner, n_iso=n_iso, jacobian="implicit", dtype=dtype,
                          fixed_B1=B1[a:a + chunk]) for a in range(0, V, chunk)]
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def theta(est: dict[str, Tensor]) -> Tensor:
        return torch.stack([est["M0"].log(), est["B1"].log(), est["T1"].log(), est["T2"].log(),
                            est["T2star"].log(), est["dw"], est["phi0"]], 1).to(dtype)

    def weights(cost: Tensor) -> Tensor:
        z = (cost / cost.median()).sqrt()
        return torch.where(z <= huber, torch.ones_like(z), huber / z).double()

    est = fit_voxels(coef, init)
    w = weights(est["cost"])
    total = (w * est["cost"].double()).sum()
    history = [total.item()]
    reduced = None
    for _ in range(n_outer):
        step, red, _ = poly_b1_gn_step(y, protocol, Phi, theta(est), coef_weights=w,
                                       info_cap=info_cap, n_iso=n_iso, chunk=chunk)
        reduced = red
        accepted = False
        for _half in range(5):
            trial_coef = coef - step
            # move each voxel along the small-angle family before refitting (T1 ∝ B1⁻², M0 ∝ B1⁻¹)
            k = (Phi @ coef - Phi @ trial_coef).exp().to(dtype)
            start = {**est, "T1": est["T1"] * k**2, "M0": est["M0"] * k}
            trial = fit_voxels(trial_coef, start)
            t_total = (w * trial["cost"].double()).sum()
            if torch.isfinite(t_total) and t_total < total:
                coef, est, total, accepted = trial_coef, trial, t_total, True
                break
            step = step / 2
        history.append(total.item())
        if not accepted:
            break
        w = weights(est["cost"])
        total = (w * est["cost"].double()).sum()
    out = dict(est)
    out.update(coef=coef, coef_cov=torch.linalg.inv(reduced), weight=w, history=history)
    return out
