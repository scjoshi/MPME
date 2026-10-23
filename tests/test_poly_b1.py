"""Joint ML with a polynomial B1 map (mle.polynomial_b1_fit)."""
import math

import pytest
import torch

from mpme.mle import _model, poly_b1_gn_step, polynomial_b1_fit, polynomial_basis
from mpme.phantom import polyfit_2d
from mpme.sequence import Protocol, Scan, paper_protocol

BASE = paper_protocol()
PR = Protocol(BASE.pathways, (BASE.scans[0], Scan(85.0, BASE.scans[0].TR, BASE.scans[0].echo_times)))


def _voxels(V, degree, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(V, 2, generator=g, dtype=torch.float64) * 1.6 - 0.8
    Phi = polynomial_basis(xy, degree)
    coef = torch.zeros(Phi.shape[1], dtype=torch.float64)
    coef[0], coef[1:3] = 0.05, torch.tensor([0.08, -0.06])
    if degree >= 2:
        coef[3:6] = torch.tensor([-0.15, 0.04, -0.12])
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(V, generator=g, dtype=torch.float64)
    T2 = u(50, 110)
    th = torch.stack([u(0.8, 1.2).log(), Phi @ coef, u(600, 2000).log(), T2.log(),
                      (0.7 * T2).log(), u(-0.05, 0.05), u(-0.5, 0.5)], 1)
    return xy, Phi, coef, th


def test_basis_matches_polyfit_2d():
    """polynomial_basis spans the same space as polyfit_2d (same coordinates and degree)."""
    H = 32
    c = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) / (H / 2)
    y, x = torch.meshgrid(c, c, indexing="ij")
    Phi = polynomial_basis(torch.stack([x.flatten(), y.flatten()], 1))
    true = Phi @ torch.linspace(-0.3, 0.4, Phi.shape[1], dtype=torch.float64)
    mask = torch.ones(H, H, dtype=torch.bool)
    fit = polyfit_2d(true.reshape(H, H), mask, n_reject=0).flatten()
    assert (fit - true).abs().max().item() < 1e-10


@pytest.mark.parametrize("seed", [0, 1])
def test_schur_step_equals_dense_gauss_newton(seed):
    """The reduced (Schur-complement) step for the coefficients equals the coefficient part of
    the full Gauss–Newton step with a central-difference Jacobian over all 6V + K parameters."""
    V, n_iso = 6, 128
    _, Phi, coef, th0 = _voxels(V, 1, seed)
    y = _model(th0, PR, n_iso)
    y = y + 0.01 * torch.randn(y.shape, generator=torch.Generator().manual_seed(seed + 7),
                               dtype=torch.float64)
    c = coef + torch.tensor([0.02, -0.01, 0.015], dtype=torch.float64)      # off the optimum
    own = [0, 2, 3, 4, 5, 6]
    x0 = torch.cat([th0[:, own].flatten(), c])

    def resid(x):
        th = torch.zeros(V, 7, dtype=torch.float64)
        th[:, own] = x[:6 * V].reshape(V, 6)
        th[:, 1] = Phi @ x[6 * V:]
        return (_model(th, PR, n_iso) - y).flatten()

    h = 1e-6
    J = torch.stack([(resid(x0 + h * e) - resid(x0 - h * e)) / (2 * h)
                     for e in torch.eye(len(x0), dtype=torch.float64)], 1)
    r = resid(x0)
    dense = torch.linalg.solve(J.T @ J, J.T @ r)[6 * V:]
    th = torch.zeros(V, 7, dtype=torch.float64)
    th[:, own] = th0[:, own]
    th[:, 1] = Phi @ c
    step, red, s = poly_b1_gn_step(y, PR, Phi, th, n_iso=n_iso)
    torch.testing.assert_close(step, dense, rtol=1e-5, atol=1e-8)
    assert (s >= 0).all()


def test_recovers_polynomial_b1_noise_free():
    """Noise-free data, 15°/85°: the fit returns the true coefficients from a perturbed start."""
    V = 150
    _, Phi, coef, th = _voxels(V, 2, seed=3)
    S = torch.view_as_complex(_model(th, PR, 256).reshape(V, 2, 3, 3, 2).contiguous())
    start_coef = coef + torch.tensor([0.06, -0.04, 0.03, 0.05, -0.03, 0.04], dtype=torch.float64)
    k = (th[:, 1] - Phi @ start_coef).exp()                                 # B1 / B1_start
    init = {"M0": th[:, 0].exp() * k, "B1": (Phi @ start_coef).exp(), "T1": th[:, 2].exp() * k**2,
            "T2": th[:, 3].exp(), "T2star": th[:, 4].exp(), "dw": th[:, 5]}
    out = polynomial_b1_fit(S, PR, Phi, init, start_coef, n_outer=15, n_inner=8, n_iso=256,
                            dtype=torch.float64)
    assert (out["coef"] - coef).abs().max().item() < 1e-4
    assert ((out["B1"] / th[:, 1].exp()).log().abs().max().item()) < 1e-4
    assert ((out["T1"] / th[:, 2].exp()).log().abs().max().item()) < 1e-3
    assert all(b <= a + 1e-12 for a, b in zip(out["history"], out["history"][1:]))
