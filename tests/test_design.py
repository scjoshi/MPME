import math

import numpy as np
import pytest
import torch

from mpme.crlb import crlb, fisher_information
from mpme.design import (alias_roots, alias_t1, bounds, ernst_offset, f_ratio, ratio_bias,
                         single_scan_fisher)
from mpme.fastjac import complex_model_and_jacobian
from mpme.sequence import Protocol, Scan, paper_protocol

PR = paper_protocol()
E, TR = PR.scans[0].echo_times, 25.0
WM = dict(M0=1.0, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3)


@pytest.fixture
def float64_default():
    """mpme.crlb builds its parameter vector with the default dtype; float64 avoids rounding it."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("a1,a2,b1", [(15, 330, 1.0), (20, 270, 0.8), (13, 329, 1.15)])
def test_decomposition_matches_two_scan_information(a1, a2, b1, float64_default):
    """F(α1, α2; B1) = F1(B1 α1) + F1(B1 α2) (Proposition decomp)."""
    F = single_scan_fisher([b1 * a1, b1 * a2], WM, TR, E, PR.pathways).sum(0)
    P = Protocol(PR.pathways, (Scan(a1, TR, E), Scan(a2, TR, E)))
    ref = fisher_information(P, {**WM, "B1": b1}, n_iso=256)
    torch.testing.assert_close(F, ref, rtol=1e-10, atol=1e-12 * ref.abs().max().item())
    assert bounds(F)[2].item() == pytest.approx(crlb(ref)["T1"], rel=1e-8)


@pytest.mark.parametrize("x,T1,T2", [(15, 850, 66), (330, 1350, 90), (95, 4000, 1500), (200, 600, 50)])
def test_t1_column_is_combination_of_b1_and_m0(x, T1, T2):
    """J_T1 = h (J_B1 / c − J_M0) for one scan (Proposition perscan)."""
    P = Protocol(PR.pathways, (Scan(1.0, TR, E),))
    th = torch.tensor([[0.0, math.log(x), math.log(T1), math.log(T2), math.log(0.8 * T2), 0.03, 0.2]],
                      dtype=torch.float64)
    _, J = complex_model_and_jacobian(th, P, 256)
    J = J[0]
    tau = TR / T1
    _, c = ernst_offset(x, T1, TR)
    pred = tau / (2 * math.sinh(tau)) * (J[:, 1] / c - J[:, 0])
    assert ((J[:, 2] - pred).norm() / J[:, 2].norm()).item() < 1e-12


@pytest.mark.parametrize("a1,a2", [(15, 330), (20, 276), (17, 254)])
def test_closed_form_bounds_and_ratio_bias(a1, a2):
    """Proposition closed (= reduced 3×3 bound) and Proposition ratio (= full first-order bias)."""
    x = np.array([a1, a2], dtype=float)
    F = single_scan_fisher(x, WM, TR, E, PR.pathways)
    _, c = ernst_offset(x, WM["T1"], TR)
    K = [np.array([[f[1, 1] / ci**2, f[1, 0] / ci], [f[1, 0] / ci, f[0, 0]]]) for f, ci in
         zip(F.numpy(), c)]
    A = np.array([[K[0][0, 0], 0, K[0][0, 1]], [0, K[1][0, 0], K[1][0, 1]],
                  [K[0][0, 1], K[1][0, 1], K[0][1, 1] + K[1][1, 1]]])
    Qi = np.linalg.inv(A[:2, :2] - np.outer(A[:2, 2], A[2, :2]) / A[2, 2])
    L = c[1] - c[0]
    tau = TR / WM["T1"]
    h = tau / (2 * math.sinh(tau))
    v = np.array([c[1], -c[0]])
    red = torch.linalg.inv(F.sum(0)[:3, :3]).diagonal().sqrt()
    assert math.sqrt(Qi[0, 0] + Qi[1, 1] - 2 * Qi[0, 1]) / abs(L) == pytest.approx(red[1].item(), rel=1e-9)
    assert math.sqrt(v @ Qi @ v) / (h * abs(L)) == pytest.approx(red[2].item(), rel=1e-9)
    full = ratio_bias(F.sum(0), F[1])
    assert full[2].item() == pytest.approx(-c[0] * c[1] / (h * L), rel=1e-8)
    assert full[1].item() == pytest.approx(c[1] / L, rel=1e-8)


def test_aliases_of_published_protocol():
    """15/330, true B1 = 1: magnitude alias 1.199 (T1' 588 ms for WM), complex alias 2.008."""
    assert alias_roots(1.0, 15, 330, (0.6, 1.4)) == []
    (m,) = alias_roots(1.0, 15, 330, (0.6, 1.4), magnitude=True)
    assert m == pytest.approx(1.19912, abs=1e-4)
    (c,) = alias_roots(1.0, 15, 330, (0.3, 3.0))
    assert c == pytest.approx(2.00785, abs=1e-4)
    assert f_ratio(c, 15, 330) == pytest.approx(f_ratio(1.0, 15, 330), rel=1e-9)
    assert alias_t1(m, 1.0, 15, 850.0, TR) == pytest.approx(588.1, abs=0.2)
    assert alias_t1(c, 1.0, 15, 850.0, TR) == pytest.approx(203.3, abs=0.2)
