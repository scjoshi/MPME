import math

import pytest
import torch

from mpme.epg import steady_state_isochromat
from mpme.paper import paper_reconstruction
from mpme.sequence import paper_protocol
from mpme.signal import add_noise, mpme_signal
from test_analytic import random_tissue


@pytest.mark.parametrize("T1,T2", [(1500.0, 70.0), (850.0, 65.0), (4000.0, 1500.0)])
@pytest.mark.parametrize("deg", [15.0, 330.0])
def test_paper_equations_hold_on_simulated_states(T1, T2, deg):
    """Eqs. 7, 10, 11, 16 and 17 applied to exact steady-state F states (Eq. 18 signs)."""
    TR, a = 25.0, math.radians(deg)
    E1, E2, c = math.exp(-TR / T1), math.exp(-TR / T2), math.cos(a)
    m = steady_state_isochromat(a, T1, T2, TR, (1, 0, -1, -2), n_iso=8192).abs()
    Fp = {1: m[0].item(), 0: m[1].item(), -1: -m[2].item(), -2: -m[3].item()}
    Fm = {1: Fp[0] * E2, -1: Fp[-2] * E2, 0: Fp[-1] * E2}                  # F⁻_k = F⇒_{k−1}
    assert Fp[1] - Fp[-1] - Fm[1] + Fm[-1] == pytest.approx(0, abs=1e-12)   # Eq. 7
    for k in (1, -1):
        X = 1 - 2 * (Fm[k] - Fp[k]) / (Fm[k] + Fm[-k])                      # Eqs. 10–11
        assert TR / math.log((X * c - 1) / (X - c)) == pytest.approx(T1, rel=1e-6)  # Eq. 16
    M0 = abs(Fp[0] - c * Fm[0] + (Fm[0] - c * Fp[0]) * E1) / (math.sqrt(1 - c * c) * (1 - E1))
    assert M0 == pytest.approx(1.0, rel=1e-6)                                # Eq. 17


@pytest.mark.parametrize("scheme", [(1, 0, -1), (0, -1, -2)])
def test_noise_free_recovery(scheme):
    pr = paper_protocol(scheme)
    gt = random_tissue(500, seed=3)
    S = mpme_signal(pr, **gt, phi0=0.7)
    est = paper_reconstruction(S, pr)
    for k, v in gt.items():
        torch.testing.assert_close(est[k], v, rtol=1e-6, atol=1e-9, msg=k)


def test_known_b1_matches_solved_b1_when_noise_free():
    pr = paper_protocol()
    gt = random_tissue(200, seed=4)
    S = mpme_signal(pr, **gt)
    a, b = paper_reconstruction(S, pr), paper_reconstruction(S, pr, B1=gt["B1"])
    torch.testing.assert_close(a["T1"], b["T1"])
    torch.testing.assert_close(a["M0"], b["M0"])


def test_beta_rescales_flip_ratio():
    # Data simulated with α2/α1 = 1.1 × nominal is recovered when β = 1.1 (Eq. 19).
    from mpme.sequence import Protocol, Scan
    pr = paper_protocol()
    s1, s2 = pr.scans
    true = Protocol(pr.pathways, (s1, Scan(s2.flip_deg * 1.1, s2.TR, s2.echo_times)))
    S = mpme_signal(true, M0=1.0, T1=1200.0, T2=80.0, R2p=0.01, dw=0.0, B1=0.95)
    est = paper_reconstruction(S, pr, beta=1.1)
    assert est["T1"].item() == pytest.approx(1200.0, rel=1e-6)
    assert est["B1"].item() == pytest.approx(0.95, rel=1e-6)


def test_noise_gives_nan_or_finite_but_no_crash():
    pr = paper_protocol()
    S = mpme_signal(pr, 1.0, 1500.0, 70.0, 0.0024, 0.0, 1.0).expand(256, 2, 3, 3)
    est = paper_reconstruction(add_noise(S, 0.03, torch.Generator().manual_seed(0)), pr)
    assert torch.isfinite(est["T2"]).all() and torch.isfinite(est["B1"]).all()
