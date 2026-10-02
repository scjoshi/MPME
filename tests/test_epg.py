import math

import pytest
import torch

from mpme.epg import epg_pathways, simulate_epg, steady_state_isochromat

# (flip deg, T1, T2, TR) in ms; includes high flip and long-T2 (CSF-like) cases
CASES = [(5, 1000, 80, 20), (30, 1000, 80, 20), (60, 800, 50, 10), (15, 4000, 2000, 30)]


def fisp_psif(alpha, T1, T2, TR):
    """Closed-form fully dephased FISP (F_0 after RF) and PSIF (echo at t = TR)."""
    E1, E2, ca = math.exp(-TR / T1), math.exp(-TR / T2), math.cos(alpha)
    p = 1 - E1 * ca - E2**2 * (E1 - ca)
    q = E2 * (1 - E1) * (1 + ca)
    r = math.sqrt(p * p - q * q)
    t = math.tan(alpha / 2)
    return t * (1 - (E1 - ca) * (1 - E2**2) / r), t * (1 - (1 - E1 * ca) * (1 - E2**2) / r)


@pytest.mark.parametrize("deg,T1,T2,TR", CASES)
def test_isochromat_matches_closed_form(deg, T1, T2, TR):
    a = math.radians(deg)
    F = steady_state_isochromat(a, T1, T2, TR, (0, -1), n_iso=4096).abs()
    fisp, psif = fisp_psif(a, T1, T2, TR)
    assert F[0].item() == pytest.approx(fisp, rel=1e-6)
    assert F[1].item() * math.exp(-TR / T2) == pytest.approx(psif, rel=1e-6)


@pytest.mark.parametrize("deg,T1,T2,TR", CASES)
def test_epg_matches_isochromat(deg, T1, T2, TR):
    a = math.radians(deg)
    paths = (2, 1, 0, -1, -2, -3)
    iso = steady_state_isochromat(a, T1, T2, TR, paths, n_iso=4096)
    F, _ = simulate_epg(a, T1, T2, TR, n_tr=int(8 * T1 / TR), K=400)
    torch.testing.assert_close(epg_pathways(F, paths), iso, atol=1e-5, rtol=0)


def test_off_resonance_only_adds_phase():
    a, T1, T2, TR = math.radians(20), 900.0, 70.0, 15.0
    paths = (1, 0, -1, -2)
    ref = steady_state_isochromat(a, T1, T2, TR, paths)
    for dw in (0.013, -0.2, 1.7):  # rad/ms
        F = steady_state_isochromat(a, T1, T2, TR, paths, dw=dw)
        phase = torch.exp(1j * torch.tensor(paths, dtype=torch.float64) * dw * TR)
        torch.testing.assert_close(F, ref * phase, atol=1e-10, rtol=0)


@pytest.mark.parametrize("phi", [0.0, 0.7, -2.1])
def test_rf_phase_convention(phi):
    a = math.radians(40)
    F, Z = simulate_epg(a, 1000.0, 100.0, 10.0, n_tr=1, K=4, phi=phi)
    F0 = epg_pathways(F, (0,))[0]
    expected = -1j * math.sin(a) * complex(math.cos(phi), math.sin(phi))
    assert abs(F0 - expected) < 1e-12
    assert Z[..., 4].real.item() == pytest.approx(math.cos(a))


def test_rf_phase_matches_between_solvers():
    a, phi = math.radians(25), 0.9
    iso = steady_state_isochromat(a, 1000.0, 80.0, 20.0, (1, 0, -1), phi=phi, n_iso=2048)
    F, _ = simulate_epg(a, 1000.0, 80.0, 20.0, n_tr=400, K=200, phi=phi)
    torch.testing.assert_close(epg_pathways(F, (1, 0, -1)), iso, atol=1e-6, rtol=0)


def test_epg_longitudinal_states_stay_hermitian():
    _, Z = simulate_epg(math.radians(35), 1000.0, 80.0, 20.0, n_tr=50, K=30, phi=0.4)
    torch.testing.assert_close(Z, torch.conj(Z.flip(-1)))


def test_diffusion_attenuates_echo_pathways_more():
    args = (math.radians(20), 1000.0, 80.0, 20.0)
    F0, _ = simulate_epg(*args, n_tr=400, K=100)
    Fd, _ = simulate_epg(*args, n_tr=400, K=100, D=3e-3, q=2 * math.pi / 1.0)  # 1 cycle/mm
    ratio = (epg_pathways(Fd, (0, -1)) / epg_pathways(F0, (0, -1))).abs()
    assert ratio[1] < ratio[0] <= 1.0


def test_broadcasting_and_gradients():
    alpha = torch.tensor([[0.1], [0.5]], dtype=torch.float64, requires_grad=True)
    T1 = torch.tensor([800.0, 1200.0, 4000.0], dtype=torch.float64, requires_grad=True)
    T2 = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)
    F = steady_state_isochromat(alpha, T1, T2, 20.0, (1, 0, -1))
    assert F.shape == (2, 3, 3)
    F.abs().sum().backward()
    for x in (alpha, T1, T2):
        assert torch.isfinite(x.grad).all() and (x.grad != 0).any()


def test_n_iso_guard():
    with pytest.raises(ValueError):
        steady_state_isochromat(0.1, 1000.0, 80.0, 20.0, (0, -8), n_iso=16)


def _leading_order(alpha, T1, T2, TR, paths, N=4096):
    """Proposition 2: m⊥(ψ) ≈ −iα / ((1 − E2 e^{iψ}) [1 + c H(ψ)]), c = α²/(2ρ)."""
    rho, E2 = TR / T1, math.exp(-TR / T2)
    z = torch.exp(2j * math.pi * torch.arange(N, dtype=torch.float64) / N)
    H = (1 - E2**2) / (1 - E2 * z).abs() ** 2
    m = -1j * alpha / ((1 - E2 * z) * (1 + alpha**2 / (2 * rho) * H))
    spec = torch.fft.fft(m) / N
    return torch.stack([spec[p % N] for p in paths])


def test_small_angle_leading_order_for_all_pathways():
    paths, T2, TR, errs = (2, 1, 0, -1, -2, -3), 70.0, 25.0, []
    for deg in (8.0, 4.0, 2.0):
        a = math.radians(deg)
        T1 = TR / (a**2 / 1.5)                                   # α²/ρ fixed
        ex = steady_state_isochromat(a, T1, T2, TR, paths, n_iso=4096)
        errs.append(((ex - _leading_order(a, T1, T2, TR, paths)).abs().max() / ex[2].abs()).item())
    # O(α²) relative remainder: error drops ~4× per halving of α
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.15)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.15)


@pytest.mark.parametrize("deg,factor", [(15.0, 1.3), (30.0, 2.0), (60.0, 0.5), (330.0, 1.1)])
def test_two_invariant_reduction(deg, factor):
    """Theorem 1: a_k depends on (α, T1) only through v·G_k(u) — exact for every pathway."""
    T1, T2, TR, dw = 1500.0, 70.0, 25.0, 0.03
    a1 = math.radians(deg)
    a2 = a1 * factor
    E1 = math.exp(-TR / T1)
    u = (E1 - math.cos(a1)) / (1 - E1 * math.cos(a1))
    E1b = (u + math.cos(a2)) / (1 + u * math.cos(a2))          # same u at the new angle
    assert 0 < E1b < 1
    T1b = -TR / math.log(E1b)
    v = lambda a, e: math.sin(a) * (1 - e) / (1 - e * math.cos(a))
    paths = (2, 1, 0, -1, -2, -3)
    A = steady_state_isochromat(a1, T1, T2, TR, paths, dw=dw, n_iso=2048)
    B = steady_state_isochromat(a2, T1b, T2, TR, paths, dw=dw, n_iso=2048)
    torch.testing.assert_close(B, A * v(a2, E1b) / v(a1, E1), rtol=1e-9, atol=1e-14)
