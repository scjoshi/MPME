import math

import pytest
import torch

from mpme.crlb import (PARAMS, conditioning, correlation, covariance, crlb, fisher_information,
                       fixed_parameter_bias, jacobian)
from mpme.sequence import Protocol, Scan, paper_protocol
from mpme.signal import mpme_signal

PR = paper_protocol()
TIS = dict(M0=1.0, B1=1.0, T1=1500.0, T2=70.0, T2star=60.0, dw=0.05, phi0=0.3)


def with_flip2(flip2):
    s1, s2 = PR.scans
    return Protocol(PR.pathways, (s1, Scan(flip2, s2.TR, s2.echo_times)))


def test_jacobian_matches_finite_differences():
    J = jacobian(PR, TIS)
    h = 1e-6
    for n, p in enumerate(PARAMS):
        def sig(delta):
            v = dict(TIS)
            v[p] = v[p] * math.exp(delta) if p in ("M0", "B1", "T1", "T2", "T2star") else v[p] + delta
            return mpme_signal(PR, v["M0"], v["T1"], v["T2"], 1 / v["T2star"] - 1 / v["T2"],
                               v["dw"], v["B1"], v["phi0"])
        fd = (sig(h) - sig(-h)) / (2 * h)
        torch.testing.assert_close(J[..., n], fd, rtol=1e-5, atol=1e-9, msg=p)


def test_information_is_symmetric_positive_definite():
    F = fisher_information(PR, TIS)
    torch.testing.assert_close(F, F.T)
    assert torch.linalg.eigvalsh(F).min() > 0


def test_bounds_scale_with_noise():
    a = crlb(fisher_information(PR, TIS))
    b = crlb(fisher_information(PR, TIS, scan_sigma=(2.0, 2.0)))
    for k in a:
        assert b[k] == pytest.approx(2 * a[k], rel=1e-10)


def test_known_parameter_never_loosens_bounds():
    F = fisher_information(PR, TIS)
    a, b = crlb(F), crlb(F, known=("B1",))
    assert all(b[k] <= a[k] + 1e-12 for k in b)


def test_small_flip_scaling_symmetry():
    # With 15°/30°, fixing B1 wrong by ε shifts T1 by ≈ −2ε and M0 by ≈ −ε:
    # the signal is nearly invariant to (M0/k, B1·k, T1/k²).
    g = fixed_parameter_bias(fisher_information(with_flip2(30.0), TIS), PARAMS, "B1")
    assert g["T1"] == pytest.approx(-2.0, abs=0.1)
    assert g["M0"] == pytest.approx(-1.0, abs=0.05)
    R = correlation(covariance(fisher_information(with_flip2(30.0), TIS)))
    assert R[1, 2].abs() > 0.999


def test_330_equals_minus_30_when_b1_is_known():
    # Magnitudes depend on α only through |F(α)|, and 330° ≡ −30°; with B1 known the two
    # protocols carry identical information. The 330° pulse helps only through ∂α/∂B1.
    a = crlb(fisher_information(PR, TIS), known=("B1",))
    b = crlb(fisher_information(with_flip2(30.0), TIS), known=("B1",))
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-6)


def test_paper_protocol_far_better_conditioned():
    c330 = conditioning(fisher_information(PR, TIS))["condition"]
    c30 = conditioning(fisher_information(with_flip2(30.0), TIS))["condition"]
    assert c30 / c330 > 1000


def test_joint_ml_recovers_noise_free_parameters():
    from mpme.analytic import analytic_reconstruction
    from mpme.mle import joint_fit
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, 1.0, 0.3).expand(4, 2, 3, 3)
    init = analytic_reconstruction(S, PR)
    init = {k: v * (1.05 if k in ("T1", "M0", "B1") else 1.0) for k, v in init.items()}
    fit = joint_fit(S, PR, init, n_iso=512, n_iter=30)
    torch.testing.assert_close(fit["T1"], torch.full((4,), 1500.0, dtype=torch.float64), rtol=1e-6, atol=0)
    torch.testing.assert_close(fit["B1"], torch.ones(4, dtype=torch.float64), rtol=1e-6, atol=0)


def test_330_and_30_differ_when_actual_angles_differ():
    # Proposition 3 is about actual angles: at B1 = 1.1 they are 363° vs 33°.
    tis = {**TIS, "B1": 1.1}
    a = crlb(fisher_information(PR, tis), known=("B1",))["T1"]
    b = crlb(fisher_information(with_flip2(30.0), tis), known=("B1",))["T1"]
    assert abs(a - b) / b > 0.01


def test_joint_ml_respects_r2p_nonnegative():
    from mpme.analytic import analytic_reconstruction
    from mpme.mle import joint_fit
    from mpme.signal import add_noise
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, 1.0, 0.3).expand(32, 2, 3, 3)
    Sn = add_noise(S, 1 / 200, torch.Generator().manual_seed(2))
    fit = joint_fit(Sn, PR, analytic_reconstruction(Sn, PR), n_iter=15)
    assert (fit["T2star"] <= fit["T2"] * (1 + 1e-12)).all()
    assert fit["at_r2p_bound"].dtype == torch.bool


def test_rician_information_factor_limits():
    from mpme.crlb import rician_information_factor
    g = rician_information_factor(torch.tensor([0.0, 0.05, 20.0], dtype=torch.float64))
    assert g[0].item() == pytest.approx(0.0, abs=1e-6)
    assert g[1].item() < 0.01
    assert g[2].item() == pytest.approx(1.0, abs=2e-3)


def test_magnitude_bound_approaches_complex_bound_at_high_snr():
    from mpme.crlb import MAG_PARAMS, fisher_information_magnitude
    snr = 1e5
    bc = crlb(fisher_information(PR, TIS))
    bm = crlb(fisher_information_magnitude(PR, TIS, snr), MAG_PARAMS)
    for k in ("T1", "B1", "T2"):
        assert bm[k] == pytest.approx(bc[k] / snr, rel=1e-3)


def test_magnitude_fit_recovers_noise_free_parameters():
    from mpme.analytic import analytic_reconstruction
    from mpme.mle import magnitude_fit
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, 1.0, 0.3).expand(4, 2, 3, 3)
    init = {k: v * (1.05 if k in ("T1", "M0", "B1") else 1.0)
            for k, v in analytic_reconstruction(S, PR).items()}
    fit = magnitude_fit(S.abs(), PR, init, n_iso=512, n_iter=30)
    torch.testing.assert_close(fit["T1"], torch.full((4,), 1500.0, dtype=torch.float64), rtol=1e-6, atol=0)


def test_fid_phase_sign_matches_branch():
    from mpme.mle import fid_phase_sign
    B1 = torch.tensor([0.9, 1.0, 1.05, 1.15, 1.3], dtype=torch.float64)
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, B1, 0.3)
    expected = torch.sign(torch.sin(B1 * PR.scans[0].alpha) * torch.sin(B1 * PR.scans[1].alpha))
    torch.testing.assert_close(fid_phase_sign(S, PR), expected)


def test_branch_constraint_recovers_from_wrong_branch_start():
    from mpme.mle import fid_phase_sign, magnitude_fit
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, 1.0, 0.3).expand(4, 2, 3, 3)
    ones = torch.ones(4, dtype=torch.float64)
    # start on the mirror branch: α2 = 330°·1.18 ≈ 390° instead of 330°
    init = dict(M0=ones, B1=1.18 * ones, T1=1100.0 * ones, T2=70.0 * ones, T2star=60.0 * ones)
    fit = magnitude_fit(S.abs(), PR, init, fid_sign=fid_phase_sign(S, PR), n_iso=512, n_iter=40)
    torch.testing.assert_close(fit["B1"], ones, rtol=1e-6, atol=0)
    torch.testing.assert_close(fit["T1"], 1500.0 * ones, rtol=1e-6, atol=0)
    assert not fit["at_branch_bound"].any()


def test_implicit_jacobian_matches_autograd():
    from mpme.fastjac import magnitude_model_and_jacobian
    from mpme.mle import _signal
    th = torch.tensor([[0.1, -0.05, math.log(1200.0), math.log(80.0), math.log(60.0)],
                       [-0.2, 0.08, math.log(700.0), math.log(50.0), math.log(45.0)]],
                      dtype=torch.float64)
    m, J = magnitude_model_and_jacobian(th, PR, n_iso=256)
    ref = _signal(th, PR, 256).abs().flatten(1)
    Ja = torch.autograd.functional.jacobian(
        lambda t: _signal(t, PR, 256).abs().flatten(1).sum(0), th, vectorize=True).permute(1, 0, 2)
    torch.testing.assert_close(m, ref, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(J, Ja, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-6), (torch.float32, 1e-3)])
def test_implicit_magnitude_fit_recovers_parameters(dtype, tol):
    from mpme.analytic import analytic_reconstruction
    from mpme.mle import fid_phase_sign, magnitude_fit
    S = mpme_signal(PR, 1.0, 1500.0, 70.0, 1 / 60 - 1 / 70, 0.05, 1.0, 0.3).expand(4, 2, 3, 3)
    init = {k: v * (1.05 if k in ("T1", "M0", "B1") else 1.0)
            for k, v in analytic_reconstruction(S, PR).items()}
    fit = magnitude_fit(S.abs(), PR, init, fid_sign=fid_phase_sign(S, PR), n_iso=256,
                        n_iter=30, jacobian="implicit", dtype=dtype)
    assert fit["T1"].dtype == dtype
    torch.testing.assert_close(fit["T1"].double(), torch.full((4,), 1500.0, dtype=torch.float64),
                               rtol=tol, atol=0)
