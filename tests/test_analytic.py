import math

import pytest
import torch

from mpme.analytic import analytic_reconstruction, estimate_b0, fid_echo_amplitudes
from mpme.epg import steady_state_isochromat
from mpme.sequence import default_protocol, paper_protocol
from mpme.signal import add_noise, mpme_signal

PR = default_protocol()


def random_tissue(n, seed=0):
    """Random tissue; B1 avoids α2 = 360° ± 2°, where the scan-2 FID vanishes and the
    flip-angle branch is undetermined."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, dtype=torch.float64, generator=g)
    B1 = u(0.7, 1.4)
    B1 = torch.where((B1 * 330 - 360).abs() < 2, B1 + 0.02, B1)
    return dict(M0=u(0.5, 1.5), T1=u(300, 3000), T2=u(30, 150), R2p=u(0.002, 0.035),
                dw=u(-0.63, 0.63), B1=B1)


@pytest.mark.parametrize("deg,T1,T2,TR", [(15, 850, 65, 25), (330, 850, 65, 25), (70, 2000, 120, 10)])
def test_closed_form_amplitudes_match_solver(deg, T1, T2, TR):
    a = math.radians(deg)
    fid, echo = fid_echo_amplitudes(*(torch.tensor(v, dtype=torch.float64) for v in (a, T1, T2, TR)))
    ref = steady_state_isochromat(a, T1, T2, TR, (0, -1), n_iso=4096).abs()
    torch.testing.assert_close(torch.stack([fid, echo]), ref, rtol=1e-6, atol=0)


@pytest.mark.parametrize("scheme", [(1, 0, -1), (0, -1, -2)])
@pytest.mark.parametrize("decay_scans", [(0,), (0, 1)])
def test_noise_free_recovery(decay_scans, scheme):
    pr = paper_protocol(scheme)
    gt = random_tissue(500)
    S = mpme_signal(pr, **gt, phi0=1.3)
    est = analytic_reconstruction(S, pr, decay_scans=decay_scans)
    for k, v in gt.items():
        torch.testing.assert_close(est[k], v, rtol=1e-5, atol=1e-8, msg=k)
    torch.testing.assert_close(est["T2star"], 1 / (1 / gt["T2"] + gt["R2p"]))


def test_b0_unambiguous_range():
    dt = 2.0  # closest echo spacing within a pathway in the default protocol (ms)
    dw = torch.tensor([-0.95, -0.5, 0.5, 0.95], dtype=torch.float64) * math.pi / dt
    S = mpme_signal(PR, M0=1.0, T1=850.0, T2=65.0, R2p=0.02, dw=dw, B1=1.0)
    torch.testing.assert_close(estimate_b0(S, PR), dw)


def test_pure_noise_gives_finite_bounded_output():
    S = add_noise(torch.zeros(64, 2, 3, 3, dtype=torch.complex128), 1.0,
                  torch.Generator().manual_seed(1))
    est = analytic_reconstruction(S, PR)
    for k, v in est.items():
        assert torch.isfinite(v).all(), k
    assert (est["T2"] >= 1.0).all() and (est["T2"] <= 5000.0).all()
