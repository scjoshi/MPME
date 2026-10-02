import math

import pytest
import torch

from mpme.epg import steady_state_isochromat
from mpme.sequence import Protocol, Scan, default_protocol
from mpme.signal import add_noise, mpme_signal

PR = default_protocol()
TISSUE = dict(M0=1.0, T1=850.0, T2=65.0, R2p=0.02, dw=0.1, B1=1.0)


def test_shape_and_broadcasting():
    T1 = torch.tensor([800.0, 1200.0, 3000.0], dtype=torch.float64)
    S = mpme_signal(PR, **{**TISSUE, "T1": T1})
    assert S.shape == (3, 2, 3, 3) and S.is_complex()


def test_magnitude_independent_of_off_resonance():
    a = mpme_signal(PR, **{**TISSUE, "dw": 0.0}).abs()
    b = mpme_signal(PR, **{**TISSUE, "dw": 0.37}).abs()
    torch.testing.assert_close(a, b)


def test_echo_phase_advances_at_dw():
    S = mpme_signal(PR, **TISSUE)
    dphi = torch.angle(S[..., 1:] * S[..., :-1].conj())
    dt = torch.stack([PR.echo_times(i).diff(dim=-1) for i in range(2)])
    torch.testing.assert_close(dphi, TISSUE["dw"] * dt)


def test_fid_and_echo_log_slopes():
    S = mpme_signal(PR, **TISSUE)[0].abs().log()
    t = PR.echo_times(0)
    slope = lambda p: (S[PR.index(p), -1] - S[PR.index(p), 0]) / (t[PR.index(p), -1] - t[PR.index(p), 0])
    R2, R2p = 1 / TISSUE["T2"], TISSUE["R2p"]
    assert slope(0).item() == pytest.approx(-(R2 + R2p))
    assert slope(-1).item() == pytest.approx(-(R2 - R2p))
    assert slope(1).item() == pytest.approx(-(R2 + R2p))


def test_amplitude_at_zero_decay_is_steady_state():
    # With T2 decay and R2′ removed analytically, S/M0 must equal a_p.
    S = mpme_signal(PR, **{**TISSUE, "dw": 0.0, "M0": 2.5})[1]
    scan = PR.scans[1]
    t = PR.echo_times(1)
    p = torch.tensor(PR.pathways, dtype=torch.float64)[:, None]
    undo = torch.exp(t / TISSUE["T2"] + TISSUE["R2p"] * (t + p * scan.TR).abs())
    a = steady_state_isochromat(scan.alpha, 850.0, 65.0, scan.TR, PR.pathways, n_iso=512)
    torch.testing.assert_close(S * undo / 2.5, a[:, None].expand(-1, 3))


def test_b1_scales_flip_angle():
    pr2 = Protocol(PR.pathways, tuple(Scan(s.flip_deg * 1.2, s.TR, s.echo_times) for s in PR.scans))
    torch.testing.assert_close(mpme_signal(PR, **{**TISSUE, "B1": 1.2}), mpme_signal(pr2, **TISSUE))


def test_noise_level():
    S = torch.zeros(200_000, dtype=torch.complex128)
    n = add_noise(S, 0.3, torch.Generator().manual_seed(0))
    assert n.real.std().item() == pytest.approx(0.3, rel=0.01)
    assert n.imag.std().item() == pytest.approx(0.3, rel=0.01)
