"""Device handling: every helper tensor must follow the device of the inputs (e.g. CUDA).

The leak test needs no GPU: inputs live on the CPU while torch's default device is set to
'meta', so any tensor created without an explicit device lands on 'meta' and the computation
fails. The agreement test runs only when CUDA is available.
"""
import pytest
import torch

from mpme.analytic import analytic_reconstruction
from mpme.design import bounds, single_scan_fisher
from mpme.fastjac import complex_model_and_jacobian
from mpme.mle import fid_phase_sign, joint_fit, magnitude_fit, polynomial_b1_fit, polynomial_basis
from mpme.paper import paper_reconstruction
from mpme.sequence import paper_protocol
from mpme.signal import mpme_signal

PR = paper_protocol()


def _data(V=40, device="cpu", dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    u = lambda lo, hi: (lo + (hi - lo) * torch.rand(V, generator=g, dtype=dtype)).to(device)
    xy = (torch.rand(V, 2, generator=g, dtype=dtype) * 1.6 - 0.8).to(device)
    p = dict(M0=u(0.8, 1.2), B1=u(0.9, 1.1), T1=u(700, 1500), T2=u(50, 90), dw=u(-0.05, 0.05))
    p["T2star"] = 0.7 * p["T2"]
    S = mpme_signal(PR, p["M0"], p["T1"], p["T2"], 1 / p["T2star"] - 1 / p["T2"], p["dw"], p["B1"],
                    0.2, n_iso=64)
    return p, S, polynomial_basis(xy, 1)


def _theta(p):
    return torch.stack([p["M0"].log(), p["B1"].log(), p["T1"].log(), p["T2"].log(),
                        p["T2star"].log(), p["dw"], torch.zeros_like(p["dw"])], 1)


CALLS = {
    "signal": lambda p, S, Phi: mpme_signal(PR, p["M0"], p["T1"], p["T2"], 1 / p["T2star"] - 1 / p["T2"],
                                            p["dw"], p["B1"], 0.2, n_iso=64),
    "jacobian": lambda p, S, Phi: complex_model_and_jacobian(_theta(p), PR, 64)[1],
    "analytic": lambda p, S, Phi: analytic_reconstruction(S, PR)["T1"],
    "paper": lambda p, S, Phi: paper_reconstruction(S, PR)["T1"],
    "joint_fit": lambda p, S, Phi: joint_fit(S, PR, p, n_iter=2, n_iso=64, jacobian="implicit")["T1"],
    "joint_fit_fixed": lambda p, S, Phi: joint_fit(S, PR, p, n_iter=2, n_iso=64, jacobian="implicit",
                                                   fixed_B1=p["B1"])["T1"],
    "magnitude_fit": lambda p, S, Phi: magnitude_fit(S.abs(), PR, p, fid_sign=fid_phase_sign(S, PR),
                                                     n_iter=2, n_iso=64, jacobian="implicit")["T1"],
    "design_bounds": lambda p, S, Phi: bounds(single_scan_fisher(
        p["B1"][:4, None] * torch.tensor([15.0, 330.0], dtype=p["B1"].dtype, device=p["B1"].device),
        dict(T1=1000.0, T2=70.0, T2star=50.0), 25.0, PR.scans[0].echo_times, PR.pathways, n_iso=64
    ).sum(1)),
    "polynomial_b1_fit": lambda p, S, Phi: polynomial_b1_fit(
        S, PR, Phi, p, torch.zeros(Phi.shape[1], dtype=Phi.dtype, device=Phi.device), n_outer=1,
        n_inner=1, n_iso=64, dtype=Phi.dtype)["coef"],
}


@pytest.mark.parametrize("name", list(CALLS))
def test_no_tensor_created_on_default_device(name):
    p, S, Phi = _data()
    old = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        out = CALLS[name](p, S, Phi)
    finally:
        torch.set_default_device(old)
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", list(CALLS))
def test_cuda_matches_cpu(name):
    torch.manual_seed(0)
    p, S, Phi = _data()
    ref = CALLS[name](p, S, Phi)
    pg, Sg, Phig = _data(device="cuda")
    out = CALLS[name](pg, Sg, Phig)
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-6, atol=1e-9, equal_nan=True)
