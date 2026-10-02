"""Benchmark the magnitude fit: forward-difference vs implicit Jacobian, float64 vs float32.

Same synthetic voxels as the earlier timing (seed 0, SNR(M0) = 1000, paper protocol, FID
sign on). Reports wall time, ms/voxel, the extrapolated time for a 256×256 slice, and the
agreement of the estimates with the forward-difference float64 reference.

    python scripts/benchmark_fit.py [N]
"""

import sys
import time

import torch

from mpme.analytic import analytic_reconstruction
from mpme.mle import fid_phase_sign, magnitude_fit
from mpme.sequence import paper_protocol
from mpme.signal import add_noise, mpme_signal

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
pr = paper_protocol()
g = torch.Generator().manual_seed(0)
T1 = 600 + 2400 * torch.rand(N, generator=g, dtype=torch.float64)
T2 = 40 + 80 * torch.rand(N, generator=g, dtype=torch.float64)
B1 = 0.8 + 0.3 * torch.rand(N, generator=g, dtype=torch.float64)
S = mpme_signal(pr, 1.0, T1, T2, 0.004, 0.05, B1, 0.3)
Sn = add_noise(S, 1 / 1000, g)
M = Sn.abs()
init = analytic_reconstruction(Sn, pr)
sign = fid_phase_sign(Sn, pr)
print(f"{N} voxels, {torch.get_num_threads()} threads")

variants = [
    ("finite diff., f64, 256 iso", dict(jacobian="fd", n_iso=256)),
    ("implicit,     f64, 256 iso", dict(jacobian="implicit", n_iso=256)),
    ("implicit,     f64, 128 iso", dict(jacobian="implicit", n_iso=128)),
    ("implicit,     f32, 256 iso", dict(jacobian="implicit", n_iso=256, dtype=torch.float32)),
    ("implicit,     f32, 128 iso", dict(jacobian="implicit", n_iso=128, dtype=torch.float32)),
]
ref = None
for name, kw in variants:
    t = time.time()
    fit = magnitude_fit(M, pr, init, fid_sign=sign, n_iter=15, **kw)
    dt = time.time() - t
    T1h, B1h = fit["T1"].double(), fit["B1"].double()
    if ref is None:
        ref = (T1h, B1h, dt)
        agree = "(reference)"
    else:
        dT = (T1h / ref[0]).log().abs()
        dB = (B1h / ref[1]).log().abs()
        agree = (f"|Δlog T1| median {dT.median():.1e} max {dT.max():.1e}; "
                 f"|Δlog B1| max {dB.max():.1e}")
    err = (T1h / T1).log()
    print(f"{name}: {dt:6.1f} s  {dt / N * 1e3:5.2f} ms/voxel  256x256: {dt / N * 65536 / 60:5.1f} min"
          f"  speed-up {ref[2] / dt:4.1f}x  | log T1 error sd {err.std():.4f} | {agree}")
