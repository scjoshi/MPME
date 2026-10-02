"""2D digital brain phantom and simulated MPME acquisitions.

The phantom is an axial-slice-like arrangement of tissue classes built from smooth shapes:
an outer CSF rim, cortical grey matter with a gyrated inner boundary, white matter, two
lateral ventricles, deep grey-matter nuclei, and a white-matter lesion. Each class has
literature-typical 3 T relaxation values, modulated by a weak smooth texture so that tissues
are not perfectly uniform. Smooth fields model the transmit B1⁺ (centre brightening), the
off-resonance (a quadratic background plus a frontal susceptibility hot spot), the receive
sensitivity C, and the receive phase φ0.

Acquired images follow the forward model voxel by voxel (no partial volume within a voxel),
with complex Gaussian noise. The published design acquires scan 2 only at low resolution;
in 2D this is modelled by keeping the central fraction of the phase-encoding (ky) lines of
scan 2 and zero-filling, applied after the noise is added (same per-sample noise, fewer
samples).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .sequence import Protocol
from .signal import mpme_signal

# label: (name, T1 ms, T2 ms, T2* ms, proton density)
TISSUES = {
    1: ("CSF", 4000.0, 1500.0, 900.0, 1.00),
    2: ("GM", 1350.0, 90.0, 60.0, 0.82),
    3: ("WM", 850.0, 66.0, 50.0, 0.69),
    4: ("deep GM", 1130.0, 58.0, 40.0, 0.75),
    5: ("lesion", 1500.0, 120.0, 80.0, 0.85),
}


@dataclass
class Phantom:
    labels: Tensor          # [H, W] int, 0 = background
    M0: Tensor              # C · PD
    T1: Tensor
    T2: Tensor
    T2star: Tensor
    B1: Tensor
    dw: Tensor              # rad/ms
    phi0: Tensor            # rad
    mask: Tensor            # [H, W] bool

    @property
    def R2p(self) -> Tensor:
        return 1 / self.T2star - 1 / self.T2


def _smooth_noise(shape, sigma_px: float, generator: torch.Generator) -> Tensor:
    """Unit-variance Gaussian random field with correlation length ``sigma_px`` (FFT filter)."""
    H, W = shape
    z = torch.randn(H, W, generator=generator, dtype=torch.float64)
    ky = torch.fft.fftfreq(H, dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(W, dtype=torch.float64)[None, :]
    filt = torch.exp(-2 * (math.pi * sigma_px) ** 2 * (kx**2 + ky**2))
    f = torch.fft.ifft2(torch.fft.fft2(z) * filt).real
    return (f - f.mean()) / f.std()


def make_phantom(n: int = 256, seed: int = 0, texture: float = 0.03) -> Phantom:
    """Build an n×n phantom. ``texture``: SD of the smooth log-modulation of T1, T2, T2*, PD."""
    g = torch.Generator().manual_seed(seed)
    c = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) / (n / 2)     # [-1, 1]
    y, x = torch.meshgrid(c, c, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    th = torch.atan2(y, x)

    def ellipse(cx, cy, ax, ay, angle=0.0):
        ca, sa = math.cos(angle), math.sin(angle)
        u = ((x - cx) * ca + (y - cy) * sa) / ax
        v = (-(x - cx) * sa + (y - cy) * ca) / ay
        return u**2 + v**2 <= 1

    head = ellipse(0, 0, 0.80, 0.92)
    brain = ellipse(0, 0, 0.76, 0.88)
    # white-matter boundary with gyral folding: radius modulated by angular harmonics
    rho = torch.sqrt((x / 0.66) ** 2 + (y / 0.78) ** 2)
    fold = 1 - 0.045 * torch.sin(9 * th) - 0.03 * torch.sin(14 * th + 1.0) - 0.02 * torch.sin(5 * th + 2)
    wm = rho <= 0.86 * fold

    labels = torch.zeros(n, n, dtype=torch.long)
    labels[head] = 1                                     # CSF rim
    labels[brain] = 2                                    # cortex
    labels[wm] = 3
    for sx in (-1, 1):                                   # lateral ventricles
        labels[ellipse(0.13 * sx, -0.05, 0.07, 0.30, 0.18 * sx)] = 1
        labels[ellipse(0.30 * sx, 0.12, 0.09, 0.12, 0.3 * sx)] = 4       # deep GM nuclei
    labels[ellipse(-0.42, -0.30, 0.07, 0.05, 0.6)] = 5   # lesion in WM

    mask = labels > 0
    T1 = torch.zeros(n, n, dtype=torch.float64)
    T2, T2s, PD = T1.clone(), T1.clone(), T1.clone()
    for lab, (_, t1, t2, t2s, pd) in TISSUES.items():
        m = labels == lab
        T1[m], T2[m], T2s[m], PD[m] = t1, t2, t2s, pd
    tex = [_smooth_noise((n, n), 6.0, g) for _ in range(4)]
    T1 = T1 * torch.exp(texture * tex[0])
    T2 = T2 * torch.exp(texture * tex[1])
    T2s = torch.minimum(T2s * torch.exp(texture * tex[2]), 0.98 * T2)  # keep R2′ > 0
    PD = PD * torch.exp(texture * tex[3])

    # transmit field: centre brightening typical of 3 T heads, slight left-right asymmetry
    B1 = 1.15 - 0.33 * r**2 + 0.04 * x
    # off-resonance: quadratic background ± ~20 Hz plus a frontal hot spot (~ +60 Hz)
    f_hz = 20 * (x**2 - y**2) + 60 * torch.exp(-((x) ** 2 + (y + 0.75) ** 2) / (2 * 0.12**2))
    dw = 2 * math.pi * f_hz / 1000
    # receive sensitivity and phase: smooth
    C = 1.0 + 0.15 * y + 0.10 * x**2
    phi0 = 0.6 * x + 0.3 * y**2

    z = torch.zeros_like(T1)
    keep = lambda a: torch.where(mask, a, z)
    return Phantom(labels=labels, M0=keep(C * PD), T1=keep(T1), T2=keep(T2), T2star=keep(T2s),
                   B1=keep(B1), dw=keep(dw), phi0=keep(phi0), mask=mask)


def simulate(ph: Phantom, protocol: Protocol, sigma: float, seed: int = 1,
             n_iso: int = 512) -> tuple[Tensor, Tensor]:
    """Noise-free and noisy complex images [H, W, n_scans, P, J] (zero outside the mask)."""
    m = ph.mask
    S_vox = mpme_signal(protocol, ph.M0[m], ph.T1[m], ph.T2[m], ph.R2p[m], ph.dw[m], ph.B1[m],
                        ph.phi0[m], n_iso=n_iso)
    H, W = m.shape
    S = torch.zeros(H, W, *S_vox.shape[1:], dtype=S_vox.dtype)
    S[m] = S_vox
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(S.shape, dtype=S.dtype, generator=g) * sigma * math.sqrt(2)
    return S, S + noise          # complex randn has variance 1 total → σ per component


def lowres_phase_encode(img: Tensor, fraction: float = 0.25) -> Tensor:
    """Keep the central ``fraction`` of ky (dim 0) lines, zero-fill, back to image space.

    Readout (dim 1) stays fully sampled, as in the published design. Applies to every
    trailing scan/pathway/echo channel.
    """
    H = img.shape[0]
    k = torch.fft.fftshift(torch.fft.fft(img, dim=0), dim=0)
    keep = max(1, int(round(H * fraction)))
    lo = (H - keep) // 2
    w = torch.zeros(H, dtype=img.real.dtype)
    w[lo:lo + keep] = 1
    k = k * w.reshape(H, *([1] * (img.dim() - 1)))
    return torch.fft.ifft(torch.fft.ifftshift(k, dim=0), dim=0)


def polyfit_2d(values: Tensor, mask: Tensor, degree: int = 4, n_reject: int = 2,
               k_mad: float = 3.0) -> Tensor:
    """Smooth a map by a 2D polynomial fitted within ``mask`` (outlier-rejecting).

    The published method fits the low-resolution flip-angle map to a polynomial; the degree
    is not stated there. Voxels with NaN/inf are ignored; ``n_reject`` rounds of rejection of
    residuals beyond ``k_mad`` × MAD make the fit robust to edge and failure voxels.
    """
    H, W = values.shape
    c = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) / (H / 2)
    d = (torch.arange(W, dtype=torch.float64) - (W - 1) / 2) / (W / 2)
    y, x = torch.meshgrid(c, d, indexing="ij")
    terms = [x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    A_all = torch.stack([t.flatten() for t in terms], 1)
    v = values.flatten().double()
    use = mask.flatten() & torch.isfinite(v)
    for _ in range(n_reject + 1):
        coef = torch.linalg.lstsq(A_all[use], v[use, None]).solution
        res = v - (A_all @ coef)[:, 0]
        mad = (res[use] - res[use].median()).abs().median()
        use = use & (res.abs() <= k_mad * 1.4826 * mad + 1e-12)
    return (A_all @ coef)[:, 0].reshape(H, W)
