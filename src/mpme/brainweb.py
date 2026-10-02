"""BrainWeb digital brain phantom with partial volume.

Uses the fuzzy (tissue-fraction) volumes of the BrainWeb normal-brain anatomical model
(Collins et al. 1998; Kwan et al. 1999; Cocosco et al. 1997): 181×217×181 voxels at 1 mm,
values = fraction of each tissue in the voxel. Files expected in ``data/brainweb/`` as
``phantom_1.0mm_normal_{csf,gry,wht,crisp}.rawb.gz`` (raw unsigned byte, gzip).

Only brain tissue is simulated (CSF, GM, WM). Voxels whose discrete label is not CSF, GM or
WM (fat, skin, skull, ...) are excluded; fat would need a chemical-shift model. Within a
voxel the signal is the proton-density-weighted sum of the compartments,

    S = C · Σ_c f_c · PD_c · s(θ_c),

with B1⁺, Δω, φ0 and the receive sensitivity C shared by the compartments (they are
properties of the location). Tissue values and fields are those of ``phantom.py``.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .phantom import TISSUES, _smooth_noise
from .sequence import Protocol
from .signal import mpme_signal

DATA = Path(__file__).resolve().parents[2] / "data" / "brainweb"
SHAPE = (181, 217, 181)                  # z, y, x
COMPARTMENTS = (("csf", "CSF"), ("gry", "GM"), ("wht", "WM"))


def load_volume(name: str, root: Path = DATA) -> np.ndarray:
    raw = gzip.open(root / f"phantom_1.0mm_normal_{name}.rawb.gz").read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)


@dataclass
class PVPhantom:
    fractions: Tensor        # [3, H, W] CSF, GM, WM volume fractions
    mask: Tensor             # [H, W] bool, brain voxels
    T1: Tensor               # [3, H, W] per-compartment parameters
    T2: Tensor
    T2star: Tensor
    PD: Tensor
    C: Tensor                # [H, W] receive sensitivity
    B1: Tensor
    dw: Tensor
    phi0: Tensor

    def weights(self) -> Tensor:
        """Signal weights w_c = f_c PD_c / Σ f PD (proton-density-weighted fractions)."""
        fw = self.fractions * self.PD
        return fw / fw.sum(0).clamp_min(1e-12)

    def effective(self, name: str) -> Tensor:
        """PD-weighted geometric mean of a parameter over compartments (the 'truth' used
        for mixed voxels): exp(Σ w_c log θ_c)."""
        return torch.exp((self.weights() * getattr(self, name).log()).sum(0))

    def M0_total(self) -> Tensor:
        return self.C * (self.fractions * self.PD).sum(0)


def make_brainweb_phantom(z_index: int = 95, n: int = 256, seed: int = 0,
                          texture: float = 0.03, root: Path = DATA) -> PVPhantom:
    """Axial slice ``z_index`` of BrainWeb, zero-padded to n×n (1 mm voxels)."""
    crisp = load_volume("crisp", root)[z_index]
    frac = np.stack([load_volume(a, root)[z_index] / 255.0 for a, _ in COMPARTMENTS])
    H0, W0 = crisp.shape
    pad_y, pad_x = (n - H0) // 2, (n - W0) // 2
    F = np.zeros((3, n, n))
    F[:, pad_y:pad_y + H0, pad_x:pad_x + W0] = frac
    lab = np.zeros((n, n), dtype=int)
    lab[pad_y:pad_y + H0, pad_x:pad_x + W0] = crisp
    F = torch.from_numpy(F)
    mask = torch.from_numpy(np.isin(lab, (1, 2, 3))) & (F.sum(0) > 0.5)

    g = torch.Generator().manual_seed(seed)
    T1 = torch.zeros(3, n, n, dtype=torch.float64)
    T2, T2s, PD = T1.clone(), T1.clone(), T1.clone()
    names = {v[0]: v for v in TISSUES.values()}
    for c, (_, tname) in enumerate(COMPARTMENTS):
        _, t1, t2, t2s, pd = names[tname]
        tex = [_smooth_noise((n, n), 6.0, g) for _ in range(4)]
        T1[c] = t1 * torch.exp(texture * tex[0])
        T2[c] = t2 * torch.exp(texture * tex[1])
        T2s[c] = torch.minimum(t2s * torch.exp(texture * tex[2]), 0.98 * T2[c])
        PD[c] = pd * torch.exp(texture * tex[3])

    # Fields as in phantom.py, in coordinates normalised to the head half-extent (~95 mm).
    c = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) / 95.0
    y, x = torch.meshgrid(c, c, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    B1 = 1.15 - 0.33 * r**2 + 0.04 * x
    f_hz = 20 * (x**2 - y**2) + 60 * torch.exp(-(x**2 + (y + 0.80) ** 2) / (2 * 0.15**2))
    dw = 2 * math.pi * f_hz / 1000
    C = 1.0 + 0.15 * y + 0.10 * x**2
    phi0 = 0.6 * x + 0.3 * y**2
    return PVPhantom(fractions=F, mask=mask, T1=T1, T2=T2, T2star=T2s, PD=PD,
                     C=C, B1=B1, dw=dw, phi0=phi0)


def simulate_pv(ph: PVPhantom, protocol: Protocol, sigma: float, seed: int = 1,
                n_iso: int = 512) -> tuple[Tensor, Tensor]:
    """Noise-free and noisy complex images [H, W, n_scans, P, J] with partial volume."""
    m = ph.mask
    S_vox = 0
    for c in range(3):
        f = ph.fractions[c][m]
        if not (f > 0).any():
            continue
        Sc = mpme_signal(protocol, ph.PD[c][m], ph.T1[c][m], ph.T2[c][m],
                         1 / ph.T2star[c][m] - 1 / ph.T2[c][m], ph.dw[m], ph.B1[m],
                         ph.phi0[m], n_iso=n_iso)
        S_vox = S_vox + f[:, None, None, None] * Sc
    S_vox = ph.C[m][:, None, None, None] * S_vox
    H, W = m.shape
    S = torch.zeros(H, W, *S_vox.shape[1:], dtype=S_vox.dtype)
    S[m] = S_vox
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(S.shape, dtype=S.dtype, generator=g) * sigma * math.sqrt(2)
    return S, S + noise
