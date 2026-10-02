"""MPME protocol definition.

Default values are placeholders chosen to be plausible, not the values used by
Cheng et al. 2019; replace them once the real protocol is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Scan:
    """One MPME acquisition.

    Attributes:
        flip_deg: nominal flip angle (degrees).
        TR: repetition time (ms).
        echo_times: echo times (ms, after the RF pulse), one tuple per pathway in
            ``Protocol.pathways`` order. All pathways must have the same number of echoes.
    """

    flip_deg: float
    TR: float
    echo_times: tuple[tuple[float, ...], ...]

    @property
    def alpha(self) -> float:
        return math.radians(self.flip_deg)


@dataclass(frozen=True)
class Protocol:
    """Pathway orders p (state F_p after the RF pulse) and the scans that sample them."""

    pathways: tuple[int, ...]
    scans: tuple[Scan, ...]

    def __post_init__(self):
        n_echo = {len(t) for s in self.scans for t in s.echo_times}
        if len(n_echo) != 1:
            raise ValueError("every pathway in every scan needs the same number of echoes")
        for s in self.scans:
            if len(s.echo_times) != len(self.pathways):
                raise ValueError("need one echo-time tuple per pathway")
            if not all(0 < t < s.TR for ts in s.echo_times for t in ts):
                raise ValueError("echo times must lie in (0, TR)")

    @property
    def n_echoes(self) -> int:
        return len(self.scans[0].echo_times[0])

    def echo_times(self, i: int, dtype=torch.float64) -> Tensor:
        """Echo times of scan ``i`` as a [P, J] tensor (ms)."""
        return torch.tensor(self.scans[i].echo_times, dtype=dtype)

    def index(self, p: int) -> int:
        return self.pathways.index(p)


def default_protocol() -> Protocol:
    """Three-pathway [1, 0, −1], three-echo protocol with α₂ = 10·α₁ (placeholder values)."""
    echoes = ((3.0, 5.0, 7.0),     # p = +1: Gx moment −1
              (9.0, 11.0, 13.0),   # p =  0: moment 0 (FID)
              (15.0, 17.0, 19.0))  # p = −1: moment +1 (echo)
    return Protocol(
        pathways=(1, 0, -1),
        scans=(Scan(flip_deg=3.0, TR=22.0, echo_times=echoes),
               Scan(flip_deg=30.0, TR=22.0, echo_times=echoes)),
    )
