"""MPME protocol definition.

``paper_protocol`` reproduces the in-vivo protocol of Cheng et al. 2019 (Table 1 and the
Gx waveform of Fig. 1). The time from the RF pulse to the first pathway echo is not given
in the paper and is an assumption here.
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


def mpme_echo_times(
    scheme: tuple[int, int, int] = (1, 0, -1),
    n_te: int = 3,
    spacing: float = 2.0,
    t_first: float = 4.5,
) -> tuple[tuple[float, ...], ...]:
    """Echo times (ms) per pathway for the Fig. 1 Gx waveform, in ``scheme`` order.

    Each of the ``n_te`` readout windows (alternating polarity) sweeps the gradient moment
    across all three pathways, ``spacing`` ms apart (= 1 / BW per pixel). Odd windows visit
    the pathways in ``scheme`` order, even windows in reverse; consecutive windows share an
    extra ``spacing`` for the turn-around. ``t_first`` is the first echo after the RF pulse.
    Gradient ramps are ignored.
    """
    times: dict[int, list[float]] = {p: [] for p in scheme}
    for w in range(n_te):
        order = scheme if w % 2 == 0 else scheme[::-1]
        for n, p in enumerate(order):
            times[p].append(t_first + spacing * (3 * w + n))
    return tuple(tuple(times[p]) for p in scheme)


def paper_protocol(scheme: tuple[int, int, int] = (1, 0, -1), spacing: float = 2.0,
                   t_first: float = 4.5) -> Protocol:
    """Cheng et al. 2019 in-vivo protocol: TR 25 ms, α 15°/330°, 3 readout windows.

    Pathway spacing 2.0 ms in vivo (1.86 ms in the phantom). The paper's scan 2 covers only
    the central 25% × 25% of ky–kz; that is handled by the imaging layer, not here.
    """
    echoes = mpme_echo_times(scheme, 3, spacing, t_first)
    return Protocol(
        pathways=tuple(scheme),
        scans=(Scan(flip_deg=15.0, TR=25.0, echo_times=echoes),
               Scan(flip_deg=330.0, TR=25.0, echo_times=echoes)),
    )


def default_protocol() -> Protocol:
    """The paper's in-vivo protocol with the [1, 0, −1] scheme."""
    return paper_protocol()
