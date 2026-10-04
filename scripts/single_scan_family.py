"""Single-scan solution family: the worked examples of Sections 2 and 5 of the technical note.

For a trial transmit scale B1', the family (eq:family) gives (T1', M0') that reproduce one scan
exactly. In Ernst-angle form (Proposition ernst): tan(alpha_E'/2) = tan(B1' alpha/2) / xi and
M0' tan(alpha_E'/2) = const, with xi = tan(alpha/2) / tan(alpha_E/2) fixed by the data.

Tables written to docs/technote/tables/:
  family_one.tex  one WM voxel: flip, Ernst angle, T1', M0', scan-1 and scan-2 misfit, and the
                  ratio xi_2/xi_1 that each member predicts for a 330 or a 30 degree scan 2
  family_two.tex  WM and GM voxels sharing B1: T1', M0', their ratios, misfits
Figure docs/technote/figures/ernst_logtan.png: the amplitude M0|v| = M0 zeta sech(s - ln zeta) on the
axis s = ln|tan(alpha/2)| for three members of the family, with the samples of scan 1 and scan 2.

    python scripts/single_scan_family.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpme.sequence import Protocol, paper_protocol
from mpme.signal import mpme_signal
from technote_figures import AQUA, BLUE, INK2, MUTED, ORANGE, save, style

TAB = Path(__file__).resolve().parents[1] / "docs" / "technote" / "tables"
TR = 25.0
WM = dict(M0=0.69, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3)
GM = dict(M0=0.82, T1=1350.0, T2=90.0, T2star=60.0, dw=-0.03, phi0=1.1)


def family(t, alpha, B1p):
    """Member of the single-scan family of tissue t (true B1 = 1) at trial transmit scale B1p."""
    zeta = math.sqrt(math.tanh(TR / (2 * t["T1"])))                # tan(alpha_E / 2)
    xi = math.tan(alpha / 2) / zeta
    zeta_p = math.tan(B1p * alpha / 2) / xi
    return TR / (2 * math.atanh(zeta_p**2)), t["M0"] * zeta / zeta_p


def xi_ratio(B1, a1_deg, a2_deg):
    """xi_2/xi_1 = tan(B1 a2/2)/tan(B1 a1/2): what two scans measure about B1 (eq:secondscan)."""
    return math.tan(math.radians(B1 * a2_deg) / 2) / math.tan(math.radians(B1 * a1_deg) / 2)


def misfit(t, scans, M0, T1, B1):
    """Largest |S - S_true| of each scan relative to the largest |S_true| of that scan."""
    kw = dict(T2=t["T2"], R2p=1 / t["T2star"] - 1 / t["T2"], dw=t["dw"], phi0=t["phi0"], n_iso=1024)
    out = []
    for p in scans:
        S0 = mpme_signal(p, M0=t["M0"], T1=t["T1"], B1=1.0, **kw)
        S = mpme_signal(p, M0=M0, T1=T1, B1=B1, **kw)
        out.append(float((S - S0).abs().max() / S0.abs().max()))
    return out


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def figure(alpha):
    """M0|v| against s = ln|tan(a/2)| for three members of the WM family, two scan-2 choices."""
    s = np.linspace(-4.5, 1.0, 600)
    members = ((0.8, AQUA), (1.0, BLUE), (1.25, ORANGE))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, a2 in zip(axes, (330, 30)):
        for B1p, col in members:
            T1p, M0p = family(WM, alpha, B1p)
            zeta = math.sqrt(math.tanh(TR / (2 * T1p)))
            ax.plot(s, M0p * zeta / np.cosh(s - math.log(zeta)), color=col, lw=1.5,
                    label=f"B1+ {B1p:g}:  T1 {T1p:.0f} ms,  M0 {M0p:.3f}")
            ax.axvline(math.log(zeta), color=col, lw=0.8, ls=":")
            for a, mk in ((15, "o"), (a2, "s")):
                si = math.log(abs(math.tan(math.radians(B1p * a) / 2)))
                ax.plot(si, M0p * zeta / math.cosh(si - math.log(zeta)), mk, ms=8, color=col,
                        markeredgecolor="white", markeredgewidth=1.2)
        style(ax, title=f"({'a' if a2 == 330 else 'b'}) scan 1 at 15°, scan 2 at {a2}°",
              xlabel="s = ln |tan(α/2)|   (dotted: Ernst angle, ln tan(αE/2))",
              ylabel="M0 |v|" if a2 == 330 else None)
    axes[0].plot([], [], "o", color=MUTED, ms=7, label="scan 1 sample (15°)")
    axes[0].plot([], [], "s", color=MUTED, ms=7, label="scan 2 sample")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=INK2,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, "ernst_logtan.png")


def main():
    pr = paper_protocol()
    scans = [Protocol(pr.pathways, (s,)) for s in pr.scans]
    alpha = math.radians(pr.scans[0].flip_deg)
    one = []
    for B1p in (0.8, 0.9, 1.0, 1.1, 1.2, 1.5):
        T1p, M0p = family({**WM, "M0": 1.0}, alpha, B1p)
        aE = math.degrees(math.acos(math.exp(-TR / T1p)))
        m1, m2 = misfit({**WM, "M0": 1.0}, scans, M0p, T1p, B1p)
        one.append(f"{B1p:.2f} & {15 * B1p:.1f} & {aE:.2f} & {T1p:.0f} & {M0p:.3f} & {sci(m1)} & "
                   f"{m2:.2f} & ${xi_ratio(B1p, 15, 330):+.2f}$ & ${xi_ratio(B1p, 15, 30):.3f}$\\\\")
        xi = math.tan(B1p * alpha / 2) / math.tan(math.radians(aE) / 2)
        print(f"B1' = {B1p}: flip / Ernst angle = {15 * B1p / aE:.4f}, xi = {xi:.4f}")
    two = []
    for B1p in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5):
        (Tw, Mw), (Tg, Mg) = family(WM, alpha, B1p), family(GM, alpha, B1p)
        mw, mg = misfit(WM, scans, Mw, Tw, B1p), misfit(GM, scans, Mg, Tg, B1p)
        two.append(f"{B1p:.1f} & {Tw:.0f} & {Tg:.0f} & {Tg / Tw:.4f} & {Mw:.3f} & {Mg:.3f} & "
                   f"{Mg / Mw:.6f} & {sci(max(mw[0], mg[0]))} & {max(mw[1], mg[1]):.2f}\\\\")
    figure(alpha)
    (TAB / "family_one.tex").write_text("\n".join(one) + "\n")
    (TAB / "family_two.tex").write_text("\n".join(two) + "\n")
    print("\n".join(one + [""] + two))


if __name__ == "__main__":
    main()
