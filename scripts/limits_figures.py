"""Figures, table and numbers for the B1/T1 limits companion note (docs/latex/mpme_b1_t1_limits).

Reference tissue T1/T2/T2* = 1500/70/60 ms (Cheng et al.'s optimisation tissue), M0 = 1, TR 25 ms,
the paper's readout; bounds per unit σ/M0, all seven parameters free unless stated.

  docs/figures/limits_lever.png     (a) rate c(x) = x/sin x; (b) B1 bound against 1/|L| over protocols
  docs/figures/limits_crossing.png  15/330 against 18/270 over B1: (a) ξ2/ξ1 and its mirror (aliases),
                                    (b) B1 and T1 bounds, (c) T1 error per unit error of a supplied B1
  docs/latex/limits_budget.tex      T1-bound budget (parameters freed one group at a time)
  results/limits_numbers.json       the numbers quoted in the note

    PYTHONPATH=src python3 scripts/limits_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpme.crlb import PARAMS, fixed_parameter_bias
from mpme.design import alias_roots, bounds, ernst_offset, f_ratio, single_scan_fisher
from mpme.sequence import paper_protocol
from technote_figures import AQUA, AXIS, BLUE, GRID, INK2, MUTED, ORANGE, SURFACE, style

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
RES, LATEX, FIG = ROOT / "results", ROOT / "docs" / "latex", ROOT / "docs" / "figures"
PR = paper_protocol()
TR, ECHOES, PATHS = PR.scans[0].TR, PR.scans[0].echo_times, PR.pathways
REF = dict(M0=1.0, T1=1500.0, T2=70.0, T2star=60.0)
H = (TR / REF["T1"]) / (2 * math.sinh(TR / REF["T1"]))       # h = τ/(2 sinh τ)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fisher(angles, tissue=REF):
    """Per-scan information at actual angles (deg) → [n_scans, 7, 7]."""
    return single_scan_fisher(np.asarray(angles, float), tissue, TR, ECHOES, PATHS)


def lever(a1, a2, b1=1.0):
    _, c = ernst_offset(np.array([b1 * a1, b1 * a2]), REF["T1"], TR)
    return float(c[1] - c[0])


def fixed_b1_factor(F):
    """d ln T1 / d ln B1 when B1 is held at a wrong value and the rest is fitted."""
    return fixed_parameter_bias(F, PARAMS, "B1")["T1"]


def collapse():
    """B1 bound and |L| for scan 1 at 15° with many scan-2 angles, and small-angle pairs."""
    rows = []
    for a1, a2 in [(15, a) for a in (30, 45, 60, 90, 120, 150, 165, 195, 210, 240, 270, 300, 315,
                                     330, 345)] + [(3.75, 7.5), (5, 10), (10, 20), (30, 60)]:
        sd = float(bounds(fisher([a1, a2]).sum(0))[1])
        L = lever(a1, a2)
        rows.append(dict(a1=a1, a2=a2, L=L, sd_lnB1=sd, product=abs(L) * sd))
    return rows


def fig_lever(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    x = np.linspace(0.5, 539.5, 3000)
    c = np.radians(x) / np.sin(np.radians(x))
    ax.plot(x, np.where(np.abs(c) > 30, np.nan, c), color=BLUE, lw=1.5)
    for v in (180, 360):
        ax.axvline(v, color=AXIS, lw=0.8, ls=":")
    ax.axhline(0, color=AXIS, lw=0.8)
    for a, off in ((15, (-12, 8)), (30, (4, 8)), (270, (6, 4)), (330, (6, -4))):
        ca = math.radians(a) / math.sin(math.radians(a))
        ax.plot(a, ca, "o", ms=7, color=ORANGE, markeredgecolor="white")
        ax.annotate(f"{a}°", (a, ca), xytext=off, textcoords="offset points", fontsize=8, color=INK2)
    ax.set_ylim(-30, 30)
    style(ax, "(a) how fast a scan's sample moves with B1", "actual flip angle x (°)", "rate c(x) = x / sin x")
    ax = axes[1]
    big = [r for r in rows if r["a1"] == 15]
    small = [r for r in rows if r["a1"] != 15]
    inv = np.array([1 / abs(r["L"]) for r in rows])
    grid = np.logspace(np.log10(inv.min() / 2), np.log10(inv.max() * 2), 50)
    lo, hi = min(r["product"] for r in rows), max(r["product"] for r in rows)
    ax.fill_between(grid, lo * grid, hi * grid, color=GRID, alpha=0.7, lw=0,
                    label=f"|L| × bound between {lo:.0f} and {hi:.0f}")
    ax.plot([1 / abs(r["L"]) for r in big], [r["sd_lnB1"] for r in big], "o", ms=7, color=BLUE,
            markeredgecolor="white", label="15° / α2 (α2 = 30°…345°)")
    ax.plot([1 / abs(r["L"]) for r in small], [r["sd_lnB1"] for r in small], "s", ms=7, color=ORANGE,
            markeredgecolor="white", label="two small angles")
    for r in rows:
        if (r["a1"], r["a2"]) in ((15, 330), (15, 30), (15, 270), (3.75, 7.5)):
            ax.annotate(f"{r['a1']:g}/{r['a2']:g}", (1 / abs(r["L"]), r["sd_lnB1"]), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "(b) the B1 bound follows the lever", "1 / |L|", "CRLB of ln B1 (σ/M0)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    save(fig, "limits_lever.png")


def crossing():
    """Over B1 for 15/330 and 18/270: ξ2/ξ1, B1 and T1 bounds, fixed-B1 factors."""
    out = {}
    for a1, a2 in ((15, 330), (18, 270)):
        cross = [180 * m / a for a in (a1, a2) for m in range(1, 4) if 0.7 <= 180 * m / a <= 1.3]
        b = np.unique(np.concatenate([np.linspace(0.7, 1.3, 1201), cross]))
        F = fisher(np.stack([b * a1, b * a2], 1).reshape(-1)).reshape(len(b), 2, 7, 7)
        Fd = F.sum(1)
        bd = bounds(Fd).numpy()
        both = np.array([fixed_b1_factor(f) for f in Fd])
        one = np.array([fixed_b1_factor(f) for f in F[:, 0]])
        out[f"{a1}/{a2}"] = dict(b=b, sdB1=bd[:, 1], sdT1=bd[:, 2], both=both, one=one)
    return out


def fig_crossing(cr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    ax = axes[0]
    b = np.linspace(0.5, 2.2, 4000)
    f = f_ratio(b, 15, 330)
    f[np.abs(f) > 25] = np.nan
    ax.plot(b, f, color=BLUE, lw=1.5, label="ξ2/ξ1 = f(B1), 15/330")
    ax.plot(b, -f, color=BLUE, lw=1.0, ls="--", label="mirror −f(B1)")
    f1 = float(f_ratio(1.0, 15, 330))
    ax.axhline(f1, color=MUTED, lw=0.8, ls=":")
    ax.axvspan(0.7, 1.3, color=GRID, alpha=0.5, lw=0)
    ax.axvline(540 / 330, color=AXIS, lw=0.8)
    roots_c = alias_roots(1.0, 15, 330, (0.5, 2.2))
    roots_m = [r for r in alias_roots(1.0, 15, 330, (0.5, 2.2), magnitude=True)
               if all(abs(r - q) > 1e-6 for q in roots_c)]         # magnitude-only roots
    ax.plot([1.0], [f1], "o", ms=8, color=INK2, markeredgecolor="white", label="truth (B1 = 1)")
    ax.plot(roots_c, [f1] * len(roots_c), "D", ms=8, color=ORANGE, markeredgecolor="white",
            label="complex alias")
    ax.plot(roots_m, [f1] * len(roots_m), "s", ms=8, color=AQUA, markeredgecolor="white",
            label="magnitude alias")
    ax.set_ylim(-25, 25)
    style(ax, "(a) aliases are other roots of one B1 equation", "B1", "ξ2 / ξ1")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3)
    for (key, d), col, ls in zip(cr.items(), (BLUE, ORANGE), ("-", "--")):
        axes[1].plot(d["b"], d["sdB1"], color=col, lw=1.4, ls=ls, label=f"ln B1, {key}")
        axes[1].plot(d["b"], d["sdT1"], color=col, lw=1.4, ls=ls, alpha=0.55, label=f"ln T1, {key}")
        axes[2].plot(d["b"], d["both"], color=col, lw=1.4, ls=ls, label=f"both scans, {key}")
    axes[2].plot(cr["15/330"]["b"], cr["15/330"]["one"], color=MUTED, lw=1.2, ls=":", label="scan 1 only")
    axes[1].set_yscale("log")
    style(axes[1], "(b) bounds across the B1 range", "B1", "CRLB (σ/M0)")
    axes[1].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower left", ncol=2)
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].axhline(0, color=AXIS, lw=0.8)
    style(axes[2], "(c) T1 error per unit error of a supplied B1", "B1", "d ln T1 / d ln B1")
    axes[2].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower left")
    save(fig, "limits_crossing.png")


def budget():
    """T1 bound with parameters freed one group at a time (15/330 and 15/30, B1 = 1)."""
    groups = [("$\\Tone$ only", ("M0", "B1", "T2", "T2star", "dw", "phi0")),
              ("$+\\ \\Ttwo,\\ \\Ttwos,\\ \\dw,\\ \\phi_0$", ("M0", "B1")),
              ("$+\\ \\Mzero$", ("B1",)), ("$+\\ \\Bone$ (all free)", ())]
    rows, nums = [], {}
    for label, known in groups:
        v = [float(bounds(fisher([15, a2]).sum(0), known=known)[2]) for a2 in (330, 30)]
        nums[label] = v
        rows.append(f"{label} & {v[0]:.1f} & {v[1]:.1f}\\\\")
    s1 = float(bounds(fisher([15]).sum(0), known=("B1",))[2])
    rows.append(f"scan 1 only, $\\Bone$ known & \\multicolumn{{2}}{{c}}{{{s1:.1f}}}\\\\")
    nums["scan1_known"] = s1
    return rows, nums


def closed_form_check():
    """−c̄_Q/h (Q-weighted mean rate) against the direct fixed-B1 factor, 15/330 at a few B1."""
    out = []
    for b1 in (0.8, 1.0, 1.2):
        x = np.array([15 * b1, 330 * b1])
        _, c = ernst_offset(x, REF["T1"], TR)
        F = fisher(x).sum(0).numpy()
        A = np.eye(7)
        A[0, :3], A[1, :3], A[2, :3] = (0, c[0], H), (0, c[1], H), (1, 0, -H)
        Ai = np.linalg.inv(A)
        G = Ai.T @ F @ Ai
        Q = G[:2, :2] - G[:2, 2:] @ np.linalg.solve(G[2:, 2:], G[2:, :2])
        cbar = float(np.ones(2) @ Q @ c / (np.ones(2) @ Q @ np.ones(2)))
        out.append(dict(B1=b1, closed=-cbar / H, direct=fixed_b1_factor(torch.as_tensor(F)), cbar=cbar))
    return out


def main():
    rows = collapse()
    fig_lever(rows)
    cr = crossing()
    fig_crossing(cr)
    brows, bnums = budget()
    (LATEX / "limits_budget.tex").write_text("\n".join(brows) + "\n")
    d = cr["15/330"]
    pick = lambda q: {f"{b:.2f}": float(np.interp(b, d["b"], d[q])) for b in (0.7, 0.8, 0.9, 1.0, 1.2, 1.3)}
    k = int(np.argmin(np.abs(d["b"] - 360 / 330)))
    nums = {
        "collapse": rows,
        "product_range": [min(r["product"] for r in rows), max(r["product"] for r in rows)],
        "sd_range": [min(r["sd_lnB1"] for r in rows), max(r["sd_lnB1"] for r in rows)],
        "lever": {f"{a1:g}/{a2:g}": lever(a1, a2) for a1, a2 in ((15, 330), (15, 30), (15, 270), (18, 270))},
        "aliases_15_330": {"complex": alias_roots(1.0, 15, 330, (0.5, 2.2)),
                           "magnitude_incl_complex": alias_roots(1.0, 15, 330, (0.5, 2.2), magnitude=True),
                           "f1": float(f_ratio(1.0, 15, 330))},
        "fixed_b1_both_15_330": pick("both"), "fixed_b1_one_15_330": pick("one"),
        "fixed_b1_both_18_270": {f"{b:.2f}": float(np.interp(b, cr["18/270"]["b"], cr["18/270"]["both"]))
                                 for b in (0.7, 0.8, 0.9, 1.0, 1.2, 1.3)},
        "fixed_b1_near_crossing": {f"{b:.3f}": float(np.interp(b, d["b"], d["both"]))
                                   for b in (1.08, 1.085, 1.09, 1.095, 1.10)},
        "sd_at_crossing_15_330": {"B1": float(d["sdB1"][k]), "T1": float(d["sdT1"][k])},
        "sd_at_1_15_330": {"B1": float(np.interp(1.0, d["b"], d["sdB1"])),
                           "T1": float(np.interp(1.0, d["b"], d["sdT1"]))},
        "sd_range_18_270": {"B1": [float(cr["18/270"]["sdB1"].min()), float(cr["18/270"]["sdB1"].max())],
                            "T1": [float(cr["18/270"]["sdT1"].min()), float(cr["18/270"]["sdT1"].max())]},
        "budget": bnums, "closed_form_fixed_b1": closed_form_check(),
    }
    RES.mkdir(exist_ok=True)
    (RES / "limits_numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps({q: v for q, v in nums.items() if q != "collapse"}, indent=1))
    for r in rows:
        print(f"{r['a1']:g}/{r['a2']:g}: L {r['L']:+.4f}  SD lnB1 {r['sd_lnB1']:.3f}  |L|*SD {r['product']:.1f}")


if __name__ == "__main__":
    main()
