"""CRLB and Fisher-information conditioning analysis for MPME (docs/crlb_analysis.md).

Prints the tables used in the write-up and saves figures to docs/figures/.

    python scripts/crlb_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpme.crlb import (PARAMS, conditioning, correlation, covariance, crlb,
                       fisher_information, fixed_parameter_bias)
from mpme.sequence import Protocol, Scan, paper_protocol
from mpme.signal import mpme_signal

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"

# Reference palette (dataviz skill, light mode) — fixed categorical order.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

LABELS = {"M0": "M0", "B1": "B1⁺", "T1": "T1", "T2": "T2", "T2star": "T2*",
          "dw": "Δω", "phi0": "φ0"}
TISSUES = {
    "WM": dict(M0=1.0, B1=1.0, T1=850.0, T2=66.0, T2star=50.0, dw=0.05, phi0=0.3),
    "GM": dict(M0=1.0, B1=1.0, T1=1350.0, T2=90.0, T2star=60.0, dw=0.05, phi0=0.3),
    "Paper ref.": dict(M0=1.0, B1=1.0, T1=1500.0, T2=70.0, T2star=60.0, dw=0.05, phi0=0.3),
}
REF = TISSUES["Paper ref."]


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def with_flip2(pr: Protocol, flip2: float) -> Protocol:
    s1, s2 = pr.scans
    return Protocol(pr.pathways, (s1, Scan(flip2, s2.TR, s2.echo_times)))


def subset(pr: Protocol, pathways) -> Protocol:
    idx = [pr.index(p) for p in pathways]
    return Protocol(tuple(pathways), tuple(
        Scan(s.flip_deg, s.TR, tuple(s.echo_times[i] for i in idx)) for s in pr.scans))


def fmt(d, keys=PARAMS):
    return " | ".join(f"{d[k]:.3g}" if k in d else "—" for k in keys)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pr = paper_protocol()
    pr30 = with_flip2(pr, 30.0)
    protocols = {"15°/330° (paper)": pr, "15°/30°": pr30}

    # ---- 1. Bounds per tissue and scenario -----------------------------------------
    print("\n## 1. CRLB per unit σ/M0 (relative sd for M0, B1, T1, T2, T2*; rad/ms for Δω)")
    print("| Protocol | Tissue | Scenario | " + " | ".join(LABELS[p] for p in PARAMS) + " |")
    print("|---" * (3 + len(PARAMS)) + "|")
    for pname, p in protocols.items():
        for tname, tis in TISSUES.items():
            F = fisher_information(p, tis)
            print(f"| {pname} | {tname} | all unknown | {fmt(crlb(F))} |")
            print(f"| {pname} | {tname} | B1 known | {fmt(crlb(F, known=('B1',)))} |")

    # ---- 2. Conditioning ------------------------------------------------------------
    print("\n## 2. Conditioning of the normalised information matrix (paper ref. tissue)")
    cond = {}
    for pname, p in protocols.items():
        F = fisher_information(p, REF)
        c = conditioning(F)
        cond[pname] = (F, c)
        print(f"\n### {pname}")
        print("eigenvalues:", np.array2string(c["eigenvalues"].numpy(), precision=6))
        print(f"condition number: {c['condition'].item():.3g}")
        print("weakest eigenvector:", {LABELS[k]: round(v, 3) for k, v in
                                       zip(PARAMS, c["eigenvectors"][:, 0].tolist())})
        print("variance inflation (F⁻¹)_ii·F_ii:", {LABELS[k]: round(v, 1) for k, v in
                                                     zip(PARAMS, c["vif"].tolist())})
        print("d log θ / d log B1 when B1 is fixed wrong:",
              {LABELS[k]: round(v, 3) for k, v in fixed_parameter_bias(F, PARAMS, "B1").items()})
        i = PARAMS.index("T1")
        print(f"T1 bound: all unknown {crlb(F)['T1']:.3g}; B1 known "
              f"{crlb(F, known=('B1',))['T1']:.3g}; B1 and M0 known "
              f"{crlb(F, known=('B1', 'M0'))['T1']:.3g}; only T1 unknown "
              f"{(1 / F[i, i].sqrt()).item():.3g}")
        Fs1 = fisher_information(p, REF, scan_sigma=(1.0, 1e6))
        print("  same, scan 1 data only:",
              {LABELS[k]: round(v, 3) for k, v in fixed_parameter_bias(Fs1, PARAMS, "B1").items()})

    # ---- 3. Scaling symmetry: signal change along (M0/k, B1·k, T1/k²) ----------------
    print("\n## 3. Relative signal change along the scaling direction (M0/k, B1·k, T1/k²)")
    print("| k − 1 | 15°/330° | 15°/30° | generic direction (T1 only, 15°/30°) |")
    print("|---|---|---|---|")
    sym_rows = []
    for eps in (0.01, 0.02, 0.05, 0.10):
        k = 1 + eps
        row = []
        for p in (pr, pr30):
            base = mpme_signal(p, 1.0, REF["T1"], REF["T2"], 1 / 60 - 1 / 70, 0.0, 1.0)
            moved = mpme_signal(p, 1 / k, REF["T1"] / k**2, REF["T2"], 1 / 60 - 1 / 70, 0.0, k)
            row.append(((moved - base).norm() / base.norm()).item())
        base = mpme_signal(pr30, 1.0, REF["T1"], REF["T2"], 1 / 60 - 1 / 70, 0.0, 1.0)
        moved = mpme_signal(pr30, 1.0, REF["T1"] / k**2, REF["T2"], 1 / 60 - 1 / 70, 0.0, 1.0)
        row.append(((moved - base).norm() / base.norm()).item())
        sym_rows.append(row)
        print(f"| {eps:.2f} | {row[0]:.2e} | {row[1]:.2e} | {row[2]:.2e} |")

    # ---- 4. Pathway subsets -----------------------------------------------------------
    print("\n## 4. Pathway subsets, 15°/330°, paper ref. tissue")
    print("| Pathways | B1 | T1 | T1 (B1 known) | T2 | condition |")
    print("|---|---|---|---|---|---|")
    for paths in ((1, 0, -1), (0, -1), (1, 0)):
        p = subset(pr, paths)
        F = fisher_information(p, REF)
        try:
            b = crlb(F)
            bk = crlb(F, known=("B1",))
            print(f"| {paths} | {b['B1']:.3g} | {b['T1']:.3g} | {bk['T1']:.3g} | {b['T2']:.3g} "
                  f"| {conditioning(F)['condition'].item():.3g} |")
        except torch.linalg.LinAlgError:
            print(f"| {paths} | singular | | | | |")

    # ---- 5. Sweeps --------------------------------------------------------------------
    flips = np.arange(20, 356, 5.0)
    sw = {"T1": [], "T1k": [], "B1": [], "cond": []}
    for f2 in flips:
        F = fisher_information(with_flip2(pr, float(f2)), REF)
        b, bk = crlb(F), crlb(F, known=("B1",))
        sw["T1"].append(b["T1"]); sw["T1k"].append(bk["T1"]); sw["B1"].append(b["B1"])
        sw["cond"].append(conditioning(F)["condition"].item())

    b1s = np.linspace(0.6, 1.5, 91)
    bsw = {"T1": [], "T1k": [], "B1": []}
    for b1 in b1s:
        F = fisher_information(pr, {**REF, "B1": float(b1)})
        b, bk = crlb(F), crlb(F, known=("B1",))
        bsw["T1"].append(b["T1"]); bsw["T1k"].append(bk["T1"]); bsw["B1"].append(b["B1"])

    t1s = np.geomspace(300, 4000, 40)
    tsw = {"T1": [], "T1k": [], "T130": []}
    for t1 in t1s:
        F = fisher_information(pr, {**REF, "T1": float(t1)})
        F30 = fisher_information(pr30, {**REF, "T1": float(t1)})
        tsw["T1"].append(crlb(F)["T1"]); tsw["T1k"].append(crlb(F, known=("B1",))["T1"])
        tsw["T130"].append(crlb(F30)["T1"])

    print("\n## 5. Selected sweep values (paper ref. tissue)")
    for f2 in (30, 90, 150, 180, 210, 270, 300, 330, 350):
        i = int(np.argmin(abs(flips - f2)))
        print(f"α2={flips[i]:.0f}°: T1 {sw['T1'][i]:.3g}, T1|B1 {sw['T1k'][i]:.3g}, "
              f"B1 {sw['B1'][i]:.3g}, cond {sw['cond'][i]:.3g}")
    for b1 in (0.7, 0.8, 0.9, 1.0, 1.05, 1.08, 1.1, 1.15, 1.3):
        i = int(np.argmin(abs(b1s - b1)))
        print(f"B1={b1s[i]:.2f} (α2={330*b1s[i]:.0f}°): T1 {bsw['T1'][i]:.3g}, "
              f"T1|B1 {bsw['T1k'][i]:.3g}, B1 {bsw['B1'][i]:.3g}")

    a1s = np.arange(5, 41, 2.5)
    a2s = np.arange(200, 356, 5.0)
    grid = np.full((len(a1s), len(a2s)), np.nan)
    for i, a1 in enumerate(a1s):
        for j, a2 in enumerate(a2s):
            s1, s2 = pr.scans
            p = Protocol(pr.pathways, (Scan(float(a1), s1.TR, s1.echo_times),
                                       Scan(float(a2), s2.TR, s2.echo_times)))
            grid[i, j] = crlb(fisher_information(p, REF))["T1"]
    i_min, j_min = np.unravel_index(np.nanargmin(grid), grid.shape)
    print(f"\n## 6. (α1, α2) map: best T1 bound {grid[i_min, j_min]:.3g} at "
          f"α1={a1s[i_min]:.1f}°, α2={a2s[j_min]:.0f}°; paper (15°, 330°): "
          f"{grid[np.argmin(abs(a1s - 15)), np.argmin(abs(a2s - 330))]:.3g}")
    for a1 in (5, 10, 15, 20, 25, 30, 40):
        r = int(np.argmin(abs(a1s - a1)))
        print(f"  α1={a1s[r]:.1f}°: best α2={a2s[np.nanargmin(grid[r])]:.0f}°, "
              f"T1 bound {np.nanmin(grid[r]):.3g}")

    # ---- Figures ------------------------------------------------------------------------
    # Fig 1: correlation matrices.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    for ax, (pname, (F, _)) in zip(axes, cond.items()):
        R = correlation(covariance(F)).numpy()
        Rd = R.copy()
        np.fill_diagonal(Rd, 0.0)                                           # neutral diagonal
        ax.imshow(Rd, cmap=cmap, vmin=-1, vmax=1)
        n = len(PARAMS)
        ax.set_xticks(range(n), [LABELS[p] for p in PARAMS])
        ax.set_yticks(range(n), [LABELS[p] for p in PARAMS])
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{R[i, j]:+.2f}" if i != j else "1", ha="center", va="center",
                        fontsize=7.5, color=INK if abs(R[i, j]) < 0.75 else "white")
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"CRLB correlation, {pname}", color=INK, fontsize=11, loc="left")
    save(fig, "crlb_correlation.png")

    # Fig 2: eigen-spectrum and weakest eigenvector.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for (pname, (_, c)), col, mk in zip(cond.items(), (BLUE, ORANGE), ("o", "s")):
        ev = c["eigenvalues"].numpy()
        ax.semilogy(range(1, len(ev) + 1), ev, mk, color=col, ms=8, label=pname,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    style(ax, "Eigenvalues of normalised information", "eigenvalue index (ascending)",
          "eigenvalue")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    x = np.arange(len(PARAMS))
    w = 0.38
    for n, ((pname, (_, c)), col) in enumerate(zip(cond.items(), (BLUE, ORANGE))):
        v = c["eigenvectors"][:, 0].numpy()
        v = v * np.sign(v[PARAMS.index("T1")])
        ax.bar(x + (n - 0.5) * w, v, w * 0.92, color=col, label=pname)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [LABELS[p] for p in PARAMS])
    style(ax, "Weakest direction (normalised coordinates)", None, "component")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, "crlb_eigen.png")

    # Fig 3: α2 sweep (two charts, one axis each).
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.semilogy(flips, sw["T1"], color=BLUE, lw=2, label="B1⁺ unknown")
    ax.semilogy(flips, sw["T1k"], color=ORANGE, lw=2, label="B1⁺ known")
    ax.semilogy(flips, sw["B1"], color=AQUA, lw=2, label="B1⁺ (its own bound)")
    ax.axvline(330, color=MUTED, lw=1, ls=":")
    ax.text(332, ax.get_ylim()[1] * 0.5, "paper", color=MUTED, fontsize=8)
    style(ax, "CRLB vs α2 (α1 = 15°)", "α2 (degrees)", "relative sd per unit σ/M0")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    ax.semilogy(flips, sw["cond"], color=BLUE, lw=2)
    ax.axvline(330, color=MUTED, lw=1, ls=":")
    style(ax, "Condition number vs α2", "α2 (degrees)", "condition number")
    save(fig, "crlb_flip_sweep.png")

    # Fig 4: B1 sweep and T1 sweep.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.semilogy(b1s, bsw["T1"], color=BLUE, lw=2, label="T1, B1⁺ unknown")
    ax.semilogy(b1s, bsw["T1k"], color=ORANGE, lw=2, label="T1, B1⁺ known")
    ax.semilogy(b1s, bsw["B1"], color=AQUA, lw=2, label="B1⁺")
    ax.axvline(360 / 330, color=MUTED, lw=1, ls=":")
    ax.text(360 / 330 + 0.01, 0.15, "α2 = 360°", color=MUTED, fontsize=8)
    style(ax, "CRLB vs true B1⁺ (15°/330°)", "true B1⁺ scale", "relative sd per unit σ/M0")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    ax.loglog(t1s, tsw["T1"], color=BLUE, lw=2, label="15°/330°, B1⁺ unknown")
    ax.loglog(t1s, tsw["T1k"], color=ORANGE, lw=2, label="B1⁺ known (either protocol)")
    ax.loglog(t1s, tsw["T130"], color=YELLOW, lw=2, label="15°/30°, B1⁺ unknown")
    ax.set_xticks([300, 500, 1000, 2000, 4000], ["300", "500", "1000", "2000", "4000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, "T1 CRLB vs T1", "T1 (ms)", "relative sd per unit σ/M0")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, "crlb_b1_t1_sweeps.png")
    # Fig 5: (α1, α2) map of the T1 bound, sequential single-hue ramp (light → dark = worse).
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    im = ax.pcolormesh(a2s, a1s, grid, cmap=ramp, shading="nearest",
                       norm=matplotlib.colors.LogNorm(vmin=np.nanmin(grid), vmax=200))
    ax.plot([330], [15], "o", ms=9, color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate("paper", (330, 15), (310, 22), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.plot([a2s[j_min]], [a1s[i_min]], "D", ms=8, color=YELLOW, markeredgecolor=SURFACE,
            markeredgewidth=2)
    ax.annotate("minimum", (a2s[j_min], a1s[i_min]), (a2s[j_min] - 45, a1s[i_min] - 4),
                color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    style(ax, "T1 CRLB per unit σ/M0, B1⁺ unknown (TR 25 ms, paper ref. tissue)",
          "α2 (degrees)", "α1 (degrees)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    cb.outline.set_visible(False)
    save(fig, "crlb_flip_map.png")
    print(f"\nFigures written to {FIG}")


if __name__ == "__main__":
    main()
