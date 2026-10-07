"""Flip-angle design note: shortlist, figures and tables from results/flip_design.npz.

Criterion (default): worst case over B1 in [0.7, 1.3] and the five tissues of the CRLB of
log T1, all seven parameters free, among designs without a complex alias in the fitting range
B1 ∈ [0.6, 1.4]. Shortlist:

  published    15/330 (Cheng et al. 2019)
  joint-worst  minimum worst-case T1 bound
  joint-mean   minimum mean T1 bound (mean over B1 and tissues)
  no-crossing  minimum worst-case T1 bound with no pulse crossing a multiple of 180° for any
               B1 in [0.7, 1.3] (α1 < 138°, 258° ≤ α2 ≤ 276°)
  margin       the same with 2.5% headroom at both ends of the B1 range (264° ≤ α2 ≤ 270°)
  two-stage    α1 minimising the worst-case scan-1 T1 bound with B1 known; α2 ≤ 330° (no more
               SAR than the published design) minimising the worst-case B1 bound for that α1
  low-SAR      minimum worst-case T1 bound with both angles ≤ 180°

    PYTHONPATH=src python3 scripts/flip_design_report.py
    → results/flip_design.json, docs/flipnote/figures/*.png, docs/flipnote/tables/*.tex
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpme.design import (alias_roots, alias_t1, bounds, ernst_offset, f_ratio, ratio_bias,
                         single_scan_fisher)
from mpme.phantom import TISSUES
from mpme.sequence import paper_protocol
from technote_figures import AQUA, AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, YELLOW, style

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG, TAB = ROOT / "docs" / "flipnote" / "figures", ROOT / "docs" / "flipnote" / "tables"
PR = paper_protocol()
TR, ECHOES, PATHS = PR.scans[0].TR, PR.scans[0].echo_times, PR.pathways
TISSUE = {n: dict(M0=1.0, T1=t1, T2=t2, T2star=t2s) for n, t1, t2, t2s, _ in TISSUES.values()}
PAR = ("GM", "WM", "deep GM", "lesion")
MAGENTA, GREEN = "#e87ba4", "#008300"
VIOLET = "#4a3aa7"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET)   # categorical slots 1-7, fixed order
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)))
LABEL = {"published": "published", "joint-worst": "worst case", "joint-mean": "mean",
         "no-crossing": "no crossing", "margin": "margin", "two-stage": "two-stage",
         "low-SAR": "low SAR"}


def load():
    d = np.load(RES / "flip_design.npz")
    names = [str(n) for n in d["names"]]
    return d, names


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ criteria and shortlist
def criteria(d, names):
    par = [names.index(n) for n in PAR]
    c = {"worst": d["T1_max"].max(0), "mean": d["T1_mean"].mean(0),
         "worst_par": d["T1_max"][par].max(0), "B1_worst": d["B1_max"].max(0),
         "T2_worst": d["T2_max"].max(0), "M0_worst": d["M0_max"].max(0),
         "T1known_worst": d["T1_knownB1_max"].max(0), "bias_worst": d["bias_T1_max"].max(0),
         "bias_at1_WM": d["bias_T1_at1"][names.index("WM")]}
    t1s = d["t1_scan1"]                                          # [tissue, B1, angle]
    c["scan1_known_worst_by_angle"] = t1s.max(axis=(0, 1))
    c["scan1_known_mean_by_angle"] = t1s.mean(axis=(0, 1))
    return c


def shortlist(d, c):
    ang, I, J, alias = d["ang"], d["I"], d["J"], d["alias"]
    a1, a2 = ang[I], ang[J]
    ok = alias < 2
    pick = lambda v, m: int(np.argmin(np.where(ok & m, v, np.inf)))
    allp = np.ones_like(ok)
    sl = {"published": int(np.nonzero((a1 == 15) & (a2 == 330))[0][0]),
          "joint-worst": pick(c["worst"], allp), "joint-mean": pick(c["mean"], allp),
          "no-crossing": pick(c["worst"], (a1 * 1.3 < 180) & (a2 * 0.7 > 180) & (a2 * 1.3 < 360)),
          "low-SAR": pick(c["worst"], a2 <= 180),
          "margin": pick(c["worst"], (a1 * 1.3 * 1.025 < 180) & (a2 >= 180 / (0.7 * 0.975))
                         & (a2 <= 360 / (1.3 * 1.025)))}
    a1_two = ang[int(np.argmin(c["scan1_known_worst_by_angle"]))]
    sl["two-stage"] = pick(c["B1_worst"], (a1 == a1_two) & (a2 <= 330))
    order = ("published", "joint-worst", "joint-mean", "no-crossing", "margin", "two-stage", "low-SAR")
    out = {}
    for k in order:                                   # merge rules that pick the same design
        same = [q for q in out if out[q] == sl[k]]
        if same:
            LABEL[same[0]] = f"{LABEL[same[0]]} = {LABEL[k]}"
        else:
            out[k] = sl[k]
    return out


def pareto(d, c, amax=np.arange(20, 541, 5)):
    ang, I, J, alias = d["ang"], d["I"], d["J"], d["alias"]
    ok = alias < 2
    out = {"amax": amax.tolist(), "worst": [], "mean": [], "design_worst": []}
    for m in amax:
        sel = ok & (ang[J] <= m)
        k = int(np.argmin(np.where(sel, c["worst"], np.inf)))
        out["worst"].append(float(c["worst"][k]))
        out["design_worst"].append([float(ang[I[k]]), float(ang[J[k]])])
        out["mean"].append(float(np.where(sel, c["mean"], np.inf).min()))
    return out


# ------------------------------------------------------------------ reduced model (step 1)
def offset_information(tissue, x_deg):
    """Per-scan information on the offset w (ℓ = ln M0ζ profiled, T2, T2*, Δω, φ0 known) and the
    information on ℓ, from one single-scan Fisher matrix: ∂S/∂w = (∂S/∂ln B1)/c, ∂S/∂ℓ = ∂S/∂ln M0."""
    F = single_scan_fisher(x_deg, tissue, TR, ECHOES, PATHS).numpy()
    w, c = ernst_offset(x_deg, tissue["T1"], TR)
    Kww, Kwl, Kll = F[:, 1, 1] / c**2, F[:, 1, 0] / c, F[:, 0, 0]
    return w, Kww - Kwl**2 / Kll, Kww, Kwl, Kll


def closed_form(F, c, h):
    """Bounds of Prop. closed from a 7×7 information F: Q is the information about (w1, w2) after
    profiling ℓ and every other row/column of F expressed in the coordinates (w1, w2, ℓ, rest)."""
    A = np.eye(7)
    A[0, :3], A[1, :3], A[2, :3] = (0, c[0], h), (0, c[1], h), (1, 0, -h)
    Ai = np.linalg.inv(A)
    G = Ai.T @ F @ Ai
    Q = G[:2, :2] - G[:2, 2:] @ np.linalg.solve(G[2:, 2:], G[2:, :2])
    Qi = np.linalg.inv(Q)
    L = c[1] - c[0]
    v = np.array([c[1], -c[0]])
    return (math.sqrt(Qi[0, 0] + Qi[1, 1] - 2 * Qi[0, 1]) / abs(L), math.sqrt(v @ Qi @ v) / (h * abs(L)))


def reduced_vs_full(a1, a2, tissue, b1=1.0):
    """SD of log B1 and log T1 from the closed form (Prop. closed) with T2, T2*, Δω, φ0 known and with
    them free, against the 3×3 and 7×7 bounds; and the closed-form ratio-error bias against the
    full first-order bias."""
    x = np.array([b1 * a1, b1 * a2])
    _, c = ernst_offset(x, tissue["T1"], TR)
    tau = TR / tissue["T1"]
    h = tau / (2 * math.sinh(tau))
    F = single_scan_fisher(x, tissue, TR, ECHOES, PATHS)
    Fd = F.sum(0)
    Fk = np.zeros((7, 7))
    Fk[:3, :3] = Fd[:3, :3].numpy()
    Fk[3:, 3:] = np.eye(4) * 1e30                              # nuisances known
    red = torch.linalg.inv(Fd[:3, :3]).diagonal().sqrt()
    full = bounds(Fd)
    lin = ratio_bias(Fd, F[1])
    L = float(c[1] - c[0])
    out = {"L": L, "c": c.tolist(), "h": h,
           "reduced": (float(red[1]), float(red[2])), "full": (float(full[1]), float(full[2])),
           "ratio_bias_closed": float(-c[0] * c[1] / (h * L)), "ratio_bias_full": float(lin[2]),
           "ratio_bias_M0_full": float(lin[0])}
    if abs(L) < 1e6:
        out["closed_known"] = closed_form(Fk, c, h)
        out["closed_free"] = closed_form(Fd.numpy(), c, h)
    return out


def alias_intervals(a1, a2, box=(0.6, 1.4), step=0.002):
    """True B1 in [0.7, 1.3] with an admissible complex / magnitude alias in ``box``."""
    out = {"complex": [], "magnitude": []}
    for b in np.round(np.arange(0.7, 1.3 + 1e-9, step), 4):
        for kind, mag in (("complex", False), ("magnitude", True)):
            roots = alias_roots(float(b), a1, a2, box, magnitude=mag)
            if any(math.isfinite(alias_t1(r, float(b), a1, t["T1"], TR)) for r in roots
                   for t in TISSUE.values()):
                out[kind].append(float(b))
    return out


def intervals_text(bs, step=0.002):
    if not bs:
        return "none"
    runs, start, prev = [], bs[0], bs[0]
    for b in bs[1:]:
        if b - prev > 1.5 * step:
            runs.append((start, prev))
            start = b
        prev = b
    runs.append((start, prev))
    return ", ".join(f"{u:.2f}--{v:.2f}" if f"{u:.2f}" != f"{v:.2f}" else f"{u:.2f}" for u, v in runs)


# ------------------------------------------------------------------ figures
def fig_theory():
    x = np.linspace(0.5, 719.5, 3000)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    _, c = ernst_offset(x, 850.0, TR)
    ax = axes[0]
    ax.plot(x, np.clip(c, -40, 40), color=BLUE, lw=1.5)
    for v in (180, 360, 540):
        ax.axvline(v, color=AXIS, lw=0.8, ls=":")
    ax.axhline(0, color=AXIS, lw=0.8)
    for a, off in ((15, (-14, 8)), (30, (4, 8)), (270, (5, 6)), (330, (6, -2))):
        ax.plot(a, a * math.pi / 180 / math.sin(math.radians(a)), "o", ms=7, color=ORANGE,
                markeredgecolor="white")
        ax.annotate(f"{a}°", (a, a * math.pi / 180 / math.sin(math.radians(a))), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_ylim(-40, 40)
    style(ax, "(a) B1 lever c = x / sin x", "actual flip angle x (°)", "c")
    ax = axes[1]
    for n, (name, t) in enumerate(TISSUE.items()):
        w, _ = ernst_offset(x, t["T1"], TR)
        ax.plot(x, w, color=SERIES[n], lw=1.3, ls=STYLES[n], label=name)
    for v in (180, 360, 540):
        ax.axvline(v, color=AXIS, lw=0.8, ls=":")
    ax.set_ylim(-4, 6)
    style(ax, "(b) offset from the Ernst angle", "actual flip angle x (°)",
          "w = ln|tan(x/2)| − ln tan(αE/2)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left", ncol=2)
    ax = axes[2]
    xs = np.linspace(0.5, 179.5, 1500)
    for n, (name, t) in enumerate(TISSUE.items()):
        w, Kp, *_ = offset_information(t, xs)
        ax.plot(w, Kp, color=SERIES[n], lw=1.3, ls=STYLES[n], label=name)
        k = int(np.argmax(Kp))
        ax.plot(w[k], Kp[k], MARKERS[n], ms=6, color=SERIES[n], markeredgecolor="white")
    ax.set_xlim(-3, 4)
    style(ax, "(c) per-scan information on w (amplitude unknown)", "offset w",
          "information (σ = M0 = 1)")
    save(fig, "theory.png")


def fig_maps(d, c, sl):
    ang, I, J, alias = d["ang"], d["I"], d["J"], d["alias"]
    n1 = 90
    keep = ang[I] <= n1
    grid = lambda v: _to_grid(v[keep], ang[I][keep], ang[J][keep], n1, len(ang))
    ok = alias < 2
    panels = (("(a) worst-case T1 bound", c["worst"], (50, 1000), (50, 100, 200, 500, 1000)),
              ("(b) mean T1 bound", c["mean"], (30, 500), (30, 50, 100, 200, 500)),
              ("(c) worst-case B1 bound", c["B1_worst"], (2, 200), (2, 5, 10, 20, 50, 100, 200)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    bad = grid(np.where(alias == 2, 1.0, np.nan))
    for ax, (title, v, (lo, hi), ticks) in zip(axes, panels):
        G = grid(v)
        im = ax.imshow(G, origin="lower", aspect="auto", extent=(0.5, n1 + 0.5, 0.5, ang[-1] + 0.5),
                       cmap="cividis_r", norm=matplotlib.colors.LogNorm(lo, hi))
        best = np.where(ok, v, np.inf).min()
        Gc = grid(np.where(ok, v, np.nan))                # contours over designs without aliases
        ax.contour(np.arange(1, n1 + 1), ang, Gc, levels=[1.05 * best, 1.1 * best],
                        colors=[INK, INK2], linewidths=[0.9, 0.7], linestyles=["-", "--"])
        ax.imshow(bad, origin="lower", aspect="auto", extent=(0.5, n1 + 0.5, 0.5, ang[-1] + 0.5),
                  cmap=matplotlib.colors.ListedColormap(["#e1e0d9"]), alpha=0.85)
        for n, (k, idx) in enumerate(sl.items()):
            ax.plot(ang[I[idx]], ang[J[idx]], MARKERS[n], ms=8, color=SERIES[n],
                    markeredgecolor="white", markeredgewidth=1.2, label=LABEL[k])
        cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=ticks)
        cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2)
        style(ax, title, "α1 (°)", "α2 (°)" if ax is axes[0] else None)
        ax.grid(False)
    axes[0].legend(frameon=True, fontsize=7, labelcolor=INK2, loc="lower right", facecolor=SURFACE,
                   edgecolor=GRID)
    save(fig, "maps.png")


def _to_grid(v, a1, a2, n1, n2):
    G = np.full((n2, n1), np.nan)
    G[(a2 - 1).astype(int), (a1 - 1).astype(int)] = v
    return G


def fig_profiles(d, sl):
    ang, I, J = d["ang"], d["I"], d["J"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for n, (k, idx) in enumerate(sl.items()):
        a1, a2 = ang[I[idx]], ang[J[idx]]
        cross = [180 * m / a for a in (a1, a2) for m in range(1, 5) if 0.7 <= 180 * m / a <= 1.3]
        b1 = np.unique(np.concatenate([np.linspace(0.7, 1.3, 601), cross]))
        for ax, (name, q) in zip(axes, (("WM", 2), ("CSF", 2), ("WM", 1))):
            F = single_scan_fisher(np.stack([b1 * a1, b1 * a2], 1), TISSUE[name], TR, ECHOES, PATHS)
            v = bounds(F.sum(1))[:, q].numpy()
            ax.plot(b1, v, color=SERIES[n], lw=1.5, ls=STYLES[n],
                    label=f"{LABEL[k]} ({a1:g}/{a2:g})")
    for ax, t in zip(axes, ("(a) T1 bound, WM", "(b) T1 bound, CSF", "(c) B1 bound, WM")):
        style(ax, t, "B1", "CRLB (σ/M0)")
    axes[2].set_yscale("log")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=INK2,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, "profiles.png")


def fig_pareto(par, c, d):
    ang = d["ang"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axes[0]
    ax.plot(par["amax"], par["worst"], color=BLUE, lw=1.5, label="worst case")
    ax.plot(par["amax"], par["mean"], color=ORANGE, lw=1.5, ls="--", label="mean")
    ax.set_yscale("log")
    style(ax, "(a) best joint T1 bound vs largest angle", "largest nominal angle (°)", "CRLB (σ/M0)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax = axes[1]
    m = ang <= 60
    ax.plot(ang[m], c["scan1_known_worst_by_angle"][m], color=BLUE, lw=1.5, label="worst case")
    ax.plot(ang[m], c["scan1_known_mean_by_angle"][m], color=ORANGE, lw=1.5, ls="--", label="mean")
    ax.set_yscale("log")
    style(ax, "(b) two-stage: scan-1 T1 bound, B1 known", "α1 (°)", "CRLB (σ/M0)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "pareto.png")


def fig_alias(d, sl):
    ang, I, J = d["ang"], d["I"], d["J"]
    b = np.linspace(0.6, 1.4, 2000)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for n, (k, idx) in enumerate(sl.items()):
        f = f_ratio(b, ang[I[idx]], ang[J[idx]])
        f[np.abs(f) > 60] = np.nan
        ax.plot(b, f, color=SERIES[n], lw=1.4, ls=STYLES[n], label=f"{LABEL[k]} ({ang[I[idx]]:g}/{ang[J[idx]]:g})")
    ax.axvspan(0.7, 1.3, color=GRID, alpha=0.5, lw=0)
    ax.set_ylim(-40, 40)
    style(ax, "ξ2/ξ1 = tan(B1 α2/2) / tan(B1 α1/2)", "B1", "ξ2/ξ1")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower left")
    save(fig, "alias.png")


# ------------------------------------------------------------------ tables
def tab_shortlist(d, c, sl, aliases):
    ang, I, J = d["ang"], d["I"], d["J"]
    rows = []
    for n, (k, idx) in enumerate(sl.items()):
        a1 = ang[I[idx]]
        s1 = c["scan1_known_worst_by_angle"][int(np.nonzero(ang == a1)[0][0])]
        rows.append(f"{LABEL[k]} & {a1:g}/{ang[J[idx]]:g} & {c['worst'][idx]:.1f} & "
                    f"{c['mean'][idx]:.1f} & {c['worst_par'][idx]:.1f} & {c['B1_worst'][idx]:.1f} & "
                    f"{c['T2_worst'][idx]:.1f} & {s1:.1f} & {c['bias_worst'][idx]:.2f} & "
                    f"{intervals_text(aliases[k]['magnitude'])}\\\\")
    return rows


def tab_reduced(sl, d):
    ang, I, J = d["ang"], d["I"], d["J"]
    rows, out = [], {}
    for k, idx in sl.items():
        a1, a2 = float(ang[I[idx]]), float(ang[J[idx]])
        r = reduced_vs_full(a1, a2, TISSUE["WM"])
        out[k] = r
        if "closed_free" in r:
            diff = max(abs(r["closed_known"][q] / r["reduced"][q] - 1) for q in (0, 1))
            diff = max(diff, *(abs(r["closed_free"][q] / r["full"][q] - 1) for q in (0, 1)))
            m, e = f"{diff:.0e}".split("e")
            dtxt, lever = f"${m}\\times10^{{{int(e)}}}$", f"${r['L']:+.2f}$"
        else:
            dtxt, lever = "(limit)", "$\\infty$"
        rows.append(f"{a1:g}/{a2:g} & {lever} & {r['reduced'][0]:.2f} & {r['full'][0]:.2f} & "
                    f"{r['reduced'][1]:.1f} & {r['full'][1]:.1f} & ${r['ratio_bias_full']:+.2f}$ & "
                    f"${r['ratio_bias_M0_full']:+.2f}$ & {dtxt}\\\\")
    return rows, out


def main():
    d, names = load()
    c = criteria(d, names)
    sl = shortlist(d, c)
    par = pareto(d, c)
    ang, I, J = d["ang"], d["I"], d["J"]
    TAB.mkdir(parents=True, exist_ok=True)
    aliases = {k: alias_intervals(float(ang[I[i]]), float(ang[J[i]])) for k, i in sl.items()}
    matched = {}
    for k, i in sl.items():                     # fitting range = the design's own crossing-free range
        a1, a2 = float(ang[I[i]]), float(ang[J[i]])
        if a1 * 1.3 < 180 and 180 < 0.7 * a2 and 1.3 * a2 < 360:
            box = (max(0.6, 180 / a2 + 1e-6), min(1.4, 360 / a2 - 1e-6))
            al = alias_intervals(a1, a2, box)
            matched[k] = {"box": box, "complex": intervals_text(al["complex"]),
                          "magnitude": intervals_text(al["magnitude"])}
    (TAB / "shortlist.tex").write_text("\n".join(tab_shortlist(d, c, sl, aliases)) + "\n")
    rows, red = tab_reduced(sl, d)
    (TAB / "reduced.tex").write_text("\n".join(rows) + "\n")
    fig_theory()
    fig_maps(d, c, sl)
    fig_profiles(d, sl)
    fig_pareto(par, c, d)
    fig_alias(d, sl)
    big = ang[I] > 90
    summary = {
        "shortlist": {k: {"angles": [float(ang[I[i]]), float(ang[J[i]])],
                          **{m: float(c[m][i]) for m in ("worst", "mean", "worst_par", "B1_worst",
                                                         "T2_worst", "M0_worst", "T1known_worst",
                                                         "bias_worst", "bias_at1_WM")},
                          "alias": int(d["alias"][i])} for k, i in sl.items()},
        "best_worst_with_alpha1_gt_90": float(np.where(d["alias"] < 2, c["worst"], np.inf)[big].min()),
        "alias_counts": np.bincount(d["alias"], minlength=3).tolist(),
        "best_worst_alias_free_all": float(np.where(d["alias"] == 0, c["worst"], np.inf).min()),
        "pareto": par, "reduced": red,
        "alias_intervals": {k: {q: intervals_text(v[q]) for q in v} for k, v in aliases.items()},
        "alias_matched_range": matched,
        "scan1_known_best_alpha1": float(ang[int(np.argmin(c["scan1_known_worst_by_angle"]))]),
        "scan1_known_best_alpha1_mean": float(ang[int(np.argmin(c["scan1_known_mean_by_angle"]))]),
    }
    (RES / "flip_design.json").write_text(json.dumps(summary, indent=1))
    n = np.bincount(d["alias"], minlength=3)
    thin = lambda v: f"{v:,}".replace(",", "\\,")
    (TAB / "macros.tex").write_text(
        f"\\newcommand{{\\AliasNone}}{{{thin(n[0])}}}\n"
        f"\\newcommand{{\\AliasMagnitude}}{{{thin(n[1])}}}\n"
        f"\\newcommand{{\\AliasComplex}}{{{thin(n[2])}}}\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pareto", "reduced")}, indent=1))
    for k, r in red.items():
        print(k, r)
    print("aliases", summary["alias_intervals"], summary["alias_matched_range"])


if __name__ == "__main__":
    main()
