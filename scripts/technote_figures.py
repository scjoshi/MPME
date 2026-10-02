"""Figures and tables for the technical note (docs/technote/).

Needs results/phantom_snr*.pt and results/phantom_summary.json (scripts/phantom_experiment.py)
and results/start_count.json (scripts/start_count.py).

    python scripts/technote_figures.py
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

from mpme.sequence import paper_protocol

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "technote" / "figures"
TAB = ROOT / "docs" / "technote" / "tables"
RES = ROOT / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIV = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SEQ = "cividis"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def show_map(ax, img, mask, cmap, vmin, vmax, title):
    a = np.where(mask, img, np.nan)
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    return im


def colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, color=INK2, fontsize=8)


def to_img(vox, mask):
    out = np.full(mask.shape, np.nan)
    out[mask] = np.asarray(vox, dtype=float)
    return out


# ------------------------------------------------------------------------------------------
def fig_sequence():
    """Gx waveform and accumulated moment m(t) for one TR of the [1, 0, −1] scheme."""
    sp = 2.0                                   # ms per unit of moment (pathway spacing)
    # (t_start, t_end, m_start, m_end) segments; prephaser played faster (2 ms for 1.5 units)
    segs = [(0.0, 1.5, 0.0, 0.0), (1.5, 3.5, 0.0, -1.5),
            (3.5, 9.5, -1.5, 1.5), (9.5, 15.5, 1.5, -1.5), (15.5, 21.5, -1.5, 1.5),
            (21.5, 22.5, 1.5, 1.0), (22.5, 25.0, 1.0, 1.0)]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})
    ax = axes[0]
    for t0, t1, m0, m1 in segs:
        G = (m1 - m0) / (t1 - t0) if t1 > t0 else 0
        ax.fill_between([t0, t1], [G, G], color=BLUE if G > 0 else ORANGE, alpha=0.85, step="pre",
                        linewidth=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.bar([0.4], [1.0], width=0.8, color=INK2)
    ax.text(1.0, 0.75, "RF", color=INK2, fontsize=8)
    style(ax, "Readout gradient (units of A per ms)")
    ax.set_ylim(-0.95, 1.1)
    ax = axes[1]
    ts, ms = [], []
    for t0, t1, m0, m1 in segs:
        ts += [t0, t1]; ms += [m0, m1]
    ax.plot(ts, ms, color=INK, lw=2)
    pr = paper_protocol()
    cols = {1: BLUE, 0: AQUA, -1: ORANGE}
    for p, times in zip(pr.pathways, pr.scans[0].echo_times):
        ax.axhline(-p, color=cols[p], lw=1, ls=":")
        ax.plot(times, [-p] * len(times), "o", ms=8, color=cols[p], markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"pathway k = {p:+d}" if p else "pathway k = 0")
    ax.set_yticks([-1.5, -1, 0, 1, 1.5])
    style(ax, "Moment since the RF pulse, m(t) (units of A); pathway k refocuses at m = −k",
          "time after the RF pulse (ms)", "m(t)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=1, loc="upper left")
    fig.tight_layout()
    save(fig, "sequence_timing.png")


# ------------------------------------------------------------------------------------------
def load(snr):
    return torch.load(RES / f"phantom_snr{snr:g}.pt", weights_only=False)


def fig_phantom(D):
    ph = D["phantom"]; m = ph.mask.numpy()
    panels = [(ph.M0, "C·M0", SEQ, 0, 1.25, None), (ph.T1, "T1 (ms)", SEQ, 500, 4200, None),
              (ph.T2, "T2 (ms)", SEQ, 40, 160, None), (ph.T2star, "T2* (ms)", SEQ, 30, 110, None),
              (ph.B1, "B1⁺ (× nominal)", SEQ, 0.85, 1.16, None),
              (ph.dw * 1000 / (2 * math.pi), "Δf (Hz)", DIV, -50, 50, None)]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    for ax, (img, title, cm, lo, hi, _) in zip(axes.flat, panels):
        im = show_map(ax, img.numpy(), m, cm, lo, hi, title)
        colorbar(fig, im, ax)
    # B1 contour at α2 = 360°
    axes[1, 1].contour(np.where(m, ph.B1.numpy(), np.nan), levels=[360 / 330], colors=[INK],
                       linewidths=1, linestyles="--")
    axes[1, 1].text(5, 250, "dashed: α2 = 360°", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "phantom_truth.png")


def fig_images(D):
    ph = D["phantom"]; m = ph.mask.numpy()
    Sn, S_low = D["Sn"], D["S_low"]
    pr = paper_protocol()
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.8))
    for i in range(2):
        for j, p in enumerate(pr.pathways):
            img = Sn[..., i, j, 0].abs().numpy()
            vmax = np.percentile(img[m], 99.5)
            show_map(axes[i, j], img, np.ones_like(m), "gray", 0, vmax,
                     f"scan {i + 1} ({pr.scans[i].flip_deg:g}°), k = {p:+d}, echo 1".replace("+0", "0"))
    img = S_low[..., 1, 1, 0].abs().numpy()
    show_map(axes[1, 3], img, np.ones_like(m), "gray", 0, np.percentile(img[m], 99.5),
             "scan 2, k = 0, low resolution")
    img = Sn[..., 0, 1, 2].abs().numpy()
    show_map(axes[0, 3], img, np.ones_like(m), "gray", 0, np.percentile(img[m], 99.5),
             "scan 1, k = 0, echo 3")
    fig.tight_layout()
    save(fig, "phantom_images.png")


def fig_scenario_A(D, snr):
    ph = D["phantom"]; mt = ph.mask; m = mt.numpy()
    R = D["recon"]
    truth = {"T1": ph.T1, "B1": ph.B1, "T2": ph.T2}
    algs = [("A_paper", "Analytic inverse"), ("A_ml", "Complex ML"), ("A_mag", "Magnitude LS + FID")]
    rng = {"T1": (500, 4200, 0.3), "B1": (0.85, 1.16, 0.05), "T2": (40, 160, 0.3)}
    for key in ("T1", "B1", "T2"):
        lo, hi, el = rng[key]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        im = show_map(axes[0, 0], truth[key].numpy(), m, SEQ, lo, hi, f"truth {key}")
        axes[1, 0].axis("off")
        for c, (k, name) in enumerate(algs, start=1):
            est = to_img(R[k][key].double().numpy(), m)
            im = show_map(axes[0, c], est, m, SEQ, lo, hi, name)
            err = np.log(est / truth[key].numpy())
            ime = show_map(axes[1, c], err, m, DIV, -el, el, f"log({key} est / true)")
        colorbar(fig, im, axes[0, 3]); colorbar(fig, ime, axes[1, 3])
        fig.suptitle(f"Scenario A (both scans full resolution), SNR(M0) = {snr:g}: {key}",
                     color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        save(fig, f"phantomA_{key}_snr{snr:g}.png")


def fig_scenario_B(D, snr):
    ph = D["phantom"]; m = ph.mask.numpy()
    R = D["recon"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    show_map(axes[0, 0], ph.B1.numpy(), m, SEQ, 0.85, 1.16, "truth B1⁺")
    show_map(axes[0, 1], to_img(R["B_paper"]["B1_low"].numpy(), m), m, SEQ, 0.85, 1.16,
             "paper: low-res B1⁺ (Eq. 15)")
    show_map(axes[0, 2], to_img(D["B1_smooth_paper"].numpy(), m), m, SEQ, 0.85, 1.16,
             "paper: polynomial fit")
    im = show_map(axes[0, 3], to_img(R["B_ml"]["B1_low"].numpy(), m), m, SEQ, 0.85, 1.16,
                  "ML: low-res B1⁺")
    colorbar(fig, im, axes[0, 3])
    for c, (k, name) in enumerate((("B_paper", "paper pipeline"), ("B_ml", "ML pipeline"))):
        est = to_img(R[k]["T1"].double().numpy(), m)
        show_map(axes[1, 2 * c], est, m, SEQ, 500, 4200, f"{name}: T1")
        ime = show_map(axes[1, 2 * c + 1], np.log(est / ph.T1.numpy()), m, DIV, -0.3, 0.3,
                       f"{name}: log(T1 est / true)")
    colorbar(fig, ime, axes[1, 3])
    fig.suptitle(f"Scenario B (published acquisition, scan 2 at 25 % of ky), SNR(M0) = {snr:g}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, f"phantomB_snr{snr:g}.png")


def fig_uncertainty(D, snr):
    ph = D["phantom"]; m = ph.mask.numpy()
    R = D["recon"]
    sd = to_img(R["A_ml_sd"]["T1"].numpy(), m)
    err = np.log(to_img(R["A_ml"]["T1"].double().numpy(), m) / ph.T1.numpy())
    z = (err / sd)[m]
    z = z[np.isfinite(z)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    im = show_map(axes[0], sd, m, SEQ, 0, np.nanpercentile(sd[m], 98), "predicted SD of log T1")
    colorbar(fig, im, axes[0])
    im = show_map(axes[1], np.abs(err), m, SEQ, 0, np.nanpercentile(sd[m], 98) * 2, "|log(T1 est / true)|")
    colorbar(fig, im, axes[1])
    ax = axes[2]
    ax.hist(np.clip(z, -5, 5), bins=80, density=True, color=BLUE, alpha=0.85)
    xs = np.linspace(-5, 5, 200)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.5, label="N(0, 1)")
    style(ax, f"z = error / predicted SD (SD {z.std():.2f})", "z", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"Uncertainty maps, complex ML, Scenario A, SNR(M0) = {snr:g}", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, f"phantom_uncertainty_snr{snr:g}.png")


# ------------------------------------------------------------------------------------------
def tables():
    S = json.loads((RES / "phantom_summary.json").read_text())
    algs = ["A: analytic inverse", "A: complex ML", "A: magnitude LS + FID sign",
            "B: paper pipeline", "B: ML pipeline"]
    short = {"A: analytic inverse": "A, analytic inverse", "A: complex ML": "A, complex ML",
             "A: magnitude LS + FID sign": "A, magnitude LS + FID",
             "B: paper pipeline": "B, paper pipeline", "B: ML pipeline": "B, ML pipeline"}
    lines = []
    for snr_key, row in S.items():
        snr = snr_key.split()[1]
        lines += [r"\begin{table}[!htbp]", r"\centering",
                  rf"\caption{{Phantom, SNR$(\Mzero)={snr}$: median and SD of $\log(\hat\theta/\theta)$ "
                  r"over each tissue (bias and spread), and the fraction of voxels with an undefined "
                  r"estimate. Truth includes a $3\%$ smooth texture.}",
                  rf"\label{{tab:roi{snr}}}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
                  r"\begin{tabular}{llcccccc}", r"\toprule",
                  r"Tissue & Method & $\Tone$ median & $\Tone$ SD & $\Ttwo$ median & $\Ttwo$ SD & "
                  r"$\Bone$ median & undef.\,\%\\", r"\midrule"]
        for tissue in ("WM", "GM", "deep GM", "lesion", "CSF"):
            first = True
            for a in algs:
                r = row["roi"][a][tissue]
                cell = lambda k, f: (f"{r[k][f]:+.3f}" if f == "median" else f"{r[k][f]:.3f}") \
                    if k in r and math.isfinite(r[k][f]) else "--"
                b1 = cell("B1", "median") if "B1" in r else "--"
                lines.append(f"{tissue if first else ''} & {short[a]} & {cell('T1', 'median')} & "
                             f"{cell('T1', 'sd')} & {cell('T2', 'median')} & {cell('T2', 'sd')} & "
                             f"{b1} & {100 * r['T1']['undefined']:.1f}\\\\")
                first = False
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table}", ""]
    (TAB / "phantom_roi.tex").write_text("\n".join(lines) + "\n")

    # run times
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Reconstruction times for the phantom (voxels in the mask; laptop CPU, "
             r"8 threads, implicit Jacobian, float64, 128 isochromats, 40 iterations).}",
             r"\label{tab:phantomtime}", r"\small", r"\begin{tabular}{lrr}", r"\toprule",
             r"Step & " + " & ".join(f"SNR {k.split()[1]}" for k in S) + r"\\", r"\midrule"]
    keys = list(next(iter(S.values()))["times_s"].keys())
    for k in keys:
        lines.append(f"{k} & " + " & ".join(f"{S[s]['times_s'][k]:.1f} s" for s in S) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "phantom_times.tex").write_text("\n".join(lines) + "\n")

    # calibration + wrong branch macros
    macros = []
    for s, row in S.items():
        tag = "".join(ch for ch in s.split()[1] if ch.isdigit())
        tagw = {"1000": "Thousand", "300": "ThreeHundred"}.get(tag, tag)
        c = row["calibration"]
        macros += [rf"\newcommand{{\CalAzT{tagw}}}{{{c['A: complex ML']['T1']['sd_z']:.2f}}}",
                   rf"\newcommand{{\CalAcovT{tagw}}}{{{100 * c['A: complex ML']['T1']['coverage95']:.1f}}}",
                   rf"\newcommand{{\CalBzT{tagw}}}{{{c['B: ML pipeline']['T1']['sd_z']:.2f}}}",
                   rf"\newcommand{{\CalBcovT{tagw}}}{{{100 * c['B: ML pipeline']['T1']['coverage95']:.1f}}}",
                   rf"\newcommand{{\WrongPaper{tagw}}}{{{row['wrong_branch']['A: analytic inverse']}}}",
                   rf"\newcommand{{\WrongML{tagw}}}{{{row['wrong_branch']['A: complex ML']}}}",
                   rf"\newcommand{{\WrongMag{tagw}}}{{{row['wrong_branch']['A: magnitude LS + FID sign']}}}",
                   rf"\newcommand{{\Voxels}}{{{row['voxels']}}}" if tag == "1000" else ""]
    sc = json.loads((RES / "start_count.json").read_text())
    for s, row in sc.items():
        tag = {"1000": "Thousand", "300": "ThreeHundred"}[s.split()[1]]
        macros += [rf"\newcommand{{\StartWorse{tag}}}{{{100 * row['one_start_worse'] / row['voxels']:.1f}}}",
                   rf"\newcommand{{\StartFlagged{tag}}}{{{100 * row['worse_and_flagged'] / max(row['one_start_worse'], 1):.0f}}}",
                   rf"\newcommand{{\StartMedian{tag}}}{{{row['median_abs_dlogT1_when_worse']:.3f}}}",
                   rf"\newcommand{{\StartMax{tag}}}{{{row['max_abs_dlogT1_when_worse']:.1f}}}"]
    (TAB / "macros.tex").write_text("\n".join(x for x in macros if x) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fig_sequence()
    for i, snr in enumerate((1000, 300)):
        p = RES / f"phantom_snr{snr}.pt"
        if not p.exists():
            continue
        D = load(snr)
        if i == 0:
            fig_phantom(D)
            fig_images(D)
        fig_scenario_A(D, snr)
        fig_scenario_B(D, snr)
        fig_uncertainty(D, snr)
    tables()
    fig_brainweb()
    print("figures and tables written to", FIG.parent)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------
# BrainWeb partial volume
def fig_brainweb():
    import json as _json
    paths = {c: RES / f"brainweb_{c}.pt" for c in ("pv_clean", "pv", "crisp")}
    if not all(p.exists() for p in paths.values()):
        return
    D = {c: torch.load(p, weights_only=False) for c, p in paths.items()}
    ph = D["pv"]["phantom"]; m = ph.mask.numpy(); mt = ph.mask
    F = ph.fractions.numpy()
    # Figure 1: fractions and effective truth
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for ax, (c, name) in zip(axes[0], ((0, "CSF fraction"), (1, "GM fraction"), (2, "WM fraction"))):
        im = show_map(ax, F[c], m, SEQ, 0, 1, name); colorbar(fig, im, ax)
    im = show_map(axes[1, 0], ph.effective("T1").numpy(), m, SEQ, 500, 4200, "effective T1 (ms)")
    colorbar(fig, im, axes[1, 0])
    im = show_map(axes[1, 1], ph.effective("T2").numpy(), m, SEQ, 40, 300, "effective T2 (ms)")
    colorbar(fig, im, axes[1, 1])
    mixed = (F.max(0) < 0.95).astype(float)
    im = show_map(axes[1, 2], np.where(F[0] >= 0.05, 2.0, mixed), m,
                  matplotlib.colors.ListedColormap(["#f0efec", AQUA, ORANGE]), -0.5, 2.5,
                  "pure / WM-GM mixed / CSF-mixed")
    fig.tight_layout(); save(fig, "brainweb_phantom.png")

    # Figure 2: maps — ML estimate and error, PV vs crisp control, T2 and T1
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    for r, (key, lo, hi) in enumerate((("T1", 500, 4200), ("T2", 40, 300))):
        eff = ph.effective(key).numpy()
        est_pv = to_img(D["pv"]["est"]["complex ML"][key].double().numpy(), m)
        crisp_ph = D["crisp"]["phantom"]
        est_cr = to_img(D["crisp"]["est"]["complex ML"][key].double().numpy(), m)
        im = show_map(axes[r, 0], est_pv, m, SEQ, lo, hi, f"ML {key}, partial volume")
        colorbar(fig, im, axes[r, 0])
        dom = F.argmax(0)
        tissue_val = np.take_along_axis(getattr(ph, key).numpy(), dom[None], 0)[0]
        ime = show_map(axes[r, 1], np.log(est_pv / tissue_val), m, DIV, -0.5, 0.5,
                       f"log({key} / dominant tissue)")
        show_map(axes[r, 2], np.log(est_pv / eff), m, DIV, -0.5, 0.5, f"log({key} / PD-weighted mean)")
        show_map(axes[r, 3], np.log(est_cr / getattr(crisp_ph, "effective")(key).numpy()), m, DIV,
                 -0.5, 0.5, f"control (no partial volume)")
        colorbar(fig, ime, axes[r, 3])
    fig.suptitle("BrainWeb, complex ML, SNR(M0) = 1000", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "brainweb_maps.png")

    # Figure 3: apparent parameters vs CSF fraction in GM/CSF voxels (noise-free)
    clean = D["pv_clean"]; e = clean["est"]; phc = clean["phantom"]
    Fv = phc.fractions[:, mt]
    sel = (Fv[2] < 0.05) & (Fv[0] > 0.0) & (Fv[1] > 0.0)
    fcsf = Fv[0][sel].numpy()
    order = np.argsort(fcsf)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, ylog in zip(axes, ("T1", "T2", "B1"), (True, True, False)):
        if key == "B1":
            truth_b1 = phc.B1[mt][sel].numpy()
            for name, col in (("complex ML", BLUE), ("analytic inverse", ORANGE)):
                v = e[name]["B1"][sel].double().numpy() / truth_b1
                ax.plot(fcsf, 100 * (v - 1), ".", ms=3, color=col, alpha=0.5, label=name)
            ax.axhline(0, color=INK2, lw=1)
            style(ax, "B1⁺ error (%) vs CSF fraction", "CSF volume fraction", "B1⁺ error (%)")
        else:
            for name, col in (("complex ML", BLUE), ("analytic inverse", ORANGE)):
                v = e[name][key][sel].double().numpy()
                ax.plot(fcsf, v, ".", ms=3, color=col, alpha=0.5, label=name)
            from mpme.phantom import TISSUES as _T
            nom = {v[0]: v for v in _T.values()}
            idx = {"T1": 1, "T2": 2}[key]
            fc = np.linspace(0.0, 1.0, 200)
            w = fc * nom["CSF"][4] / (fc * nom["CSF"][4] + (1 - fc) * nom["GM"][4])
            eff = np.exp(w * np.log(nom["CSF"][idx]) + (1 - w) * np.log(nom["GM"][idx]))
            ax.plot(fc, eff, color=INK, lw=1.8, label="PD-weighted mean (nominal)")
            ax.axhline(nom["GM"][idx], color=MUTED, lw=1.5, ls="--", label="GM value (nominal)")
            ax.set_yscale("log")
            style(ax, f"apparent {key} (ms) vs CSF fraction", "CSF volume fraction", f"{key} (ms)")
        ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9, fontsize=7,
                  labelcolor=INK2, markerscale=3, loc="lower right" if key == "B1" else "upper left")
    fig.suptitle("BrainWeb GM/CSF voxels, noise-free: the single-compartment fit of a mixture",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); save(fig, "brainweb_mixture_curves.png")

    # Tables
    S = _json.loads((RES / "brainweb_summary.json").read_text())
    cats = ["pure CSF", "pure GM", "pure WM", "WM/GM mixed", "CSF 5-25%", "CSF 25-50%", "CSF 50-95%"]
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{BrainWeb, noise-free: median of $\log(\hat\theta/\theta_{\rm eff})$ per voxel category, "
             r"where $\theta_{\rm eff}$ is the PD-weighted geometric mean of the compartment values "
             r"($\Bone$: relative to the true field). Last columns: RMS mismatch residual per "
             r"measurement in units of $\sigma$ at SNR 1000, and the probability that a $\chi^2$ test "
             r"(99.9\,\%) flags the voxel at SNR 1000 and 3000.}",
             r"\label{tab:bwclean}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrcccccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Complex ML} & \multicolumn{3}{c}{Analytic inverse} & & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Category & voxels & $\Tone$ & $\Ttwo$ & $\Bone$ & $\Tone$ & $\Ttwo$ & $\Bone$ & RMS/$\sigma$ & flagged 1000 / 3000\\",
             r"\midrule"]
    sc = S["pv_clean"]
    for c in cats:
        ml, pa = sc["stats"]["complex ML"][c], sc["stats"]["analytic inverse"][c]
        f = lambda d, k: f"{d[k]['median']:+.3f}".replace("-0.000", "+0.000")
        det = sc["detection_power_chi2_99.9"]
        lines.append(f"{c.replace('%', chr(92) + '%')} & {ml['voxels']} & {f(ml, 'T1')} & {f(ml, 'T2')} & {f(ml, 'B1')} & "
                     f"{f(pa, 'T1')} & {f(pa, 'T2')} & {f(pa, 'B1')} & "
                     f"{sc['mismatch_rms_over_sigma1000'][c]:.2f} & "
                     f"{100 * det['SNR 1000'][c]:.1f}\\,\\% / {100 * det['SNR 3000'][c]:.1f}\\,\\%\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines += [r"\begin{table}[!htbp]", r"\centering",
              r"\caption{BrainWeb at SNR$(\Mzero)=1000$, complex ML: SD of $\log(\hat\Tone/\Tone_{\rm eff})$ with "
              r"partial volume and in the control without it, coverage of $95\%$ uncertainty intervals, "
              r"and the fraction of voxels flagged by the $\chi^2$ test.}",
              r"\label{tab:bwnoisy}", r"\footnotesize",
              r"\begin{tabular}{lccccc}", r"\toprule",
              r"Category & $\Tone$ SD (PV) & $\Tone$ SD (control) & median (PV) & coverage (PV) & flagged (PV)\\",
              r"\midrule"]
    pv, cr = S["pv"], S["crisp"]
    for c in cats:
        a, b = pv["stats"]["complex ML"][c]["T1"], cr["stats"]["complex ML"][c]["T1"]
        lines.append(f"{c.replace('%', chr(92) + '%')} & {a['sd']:.3f} & {b['sd']:.3f} & {a['median']:+.3f} & "
                     f"{100 * pv['coverage_T1'][c]['coverage95']:.1f}\\,\\% & "
                     f"{100 * pv['flagged_chi2_99.9'][c]:.1f}\\,\\%\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "brainweb.tex").write_text("\n".join(lines) + "\n")
