"""Turn results/estimator_efficiency.json into docs/latex/mc_table.tex.

Three tables: log T1 errors in full (tab:mc), log B1 / T2 / M0 (tab:mcb1), and the ML
diagnostics with the matched bounds (tab:mcdiag).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "estimator_efficiency.json"
res = json.loads(src.read_text())
res.pop("settings", None)

ORDER = ["joint ML", "model-fit", "paper", "paper, B1 known"]
NAMES = {"joint ML": "Joint ML", "model-fit": "Model fit",
         "paper": "Analytic inverse",
         "paper, B1 known": "Analytic, $\\Bone$ given"}


def f3(x):
    if x is None:
        return "--"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def sgn(x, nd=3):
    s = f"{x:+.{nd}f}"
    return s.replace("-0." + "0" * nd, "+0." + "0" * nd)


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def proto_tex(case):
    proto, _, snr = case.split()
    return (proto.replace("15/330", r"$15^\circ/330^\circ$").replace("15/30", r"$15^\circ/30^\circ$"),
            snr)


out = []
# ---- Table: log T1 --------------------------------------------------------------------
out += [r"\begin{table}[p]", r"\centering",
        r"\caption{Monte Carlo errors of $\log\hat\Tone$ (1000 noise realisations, reference "
        r"tissue, $\Bone=1$). SD, mean (bias), RMSE and robust spread (IQR$/1.349$) are over "
        r"voxels with a defined estimate. ``Undef.'': fraction with no valid $\Tone$; "
        r"``bound'': fraction with a constraint or search limit active (for ML: $\Rtwop=0$ or "
        r"the safeguard box). CRLB: bound on the SD for unbiased estimators using both scans "
        r"with $\Bone$ unknown.}",
        r"\label{tab:mc}", r"\small",
        r"\begin{tabular}{lllcccccc}", r"\toprule",
        r"Protocol & SNR$(\Mzero)$ & Estimator & SD & mean & RMSE & robust & undef.\,\% & bound\,\%\\",
        r"\midrule"]
for case, row in res.items():
    proto, snr = proto_tex(case)
    out.append(f"{proto} & {snr} & CRLB & {f3(row['CRLB']['all unknown']['T1'])} & & & & & \\\\")
    for est in ORDER:
        if est not in row:
            continue
        d = row[est]["T1"]
        out.append(f" & & {NAMES[est]} & {f3(d['sd'])} & {sgn(d['mean'])} & {f3(d['rmse'])} & "
                   f"{f3(d['robust_sd'])} & {pct(d['undefined_frac'])} & {pct(d['boundary_frac'])}\\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
out += [r"\end{tabular}", r"\end{table}", ""]

# ---- Table: log B1, T2, M0 ----------------------------------------------------------------
out += [r"\begin{table}[p]", r"\centering",
        r"\caption{Monte Carlo SD (robust spread in parentheses) of $\log\hat\Bone$, "
        r"$\log\hat\Ttwo$ and $\log\hat\Mzero$, and the mean of $\log\hat\Bone$. Same runs as "
        r"Table~\ref{tab:mc}.}",
        r"\label{tab:mcb1}", r"\footnotesize",
        r"\begin{tabular}{lllcccc}", r"\toprule",
        r"Protocol & SNR & Estimator & $\Bone$ SD (rob.) & $\Bone$ mean & "
        r"$\Ttwo$ SD (rob.) & $\Mzero$ SD (rob.)\\",
        r"\midrule"]
for case, row in res.items():
    proto, snr = proto_tex(case)
    c = row["CRLB"]["all unknown"]
    out.append(f"{proto} & {snr} & CRLB & {f3(c['B1'])} & & {f3(c['T2'])} & {f3(c['M0'])}\\\\")
    for est in ORDER:
        if est not in row or "B1" not in row[est]:
            continue
        d = row[est]
        cell = lambda k: f"{f3(d[k]['sd'])} ({f3(d[k]['robust_sd'])})"
        out.append(f" & & {NAMES[est]} & {cell('B1')} & {sgn(d['B1']['mean'], 4)} & {cell('T2')} & "
                   f"{cell('M0')}\\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
out += [r"\end{tabular}", r"\end{table}", ""]

# ---- Table: diagnostics and matched bounds ------------------------------------------------
out += [r"\begin{table}[h]", r"\centering",
        r"\caption{Joint ML diagnostics, and the analytic inverse with $\Bone$ given compared "
        r"with matched bounds: both scans with $\Bone$ known (28.0 per unit $\sigma/\Mzero$) and "
        r"scan 1 only with $\Bone$ known (57.1), the data its $\Tone$ step actually uses.}",
        r"\label{tab:mcdiag}", r"\small",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r" & & \multicolumn{3}{c}{Joint ML} & \multicolumn{2}{c}{Analytic, $\Bone$ given: SD / bound}\\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"Protocol & SNR & $\Rtwop=0$\,\% & box\,\% & cost $>$ cost(truth) & both scans & scan 1 only\\",
        r"\midrule"]
for case, row in res.items():
    proto, snr = proto_tex(case)
    g = row["joint ML diagnostics"]
    if "paper, B1 known" in row:
        sd = row["paper, B1 known"]["T1"]["sd"]
        r2 = sd / row["CRLB"]["B1 known"]["T1"]
        r1 = sd / row["CRLB"]["scan 1, B1 known"]["T1"]
        ratios = f"{r2:.2f} & {r1:.2f}"
    else:
        ratios = "-- & --"
    out.append(f"{proto} & {snr} & {pct(g['r2p_bound_frac'])} & {pct(g['box_frac'])} & "
               f"{g['cost_above_truth']} & {ratios}\\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

(ROOT / "docs" / "latex" / "mc_table.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
