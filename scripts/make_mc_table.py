"""Turn results/estimator_efficiency.json into docs/latex/mc_table.tex."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "results" / "estimator_efficiency.json").read_text())

ORDER = ["CRLB", "joint ML", "model-fit", "paper", "paper, B1 known"]
NAMES = {"CRLB": "CRLB", "joint ML": "Joint ML", "model-fit": "Model fit",
         "paper": "Analytic inverse~\\cite{cheng2019}",
         "paper, B1 known": "Analytic inverse, $\\Bone$ given"}
KEYS = [("T1", "$\\Tone$"), ("B1", "$\\Bone$"), ("T2", "$\\Ttwo$"), ("M0", "$\\Mzero$")]


def cell(d, k):
    if k not in d:
        return "--"
    v = 100 * d[k]["robust_sd"]
    return f"{v:.1f}" if v < 1000 else f"{v:.0f}"


lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\caption{Monte Carlo robust standard deviation (\%) of the log-estimates, 1000 noise "
    r"realisations, reference tissue, $\Bone=1$. SNR is relative to $\Mzero$ (peak image SNR "
    r"$\approx0.088\times$). Last column: fraction of voxels with undefined $\Tone$.}",
    r"\label{tab:mc}",
    r"\small",
    r"\begin{tabular}{lllccccc}",
    r"\toprule",
    "Protocol & SNR$(\\Mzero)$ & Estimator & " + " & ".join(n for _, n in KEYS) +
    r" & $\Tone$ undefined\\",
    r"\midrule",
]
for case, row in res.items():
    proto, _, snr = case.split()
    proto = proto.replace("15/330", r"$15^\circ/330^\circ$").replace("15/30", r"$15^\circ/30^\circ$")
    first = True
    for est in ORDER:
        if est not in row:
            continue
        d = row[est]
        nan = d.get("T1", {}).get("nan_frac")
        nan_s = "--" if nan is None else f"{100 * nan:.1f}\\%"
        lines.append(f"{proto if first else ''} & {snr if first else ''} & {NAMES[est]} & "
                     + " & ".join(cell(d, k) for k, _ in KEYS) + f" & {nan_s}\\\\")
        first = False
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(ROOT / "docs" / "latex" / "mc_table.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
