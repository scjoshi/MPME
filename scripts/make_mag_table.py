"""Turn results/magnitude_comparison.json into docs/latex/mag_table.tex and mag_counts.tex."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "magnitude_comparison.json"
res = json.loads(src.read_text())
res.pop("settings", None)

ORDER = [("paper", "Analytic inverse~\\cite{cheng2019}"),
         ("magnitude LS", "Magnitude LS"),
         ("magnitude LS + FID sign", "Magnitude LS + FID sign"),
         ("joint ML (complex)", "Complex ML")]
WRONG = {"magnitude LS": "magnitude LS diagnostics",
         "magnitude LS + FID sign": "magnitude LS + FID sign diagnostics"}


def sgn(x):
    s = f"{x:+.3f}"
    return s.replace("-0.000", "+0.000")


lines = [r"\begin{table}[h]", r"\centering",
         r"\caption{Magnitude versus complex estimation on identical data (paper protocol, "
         r"reference tissue, 1000 realisations). SD and mean of the log-estimates; bounds for "
         r"complex data and for magnitude (Rician) data. Last column: voxels on the wrong side "
         r"of $\alpha_2=360^\circ$ (``wrong'').}",
         r"\label{tab:mag}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"SNR & & $\Tone$ SD & $\Tone$ mean & $\Bone$ SD & $\Ttwo$ SD & wrong\\",
         r"\midrule"]
for case, row in res.items():
    snr = case.split()[1]
    c, r = row["CRLB complex"], row["CRLB magnitude (Rician)"]
    lines.append(f"{snr} & CRLB complex/Rician & {c['T1']:.4f}/{r['T1']:.4f} & & "
                 f"{c['B1']:.4f}/{r['B1']:.4f} & {c['T2']:.4f}/{r['T2']:.4f} & \\\\")
    for key, name in ORDER:
        d = row[key]
        wrong = row[WRONG[key]]["wrong_branch"] if key in WRONG else "--"
        lines.append(f" & {name} & {d['T1']['sd']:.4f} & {sgn(d['T1']['mean'])} & "
                     f"{d['B1']['sd']:.4f} & {d['T2']['sd']:.4f} & {wrong}\\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(ROOT / "docs" / "latex" / "mag_table.tex").write_text("\n".join(lines) + "\n")

w3 = res.get("SNR 3000", {}).get("magnitude LS diagnostics", {}).get("wrong_branch", "?")
(ROOT / "docs" / "latex" / "mag_counts.tex").write_text(
    f"\\newcommand{{\\MagWrongThree}}{{{w3}}}\n")
print("\n".join(lines))
