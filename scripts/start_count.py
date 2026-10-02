"""Is one ML start enough? One start (model-fit estimate) vs the best of four.

For each voxel, compares the final cost from the single start with the best of four starts
(model fit, analytic inverse, T1 = 700 ms, T1 = 2000 ms), and tests a χ² check on the final
cost as a detector of poorly converged voxels: flag when ‖r‖²/σ² exceeds the 99.9 %
quantile of χ² with (36 − 7) degrees of freedom.

    python scripts/start_count.py   → results/start_count.json
"""

import json
from pathlib import Path

import torch
from scipy.stats import chi2

from mpme.analytic import analytic_reconstruction
from mpme.mle import joint_fit
from mpme.paper import paper_reconstruction
from mpme.sequence import paper_protocol
from mpme.signal import add_noise, mpme_signal

RES = Path(__file__).resolve().parents[1] / "results"
FIT = dict(jacobian="implicit", n_iso=128, n_iter=40, b1_max=540 / 330)


def main():
    pr = paper_protocol()
    N = 2000
    g = torch.Generator().manual_seed(11)
    d = torch.float64
    T1 = 600 + 2400 * torch.rand(N, generator=g, dtype=d)
    T2 = 40 + 100 * torch.rand(N, generator=g, dtype=d)
    T2s = T2 * (0.6 + 0.35 * torch.rand(N, generator=g, dtype=d))
    B1 = 0.85 + 0.3 * torch.rand(N, generator=g, dtype=d)
    dw = (torch.rand(N, generator=g, dtype=d) - 0.5) * 0.6
    S = mpme_signal(pr, 1.0, T1, T2, 1 / T2s - 1 / T2, dw, B1, 0.3)
    out = {}
    for snr in (300.0, 1000.0):
        sigma = 1 / snr
        Sn = add_noise(S, sigma, torch.Generator().manual_seed(int(snr)))
        fit = analytic_reconstruction(Sn, pr)
        pap = paper_reconstruction(Sn, pr)
        starts = [fit, {k: torch.where(torch.isfinite(pap[k]) & (pap[k] > 0), pap[k], fit[k]) for k in fit}]
        for t1 in (700.0, 2000.0):
            starts.append({**fit, "T1": torch.full_like(fit["T1"], t1),
                           "T2star": torch.minimum(fit["T2star"], 0.85 * fit["T2"])})
        fits = [joint_fit(Sn, pr, s, **FIT) for s in starts]
        cost = torch.stack([f["cost"] for f in fits])
        best = cost.min(0).values
        one = cost[0]
        worse = one > best * (1 + 1e-6) + 1e-30
        thr = chi2.ppf(0.999, 36 - 7) * sigma**2
        flag = one > thr
        dT1 = (fits[0]["T1"] / torch.stack([f["T1"] for f in fits]).gather(0, cost.argmin(0)[None])[0]).log().abs()
        row = {"voxels": N, "one_start_worse": int(worse.sum()),
               "worse_and_flagged": int((worse & flag).sum()),
               "flagged_total": int(flag.sum()),
               "false_flag_rate": (flag & ~worse).float().mean().item(),
               "max_abs_dlogT1_when_worse": dT1[worse].max().item() if worse.any() else 0.0,
               "median_abs_dlogT1_when_worse": dT1[worse].median().item() if worse.any() else 0.0}
        out[f"SNR {snr:g}"] = row
        print(f"SNR {snr:g}: {row}")
    (RES / "start_count.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
