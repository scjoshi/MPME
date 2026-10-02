# References

## Core

- **Cheng CC, Preiswerk F, Hoge WS, Kuo TH, Madore B.** Multipathway multi-echo (MPME)
  imaging: all main MR parameters mapped based on a single 3D scan.
  *Magn Reson Med.* 2019;81(3):1699–1713. doi:10.1002/mrm.27525 · PMID 30320945 · PMC6347518
  - Analytic, sequential reconstruction (no NN / dictionary): B0 from multi-echo phase →
    T2, T2* via Eq. 1 → mixing factor X (Eqs. 10–11) → flip angle via Eq. 15 using two scans,
    with off-resonance nutation correction (Eq. 8) → T1, M0 via Eqs. 16–17. In-vivo flip
    ratio calibration β = 1.24 (Eq. 19). Implemented in `src/mpme/paper.py`.
  - Protocol (Table 1): TR 25/25 ms, α 15°/330°, 3 readout windows, pathway spacing 2 ms
    (1.86 ms phantom), BW 501 Hz/px, 1.2×1.0×1.2 mm³, 160×192×160, 11.5 min; scan 2 samples
    only the central 25%×25% of ky–kz. Siemens Trio 3T, 12-ch head coil, 5 volunteers.
  - Simulations (isochromats, 500 TRs, Lorentzian R2′) used for validation and protocol
    optimisation (Fig. 3). Discussion suggests regularisation or CNNs to control noise.
  - Local copy: `nihms-986644.pdf` (author manuscript, PMC).

## Related

- **Cheng CC, Preiswerk F, Madore B.** Multi-pathway multi-echo acquisition and neural
  contrast translation to generate a variety of quantitative and qualitative image contrasts.
  *Magn Reson Med.* 2020;83(6):2310–2321. doi:10.1002/mrm.28077 · PMID 31755588
  - Neural networks trained on in-vivo reference scans (SE, MPRAGE, FLAIR; 8 volunteers),
    not simulated signals. Predicts T1/T2 maps and synthetic MPRAGE, FLAIR, T1w, T2w, PDw.

- **Cheng CC, Mei CS, Duryea J, Chung HW, Chao TC, Panych LP, Madore B.** Dual-pathway
  multi-echo sequence for simultaneous frequency and T2 mapping.
  *J Magn Reson.* 2016;265:177–187. doi:10.1016/j.jmr.2016.01.019 · PMID 26923150
  - Two-pathway predecessor of MPME.
