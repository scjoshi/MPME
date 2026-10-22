PY := PYTHONPATH=src python3
FAST := --jacobian implicit

.PHONY: test technote technote-results technote-pdf technote-reference flipnote flipnote-pdf limits

test:
	$(PY) -m pytest -q

# Fast path: regenerate results, figures, tables, and the PDF of the technical note.
technote: technote-results technote-pdf

technote-results:
	$(PY) scripts/crlb_analysis.py
	$(PY) scripts/single_scan_family.py
	$(PY) scripts/shared_b1_check.py
	$(PY) scripts/beyond_ideal_model.py
	$(PY) scripts/start_count.py
	$(PY) scripts/two_scan_design.py
	$(PY) scripts/phantom_experiment.py --snr 1000
	$(PY) scripts/phantom_experiment.py --snr 300
	PYTHONPATH=src:scripts python3 scripts/brainweb_experiment.py --case pv_clean
	PYTHONPATH=src:scripts python3 scripts/brainweb_experiment.py --case pv
	PYTHONPATH=src:scripts python3 scripts/brainweb_experiment.py --case crisp
	$(PY) scripts/technote_figures.py

technote-pdf:
	cd docs/technote && pdflatex -interaction=nonstopmode mpme_technote.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode mpme_technote.tex | grep -E "Output written|^!"

# Reference (forward-difference) Monte Carlo used for the efficiency and magnitude tables.
technote-reference:
	$(PY) scripts/estimator_efficiency.py
	$(PY) scripts/magnitude_comparison.py
	$(PY) scripts/make_mc_table.py
	$(PY) scripts/make_mag_table.py

# Flip-angle design note (docs/flipnote): design maps, shortlist, robustness, PDF.
flipnote:
	$(PY) scripts/flip_design.py
	$(PY) scripts/flip_design_report.py
	PYTHONPATH=src:scripts python3 scripts/flip_design_robustness.py
	$(MAKE) flipnote-pdf

flipnote-pdf:
	cd docs/flipnote && pdflatex -interaction=nonstopmode flip_design.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode flip_design.tex | grep -E "Output written|^!"

# B1/T1 limits companion note (docs/latex/mpme_b1_t1_limits.tex): figures, table, PDF.
limits:
	$(PY) scripts/limits_figures.py
	cd docs/latex && pdflatex -interaction=nonstopmode mpme_b1_t1_limits.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode mpme_b1_t1_limits.tex | grep -E "Output written|^!"
