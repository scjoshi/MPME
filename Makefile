PY := PYTHONPATH=src python3
FAST := --jacobian implicit

.PHONY: test technote technote-results technote-pdf technote-reference

test:
	$(PY) -m pytest -q

# Fast path: regenerate results, figures, tables, and the PDF of the technical note.
technote: technote-results technote-pdf

technote-results:
	$(PY) scripts/crlb_analysis.py
	$(PY) scripts/start_count.py
	$(PY) scripts/phantom_experiment.py
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
