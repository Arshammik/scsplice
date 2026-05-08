#!/usr/bin/env bash
# Orchestrates the full end-to-end validation on samples A01 + B01:
#   1. R pipeline    -> validation/e2e/data/r_pipeline.h5
#   2. Python pipe   -> validation/e2e/data/py_pipeline.h5ad
#   3. Compare       -> structural + bit-exact equality on M1, M2, eventdata
#
# Run from the repo root on c170 (or any node with R splikit + the splikit-py
# editable install).

set -euo pipefail

REPO_ROOT="/home/arsham79/projects/rrg-hsn/arsham79/splikitpy"
VENV_PY="/home/arsham79/projects/rrg-hsn/arsham79/multigedipy_pkg/.venv/bin/python"

cd "$REPO_ROOT"

echo "[run_all] Step 1/3: R pipeline (make_junction_ab + make_m1 + make_m2)..."
module load r/4.4.0
Rscript validation/e2e/run_r_pipeline.R

echo
echo "[run_all] Step 2/3: Python pipeline (read_starsolo + make_m2)..."
module load eigen/3.4.0
"$VENV_PY" validation/e2e/run_py_pipeline.py

echo
echo "[run_all] Step 3/3: Comparing R vs Python outputs..."
"$VENV_PY" validation/e2e/compare_outputs.py

echo
echo "[run_all] DONE."
