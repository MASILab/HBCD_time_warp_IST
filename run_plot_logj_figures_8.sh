#!/usr/bin/env bash
# De-mean saved warp profiles and build dual-set (non-demeaned / image-demeaned)
# Figures 2-4 plus shared Figures 3/5 for the manual 6-subject HBCD_diff_morph run.
set -euo pipefail

PROJ="${PROJ:-/nfs/masi/taylogc1/HBCD_diff_morph}"
TOOLS="${TOOLS:-/home-local/taylogc1/hbcd_dm_fig8}"
PYTHON="${PYTHON:-${TOOLS}/.venv/bin/python}"

SUBJECTS=(sub-2368305128 sub-3658433742 sub-3980628532 sub-7494979436 sub-7857079557 sub-9511225120)

cd "${PROJ}"
"${PYTHON}" plot_logj_figures_8.py \
  --t2w-output-root derivatives_2/T2w_diff_morph \
  --tract-output-root derivatives_2/tract2tract_diff_morph \
  --preproc-t2w-root preproc_T2w \
  --demeaned-out-root derivatives_2_de_meaned \
  --out-dir figures_8 \
  --subjects "${SUBJECTS[@]}" "$@"
