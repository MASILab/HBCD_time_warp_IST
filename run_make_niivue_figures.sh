#!/usr/bin/env bash
# Niivue 3D interactive HTML + static grids + orbit movies for both metric sets
# of the manual 6-subject HBCD_diff_morph run. Run run_plot_logj_figures_8.sh
# first (it creates derivatives_2_de_meaned/).
set -euo pipefail

PROJ="${PROJ:-/nfs/masi/taylogc1/HBCD_diff_morph}"
TOOLS="${TOOLS:-/home-local/taylogc1/hbcd_dm_fig8}"
PYTHON="${PYTHON:-${TOOLS}/.venv/bin/python}"

SUBJECTS=(sub-2368305128 sub-3658433742 sub-3980628532 sub-7494979436 sub-7857079557 sub-9511225120)

cd "${PROJ}"
"${PYTHON}" make_niivue_figures.py \
  --t2w-deriv-root derivatives_2/T2w_diff_morph \
  --demeaned-t2w-root derivatives_2_de_meaned/T2w_diff_morph \
  --preproc-t2w-root preproc_T2w \
  --out-dir niivue_fig \
  --geom-subject sub-7857079557 \
  --niivue-js "$(dirname "$0")/niivue.umd.js" \
  --subjects "${SUBJECTS[@]}" "$@"
