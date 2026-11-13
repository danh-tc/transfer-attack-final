#!/usr/bin/env bash
# E87 smoke: 3 ảnh đầu x200 (KHÔNG đụng devB100), boot 20 — chỉ kiểm pipeline, KHÔNG đọc số.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python scripts/e87_c2_weight.py --list smoke3 --boot 20 --tag smoke
python scripts/e87_diag.py --list smoke3 --tag smoke
