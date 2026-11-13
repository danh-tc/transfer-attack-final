#!/usr/bin/env bash
# E87 @ devB100 (khóa trong progress_log 2026-11-13 TRƯỚC khi chạy). 2 worker sinh -> eval 9 model -> chẩn đoán.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python scripts/e87_c2_weight.py --gen-only --cells CNX-A1,CNX-A05 > results/e87/gen_w1.log 2>&1 &
P1=$!
python scripts/e87_c2_weight.py --gen-only --cells CNX-A0,R50-A1,R50-A0 > results/e87/gen_w2.log 2>&1 &
P2=$!
wait $P1 $P2
python scripts/e87_c2_weight.py > results/e87/eval.log 2>&1
python scripts/e87_diag.py > results/e87/diag.log 2>&1
echo "[e87] XONG $(TZ=Asia/Ho_Chi_Minh date '+%F %H:%M GMT+7')"
