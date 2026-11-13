#!/usr/bin/env bash
# E88 baseline fidelity nửa đầu (khóa trong progress_log TRƯỚC khi chạy). Dùng: run.sh [dev100|smoke3] [boot] [tag]
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=${1:-dev100}; BOOT=${2:-1000}; TAG=${3:-}
A=(--list "$L"); [ -n "$TAG" ] && A+=(--tag "$TAG")
P=results/e88/${L}${TAG:+_$TAG}
python scripts/e88_ens_baselines.py "${A[@]}" --gen-only --cells OSFD-PUB,OSFD-REF,ENS-WP,ENS-OSFD-AVG > ${P}_gen_w1.log 2>&1 &
P1=$!
python scripts/e88_ens_baselines.py "${A[@]}" --gen-only --cells RD-CNX,ENS-OSFD-GN,ENS-RD-AVG,ENS-RD-GN > ${P}_gen_w2.log 2>&1 &
P2=$!
wait $P1; wait $P2
python scripts/e88_ens_baselines.py "${A[@]}" --boot "$BOOT" > ${P}_eval.log 2>&1
echo "[e88] XONG $L $(TZ=Asia/Ho_Chi_Minh date '+%F %H:%M GMT+7')"
