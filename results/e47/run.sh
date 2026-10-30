#!/usr/bin/env bash
# E47 (progress_log 2026-10-30): TF32 có phải nguồn lệch Swin-L giữa V100 (E45) và 3090 (E46)? RD B36 + OSFD B50 @ dev30, FP32 thuần.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --no-tf32"
python scripts/run_baselines.py $A --budget 36 --methods C2-OCT-1V-WP2-RD --out metrics_e47_dev30.json
python scripts/run_baselines.py $A --budget 50 --methods OSFD --out metrics_e47_dev30.json
for R in dev300_B36_eps5_limit100_fp32:C2-OCT-1V-WP2-RD dev300_B50_eps5_limit100_fp32:OSFD; do
  python scripts/eval_generalization.py --run ${R%:*} --subset dev30 --methods ${R#*:} --models dino_swin_l \
    --dets dets_e47_dev30.json --out generalization_e47_dev30.json
done
