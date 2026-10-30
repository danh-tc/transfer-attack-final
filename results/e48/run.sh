#!/usr/bin/env bash
# E48 (progress_log 2026-10-30): lệch DINO-Swin-L V100 vs 3090 đến từ WP và/hoặc AMP fp16? @ dev30 B36 trên 3090.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="C2-OCT-1V C2-OCT-1V-WP2 C2-OCT-1V-WP2-RD-NOAMP"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M C2-OCT-1V-WP2-RD --out metrics_e48_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M C2-OCT-1V-WP2-RD --models dino_swin_l \
  --dets dets_e48_dev30.json --out generalization_e48_dev30.json
