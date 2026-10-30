#!/usr/bin/env bash
# E61 (progress_log 2026-10-30): đẩy liều flush gradient p ∈ {0.8, 0.9} — RD-GQ tới mốc V100 37.7? R50 vỡ ở đâu? @ dev30 B36.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="C2-OCT-1V-WP2-RD-GQ80 C2-OCT-1V-WP2-RD-GQ90"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e61_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --models dino_swin_l yolox_l --dets dets_e61_dev30.json --out generalization_e61_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --gen-models dino_swin_l yolox_l --out results/e60/panel_ext_dev30.json --cells \
  C2-OCT-1V-WP2-RD@$R:dets_e60_dev30.json C2-OCT-1V-WP2-RD-GQ70@$R:dets_e60_dev30.json \
  C2-OCT-1V-WP2-RD-GQ80@$R:dets_e61_dev30.json C2-OCT-1V-WP2-RD-GQ90@$R:dets_e61_dev30.json
