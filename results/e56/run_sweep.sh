#!/usr/bin/env bash
# E57 (progress_log 2026-10-30): quét liều CWT n ∈ {4,6} + biến thể xoay (n=3, ±15°) trên nền RD @ dev30 B36.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="C2-OCT-1V-WP2-RD-CWTN4 C2-OCT-1V-WP2-RD-CWTN6 C2-OCT-1V-WP2-RD-CWTN3R"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e57_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e57_dev30.json --out generalization_e57_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e56/panel_sweep_dev30.json --cells \
  C2-OCT-1V-WP2-RD@$R:dets_e56_dev30.json C2-OCT-1V-WP2-RD-CWTN3@$R:dets_e56_dev30.json \
  C2-OCT-1V-WP2-RD-CWTN4@$R:dets_e57_dev30.json C2-OCT-1V-WP2-RD-CWTN6@$R:dets_e57_dev30.json C2-OCT-1V-WP2-RD-CWTN3R@$R:dets_e57_dev30.json
