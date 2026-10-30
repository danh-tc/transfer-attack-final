#!/usr/bin/env bash
# E55 (progress_log 2026-10-30): RD-SCR — regularizer "bền xáo ô" (suy từ E53 H_steep + E54 scramble) @ dev30 B36.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="C2-OCT-1V-WP2-RD-SCR C2-OCT-1V-WP2-RD-SCR05 C2-OCT-1V-WP2-RD-SCR2"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M C2-OCT-1V-WP2-RD --out metrics_e55_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M C2-OCT-1V-WP2-RD --dets dets_e55_dev30.json --out generalization_e55_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e55/panel_dev30.json --cells \
  OSFD@dev300_B50_eps5_limit100:dets_e46_dev30.json C2-OCT-1V-WP2-RD@$R:dets_e55_dev30.json \
  C2-OCT-1V-WP2-RD-SCR@$R:dets_e55_dev30.json C2-OCT-1V-WP2-RD-SCR05@$R:dets_e55_dev30.json C2-OCT-1V-WP2-RD-SCR2@$R:dets_e55_dev30.json
