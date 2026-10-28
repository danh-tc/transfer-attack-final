#!/usr/bin/env bash
# E45 (progress_log 2026-10-28): RD-V (loss cls vanishing) @ dev30 B36, so RD + OSFD trên panel chốt (AP drop + ASR).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; R=dev300_B36_eps5_limit100
python scripts/run_baselines.py $A --budget 36 --stage attack --methods C2-OCT-1V-WP2-RDV
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RDV --out metrics_e45_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RDV --dets dets_e45_dev30.json --out generalization_e45_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e45/panel_dev30.json --cells \
  OSFD@dev300_B50_eps5_limit100:dets_e40_dev30.json \
  C2-OCT-1V-WP2-RD@$R:dets_e45_dev30.json C2-OCT-1V-WP2-RDV@$R:dets_e45_dev30.json
