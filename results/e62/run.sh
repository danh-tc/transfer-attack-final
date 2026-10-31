#!/usr/bin/env bash
# E62 (progress_log 2026-10-31): kiểm máy RTX A4500 — RD B36 + OSFD B50 @ dev30, full panel, đối chiếu mốc 3090 (E46/E60).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; R=dev300_B36_eps5_limit100; R50=dev300_B50_eps5_limit100
python scripts/run_baselines.py $A --budget 36 --stage attack --methods C2-OCT-1V-WP2-RD
python scripts/run_baselines.py $A --budget 50 --stage attack --methods OSFD
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2-RD --out metrics_e62_dev30.json
python scripts/run_baselines.py $A --budget 50 --stage eval --methods OSFD --out metrics_e62_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods C2-OCT-1V-WP2-RD --dets dets_e62_dev30.json --out generalization_e62_dev30.json
python scripts/eval_generalization.py --run $R50 --subset dev30 --methods OSFD --dets dets_e62_dev30.json --out generalization_e62_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e62/panel_dev30.json --cells \
  OSFD@$R50:dets_e62_dev30.json C2-OCT-1V-WP2-RD@$R:dets_e62_dev30.json
