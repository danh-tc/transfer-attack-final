#!/usr/bin/env bash
# E63 (progress_log 2026-10-31): surrogate ConvNeXt-T — RD port (λ 0.2, WP cả Linear) + đối chứng @ dev30 B36, full panel, A4500.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36 --surrogate convnext_t"; R=dev300_B36_eps5_limit100_sur-convnext_t; R0=dev300_B36_eps5_limit100
M="C2-OCT-1V-WP2-RD-CNX C2-OCT-1V-RD-CNX-NOWP C2-OCT-1V-WP2-RD"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e63_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e63_dev30.json --out generalization_e63_dev30.json
python scripts/panel_table.py --subset dev30 --ref RD-R50 --out results/e63/panel_dev30.json --cells \
  RD-R50=C2-OCT-1V-WP2-RD@$R0:dets_e62_dev30.json \
  RD-CNX=C2-OCT-1V-WP2-RD-CNX@$R:dets_e63_dev30.json \
  CNX-NOWP=C2-OCT-1V-RD-CNX-NOWP@$R:dets_e63_dev30.json \
  CNX-NGUYEN=C2-OCT-1V-WP2-RD@$R:dets_e63_dev30.json
