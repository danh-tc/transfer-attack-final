#!/usr/bin/env bash
# E66 (progress_log 2026-10-31): ablation nguyên liệu bắc cầu trên surrogate ConvNeXt-T @ dev30 B36 — tách OCT-feat / cls / WP.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36 --surrogate convnext_t"; R=dev300_B36_eps5_limit100_sur-convnext_t
M="RD-CNX-FEAT RD-CNX-CLS"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e66_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e66_dev30.json --out generalization_e66_dev30.json
python scripts/panel_table.py --subset dev30 --ref RD-CNX --out results/e66/panel_dev30.json --cells \
  RD-CNX=C2-OCT-1V-WP2-RD-CNX@$R:dets_e63_dev30.json \
  NOWP=C2-OCT-1V-RD-CNX-NOWP@$R:dets_e63_dev30.json \
  FEAT=RD-CNX-FEAT@$R:dets_e66_dev30.json \
  CLS=RD-CNX-CLS@$R:dets_e66_dev30.json
