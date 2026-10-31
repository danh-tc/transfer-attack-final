#!/usr/bin/env bash
# E65 (progress_log 2026-10-31): baseline trên surrogate ConvNeXt-T @ dev30 B50, tách hiệu ứng surrogate khỏi hiệu ứng method; so RD-CNX B36 (E63).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 50 --surrogate convnext_t"; R=dev300_B50_eps5_limit100_sur-convnext_t
R36=dev300_B36_eps5_limit100_sur-convnext_t; R50=dev300_B50_eps5_limit100
M="OSFD MI-FGSM M-DI2-FGSM"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e65_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e65_dev30.json --out generalization_e65_dev30.json
python scripts/panel_table.py --subset dev30 --ref RD-CNX --out results/e65/panel_dev30.json --cells \
  RD-CNX=C2-OCT-1V-WP2-RD-CNX@$R36:dets_e63_dev30.json \
  OSFD-CNX=OSFD@$R:dets_e65_dev30.json \
  MI-CNX=MI-FGSM@$R:dets_e65_dev30.json \
  MDI-CNX=M-DI2-FGSM@$R:dets_e65_dev30.json \
  OSFD-R50=OSFD@$R50:dets_e62_dev30.json
