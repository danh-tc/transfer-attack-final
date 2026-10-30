#!/usr/bin/env bash
# E52 (progress_log 2026-10-30): kênh ngữ cảnh từ R50 — δ chỉ ở OUT / RING, loss RD gốc (-F) vs loss chỉ object (-O) @ dev30 B36.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="RD-OUTF RD-OUTO RD-RINGF RD-RINGO"
python scripts/run_baselines.py $A --stage attack --methods $M
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e52_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e52_dev30.json --out generalization_e52_dev30.json
python scripts/panel_table.py --subset dev30 --ref RD-OUT --out results/e52/panel_dev30.json --cells \
  RD-OUT@$R:dets_e50_dev30.json RD-RING@$R:dets_e50_dev30.json \
  RD-OUTF@$R:dets_e52_dev30.json RD-OUTO@$R:dets_e52_dev30.json RD-RINGF@$R:dets_e52_dev30.json RD-RINGO@$R:dets_e52_dev30.json
