#!/usr/bin/env bash
# E50 (progress_log 2026-10-30): δ_RD che theo GT box (IN / OUT / RING) @ dev30, hậu xử lý PNG E46, chỉ eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="C2-OCT-1V-WP2-RD RD-IN RD-OUT RD-RING"
python scripts/e50_bg_mask.py
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e50_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e50_dev30.json --out generalization_e50_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e50/panel_dev30.json --cells \
  C2-OCT-1V-WP2-RD@$R:dets_e50_dev30.json RD-IN@$R:dets_e50_dev30.json RD-OUT@$R:dets_e50_dev30.json RD-RING@$R:dets_e50_dev30.json
python scripts/e50_bg_mask.py fp
