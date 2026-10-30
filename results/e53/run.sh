#!/usr/bin/env bash
# E53 (progress_log 2026-10-30): giả thuyết "sai thứ tự độ dốc" — δ_S thoải trên R50, dốc trên DINO-Swin-L? @ dev30, đo tại ảnh sạch.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"
python scripts/run_baselines.py $A --budget 41 --surrogate swin_t --stage attack --methods C2-OCT-1V   # δ_S (resume: có PNG thì bỏ qua)
python scripts/run_baselines.py $A --budget 41 --surrogate swin_t --stage eval --methods C2-OCT-1V --out metrics_e53_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100_sur-swin_t --subset dev30 --methods C2-OCT-1V --models dino_swin_l \
  --dets dets_e53_dev30.json --out generalization_e53_dev30.json
python scripts/e53_steepness.py
