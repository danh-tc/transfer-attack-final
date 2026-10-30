#!/usr/bin/env bash
# E46 (progress_log 2026-10-30): RD-CV (6·CE RoI + 0.3·vanishing) @ dev30 B36, so RD / RD-V / OSFD trên panel chốt (AP drop + ASR).
# Máy mới không có PNG HF -> bước attack sinh lại RD / RD-V (B36) và OSFD (B50); có sẵn PNG thì tự bỏ qua.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; R=dev300_B36_eps5_limit100; R50=dev300_B50_eps5_limit100
M="C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RDV C2-OCT-1V-WP2-RDCV"
python scripts/run_baselines.py $A --budget 36 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 50 --stage attack --methods OSFD
python scripts/run_baselines.py $A --budget 36 --stage eval --methods $M --out metrics_e46_dev30.json
python scripts/run_baselines.py $A --budget 50 --stage eval --methods OSFD --out metrics_e46_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --dets dets_e46_dev30.json --out generalization_e46_dev30.json
python scripts/eval_generalization.py --run $R50 --subset dev30 --methods OSFD --dets dets_e46_dev30.json --out generalization_e46_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --out results/e46/panel_dev30.json --cells \
  OSFD@$R50:dets_e46_dev30.json \
  C2-OCT-1V-WP2-RD@$R:dets_e46_dev30.json C2-OCT-1V-WP2-RDV@$R:dets_e46_dev30.json C2-OCT-1V-WP2-RDCV@$R:dets_e46_dev30.json
