#!/usr/bin/env bash
# E59 (progress_log 2026-10-30): quét seed attack trên cùng 3090 — RD B36 dev30, seeds {100,200,300} (+ seed0 từ E56).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --methods C2-OCT-1V-WP2-RD --models dino_swin_l --dets dets_e56_dev30.json --out generalization_e59_dev30.json
for S in 100 200 300; do
  A="--split dev300 --subset dev30 --limit 100 --budget 36 --seed-base $S"
  python scripts/run_baselines.py $A --stage attack --methods C2-OCT-1V-WP2-RD
  python scripts/run_baselines.py $A --stage eval --methods C2-OCT-1V-WP2-RD --out metrics_e59_dev30.json
  python scripts/eval_generalization.py --run dev300_B36_eps5_limit100_seed${S} --subset dev30 --methods C2-OCT-1V-WP2-RD --models dino_swin_l --dets dets_e59_dev30.json --out generalization_e59_dev30.json
done
python scripts/e59_seed_spread.py
