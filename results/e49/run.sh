#!/usr/bin/env bash
# E49 (progress_log 2026-10-30): gộp 3 quỹ đạo RD (E46 / E47 / E48) @ dev30 — phép dò, chỉ eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
M="RD-VOTE3 RD-MEAN3"
python scripts/e49_traj_vote.py
python scripts/run_baselines.py $A --stage eval --methods $M C2-OCT-1V-WP2-RD --out metrics_e49_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M C2-OCT-1V-WP2-RD --models dino_swin_l yolox_l \
  --dets dets_e49_dev30.json --out generalization_e49_dev30.json
python scripts/eval_generalization.py --run ${R}_fp32 --subset dev30 --methods C2-OCT-1V-WP2-RD --models dino_swin_l yolox_l \
  --dets dets_e47_dev30.json --out generalization_e49_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods C2-OCT-1V-WP2-RD-NOAMP --models yolox_l \
  --dets dets_e48_dev30.json --out generalization_e49b_dev30.json
python scripts/e49_traj_vote.py corr
