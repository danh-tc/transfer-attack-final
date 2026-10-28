#!/usr/bin/env bash
# E39 (progress_log 2026-10-28): WP2 rút gọn @ dev30 B36, so WP2 (PNG E38 cùng ngày).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e39_dev30.json
M="C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD4"
python scripts/run_baselines.py $A --budget 36 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2 $M --out metrics_e39_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 $M --dets $D --out generalization_e39_dev30.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e39_dev30 --pairs C2-OCT-1V-WP2-RD:C2-OCT-1V-WP2 C2-OCT-1V-WP2-RD4:C2-OCT-1V-WP2
