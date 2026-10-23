#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
M="C2-OCT-1V-WP2 C2-OCT-1V-WPA"
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 36 --methods $M --out metrics_wpa_dev100.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods $M --dets dets_wpa_dev100.json --out generalization_wpa_dev100.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev100 --dets dets_wpa_dev100.json --tag wpa_dev100 --exclude dev30 --pairs C2-OCT-1V-WPA:C2-OCT-1V-WP2
# Đo thời gian cùng lần (chỉ attack, run riêng)
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 12 --budget 36 --stage attack --methods $M --out metrics_wpa_timing.json
