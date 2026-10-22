#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --methods C2-OCT-1V-WP2 C2-OCT-1V-WP2-SZC --out metrics_szc_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 C2-OCT-1V-WP2-SZC --dets dets_szc_dev30.json --out generalization_szc_dev30.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets dets_szc_dev30.json --tag szc_dev30 --pairs C2-OCT-1V-WP2-SZC:C2-OCT-1V-WP2
