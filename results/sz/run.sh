#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
M="C2-OCT-1V-WP2 C2-OCT-1V-WP2-SZ C2-OCT-1V-WP2-R C2-OCT-1V-WP2-SZR"
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --methods $M --out metrics_sz_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets dets_sz_dev30.json --out generalization_sz_dev30.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets dets_sz_dev30.json --tag sz_dev30 --pairs \
  C2-OCT-1V-WP2-SZ:C2-OCT-1V-WP2 C2-OCT-1V-WP2-R:C2-OCT-1V-WP2 C2-OCT-1V-WP2-SZR:C2-OCT-1V-WP2
