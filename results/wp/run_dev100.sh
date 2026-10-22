#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-WP2 --out metrics_wp_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 36 --methods C2-OCT-1V-WP2 --out metrics_wp_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 50 --methods OSFD --out metrics_wp_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-WP2 --dets dets_wp_dev100.json --out generalization_wp_dev100.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 --dets dets_wp_dev100.json --out generalization_wp_dev100.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods OSFD --dets dets_wp_dev100.json --out generalization_wp_dev100.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_wp_dev100.json --tag wp_dev100 --exclude dev30 --pairs \
  C2-OCT-1V-WP2@dev300_B36_eps5_limit100:C2-OCT-1V C2-OCT-1V-WP2:C2-OCT-1V \
  C2-OCT-1V-WP2@dev300_B36_eps5_limit100:OSFD@dev300_B50_eps5_limit100 C2-OCT-1V:OSFD@dev300_B50_eps5_limit100
