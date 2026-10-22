#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 100 --epsilon 8 --budget 41 --methods C2-OCT-1V --out metrics_eps8_dev10.json
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 100 --epsilon 8 --budget 36 --methods C2-OCT-1V-WP2 --out metrics_eps8_dev10.json
python scripts/run_baselines.py --split dev300 --subset dev10 --limit 100 --epsilon 8 --budget 50 --methods OSFD --out metrics_eps8_dev10.json
python scripts/eval_generalization.py --run dev300_B41_eps8_limit100 --subset dev10 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets dets_eps8_dev10.json --out generalization_eps8_dev10.json
python scripts/eval_generalization.py --run dev300_B36_eps8_limit100 --subset dev10 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 --dets dets_eps8_dev10.json --out generalization_eps8_dev10.json
python scripts/eval_generalization.py --run dev300_B50_eps8_limit100 --subset dev10 --models yolox_s dino_swin_l --methods OSFD --dets dets_eps8_dev10.json --out generalization_eps8_dev10.json
python scripts/pair_screen.py --run dev300_B41_eps8_limit100 --subset dev10 --dets dets_eps8_dev10.json --tag eps8_dev10 --pairs \
  C2-OCT-1V-WP2@dev300_B36_eps8_limit100:C2-OCT-1V C2-OCT-1V-WP2@dev300_B36_eps8_limit100:OSFD@dev300_B50_eps8_limit100 C2-OCT-1V:OSFD@dev300_B50_eps8_limit100
