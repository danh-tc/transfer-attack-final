#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
X="C2-OCT-1V-WP2-XBG C2-OCT-1V-WP2-XBGS"
R30=dev300_B30_eps5_limit100; R36=dev300_B36_eps5_limit100
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --methods C2-OCT-1V-WP2 $X --out metrics_xbg_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 30 --methods $X --out metrics_xbg_dev30.json
python scripts/eval_generalization.py --run $R36 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 $X --dets dets_xbg_dev30.json --out generalization_xbg_dev30.json
python scripts/eval_generalization.py --run $R30 --subset dev30 --models yolox_s dino_swin_l --methods $X --dets dets_xbg_dev30.json --out generalization_xbg_dev30.json
python scripts/pair_screen.py --run $R30 --subset dev30 --dets dets_xbg_dev30.json --tag xbg_dev30 --pairs \
  C2-OCT-1V-WP2-XBG:C2-OCT-1V-WP2@$R36 C2-OCT-1V-WP2-XBGS:C2-OCT-1V-WP2@$R36 \
  C2-OCT-1V-WP2-XBG@$R36:C2-OCT-1V-WP2@$R36 C2-OCT-1V-WP2-XBGS@$R36:C2-OCT-1V-WP2@$R36
