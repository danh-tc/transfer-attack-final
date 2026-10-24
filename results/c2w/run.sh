#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
M="C2-OCT-1V-WP2 C2-OCT-1V-WP2-C2W03 C2-OCT-1V-WP2-C2W1"
# sức khỏe white-box (ảnh 1–4 dev300, chỉ R50)
python scripts/run_baselines.py --split dev300 --limit 5 --budget 36 --stage attack --methods $M
python scripts/tx_calib.py --run dev300_B36_eps5_limit5 --methods $M
# dev30
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --methods $M --out metrics_c2w_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets dets_c2w_dev30.json --out generalization_c2w_dev30.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets dets_c2w_dev30.json --tag c2w_dev30 --pairs C2-OCT-1V-WP2-C2W03:C2-OCT-1V-WP2 C2-OCT-1V-WP2-C2W1:C2-OCT-1V-WP2
