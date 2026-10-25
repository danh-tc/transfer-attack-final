#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --surrogate dino_r50 --methods C2-OCT-1V-DETR --out metrics_c2_1v_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100_sur-dino_r50 --subset dev100 --models yolox_s dino_swin_l dino_r50 --methods C2-OCT-1V-DETR --dets dets_c2_1v_dev100.json --out generalization_surdino_dev100.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_c2_1v_dev100.json --tag sur_dino_dev100 --exclude dev30 \
  --pairs C2-OCT-1V-DETR@dev300_B41_eps5_limit100_sur-dino_r50:C2-OCT-1V
