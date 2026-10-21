#!/usr/bin/env bash
# Chẩn đoán đổi surrogate Swin-T (progress_log 2026-10-21). Resume được — chạy lại đúng lệnh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V --out metrics_sur_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --surrogate swin_t --methods C2-OCT-1V --out metrics_sur_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets dets_sur_dev100.json --out generalization_sur_dev100.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100_sur-swin_t --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets dets_sur_dev100.json --out generalization_sur_dev100.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev100 --dets dets_sur_dev100.json --tag sur_swint_dev100 --exclude dev30 \
  --pairs C2-OCT-1V@dev300_B41_eps5_limit100_sur-swin_t:C2-OCT-1V
