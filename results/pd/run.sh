#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-PD16 C2-OCT-1V-PD32 --out metrics_pd_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-PD16 C2-OCT-1V-PD32 --dets dets_pd_dev30.json --out generalization_pd_dev30.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev30 --dets dets_pd_dev30.json --tag pd_dev30 --pairs C2-OCT-1V-PD16:C2-OCT-1V C2-OCT-1V-PD32:C2-OCT-1V
