#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --methods C2-OCT-1V C2-OCT-1V-GLB7 C2-OCT-1V-GLBG --out metrics_glb_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-GLB7 C2-OCT-1V-GLBG --dets dets_glb_dev30.json --out generalization_glb_dev30.json
python scripts/pair_screen.py --run dev300_B41_eps5_limit100 --subset dev30 --dets dets_glb_dev30.json --tag glb_dev30 --pairs C2-OCT-1V-GLB7:C2-OCT-1V C2-OCT-1V-GLBG:C2-OCT-1V
