#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="PIA PIA-noT CAM-R30 NOISE FLIP POOL"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_pool_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets dets_pool_dev30.json --out generalization_pool_dev30.json
python scripts/pair_screen.py --subset dev30 --dets dets_pool_dev30.json --tag pool_dev30 --pairs FLIP:NOISE PIA-noT:NOISE POOL:PIA POOL:PIA-noT POOL:CAM-R30
echo PIPELINE_DONE
