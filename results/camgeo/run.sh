#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="OSFD GEO-C0 PIA PIA-noT CAM-R15 CAM-R30 LENS CAM-R15+LENS"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_cam_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets dets_cam_dev30.json --out generalization_cam_dev30.json
python scripts/pair_screen.py --subset dev30 --dets dets_cam_dev30.json --tag cam_dev30 --pairs CAM-R15:PIA CAM-R30:PIA LENS:PIA CAM-R15+LENS:PIA CAM-R15+LENS:CAM-R15 CAM-R30:PIA-noT
echo PIPELINE_DONE
