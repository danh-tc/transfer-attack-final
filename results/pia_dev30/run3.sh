#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="OSFD PROBE-osfd+cls C2-PIA C2-MORGD C2-SMOOTH"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_morgd_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l \
  --methods $M --dets dets_morgd_dev30.json --out generalization_morgd_dev30.json
python scripts/ablation_screen.py --subset dev30 --dets dets_morgd_dev30.json --tag morgd_dev30 \
  --cells PROBE-osfd+cls C2-PIA C2-MORGD C2-SMOOTH
echo PIPELINE_DONE
