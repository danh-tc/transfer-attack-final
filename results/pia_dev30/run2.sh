#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="OSFD PROBE-osfd+cls GEO-C0 PIA C2-PIA PIA-noCT"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_pia2_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l \
  --methods $M --dets dets_pia2_dev30.json --out generalization_pia2_dev30.json
python scripts/ablation_screen.py --subset dev30 --dets dets_pia2_dev30.json --tag pia2_dev30 \
  --cells PROBE-osfd+cls GEO-C0 PIA C2-PIA PIA-noCT
echo PIPELINE_DONE
