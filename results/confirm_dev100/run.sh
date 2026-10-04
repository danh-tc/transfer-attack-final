#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="OSFD PROBE-osfd+cls GEO-C0 PIA C2-PIA"
python scripts/run_baselines.py --split dev300 --subset dev100 --methods $M --out metrics_pia_dev100.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev100 --models yolox_s dino_swin_l --methods $M --dets dets_pia_dev100.json --out generalization_pia_dev100.json
python scripts/pair_screen.py --subset dev100 --dets dets_pia_dev100.json --tag confirm_dev100 --pairs C2-PIA:PROBE-osfd+cls PIA:GEO-C0 C2-PIA:OSFD --exclude dev30
echo PIPELINE_DONE
