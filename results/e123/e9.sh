#!/usr/bin/env bash
# E9 (progress_log 2026-10-06): C2 + C2-PIA @ dev30, tách theo kích thước. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONUTF8=1
M="OSFD PIA PROBE-osfd+cls C2-PIA"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_e9_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l \
  --methods $M --dets dets_e9_dev30.json --out generalization_e9_dev30.json
python scripts/e4_target_side.py --run dev300_B50_eps5 --subset dev30 --dets dets_e9_dev30.json --methods $M \
  --base OSFD --tag _e9_vsOSFD --targets dino_swin_l swin_t convnext_t
python scripts/e4_target_side.py --run dev300_B50_eps5 --subset dev30 --dets dets_e9_dev30.json --methods PIA C2-PIA \
  --base PIA --tag _e9_vsPIA --targets dino_swin_l swin_t convnext_t
echo PIPELINE_DONE
