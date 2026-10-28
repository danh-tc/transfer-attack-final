#!/usr/bin/env bash
# E42 (progress_log 2026-10-28): đặt cạnh AugTrans Table 2 — eval DETR-R50 + DINO-R50 trên PNG dev30 có sẵn (không attack).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
G="--subset dev30 --models detr_r50 dino_r50 --dets dets_e42_dev30.json --out generalization_e42_dev30.json"
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --methods C2-OCT-1V C2-OCT-1V-WP2 C2-OCT-1V-WP2-RD $G
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --methods OSFD $G
