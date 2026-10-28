#!/usr/bin/env bash
# E42 (progress_log 2026-10-28): đặt cạnh AugTrans Table 2 — eval DETR-R50 + DINO-R50 trên PNG dev30 có sẵn (không attack).
# eval_generalization cần dets Controlled CÙNG run chứa đủ method -> dùng dets của E38 / E39 / E40.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
G="--subset dev30 --models detr_r50 dino_r50"
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --methods C2-OCT-1V C2-OCT-1V-WP2 $G --dets dets_e38_dev30.json --out generalization_e42a_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --methods C2-OCT-1V-WP2-RD $G --dets dets_e39_dev30.json --out generalization_e42b_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --methods OSFD $G --dets dets_e40_dev30.json --out generalization_e42_dev30.json
