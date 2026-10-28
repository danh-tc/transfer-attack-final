#!/usr/bin/env bash
# E43 (progress_log 2026-10-28): nhóm B panel chốt — eval YOLOv3-D53 + YOLOX-L trên PNG dev30 có sẵn (không attack).
# eval_generalization cần dets Controlled CÙNG run chứa đủ method -> dùng dets của E38 / E39 / E40 (như E42).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
G="--subset dev30 --models yolov3_d53 yolox_l"
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --methods C2-OCT-1V C2-OCT-1V-WP2 $G --dets dets_e38_dev30.json --out generalization_e43a_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --methods C2-OCT-1V-WP2-RD $G --dets dets_e39_dev30.json --out generalization_e43b_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --methods OSFD $G --dets dets_e40_dev30.json --out generalization_e43_dev30.json
