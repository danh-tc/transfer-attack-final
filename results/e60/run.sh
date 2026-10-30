#!/usr/bin/env bash
# E60 (progress_log 2026-10-30): RD-GQ — flush gradient fp16-kiểu-V100 (giả lập cơ chế V100) @ dev30 B36. Tái tạo +5 Swin-L?
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36"; R=dev300_B36_eps5_limit100
rm -rf artifacts/runs/$R/adv/C2-OCT-1V-WP2-RD   # xóa δ code-cũ/V100 đang nằm đây -> sinh lại RD code hiện tại
M="C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD-GQ30 C2-OCT-1V-WP2-RD-GQ50 C2-OCT-1V-WP2-RD-GQ70"
python scripts/run_baselines.py $A --stage attack --methods $M
rm -f artifacts/runs/$R/gen_dets/dino_swin_l.json artifacts/runs/$R/gen_dets/yolox_l.json
python scripts/run_baselines.py $A --stage eval --methods $M --out metrics_e60_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --methods $M --models dino_swin_l yolox_l --dets dets_e60_dev30.json --out generalization_e60_dev30.json
python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD --gen-models dino_swin_l yolox_l --out results/e60/panel_dev30.json --cells \
  C2-OCT-1V-WP2-RD@$R:dets_e60_dev30.json C2-OCT-1V-WP2-RD-GQ30@$R:dets_e60_dev30.json C2-OCT-1V-WP2-RD-GQ50@$R:dets_e60_dev30.json C2-OCT-1V-WP2-RD-GQ70@$R:dets_e60_dev30.json
