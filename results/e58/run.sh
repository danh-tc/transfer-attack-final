#!/usr/bin/env bash
# E58 (progress_log 2026-10-30): XÁC NHẬN RD-CWT(n=3) @ dev100 cùng máy — RD-CWTN3 vs RD vs OSFD (bản gọn: Swin-L + Controlled + YOLOX-L).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev100 --limit 100"; R=dev300_B36_eps5_limit100; R50=dev300_B50_eps5_limit100
python scripts/run_baselines.py $A --budget 36 --stage attack --methods C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD-CWTN3
python scripts/run_baselines.py $A --budget 50 --stage attack --methods OSFD
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD-CWTN3 --out metrics_e58_dev100.json
python scripts/run_baselines.py $A --budget 50 --stage eval --methods OSFD --out metrics_e58_dev100.json
python scripts/eval_generalization.py --run $R --subset dev100 --methods C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD-CWTN3 --models dino_swin_l yolox_l --dets dets_e58_dev100.json --out generalization_e58_dev100.json
python scripts/eval_generalization.py --run $R50 --subset dev100 --methods OSFD --models dino_swin_l yolox_l --dets dets_e58_dev100.json --out generalization_e58_dev100.json
python scripts/panel_table.py --subset dev100 --ref C2-OCT-1V-WP2-RD --gen-models dino_swin_l yolox_l --out results/e58/panel_dev100.json --cells \
  OSFD@$R50:dets_e58_dev100.json C2-OCT-1V-WP2-RD@$R:dets_e58_dev100.json C2-OCT-1V-WP2-RD-CWTN3@$R:dets_e58_dev100.json
