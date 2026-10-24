#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e23_dev30.json
python scripts/run_baselines.py $A --budget 33 --stage attack --methods C2-OCT-1V-WP2-DEFCLS
python scripts/run_baselines.py $A --budget 34 --stage attack --methods C2-OCT-1V-WP2-DEFFEAT
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2 --out metrics_e23_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 --dets $D --out generalization_e23_dev30.json
for B in 33:C2-OCT-1V-WP2-DEFCLS 34:C2-OCT-1V-WP2-DEFFEAT; do
  python scripts/run_baselines.py $A --budget ${B%%:*} --stage eval --methods ${B#*:} --out metrics_e23_dev30.json
  python scripts/eval_generalization.py --run dev300_B${B%%:*}_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods ${B#*:} --dets $D --out generalization_e23_dev30.json
done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e23_dev30 \
  --pairs C2-OCT-1V-WP2-DEFCLS@dev300_B33_eps5_limit100:C2-OCT-1V-WP2 C2-OCT-1V-WP2-DEFFEAT@dev300_B34_eps5_limit100:C2-OCT-1V-WP2
