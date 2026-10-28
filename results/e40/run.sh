#!/usr/bin/env bash
# E40 (progress_log 2026-10-28): quét B của WP2-RD @ dev30 (36 có sẵn E39) + OSFD B50 đo thời gian cùng lần.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e40_dev30.json; R=C2-OCT-1V-WP2-RD
python scripts/run_baselines.py $A --budget 50 --stage attack --methods OSFD $R
for B in 75 100; do python scripts/run_baselines.py $A --budget $B --stage attack --methods $R; done
python scripts/run_baselines.py $A --budget 50 --stage eval --methods OSFD $R --out metrics_e40_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods OSFD $R --dets $D --out generalization_e40_dev30.json
for B in 36 75 100; do
  python scripts/run_baselines.py $A --budget $B --stage eval --methods $R --out metrics_e40_dev30.json
  python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $R --dets $D --out generalization_e40_dev30.json
done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e40_dev30 --pairs \
  $R@dev300_B50_eps5_limit100:$R $R@dev300_B75_eps5_limit100:$R@dev300_B50_eps5_limit100 \
  $R@dev300_B100_eps5_limit100:$R@dev300_B75_eps5_limit100 $R:OSFD@dev300_B50_eps5_limit100
