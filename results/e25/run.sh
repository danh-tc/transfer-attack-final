#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e25_dev30.json; S=dev300_B41_eps5_limit100_sur-swin_t
python scripts/run_baselines.py $A --budget 36 --snapshots 9 --stage attack --methods C2-OCT-1V-WP2-ENS-IS --init-from $S
for B in 9 36; do
  python scripts/run_baselines.py $A --budget $B --stage eval --methods C2-OCT-1V-WP2-ENS-IS --out metrics_e25_dev30.json
  python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2-ENS-IS --dets $D --out generalization_e25_dev30.json
done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e25_dev30 \
  --pairs C2-OCT-1V-WP2-ENS-IS:C2-OCT-1V-WP2 C2-OCT-1V-WP2-ENS-IS@dev300_B9_eps5_limit100:C2-OCT-1V-WP2
