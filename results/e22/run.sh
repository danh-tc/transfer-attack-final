#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e22_dev30.json
M="C2-OCT-1V-WP2-RAND C2-OCT-1V-WP2-ASC C2-OCT-1V-WP2-L3 C2-OCT-1V-WP2-noL3 C2-OCT-1V-WPH C2-OCT-1V-WP2+H"
python scripts/run_baselines.py $A --budget 36 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 41 --stage attack --methods C2-OCT-1V
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2 $M --out metrics_e22_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 $M --dets $D --out generalization_e22_dev30.json
python scripts/run_baselines.py $A --budget 41 --stage eval --methods C2-OCT-1V --out metrics_e22_dev30.json
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets $D --out generalization_e22_dev30.json
P=""; for m in $M; do P="$P $m:C2-OCT-1V-WP2"; done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e22_dev30 \
  --pairs C2-OCT-1V-WP2:C2-OCT-1V@dev300_B41_eps5_limit100 $P C2-OCT-1V-WP2-L3:C2-OCT-1V@dev300_B41_eps5_limit100 C2-OCT-1V-WP2-noL3:C2-OCT-1V@dev300_B41_eps5_limit100
