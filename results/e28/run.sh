#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e28_dev30.json
M="C2-OCT-1V-WSPK1 C2-OCT-1V-WSPK2"
python scripts/run_baselines.py $A --budget 36 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 36 --stage eval --methods C2-OCT-1V-WP2 $M --out metrics_e28_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 $M --dets $D --out generalization_e28_dev30.json
P=""; for m in $M; do P="$P $m:C2-OCT-1V-WP2"; done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e28_dev30 --pairs $P
