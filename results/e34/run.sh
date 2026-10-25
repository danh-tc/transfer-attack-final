#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; R=dev300_B41_eps5_limit100; D=dets_e34_dev30.json
M="C2-OCT-1V-RJ30 C2-OCT-1V-RJ60"
python scripts/run_baselines.py $A --budget 41 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 41 --stage eval --methods C2-OCT-1V $M --out metrics_e34_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V $M --dets $D --out generalization_e34_dev30.json
P=""; for m in $M; do P="$P $m:C2-OCT-1V"; done
python scripts/pair_screen.py --run $R --subset dev30 --dets $D --tag e34_dev30 --pairs $P
