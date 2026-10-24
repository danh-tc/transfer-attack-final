#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
S=dev300_B41_eps5_limit100_sur-swin_t; D=dets_otsv_dev30.json; O=C2-OCT-1V-WP2-OTSV50-IS; W=C2-OCT-1V-WP2-IS
A="--split dev300 --subset dev30 --limit 100"
# cổng oracle-retention: OTSV50-IS B36 (snapshot B1/B4/B12); đối chứng WP2-IS bổ sung B1/B4
python scripts/run_baselines.py $A --budget 36 --snapshots 1 4 12 --stage attack --methods $O --init-from $S
python scripts/run_baselines.py $A --budget 4 --snapshots 1 --stage attack --methods $W --init-from $S
for B in 36 12 4 1; do
  python scripts/run_baselines.py $A --budget $B --stage eval --methods $O $W --out metrics_otsv_dev30.json
  python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $O $W --dets $D --out generalization_otsv_dev30.json
done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag otsv_gate_dev30 \
  --pairs $O:$W $O@dev300_B12_eps5_limit100:$W@dev300_B12_eps5_limit100 $O@dev300_B4_eps5_limit100:$W@dev300_B4_eps5_limit100 $O@dev300_B1_eps5_limit100:$W@dev300_B1_eps5_limit100
python scripts/otsv_delta_move.py
