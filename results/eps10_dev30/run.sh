#!/usr/bin/env bash
# Quick test ε=10 @ dev30 (progress_log 2026-10-25). Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --epsilon 10"; O=metrics_eps10_dev30.json; D=dets_eps10_dev30.json
R50=dev300_B50_eps10_limit100; R36=dev300_B36_eps10_limit100
python scripts/run_baselines.py $A --budget 50 --methods OSFD M-DI2-FGSM --out $O
python scripts/run_baselines.py $A --budget 36 --methods C2-OCT-1V-WP2 --out $O
python scripts/eval_generalization.py --run $R50 --subset dev30 --models yolox_s dino_swin_l --methods OSFD M-DI2-FGSM --dets $D --out generalization_eps10_dev30.json
python scripts/eval_generalization.py --run $R36 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 --dets $D --out generalization_eps10_dev30.json
python scripts/pair_screen.py --run $R50 --subset dev30 --dets $D --tag eps10_dev30 \
  --pairs C2-OCT-1V-WP2@$R36:OSFD M-DI2-FGSM:OSFD C2-OCT-1V-WP2@$R36:M-DI2-FGSM
echo "[run.sh] XONG $(date '+%F %T')"
