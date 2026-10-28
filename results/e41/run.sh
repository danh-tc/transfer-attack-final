#!/usr/bin/env bash
# E41 (progress_log 2026-10-28): RD-SAT / RAND5 hậu xử lý từ PNG RD B36 (E39), chỉ eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e41_dev30.json
M="C2-OCT-1V-WP2-RD C2-OCT-1V-WP2-RD-SAT RAND5"
python scripts/e41_saturate.py
python scripts/run_baselines.py $A --budget 36 --stage eval --methods $M --out metrics_e41_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets $D --out generalization_e41_dev30.json
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e41_dev30 --pairs C2-OCT-1V-WP2-RD-SAT:C2-OCT-1V-WP2-RD RAND5:C2-OCT-1V-WP2-RD
