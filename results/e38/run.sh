#!/usr/bin/env bash
# E38 (progress_log 2026-10-28): quét liều ρ của WP @ dev30 B36 + đo Sm / Sim.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100"; D=dets_e38_dev30.json
M="C2-OCT-1V C2-OCT-1V-WP1 C2-OCT-1V-WP2 C2-OCT-1V-WP5 C2-OCT-1V-WP10"
python scripts/run_baselines.py $A --budget 36 --stage attack --methods $M
python scripts/run_baselines.py $A --budget 36 --stage eval --methods $M --out metrics_e38_dev30.json
python scripts/eval_generalization.py --run dev300_B36_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods $M --dets $D --out generalization_e38_dev30.json
P=""; for m in $M; do [ "$m" != C2-OCT-1V-WP2 ] && P="$P $m:C2-OCT-1V-WP2"; done
python scripts/pair_screen.py --run dev300_B36_eps5_limit100 --subset dev30 --dets $D --tag e38_dev30 --pairs $P
python scripts/e38_rho_tradeoff.py
