#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
R=dev300_B36_eps5_limit100; D=dets_e20pfa_dev30.json; S=dev300_B41_eps5_limit100_sur-swin_t
M="C2-OCT-1V-WP2 C2-OCT-1V-WP2-PFA50 C2-OCT-1V-WP2-PFA75 C2-OCT-1V-WP2-IS"
# PFA dev30 (WP2 + IS đã có PNG)
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --stage attack --methods C2-OCT-1V-WP2-PFA50 C2-OCT-1V-WP2-PFA75
# eval B36 (WP2, PFA, IS), B12 (IS snapshot), δ_S (sur Swin-T B41)
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 36 --stage eval --methods $M --out metrics_e20pfa_dev30.json
python scripts/eval_generalization.py --run $R --subset dev30 --models yolox_s dino_swin_l --methods $M --dets $D --out generalization_e20pfa_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 12 --stage eval --methods C2-OCT-1V-WP2-IS --out metrics_e20pfa_dev30.json
python scripts/eval_generalization.py --run dev300_B12_eps5_limit100 --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2-IS --dets $D --out generalization_e20pfa_dev30.json
python scripts/run_baselines.py --split dev300 --subset dev30 --limit 100 --budget 41 --surrogate swin_t --stage eval --methods C2-OCT-1V --out metrics_e20pfa_dev30.json
python scripts/eval_generalization.py --run $S --subset dev30 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets $D --out generalization_e20pfa_dev30.json
python scripts/pair_screen.py --run $R --subset dev30 --dets $D --tag pfa_dev30 --pairs C2-OCT-1V-WP2-PFA75:C2-OCT-1V-WP2 C2-OCT-1V-WP2-PFA50:C2-OCT-1V-WP2
python scripts/pair_screen.py --run $R --subset dev30 --dets $D --tag e20_dev30 --pairs C2-OCT-1V-WP2-IS:C2-OCT-1V-WP2 C2-OCT-1V-WP2-IS@dev300_B12_eps5_limit100:C2-OCT-1V-WP2 C2-OCT-1V@$S:C2-OCT-1V-WP2
