#!/usr/bin/env bash
# progress_log 2026-10-20 (b): margin CW @ dev100 B57, A4500. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 57 --methods C2-OCT-1V C2-OCT-1V-MRG C2-OCT-1V-MRGF --out metrics_sat_dev100.json
python scripts/eval_generalization.py --run dev300_B57_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V C2-OCT-1V-MRG C2-OCT-1V-MRGF --dets dets_sat_dev100.json --out generalization_sat_dev100.json
python scripts/pair_screen.py --run dev300_B57_eps5_limit100 --subset dev100 --dets dets_sat_dev100.json --tag mrg_dev100 --exclude dev30 --pairs \
  C2-OCT-1V-MRG:C2-OCT-1V C2-OCT-1V-MRGF:C2-OCT-1V C2-OCT-1V-MRG:OSFD@dev300_B50_eps5_limit100 C2-OCT-1V-MRGF:OSFD@dev300_B50_eps5_limit100
echo ALLDONE
