#!/usr/bin/env bash
# progress_log 2026-10-20: (a) bão hòa B của C2-OCT-1V + (d) DM / DMA @ dev100, A4500. Resume được: chạy lại y hệt.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
C="--split dev300 --subset dev100 --limit 100 --out metrics_sat_dev100.json"
G="--subset dev100 --models yolox_s dino_swin_l --dets dets_sat_dev100.json --out generalization_sat_dev100.json"
python scripts/run_baselines.py $C --methods OSFD C2-OCT-1V-DM
python scripts/run_baselines.py $C --budget 100 --snapshots 20 41 57 65 --methods C2-OCT-1V
python scripts/run_baselines.py $C --budget 81 --methods C2-OCT-1V-DMA
for B in 20 41 57 65; do python scripts/run_baselines.py $C --budget $B --stage eval --methods C2-OCT-1V; done
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 $G --methods OSFD C2-OCT-1V-DM
for B in 20 41 57 65 100; do python scripts/eval_generalization.py --run dev300_B${B}_eps5_limit100 $G --methods C2-OCT-1V; done
python scripts/eval_generalization.py --run dev300_B81_eps5_limit100 $G --methods C2-OCT-1V-DMA
R=dev300_B
python scripts/pair_screen.py --run ${R}50_eps5_limit100 --subset dev100 --dets dets_sat_dev100.json --tag sat_dm_dev100 --exclude dev30 --pairs \
  C2-OCT-1V@${R}57_eps5_limit100:OSFD C2-OCT-1V@${R}41_eps5_limit100:OSFD C2-OCT-1V@${R}20_eps5_limit100:OSFD \
  C2-OCT-1V@${R}65_eps5_limit100:OSFD C2-OCT-1V@${R}100_eps5_limit100:OSFD \
  C2-OCT-1V@${R}100_eps5_limit100:C2-OCT-1V@${R}57_eps5_limit100 C2-OCT-1V@${R}57_eps5_limit100:C2-OCT-1V@${R}41_eps5_limit100 \
  C2-OCT-1V-DM:C2-OCT-1V@${R}57_eps5_limit100 C2-OCT-1V-DMA@${R}81_eps5_limit100:C2-OCT-1V@${R}57_eps5_limit100 \
  C2-OCT-1V-DMA@${R}81_eps5_limit100:OSFD
echo DONE
