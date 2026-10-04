#!/bin/bash
set -e
cd /workspace/transfer-attack-final && source .venv/bin/activate
M="OSFD GEO-C0 PIA PIA-noT PIA-noR PIA-noS PIA-noK PIA-noP"
P="PIA:PIA-noT PIA:PIA-noR PIA:PIA-noS PIA:PIA-noK PIA:PIA-noP PIA:OSFD"
for S in dev30 dev100; do
  python scripts/run_baselines.py --split dev300 --subset $S --methods $M --out metrics_piafac_$S.json
  python scripts/eval_generalization.py --run dev300_B50_eps5 --subset $S --models yolox_s dino_swin_l --methods $M --dets dets_piafac_$S.json --out generalization_piafac_$S.json
  python scripts/pair_screen.py --subset $S --dets dets_piafac_$S.json --tag piafac_$S --pairs $P
  echo "STAGE_DONE $S"
done
echo PIPELINE_DONE
