#!/usr/bin/env bash
# Quét liều phép biến đổi cơ bản @ dev30 (progress_log 2026-10-05). Resume được: chạy lại đúng file này.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONUTF8=1
M="OSFD NOISE FLIP PIA PIA-noT CAM-R15 CAM-R30 ROT7 ROT15 ROT30 SHR15 SHR30 ANI10 ANI25 SCL25"
python scripts/run_baselines.py --split dev300 --subset dev30 --methods $M --out metrics_elem_dev30.json
python scripts/eval_generalization.py --run dev300_B50_eps5 --subset dev30 --models yolox_s dino_swin_l \
  --methods $M --dets dets_elem_dev30.json --out generalization_elem_dev30.json
P=""; for c in $M; do [ "$c" != OSFD ] && P="$P $c:OSFD"; done
for c in ROT7 ROT15 ROT30 SHR15 SHR30 ANI10 ANI25 SCL25 FLIP CAM-R15 CAM-R30; do P="$P $c:NOISE"; done
python scripts/pair_screen.py --subset dev30 --dets dets_elem_dev30.json --tag elem_dev30 --pairs $P
echo PIPELINE_DONE
