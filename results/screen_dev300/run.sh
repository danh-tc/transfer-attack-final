#!/usr/bin/env bash
# Sàng lọc dev300: C2-OCT-1V B41 vs OSFD B50 (progress_log 2026-10-18). Resume được: chạy lại đúng lệnh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
O=metrics_screen_dev300.json; D=dets_screen_dev300.json; G=generalization_screen_dev300.json
R50=dev300_B50_eps5_limit300; R41=dev300_B41_eps5_limit300
python scripts/run_baselines.py --split dev300 --limit 300 --methods OSFD --out $O
python scripts/run_baselines.py --split dev300 --limit 300 --budget 41 --methods C2-OCT-1V --out $O
python scripts/eval_generalization.py --run $R50 --models yolox_s dino_swin_l --methods OSFD --dets $D --out $G
python scripts/eval_generalization.py --run $R41 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets $D --out $G
python scripts/pair_screen.py --run $R50 --subset dev300 --dets $D --tag screen_c2oct1v_dev300 --exclude dev100 \
  --pairs C2-OCT-1V@$R41:OSFD
echo "[run.sh] XONG $(date '+%F %T')"
