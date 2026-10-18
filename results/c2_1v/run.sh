#!/usr/bin/env bash
# C2-OCT-1V B41 / C2-PIA-1V B45 vs OSFD B50 @ dev100 (progress_log 2026-10-18). Resume được: chạy lại đúng lệnh.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/bootstrap.sh
source .venv/bin/activate
O=metrics_c2_1v_dev100.json; D=dets_c2_1v_dev100.json; G=generalization_c2_1v_dev100.json
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --methods OSFD --out $O
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 41 --methods C2-OCT-1V --out $O
python scripts/run_baselines.py --split dev300 --subset dev100 --limit 100 --budget 45 --methods C2-PIA-1V --out $O
python scripts/eval_generalization.py --run dev300_B50_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods OSFD --dets $D --out $G
python scripts/eval_generalization.py --run dev300_B41_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-OCT-1V --dets $D --out $G
python scripts/eval_generalization.py --run dev300_B45_eps5_limit100 --subset dev100 --models yolox_s dino_swin_l --methods C2-PIA-1V --dets $D --out $G
python scripts/pair_screen.py --run dev300_B50_eps5_limit100 --subset dev100 --dets $D --tag c2_1v_dev100 --exclude dev30 \
  --pairs C2-OCT-1V@dev300_B41_eps5_limit100:OSFD C2-PIA-1V@dev300_B45_eps5_limit100:OSFD
echo "[run.sh] XONG"
