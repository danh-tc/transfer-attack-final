#!/usr/bin/env bash
# Sàng lọc C2-OCT-1V-WP2 @ dev300 (progress_log 2026-10-25). Resume được. OSFD/C2-OCT-1V kéo từ HF, chỉ attack WP2 B36.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export HF_TOKEN=$(python3 -c "ls=[l for l in open('.env') if l.strip().startswith('HF_TOKEN')]; print(ls[0].split('=',1)[1].strip().strip(chr(34)).strip(chr(39)))")
O=metrics_screen_dev300.json; D=dets_screen_dev300.json; G=generalization_screen_dev300.json
R36=dev300_B36_eps5_limit300; R41=dev300_B41_eps5_limit300; R50=dev300_B50_eps5_limit300
# 1) kéo dets baseline (idempotent)
for R in $R41 $R50; do [ -f artifacts/runs/$R/$D ] || python scripts/sync_artifacts.py download $R; done
# 2) attack + eval WP2 B36 (Controlled Panel), rồi gen panel
python scripts/run_baselines.py --split dev300 --limit 300 --budget 36 --methods C2-OCT-1V-WP2 --out $O
python scripts/eval_generalization.py --run $R36 --models yolox_s dino_swin_l --methods C2-OCT-1V-WP2 --dets $D --out $G
# 3) sàng lọc 3 chiều
python scripts/pair_screen.py --run $R50 --subset dev300 --dets $D --tag screen_wp2_dev300 --exclude dev100 \
  --pairs C2-OCT-1V-WP2@$R36:C2-OCT-1V@$R41 C2-OCT-1V-WP2@$R36:OSFD C2-OCT-1V@$R41:OSFD
echo "[run.sh] XONG $(date '+%F %T')"
