#!/usr/bin/env bash
# E73 (progress_log 2026-10-31): XÁC NHẬN dev100 — ENS-WP / RD-CNX / RD-R50 / OSFD-R50, panel 9 model, boot 300. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
D="--split dev300 --subset dev100 --limit 100"
python scripts/run_baselines.py $D --budget 36 --surrogate convnext_t --stage attack --methods C2-OCT-1V-WP2-RD-CNX
python scripts/run_baselines.py $D --budget 36 --stage attack --methods C2-OCT-1V-WP2-RD
python scripts/run_baselines.py $D --budget 50 --stage attack --methods OSFD
python scripts/e73_dev100.py
echo "[run.sh] XONG $(date '+%F %T')"
