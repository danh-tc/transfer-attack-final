#!/usr/bin/env bash
# E74 (progress_log 2026-10-31): quét ngân sách B cho RD-CNX (ConvNeXt đơn) @ dev100 — B50, B75 (B36 đã có ở E73). Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
D="--split dev300 --subset dev100 --limit 100 --surrogate convnext_t --stage attack --methods C2-OCT-1V-WP2-RD-CNX"
python scripts/run_baselines.py $D --budget 50
python scripts/run_baselines.py $D --budget 75
python scripts/e74_cnx_budget.py
echo "[run.sh] XONG $(date '+%F %T')"
