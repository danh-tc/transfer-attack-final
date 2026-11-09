#!/usr/bin/env bash
# E81 (progress_log 2026-11-09): phân bổ ngân sách 2 surrogate + liều ρ WP @ dev30 B36, V100. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "[run.sh] BẮT ĐẦU $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
.venv/bin/python scripts/e81_ens_alloc_rho.py
echo "[run.sh] XONG $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
