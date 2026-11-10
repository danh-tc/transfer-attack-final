#!/usr/bin/env bash
# E82 (progress_log 2026-11-10): strict-1 + WP @ dev30, panel gọn 4 model, boot 100. Resume được.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "[run.sh] BẮT ĐẦU $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
.venv/bin/python scripts/e82_strict1_wp.py "$@"
echo "[run.sh] XONG $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
