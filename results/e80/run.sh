#!/usr/bin/env bash
# E80-ENSB (progress_log 2026-11-09): ENS-WP B50 @ dev30, panel gọn 4 model, boot 100, máy V100.
# Resume được: PNG đã có thì bỏ qua; json ghi sau mỗi model.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "[run.sh] BẮT ĐẦU $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
.venv/bin/python scripts/e80_ens_budget.py
echo "[run.sh] XONG $(TZ=Asia/Bangkok date '+%Y-%m-%d %H:%M:%S GMT+7')"
