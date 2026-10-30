#!/usr/bin/env bash
# E54 (progress_log 2026-10-30): dò tính chất chỉ-R50 tách δ_S khỏi δ_RD dọc trục C -> RD -> S @ dev30, forward/backward R50.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/e54_property_probe.py
