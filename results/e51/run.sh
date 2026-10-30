#!/usr/bin/env bash
# E51 (progress_log 2026-10-30): cổng oracle white-box DINO-Swin-L theo vùng (FULL / IN / OUT / RING) @ dev30 B20.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
python scripts/e51_wb_bgmask.py --budget 20
