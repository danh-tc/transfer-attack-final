#!/usr/bin/env bash
# Nối tiếp run.sh sau khi E1 xong (bước phân tích E1 lỗi ghi JSON đã sửa, chạy --analyze-only riêng).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONUTF8=1
python scripts/eval_generalization.py --run n300_B50_eps5 --models dino_swin_l --out generalization_e2_swinl.json
python scripts/e2_survival.py --run n300_B50_eps5 --tag n300
python scripts/e3_equivariance.py --subset dev30
echo PIPELINE_DONE
