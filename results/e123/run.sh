#!/usr/bin/env bash
# E1–E3 (progress_log 2026-10-06). Chạy lại được (E1/E2/E3 tính lại từ đầu; predict Swin-L resume theo gen_dets).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONUTF8=1
python scripts/e1_expected_grad.py --subset dev30
python scripts/eval_generalization.py --run n300_B50_eps5 --models dino_swin_l --out generalization_e2_swinl.json
python scripts/e2_survival.py --run n300_B50_eps5 --tag n300
python scripts/e3_equivariance.py --subset dev30
echo PIPELINE_DONE
