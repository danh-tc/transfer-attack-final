#!/usr/bin/env bash
# E72-RGG (progress_log 2026-10-31): 5 biến thể RGG trên surrogate ConvNeXt @ dev30 B36, rồi eval + ablation.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
A="--split dev300 --subset dev30 --limit 100 --budget 36 --surrogate convnext_t"
python scripts/run_baselines.py $A --stage attack --methods RD-CNX-RGG-node RD-CNX-RGG-spatial RD-CNX-RGG-grad RD-CNX-RGG-shuffle RD-CNX-RGG-smooth
python scripts/e72_rgg.py
