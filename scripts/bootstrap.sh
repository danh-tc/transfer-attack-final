#!/usr/bin/env bash
# Bootstrap toàn bộ cho một máy GPU thuê mới hoàn toàn (không có gì tồn tại ngoài repo git).
# Gộp: dựng venv -> tải checkpoint -> tải dataset -> chốt danh sách ảnh n=300/n=1000.
# Chạy: bash scripts/bootstrap.sh
# Idempotent: mỗi bước con tự bỏ qua phần đã xong (xem log của từng script).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\n[bootstrap] ===== $* =====\n"; }

log "1/4 Dựng môi trường Python (venv, torch, mmcv, mmdet)"
bash scripts/setup_env.sh

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

log "2/4 Tải checkpoint Controlled Panel"
bash scripts/download_checkpoints.sh

log "3/4 Tải dataset COCO val2017"
bash scripts/download_dataset.sh

log "4/4 Chốt danh sách ảnh n=300/n=1000 (idempotent — bỏ qua nếu đã khóa từ trước)"
python scripts/generate_image_lists.py

log "Xong toàn bộ bootstrap."
echo "Nhắc: kích hoạt venv trong shell hiện tại bằng: source .venv/bin/activate"
echo "Nhắc: nếu data/image_lists/ mới được tạo lần đầu (chưa từng commit), commit + push ngay để khóa lại."
