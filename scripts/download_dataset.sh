#!/usr/bin/env bash
# Tải + giải nén COCO 2017 val images + annotations vào data/coco/.
# Chạy: bash scripts/download_dataset.sh
# Idempotent: bỏ qua phần nào đã có sẵn đúng số lượng file mong đợi.
# Không commit data/coco/ (xem .gitignore) — tải lại mỗi phiên GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCO_DIR="$REPO_ROOT/data/coco"
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

log() { echo -e "\n[download_dataset] $*"; }

# images.cocodataset.org bị lỗi cert HTTPS phía server chính thức (cert trả về là của
# s3.amazonaws.com, không match hostname) — lỗi đã biết từ lâu, không phải do máy/mạng mình.
# Workaround: dùng http:// thay vì https:// (server vẫn phục vụ HTTP bình thường).
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"

if [ -f "annotations/instances_val2017.json" ]; then
  log "annotations/ đã có sẵn, bỏ qua tải lại."
else
  log "Tải annotations_trainval2017.zip (~252MB)..."
  curl -sS --max-time 600 -o annotations_trainval2017.zip "$ANNOTATIONS_URL"
  log "Giải nén annotations..."
  unzip -q -o annotations_trainval2017.zip
  rm -f annotations_trainval2017.zip
fi

VAL_IMG_COUNT=0
if [ -d "val2017" ]; then
  VAL_IMG_COUNT=$(find val2017 -maxdepth 1 -name '*.jpg' | wc -l)
fi
if [ "$VAL_IMG_COUNT" -eq 5000 ]; then
  log "val2017/ đã có đủ 5000 ảnh, bỏ qua tải lại."
else
  log "Tải val2017.zip (~815MB)..."
  curl -sS --max-time 1800 -o val2017.zip "$VAL_IMAGES_URL"
  log "Giải nén val2017..."
  unzip -q -o val2017.zip
  rm -f val2017.zip
fi

log "Xong. data/coco/ chứa $(find val2017 -maxdepth 1 -name '*.jpg' | wc -l) ảnh val2017 + annotations."
