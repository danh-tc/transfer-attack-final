#!/usr/bin/env bash
# Tải 4 checkpoint Controlled Panel (surrogate + 3 target) đúng theo docs/protocol_lock.md.
# Chạy sau khi scripts/setup_env.sh đã xong (cần mim/mmdet đã cài trong .venv).
# Chạy: bash scripts/download_checkpoints.sh          (Controlled Panel)
#       bash scripts/download_checkpoints.sh --gen    (thêm Generalization Panel, idea.md §6)
# Idempotent: mim download bỏ qua nếu checkpoint đã tồn tại đúng tên tại đích.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="$REPO_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

if [ ! -f "$REPO_ROOT/.venv/bin/activate" ]; then
  echo "[download_checkpoints][LỖI] Chưa thấy .venv — chạy scripts/setup_env.sh trước." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

log() { echo -e "\n[download_checkpoints] $*"; }

# Tên config phải khớp đúng docs/protocol_lock.md — sửa ở đó trước rồi mới sửa ở đây.
# Lưu ý: mim (v3.3.0 model-index) định danh checkpoint bằng đúng tên config file (bỏ .py,
# giữ nguyên "mask-rcnn" có gạch nối) — KHÔNG phải tên checkpoint kiểu v2 cũ "mask_rcnn_..."
# đã ghi trong protocol_lock.md/model_registry.md (đã sửa lại 2 file đó cho khớp).
CONFIGS=(
  "mask-rcnn_r50_fpn_mstrain-poly_3x_coco"                # Surrogate
  "mask-rcnn_r101_fpn_ms-poly-3x_coco"                    # Same-family target
  "mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco"    # Cross-CNN target
  "mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco"        # CNN->Transformer target
)

# Generalization Panel (docs/protocol_lock.md) — chỉ tải khi có --gen. YOLOX tải cả S và L
# vì size chưa chốt.
if [ "${1:-}" = "--gen" ]; then
  CONFIGS+=(
    "fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco"
    "detr_r50_8xb2-150e_coco"
    "yolox_s_8x8_300e_coco"            # mim id khác tên file config (yolox_s_8xb8-300e_coco.py)
    "yolox_l_8x8_300e_coco"
    "dino-5scale_swin-l_8xb2-36e_coco"
  )
fi

for cfg in "${CONFIGS[@]}"; do
  log "Tải checkpoint cho config: $cfg"
  mim download mmdet --config "$cfg" --dest "$CKPT_DIR"
done

log "Xong. Checkpoint nằm tại $CKPT_DIR (không commit — xem .gitignore)."
ls -la "$CKPT_DIR"
