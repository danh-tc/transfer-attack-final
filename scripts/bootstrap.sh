#!/usr/bin/env bash
# Bootstrap toàn bộ cho một máy GPU thuê mới hoàn toàn (không có gì tồn tại ngoài repo git).
# Gộp: dựng venv -> tải checkpoint (Controlled + Generalization Panel + chẩn đoán) -> tải dataset
# -> chốt danh sách ảnh -> (nếu có HF_TOKEN trong .env) kéo ảnh adv baseline n300 từ HF.
# Chạy 1 lần: bash scripts/bootstrap.sh
# Idempotent: venv đã dùng được thì không dựng lại (setup_env.sh luôn xóa venv); các bước khác tự bỏ qua phần đã có.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\n[bootstrap] ===== $* =====\n"; }

log "1/5 Dựng môi trường Python (venv, torch, mmcv, mmdet)"
if [ -x .venv/bin/python ] && .venv/bin/python -c \
    "import torch, mmcv, mmdet, mmpretrain; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "[bootstrap] .venv đã dùng được (torch + CUDA + mmdet) — bỏ qua setup_env.sh."
else
  bash scripts/setup_env.sh
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

log "2/5 Tải checkpoint: Controlled + Generalization Panel + model chẩn đoán (--gen)"
bash scripts/download_checkpoints.sh --gen

log "3/5 Tải dataset COCO val2017"
bash scripts/download_dataset.sh

log "4/5 Chốt danh sách ảnh n=300/n=1000 (idempotent — bỏ qua nếu đã khóa từ trước)"
python scripts/generate_image_lists.py

log "5/5 Ảnh adv baseline n300 từ HF (cần HF_TOKEN trong .env — gitignored)"
if [ -f .env ] && grep -Eq '^HF_TOKEN *=' .env; then
  if [ -d artifacts/runs/n300_B50_eps5/adv ]; then
    echo "[bootstrap] artifacts/runs/n300_B50_eps5/ đã có — bỏ qua."
  else
    # Không in token: chỉ export vào môi trường của lệnh tải.
    HF_TOKEN="$(sed -n 's/^HF_TOKEN *= *//p' .env | head -1 | tr -d '\r"'"'"' ')" \
      python scripts/sync_artifacts.py download n300_B50_eps5
  fi
else
  echo "[bootstrap] Không có .env/HF_TOKEN — bỏ qua (chỉ cần cho GO @ n300; ảnh dev300 chưa từng lên HF)."
fi

log "Xong toàn bộ bootstrap."
echo "Nhắc: kích hoạt venv trong shell hiện tại bằng: source .venv/bin/activate"
echo "Nhắc: máy mới chưa có git identity — đặt cho repo: git config user.name/user.email (theo tác giả commit cũ)."
echo "Nhắc: nếu data/image_lists/ mới được tạo lần đầu (chưa từng commit), commit + push ngay để khóa lại."
