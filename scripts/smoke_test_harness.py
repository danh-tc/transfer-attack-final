#!/usr/bin/env python3
"""Kiểm tra nhanh khung harness mmdet v3 (attack/) chạy đúng trên GPU thật.

Không phải attack method thật (chưa có IFGSM/MI/DI/OSFD) — chỉ verify 3 thứ:
1. AttackDataset load được ảnh + GT từ data/image_lists/n300.csv.
2. Gradient lan truyền được từ loss thật (model.loss, GT-assisted) ngược về
   pixel tensor qua model.data_preprocessor (điều kiện tiên quyết để bất kỳ
   attack pixel-space nào hoạt động).
3. extract_features (nền tảng cho OSFD) và predict (nền tảng cho eval mAP)
   chạy không lỗi.

Chạy: python scripts/smoke_test_harness.py (từ repo root, venv đã activate).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.data import build_attack_dataset
from attack.models import load_surrogate
from attack.preprocess import compute_gt_loss, extract_features, predict


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    print("[smoke] Load surrogate (R50)...")
    model = load_surrogate(device=device)

    print("[smoke] Build AttackDataset(n300)...")
    dataset = build_attack_dataset("n300")
    print(f"[smoke] len(dataset)={len(dataset)}")

    sample = dataset[0]
    img_id, clean_pixels, data_sample = sample["img_id"], sample["inputs"], sample["data_sample"]
    clean_pixels = clean_pixels.to(device)
    print(f"[smoke] sample img_id={img_id}, inputs.shape={tuple(clean_pixels.shape)}, "
         f"n_gt_boxes={len(data_sample.gt_instances.bboxes)}")

    # --- Test 1: gradient lan truyền qua data_preprocessor tới pixel space ---
    noise = torch.zeros_like(clean_pixels, requires_grad=True)
    adv_pixels = torch.clamp(clean_pixels + noise, min=0.0, max=255.0)
    loss = compute_gt_loss(model, adv_pixels, data_sample)
    loss.backward()
    assert noise.grad is not None, "noise.grad is None — gradient KHÔNG lan truyền được!"
    grad_norm = noise.grad.norm().item()
    print(f"[smoke] GT loss={loss.item():.4f}, grad_norm={grad_norm:.6f}")
    assert grad_norm > 0, "grad_norm == 0 — nghi ngờ đồ thị tính toán bị đứt."
    print("[smoke] OK: gradient lan truyền được từ loss về pixel-space noise.")

    # --- Test 2: extract_features (nền tảng OSFD) ---
    with torch.no_grad():
        feats_clean = extract_features(model, clean_pixels, data_sample)
    print(f"[smoke] extract_features: {len(feats_clean)} stage, "
         f"shapes={[tuple(f.shape) for f in feats_clean]}")

    # --- Test 3: predict (nền tảng eval mAP) ---
    result = predict(model, clean_pixels, data_sample, rescale=True)
    n_pred = len(result.pred_instances.bboxes)
    top_score = result.pred_instances.scores.max().item() if n_pred > 0 else 0.0
    print(f"[smoke] predict (clean): {n_pred} detections, top_score={top_score:.4f}")

    print("\n[smoke] TẤT CẢ OK — harness sẵn sàng để cắm attack method vào.")


if __name__ == "__main__":
    main()
