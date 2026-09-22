#!/usr/bin/env python3
"""Kiểm tra 3 attack core (MI-FGSM, DI-FGSM, OSFD) chạy đúng trên GPU thật.

Không phải baseline table thật (chỉ 1 ảnh, steps nhỏ) — chỉ verify:
1. Chạy hết `steps` không lỗi, không NaN.
2. Noise cuối cùng nằm trong [-epsilon, epsilon] (đúng constraint).
3. Ảnh adversarial thực sự đánh lừa được surrogate (near white-box): số
   detection giảm hoặc confidence giảm rõ rệt so với ảnh sạch — nếu không,
   nghi ngờ core logic (dấu gradient, ascent/descent) bị sai.

Chạy: python scripts/smoke_test_attacks.py (từ repo root, venv đã activate).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.data import build_attack_dataset
from attack.methods.augtrans import augtrans_attack
from attack.methods.baselines import di_fgsm_attack, mi_fgsm_attack, osfd_attack
from attack.models import load_surrogate
from attack.preprocess import predict

STEPS = 20
EPSILON = 5.0
AUGTRANS_BUDGET_B = 50  # -> K_max=5 (N_EOT=10) — nhỏ, chỉ để smoke-test chạy nhanh


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def gt_matched_confidence(result, gt_boxes, gt_labels, iou_thr=0.5):
    """Với mỗi GT box, tìm predicted box tốt nhất (IoU cao nhất) và trả về
    confidence của nó — đây là tín hiệu ĐÚNG để đánh giá attack (không phải
    đếm tổng số box: attack untargeted có thể sinh rất nhiều false positive
    tràn lan mà KHÔNG hề suppress detection thật, vẫn là attack thành công
    vì làm sập Precision/AP — xem docs/progress_log.md)."""
    boxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    confidences = []
    for gb, gl in zip(gt_boxes, gt_labels):
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(boxes, scores, labels):
            if l != gl:
                continue
            v = box_iou(gb, b)
            if v > best_iou:
                best_iou, best_score = v, s
        confidences.append(best_score if best_iou > iou_thr else 0.0)
    return confidences


def run_one(name, attack_fn, model, clean_pixels, data_sample, gt_boxes, gt_labels, **attack_kwargs):
    print(f"\n[smoke-attack] === {name} ===")
    noise = attack_fn(model, clean_pixels, data_sample, epsilon=EPSILON, **attack_kwargs)
    assert not torch.isnan(noise).any(), f"{name}: noise có NaN"
    linf = noise.abs().max().item()
    print(f"  realized L_inf={linf:.4f} (epsilon={EPSILON})")
    assert linf <= EPSILON + 1e-4, f"{name}: vi phạm epsilon-ball ({linf} > {EPSILON})"

    adv_pixels = torch.clamp(clean_pixels + noise, min=0.0, max=255.0)
    with torch.no_grad():
        clean_result = predict(model, clean_pixels, data_sample, rescale=True)
        adv_result = predict(model, adv_pixels, data_sample, rescale=True)

    n_clean = len(clean_result.pred_instances.bboxes)
    n_adv = len(adv_result.pred_instances.bboxes)
    print(f"  tổng số detection: clean={n_clean}, adv={n_adv} "
         f"(KHÔNG dùng số này để đánh giá — attack untargeted có thể sinh false positive tràn lan)")

    conf_clean = gt_matched_confidence(clean_result, gt_boxes, gt_labels)
    conf_adv = gt_matched_confidence(adv_result, gt_boxes, gt_labels)
    print(f"  confidence trên GT-matched box: clean={[f'{c:.3f}' for c in conf_clean]}")
    print(f"                                  adv  ={[f'{c:.3f}' for c in conf_adv]}")

    mean_drop = sum(conf_clean) / len(conf_clean) - sum(conf_adv) / len(conf_adv)
    if mean_drop > 0.01:
        print(f"  -> OK: confidence trung bình trên GT giảm {mean_drop:.3f} — attack đúng hướng.")
    else:
        print(f"  -> CẢNH BÁO: confidence trên GT không giảm (delta={mean_drop:.3f}) — kiểm tra lại core logic.")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-attack] device={device}, steps={STEPS}, epsilon={EPSILON}")

    model = load_surrogate(device=device)
    dataset = build_attack_dataset("n300")
    sample = dataset[0]
    clean_pixels = sample["inputs"].to(device)
    data_sample = sample["data_sample"]
    print(f"[smoke-attack] img_id={sample['img_id']}, shape={tuple(clean_pixels.shape)}, "
         f"n_gt={len(data_sample.gt_instances.bboxes)}")
    raw_boxes = data_sample.gt_instances.bboxes
    gt_boxes = (raw_boxes.tensor if hasattr(raw_boxes, "tensor") else raw_boxes).cpu().numpy()
    gt_labels = data_sample.gt_instances.labels.cpu().numpy()

    run_one("MI-FGSM", mi_fgsm_attack, model, clean_pixels, data_sample, gt_boxes, gt_labels, steps=STEPS)
    run_one("DI-FGSM", di_fgsm_attack, model, clean_pixels, data_sample, gt_boxes, gt_labels, steps=STEPS)
    run_one("OSFD", osfd_attack, model, clean_pixels, data_sample, gt_boxes, gt_labels, steps=STEPS)
    run_one("AugTrans", augtrans_attack, model, clean_pixels, data_sample, gt_boxes, gt_labels,
           budget_B=AUGTRANS_BUDGET_B)

    print("\n[smoke-attack] Xong.")


if __name__ == "__main__":
    main()
