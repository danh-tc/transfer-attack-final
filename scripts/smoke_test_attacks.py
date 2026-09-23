#!/usr/bin/env python3
"""Kiểm tra 3 attack baseline (MI-FGSM, DI-FGSM, OSFD — AugTrans tạm bỏ khỏi plan) chạy đúng trên GPU thật.

Không phải baseline table thật (vài ảnh, steps nhỏ) — chỉ verify:
0. GT (box + mask) nằm ĐÚNG khung ảnh đã resize (bug cũ: LoadAnnotations đứng
   sau Resize trong test pipeline -> GT ở tọa độ ảnh gốc, xem docs/progress_log.md),
   và detection sạch của surrogate khớp GT.
1. Co-transform GT của DI: bbox của mask đã biến đổi khớp box đã biến đổi.
2. Chạy hết `steps` không lỗi, không NaN; noise nằm trong [-epsilon, epsilon].
3. Ảnh adversarial thực sự đánh lừa được surrogate (near white-box): confidence
   trên GT-matched box giảm rõ — nếu không, nghi ngờ core logic.
4. A/B với GT lệch khung kiểu bug cũ (--no-legacy để bỏ qua) — đo ảnh hưởng của bug.

Chạy: python scripts/smoke_test_attacks.py (từ repo root, venv đã activate).
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.data import build_attack_dataset
from attack.methods.baselines import di_fgsm_attack, mi_fgsm_attack, osfd_attack
from attack.methods.box_transforms import get_gt
from attack.methods.diversity import input_diversity_with_boxes
from attack.models import load_surrogate
from attack.preprocess import predict

N_IMAGES = 3
STEPS = 20
EPSILON = 5.0


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
    vì làm sập Precision/AP — xem docs/progress_log.md).

    result phải ở CÙNG khung tọa độ với gt_boxes (predict rescale=False,
    khung ảnh đã resize)."""
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


def mask_bbox(mask):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if len(xs) == 0:
        return None
    return [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


def check_gt_frame(model, clean_pixels, data_sample, gt_boxes, gt_labels):
    """(0) GT nằm đúng khung ảnh đã resize + detection sạch khớp GT."""
    h, w = clean_pixels.shape[-2:]
    boxes, masks = get_gt(data_sample, clean_pixels.device)
    assert masks is not None and tuple(masks.shape[-2:]) == (h, w), \
        f"mask shape {None if masks is None else tuple(masks.shape)} != ảnh {(h, w)}"
    assert boxes[:, [0, 2]].max() <= w + 1 and boxes[:, [1, 3]].max() <= h + 1, "GT box vượt khung ảnh"
    # bbox của mask phải khớp GT box (COCO box ~ tight bbox của polygon)
    ious = [box_iou(b.tolist(), mb) for b, m in zip(boxes, masks) if (mb := mask_bbox(m))]
    clean = predict(model, clean_pixels, data_sample, rescale=False)
    conf = gt_matched_confidence(clean, gt_boxes, gt_labels)
    matched = sum(c > 0.3 for c in conf) / len(conf)
    print(f"  [GT frame] ảnh {w}x{h}, GT box max=({boxes[:, 2].max():.0f},{boxes[:, 3].max():.0f}), "
          f"mean IoU(mask-bbox, box)={sum(ious) / len(ious):.3f}, "
          f"GT được detect (clean, score>0.3)={matched:.0%}")
    assert sum(ious) / len(ious) > 0.8, "mask không khớp box"
    return matched


def check_cotransform(name, view_fn, clean_pixels, data_sample, n_trials=5):
    """(1) Sau biến đổi của DI, bbox của mask vẫn khớp box đã biến đổi."""
    ious = []
    for _ in range(n_trials):
        img, ds = view_fn(clean_pixels, data_sample)
        assert img.shape == clean_pixels.shape
        boxes, masks = get_gt(ds, clean_pixels.device)
        assert masks is not None and masks.shape[-2:] == img.shape[-2:]
        for b, m in zip(boxes, masks):
            mb = mask_bbox(m)
            if mb is not None and (b[2] - b[0]) > 4 and (b[3] - b[1]) > 4:
                ious.append(box_iou(b.tolist(), mb))
    mean_iou = sum(ious) / max(1, len(ious))
    print(f"  [co-transform {name}] mean IoU(mask-bbox, box) sau biến đổi = {mean_iou:.3f} ({len(ious)} cặp)")
    return mean_iou


def to_legacy_frame(data_sample):
    """Tái tạo bug cũ: GT ở tọa độ ảnh GỐC trong khi ảnh đã resize."""
    ds = copy.deepcopy(data_sample)
    sx, sy = ds.scale_factor
    boxes = ds.gt_instances.bboxes.tensor.clone()
    boxes[:, [0, 2]] /= sx
    boxes[:, [1, 3]] /= sy
    ds.gt_instances.bboxes = boxes
    ds.gt_instances.masks = ds.gt_instances.masks.resize(ds.ori_shape)
    return ds


def run_one(name, attack_fn, model, clean_pixels, data_sample, gt_boxes, gt_labels,
            attack_data_sample=None, **attack_kwargs):
    noise = attack_fn(model, clean_pixels, attack_data_sample or data_sample,
                      epsilon=EPSILON, **attack_kwargs)
    assert not torch.isnan(noise).any(), f"{name}: noise có NaN"
    linf = noise.abs().max().item()
    assert linf <= EPSILON + 1e-4, f"{name}: vi phạm epsilon-ball ({linf} > {EPSILON})"

    adv_pixels = torch.clamp(clean_pixels + noise, min=0.0, max=255.0)
    clean_result = predict(model, clean_pixels, data_sample, rescale=False)
    adv_result = predict(model, adv_pixels, data_sample, rescale=False)
    conf_clean = gt_matched_confidence(clean_result, gt_boxes, gt_labels)
    conf_adv = gt_matched_confidence(adv_result, gt_boxes, gt_labels)
    mean_drop = sum(conf_clean) / len(conf_clean) - sum(conf_adv) / len(conf_adv)
    print(f"  {name:<22} L_inf={linf:.3f}  det clean/adv={len(clean_result.pred_instances)}/"
          f"{len(adv_result.pred_instances)}  GT-conf clean={sum(conf_clean) / len(conf_clean):.3f} "
          f"adv={sum(conf_adv) / len(conf_adv):.3f}  drop={mean_drop:+.3f}")
    return mean_drop


def main():
    legacy = "--no-legacy" not in sys.argv
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-attack] device={device}, n_images={N_IMAGES}, steps={STEPS}, epsilon={EPSILON}")

    model = load_surrogate(device=device)
    dataset = build_attack_dataset("n300")
    drops = {}
    for i in range(N_IMAGES):
        sample = dataset[i]
        clean_pixels = sample["inputs"].to(device)
        data_sample = sample["data_sample"]
        gt_boxes = data_sample.gt_instances.bboxes.tensor.cpu().numpy()
        gt_labels = data_sample.gt_instances.labels.cpu().numpy()
        print(f"\n[smoke-attack] img_id={sample['img_id']}, shape={tuple(clean_pixels.shape)}, n_gt={len(gt_boxes)}")

        check_gt_frame(model, clean_pixels, data_sample, gt_boxes, gt_labels)
        di_iou = check_cotransform("DI", lambda x, ds: input_diversity_with_boxes(x, ds, prob=1.0),
                                   clean_pixels, data_sample)
        assert di_iou > 0.8, "co-transform mask/box lệch"

        runs = [
            ("MI-FGSM", mi_fgsm_attack, None, dict(steps=STEPS)),
            ("DI-FGSM", di_fgsm_attack, None, dict(steps=STEPS)),
            ("OSFD", osfd_attack, None, dict(steps=STEPS)),
        ]
        if legacy:
            legacy_ds = to_legacy_frame(data_sample)
            runs += [
                ("MI-FGSM [GT lệch cũ]", mi_fgsm_attack, legacy_ds, dict(steps=STEPS)),
                ("DI-FGSM [GT lệch cũ]", di_fgsm_attack, legacy_ds, dict(steps=STEPS)),
            ]
        for name, fn, attack_ds, kwargs in runs:
            drop = run_one(name, fn, model, clean_pixels, data_sample, gt_boxes, gt_labels,
                           attack_data_sample=attack_ds, **kwargs)
            drops.setdefault(name, []).append(drop)

    print("\n[smoke-attack] === Tổng hợp: mean GT-confidence drop qua", N_IMAGES, "ảnh ===")
    for name, ds in drops.items():
        mean = sum(ds) / len(ds)
        flag = "OK" if mean > 0.01 else "CẢNH BÁO: không giảm"
        print(f"  {name:<22} {mean:+.3f}   ({', '.join(f'{d:+.3f}' for d in ds)})  {flag}")
    print("\n[smoke-attack] Xong.")


if __name__ == "__main__":
    main()
