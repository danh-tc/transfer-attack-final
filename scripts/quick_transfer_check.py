#!/usr/bin/env python3
"""Kiểm tra transfer SƠ BỘ (không phải baseline table chính thức): n ảnh đầu của
n300, sinh adv trên surrogate R50, đánh giá trên cả 4 model Controlled Panel.

Metric:
- bbox AP (pycocotools, giới hạn imgIds = tập ảnh đang chạy, predict rescale=True
  về tọa độ gốc).
- GT-matched confidence trung bình (khung ảnh đã resize, predict rescale=False).

Mỗi method đánh giá 2 kiểu: "<method>" = tensor float ở khung ảnh đã resize
(như harness hiện tại), "<method>@img" = đưa về ảnh thật (kích thước gốc,
uint8) rồi target tự resize lại — xem to_image_space().

Chạy: python scripts/quick_transfer_check.py [n_images] [budget_B]
"""
import json
import os
import random
import sys
import time

import cv2
import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset
from attack.methods.baselines import di_fgsm_attack, mi_fgsm_attack, osfd_attack
from attack.models import CONTROLLED_PANEL, SURROGATE_KEY, load_model
from attack.preprocess import predict
from smoke_test_attacks import gt_matched_confidence

EPSILON = 5.0
SEED = 0
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")

METHODS = {
    "MI-FGSM": lambda m, x, ds, B: mi_fgsm_attack(m, x, ds, steps=B, epsilon=EPSILON),
    "DI-FGSM": lambda m, x, ds, B: di_fgsm_attack(m, x, ds, steps=B, epsilon=EPSILON),
    "OSFD": lambda m, x, ds, B: osfd_attack(m, x, ds, steps=B, epsilon=EPSILON),
}


def to_image_space(adv, clean, data_sample):
    """Đưa adv (float, khung ảnh đã resize) về ẢNH THẬT: lấy delta = adv - clean,
    thu delta về ori_shape (INTER_AREA — trung bình nên |delta| vẫn <= eps), cộng
    vào ảnh GỐC uint8 đọc từ file, làm tròn/clip uint8, rồi chạy lại đúng Resize
    của test pipeline (mmcv.imresize bilinear) như target nhận 1 file ảnh bình
    thường. delta=0 -> trả lại ĐÚNG clean (ảnh gốc không bị blur qua down/up).
    Channel order BGR giữ nguyên (LoadImageFromFile mặc định BGR)."""
    h, w = adv.shape[-2:]
    oh, ow = data_sample.ori_shape
    orig = mmcv.imread(data_sample.img_path).astype(np.float32)
    delta = (adv - clean).permute(1, 2, 0).cpu().numpy()
    delta = cv2.resize(delta, (ow, oh), interpolation=cv2.INTER_AREA)
    img = np.clip(np.round(orig + delta), 0, 255).astype(np.uint8)
    img = mmcv.imresize(img, (w, h), interpolation="bilinear", backend="cv2")
    return torch.from_numpy(img).permute(2, 0, 1).float().to(adv.device)


def to_coco_dets(result, img_id, cat_ids):
    b = result.pred_instances.bboxes.cpu().numpy()
    s = result.pred_instances.scores.cpu().numpy()
    l = result.pred_instances.labels.cpu().numpy()
    return [{"image_id": img_id, "category_id": cat_ids[int(li)],
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(si)}
            for (x1, y1, x2, y2), si, li in zip(b, s, l)]


def coco_ap(coco_gt, dets, img_ids):
    if not dets:
        return 0.0, 0.0
    ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
    ev.params.imgIds = img_ids
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return 100 * ev.stats[0], 100 * ev.stats[1]


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    budget_B = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    device = "cuda:0"
    models = {k: load_model(k, device) for k in CONTROLLED_PANEL}
    surrogate = models[SURROGATE_KEY]
    dataset = build_attack_dataset("n300")
    cat_ids = dataset._mmdet_dataset.cat_ids
    coco_gt = COCO(ANN_FILE)

    conds = ["clean"] + list(METHODS) + [f"{m}@img" for m in METHODS]
    dets = {c: {k: [] for k in models} for c in conds}
    conf = {c: {k: [] for k in models} for c in conds}
    runtime = {m: 0.0 for m in METHODS}
    img_ids = []

    for i in range(n_images):
        sample = dataset[i]
        img_id, x, ds = sample["img_id"], sample["inputs"].to(device), sample["data_sample"]
        img_ids.append(img_id)
        gt_boxes = ds.gt_instances.bboxes.tensor.cpu().numpy()
        gt_labels = ds.gt_instances.labels.cpu().numpy()

        advs = {"clean": x}
        for name, fn in METHODS.items():
            random.seed(SEED + i); torch.manual_seed(SEED + i)
            t0 = time.time()
            noise = fn(surrogate, x, ds, budget_B)
            torch.cuda.synchronize(); runtime[name] += time.time() - t0
            assert noise.abs().max() <= EPSILON + 1e-4
            advs[name] = torch.clamp(x + noise, 0.0, 255.0)
            advs[f"{name}@img"] = to_image_space(advs[name], x, ds)

        for c, img in advs.items():
            for k, m in models.items():
                dets[c][k] += to_coco_dets(predict(m, img, ds, rescale=True), img_id, cat_ids)
                c_list = gt_matched_confidence(predict(m, img, ds, rescale=False), gt_boxes, gt_labels)
                conf[c][k].append(float(np.mean(c_list)))
        print(f"[{i + 1}/{n_images}] img_id={img_id} n_gt={len(gt_boxes)}", flush=True)

    summary = {"n_images": n_images, "budget_B": budget_B, "epsilon": EPSILON,
               "img_ids": img_ids, "runtime_s_per_img": {m: t / n_images for m, t in runtime.items()},
               "ap": {}, "ap50": {}, "gt_conf": {}}
    for c in conds:
        summary["ap"][c], summary["ap50"][c], summary["gt_conf"][c] = {}, {}, {}
        for k in models:
            ap, ap50 = coco_ap(coco_gt, dets[c][k], img_ids)
            summary["ap"][c][k], summary["ap50"][c][k] = ap, ap50
            summary["gt_conf"][c][k] = float(np.mean(conf[c][k]))

    keys = list(models)
    print(f"\n=== bbox AP (n={n_images}, B={budget_B}, eps={EPSILON}) — trong ngoặc: relative AP drop ===")
    print(f"{'':<14}" + "".join(f"{k:>20}" for k in keys))
    for c in conds:
        row = f"{c:<14}"
        for k in keys:
            ap, ap0 = summary["ap"][c][k], summary["ap"]["clean"][k]
            row += f"{ap:>11.1f} ({100 * (ap0 - ap) / ap0:>5.1f}%)" if c != "clean" else f"{ap:>20.1f}"
        print(row)
    print(f"\n=== mean GT-matched confidence ===")
    print(f"{'':<14}" + "".join(f"{k:>12}" for k in keys))
    for c in conds:
        print(f"{c:<14}" + "".join(f"{summary['gt_conf'][c][k]:>12.3f}" for k in keys))
    print("\nruntime/ảnh (s):", {m: round(t, 1) for m, t in summary["runtime_s_per_img"].items()})

    out_dir = os.path.join(REPO_ROOT, "results/quick_transfer")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"n{n_images}_B{budget_B}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
