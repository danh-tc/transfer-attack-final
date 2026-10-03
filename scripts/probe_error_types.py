#!/usr/bin/env python3
"""Probe hướng 3: AP drop đến từ MẤT GT hay từ FALSE POSITIVE? (mô tả; không GPU, dùng dets có sẵn)

dev300, điều kiện OSFD / ABL-task-loss (đủ 300 ảnh), 6 model. Detection score >= 0.3 coi là "được phát hiện".
- recall@0.3: tỉ lệ GT (không crowd) có detection đúng lớp, IoU >= 0.5, score >= 0.3.
- FP/ảnh (score >= 0.3), tách: định vị (cùng lớp, 0.1 <= IoU < 0.5), nhầm lớp (khác lớp, IoU >= 0.5),
  nền (IoU < 0.1 với mọi GT), còn lại "khác".
- AP_noFP: AP khi chỉ giữ detection khớp 1 GT cùng lớp IoU >= 0.5 (bỏ mọi FP) -> AP chỉ còn phản ánh mất GT +
  thứ tự score. Phần AP mất do FP ≈ AP_noFP − AP.

Chạy: python scripts/probe_error_types.py   -> results/ablation/mech/error_types.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CONDS = ["clean", "OSFD", "ABL-task-loss"]
MODELS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
SCORE = 0.3


def iou_mat(a, b):
    """a [N,4], b [M,4] xywh -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def analyse(coco, dets, img_ids):
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    n_gt = hit = 0
    fp = defaultdict(int)
    keep_tp = []
    for img_id in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        g = np.array([a["bbox"] for a in anns]).reshape(-1, 4)
        gc = np.array([a["category_id"] for a in anns])
        ds = by_img.get(img_id, [])
        d = np.array([x["bbox"] for x in ds]).reshape(-1, 4)
        dc = np.array([x["category_id"] for x in ds])
        sc = np.array([x["score"] for x in ds])
        M = iou_mat(d, g)
        same = (dc[:, None] == gc[None]) if len(ds) and len(anns) else np.zeros((len(ds), len(anns)), bool)
        n_gt += len(anns)
        hi = sc >= SCORE
        if len(anns):
            hit += int(((M >= 0.5) & same & hi[:, None]).any(0).sum())
        for k, x in enumerate(ds):
            is_tp = len(anns) > 0 and bool(((M[k] >= 0.5) & same[k]).any())
            if is_tp:
                keep_tp.append(x)
            if not hi[k] or is_tp:
                continue
            if len(anns) == 0 or M[k].max() < 0.1:
                fp["background"] += 1
            elif (same[k] & (M[k] >= 0.1)).any():
                fp["localization"] += 1
            elif ((~same[k]) & (M[k] >= 0.5)).any():
                fp["class_confusion"] += 1
            else:
                fp["other"] += 1
    n = len(img_ids)
    ap = CocoImageEval(coco, dets, img_ids).ap(np.ones(n))[0] * 100
    ap_nofp = CocoImageEval(coco, keep_tp, img_ids).ap(np.ones(n))[0] * 100
    return {"recall@0.3": hit / n_gt, "fp_per_img": {k: v / n for k, v in fp.items()},
            "fp_per_img_total": sum(fp.values()) / n, "AP": ap, "AP_noFP": ap_nofp}


def main():
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, "dets_abl_dev300.json")) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in CONDS} for k in MODELS[:4]}
    for k in MODELS[4:]:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in CONDS}
    out = {m: {c: analyse(coco, dets[m][c], img_ids) for c in CONDS} for m in MODELS}
    path = os.path.join(REPO_ROOT, "results/ablation/mech/error_types.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{'model':<12}{'cond':<15}{'AP':>6}{'AP_noFP':>9}{'FPloss':>8}{'recall':>8}{'FP/img':>8}  bg / loc / cls")
    for m in MODELS:
        for c in CONDS:
            r = out[m][c]; f = r["fp_per_img"]
            print(f"{m:<12}{c:<15}{r['AP']:6.1f}{r['AP_noFP']:9.1f}{r['AP_noFP'] - r['AP']:8.1f}{r['recall@0.3']:8.2f}"
                  f"{r['fp_per_img_total']:8.1f}  {f.get('background', 0):.1f} / {f.get('localization', 0):.1f} / {f.get('class_confusion', 0):.1f}")


if __name__ == "__main__":
    main()
