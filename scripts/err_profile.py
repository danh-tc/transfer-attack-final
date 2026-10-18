#!/usr/bin/env python3
"""Hồ sơ lỗi theo model (progress_log 2026-10-18): vì sao Mask R-CNN Swin-T dễ phá mà DINO-Swin-L kháng? — mô tả, không GPU.

Dùng dets có sẵn (sạch / OSFD B50 / C2-OCT-1V B41 @ dev100). Với mỗi model × điều kiện:
- recall@0.3 theo kích thước GT (COCO: nhỏ < 32², vừa < 96², lớn); GT "phát hiện" = det cùng lớp, IoU ≥ 0.5, score ≥ 0.3.
- conf_ratio: với GT phát hiện được ở CẢ sạch và adv, TB score_adv / score_sạch (det khớp score cao nhất).
- FP (score ≥ 0.3, không khớp GT cùng lớp IoU ≥ 0.5): / ảnh, tách theo
  vị trí: "sát object" (max IoU với GT bất kỳ ≥ 0.1) | "nền";
  cỡ: √(diện tích FP) / √(diện tích ảnh) — < 0.1 nhỏ, < 0.3 vừa, còn lại lớn;
  score: [0.3, 0.5) | ≥ 0.5.

Chạy: python scripts/err_profile.py -> results/mechanism/err_profile_dev100.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids  # noqa: E402
from scripts.probe_error_types import iou_mat  # noqa: E402

A = os.path.join(REPO_ROOT, "artifacts/runs")
SRC = {"clean": ("dev300_B50_eps5_limit100", "clean"), "OSFD": ("dev300_B50_eps5_limit100", "OSFD"),
       "C2-OCT-1V": ("dev300_B41_eps5_limit100", "C2-OCT-1V")}
MODELS = ["r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
SCORE = 0.3


def load(model, run, name):
    if model in ("yolox_s", "dino_swin_l"):
        d = json.load(open(os.path.join(A, run, "gen_dets", f"{model}.json")))[name]
    else:
        d = json.load(open(os.path.join(A, run, "dets_c2_1v_dev100.json")))[name][model]
    out = defaultdict(list)
    for x in d:
        if x["score"] >= SCORE:
            out[x["image_id"]].append(x)
    return out


def size_bucket(area):
    return "nhỏ" if area < 32 ** 2 else ("vừa" if area < 96 ** 2 else "lớn")


def profile(coco, dets, img_ids, clean_hits=None):
    n_gt, hit = defaultdict(int), defaultdict(int)
    fp = defaultdict(float)
    hits = {}  # (img, ann_id) -> score det khớp tốt nhất
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        img_area = info["width"] * info["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        g = np.array([a["bbox"] for a in anns]).reshape(-1, 4)
        gc = np.array([a["category_id"] for a in anns])
        ds = dets.get(img_id, [])
        d = np.array([x["bbox"] for x in ds]).reshape(-1, 4)
        dc = np.array([x["category_id"] for x in ds])
        M = iou_mat(d, g)
        same = (dc[:, None] == gc[None]) if len(ds) and len(anns) else np.zeros((len(ds), len(anns)), bool)
        ok = (M >= 0.5) & same
        for j, a in enumerate(anns):
            b = size_bucket(a["area"])
            n_gt[b] += 1
            if len(ds) and ok[:, j].any():
                hit[b] += 1
                hits[(img_id, a["id"])] = max(ds[k]["score"] for k in np.where(ok[:, j])[0])
        for k, x in enumerate(ds):
            if len(anns) and ok[k].any():
                continue
            near = len(anns) > 0 and M[k].max() >= 0.1
            rel = np.sqrt(x["bbox"][2] * x["bbox"][3] / img_area)
            sz = "nhỏ" if rel < 0.1 else ("vừa" if rel < 0.3 else "lớn")
            sc = "≥0.5" if x["score"] >= 0.5 else "0.3–0.5"
            fp["tổng"] += 1
            fp["sát object" if near else "nền"] += 1
            fp[f"cỡ {sz}"] += 1
            fp[f"score {sc}"] += 1
    n = len(img_ids)
    out = {"recall": {b: hit[b] / max(1, n_gt[b]) for b in ("nhỏ", "vừa", "lớn")},
           "n_gt": dict(n_gt), "fp_per_img": {k: v / n for k, v in fp.items()}}
    if clean_hits is not None:
        common = [k for k in hits if k in clean_hits]
        out["conf_ratio"] = float(np.mean([hits[k] / clean_hits[k] for k in common])) if common else None
        out["n_common"] = len(common)
    return out, hits


def main():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    coco = COCO(os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"))
    res = {}
    for m in MODELS:
        res[m] = {}
        clean_prof, clean_hits = profile(coco, load(m, *SRC["clean"]), ids)
        res[m]["clean"] = clean_prof
        for c in ("OSFD", "C2-OCT-1V"):
            res[m][c], _ = profile(coco, load(m, *SRC[c]), ids, clean_hits)
    path = os.path.join(REPO_ROOT, "results/mechanism/err_profile_dev100.json")
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    keys = ["sát object", "nền", "cỡ nhỏ", "cỡ vừa", "cỡ lớn", "score 0.3–0.5", "score ≥0.5"]
    print(f"{'model':<12}{'cond':<11}{'recall nhỏ/vừa/lớn':>22}{'conf':>6}{'FP/ảnh':>8}  " + " | ".join(keys))
    for m in MODELS:
        for c in ("clean", "OSFD", "C2-OCT-1V"):
            r = res[m][c]; f = r["fp_per_img"]
            rc = "/".join(f"{r['recall'][b]:.2f}" for b in ("nhỏ", "vừa", "lớn"))
            cr = f"{r['conf_ratio']:.2f}" if r.get("conf_ratio") else "  – "
            print(f"{m:<12}{c:<11}{rc:>22}{cr:>6}{f.get('tổng', 0):8.1f}  " + " | ".join(f"{f.get(k, 0):.1f}" for k in keys))
    print("n_gt nhỏ/vừa/lớn:", res["swin_t"]["clean"]["n_gt"])


if __name__ == "__main__":
    main()
