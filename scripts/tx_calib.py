#!/usr/bin/env python3
"""Hiệu chỉnh liều WP2-TX (progress_log 2026-10-22, "RRB cho texture") CHỈ theo sức khỏe white-box — KHÔNG đo target nào.
Ảnh 1–4 dev300 (ảnh 0 = khởi động), PNG từ run_baselines --limit 5 --stage attack. Mỗi ô: tỉ lệ pixel |δ| = 5 (bão hòa
ngân sách), tỉ lệ object R50 bị xóa (GT R50 phát hiện được khi sạch; xóa = không còn box cùng lớp IoU ≥ 0.5, score ≥ 0.3).
Cách dùng: python scripts/tx_calib.py --run dev300_B36_eps5_limit5 --methods C2-OCT-1V-WP2 C2-OCT-1V-WP2-TXC2 ..."""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.models import CONTROLLED_PANEL

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    args = p.parse_args()
    from mmdet.apis import inference_detector, init_detector
    coco = COCO(ANN_FILE)
    cat_ids = sorted(coco.getCatIds())
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))[1:5]
    cfg = CONTROLLED_PANEL["r50"]
    model = init_detector(cfg.config_path, cfg.checkpoint_path, device="cuda:0")

    def found(path, img_id):
        dets = [d for d in to_coco_dets(inference_detector(model, path), img_id, cat_ids) if d["score"] >= 0.3]
        return {g["id"]: any(d["category_id"] == g["category_id"] and iou(d["bbox"], g["bbox"]) >= .5 for d in dets)
                for g in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))}

    clean = {i: found(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"]), i) for i in img_ids}
    for m in args.methods:
        kill, sat = [], []
        for i in img_ids:
            png = os.path.join(REPO_ROOT, "artifacts/runs", args.run, "adv", m, f"{i}.png")
            orig = np.asarray(Image.open(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).convert("RGB"), dtype=int)
            sat.append(float((np.abs(np.asarray(Image.open(png).convert("RGB"), dtype=int) - orig) >= 5).mean()))
            adv = found(png, i)
            kill += [not adv[g] for g, ok in clean[i].items() if ok]
        print(f"{m}: |δ| = 5 ở {100 * np.mean(sat):.0f}% pixel | R50 white-box xóa {100 * np.mean(kill):.0f}% "
              f"({sum(kill)}/{len(kill)} object)")


if __name__ == "__main__":
    main()
