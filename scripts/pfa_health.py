#!/usr/bin/env python3
"""PFA — cổng sức khỏe H1 (progress_log 2026-10-24): AP sạch của R50 có PFA (lệch cửa sổ 0, τ = 0.1, w = 7) trên dev30
so R50 thường; qua nếu ≥ 85%. CHỈ R50, ảnh sạch — không đo target nào.
Chạy: python scripts/pfa_health.py --alphas 0.25 0.5 0.75"""
import argparse
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.methods.pia import pfa_hook
from attack.models import CONTROLLED_PANEL

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--subset", default="dev30")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/pfa/health_h1.json"))
    args = p.parse_args()
    import contextlib
    from mmdet.apis import inference_detector, init_detector
    coco = COCO(ANN_FILE)
    cat_ids = sorted(coco.getCatIds())
    img_ids = load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.subset}.csv"))
    cfg = CONTROLLED_PANEL["r50"]
    model = init_detector(cfg.config_path, cfg.checkpoint_path, device="cuda:0")
    res = {}
    for a in [0.0] + args.alphas:
        ctx = pfa_hook(model, a, 0.1, 7, {"off": (0, 0)}) if a > 0 else contextlib.nullcontext()
        dets = []
        with ctx:
            for i in img_ids:
                dets += to_coco_dets(inference_detector(model, os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])),
                                     i, cat_ids)
        ap = float(CocoImageEval(coco, dets, img_ids).ap(np.ones(len(img_ids)))[0])
        res[str(a)] = ap
        rel = ap / res["0.0"]
        print(f"α = {a}: AP sạch R50 = {100 * ap:.1f} ({100 * rel:.0f}% R50 thường) -> H1 "
              f"{'QUA' if a == 0 or rel >= 0.85 else 'TRƯỢT'}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"subset": args.subset, "tau": 0.1, "window": 7, "ap_clean_r50": res}, f, indent=1)


if __name__ == "__main__":
    main()
