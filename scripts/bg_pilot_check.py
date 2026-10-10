#!/usr/bin/env python3
"""Kiểm thao tác pilot "object → nền" (progress_log 2026-10-10): s (tỉ lệ khoảng cách object → nền đã đi, như E14c) trên
Swin-L bb3 và R50 C4 cho các ô dev30. Chỉ forward.
  python scripts/bg_pilot_check.py [--cells OSFD OCT BG-OCT OCT+BG]   -> results/bg/pilot_check_dev30.json
"""
import argparse
import glob
import json
import os
import sys

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.preprocess import pipeline_resize, to_batch
from scripts.e14_patching import CFG
from scripts.e14c_bgshift import assign, bg_dirs, dir_stats

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=["OSFD", "OCT", "BG-OCT", "OCT+BG"])
    args = p.parse_args()
    from mmdet.apis import init_detector
    from attack.models import load_surrogate
    dev = "cuda:0"
    models = {"swinl": init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev),
              "r50": load_surrogate(dev)}
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    rec = {side: {c: [] for c in args.cells} for side in models}
    for img_id in ids:
        smp = data[pos[img_id]]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        for side, m in models.items():
            c3 = m.backbone(to_batch(m, [x], [ds])[0])[2][0]
            lab = assign(ds, *c3.shape[1:], dev)
            U, ok = bg_dirs(c3, lab)
            if not ok:
                continue
            for cell in args.cells:
                a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, cell, f"{img_id}.png")))
                                    .permute(2, 0, 1).float().to(dev), x.shape[-2:])
                rec[side][cell].append(dir_stats(m.backbone(to_batch(m, [a], [ds])[0])[2][0] - c3, U, lab >= 0))
    out = {side: {c: {"s": float(np.mean([r[0] for r in v])), "e_u": float(np.mean([r[1] for r in v])), "n_img": len(v)}
                  for c, v in d.items()} for side, d in rec.items()}
    path = os.path.join(REPO_ROOT, "results/bg/pilot_check_dev30.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for side, d in out.items():
        print(side, {c: round(v["s"], 3) for c, v in d.items()})


if __name__ == "__main__":
    main()
