#!/usr/bin/env python3
"""E3 (quy tắc khóa: progress_log 2026-10-06, chỉ mô tả): độ ĐỒNG BIẾN của feature backbone mỗi target với biến đổi hình
học, ảnh sạch dev30. cosd_s = 1 − cos(F_s(T'x), T'(F_s(x))) trên vùng hợp lệ của stage s, T' = các phép cố định của E2
(xoay +10°, trượt 0.15, phối cảnh yaw 15°, tịnh tiến (+7, +5) px), áp lên feature bằng CÙNG phép (đổi tọa độ theo stride).
Mỗi model dùng khung ảnh của chính nó (Resize theo test pipeline của nó).
  python scripts/e3_equivariance.py -> results/mechanism/e3_equivariance.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.methods.pia import warp, warp_grid
from attack.preprocess import resize_to_model, to_batch
from attack.targets import load_any, target_dataset
from scripts.e2_survival import transforms

MODELS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l", "dino_r50"]
OUT = os.path.join(REPO_ROOT, "results/mechanism")


def feat_grid(H, h, w, fh, fw, sy, sx, device):
    """Lưới grid_sample (align_corners=False) ở lưới feature fh × fw: vị trí feature (i, j) có tâm ảnh
    ((j + .5)·sx − .5, (i + .5)·sy − .5); nguồn = H·tâm; trả (grid [1,fh,fw,2], valid [fh,fw])."""
    ys, xs = torch.meshgrid(torch.arange(fh, dtype=torch.float64), torch.arange(fw, dtype=torch.float64), indexing="ij")
    px, py = (xs + 0.5) * sx - 0.5, (ys + 0.5) * sy - 0.5
    q = torch.stack([px, py, torch.ones_like(px)], -1) @ H.T
    u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    valid = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (px <= w - 1) & (py <= h - 1)
    fu, fv = (u + 0.5) / sx - 0.5, (v + 0.5) / sy - 0.5
    grid = torch.stack([(2 * fu + 1) / fw - 1, (2 * fv + 1) / fh - 1], -1)
    return grid.float()[None].to(device), valid.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="dev30")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    dev = "cuda:0"
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv"))
    ids = ids[:args.limit] if args.limit else ids
    res = {}
    for key in args.models:
        model = load_any(key, dev)
        tds = target_dataset(key, ids)
        acc = {}
        t0 = time.time()
        for i in range(len(tds)):
            smp = tds[i]
            ds = smp["data_sample"]
            x = resize_to_model(smp["orig_inputs"].to(dev), tuple(ds.img_shape))
            h, w = x.shape[-2:]
            with torch.no_grad():
                inp, _ = to_batch(model, [x], [ds])
                f0 = model.backbone(inp)
                for tn, H in transforms(h, w).items():
                    if H is None:
                        continue
                    inp_t, _ = to_batch(model, [warp(x, warp_grid(H, h, w, dev))], [ds])
                    ft = model.backbone(inp_t)
                    ch, cw = inp.shape[-2:]
                    for s, (a, b) in enumerate(zip(f0, ft)):
                        fh, fw = a.shape[-2:]
                        g, valid = feat_grid(H, h, w, fh, fw, ch / fh, cw / fw, dev)
                        aw = F.grid_sample(a, g, mode="bilinear", padding_mode="zeros", align_corners=False)
                        va, vb = aw[0][:, valid], b[0][:, valid]
                        cd = 1 - float(F.cosine_similarity(va.flatten(), vb.flatten(), dim=0))
                        acc.setdefault(f"{tn}/s{s}", []).append(cd)
        res[key] = {k: float(np.mean(v)) for k, v in acc.items()}
        res[key]["_per_image"] = acc
        print(f"[E3] {key}: {time.time() - t0:.0f}s  " + "  ".join(f"{k}={v:.3f}" for k, v in res[key].items()
                                                                 if not k.startswith("_")), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e3_equivariance.json"), "w") as f:
        json.dump({"subset": args.subset, "n_images": len(ids), "cosd": res}, f, indent=1)


if __name__ == "__main__":
    main()
