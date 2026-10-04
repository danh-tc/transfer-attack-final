#!/usr/bin/env python3
"""Độ trung thực của trường MORGD (quy tắc khóa: progress_log 2026-10-04 "Cải tiến MORGD"). Forward-only, dev30.

--version v1: MORGD gốc (morgd.sample_anchors + flow_grid), so neo object vs random (SMOOTH).
  Mỗi object (bản object): dời tâm thực tế (AABB mặt nạ chữ nhật warp) so với Δ_i -> ratio_pos, cos.
  Cả hai bản: D_pair_pos_box, D_pair_scale_box (grgd._box_pair_stats), RMS |u|.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.methods.box_transforms import get_gt
from attack.methods.grgd import _box_pair_stats
from attack.methods.morgd import flow_field, pixel_coords, sample_anchors, to_grid, warp_boxes_by_grid
from attack.preprocess import resize_to_model

SEED = 0  # = run_baselines.SEED


def centers(b):
    return torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def v1_view(boxes, h, w, anchors, xs, ys):
    anc = sample_anchors(boxes, h, w, anchors)
    ux, uy, _ = flow_field(anc, xs, ys)
    grid = to_grid(xs, ys, ux, uy, h, w)
    nb, keep = warp_boxes_by_grid(boxes, grid, h, w)
    st = {"rms_u_px": float((ux ** 2 + uy ** 2).mean().sqrt())}
    st.update(_box_pair_stats(boxes, nb, keep))
    obj = []
    if anchors == "object" and keep.any():
        tgt = torch.tensor([(a[4], a[5]) for a in anc], device=boxes.device)[keep]
        real = centers(nb) - centers(boxes[keep])
        ok = tgt.norm(dim=1) >= 1
        for r, t in zip(real[ok], tgt[ok]):
            obj.append((float(r.norm() / t.norm()), float(torch.nn.functional.cosine_similarity(r, t, dim=0))))
    return st, obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1"], default="v1")
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/morgd2/fidelity_v1.json"))
    args = p.parse_args()

    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ds = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    rows = {"object": [], "random": []}
    objs = []
    for i in range(len(ds)):
        smp = ds[i]
        d = smp["data_sample"]
        im = resize_to_model(smp["orig_inputs"].cuda(), tuple(d.img_shape))
        h, w = im.shape[-2:]
        boxes, _ = get_gt(d, im.device)
        ys, xs = pixel_coords(h, w, im.device)
        seed = SEED + split.index(smp["img_id"])
        for anc in rows:
            random.seed(seed); torch.manual_seed(seed)
            for _ in range(args.views):
                st, ob = v1_view(boxes, h, w, anc, xs, ys)
                st["n_objects"] = len(boxes)
                rows[anc].append(st)
                if anc == "object":
                    objs += ob

    o = np.array(objs)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs if k in r]))
    ratio = mean(rows["object"], "d_pair_pos_box") / mean(rows["random"], "d_pair_pos_box")
    med = float(np.median(o[:, 0]))
    out = {"version": args.version, "split": "dev30", "views_per_image": args.views, "n_objects_measured": len(o),
           "median_ratio_pos": med, "mean_ratio_pos": float(o[:, 0].mean()), "mean_cos": float(o[:, 1].mean()),
           "ratio_pos_by_quartile": [float(x) for x in np.percentile(o[:, 0], [25, 50, 75])],
           "d_pair_pos_box": {k: mean(v, "d_pair_pos_box") for k, v in rows.items()},
           "d_pair_scale_box": {k: mean(v, "d_pair_scale_box") for k, v in rows.items()},
           "rms_u_px": {k: mean(v, "rms_u_px") for k, v in rows.items()},
           "object_over_random_d_pair_pos_box": ratio,
           "hypothesis_supported": bool(med < 0.75 or ratio < 1.5)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
