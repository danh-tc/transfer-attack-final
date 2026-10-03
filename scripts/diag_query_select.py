#!/usr/bin/env python3
"""Chẩn đoán nhánh "tấn công cả DINO-Swin-L" (progress_log 2026-10-03) — chỉ forward, không attack mới.

Kênh "xếp hạng confidence" có phải đường lọt qua encoder DINO không? Tái tạo bước chọn query của DINO
(`DINO.pre_decoder`): điểm encoder s_i = max_c sigmoid(cls_branch[L](output_memory)), chọn top-900.
Mỗi (model, điều kiện, ảnh), 100 ảnh dev100:
- top100_mean: trung bình s của 100 token điểm cao nhất;
- overlap100: |top100(sạch) ∩ top100(adv)| / 100 (cùng vị trí token);
- GT: với mỗi GT, s_gt = max s trên các proposal encoder có IoU >= 0.5 với GT; in_top900 = GT có ít nhất
  1 proposal IoU >= 0.5 nằm trong top-900 (đầu vào decoder).
Model: DINO-Swin-L, DINO-R50 (cùng detector). Điều kiện: OSFD, C1 (task loss), C2 (OSFD + 4·cls),
chỉ-cls, bỏ RRB.

Chạy: python scripts/diag_query_select.py   -> results/ablation/mech/query_select.json
"""
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

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CONDS = ["OSFD", "ABL-task-loss", "PROBE-osfd+cls", "ABL-task-cls", "ABL-no-RRB"]
DINO = {"dino_swin_l": ("dino/dino-5scale_swin-l_8xb2-36e_coco.py", "dino-5scale_swin-l*"),
        "dino_r50": ("dino/dino-4scale_r50_8xb2-12e_coco.py", "dino-4scale_r50*")}


@torch.no_grad()
def encoder_scores(model, pix, ds):
    """-> (s [N] điểm token, boxes [N,4] xyxy khung ảnh model)."""
    x, dss = to_batch(model, [pix], [ds])
    enc_in, _ = model.pre_transformer(model.neck(model.backbone(x)), dss)
    enc = model.forward_encoder(**enc_in)
    out_mem, props = model.gen_encoder_output_proposals(enc["memory"], enc["memory_mask"], enc["spatial_shapes"])
    L = model.decoder.num_layers
    s = model.bbox_head.cls_branches[L](out_mem).sigmoid().max(-1)[0][0]
    c = (model.bbox_head.reg_branches[L](out_mem) + props).sigmoid()[0]  # cx, cy, w, h chuẩn hóa
    h, w = dss[0].img_shape
    b = torch.stack([(c[:, 0] - c[:, 2] / 2) * w, (c[:, 1] - c[:, 3] / 2) * h,
                     (c[:, 0] + c[:, 2] / 2) * w, (c[:, 1] + c[:, 3] / 2) * h], 1)
    return s, b


def iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda t: (t[:, 2:] - t[:, :2]).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def stats(s, b, gt, top_clean=None):
    top100 = torch.topk(s, 100).indices
    top900 = torch.topk(s, min(900, len(s))).indices
    r = {"top100_mean": float(s[top100].mean())}
    if top_clean is not None:
        r["overlap100"] = len(set(top100.tolist()) & set(top_clean.tolist())) / 100
    if len(gt):
        M = iou(gt, b) >= 0.5                                 # [G, N]
        sg = torch.where(M, s[None], torch.zeros_like(s)[None]).max(1)[0]
        in900 = M[:, top900].any(1).float()
        r["gt_score_mean"], r["gt_in_top900"] = float(sg.mean()), float(in900.mean())
    return r, top100


def main():
    dev = "cuda:0"
    from mmdet.apis import init_detector
    models = {k: init_detector(os.path.join(CFG, cfg), glob.glob(os.path.join(REPO_ROOT, "checkpoints", p + ".pth"))[0],
                               device=dev) for k, (cfg, p) in DINO.items()}
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    img_ids = [i for i in data.image_ids if i in keep]
    acc = {m: {c: [] for c in ["clean"] + CONDS} for m in models}
    for j, img_id in enumerate(img_ids):
        s0 = data[pos[img_id]]
        x, ds = s0["inputs"].to(dev), s0["data_sample"]
        gtb = ds.gt_instances.bboxes
        gt = (gtb.tensor if hasattr(gtb, "tensor") else gtb).float().to(dev)
        advs = {c: pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", c, f"{img_id}.png")))
                                   .permute(2, 0, 1).float().to(dev), x.shape[-2:]) for c in CONDS}
        for m, model in models.items():
            s, b = encoder_scores(model, x, ds)
            r, top_c = stats(s, b, gt)
            acc[m]["clean"].append(r)
            for c, a in advs.items():
                s, b = encoder_scores(model, a, ds)
                acc[m][c].append(stats(s, b, gt, top_c)[0])
        if (j + 1) % 20 == 0:
            print(f"[query] {j + 1}/{len(img_ids)}", flush=True)
    out = {"n_images": len(img_ids), "conds": ["clean"] + CONDS, "mean": {}}
    for m in models:
        out["mean"][m] = {c: {k: float(np.mean([r[k] for r in rs if k in r]))
                              for k in sorted(set().union(*(r.keys() for r in rs)))}
                          for c, rs in acc[m].items()}
    path = os.path.join(REPO_ROOT, "results/ablation/mech/query_select.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for m in models:
        print(f"\n=== {m} ===  (top100_mean | overlap100 | gt_score_mean | gt_in_top900)")
        for c, v in out["mean"][m].items():
            print(f"{c:<16} {v['top100_mean']:.3f} | {v.get('overlap100', 1.0):.2f} | {v.get('gt_score_mean', float('nan')):.3f} | {v.get('gt_in_top900', float('nan')):.3f}")


if __name__ == "__main__":
    main()
