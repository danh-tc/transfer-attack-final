#!/usr/bin/env python3
"""Vì sao DINO-Swin-L bền? (câu hỏi phụ của docs/plan_sota.md, chỉ forward, không model mới.)

Manh mối từ bước 2: với OSFD, D_4 trên backbone Swin-L (0.91) còn CAO hơn Swin-T (0.70) nhưng DINO-Swin-L
chỉ bị lừa ~25%. Hai thí nghiệm, 100 ảnh dev100, điều kiện OSFD / ABL-task-loss / ABL-no-RRB:

A. Distortion bị dập ở tầng nào: đo qua backbone -> neck -> encoder (memory) của DINO-Swin-L và DINO-R50
   (CÙNG detector DINO, chỉ khác backbone). Kèm Mask R-CNN Swin-T (backbone -> FPN) làm tham chiếu.
   Hai chỉ số không phụ thuộc thang đo kiến trúc: rel = ‖a − c‖/‖c‖ và cosd = 1 − cos(a, c) (vector phẳng).
B. Theo ảnh: Spearman(distortion ở từng tầng, retained confidence của chính model đó).

Chạy (sau khi có gen_dets/dino_r50.json cho dev100):
  python scripts/mech_swinl.py   -> results/ablation/mech/swinl.json
"""
import glob
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import per_image_transfer
from attack.models import load_model
from attack.preprocess import pipeline_resize, to_batch

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CONDS = ["OSFD", "ABL-task-loss", "ABL-no-RRB"]
DINO = {"dino_swin_l": ("dino/dino-5scale_swin-l_8xb2-36e_coco.py", "dino-5scale_swin-l*"),
        "dino_r50": ("dino/dino-4scale_r50_8xb2-12e_coco.py", "dino-4scale_r50*")}


@torch.no_grad()
def dino_levels(model, pix, ds):
    """-> {'backbone': [..], 'neck': [..], 'encoder': [memory]} (tensor phẳng theo batch 1)."""
    x, dss = to_batch(model, [pix], [ds])
    bb = model.backbone(x)
    neck = model.neck(bb)
    enc_in, _ = model.pre_transformer(neck, dss)
    mem = model.forward_encoder(**enc_in)["memory"]
    return {"backbone": list(bb), "neck": list(neck), "encoder": [mem]}


@torch.no_grad()
def mrcnn_levels(model, pix, ds):
    x, _ = to_batch(model, [pix], [ds])
    bb = model.backbone(x)
    return {"backbone": list(bb), "neck": list(model.neck(bb))}


def dist(a, c):
    a, c = a.flatten().double(), c.flatten().double()
    return float((a - c).norm() / c.norm()), float(1 - torch.dot(a, c) / (a.norm() * c.norm()))


def main():
    dev = "cuda:0"
    from mmdet.apis import init_detector
    models = {}
    for k, (cfg, pat) in DINO.items():
        models[k] = init_detector(os.path.join(CFG, cfg), glob.glob(os.path.join(REPO_ROOT, "checkpoints", pat + ".pth"))[0], device=dev)
    models["swin_t"] = load_model("swin_t", dev)
    fn = {"dino_swin_l": dino_levels, "dino_r50": dino_levels, "swin_t": mrcnn_levels}

    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    img_ids = [i for i in data.image_ids if i in keep]
    rec = {m: {c: {} for c in CONDS} for m in models}  # rec[m][c][tầng] = [n, 2] (rel, cosd)
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds = s["inputs"].to(dev), s["data_sample"]
        advs = {c: pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", c, f"{img_id}.png")))
                                   .permute(2, 0, 1).float().to(dev), x.shape[-2:]) for c in CONDS}
        for m, model in models.items():
            clean = fn[m](model, x, ds)
            for c, a in advs.items():
                out = fn[m](model, a, ds)
                for lvl, feats in out.items():
                    for li, (fa, fc) in enumerate(zip(feats, clean[lvl])):
                        rec[m][c].setdefault(f"{lvl}{li}", []).append(dist(fa, fc))
        if (j + 1) % 20 == 0:
            print(f"[swinl] {j + 1}/{len(img_ids)}", flush=True)

    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, "dets_abl_dev100.json")) as f:
        ctrl = json.load(f)
    dets = {"swin_t": {c: ctrl[c]["swin_t"] for c in ["clean"] + CONDS}}
    for k in DINO:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in ["clean"] + CONDS}

    out = {"n_images": len(img_ids), "conds": CONDS, "levels": {}, "per_image_spearman_vs_retained": {},
           "retained_mean": {}}
    for m in models:
        out["levels"][m], out["per_image_spearman_vs_retained"][m], out["retained_mean"][m] = {}, {}, {}
        for c in CONDS:
            ret = per_image_transfer(coco, dets[m]["clean"], dets[m][c], img_ids)["retained"]
            ok = ~np.isnan(ret)
            out["retained_mean"][m][c] = float(np.nanmean(ret))
            out["levels"][m][c] = {l: {"rel": float(np.mean([v[0] for v in vs])), "cosd": float(np.mean([v[1] for v in vs]))}
                                  for l, vs in rec[m][c].items()}
            out["per_image_spearman_vs_retained"][m][c] = {
                l: float(spearmanr(np.array([v[0] for v in vs])[ok], ret[ok])[0]) for l, vs in rec[m][c].items()}
    path = os.path.join(REPO_ROOT, "results/ablation/mech/swinl.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for m in models:
        print(f"\n=== {m} ===")
        for c in CONDS:
            L = out["levels"][m][c]; R = out["per_image_spearman_vs_retained"][m][c]
            print(f"{c:<15} retained {out['retained_mean'][m][c]:.2f} | " +
                  "  ".join(f"{l}: rel {v['rel']:.2f} cosd {v['cosd']:.3f} ρ {R[l]:+.2f}" for l, v in L.items()))


if __name__ == "__main__":
    main()
