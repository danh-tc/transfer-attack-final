#!/usr/bin/env python3
"""E51 (progress_log 2026-10-30): cổng ORACLE — pixel nền CÓ THỂ tác động DINO-Swin-L không? White-box trên CHÍNH
DINO-Swin-L (chỉ phân tích, như E8; không vào method), δ giới hạn theo vùng GT box của E50:
  WB-FULL (cả ảnh) | WB-IN (trong hợp box) | WB-OUT (ngoài box) | WB-RING (vành 25% trừ IN).
Loss = số hạng phân loại không-dn của DINO (target_grads objective 'cls'), ascent; MI μ = 1, α = 1, B bước, δ ← δ·mask mỗi bước.
  python scripts/e51_wb_bgmask.py --budget 20 [--limit 1]
"""
import argparse
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.mechanism import per_image_transfer
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads
from scripts.e50_bg_mask import ANN, masks

ART = os.path.join(REPO_ROOT, "artifacts/e51")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e51_wb_bgmask.json")
CONDS = {"WB-FULL": None, "WB-IN": "RD-IN", "WB-OUT": "RD-OUT", "WB-RING": "RD-RING"}
SEED = 0


def attack(model, orig, ds, m, steps, eps, alpha=1.0, mu=1.0):
    d = torch.zeros_like(orig)
    acc = torch.zeros_like(orig)
    for _ in range(steps):
        g_img, _, _ = target_grads(model, "dino_swin_l", torch.clamp(orig + d, 0, 255), ds, ("cls",))
        g = g_img["cls"] * m
        g = g / (g.abs().sum() / m.sum().clamp(min=1) / 3 + 1e-12)  # chuẩn hóa theo TB |g| trên vùng được phép
        acc = mu * acc + g
        d = torch.clamp(d + alpha * torch.sign(acc), -eps, eps) * m
    return d


def evaluate(coco, ids):
    from mmdet.apis import inference_detector
    cat_ids = sorted(coco.getCatIds())
    model = load_any("dino_swin_l")
    clean = lambda i: os.path.join(REPO_ROOT, "data/coco/val2017", coco.loadImgs(i)[0]["file_name"])
    dets = {c: [] for c in ["clean", *CONDS]}
    for i in ids:
        for c in dets:
            dets[c] += to_coco_dets(inference_detector(model, clean(i) if c == "clean" else os.path.join(ART, c, f"{i}.png")), i, cat_ids)
    ap = {c: CocoImageEval(coco, dets[c], ids).ap(np.ones(len(ids)))[0] for c in dets}
    res = {}
    for c in CONDS:
        sup = 100 * per_image_transfer(coco, dets["clean"], dets[c], ids)["suppression"]
        res[c] = {"ap_drop": float(100 * (ap["clean"] - ap[c]) / ap["clean"]), "asr": float(np.nanmean(sup))}
    full = res["WB-FULL"]["ap_drop"]
    for c in CONDS:
        res[c]["share_of_full"] = res[c]["ap_drop"] / full if full else None
        print(f"{c:<8} AP drop {res[c]['ap_drop']:5.1f}  ASR {res[c]['asr']:5.1f}  share {res[c]['share_of_full']:.2f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    split = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
    ids = ids[:args.limit] if args.limit else ids
    model = load_any("dino_swin_l")
    tds = target_dataset("dino_swin_l", ids)
    t0 = time.time()
    for i in range(len(tds)):
        smp = tds[i]
        orig = smp["orig_inputs"].cuda()
        mk = masks(coco, smp["img_id"], *orig.shape[1:])
        for c, key in CONDS.items():
            png = os.path.join(ART, c, f"{smp['img_id']}.png")
            if os.path.exists(png):
                continue
            os.makedirs(os.path.dirname(png), exist_ok=True)
            torch.manual_seed(SEED + split.index(smp["img_id"]))
            m = torch.ones_like(orig) if key is None else torch.from_numpy(mk[key]).cuda()[None].float().expand_as(orig)
            adv = to_adv_image(orig, attack(model, orig, smp["data_sample"], m, args.budget, args.epsilon))
            assert float((adv - orig).abs().max()) <= args.epsilon + 1e-4
            assert float(((adv - orig) * (1 - m)).abs().max()) == 0.0
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png)
        print(f"[E51] {i + 1}/{len(tds)} ({time.time() - t0:.0f}s) peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    del model
    torch.cuda.empty_cache()
    res = evaluate(coco, ids)
    if not args.limit:
        json.dump({"budget": args.budget, "n_images": len(ids), "results": res}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
