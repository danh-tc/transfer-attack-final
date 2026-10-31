#!/usr/bin/env python3
"""E68 (progress_log 2026-10-31): cầu ConvNeXt→Swin-L nằm ở ĐƯỜNG GRADIENT? (E67 bác tương đồng feature).
Trên ảnh SẠCH dev30, gradient loss cls (thành phần bắc cầu chính, E66) theo ẢNH GỐC cho 5 model (attack.targets.target_grads,
khung ảnh gốc chung) -> cosine + đồng thuận dấu từng cặp. Trục chính: cos(∇ConvNeXt, ∇Swin-L) vs cos(∇R50, ∇Swin-L).
Nếu ∇ConvNeXt gần ∇Swin-L hơn ∇R50 -> cầu ở gradient (dù feature ConvNeXt giống ResNet, E67).
  python scripts/e68_grad_align.py   -> results/mechanism/e68_grad_align.json"""
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.targets import load_any, target_dataset, target_grads

OUT = os.path.join(REPO_ROOT, "results/mechanism/e68_grad_align.json")
MODELS = ["r50", "r101", "convnext_t", "swin_t", "dino_swin_l"]
BOOT_SEED, N_BOOT = 2026, 1000


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def sign_agree(a, b):
    a, b = a.flatten(), b.flatten()
    m = (a != 0) & (b != 0)
    return float((torch.sign(a[m]) == torch.sign(b[m])).float().mean())


def main():
    dev = "cuda:0"
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    # dataset riêng từng model (GT co-transform đúng khung model); ảnh gốc chung lấy từ dataset đầu.
    ds_by = {m: {s["img_id"]: s for s in target_dataset(m, ids)} for m in MODELS}
    pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
    cosv = {f"{a}|{b}": [] for a, b in pairs}
    sgnv = {f"{a}|{b}": [] for a, b in pairs}
    for n, iid in enumerate(ids):
        g = {}
        for m in MODELS:
            model = load_any(m, dev)
            s = ds_by[m][iid]
            orig = s["orig_inputs"].to(dev).float()
            grads, _, _ = target_grads(model, m, orig, s["data_sample"], objectives=("cls",))
            g[m] = grads["cls"].cpu()
            del model
            torch.cuda.empty_cache()
        for a, b in pairs:
            cosv[f"{a}|{b}"].append(cos(g[a], g[b]))
            sgnv[f"{a}|{b}"].append(sign_agree(g[a], g[b]))
        print(f"[e68] {n + 1}/30", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, 30, size=(N_BOOT, 30))

    def summ(v):
        v = np.array(v)
        bs = v[boot].mean(1)
        return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    def paired(p1, p2):  # hiệu v[p1]-v[p2] theo ảnh, paired bootstrap
        d = np.array(cosv[p1]) - np.array(cosv[p2])
        bs = d[boot].mean(1)
        return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    def key(a, b):
        return f"{a}|{b}" if f"{a}|{b}" in cosv else f"{b}|{a}"

    res = {"n_images": 30, "objective": "cls", "note": "cosine/sign-agree gradient loss cls theo ảnh gốc; [point, lo, hi]",
           "cos": {p: summ(cosv[p]) for p in cosv}, "sign_agree": {p: summ(sgnv[p]) for p in sgnv}}
    res["trucchinh_dino_swin_l"] = {
        "cos_convnext_vs_swinl": summ(cosv[key("convnext_t", "dino_swin_l")]),
        "cos_r50_vs_swinl": summ(cosv[key("r50", "dino_swin_l")]),
        "delta_convnext_minus_r50": paired(key("convnext_t", "dino_swin_l"), key("r50", "dino_swin_l")),
    }
    res["phu_swin_t"] = {
        "cos_convnext_vs_swint": summ(cosv[key("convnext_t", "swin_t")]),
        "cos_r50_vs_swint": summ(cosv[key("r50", "swin_t")]),
        "delta_convnext_minus_r50": paired(key("convnext_t", "swin_t"), key("r50", "swin_t")),
    }
    res["cos_r50_convnext"] = summ(cosv[key("r50", "convnext_t")])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("trucchinh_dino_swin_l", "phu_swin_t", "cos_r50_convnext")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
