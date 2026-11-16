"""H0 "match the back-end" (progress_log 2026-11-16 21:15, KHÓA). Surrogate DINO-R50 vs Mask R-CNN R50, CÙNG công thức:
MI-FGSM (μ 1, chuẩn L1) + task loss GT (DINO: số hạng không-dn), ε 5, α 1, B 50, δ0 = 0, seed torch 1000·(n+1).
  python scripts/h0_backend.py --n 30          # sinh (resume theo PNG) + eval + phán quyết
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
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs/h0_backend")
EPS, ALPHA, STEPS, MU, BOOT_SEED = 5.0, 1.0, 50, 1.0, 2026
CELLS = {"DINO-R50": "dino_r50", "MRCNN-R50": "r50"}
PANEL = ["dino_swin_l", "swin_t", "r101", "fcos_r50", "yolox_l", "convnext_t", "r50", "dino_r50"]


def png(cell, i):
    return os.path.join(ART, "adv", cell, f"{i}.png")


def attack(model, key, orig, dsm):
    delta = torch.zeros_like(orig)
    g = torch.zeros_like(orig)
    for _ in range(STEPS):
        x = torch.clamp(orig + delta, 0, 255)
        grad = target_grads(model, key, x, dsm, objectives=("task",))[0]["task"]
        g = MU * g + grad / (grad.abs().mean() + 1e-12)
        delta = torch.clamp(delta + ALPHA * torch.sign(g), -EPS, EPS)
        delta = torch.clamp(orig + delta, 0, 255) - orig
    return delta


def generate(ids):
    timing = {}
    for cell, key in CELLS.items():
        if all(os.path.exists(png(cell, i)) for i in ids):
            print(f"[h0] {cell}: PNG đủ", flush=True)
            continue
        model = load_any(key)
        ds = {s["img_id"]: s for s in target_dataset(key, ids)}
        ts = []
        for n, i in enumerate(ids):
            if os.path.exists(png(cell, i)):
                continue
            torch.manual_seed(1000 * (n + 1))
            orig = ds[i]["orig_inputs"].cuda().float()
            torch.cuda.synchronize(); t0 = time.time()
            delta = attack(model, key, orig, ds[i]["data_sample"])
            torch.cuda.synchronize(); ts.append(time.time() - t0)
            adv = to_adv_image(orig, delta)
            assert float((adv - orig).abs().max()) <= EPS + 1e-4
            os.makedirs(os.path.dirname(png(cell, i)), exist_ok=True)
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png(cell, i))
            print(f"[h0 {cell}] {n + 1}/{len(ids)} {np.mean(ts):.1f}s/ảnh", flush=True)
        timing[cell] = float(np.mean(ts)) if ts else None
        del model
        torch.cuda.empty_cache()
    return timing


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    timing = generate(ids)
    from mmdet.apis import inference_detector
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), args.boot, BOOT_SEED)])
    S = {}
    for key in PANEL:
        cache = os.path.join(ART, f"dets_{key}_n{len(ids)}.json")
        if os.path.exists(cache):
            dets = json.load(open(cache))
        else:
            model = load(key)
            dets = {c: [] for c in ["clean", *CELLS]}
            with torch.no_grad():
                for i in ids:
                    dets["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)
                    for c in CELLS:
                        dets[c] += to_coco_dets(inference_detector(model, png(c, i)), i, cat_ids)
            json.dump(dets, open(cache, "w"))
            del model
            torch.cuda.empty_cache()
        ev = CocoImageEval(coco, dets["clean"], ids)
        clean = np.array([ev.ap(w)[0] for w in W])
        for c in CELLS:
            ev = CocoImageEval(coco, dets[c], ids)
            S[f"{key}|{c}"] = 100 * (clean - np.array([ev.ap(w)[0] for w in W])) / clean
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)
    ci = lambda v: summarize(v[0], v[1:])
    res = {"n": len(ids), "boot": args.boot, "gpu": torch.cuda.get_device_name(0), "timing_s_per_img": timing,
           "drop": {k: ci(v) for k, v in S.items()}}
    d = ci(S["dino_swin_l|DINO-R50"] - S["dino_swin_l|MRCNN-R50"])
    res["delta_be_SwinL"] = d
    if d["point"] >= 15 and d["ci95"][0] > 0:
        v = "H_be ỦNG HỘ (xu hướng) -> khóa xác nhận dev100 + mở encoder-aware objective"
    elif d["ci95"][1] < 5:
        v = "NO-GO -> khép hướng back-end"
    else:
        v = "UNCLEAR"
    res["verdict"] = v
    print(f"\nΔ_be (SwinL, DINO-R50 − MRCNN-R50) = {d['point']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
    print(f"H0 VERDICT (khóa): {v}")
    out = os.path.join(REPO_ROOT, f"results/mechanism/h0_backend_dev{len(ids)}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[h0] -> {out}")


if __name__ == "__main__":
    main()
