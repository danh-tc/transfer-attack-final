#!/usr/bin/env python3
"""E69 (progress_log 2026-10-31): ensemble cross-family 2 surrogate (R50 ⊕ ConvNeXt) CÙNG budget B=36 — giữ cả hai họ?
Cơ chế (E68): ∇R50 trỏ ResNet, ∇ConvNeXt trỏ Swin-L, gần trực giao -> δ qua CẢ HAI surrogate phủ cả hai subspace.
MVP warm-start (không sửa core): 18 bước RD trên surrogate A -> δ1; 18 bước RD trên surrogate B khởi tạo từ δ1 (init_noise).
Tổng 36 backward = cùng B (KHÔNG x2 thời gian; chỉ đổi threat model: 2 surrogate). Hai thứ tự: RC (R50->CNX), CR (CNX->R50).
So RD-CNX (E63 PNG) và RD-R50 (E62 PNG) trên full panel, cùng đường eval (inference trên PNG).
  python scripts/e69_ensemble.py   -> results/mechanism/e69_ensemble.json"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.run_baselines import METHODS
from scripts.e64_cnx_drop import load  # load(key): controlled + gen panel

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
ENSRUN = "dev300_B36_eps5_limit100_ens"
OUT = os.path.join(REPO_ROOT, "results/mechanism/e69_ensemble.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
FIX = {"RD-CNX": ("dev300_B36_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX"),
       "RD-R50": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"),
       "OSFD-R50": ("dev300_B50_eps5_limit100", "OSFD"),
       "OSFD-CNX": ("dev300_B50_eps5_limit100_sur-convnext_t", "OSFD")}
BOOT_SEED, N_BOOT = 2026, 1000


def attack_ensemble(ids):
    """Sinh PNG ENS-RC / ENS-CR (resume: bỏ qua nếu PNG đã có)."""
    need = any(not os.path.exists(os.path.join(ART, ENSRUN, "adv", c, f"{i}.png")) for c in ("ENS-RC", "ENS-CR") for i in ids)
    if not need:
        return
    fn_r50 = METHODS["C2-OCT-1V-WP2-RD"][0]
    fn_cnx = METHODS["C2-OCT-1V-WP2-RD-CNX"][0]
    r50, cnx = load_model("r50", "cuda:0"), load_model("convnext_t", "cuda:0")
    ds_r = {s["img_id"]: s for s in target_dataset("r50", ids)}
    ds_c = {s["img_id"]: s for s in target_dataset("convnext_t", ids)}
    for c in ("ENS-RC", "ENS-CR"):
        os.makedirs(os.path.join(ART, ENSRUN, "adv", c), exist_ok=True)
    for n, i in enumerate(ids):
        orig = ds_r[i]["orig_inputs"].cuda().float()
        # RC: R50 18 bước -> ConvNeXt 18 bước (kết thúc ở ConvNeXt, mang hướng Swin-L)
        d1 = fn_r50(r50, orig, ds_r[i]["data_sample"], steps=18, epsilon=5.0)
        d_rc = fn_cnx(cnx, orig, ds_c[i]["data_sample"], steps=18, epsilon=5.0, init_noise=d1)
        # CR: ConvNeXt 18 -> R50 18 (kết thúc ở R50, mang hướng ResNet)
        d2 = fn_cnx(cnx, orig, ds_c[i]["data_sample"], steps=18, epsilon=5.0)
        d_cr = fn_r50(r50, orig, ds_r[i]["data_sample"], steps=18, epsilon=5.0, init_noise=d2)
        for c, d in (("ENS-RC", d_rc), ("ENS-CR", d_cr)):
            adv = to_adv_image(orig, d)
            assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(ART, ENSRUN, "adv", c, f"{i}.png"))
        print(f"[e69 attack] {n + 1}/{len(ids)}", flush=True)
    del r50, cnx
    torch.cuda.empty_cache()


def png_path(cond, i):
    if cond in FIX:
        run, name = FIX[cond]
        return os.path.join(ART, run, "adv", name, f"{i}.png")
    return os.path.join(ART, ENSRUN, "adv", cond, f"{i}.png")


def main():
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    attack_ensemble(ids)
    conds = ["ENS-RC", "ENS-CR", "RD-CNX", "RD-R50", "OSFD-R50", "OSFD-CNX"]
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": n, "conds": conds, "drop": {m: {} for m in PANEL}, "note": "relative AP drop % dev30, cùng đường eval PNG"}
    res["conds"] = conds
    for key in PANEL:
        res["drop"].setdefault(key, {})
        if all(c in res["drop"][key] for c in conds):
            print(key, "đã đủ, bỏ qua", flush=True); continue
        model = load(key)
        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in conds:
            if c in res["drop"][key]:
                continue
            adv = ap_of([png_path(c, i) for i in ids])
            res["drop"][key][c] = summarize(*(lambda a: (a[0], a[1:]))(100 * (clean - adv) / clean))
        del model
        torch.cuda.empty_cache()
        print(key, {c: round(res["drop"][key][c]["point"], 1) for c in conds}, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
