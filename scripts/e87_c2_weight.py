"""E87 — can thiệp trọng số C2 trong L_OSFD (kiểm chứng CUỐI nhánh cơ chế feature-stage gradient imbalance), devB100.

Giả thuyết hậu kỳ (E86, phát sinh từ E84/E85 — KHÔNG phải giả thuyết đăng ký từ đầu dự án):
  "The disproportionately large shallow-stage gradient contribution in ConvNeXt alters the attack update direction,
   potentially suppressing transferable information from deeper feature stages."
Ô (objective = L_OSFD thuần, geo_osfd_attack OCT 1 view, co-transform, amp, B=36, ε=5 — như ô OCT của E85):
  L = α·L_C2 + L_C3 + L_C4 + L_C5  (feat_stage_w = (α, 1, 1, 1); α=1 ≡ C2–C5, α=0 ≡ C3–C5, unit test e87_unit_test.py)
  CNX α ∈ {1, 0.5, 0};  R50 α ∈ {1, 0} (negative control).
PRIMARY (cổng bằng chứng, T5C): Δ_CNX = Drop(CNX α=0.5) − Drop(CNX α=1); cận dưới CI95 paired > 0 -> hỗ trợ giả thuyết
  "giảm đóng góp C2 có lợi" (KHÔNG tự chứng minh cơ chế sign conflict).
SUPPORTING: I_stage = (CNX α0 − CNX α1) − (R50 α0 − R50 α1) trên T5C, CI trực tiếp.
Mô tả: R = (α0.5 − α1)/(α0 − α1), mọi contrast trên mọi target.
Seed: torch + random + numpy = 1000·(n+1), n = chỉ số trong devB100. Eval: panel 9 model, boot 1000, BOOT_SEED 2026.

  python scripts/e87_c2_weight.py --gen-only --cells CNX-A1,CNX-A05   # 1 worker sinh
  python scripts/e87_c2_weight.py                                       # sinh còn thiếu + eval
  python scripts/e87_c2_weight.py --list smoke3 --boot 20 --tag smoke   # smoke: 3 ảnh đầu x200 (KHÔNG đọc số)
"""
import argparse
import gzip
import json
import os
import random
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
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STEPS, EPS, BOOT_SEED = 36, 5.0, 2026
OCT_KW = dict(geo="elem", elem=_C2OCT["elem"], n_views=1, amp=True)
ALPHAS = {"A1": 1.0, "A05": 0.5, "A0": 0.0}
CELLS = {}                                                   # cell -> (surrogate, kwargs)
for _a in ("A1", "A05", "A0"):
    CELLS[f"CNX-{_a}"] = ("convnext_t", dict(OCT_KW, feat_stage_w=(ALPHAS[_a], 1.0, 1.0, 1.0)))
for _a in ("A1", "A0"):
    CELLS[f"R50-{_a}"] = ("r50", dict(OCT_KW, feat_stage_w=(ALPHAS[_a], 1.0, 1.0, 1.0)))

PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
T5C = ["fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]   # chung (không G0 với surrogate nào)
PRIMARY = "PRIMARY Δ_CNX (α0.5 − α1)"
SUPPORT = "I_stage (CNX α0−α1) − (R50 α0−α1)"
CONTRASTS = {
    PRIMARY: {"CNX-A05": 1, "CNX-A1": -1},
    SUPPORT: {"CNX-A0": 1, "CNX-A1": -1, "R50-A0": -1, "R50-A1": 1},
    "CNX α0 − α1": {"CNX-A0": 1, "CNX-A1": -1},
    "CNX α0 − α0.5": {"CNX-A0": 1, "CNX-A05": -1},
    "R50 α0 − α1": {"R50-A0": 1, "R50-A1": -1},
}


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def adv_dir(run_dir, cell):
    return os.path.join(ART, run_dir, "adv", cell)


def image_ids(name):
    if name == "smoke3":                                     # 3 ảnh đầu x200 — KHÔNG đụng devB100
        d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
        return [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100][:3]
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devB100.csv"))
    assert len(ids) == 100, len(ids)
    return ids


def generate(ids, cell, run_dir):
    sur, kw = CELLS[cell]
    out = adv_dir(run_dir, cell)
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        print(f"[e87] {cell}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    model = load_model(sur, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(sur, ids)}
    t0, done = time.time(), 0
    for n, i in enumerate(ids):
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        orig = ds[i]["orig_inputs"].cuda().float()
        seed_all(1000 * (n + 1))
        delta = geo_osfd_attack(model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS, **kw)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= EPS + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png")
        os.replace(p + ".tmp.png", p)              # ghi nguyên tử
        done += 1
        print(f"[e87 {cell}] {n + 1}/{len(ids)}  {(time.time() - t0) / done:.1f}s/ảnh", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="devB100", choices=["devB100", "smoke3"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cells", default=None, help="danh sách ô, phẩy ngăn (mặc định: tất cả)")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cells = args.cells.split(",") if args.cells else list(CELLS)
    assert all(c in CELLS for c in cells), cells
    gpu = torch.cuda.get_device_name(0)
    mtag = gpu.split()[-1].replace("-", "").lower()
    sfx = f"_{args.tag}" if args.tag else ""
    run_dir = f"e87_c2w_B{STEPS}_eps5_{mtag}{sfx}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e87_c2w_{args.list}{sfx}.json")
    print(f"[e87] GPU = {gpu} | run dir {run_dir} | ô {cells}", flush=True)

    ids = image_ids(args.list)
    for c in cells:
        generate(ids, c, run_dir)
    if args.gen_only:
        return
    assert set(cells) == set(CELLS), "eval cần đủ mọi ô"

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])   # hàng 0 = mẫu gốc
    npz = out_json.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
    meta = {"list": args.list, "n_images": n, "n_boot": args.boot, "boot_seed": BOOT_SEED, "gpu": gpu,
            "steps": STEPS, "eps": EPS, "run_dir": run_dir,
            "cells": {c: {"surrogate": v[0], "kw": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v[1].items()}}
                      for c, v in CELLS.items()}}
    from mmdet.apis import inference_detector
    dets_dir = os.path.join(ART, run_dir, "dets")
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in CELLS):
            continue
        model = load(key)
        os.makedirs(os.path.join(dets_dir, key), exist_ok=True)

        def ap_of(name, paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            with gzip.open(os.path.join(dets_dir, key, f"{name}.json.gz"), "wt") as f:
                json.dump(dets, f)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        clean = ap_of("clean", [os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in CELLS:
            adv = ap_of(c, [os.path.join(adv_dir(run_dir, c), f"{i}.png") for i in ids])
            S[f"{key}|{c}"] = (100 * (clean - adv) / clean).astype(np.float64)
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)

    groups = {m: [m] for m in PANEL}
    groups.update({"T5C": T5C, "T7": T7, "TF": ["swin_t", "dino_swin_l"]})   # T7 không chứa r50 / convnext_t -> = T7\own
    G = {g: {c: np.mean([S[f"{m}|{c}"] for m in ms], axis=0) for c in CELLS} for g, ms in groups.items()}
    res = dict(meta, drop={}, contrast={}, ratio_R={}, groups=groups)
    for g in groups:
        res["drop"][g] = {c: summarize(G[g][c][0], G[g][c][1:]) for c in CELLS}
        res["contrast"][g] = {}
        for name, coef in CONTRASTS.items():
            v = sum(w * G[g][c] for c, w in coef.items())
            res["contrast"][g][name] = summarize(v[0], v[1:])
        num, den = G[g]["CNX-A05"] - G[g]["CNX-A1"], G[g]["CNX-A0"] - G[g]["CNX-A1"]
        r = num / np.where(np.abs(den) < 1e-9, np.nan, den)
        rb = r[1:][np.isfinite(r[1:])]
        res["ratio_R"][g] = {"point": float(r[0]), "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                             "note": "mô tả; chỉ đọc khi CI(CNX α0 − α1) loại 0"}
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    show = PANEL + ["T7", "T5C"]
    print(f"\n=== E87 @ {args.list}, {gpu}, B={STEPS}, boot {args.boot} — relative AP drop % ===")
    print(f"{'cell':9}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in CELLS:
        print(f"{c:9}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    print("\n--- contrast (mọi target) ---")
    for name in CONTRASTS:
        print(f"{name:38}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][name])}" for g in ["T5C", "dino_swin_l", "T7", "TF"]))
    for name in CONTRASTS:
        print(f"  {name}: " + "  ".join(f"{m} {fmt(res['contrast'][m][name])}" for m in PANEL))
    p = res["contrast"]["T5C"][PRIMARY]
    rr = res["ratio_R"]["T5C"]
    print(f"\nR (T5C) = {rr['point']:+.2f} [{rr['ci95'][0]:+.2f}, {rr['ci95'][1]:+.2f}]  (mô tả)")
    print(f"PRIMARY T5C: {fmt(p)} -> " + ("CẬN DƯỚI > 0: HỖ TRỢ giả thuyết" if p["ci95"][0] > 0 else "CI KHÔNG loại 0: KHÔNG hỗ trợ"))
    res["verdict"] = {"primary_T5C": p, "supported": bool(p["ci95"][0] > 0)}
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[e87] -> {out_json}")


if __name__ == "__main__":
    main()
