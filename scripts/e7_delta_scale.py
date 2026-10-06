#!/usr/bin/env python3
"""E7 (quy tắc khóa: progress_log 2026-10-06): phần δ tác dụng lên object LỚN của Swin-L là cấu trúc quy mô lớn hay cục bộ?
Biến đổi δ = PNG adv − ảnh gốc trong từng GT box bằng ĐÚNG phép của E6 (BLUR_δ | SHUF_δ), ngoài box giữ nguyên. Chỉ forward.
Ảnh sạch dùng dets của E6 (artifacts/e6/dets.json, cùng inference_detector); δ nguyên predict lại cùng script (cùng đường đọc).
  python scripts/e7_delta_scale.py   -> results/mechanism/e7_delta_scale.json
"""
import argparse
import json
import os
import sys
import time

import mmcv
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e4_target_side import SIZES, matched
from scripts.e6_shape_texture import CONDS, IMG_DIR, OTHERS, PAIRS, TARGETS, corrupt

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
ADV = os.path.join(REPO_ROOT, "artifacts/runs/n300_B50_eps5/adv")
OUT = os.path.join(REPO_ROOT, "results/mechanism")
DETS = os.path.join(REPO_ROOT, "artifacts/e7/dets.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def run(args, coco, ids):
    from mmdet.apis import inference_detector
    from attack.targets import load_any
    import torch
    cat_ids = sorted(coco.getCatIds())
    conds = [f"{m}|{c}" for m in args.methods for c in ["ID"] + list(CONDS)]
    dets = {t: {c: [] for c in conds} for t in TARGETS}
    for t in TARGETS:
        model = load_any(t)
        t0 = time.time()
        for n, i in enumerate(ids):
            x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
            for m in args.methods:
                d = mmcv.imread(os.path.join(ADV, m, f"{i}.png")).astype(np.float32) - x
                ds = {"ID": d}
                for c, (kind, p) in CONDS.items():
                    ds[c] = corrupt(d, anns, kind, p, np.random.default_rng(2026 + i))
                for c, dd in ds.items():
                    im = np.clip(np.round(x + dd), 0, 255).astype(np.uint8)
                    dets[t][f"{m}|{c}"] += to_coco_dets(inference_detector(model, im), i, cat_ids)
            if (n + 1) % 100 == 0:
                print(f"[E7 {t}] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump(dets, f)


def analyze(args, coco, ids):
    dets = json.load(open(DETS))
    clean_all = json.load(open(os.path.join(REPO_ROOT, "artifacts/e6/dets.json")))
    pos = {i: k for k, i in enumerate(ids)}
    area = {a["id"]: a["area"] for a in coco.loadAnns(coco.getAnnIds(imgIds=ids, iscrowd=False))}
    zone = lambda ar: next(n for n, lo, hi in SIZES if lo <= ar < hi)
    zs = [n for n, _, _ in SIZES]
    rng = np.random.default_rng(BOOT_SEED)
    W = np.vstack([np.ones((1, len(ids))),
                   [np.bincount(r, minlength=len(ids)) for r in rng.integers(0, len(ids), (N_BOOT, len(ids)))]])
    A = {}  # (t, cond, z) -> [1 + N_BOOT] sức attack 1 − retained
    for t in TARGETS:
        cl = matched(coco, clean_all[t]["clean"], ids)
        for cond in dets[t]:
            cc = matched(coco, dets[t][cond], ids)
            rec = [(pos[i], zone(area[a]), sa / sc) for i in ids
                   for a, sc, sa in zip(cl[i][0], cl[i][1], cc[i][1]) if sc >= SCORE]
            for z in zs:
                r = [x for x in rec if x[1] == z]
                img = np.array([x[0] for x in r]); y = np.array([x[2] for x in r]); w = W[:, img]
                A[(t, cond, z)] = 1 - (w * y).sum(1) / np.maximum(w.sum(1), 1e-9)
    out = {"n_images": len(ids), "A": {}, "F": {}, "I": {}}
    F = lambda t, m, c, z: A[(t, f"{m}|{c}", z)] / A[(t, f"{m}|ID", z)]
    for m in args.methods:
        out["A"][m] = {t: {c: {z: float(A[(t, f"{m}|{c}", z)][0]) for z in zs} for c in ["ID"] + list(CONDS)} for t in TARGETS}
        out["F"][m] = {t: {c: {z: float(F(t, m, c, z)[0]) for z in zs} for c in CONDS} for t in TARGETS}
        out["I"][m] = {}
        for b, s in PAIRS:
            out["I"][m][f"{b}|{s}"] = {}
            for z in zs:
                gap = lambda c: F("dino_swin_l", m, c, z) - np.mean([F(t, m, c, z) for t in OTHERS], 0)
                I = gap(b) - gap(s)
                out["I"][m][f"{b}|{s}"][z] = {"point": float(I[0]), "ci95": [float(x) for x in np.nanpercentile(I[1:], [2.5, 97.5])]}
    ok = all(out["I"]["OSFD"][f"{b}|{s}"]["large"]["ci95"][0] > 0 and
             out["I"]["OSFD"][f"{b}|{s}"]["large"]["point"] > out["I"]["OSFD"][f"{b}|{s}"]["small"]["point"] for b, s in PAIRS)
    out["supported_OSFD"] = bool(ok)
    with open(os.path.join(OUT, "e7_delta_scale.json"), "w") as f:
        json.dump(out, f, indent=2)
    for m in args.methods:
        print(f"\n=== {m}: sức attack A (nhỏ/vừa/lớn) δ nguyên | F = phần còn lại sau BLUR05 / SHUF6 (lớn) ===")
        for t in TARGETS:
            a = out["A"][m][t]["ID"]
            print(f"{t:<12} A={a['small']:.2f}/{a['medium']:.2f}/{a['large']:.2f}   " + "  ".join(
                f"F[{c}] lớn={out['F'][m][t][c]['large']:.2f}" for c in CONDS))
        for k, v in out["I"][m].items():
            print(f"I_δ {k}: " + "  ".join(f"{z}: {x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for z, x in v.items()))
    print("\nỦNG HỘ (OSFD):", ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["OSFD", "M-DI2-FGSM"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/n300.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(args, coco, ids)
    analyze(args, coco, ids)


if __name__ == "__main__":
    main()
