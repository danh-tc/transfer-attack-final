#!/usr/bin/env python3
"""E6 (quy tắc khóa: progress_log 2026-10-06): target nhận object bằng HÌNH DẠNG hay CẤU TRÚC CỤC BỘ? Ảnh sạch n300,
làm hỏng trong từng GT box: BLUR (σ = f·√(wh), giữ hình dạng) | SHUF (hoán vị g × g ô, giữ cục bộ). Chỉ forward.
  python scripts/e6_shape_texture.py            -> results/mechanism/e6_shape_texture.json
  python scripts/e6_shape_texture.py --analyze-only
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import mmcv
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e4_target_side import SIZES, matched

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
OUT = os.path.join(REPO_ROOT, "results/mechanism")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3
TARGETS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
OTHERS = ["convnext_t", "swin_t", "yolox_s"]
CONDS = {"BLUR02": ("blur", 0.02), "BLUR05": ("blur", 0.05), "SHUF3": ("shuf", 3), "SHUF6": ("shuf", 6)}
PAIRS = [("BLUR02", "SHUF3"), ("BLUR05", "SHUF6")]


def box_int(a, H, W):
    x, y, w, h = a["bbox"]
    return int(max(0, np.floor(x))), int(max(0, np.floor(y))), int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))


def corrupt(img, anns, kind, p, rng):
    out = img.copy()
    H, W = img.shape[:2]
    for a in sorted(anns, key=lambda a: -a["area"]):  # box lớn trước -> nhỏ sau
        x0, y0, x1, y1 = box_int(a, H, W)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patch = out[y0:y1, x0:x1]
        if kind == "blur":
            s = max(0.5, p * np.sqrt(a["bbox"][2] * a["bbox"][3]))
            out[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), s)
        else:
            g = int(p)
            ys = np.linspace(0, y1 - y0, g + 1).astype(int); xs = np.linspace(0, x1 - x0, g + 1).astype(int)
            # chỉ hoán vị giữa các ô CÙNG kích thước (dùng lưới đều, cắt phần dư)
            th, tw = (y1 - y0) // g, (x1 - x0) // g
            if th < 1 or tw < 1:
                continue
            tiles = [patch[i * th:(i + 1) * th, j * tw:(j + 1) * tw].copy() for i in range(g) for j in range(g)]
            perm = rng.permutation(len(tiles))
            for k, src in enumerate(perm):
                i, j = divmod(k, g)
                out[y0 + i * th:y0 + (i + 1) * th, x0 + j * tw:x0 + (j + 1) * tw] = tiles[src]
    return out


def run(args, coco, ids):
    from mmdet.apis import inference_detector
    from attack.targets import load_any
    cat_ids = sorted(coco.getCatIds())
    dets = {t: {c: [] for c in ["clean"] + list(CONDS)} for t in TARGETS}
    for t in TARGETS:
        model = load_any(t)
        t0 = time.time()
        for n, i in enumerate(ids):
            img = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"]))
            anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
            imgs = {"clean": img}
            for c, (kind, p) in CONDS.items():
                imgs[c] = corrupt(img, anns, kind, p, np.random.default_rng(2026 + i))
            for c, im in imgs.items():
                dets[t][c] += to_coco_dets(inference_detector(model, im), i, cat_ids)
            if (n + 1) % 100 == 0:
                print(f"[E6 {t}] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(REPO_ROOT, "artifacts/e6"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "artifacts/e6/dets.json"), "w") as f:
        json.dump(dets, f)


def analyze(args, coco, ids):
    dets = json.load(open(os.path.join(REPO_ROOT, "artifacts/e6/dets.json")))
    pos = {i: k for k, i in enumerate(ids)}
    area = {a["id"]: a["area"] for a in coco.loadAnns(coco.getAnnIds(imgIds=ids, iscrowd=False))}
    zone = lambda ar: next(n for n, lo, hi in SIZES if lo <= ar < hi)
    # per (t, c): list (img_pos, size, retained)
    rec = {}
    for t in TARGETS:
        cl = matched(coco, dets[t]["clean"], ids)
        for c in CONDS:
            cc = matched(coco, dets[t][c], ids)
            rec[(t, c)] = [(pos[i], zone(area[a]), sa / sc) for i in ids
                           for a, sc, sa in zip(cl[i][0], cl[i][1], cc[i][1]) if sc >= SCORE]
    rng = np.random.default_rng(BOOT_SEED)
    W = np.vstack([np.ones((1, len(ids))), [np.bincount(r, minlength=len(ids)) for r in rng.integers(0, len(ids), (N_BOOT, len(ids)))]])

    def R(t, c, z):  # [1 + N_BOOT] mean retained có trọng số ảnh
        r = [x for x in rec[(t, c)] if x[1] == z]
        img = np.array([x[0] for x in r]); y = np.array([x[2] for x in r])
        w = W[:, img]
        return (w * y).sum(1) / np.maximum(w.sum(1), 1e-9)

    out = {"n_images": len(ids), "R": {}, "I": {}}
    zs = [n for n, _, _ in SIZES]
    for t in TARGETS:
        out["R"][t] = {c: {z: float(R(t, c, z)[0]) for z in zs} for c in CONDS}
        out["R"][t]["n_gt"] = {z: sum(1 for x in rec[(t, "BLUR02")] if x[1] == z) for z in zs}
    ok = True
    for b, s in PAIRS:
        out["I"][f"{b}|{s}"] = {}
        for z in zs:
            gap = lambda c: R("dino_swin_l", c, z) - np.mean([R(t, c, z) for t in OTHERS], 0)
            I = gap(b) - gap(s)
            out["I"][f"{b}|{s}"][z] = {"point": float(I[0]), "ci95": [float(x) for x in np.percentile(I[1:], [2.5, 97.5])]}
        L, S = out["I"][f"{b}|{s}"]["large"], out["I"][f"{b}|{s}"]["small"]
        ok &= L["ci95"][0] > 0 and L["point"] > S["point"]
    out["shape_hypothesis_supported"] = bool(ok)
    with open(os.path.join(OUT, "e6_shape_texture.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=== E6 mean retained (small / medium / large) ===")
    for t in TARGETS:
        print(f"{t:<12} " + "  ".join(f"{c}: " + "/".join(f"{out['R'][t][c][z]:.2f}" for z in zs) for c in CONDS)
              + f"   nGT {out['R'][t]['n_gt']}")
    for k, v in out["I"].items():
        print(f"I {k}: " + "  ".join(f"{z}: {x['point']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for z, x in v.items()))
    print("ỦNG HỘ giả thuyết hình dạng:", ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="n300")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.split}.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(args, coco, ids)
    analyze(args, coco, ids)


if __name__ == "__main__":
    main()
