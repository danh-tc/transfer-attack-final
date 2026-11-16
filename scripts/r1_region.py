"""R1 (progress_log 2026-11-17 00:05, KHÓA): vùng pixel nào quan trọng cho transfer. δ = adv − clean (uint8, BGR),
che theo GT box: keep_R (chỉ giữ δ trong R), rem_R (R trả về clean), rnd_IN (đảo dấu δ trong IN, giữ |δ|).
Quét mỗi method trên target dễ→khó; chỉ số = tỉ lệ object clean-TP bị giết. Forward-only, 10 ảnh.
  python scripts/r1_region.py --n 10
"""
import argparse
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
from attack.evaluation import to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ADV = os.path.join(REPO_ROOT, "artifacts/runs/f1b_r1a_eps5_3090/adv")
ATT = {"RD-CNX@50": ("RD-CNX-B100", "snap50"), "ENS-WP": ("ENS-WP-B36", "final"),
       "OSFD-PUB": ("OSFD-PUB-B50", "final"), "RD-R50": ("RD-R50-B36", "final")}
TARGETS = ["r101", "swin_t", "dino_swin_l"]   # dễ -> khó
CELLS = ["full", "keep_IN", "keep_RING", "keep_FAR", "rem_IN", "rem_RING", "rem_FAR", "rnd_IN"]
OUT = os.path.join(REPO_ROOT, "results/mechanism/r1_region_dev10.json")


def region_masks(anns, h, w):
    m_in = np.zeros((h, w), np.float32)
    m_big = np.zeros((h, w), np.float32)
    for a in anns:
        x, y, bw, bh = a["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(np.ceil(x + bw)), int(np.ceil(y + bh))
        m_in[max(y0, 0):y1, max(x0, 0):x1] = 1.0
        ex, ey = 0.25 * bw, 0.25 * bh
        bx0, by0 = int(x - ex), int(y - ey)
        bx1, by1 = int(np.ceil(x + bw + ex)), int(np.ceil(y + bh + ey))
        m_big[max(by0, 0):by1, max(bx0, 0):bx1] = 1.0
    m_ring = np.clip(m_big - m_in, 0, 1)
    m_far = 1.0 - m_big
    return {"IN": m_in, "RING": m_ring, "FAR": m_far}


def apply_cell(clean, delta, cell, M, rng):
    if cell == "full":
        out = clean.astype(np.int32) + delta
    elif cell.startswith("keep_"):
        out = clean.astype(np.int32) + delta * M[cell[5:]][:, :, None]
    elif cell.startswith("rem_"):
        out = clean.astype(np.int32) + delta * (1.0 - M[cell[4:]])[:, :, None]
    else:  # rnd_IN
        sign = rng.choice([-1, 1], size=delta.shape).astype(np.int32)
        out = clean.astype(np.int32) + (np.abs(delta) * sign) * M["IN"][:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    from mmdet.apis import inference_detector
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]

    # cache: clean array, delta mỗi attack, mask vùng, danh sách GT clean-TP mỗi (target sẽ tính sau)
    clean_arr, deltas, Masks, gts = {}, {}, {}, {}
    for i in ids:
        fn = coco.loadImgs(i)[0]["file_name"]
        c = mmcv.imread(os.path.join(IMGDIR, fn))
        clean_arr[i] = c
        h, w = c.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        gts[i] = anns
        Masks[i] = region_masks(anns, h, w)
        deltas[i] = {}
        for att, (g, sub) in ATT.items():
            adv = mmcv.imread(os.path.join(ADV, g, sub, f"{i}.png"))
            deltas[i][att] = adv.astype(np.int32) - c.astype(np.int32)

    res = {"n": len(ids), "targets": TARGETS, "kill": {}, "n_clean_tp": {}}
    for key in TARGETS:
        model = load(key)
        res["kill"][key] = {}
        # clean fate: object nào là clean-TP (survived trên clean)
        clean_grp = {}
        tp = {}
        with torch.no_grad():
            for i in ids:
                cg = group(to_coco_dets(inference_detector(model, clean_arr[i]), i, cat_ids))
                clean_grp[i] = cg.get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], clean_grp[i])[0] == "survived"]
        n_tp = sum(len(v) for v in tp.values())
        res["n_clean_tp"][key] = n_tp
        for att in ATT:
            rng = np.random.default_rng(42)
            kills = {}
            with torch.no_grad():
                for cell in CELLS:
                    k = 0
                    for i in ids:
                        if not tp[i]:
                            continue
                        img = apply_cell(clean_arr[i], deltas[i][att], cell, Masks[i], rng)
                        g = group(to_coco_dets(inference_detector(model, img), i, cat_ids)).get(i, [])
                        for a in tp[i]:
                            if fate(a["bbox"], a["category_id"], g)[0] != "survived":
                                k += 1
                    kills[cell] = k
            kr = {c: kills[c] / max(n_tp, 1) for c in CELLS}
            share = {c: (kills[c] / kills["full"] if kills["full"] else 0.0) for c in CELLS}
            res["kill"][key][att] = {"kill_rate": kr, "share": share, "kills": kills}
            print(f"{key:12} {att:10} kill% " + " ".join(f"{c.replace('_','')}={kr[c]*100:4.0f}" for c in CELLS), flush=True)
        del model
        torch.cuda.empty_cache()

    # overview: share keep_only mỗi vùng, mỗi method × target
    print("\n=== share keep-only (phần thiệt hại giữ được khi chỉ δ ở vùng đó) ===")
    for att in ATT:
        print(f"\n{att}:")
        print(f"{'target':12} {'IN':>6} {'RING':>6} {'FAR':>6}  | rnd_IN")
        for key in TARGETS:
            s = res["kill"][key][att]["share"]
            print(f"{key:12} {s['keep_IN']:6.2f} {s['keep_RING']:6.2f} {s['keep_FAR']:6.2f}  | {s['rnd_IN']:.2f}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n[r1] -> {OUT}")


if __name__ == "__main__":
    main()
