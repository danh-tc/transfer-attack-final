#!/usr/bin/env python3
"""Box mồi có chuyển giao không? (progress_log 2026-10-18, hướng A) — mô tả, không GPU.

Với mỗi ô FAB và mỗi model: tỉ lệ box mồi (ảnh gốc, từ attack_stats.jsonl) có detection score ≥ 0.3, IoU ≥ 0.5,
(a) đúng lớp mồi, (b) lớp bất kỳ. Mốc "tình cờ": cùng box mồi đo trên ảnh SẠCH và ảnh của ô gốc C2-OCT-1V.

Chạy: python scripts/fab_check.py --run dev300_B41_eps5_limit100 --subset dev30 --dets dets_fab_dev30.json \
          --cells C2-OCT-1V-FAB4 C2-OCT-1V-FAB8 --base C2-OCT-1V
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids  # noqa: E402
from scripts.probe_error_types import iou_mat  # noqa: E402

MODELS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--dets", required=True)
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--base", default="C2-OCT-1V")
    args = p.parse_args()
    art = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    ids = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
    stats = {}
    for line in open(os.path.join(REPO_ROOT, "results/runs", args.run, "attack_stats.jsonl")):
        r = json.loads(line)
        if r["method"] in args.cells and r["img_id"] in ids:
            stats[(r["method"], r["img_id"])] = r  # bản ghi sau cùng thắng
    ctrl = json.load(open(os.path.join(art, args.dets)))
    gen = {m: json.load(open(os.path.join(art, "gen_dets", f"{m}.json"))) for m in ("yolox_s", "dino_swin_l")}

    def dets_of(cond, m):
        d = gen[m][cond] if m in gen else ctrl[cond][m]
        out = defaultdict(list)
        for x in d:
            if x["score"] >= 0.3:
                out[x["image_id"]].append(x)
        return out

    res = {}
    for cell in args.cells:
        res[cell] = {}
        for m in MODELS:
            row = {}
            for cond in ("clean", args.base, cell):
                dd = dets_of(cond, m)
                hc = ha = n = 0
                for (c, img), r in stats.items():
                    if c != cell:
                        continue
                    fb = np.array(r["fab_boxes_orig"]).reshape(-1, 4)
                    fb_xywh = np.concatenate([fb[:, :2], fb[:, 2:] - fb[:, :2]], 1)
                    lab = np.array([COCO_IDS[v] for v in r["fab_labels"]])
                    ds = dd.get(img, [])
                    d = np.array([x["bbox"] for x in ds]).reshape(-1, 4)
                    dc = np.array([x["category_id"] for x in ds])
                    M = iou_mat(fb_xywh, d)
                    n += len(fb)
                    if len(ds):
                        ha += int((M >= 0.5).any(1).sum())
                        hc += int(((M >= 0.5) & (dc[None] == lab[:, None])).any(1).sum())
                row[cond] = {"đúng lớp": hc / max(1, n), "bất kỳ": ha / max(1, n), "n_mồi": n}
            res[cell][m] = row
    out = os.path.join(REPO_ROOT, "results/mechanism", f"fab_check_{args.subset}.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    for cell in args.cells:
        print(f"\n{cell}: tỉ lệ mồi được phát hiện (đúng lớp | bất kỳ) — sạch / {args.base} / {cell}")
        for m in MODELS:
            r = res[cell][m]
            print(f"  {m:<12}" + "   ".join(f"{r[c]['đúng lớp']:.2f} | {r[c]['bất kỳ']:.2f}" for c in ("clean", args.base, cell)))


if __name__ == "__main__":
    main()
