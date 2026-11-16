"""M0 (progress_log 2026-11-16 22:55, KHÓA): RTMDet-L làm target phân tích trên PNG F1b (dev30), forward-only."""
import gzip
import json
import os
import sys

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
RUN = os.path.join(REPO_ROOT, "artifacts/runs/f1b_r1a_eps5_3090")
CELLS = {"RD-CNX@50": ("RD-CNX-B100", "snap50"), "ENS-WP": ("ENS-WP-B36", "final"),
         "OSFD-PUB": ("OSFD-PUB-B50", "final"), "RD-R50": ("RD-R50-B36", "final")}
REF = ["dino_swin_l", "yolox_l", "r101"]
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py")
CKPT = os.path.join(REPO_ROOT, "checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth")
OUT = os.path.join(REPO_ROOT, "results/mechanism/m0_rtmdet_dev30.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector, init_detector
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), 200, 2026)])
    idset = set(ids)
    dets = {}
    model = init_detector(CFG, CKPT, device="cuda:0")
    d = {c: [] for c in ["clean", *CELLS]}
    with torch.no_grad():
        for i in ids:
            d["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)
            for c, (g, sub) in CELLS.items():
                d[c] += to_coco_dets(inference_detector(model, os.path.join(RUN, "adv", g, sub, f"{i}.png")), i, cat_ids)
    dets["rtmdet_l"] = d
    json.dump(d, open(os.path.join(REPO_ROOT, "results/m0/dets_rtmdet_l_n30.json"), "w"))
    for m in REF:   # dets F1b đã lưu, lọc 30 ảnh
        a = json.load(gzip.open(os.path.join(RUN, "dets", m, "all.json.gz"), "rt"))
        dets[m] = {c: [x for x in a[c] if x["image_id"] in idset] for c in ["clean", *CELLS]}
    S = {}
    for m, d in dets.items():
        ev = CocoImageEval(coco, d["clean"], ids); clean = np.array([ev.ap(w)[0] for w in W])
        for c in CELLS:
            ev = CocoImageEval(coco, d[c], ids)
            S[f"{m}|{c}"] = 100 * (clean - np.array([ev.ap(w)[0] for w in W])) / clean
        print(f"{m:12} clean AP {clean[0] * 100:.1f}  " + "  ".join(f"{c} {S[f'{m}|{c}'][0]:.1f}" for c in CELLS), flush=True)
    ci = lambda v: summarize(v[0], v[1:])
    res = {"drop": {k: ci(v) for k, v in S.items()}, "paired": {}}
    for c in CELLS:
        for ref in ("dino_swin_l", "yolox_l"):
            res["paired"][f"{c} rtmdet−{ref}"] = ci(S[f"rtmdet_l|{c}"] - S[f"{ref}|{c}"])
            p = res["paired"][f"{c} rtmdet−{ref}"]
            print(f"{c:10} RTMDet − {ref:11} {p['point']:+6.1f} [{p['ci95'][0]:+.1f}, {p['ci95'][1]:+.1f}]")
    main3 = ["RD-CNX@50", "ENS-WP", "OSFD-PUB"]
    hard = sum(S[f"rtmdet_l|{c}"][0] <= S[f"dino_swin_l|{c}"][0] + 10 for c in main3)
    easy = sum(S[f"rtmdet_l|{c}"][0] >= S[f"yolox_l|{c}"][0] - 5 for c in main3)
    res["verdict"] = "RTMDet KHÓ" if hard >= 2 else "RTMDet DỄ" if easy >= 2 else "TRUNG GIAN"
    print(f"M0 VERDICT (khóa): {res['verdict']}  (khó {hard}/3, dễ {easy}/3)")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
