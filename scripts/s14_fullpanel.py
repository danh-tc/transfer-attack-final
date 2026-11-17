"""S14 (progress_log 2026-11-17): TOAN PANEL dev30, 1 metric nhat quan (relative AP drop % + kill %), delta DONG BANG.
Gop panel goc + controlled + modern. delta: RD-ALL (RD-CNX), OSFD-FEAT (OSFD baseline), R50-ALL (RD tren R50).
Eval-only tren PNG co san (dev30). Target R50/surrogate -> white-box cho R50-ALL (danh dau WB).
  python scripts/s14_fullpanel.py
"""
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
from attack.evaluation import CocoImageEval, to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.s3_region import ADV, ANN, IMGDIR
from scripts.s13_modern import TARGETS as MODERN

# (ten, 'e64' | duong_dan_cfg) ; thu tu trinh bay
E64 = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
DELTAS = ["RD-ALL", "OSFD-FEAT", "R50-ALL"]
SUR_WB = {"R50-ALL": {"r50"}, "RD-ALL": {"convnext_t"}, "OSFD-FEAT": set()}  # surrogate -> white-box, chu thich


def eval_target(model, inf, ids, coco, cat, gts, clean):
    with torch.no_grad():
        cl, tp = [], {}
        for i in ids:
            det = to_coco_dets(inf(model, clean[i]), i, cat); cl += det
            g = group(det).get(i, [])
            tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
        apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]; ntp = sum(len(v) for v in tp.values())
        out = {"clean_ap": apc * 100, "n_tp": ntp, "drop": {}, "kill": {}}
        for d in DELTAS:
            ad, k = [], 0
            for i in ids:
                det = to_coco_dets(inf(model, mmcv.imread(os.path.join(ADV, d, f"{i}.png"))), i, cat); ad += det
                g = group(det).get(i, [])
                k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
            apa = CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]
            out["drop"][d] = 100 * (apc - apa) / max(apc, 1e-9); out["kill"][d] = 100 * k / max(ntp, 1)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector, init_detector
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])) for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    res = {}
    for t in E64:
        m = load(t); res[t] = eval_target(m, inference_detector, ids, coco, cat, gts, clean)
        print(f"[s14] {t:16} " + " ".join(f"{d} {res[t]['drop'][d]:.0f}/{res[t]['kill'][d]:.0f}" for d in DELTAS), flush=True)
        del m; torch.cuda.empty_cache()
    for t, (cfg, ck) in MODERN.items():
        if not os.path.exists(ck):
            continue
        m = init_detector(cfg, ck, device="cuda:0"); res[t] = eval_target(m, inference_detector, ids, coco, cat, gts, clean)
        print(f"[s14] {t:16} " + " ".join(f"{d} {res[t]['drop'][d]:.0f}/{res[t]['kill'][d]:.0f}" for d in DELTAS), flush=True)
        del m; torch.cuda.empty_cache()

    order = E64 + [t for t in MODERN if t in res]
    print(f"\n=== TOAN PANEL dev30 — relative AP drop % (kill %) — delta dong bang ===")
    print(f"{'target':18}{'cleanAP':>8}" + "".join(f"{d:>14}" for d in DELTAS))
    for t in order:
        r = res[t]
        row = f"{t:18}{r['clean_ap']:8.1f}"
        for d in DELTAS:
            wb = " *" if t in SUR_WB[d] else "  "
            row += f"{r['drop'][d]:7.0f}({r['kill'][d]:3.0f}){wb}"
        print(row)
    print("  (* = white-box: target trùng surrogate của δ đó)")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s14_fullpanel_dev30.json"), "w"), indent=1, ensure_ascii=False)
    print("[s14] -> results/mechanism/s14_fullpanel_dev30.json")


if __name__ == "__main__":
    main()
