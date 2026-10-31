"""E72-RGG (progress_log 2026-10-31): Relational Gradient Graph — reweight gradient theo vùng trên nền RD-CNX, 1 surrogate.
RQ: quan hệ gradient giữa các vùng có cải thiện transfer VƯỢT region-importance độc lập không?
5 biến thể (sur ConvNeXt, B36, ε=5, cùng OCT/WP/momentum/36 backward): node (region importance), spatial (topology không
gradient), grad (quan hệ gradient có dấu), shuffle (grad nhưng hoán vị lân cận), smooth (lowpass ‖g‖ — đối chứng 'graph=làm mượt').
So RD-CNX (E63 PNG) cùng đường eval inference-PNG. DINO-Swin-L HELD-OUT (không tune). Null ở n=30 = vô kết luận.
  python scripts/e72_rgg.py   -> results/mechanism/e72_rgg.json"""
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
RUN = "dev300_B36_eps5_limit100_sur-convnext_t"
OUT = os.path.join(REPO_ROOT, "results/mechanism/e72_rgg.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
RGG = [f"RD-CNX-RGG-{m}" for m in ("node", "spatial", "grad", "shuffle", "smooth")]
CONDS = ["C2-OCT-1V-WP2-RD-CNX"] + RGG   # baseline RD-CNX + 5 biến thể
BOOT_SEED, N_BOOT = 2026, 300  # pilot diagnostic -> 300 (đủ CI, nhanh hơn)


def png(cond, i):
    return os.path.join(ART, RUN, "adv", cond, f"{i}.png")


def main():
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": n, "n_boot": N_BOOT, "drop": {}}
    for key in PANEL:
        if key in res["drop"] and all(c in res["drop"][key] for c in CONDS):
            continue
        model = load(key)
        res["drop"].setdefault(key, {})

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in CONDS:
            if c in res["drop"][key]:
                continue
            adv = ap_of([png(c, i) for i in ids])
            res["drop"][key][c] = summarize(*(lambda a: (a[0], a[1:]))(100 * (clean - adv) / clean))
        del model
        import torch
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)  # ghi SAU MỖI model -> eval resume per-model được
        print(key, {c.replace("RD-CNX-RGG-", "").replace("C2-OCT-1V-WP2-RD-CNX", "base"): round(res["drop"][key][c]["point"], 1) for c in CONDS}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    # Bảng + ablation then chốt
    lbl = {"C2-OCT-1V-WP2-RD-CNX": "base", **{c: c.split("-")[-1] for c in RGG}}
    p = lambda c, m: res["drop"][m][c]["point"]
    t7 = lambda c: sum(p(c, m) for m in T7) / 7
    print("\n" + " " * 8 + " ".join(m[:9].rjust(9) for m in PANEL) + "    T7")
    for c in CONDS:
        print(f"{lbl[c]:8}" + " ".join(f"{p(c, m):9.1f}" for m in PANEL) + f" {t7(c):6.1f}")
    print("\n--- ablation (T7, Swin-L, R101) ---")
    for c in RGG:
        print(f"{lbl[c]:8} T7 {t7(c):.1f}  SwinL {p(c,'dino_swin_l'):.1f}  R101 {p(c,'r101'):.1f}")
    g = "RD-CNX-RGG-grad"
    print(f"\nGRAD − node  T7 {t7(g)-t7('RD-CNX-RGG-node'):+.1f}  SwinL {p(g,'dino_swin_l')-p('RD-CNX-RGG-node','dino_swin_l'):+.1f}")
    print(f"GRAD − smooth T7 {t7(g)-t7('RD-CNX-RGG-smooth'):+.1f}  SwinL {p(g,'dino_swin_l')-p('RD-CNX-RGG-smooth','dino_swin_l'):+.1f}")
    print(f"GRAD − shuffle T7 {t7(g)-t7('RD-CNX-RGG-shuffle'):+.1f}  SwinL {p(g,'dino_swin_l')-p('RD-CNX-RGG-shuffle','dino_swin_l'):+.1f}")
    print(f"GRAD − base   T7 {t7(g)-t7('C2-OCT-1V-WP2-RD-CNX'):+.1f}  SwinL {p(g,'dino_swin_l')-p('C2-OCT-1V-WP2-RD-CNX','dino_swin_l'):+.1f}")


if __name__ == "__main__":
    main()
