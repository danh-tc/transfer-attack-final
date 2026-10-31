"""E73 (progress_log 2026-10-31): XÁC NHẬN dev100 — ENS-WP vs RD-CNX vs RD-R50 vs OSFD-R50, panel 9 model, boot 300.
Sinh ENS-WP (ensemble momentum-chung khối-2, tái dùng e71.attack) trên dev100; RD-CNX/RD-R50/OSFD sinh bằng run_baselines
(run.sh). Eval cùng đường inference-PNG, incremental save (resume per-model). DINO-Swin-L trong panel (không held-out ở đây —
là target báo cáo). n=100 co CI so dev30.
  python scripts/e73_dev100.py   -> results/mechanism/e73_dev100.json"""
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
from scripts.e64_cnx_drop import load
from scripts.e71_ensemble_wp import attack as ens_attack, KW_WP

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e73_dev100.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
# cond -> (run dir, tên thư mục adv)
FIX = {"OSFD-R50": ("dev300_B50_eps5_limit100", "OSFD"),
       "RD-R50": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"),
       "RD-CNX": ("dev300_B36_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX"),
       "ENS-WP": ("dev300_B36_eps5_limit100_ens", "ENS-WP")}
CONDS = ["OSFD-R50", "RD-R50", "RD-CNX", "ENS-WP"]
BOOT_SEED, N_BOOT = 2026, 300


def png(cond, i):
    run, folder = FIX[cond]
    return os.path.join(ART, run, "adv", folder, f"{i}.png")


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100
    ens_attack(ids, "ENS-WP", KW_WP)  # sinh ENS-WP dev100 (resume: bỏ qua 30 ảnh dev30 đã có)
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
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)  # incremental -> resume per-model
        print(key, {c: round(res["drop"][key][c]["point"], 1) for c in CONDS}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    p = lambda c, m: res["drop"][m][c]["point"]
    t7 = lambda c: sum(p(c, m) for m in T7) / 7
    print("\n" + " " * 9 + " ".join(m[:9].rjust(9) for m in PANEL) + "    T7")
    for c in CONDS:
        print(f"{c:9}" + " ".join(f"{p(c, m):9.1f}" for m in PANEL) + f" {t7(c):6.1f}")
    print("\n--- so ENS-WP ---")
    for c in ("RD-R50", "RD-CNX", "OSFD-R50"):
        print(f"ENS-WP − {c:9} T7 {t7('ENS-WP')-t7(c):+.1f}  SwinL {p('ENS-WP','dino_swin_l')-p(c,'dino_swin_l'):+.1f}  R101 {p('ENS-WP','r101')-p(c,'r101'):+.1f}")
    for c in CONDS:
        ci = res["drop"]["dino_swin_l"][c]["ci95"]
        print(f"{c:9} Swin-L {p(c,'dino_swin_l'):.1f} CI[{ci[0]:.1f},{ci[1]:.1f}]")


if __name__ == "__main__":
    main()
