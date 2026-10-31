#!/usr/bin/env python3
"""E70 (progress_log 2026-10-31): ensemble momentum-CHUNG thật, luân phiên R50/ConvNeXt mỗi bước, cùng δ + cùng momentum MI.
Khác E69 (warm-start nối tiếp, reset momentum): ở đây 1 momentum xuyên 36 bước -> hai hướng gần trực giao CỘNG DỒN vào cùng
vector, mỗi bước sign phản ánh cả hai subspace; đối xứng thứ tự. Cùng B=36, 1 backward/bước (KHÔNG x2). Loss RD-NOWP
(WP trễ-1-bước xung đột luân phiên -> bỏ WP cho bản ensemble này). Dùng core.run_iterative_attack 1-step + mom_in/return_state.
So ENS-SM với ENS-RC (E69), RD-CNX, RD-R50, OSFD-R50 (số E69, cùng đường eval inference-PNG).
  python scripts/e70_ensemble_sm.py   -> results/mechanism/e70_ensemble_sm.json"""
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
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.run_baselines import _C2OCT
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
RUN = "dev300_B36_eps5_limit100_ens"
OUT = os.path.join(REPO_ROOT, "results/mechanism/e70_ensemble_sm.json")
E69 = os.path.join(REPO_ROOT, "results/mechanism/e69_ensemble.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
COND = "ENS-SM"
BOOT_SEED, N_BOOT, STEPS = 2026, 1000, 36
# RD-NOWP kwargs mỗi surrogate (= RD trừ weight_rho): OCT + L_OSFD C3-C5 + λ·cls RoI.
KW = {"r50": dict(_C2OCT, cls_weight=6.0), "convnext_t": dict(_C2OCT, cls_weight=0.2)}
COMMON = dict(cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))


def attack(ids):
    if all(os.path.exists(os.path.join(ART, RUN, "adv", COND, f"{i}.png")) for i in ids):
        return
    os.makedirs(os.path.join(ART, RUN, "adv", COND), exist_ok=True)
    models = {"r50": load_model("r50", "cuda:0"), "convnext_t": load_model("convnext_t", "cuda:0")}
    ds = {"r50": {s["img_id"]: s for s in target_dataset("r50", ids)},
          "convnext_t": {s["img_id"]: s for s in target_dataset("convnext_t", ids)}}
    order = ["r50", "convnext_t"]
    for n, i in enumerate(ids):
        orig = ds["r50"][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        for step in range(STEPS):
            key = order[step % 2]
            torch.manual_seed(1000 * (n + 1) + step)  # view OCT đa dạng theo bước + tái lập
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=1,
                                         init_noise=delta, mom_in=mom, return_state=True, **KW[key], **COMMON)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(ART, RUN, "adv", COND, f"{i}.png"))
        print(f"[e70 attack] {n + 1}/{len(ids)}", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    attack(ids)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": n, "drop": {}}
    for key in PANEL:
        if key in res["drop"]:
            continue
        model = load(key)

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        adv = ap_of([os.path.join(ART, RUN, "adv", COND, f"{i}.png") for i in ids])
        res["drop"][key] = summarize(*(lambda a: (a[0], a[1:]))(100 * (clean - adv) / clean))
        del model
        torch.cuda.empty_cache()
        print(key, round(res["drop"][key]["point"], 1), flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    # Bảng so sánh với E69
    e69 = json.load(open(E69))["drop"]
    T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
    conds = ["OSFD-R50", "RD-R50", "RD-CNX", "ENS-RC", "ENS-SM"]
    get = lambda c, m: res["drop"][m]["point"] if c == "ENS-SM" else e69[m][c]["point"]
    print("\n" + " " * 9 + " ".join(m[:9].rjust(9) for m in PANEL) + "    T7")
    for c in conds:
        print(f"{c:9}" + " ".join(f"{get(c, m):9.1f}" for m in PANEL) + f" {sum(get(c, m) for m in T7) / 7:6.1f}")


if __name__ == "__main__":
    main()
