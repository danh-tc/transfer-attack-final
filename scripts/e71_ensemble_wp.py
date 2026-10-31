"""E71 (progress_log 2026-10-31): WP-trong-ensemble — luân phiên R50/ConvNeXt theo KHỐI 2 bước, momentum CHUNG, cùng B=36.
Trong mỗi khối 2 bước cùng surrogate, WP trễ-1-bước của RD chạy đúng (bước 0 set θ', bước 1 forward trên θ'); momentum + δ
mang xuyên khối qua mom_in/return_state. Tái dùng toàn bộ code WP, không logic WP mới. Mục tiêu: lấy lại ~+7 Swin-L mà ENS-SM
(E70, bỏ WP) thiếu so RD-CNX, giữ nguyên ResNet, cùng B.
Hai ô: ENS-WP (khối-2 + WP ρ0.002, weight_linear cho CNX), ENS-B2 (khối-2 KHÔNG WP — đối chứng tách 'khối' khỏi 'WP').
So E70 ENS-SM (strict-1, noWP) + mốc E69. Eval inference-PNG, 9 model panel.
  python scripts/e71_ensemble_wp.py   -> results/mechanism/e71_ensemble_wp.json"""
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
OUT = os.path.join(REPO_ROOT, "results/mechanism/e71_ensemble_wp.json")
E69 = os.path.join(REPO_ROOT, "results/mechanism/e69_ensemble.json")
E70 = os.path.join(REPO_ROOT, "results/mechanism/e70_ensemble_sm.json")
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
BLOCK, STEPS, BOOT_SEED, N_BOOT = 2, 36, 2026, 1000
COMMON = dict(cls_mode="roi", stage_w=(0.0, 1.0, 1.0, 1.0))
# ENS-WP: RD-NOWP + WP (ρ0.002; weight_linear cho ConvNeXt vì 93% trọng số là Linear). ENS-B2: không WP.
KW_WP = {"r50": dict(_C2OCT, cls_weight=6.0, weight_rho=0.002),
         "convnext_t": dict(_C2OCT, cls_weight=0.2, weight_rho=0.002, weight_linear=True)}
KW_B2 = {"r50": dict(_C2OCT, cls_weight=6.0), "convnext_t": dict(_C2OCT, cls_weight=0.2)}


def attack(ids, cond, kw):
    if all(os.path.exists(os.path.join(ART, RUN, "adv", cond, f"{i}.png")) for i in ids):
        return
    os.makedirs(os.path.join(ART, RUN, "adv", cond), exist_ok=True)
    models = {"r50": load_model("r50", "cuda:0"), "convnext_t": load_model("convnext_t", "cuda:0")}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in models}
    order = ["r50", "convnext_t"]
    nblocks = STEPS // BLOCK
    for n, i in enumerate(ids):
        orig = ds["r50"][i]["orig_inputs"].cuda().float()
        delta, mom = None, None
        for b in range(nblocks):
            key = order[b % 2]
            torch.manual_seed(1000 * (n + 1) + b)
            delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=BLOCK,
                                         init_noise=delta, mom_in=mom, return_state=True, **kw[key], **COMMON)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= 5.0 + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(ART, RUN, "adv", cond, f"{i}.png"))
        print(f"[e71 {cond}] {n + 1}/{len(ids)}", flush=True)
    del models
    torch.cuda.empty_cache()


def main():
    coco = COCO(ANN)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    attack(ids, "ENS-WP", KW_WP)
    attack(ids, "ENS-B2", KW_B2)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    from mmdet.apis import inference_detector
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n_images": n, "drop": {}}
    for key in PANEL:
        if key in res["drop"] and all(c in res["drop"][key] for c in ("ENS-WP", "ENS-B2")):
            continue
        model = load(key)
        res["drop"].setdefault(key, {})

        def ap_of(paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            return np.array([CocoImageEval(coco, dets, ids).ap(w) for w in W])[:, 0] * 100
        clean = ap_of([os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in ("ENS-WP", "ENS-B2"):
            if c in res["drop"][key]:
                continue
            adv = ap_of([os.path.join(ART, RUN, "adv", c, f"{i}.png") for i in ids])
            res["drop"][key][c] = summarize(*(lambda a: (a[0], a[1:]))(100 * (clean - adv) / clean))
        del model
        torch.cuda.empty_cache()
        print(key, {c: round(res["drop"][key][c]["point"], 1) for c in ("ENS-WP", "ENS-B2")}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    e69 = json.load(open(E69))["drop"]
    e70 = json.load(open(E70))["drop"]
    T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]

    def get(c, m):
        if c in ("ENS-WP", "ENS-B2"):
            return res["drop"][m][c]["point"]
        if c == "ENS-SM":
            return e70[m]["point"]
        return e69[m][c]["point"]
    conds = ["OSFD-R50", "RD-R50", "RD-CNX", "ENS-SM", "ENS-B2", "ENS-WP"]
    print("\n" + " " * 9 + " ".join(m[:9].rjust(9) for m in PANEL) + "    T7")
    for c in conds:
        print(f"{c:9}" + " ".join(f"{get(c, m):9.1f}" for m in PANEL) + f" {sum(get(c, m) for m in T7) / 7:6.1f}")
    print("ENS-WP Swin-L CI", [round(x, 1) for x in res["drop"]["dino_swin_l"]["ENS-WP"]["ci95"]])


if __name__ == "__main__":
    main()
