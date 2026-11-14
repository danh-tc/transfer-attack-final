"""E93 — HardBank (worst-view, E37) trên ConvNeXt với recipe RD-CNX, từ δ = 0 (progress_log 2026-11-14 E93). PILOT.
Ô: RD-CNX = METHODS["C2-OCT-1V-WP2-RD-CNX"] B50 (đóng băng) | RD-CNX-HB = y hệt + hardview=(4, 2) (bank 8 view cố định
   4 scale × 2 pha như E37; chọn view loss THẤP NHẤT bằng forward no-grad, backward chỉ view đó). Cùng B50 = 50 update =
   50 backward, ε 5, momentum, WP ρ 0.002, loss RD (C3–C5 + 0.2·RoI cls) — loss HB đã kiểm trùng RD (e93_unit_test).
   HB thêm 8 forward no-grad / bước (+ 8 forward feature sạch). Đo wall-clock từng ô (1 tiến trình, tuần tự).
Ảnh: devC300[120:120+N] (chưa dùng). Seed 1000·(n+1). Target: dino_swin_l (chính), yolox_l (phụ). Relative AP drop,
   paired bootstrap 1000 (seed 2026) trong tiến trình.
  OMP_NUM_THREADS=4 python scripts/e93_hardbank_cnx.py [--n N] [--smoke]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import METHODS

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
TGT = ["dino_swin_l", "yolox_l"]
EPS, STEPS, BOOT_SEED, NB = 5.0, 50, 2026, 1000
RD_KW = METHODS["C2-OCT-1V-WP2-RD-CNX"][0].keywords
CELLS = {"RD-CNX": dict(RD_KW), "RD-CNX-HB": dict(RD_KW, hardview=(4, 2))}
OUT = os.path.join(REPO_ROOT, "results/mechanism/e93_hardbank_cnx_devC.json")


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devC300.csv"))[120:120 + (2 if a.smoke else a.n)]
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    from mmdet.apis import inference_detector
    t0 = time.time()
    model = load_model("convnext_t", "cuda:0")
    ds = {x["img_id"]: x for x in target_dataset("convnext_t", ids)}
    clean = {i: ds[i]["orig_inputs"].permute(1, 2, 0).numpy().astype(np.uint8) for i in ids}
    adv, wall = {c: {} for c in CELLS}, {}
    for c, kw in CELLS.items():
        torch.cuda.synchronize(); tc = time.time()
        for k, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float()
            seed_all(1000 * (k + 1))
            d = geo_osfd_attack(model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS, **kw)
            x = to_adv_image(orig, d)
            assert float((x - orig).abs().max()) <= EPS + 1e-4
            adv[c][i] = x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        wall[c] = (time.time() - tc) / len(ids)
        print(f"[e93] gen {c}: {wall[c]:.2f} s/ảnh ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), 20 if a.smoke else NB, BOOT_SEED)])
    S = {}
    for tg in TGT:
        m = load(tg)

        def apv(imgs):
            dets = []
            for i in ids:
                dets += to_coco_dets(inference_detector(m, imgs[i]), i, cat_ids)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        cl = apv(clean)
        for c in CELLS:
            S[(tg, c)] = 100 * (cl - apv(adv[c])) / cl
        del m
        torch.cuda.empty_cache()
        print(f"[e93] eval {tg} ({time.time() - t0:.0f}s)", flush=True)
    if a.smoke:
        print(f"[smoke] n {len(ids)} wall {json.dumps({k: round(v, 2) for k, v in wall.items()})} total {time.time() - t0:.0f}s "
              f"(KHÔNG đọc số)")
        return
    res = {"ids": ids, "n": len(ids), "wall_s_per_img": {k: round(v, 3) for k, v in wall.items()},
           "wall_ratio_HB_over_RD": round(wall["RD-CNX-HB"] / wall["RD-CNX"], 3),
           "forward_per_step_note": "RD: 1 view fwd+bwd + 1 fwd sạch; HB: 8 fwd no-grad (bank) + 8 fwd sạch + 1 fwd sạch + 1 fwd+bwd",
           "cells": {}, "delta_HB_minus_RD": {}}
    for tg in TGT:
        for c in CELLS:
            res["cells"][f"{tg}|{c}"] = [round(float(S[(tg, c)][0]), 2), ci(S[(tg, c)][1:])]
        d = S[(tg, "RD-CNX-HB")] - S[(tg, "RD-CNX")]
        res["delta_HB_minus_RD"][tg] = {"point": round(float(d[0]), 2), "ci95": ci(d[1:])}
    p = res["delta_HB_minus_RD"]["dino_swin_l"]
    if p["ci95"][1] < 0:
        verdict = "NO-GO (HB giảm transfer Swin-L)"
    elif p["ci95"][0] > 0:
        verdict = ("CẢI THIỆN RÕ Swin-L — CHỈ đi tiếp nếu matched-runtime vẫn lợi"
                   + (" (wall > +10%: BẮT BUỘC đối chứng cùng thời gian)" if res["wall_ratio_HB_over_RD"] > 1.1 else ""))
    else:
        verdict = "KHÔNG có bằng chứng cải thiện (CI chứa 0) -> dừng nhánh; KHÔNG kết luận tương đương"
    res["verdict"] = verdict
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
