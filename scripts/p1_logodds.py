"""P1 (progress_log 2026-11-16 23:20, KHÓA): RD-CNX với cls log-odds (không bão hòa) vs RD-CNX, cả hai α = 2, B50, dev30.
  python scripts/p1_logodds.py --calib     # s = trung vị mean|∇CE_RoI| / mean|∇L_lo| tại δ=0, 5 ảnh đầu dev100 + unit test
  python scripts/p1_logodds.py --scale S   # sinh + eval + phán quyết
"""
import argparse
import contextlib
import functools
import gzip
import json
import math
import os
import random
import statistics
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.repair import roi_logodds
from attack.models import load_model
from attack.preprocess import resize_to_model, to_adv_image, to_batch
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.e88_ens_baselines import C, seed_all
from scripts.run_baselines import METHODS

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs/p1_logodds")
EPS, ALPHA, STEPS = 5.0, 2.0, 50
PANEL = ["convnext_t", "r101", "fcos_r50", "dino_r50", "swin_t", "yolov3_d53", "yolox_l", "dino_swin_l", "rtmdet_l"]
RES3 = ["r101", "fcos_r50", "dino_r50"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
RTM = (os.path.join(REPO_ROOT, "third_party/mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"),
       os.path.join(REPO_ROOT, "checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"))


def ids_dev(n):
    return load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:n]


def calib():
    model = load_model(C, "cuda:0")
    model.roi_head.mask_head = None
    ids = ids_dev(5)
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}

    def grad(orig, dsm, ctx, seed):
        random.seed(seed); torch.manual_seed(seed)
        n = torch.zeros_like(orig, requires_grad=True)
        bi, bds = to_batch(model, [resize_to_model(orig + n, tuple(dsm.img_shape))], [dsm])
        with ctx():
            l = model.loss(bi, bds)["loss_cls"]
        return torch.autograd.grad(l, n)[0], float(l)
    r = []
    for j, i in enumerate(ids):
        orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
        if "masks" in dsm.gt_instances:
            del dsm.gt_instances.masks
        g_ce, l_ce = grad(orig, dsm, contextlib.nullcontext, j)
        g_lo, l_lo = grad(orig, dsm, lambda: roi_logodds(model, 1.0), j)
        r.append(float(g_ce.abs().mean() / (g_lo.abs().mean() + 1e-12)))
        print(f"img {i}: CE {l_ce:.3f}  logodds {l_lo:.3f}  |∇CE|/|∇lo| {r[-1]:.4f}")
    # unit test: forward khôi phục + giá trị log-odds tính tay
    crit = model.roi_head.bbox_head.loss_cls
    z = torch.randn(7, 81, device="cuda"); lab = torch.randint(0, 81, (7,), device="cuda")
    with roi_logodds(model, 1.0):
        v = float(crit(z, lab, torch.ones(7, device="cuda"), avg_factor=7))
    p = torch.softmax(z, 1).gather(1, lab[:, None])[:, 0]
    ref = float(torch.log((1 - p) / p).mean())
    restored = "forward" not in crit.__dict__
    ok = abs(v - ref) < 1e-4 and restored
    print(f"unit test: logodds {v:.5f} vs tay {ref:.5f}, forward khôi phục {restored} -> {'PASS' if ok else 'FAIL'}")
    s = statistics.median(r)
    print(f"SCALE = {s:.4f} (từ {[round(x, 4) for x in r]})")
    json.dump({"ratios": r, "scale": s, "unit_test": ok}, open(os.path.join(REPO_ROOT, "results/p1/calib.json"), "w"), indent=1)


def png(cell, i):
    return os.path.join(ART, "adv", cell, f"{i}.png")


def generate(ids, scale):
    base, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    cells = {"RD-CNX-a2": base, "RD-CNX-LO-a2": functools.partial(base, logodds_scale=scale)}
    model = load_model(C, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}
    t = {}
    for cell, fn in cells.items():
        ts = []
        for n, i in enumerate(ids):
            if os.path.exists(png(cell, i)):
                continue
            orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
            seed_all(1000 * (n + 1))
            torch.cuda.synchronize(); t0 = time.time()
            delta = fn(model, orig, dsm, steps=STEPS, epsilon=EPS, alpha=ALPHA)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
            adv = to_adv_image(orig, delta)
            assert float((adv - orig).abs().max()) <= EPS + 1e-4
            os.makedirs(os.path.dirname(png(cell, i)), exist_ok=True)
            mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png(cell, i))
            print(f"[p1 {cell}] {n + 1}/{len(ids)} {np.mean(ts):.1f}s/ảnh", flush=True)
        t[cell] = float(np.mean(ts)) if ts else None
    del model
    torch.cuda.empty_cache()
    return list(cells), t


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--scale", type=float)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO_ROOT, "results/p1"), exist_ok=True)
    if args.calib:
        return calib()
    ids = ids_dev(args.n)
    cells, timing = generate(ids, args.scale)
    from mmdet.apis import inference_detector, init_detector
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), 200, 2026)])
    S, conf = {}, {}
    for key in PANEL:
        cache = os.path.join(ART, f"dets_{key}_n{len(ids)}.json")
        if os.path.exists(cache):
            dets = json.load(open(cache))
        else:
            model = init_detector(*RTM, device="cuda:0") if key == "rtmdet_l" else load(key)
            dets = {c: [] for c in ["clean", *cells]}
            with torch.no_grad():
                for i in ids:
                    dets["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)
                    for c in cells:
                        dets[c] += to_coco_dets(inference_detector(model, png(c, i)), i, cat_ids)
            json.dump(dets, open(cache, "w"))
            del model
            torch.cuda.empty_cache()
        ev = CocoImageEval(coco, dets["clean"], ids); clean = np.array([ev.ap(w)[0] for w in W])
        for c in cells:
            ev = CocoImageEval(coco, dets[c], ids)
            S[f"{key}|{c}"] = 100 * (clean - np.array([ev.ap(w)[0] for w in W])) / clean
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in cells), flush=True)
        if key == "convnext_t":   # kiểm cơ chế: object CNX white-box còn sống với điểm ≥ 0.9
            G = {c: group(dets[c]) for c in ["clean", *cells]}
            for c in cells:
                cnt = 0
                for k, i in enumerate(ids):
                    for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
                        if fate(a["bbox"], a["category_id"], G["clean"].get(i, []))[0] != "survived":
                            continue
                        f, s = fate(a["bbox"], a["category_id"], G[c].get(i, []))
                        cnt += int(f == "survived" and s >= 0.9)
                conf[c] = cnt
            print(f"CNX white-box sống & ≥0.9: {conf}", flush=True)
    ci = lambda v: summarize(v[0], v[1:])
    grp = lambda g, c: np.mean([S[f"{m}|{c}"] for m in g], axis=0)
    a, b = "RD-CNX-LO-a2", "RD-CNX-a2"
    res = {"n": len(ids), "scale": args.scale, "timing": timing, "conf_survivors_cnx": conf,
           "drop": {k: ci(v) for k, v in S.items()},
           "dRES3": ci(grp(RES3, a) - grp(RES3, b)), "dSwinL": ci(S[f"dino_swin_l|{a}"] - S[f"dino_swin_l|{b}"]),
           "dT7": ci(grp(T7, a) - grp(T7, b)), "dRTMDet": ci(S[f"rtmdet_l|{a}"] - S[f"rtmdet_l|{b}"]),
           "dCNX_WB": ci(S[f"convnext_t|{a}"] - S[f"convnext_t|{b}"])}
    for k in ("dRES3", "dSwinL", "dT7", "dRTMDet", "dCNX_WB"):
        print(f"{k:8} {res[k]['point']:+6.2f} [{res[k]['ci95'][0]:+.2f}, {res[k]['ci95'][1]:+.2f}]")
    r3, sl = res["dRES3"], res["dSwinL"]
    if r3["point"] >= 1.5 and r3["ci95"][0] > 0 and sl["point"] >= -2:
        v = "PROMISING -> khóa xác nhận dev100"
    elif r3["ci95"][1] < 0.5 or sl["point"] <= -3:
        v = "NO-GO"
    else:
        v = "UNCLEAR"
    res["verdict"] = v
    print(f"P1 VERDICT (khóa): {v} | cơ chế: object CNX tự tin còn sống {conf}")
    json.dump(res, open(os.path.join(REPO_ROOT, f"results/mechanism/p1_logodds_dev{len(ids)}.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
