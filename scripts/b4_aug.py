"""B4 (progress_log 2026-11-15): trục AUGMENTATION/EoT cho CNX có BẤT KỲ headroom RES3 nào không? SCREEN dev30 (rẻ,
directional — winner PHẢI xác nhận devC, bài học CWT dev30->dev100). CNX-only, giữ single-surrogate, GIỮ NGUYÊN B=50
(n_views gộp vào 1 backward -> không tăng số backward). Recipe RD-CNX đóng băng, chỉ đổi augmentation.
Ô (seed_all per ảnh = 1000*(n+1), như E88/F1b):
  base     RD-CNX gốc: OCT{scale, shift:ph8}, n_views=1
  nv4      n_views=4 (EoT trên 4 mẫu OCT), còn lại y hệt
  rot      elem {scale, rot:10°, shift:ph8}, n_views=1
  rotnv4   elem {scale, rot:10°, shift:ph8}, n_views=4
Eval panel {r101, fcos_r50, dino_r50, swin_t, dino_swin_l}, relative AP drop, boot 1000 seed 2026, paired trong tiến trình.
PROMISING(v) (khóa, directional dev30): ΔRES3(v−base) point ≥ +1.5 VÀ CI lower > 0 VÀ ΔSwinL ≥ −2 -> đưa lên devC.
  python scripts/b4_aug.py --gen-only / python scripts/b4_aug.py
"""
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e88_ens_baselines import C, seed_all
from scripts.f1b_r1a import ANN, ART, IMGDIR, EPS, BOOT_SEED, save_png
from scripts.run_baselines import _C2OCT, _E63, _OCT

PANEL = ["r101", "fcos_r50", "dino_r50", "swin_t", "dino_swin_l"]
RES3 = ["r101", "fcos_r50", "dino_r50"]
RDCNX = {**{k: v for k, v in _C2OCT.items() if k not in ("cls_weight", "elem", "n_views")},
         "cls_weight": 0.2, **_E63, "weight_rho": 0.002, "weight_linear": True}   # phần đóng băng (bỏ elem/n_views để ô tự đặt)
CELLS = {
    "base":   dict(elem={"scale": _OCT, "shift": "ph8"}, n_views=1),
    "nv4":    dict(elem={"scale": _OCT, "shift": "ph8"}, n_views=4),
    "rot":    dict(elem={"scale": _OCT, "rot": 10.0, "shift": "ph8"}, n_views=1),
}  # rotnv4 (OCT+rot 4V) cắt 2026-11-15 theo user (tiết kiệm ~9'); combo để devC nếu nv4/rot có tín hiệu
OUT = os.path.join(REPO_ROOT, "results/mechanism/b4_aug_dev30.json")


def ids_dev30():
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    return ids


def png(run_dir, cell, i):
    return os.path.join(ART, run_dir, "adv", cell, f"{i}.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-only", action="store_true")
    args = ap.parse_args()
    ids = ids_dev30()
    gpu = torch.cuda.get_device_name(0)
    run_dir = f"b4_aug_dev30_eps5_{gpu.split()[-1].lower()}"
    tpath = os.path.join(REPO_ROOT, "results/mechanism/b4_aug_timing.json")
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}

    model = None
    for cell, extra in CELLS.items():
        if all(os.path.exists(png(run_dir, cell, i)) for i in ids):
            continue
        if model is None:
            model = load_model(C, "cuda:0")
            ds = {s["img_id"]: s for s in target_dataset(C, ids)}
        t_all = timing.setdefault(cell, {})
        for n, i in enumerate(ids):
            if os.path.exists(png(run_dir, cell, i)):
                continue
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            seed_all(1000 * (n + 1))
            torch.cuda.synchronize(); t0 = time.time()
            delta = geo_osfd_attack(model, orig, dsm, steps=50, epsilon=EPS, **RDCNX, **extra)
            torch.cuda.synchronize(); t_all[str(i)] = time.time() - t0
            save_png(orig, delta, png(run_dir, cell, i))
            print(f"[b4 {cell}] {n + 1}/{len(ids)}  {np.mean(list(t_all.values())):.2f}s/ảnh", flush=True)
        json.dump(timing, open(tpath, "w"), indent=1)
    if model is not None:
        del model; torch.cuda.empty_cache()
    if args.gen_only:
        return

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, 1000, BOOT_SEED)])
    npz = OUT.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
    from mmdet.apis import inference_detector
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in CELLS):
            continue
        m = load(key)
        with torch.no_grad():
            clean_d = [d for i in ids for d in to_coco_dets(inference_detector(m, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)]
            clean = np.array([CocoImageEval(coco, clean_d, ids).ap(w)[0] for w in W]) * 100
            for c in CELLS:
                adv_d = [d for i in ids for d in to_coco_dets(inference_detector(m, png(run_dir, c, i)), i, cat_ids)]
                S[f"{key}|{c}"] = 100 * (clean - np.array([CocoImageEval(coco, adv_d, ids).ap(w)[0] for w in W]) * 100) / clean
        del m; torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)

    ci = lambda v: summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    grp = lambda g, c: np.mean([S[f"{m}|{c}"] for m in RES3], axis=0) if g == "RES3" else S[f"{g}|{c}"]
    res = {"n_images": n, "gpu": gpu, "run_dir": run_dir,
           "timing_s_per_img": {k: float(np.mean(list(v.values()))) for k, v in timing.items() if v},
           "drop": {g: {c: ci(grp(g, c)) for c in CELLS} for g in PANEL + ["RES3"]}, "contrast": {}, "promising": []}
    print(f"\n=== B4 augmentation @ dev30 (screen, directional) — relative AP drop % ===")
    show = PANEL + ["RES3"]
    print(f"{'cell':8}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in CELLS:
        print(f"{c:8}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    for v in CELLS:
        if v == "base":
            continue
        dR = ci(grp("RES3", v) - grp("RES3", "base"))
        dS = ci(grp("dino_swin_l", v) - grp("dino_swin_l", "base"))
        dT = ci(grp("swin_t", v) - grp("swin_t", "base"))
        res["contrast"][v] = {"dRES3": dR, "dSwinL": dS, "dSwinT": dT}
        prom = dR["point"] >= 1.5 and dR["ci95"][0] > 0 and dS["point"] >= -2
        if prom:
            res["promising"].append(v)
        print(f"{v:8} ΔRES3 {fmt(dR)}  ΔSwinL {fmt(dS)}  ΔSwinT {fmt(dT)}  {'-> PROMISING' if prom else ''}")
    print(f"\nPROMISING (đưa lên devC): {res['promising'] or 'không có'}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[b4] -> {OUT}")


if __name__ == "__main__":
    main()
