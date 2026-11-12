"""E85 (bước 3b, trục mới) — tầng feature × họ surrogate trên 4 surrogate Mask R-CNN, dev100, cùng máy.

Mọi ô factorial: objective = L_OSFD thuần, geo_osfd_attack (co-transform, amp), B=36, ε=5 (định nghĩa y hệt E84):
  {R50, CNX, R101, SWT} × AUG {RRB, OCT} × STAGE {C2–C5, C3–C5}   = 16 ô
  + CLS-only cho R101, SWT (như ô CLS của E83; MI chuẩn hóa rồi sign -> λ không tác dụng, đặt 1.0) = 2 ô.
  Ô CLS của R50/CNX trên dev100 lấy từ E83 (cùng máy, cùng seed) cho 3c, không sinh lại.
MÔ TẢ, KHÔNG có cổng (plan 3b: không trình bày như xác nhận độc lập). dev100 đã dùng ở E83 (nơi phát hiện hậu kỳ).
Target: mỗi surrogate loại checkpoint của chính nó (G0); surrogate SWT: DINO-Swin-L là G1 (cùng họ Swin).
Seed: torch + random + numpy = 1000·(n+1), n = chỉ số trong dev100 (trùng seed E83).
Eval: panel 9 model, paired bootstrap 1000 (BOOT_SEED 2026), contrast trong tiến trình. Lưu dets từng ảnh cho 3c.

  python scripts/e85_surrogate_stage.py --gen-only --cells R50-RRB-C25,...   # 1 worker sinh
  python scripts/e85_surrogate_stage.py                                       # sinh còn thiếu + eval
  python scripts/e85_surrogate_stage.py --list dev5 --boot 20 --tag smoke     # smoke (KHÔNG đọc số)
"""
import argparse
import gzip
import json
import os
import random
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
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STEPS, EPS, BOOT_SEED = 36, 5.0, 2026
SURS = {"r50": "R50", "convnext_t": "CNX", "r101": "R101", "swin_t": "SWT"}
RRB_KW = dict(geo="rrb", amp=True)                       # 2 view/backward (rrb_views), co-transform
OCT_KW = dict(geo="elem", elem=_C2OCT["elem"], n_views=1, amp=True)
STAGES = {"C25": None, "C35": (1, 2, 3)}                   # None = cả 4 stage C2–C5 (như OSFD)
CLS_KW = dict(_C2OCT, cls_weight=1.0, cls_mode="roi", feature=False)   # = ô CLS E83 (λ vô tác dụng do sign)

# cell -> (surrogate, hàm attack, kwargs, vai trò)
CELLS = {}
for _s, p in SURS.items():
    for aug, akw in (("RRB", RRB_KW), ("OCT", OCT_KW)):
        for st, fs in STAGES.items():
            CELLS[f"{p}-{aug}-{st}"] = (_s, geo_osfd_attack, dict(akw, feat_stages=fs), "factorial")
    if p in ("R101", "SWT"):
        CELLS[f"{p}-CLS"] = (_s, geo_osfd_attack, CLS_KW, "3c")

PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
T5C = ["fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]   # chung cho mọi surrogate (không G0)
AUG = {"OCT-C25": .5, "OCT-C35": .5, "RRB-C25": -.5, "RRB-C35": -.5}
STG = {"RRB-C35": .5, "OCT-C35": .5, "RRB-C25": -.5, "OCT-C25": -.5}


def own_groups(p):
    """nhóm target cho contrast TRONG surrogate p: loại checkpoint của chính nó."""
    sur = [k for k, v in SURS.items() if v == p][0]
    return {f"T7\\{p}": [m for m in T7 if m != sur],
            f"TF\\{p}": [m for m in ("swin_t", "dino_swin_l") if m != sur]}


def contrasts():
    """tên -> {cell: hệ số}. MÔ TẢ, không cổng."""
    d = {}
    for p in SURS.values():
        d[f"{p}: stage (C35−C25)"] = {f"{p}-{k}": w for k, w in STG.items()}
        d[f"{p}: aug (OCT−RRB)"] = {f"{p}-{k}": w for k, w in AUG.items()}
        d[f"{p}: recipe (OCT-C35 − RRB-C25)"] = {f"{p}-OCT-C35": 1, f"{p}-RRB-C25": -1}
        if f"{p}-CLS" in CELLS:
            d[f"{p}: CLS − OCT-C35"] = {f"{p}-CLS": 1, f"{p}-OCT-C35": -1}
    for p in ("CNX", "R101", "SWT"):                              # tương tác so với R50 (đọc trên T5C)
        d[f"I_stage ({p} − R50)"] = {**{f"{p}-{k}": w for k, w in STG.items()},
                                     **{f"R50-{k}": -w for k, w in STG.items()}}
        d[f"I_recipe ({p} − R50)"] = {f"{p}-OCT-C35": 1, f"{p}-RRB-C25": -1, "R50-OCT-C35": -1, "R50-RRB-C25": 1}
    return d


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def adv_dir(run_dir, cell):
    return os.path.join(ART, run_dir, "adv", cell)


def generate(ids, cell, run_dir):
    sur, fn, kw, _ = CELLS[cell]
    out = adv_dir(run_dir, cell)
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        print(f"[e85] {cell}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    model = load_model(sur, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(sur, ids)}
    t0, done = time.time(), 0
    for n, i in enumerate(ids):
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        orig = ds[i]["orig_inputs"].cuda().float()
        seed_all(1000 * (n + 1))
        delta = fn(model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS, **kw)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= EPS + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png")
        os.replace(p + ".tmp.png", p)              # ghi nguyên tử: PNG dở không bị coi là xong khi resume
        done += 1
        print(f"[e85 {cell}] {n + 1}/{len(ids)}  {(time.time() - t0) / done:.1f}s/ảnh", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")       # tmux/SSH locale ASCII làm vỡ print '−', 'Δ'
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="dev100", choices=["dev5", "dev100"],
                    help="dev100; dev5 = 5 ảnh đầu dev100 (smoke)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cells", default=None, help="danh sách ô, phẩy ngăn (mặc định: tất cả)")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--tag", default="", help="hậu tố run dir / json (vd 'smoke')")
    args = ap.parse_args()
    cells = args.cells.split(",") if args.cells else list(CELLS)
    assert all(c in CELLS for c in cells), cells
    gpu = torch.cuda.get_device_name(0)
    mtag = gpu.split()[-1].replace("-", "").lower()
    sfx = f"_{args.tag}" if args.tag else ""
    run_dir = f"e85_surstage_B{STEPS}_eps5_{mtag}{sfx}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e85_surstage_{args.list}{sfx}.json")
    print(f"[e85] GPU = {gpu} | run dir {run_dir} | ô {cells}", flush=True)

    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100, len(ids)
    ids = ids[:int(args.list[3:])]
    for c in cells:
        generate(ids, c, run_dir)
    if args.gen_only:
        return
    assert set(cells) == set(CELLS), "eval cần đủ mọi ô"

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])   # hàng 0 = mẫu gốc
    npz = out_json.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}       # "model|cell" -> [1+boot] drop %
    meta = {"list": args.list, "n_images": n, "n_boot": args.boot, "boot_seed": BOOT_SEED, "gpu": gpu,
            "steps": STEPS, "eps": EPS, "run_dir": run_dir,
            "cells": {c: {"surrogate": v[0], "fn": v[1].__name__, "role": v[3],
                          "kw": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v[2].items()}}
                      for c, v in CELLS.items()}}
    from mmdet.apis import inference_detector
    dets_dir = os.path.join(ART, run_dir, "dets")
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in CELLS):
            continue
        model = load(key)
        os.makedirs(os.path.join(dets_dir, key), exist_ok=True)

        def ap_of(name, paths):
            dets = []
            for i, p in zip(ids, paths):
                dets += to_coco_dets(inference_detector(model, p), i, cat_ids)
            with gzip.open(os.path.join(dets_dir, key, f"{name}.json.gz"), "wt") as f:   # cho 3c (outcome theo ảnh)
                json.dump(dets, f)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        clean = ap_of("clean", [os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in CELLS:
            adv = ap_of(c, [os.path.join(adv_dir(run_dir, c), f"{i}.png") for i in ids])
            S[f"{key}|{c}"] = (100 * (clean - adv) / clean).astype(np.float64)
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)                                       # incremental -> resume theo model
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)

    # ---- bảng + contrast (paired: cùng W, cùng clean) ----
    groups = {m: [m] for m in PANEL}
    groups["T7"] = T7
    groups["T5C"] = T5C
    for p in SURS.values():
        groups.update(own_groups(p))
    G = {g: {c: np.mean([S[f"{m}|{c}"] for m in ms], axis=0) for c in CELLS} for g, ms in groups.items()}
    res = dict(meta, drop={}, contrast={}, groups=groups)
    for g in groups:
        res["drop"][g] = {c: summarize(G[g][c][0], G[g][c][1:]) for c in CELLS}
        res["contrast"][g] = {}
        for name, coef in contrasts().items():
            v = sum(w * G[g][c] for c, w in coef.items())
            res["contrast"][g][name] = summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    show = PANEL + ["T7", "T5C"]
    print(f"\n=== E85 @ {args.list}, {gpu}, B={STEPS}, boot {args.boot} — relative AP drop % (G0 = cột của chính surrogate) ===")
    print(f"{'cell':12}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in CELLS:
        print(f"{c:12}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    print("\n--- contrast trong từng surrogate (target loại G0) ---")
    for name in contrasts():
        if name.startswith("I_"):
            continue
        p = name.split(":")[0]
        gs = list(own_groups(p)) + ["dino_swin_l", "T5C"]
        print(f"{name:32}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][name])}" for g in gs))
    print("\n--- tương tác so với R50 (T5C chung, không G0) ---")
    for name in contrasts():
        if name.startswith("I_"):
            print(f"{name:32}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][name])}" for g in ["T5C", "dino_swin_l"]))
    print("\n(MÔ TẢ — không có cổng; SWT: dino_swin_l là G1 cùng họ Swin)")
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(f"[e85] -> {out_json}")

if __name__ == "__main__":
    main()
