"""E84 — xác nhận phát hiện hậu kỳ của E83: tương tác surrogate × CÔNG THỨC FEATURE (augmentation × stage), trên
200 ảnh dev300 \\ dev100 (CHƯA dùng cho câu hỏi này), cùng máy.

Mọi ô: objective = L_OSFD thuần (không cls, không WP), geo_osfd_attack (co-transform feature sạch, amp), B=36, ε=5.
  {R50, CNX} × AUG {RRB (2 view/backward, như OSFD), OCT (1 view)} × STAGE {C2–C5, C3–C5}  = 8 ô factorial
  Phụ (không làm cổng):
    OCT2V-C35 = OCT 2 view/backward, C3–C5 -> kiểm phần lợi OCT có phải do số view (RRB 2 view vs OCT 1 view)
    OSFD      = osfd_attack gốc (RRB, C2–C5, feature sạch KHÔNG co-transform, fp32) -> neo về ô OSFD của E83
Seed: torch + python random + numpy = 1000·(n+1), n = chỉ số ảnh trong danh sách 200 ảnh.
Eval: panel 9 model, paired bootstrap 1000 (BOOT_SEED 2026), mọi contrast tính TRONG tiến trình này.
Quy tắc phán quyết: progress_log entry "E84" (khóa + commit TRƯỚC khi chạy).

  python scripts/e84_recipe_factorial.py --gen-only --cells R50-RRB-C25,R50-OCT-C35   # 1 worker sinh
  python scripts/e84_recipe_factorial.py                                               # sinh còn thiếu + eval + phán quyết
  python scripts/e84_recipe_factorial.py --list x5 --boot 20 --tag smoke               # smoke (KHÔNG đọc số)
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
from attack.methods.baselines import osfd_attack
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
STEPS, EPS, RHO, BOOT_SEED = 36, 5.0, 0.002, 2026
R, C = "r50", "convnext_t"
SHORT = {R: "R50", C: "CNX"}
RRB_KW = dict(geo="rrb", amp=True)                       # 2 view/backward (rrb_views), co-transform
OCT_KW = dict(geo="elem", elem=_C2OCT["elem"], n_views=1, amp=True)
STAGES = {"C25": None, "C35": (1, 2, 3)}                   # None = cả 4 stage C2–C5 (như OSFD)

# cell -> (surrogate, hàm attack, kwargs, vai trò)
CELLS = {}
for _s in (R, C):
    p = SHORT[_s]
    for aug, akw in (("RRB", RRB_KW), ("OCT", OCT_KW)):
        for st, fs in STAGES.items():
            CELLS[f"{p}-{aug}-{st}"] = (_s, geo_osfd_attack, dict(akw, feat_stages=fs), "primary")
    CELLS[f"{p}-OCT2V-C35"] = (_s, geo_osfd_attack, dict(OCT_KW, n_views=2, feat_stages=(1, 2, 3)), "aux")
    CELLS[f"{p}-OSFD"] = (_s, osfd_attack, {}, "aux")

PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
PRIMARY_TARGETS = ["swin_t", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
PRIMARY = "I_recipe (OCT-C35 − RRB-C25)"


def contrasts():
    """tên -> {cell: hệ số}. PRIMARY là cổng; còn lại secondary / mô tả. Hiệu số surrogate: CNX − R50."""
    def sur_diff(coef):  # coef trên tên ô không có tiền tố surrogate -> (CNX·coef) − (R50·coef)
        d = {}
        for k, w in coef.items():
            d[f"CNX-{k}"] = d.get(f"CNX-{k}", 0) + w
            d[f"R50-{k}"] = d.get(f"R50-{k}", 0) - w
        return d
    aug = {"OCT-C25": .5, "OCT-C35": .5, "RRB-C25": -.5, "RRB-C35": -.5}       # hiệu ứng chính AUG (TB stage)
    stg = {"RRB-C35": .5, "OCT-C35": .5, "RRB-C25": -.5, "OCT-C25": -.5}       # hiệu ứng chính STAGE (TB aug)
    a3 = {"OCT-C35": 1, "OCT-C25": -1, "RRB-C35": -1, "RRB-C25": 1}           # aug × stage
    d = {PRIMARY: sur_diff({"OCT-C35": 1, "RRB-C25": -1}),
         "I_aug (sur × aug)": sur_diff(aug),
         "I_stage (sur × stage)": sur_diff(stg),
         "I_3way (sur × aug × stage)": sur_diff(a3),
         "I_views (OCT2V−OCT1V, C35)": sur_diff({"OCT2V-C35": 1, "OCT-C35": -1}),
         "I_impl (RRB-C25 − OSFD gốc)": sur_diff({"RRB-C25": 1, "OSFD": -1})}
    for p in ("R50", "CNX"):
        d[f"{p}: OCT-C35 − RRB-C25"] = {f"{p}-OCT-C35": 1, f"{p}-RRB-C25": -1}
        d[f"{p}: aug (OCT−RRB)"] = {f"{p}-{k}": w for k, w in aug.items()}
        d[f"{p}: stage (C35−C25)"] = {f"{p}-{k}": w for k, w in stg.items()}
        d[f"{p}: RRB-C25 − OSFD gốc"] = {f"{p}-RRB-C25": 1, f"{p}-OSFD": -1}
    return d


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def adv_dir(run_dir, cell):
    return os.path.join(ART, run_dir, "adv", cell)


def generate(ids, cell, run_dir):
    sur, fn, kw, _ = CELLS[cell]
    out = adv_dir(run_dir, cell)
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        print(f"[e84] {cell}: PNG đã đủ, bỏ qua sinh", flush=True)
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
        print(f"[e84 {cell}] {n + 1}/{len(ids)}  {(time.time() - t0) / done:.1f}s/ảnh", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")       # tmux/SSH locale ASCII làm vỡ print '−', 'Δ'
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="x200", choices=["x5", "x200"],
                    help="x200 = dev300 \\ dev100 theo thứ tự dev300; x5 = 5 ảnh đầu của nó (smoke)")
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
    run_dir = f"e84_recipe_B{STEPS}_eps5_{mtag}{sfx}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e84_recipe_{args.list}{sfx}.json")
    print(f"[e84] GPU = {gpu} | run dir {run_dir} | ô {cells}", flush=True)

    d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100]
    assert len(ids) == 200, len(ids)
    ids = ids[:int(args.list[1:])]
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
    groups["PRIMARY(Swin-T,Swin-L)"] = PRIMARY_TARGETS
    groups["T7"] = T7
    G = {g: {c: np.mean([S[f"{m}|{c}"] for m in ms], axis=0) for c in CELLS} for g, ms in groups.items()}
    res = dict(meta, drop={}, contrast={})
    for g in groups:
        res["drop"][g] = {c: summarize(G[g][c][0], G[g][c][1:]) for c in CELLS}
        res["contrast"][g] = {}
        for name, coef in contrasts().items():
            v = sum(w * G[g][c] for c, w in coef.items())
            res["contrast"][g][name] = summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    print(f"\n=== E84 @ {args.list}, {gpu}, B={STEPS}, boot {args.boot} — relative AP drop % ===")
    print(f"{'cell':12}" + "".join(f"{g[:10]:>11}" for g in groups))
    for c in CELLS:
        print(f"{c:12}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in groups))
    print("\n--- contrast (điểm [CI95 paired]) ---")
    for name in contrasts():
        print(f"{name:30}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][name])}"
                                        for g in ["PRIMARY(Swin-T,Swin-L)", "dino_swin_l", "swin_t", "T7"]))
    print(f"\n{PRIMARY} mọi target:")
    for m in PANEL:
        tagm = " (G0: surrogate của 1 nhánh)" if m in (R, C) else ""
        print(f"  {m:12} {fmt(res['contrast'][m][PRIMARY])}{tagm}")

    # ---- phán quyết theo quy tắc khóa (progress_log "E84") ----
    I = res["contrast"]["PRIMARY(Swin-T,Swin-L)"][PRIMARY]
    ok = I["ci95"][0] > 0
    res["verdict"] = {"I_primary": I, "confirmed": bool(ok)}
    print(f"\n=> {PRIMARY} trên TB(Swin-T, DINO-Swin-L) = {fmt(I)}")
    print("   " + ("cận dưới CI > 0 -> XÁC NHẬN: trụ cột = tương tác surrogate × công thức feature" if ok else
                   "CI chứa 0 (hoặc âm) -> KHÔNG xác nhận -> paper hạ xuống mô tả hiện tượng (plan R2)"))
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(f"[e84] -> {out_json}")


if __name__ == "__main__":
    main()
