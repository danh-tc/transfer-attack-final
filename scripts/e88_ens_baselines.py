"""E88 — baseline fidelity nửa đầu (bước 4 plan_paper_lock, lớp B): OSFD 2 cấu hình + Ens-OSFD / Ens-RD, dev100, cùng máy.

Câu hỏi:
  Q1 OSFD-R50 B50: θ=10/ρ=1.0 (reference-code behavior, bug truyền tham số đã xác minh 2026-11-13) vs θ=7/ρ=0.8
     (published config) chênh bao nhiêu?
  Q2 ENS-WP (luân phiên khối 2 bước + momentum chung) có hơn ensemble TRUNG BÌNH đơn giản trên CÙNG 2 surrogate,
     CÙNG objective từng surrogate, CÙNG tổng 36 backward không? (phản biện "chỉ vì có 2 model")
Ô (ε=5, dev100, seed torch+random+numpy):
  OSFD-PUB      osfd_attack R50 B50, θ7 ρ0.8 (port mặc định)                     seed 1000·(n+1)
  OSFD-REF      osfd_attack R50 B50, θ10 ρ1.0                                     seed 1000·(n+1)
  RD-CNX        C2-OCT-1V-WP2-RD-CNX B50 (đóng băng, mốc sinh lại)                seed 1000·(n+1)
  ENS-WP        đóng băng (= e82 ô ENS-WP-B36): khối 2 bước luân phiên, momentum chung, θ' không mang  seed 1000·(n+1)+b
  ENS-{OSFD,RD}-{AVG,GN}: 18 bước × (R50 + CNX) = 36 backward. Mỗi bước gọi từng surrogate steps=1, alpha=0 (không
     bước) lấy gradient THÔ qua grad_tap (cùng views / WP / co-transform của objective đơn); gộp:
       AVG = (g_R50 + g_CNX)/2  (= gradient của loss trung bình)
       GN  = g_R50/mean|g_R50| + g_CNX/mean|g_CNX|  (chuẩn hóa từng surrogate — tránh 1 model áp đảo thang loss)
     rồi MI (momentum 1, chuẩn hóa mean|g|, sign, α=1). Objective từng surrogate: OSFD = osfd_attack θ7 ρ0.8;
     RD = kw_for(·, ρ0.002) + COMMON (y hệt ENS-WP), θ' WP MANG qua bước của chính surrogate (wp_in/wp_out).
     seed 1000·(n+1) + 2t + j (t bước, j surrogate).
Eval: panel 9 model, boot 1000, BOOT_SEED 2026, contrast paired trong tiến trình. Quy tắc đọc: progress_log E88.

  python scripts/e88_ens_baselines.py --gen-only --cells OSFD-PUB,...
  python scripts/e88_ens_baselines.py
  python scripts/e88_ens_baselines.py --list smoke3 --boot 20 --tag smoke
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
from scripts.e81_ens_alloc_rho import COMMON, C, R, kw_for
from scripts.run_baselines import METHODS

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
EPS, RHO, BOOT_SEED, ENS_STEPS = 5.0, 0.002, 2026, 18
OSFD_PUB = dict(rrb_theta=7.0, rrb_rho=0.8)
OSFD_REF = dict(rrb_theta=10.0, rrb_rho=1.0)
CELLS = ["OSFD-PUB", "OSFD-REF", "RD-CNX", "ENS-WP", "ENS-OSFD-AVG", "ENS-OSFD-GN", "ENS-RD-AVG", "ENS-RD-GN"]
SURS = {"OSFD-PUB": [R], "OSFD-REF": [R], "RD-CNX": [C]}

PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
T7 = ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
GROUPS = {"T7": T7, "G1": ["r101", "fcos_r50", "dino_r50"], "G2cnn": ["yolov3_d53", "yolox_l"],
          "TF": ["swin_t", "dino_swin_l"]}


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def ens_step_grad(kind, key, model, orig, dsm, delta, wp_state):
    """Gradient THÔ của objective đơn surrogate `key` tại δ (1 backward, không bước)."""
    tap = []
    if kind == "OSFD":
        osfd_attack(model, orig, dsm, steps=1, epsilon=EPS, alpha=0.0, init_noise=delta, grad_tap=tap, **OSFD_PUB)
    else:
        out = []
        geo_osfd_attack(model, orig, dsm, steps=1, epsilon=EPS, alpha=0.0, init_noise=delta, grad_tap=tap,
                        wp_in=wp_state.get(key), wp_out=out, **kw_for(key, RHO), **COMMON)
        wp_state[key] = out
    assert len(tap) == 1
    return tap[0]


def ens_avg_attack(kind, agg, models, ds, i, n, steps=ENS_STEPS, n_backward=None):
    orig = ds[R][i]["orig_inputs"].cuda().float()
    delta, mom, wp_state = torch.zeros_like(orig), torch.zeros_like(orig), {}
    for t in range(steps):
        gs = []
        for j, key in enumerate((R, C)):
            seed_all(1000 * (n + 1) + 2 * t + j)
            gs.append(ens_step_grad(kind, key, models[key], orig, ds[key][i]["data_sample"], delta, wp_state))
            if n_backward is not None:
                n_backward[0] += 1
        if agg == "AVG":
            g = (gs[0] + gs[1]) / 2
        else:
            g = sum(x / (x.abs().mean() + 1e-12) for x in gs)
        with torch.no_grad():
            mom = mom + g / (g.abs().mean() + 1e-12)               # MI momentum 1.0 (như core)
            delta = torch.clamp(delta + torch.sign(mom), -EPS, EPS)
    return orig, delta


def ens_wp_attack(models, ds, i, n):
    """= scripts/e82_strict1_wp.py ô ENS-WP-B36 (block=2, carry=False)."""
    orig = ds[R][i]["orig_inputs"].cuda().float()
    delta, mom = None, None
    for b in range(36 // 2):
        key = (R, C)[b % 2]
        seed_all(1000 * (n + 1) + b)
        delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=2, init_noise=delta,
                                     mom_in=mom, return_state=True, **kw_for(key, RHO), **COMMON)
    return orig, delta


def single_attack(cell, model, ds, i, n):
    orig = ds[i]["orig_inputs"].cuda().float()
    seed_all(1000 * (n + 1))
    if cell.startswith("OSFD"):
        kw = OSFD_PUB if cell == "OSFD-PUB" else OSFD_REF
        return orig, osfd_attack(model, orig, ds[i]["data_sample"], steps=50, epsilon=EPS, **kw)
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    return orig, fn(model, orig, ds[i]["data_sample"], steps=50, epsilon=EPS)


def adv_dir(run_dir, cell):
    return os.path.join(ART, run_dir, "adv", cell)


def image_ids(name):
    if name == "smoke3":                                     # 3 ảnh đầu x200
        d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
        return [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100][:3]
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100
    return ids


def generate(ids, cell, run_dir):
    out = adv_dir(run_dir, cell)
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        print(f"[e88] {cell}: PNG đã đủ, bỏ qua sinh", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    keys = SURS.get(cell, [R, C])
    models = {k: load_model(k, "cuda:0") for k in keys}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in keys}
    t0, done = time.time(), 0
    for n, i in enumerate(ids):
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        if cell in SURS:
            orig, delta = single_attack(cell, models[keys[0]], ds[keys[0]], i, n)
        elif cell == "ENS-WP":
            orig, delta = ens_wp_attack(models, ds, i, n)
        else:
            _, kind, agg = cell.split("-")
            orig, delta = ens_avg_attack(kind, agg, models, ds, i, n)
        adv = to_adv_image(orig, delta)
        assert float((adv - orig).abs().max()) <= EPS + 1e-4
        mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png")
        os.replace(p + ".tmp.png", p)
        done += 1
        print(f"[e88 {cell}] {n + 1}/{len(ids)}  {(time.time() - t0) / done:.1f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def contrasts():
    d = {"Q1 OSFD REF − PUB": {"OSFD-REF": 1, "OSFD-PUB": -1}}
    for e in ("ENS-RD-AVG", "ENS-RD-GN", "ENS-OSFD-AVG", "ENS-OSFD-GN"):
        d[f"Q2 ENS-WP − {e}"] = {"ENS-WP": 1, e: -1}
    for e in ("ENS-RD-AVG", "ENS-RD-GN"):
        d[f"{e} − RD-CNX"] = {e: 1, "RD-CNX": -1}
    for e in ("ENS-OSFD-AVG", "ENS-OSFD-GN"):
        d[f"{e} − OSFD-PUB"] = {e: 1, "OSFD-PUB": -1}
    d["ENS-WP − RD-CNX"] = {"ENS-WP": 1, "RD-CNX": -1}
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="dev100", choices=["dev100", "smoke3"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cells = args.cells.split(",") if args.cells else CELLS
    assert all(c in CELLS for c in cells), cells
    gpu = torch.cuda.get_device_name(0)
    mtag = gpu.split()[-1].replace("-", "").lower()
    sfx = f"_{args.tag}" if args.tag else ""
    run_dir = f"e88_ensbase_eps5_{mtag}{sfx}"
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/e88_ensbase_{args.list}{sfx}.json")
    print(f"[e88] GPU = {gpu} | run dir {run_dir} | ô {cells}", flush=True)
    ids = image_ids(args.list)
    for c in cells:
        generate(ids, c, run_dir)
    if args.gen_only:
        return
    assert set(cells) == set(CELLS)

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])
    npz = out_json.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
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
            with gzip.open(os.path.join(dets_dir, key, f"{name}.json.gz"), "wt") as f:
                json.dump(dets, f)
            ev = CocoImageEval(coco, dets, ids)
            return np.array([ev.ap(w)[0] for w in W]) * 100
        clean = ap_of("clean", [os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"]) for i in ids])
        for c in CELLS:
            adv = ap_of(c, [os.path.join(adv_dir(run_dir, c), f"{i}.png") for i in ids])
            S[f"{key}|{c}"] = (100 * (clean - adv) / clean).astype(np.float64)
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)

    groups = {m: [m] for m in PANEL}
    groups.update(GROUPS)
    G = {g: {c: np.mean([S[f"{m}|{c}"] for m in ms], axis=0) for c in CELLS} for g, ms in groups.items()}
    res = {"list": args.list, "n_images": n, "n_boot": args.boot, "gpu": gpu, "run_dir": run_dir,
           "drop": {}, "contrast": {}, "groups": groups}
    for g in groups:
        res["drop"][g] = {c: summarize(G[g][c][0], G[g][c][1:]) for c in CELLS}
        res["contrast"][g] = {k: summarize(*(lambda v: (v[0], v[1:]))(sum(w * G[g][c] for c, w in co.items())))
                              for k, co in contrasts().items()}
    # Quy tắc khóa: Ens-RD đối chứng chính = ô MẠNH HƠN theo điểm T7 (bất lợi cho ENS-WP)
    strong = max(("ENS-RD-AVG", "ENS-RD-GN"), key=lambda c: G["T7"][c][0])
    res["ens_rd_strong"] = strong
    res["contrast_primary"] = {g: summarize(*(lambda v: (v[0], v[1:]))(G[g]["ENS-WP"] - G[g][strong]))
                               for g in groups}
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    show = PANEL + list(GROUPS)
    print(f"\n=== E88 @ {args.list}, {gpu}, boot {args.boot} — relative AP drop % ===")
    print(f"{'cell':13}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in CELLS:
        print(f"{c:13}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    print("\n--- contrast ---")
    for k in contrasts():
        print(f"{k:28}" + " | ".join(f"{g[:8]} {fmt(res['contrast'][g][k])}" for g in ["T7", "dino_swin_l", "G1", "G2cnn", "TF"]))
    pr = res["contrast_primary"]["T7"]
    print(f"\nQ2 PRIMARY: ENS-WP − {strong} (T7) = {fmt(pr)} | Swin-L {fmt(res['contrast_primary']['dino_swin_l'])} -> "
          + ("CI loại 0: phối hợp có lợi thêm" if pr["ci95"][0] > 0 else "CI chứa 0 (hoặc âm): chỉ claim lợi do 2 surrogate"))
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[e88] -> {out_json}")


if __name__ == "__main__":
    main()
