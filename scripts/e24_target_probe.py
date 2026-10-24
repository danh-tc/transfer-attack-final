#!/usr/bin/env python3
"""E24 (progress_log 2026-10-24): có thước đo φ chỉ-dùng-R50 nào xếp δ theo mức phá DINO-Swin-L không?
Forward-only trên PNG dev30 đã có (KHÔNG attack mới). y = suppression DINO-Swin-L theo ảnh.
Chạy: python scripts/e24_target_probe.py"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.mechanism import boot_spearman, boot_weights, per_image_transfer
from attack.methods.core import wp_update
from attack.methods.repair import cls_term
from attack.models import load_model
from attack.preprocess import extract_features, pipeline_resize, resize_to_model, to_batch

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
K, LAM, RHO = 3.0, 4.0, 0.002
# (run, method) của 23 δ dev30 — tất cả đã có PNG + gen_dets/dino_swin_l.json
DELTAS = [
    ("dev300_B41_eps5_limit100", "C2-OCT-1V"),
    ("dev300_B41_eps5_limit100_sur-swin_t", "C2-OCT-1V"),  # δ_SwinT (oracle)
    *[("dev300_B36_eps5_limit100", m) for m in (
        "C2-OCT-1V-WP2", "C2-OCT-1V-WP2-RAND", "C2-OCT-1V-WP2-ASC", "C2-OCT-1V-WP2-L3",
        "C2-OCT-1V-WP2-noL3", "C2-OCT-1V-WPH", "C2-OCT-1V-WP2+H", "C2-OCT-1V-WP2-C2W03",
        "C2-OCT-1V-WP2-C2W1", "C2-OCT-1V-WP2-PFA50", "C2-OCT-1V-WP2-PFA75")],
    ("dev300_B33_eps5_limit100", "C2-OCT-1V-WP2-DEFCLS"),
    ("dev300_B34_eps5_limit100", "C2-OCT-1V-WP2-DEFFEAT"),
    *[(f"dev300_B{b}_eps5_limit100", m) for b in (1, 4, 12, 36)
      for m in ("C2-OCT-1V-WP2-IS", "C2-OCT-1V-WP2-OTSV50-IS")],
]
IS_TRAJ = [(f"dev300_B{b}_eps5_limit100", "C2-OCT-1V-WP2-IS", b) for b in (1, 4, 12, 36)]


def backbone_conv_params(model):
    return [m.weight for m in model.backbone.modules()
            if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad]


def l_plain(model, x, ds, fc, feat_only=False, cls_only=False):
    """L_OSFD(k) + LAM·cls trên θ hiện tại. feat_only / cls_only: chỉ 1 thành phần."""
    cap = {}
    h = model.backbone.register_forward_hook(lambda m, i, o: cap.update(f=o))
    try:
        bi, bds = to_batch(model, [x], [ds])
        c = 0.0 if feat_only else cls_term(model, bi, bds, "c2")
    finally:
        h.remove()
    feat = 0.0 if cls_only else sum(F.mse_loss(K * a, b) for a, b in zip(fc, cap["f"]))
    return (0.0 if cls_only else feat) + (0.0 if feat_only else LAM * c), cap["f"]


def measures(model, x, ds, fc, params, base, rng, n_mc=8):
    out = {}
    # L_plain + cosd + cls_only (1 forward có grad để lấy ∇θ cho def)
    xr = x.clone().requires_grad_(True)
    L, fa = l_plain(model, xr, ds, fc)
    out["L_plain"] = float(L.detach())
    out["cosd_C45"] = float(np.mean([1 - F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
                                     for a, b in list(zip(fc, fa))[2:]]))
    gtheta = torch.autograd.grad(L, params, retain_graph=False)
    with torch.no_grad():
        out["L_cls_only"] = float(l_plain(model, x, ds, fc, cls_only=True)[0])
        # L_def1 / L_def3: θ' phòng thủ (giảm L), đo lại L tại θ'
        orig = [p.detach().clone() for p in params]
        for p, p0, g in zip(params, base, gtheta):
            p.copy_(wp_update(p, p0, g, RHO))
        out["L_def1"] = float(l_plain(model, x, ds, fc)[0])
        for p, o in zip(params, orig):
            p.copy_(o)
    # L_def3: 3 bước tích lũy
    with torch.no_grad():
        orig = [p.detach().clone() for p in params]
    th = [p0.clone() for p0 in base]
    for _ in range(3):
        for p, t in zip(params, th):
            p.data.copy_(t)
        xr2 = x.clone().requires_grad_(True)
        L3, _ = l_plain(model, xr2, ds, fc)
        g3 = torch.autograd.grad(L3, params)
        with torch.no_grad():
            for ti, (p0, g) in enumerate(zip(base, g3)):
                th[ti] = wp_update(th[ti], p0, g, RHO / 3)
    with torch.no_grad():
        for p, t in zip(params, th):
            p.data.copy_(t)
        out["L_def3"] = float(l_plain(model, x, ds, fc)[0])
        for p, o in zip(params, orig):
            p.copy_(o)
        # L_rand: cắt tham số Gauss lớn (2 ρ × n_mc/2 mẫu)
        vals = []
        for rho in (0.01, 0.02):
            for _ in range(n_mc // 2):
                for p, p0 in zip(params, base):
                    r = torch.randn(p0.shape, generator=rng, device=p0.device, dtype=p0.dtype)
                    p.copy_(p0 + rho * p0.norm() * r / (r.norm() + 1e-12))
                vals.append(l_plain(model, x, ds, fc)[0])
        out["L_rand"] = float(np.mean([float(v) for v in vals]))
        for p, o in zip(params, base):
            p.copy_(o)
        # L_octbig: L_OSFD dưới co giãn ±1 octave + blur (feature-only, GT không cần)
        import math
        from attack.methods.pia import _blur, warp, warp_grid
        from attack.methods.elem import linear_homography, shift_h
        hh, ww = x.shape[-2:]
        ov = []
        for _ in range(n_mc):
            sc = math.exp((rng_py.random() * 2 - 1) * math.log(2))
            Hm = linear_homography(torch.tensor([[sc, 0.0], [0.0, sc]], dtype=torch.float64), hh, ww) @ shift_h(
                rng_py.randrange(8), rng_py.randrange(8))
            grid = warp_grid(Hm, hh, ww, x.device)
            xv = _blur(warp(x, grid), rng_py.choice([0.0, 2.0, 4.0]))
            cv = warp(clean_cache["c"], grid)
            with torch.enable_grad():
                pass
            fcv = extract_features(model, cv, ds)
            fav = extract_features(model, xv, ds)
            ov.append(float(sum(F.mse_loss(K * a, b) for a, b in zip(fcv, fav))))
        out["L_octbig"] = float(np.mean(ov))
    return out


def main():
    global rng_py, clean_cache
    import random
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="dev30")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/mechanism/e24_target_probe.json"))
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "results/e24"), exist_ok=True)
    coco = COCO(ANN_FILE)
    img_ids = load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.subset}.csv"))
    ds_all = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
    idx = {i: j for j, i in enumerate(ds_all.image_ids)}
    model = load_model("r50", "cuda:0")
    params = backbone_conv_params(model)
    base = [p.detach().clone() for p in params]
    rng = torch.Generator(device="cuda:0").manual_seed(2026)
    rng_py = random.Random(2026)
    clean_cache = {}
    MEAS = ["L_plain", "L_def1", "L_def3", "L_rand", "L_octbig", "cosd_C45", "L_cls_only"]

    # φ[meas][delta_key] = np.array theo ảnh; y[delta_key] = suppression Swin-L theo ảnh
    phi = {m: {} for m in MEAS}
    y = {}
    for run, method in DELTAS:
        key = f"{run}:{method}"
        gen = json.load(open(os.path.join(REPO_ROOT, "artifacts/runs", run, "gen_dets", "dino_swin_l.json")))
        y[key] = per_image_transfer(coco, gen["clean"], gen[method], img_ids)["suppression"]
        rows = {m: [] for m in MEAS}
        for i in img_ids:
            ds = ds_all[idx[i]]["data_sample"]
            png = os.path.join(REPO_ROOT, "artifacts/runs", run, "adv", method, f"{i}.png")
            adv_orig = torch.from_numpy(mmcv.imread(png)).permute(2, 0, 1).float().to("cuda:0")
            x = pipeline_resize(adv_orig, tuple(ds.img_shape))
            if i not in clean_cache:
                clean_cache[i] = resize_to_model(ds_all[idx[i]]["orig_inputs"].to("cuda:0"), tuple(ds.img_shape))
            clean_cache["c"] = clean_cache[i]
            with torch.no_grad():
                fc = tuple(f.detach() for f in extract_features(model, clean_cache[i], ds))
            mv = measures(model, x, ds, fc, params, base, rng)
            for m in MEAS:
                rows[m].append(mv[m])
        for m in MEAS:
            phi[m][key] = np.array(rows[m])
        print(f"{key}: y(sup Swin-L) {np.nanmean(y[key]):.3f}", flush=True)

    # Spearman gộp mọi δ (mỗi (ảnh, δ) là 1 điểm); bootstrap theo ẢNH (paired, cùng idx mọi meas)
    n = len(img_ids)
    bidx = boot_weights(n, 1000, 2026)
    keys = [f"{r}:{m}" for r, m in DELTAS]
    Y = np.concatenate([y[k] for k in keys])
    bidx_flat = np.concatenate([bidx + d * n for d in range(len(keys))], axis=1)  # resample ảnh, giữ mọi δ
    res = {"n_images": n, "n_deltas": len(keys), "measures": {}}
    for m in MEAS:
        P = np.concatenate([phi[m][k] for k in keys])
        res["measures"][m] = boot_spearman(P, Y, bidx_flat)
        print(f"rho({m}, sup Swin-L) = {res['measures'][m]['point']:.3f} "
              f"[{res['measures'][m]['ci95'][0]:.3f}, {res['measures'][m]['ci95'][1]:.3f}]", flush=True)

    # Trên quỹ đạo IS (B1→B36): φ có giảm khi Swin-L sụp không? Spearman(φ_trung-bình-ảnh, B)
    bs = [b for _, _, b in IS_TRAJ]
    res["is_trajectory"] = {}
    for m in MEAS:
        means = [float(np.nanmean(phi[m][f"{r}:{mm}"])) for r, mm, _ in IS_TRAJ]
        from scipy.stats import spearmanr
        res["is_trajectory"][m] = {"phi_by_B": dict(zip(bs, [round(v, 2) for v in means])),
                                   "spearman_phi_B": float(spearmanr(bs, means)[0])}
    res["is_swinl_by_B"] = {b: round(float(np.nanmean(y[f"{r}:{mm}"])), 3) for r, mm, b in IS_TRAJ}
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
