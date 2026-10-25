#!/usr/bin/env python3
"""E32 (progress_log 2026-10-25): WP2 thắng nhờ TÍNH CHẤT gì của δ? δ_W (C2-OCT-1V-WP2 B36) vs δ_C (C2-OCT-1V B41), dev100,
forward-only trên PNG có sẵn (HF). Mỗi ảnh, X ∈ {C, W}:
(a) độ phẳng weight-space R50: G(θ; X) = L(x + δ_X; θ) − L(x; θ), L = L_OSFD (k = 3, feature sạch tại CÙNG θ) + 4·C2 cls;
    r_ρ = TB_ξ G(θ0 + ρ‖θ0‖ξ̂) / G(θ0), ρ ∈ RHOS, 4 ξ / ρ (CÙNG ξ cho C, W); r_def: θ' = WP2 phòng thủ 1 bước. G trên R101.
(b) D_k backbone stage 1–4 trên R50 / R101 / ConvNeXt-T / Swin-T / DINO-Swin-L; T_k = D_k(m) / D_k(R50).
(c) DINO-Swin-L cosd backbone / neck / encoder, A_enc = cosd_enc / cosd_bb4.
(d) δ: 3 dải tần, cos(δ_W, δ_C), cos(δ_X, δ_S), năng lượng trong box GT; loại lỗi (dets có sẵn, như err_profile).
Chạy: python scripts/e32_wp2_dissect.py   (--limit n: smoke test, không đọc số)"""
import argparse
import json
import os
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import boot_mean, boot_spearman, boot_weights, per_image_transfer
from attack.methods.core import wp_update
from attack.methods.repair import cls_term
from attack.models import load_model
from attack.preprocess import pipeline_resize, to_batch
from attack.targets import load_any
from scripts.err_profile import profile

ART = os.path.join(REPO_ROOT, "artifacts/runs")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SRC = {"C": ("dev300_B41_eps5_limit100", "C2-OCT-1V", "dets_c2_1v_dev100.json"),
       "W": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2", "dets_wpa_dev100.json")}
S_SRC = ("dev300_B41_eps5_limit100_sur-swin_t", "C2-OCT-1V")
K, LAM, RHO_DEF = 3.0, 4.0, 0.002
RHOS, N_XI = (0.002, 0.005, 0.01, 0.02), 4
MASK_RCNN = ("r50", "r101", "convnext_t", "swin_t")


def conv_params(model):
    return [m.weight for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad]


def attack_loss(model, pix, ds, fc):
    """L_OSFD(k; fc) + LAM·C2 cls tại θ hiện tại -> (L, feat backbone). fc None -> fc = feature của chính pix."""
    cap = {}
    h = model.backbone.register_forward_hook(lambda m, i, o: cap.update(f=o))
    try:
        torch.manual_seed(0)  # RoI sampling cùng hạt giống ở mọi lần gọi
        bi, bds = to_batch(model, [pix], [ds])
        c = cls_term(model, bi, bds, "c2")
    finally:
        h.remove()
    fa = cap["f"]
    fc = tuple(f.detach() for f in fa) if fc is None else fc
    return sum(F.mse_loss(K * a, b) for a, b in zip(fc, fa)) + LAM * c, fc


@torch.no_grad()
def gains(model, x, advs, ds):
    """{X: G} tại θ hiện tại; L sạch tính 1 lần, dùng chung."""
    lc, fc = attack_loss(model, x, ds, None)
    return {X: float(attack_loss(model, a, ds, fc)[0] - lc) for X, a in advs.items()}


def set_params(params, vals):
    with torch.no_grad():
        for p, v in zip(params, vals):
            p.copy_(v)


def flatness(model, params, base, x, advs, ds, gen):
    out = {X: {"G0": None} for X in advs}
    g0 = gains(model, x, advs, ds)
    for X in advs:
        out[X]["G0"] = g0[X]
    for rho in RHOS:
        acc = defaultdict(list)
        for _ in range(N_XI):
            vals = []
            for p0 in base:
                xi = torch.randn(p0.shape, generator=gen, device=p0.device, dtype=p0.dtype)
                vals.append(p0 + rho * p0.norm() * xi / (xi.norm() + 1e-12))
            set_params(params, vals)
            for X, g in gains(model, x, advs, ds).items():
                acc[X].append(g)
        for X in advs:
            out[X][f"r_{rho}"] = float(np.mean(acc[X]) / g0[X])
    set_params(params, base)
    for X, a in advs.items():  # r_def: WP2 phòng thủ 1 bước chống CHÍNH δ_X
        _, fc = attack_loss(model, x, ds, None)
        L, _ = attack_loss(model, a, ds, tuple(f.detach() for f in fc))
        gt = torch.autograd.grad(L, params)
        set_params(params, [wp_update(p0, p0, g, RHO_DEF) for p0, g in zip(base, gt)])
        out[X]["r_def"] = float(gains(model, x, {X: a}, ds)[X] / g0[X])
        set_params(params, base)
    return out


@torch.no_grad()
def levels(key, model, pix, ds):
    bi, dss = to_batch(model, [pix], [ds])
    bb = model.backbone(bi)
    out = {"bb": list(bb)}
    if key == "dino_swin_l":
        neck = model.neck(bb)
        enc_in, _ = model.pre_transformer(neck, dss)
        out["neck"] = list(neck)
        out["enc"] = [model.forward_encoder(**enc_in)["memory"]]
    return out


def dist(a, c):
    a, c = a.flatten().double(), c.flatten().double()
    return float((a - c).norm() / c.norm()), float(1 - torch.dot(a, c) / (a.norm() * c.norm()))


def bands(d):
    """d [3, H, W] -> năng lượng 3 dải radial (thấp < 1/6, trung < 1/3, cao), chuẩn hóa tổng = 1 (như A4)."""
    e = (torch.fft.fft2(d.double()).abs() ** 2).sum(0)
    fy = torch.fft.fftfreq(d.shape[1], device=d.device).view(-1, 1)
    fx = torch.fft.fftfreq(d.shape[2], device=d.device).view(1, -1)
    f = torch.sqrt(fx ** 2 + fy ** 2)
    tot = e.sum()
    return [float(e[f < 1 / 6].sum() / tot), float(e[(f >= 1 / 6) & (f < 1 / 3)].sum() / tot), float(e[f >= 1 / 3].sum() / tot)]


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def box_ratio(d, anns):
    m = torch.zeros(d.shape[1:], dtype=torch.bool, device=d.device)
    for a in anns:
        x, y, w, h = a["bbox"]
        m[int(y):int(np.ceil(y + h)), int(x):int(np.ceil(x + w))] = True
    frac = float(m.float().mean())
    if frac == 0 or frac == 1:
        return np.nan
    e = (d.double() ** 2).sum(0)
    return float(e[m].sum() / e.sum()) / frac


def load_dets(model, run, name, dets_file):
    if model in ("yolox_s", "dino_swin_l"):
        d = json.load(open(os.path.join(ART, run, "gen_dets", f"{model}.json")))[name]
    else:
        d = json.load(open(os.path.join(ART, run, dets_file)))[name][model]
    out = defaultdict(list)
    for x in d:
        if x["score"] >= 0.3:
            out[x["image_id"]].append(x)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/mechanism/e32_wp2_dissect.json"))
    args = p.parse_args()
    dev = "cuda:0"
    coco = COCO(ANN_FILE)
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.limit]
    models = {k: load_model(k, dev) for k in MASK_RCNN}
    models["dino_swin_l"] = load_any("dino_swin_l", dev)
    params, params101 = conv_params(models["r50"]), conv_params(models["r101"])
    base, base101 = [p.detach().clone() for p in params], [p.detach().clone() for p in params101]

    rec = defaultdict(lambda: defaultdict(list))  # rec[X][đại lượng] = list theo ảnh
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds, orig = s["inputs"].to(dev).float(), s["data_sample"], s["orig_inputs"].to(dev).float()
        png = {X: torch.from_numpy(mmcv.imread(os.path.join(ART, r, "adv", m, f"{img_id}.png"))).permute(2, 0, 1)
               .float().to(dev) for X, (r, m, _) in SRC.items()}
        png_s = torch.from_numpy(mmcv.imread(os.path.join(ART, S_SRC[0], "adv", S_SRC[1], f"{img_id}.png"))).permute(
            2, 0, 1).float().to(dev)
        advs = {X: pipeline_resize(v, tuple(x.shape[-2:])) for X, v in png.items()}
        # (a)
        gen = torch.Generator(device=dev).manual_seed(2026 + img_id)
        fl = flatness(models["r50"], params, base, x, advs, ds, gen)
        g101 = gains(models["r101"], x, advs, ds)
        for X in SRC:
            for q, v in fl[X].items():
                rec[X][q].append(v)
            rec[X]["G_r101"].append(g101[X])
        # (b) + (c)
        for key, model in models.items():
            cl = levels(key, model, x, ds)
            for X, a in advs.items():
                lv = levels(key, model, a, ds)
                for name, feats in lv.items():
                    for li, (fa, fc) in enumerate(zip(feats, cl[name])):
                        r, cd = dist(fa, fc)
                        rec[X][f"D_{key}_{name}{li}"].append(r)
                        rec[X][f"cosd_{key}_{name}{li}"].append(cd)
        # (d)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        dS = png_s - orig
        dlt = {X: v - orig for X, v in png.items()}
        for X, d in dlt.items():
            for bname, v in zip(("band_lo", "band_mid", "band_hi"), bands(d)):
                rec[X][bname].append(v)
            rec[X]["cos_S"].append(cos(d, dS))
            rec[X]["box_ratio"].append(box_ratio(d, anns))
            rec[X]["l2"].append(float(d.norm()))
        rec["W"]["cos_WC"].append(cos(dlt["W"], dlt["C"]))
        if (j + 1) % 10 == 0:
            print(f"[e32] {j + 1}/{len(img_ids)}  r_0.01 C {np.mean(rec['C']['r_0.01']):.3f} W {np.mean(rec['W']['r_0.01']):.3f}",
                  flush=True)

    n = len(img_ids)
    bidx = boot_weights(n, 1000, 2026)
    A = {X: {q: np.array(v, dtype=float) for q, v in rec[X].items()} for X in SRC}
    for X in SRC:  # chỉ số dẫn xuất theo ảnh
        for key in models:
            for k in range(4):
                A[X][f"T_{key}_{k}"] = A[X][f"D_{key}_bb{k}"] / A[X][f"D_r50_bb{k}"]
        A[X]["A_enc"] = A[X]["cosd_dino_swin_l_enc0"] / A[X]["cosd_dino_swin_l_bb3"]
    sup = {}
    for X, (r, m, _) in SRC.items():
        g = json.load(open(os.path.join(ART, r, "gen_dets", "dino_swin_l.json")))
        sup[X] = per_image_transfer(coco, g["clean"], g[m], img_ids)["suppression"]
    res = {"n_images": n, "C": {}, "W": {}, "delta_W_minus_C": {}, "cos_WC": boot_mean(np.array(rec["W"]["cos_WC"]), bidx)}
    for q in A["C"]:
        res["C"][q] = float(np.nanmean(A["C"][q]))
        res["W"][q] = float(np.nanmean(A["W"][q]))
        res["delta_W_minus_C"][q] = boot_mean(A["W"][q] - A["C"][q], bidx)
    res["sup_swinl"] = {"C": float(np.nanmean(sup["C"])), "W": float(np.nanmean(sup["W"])),
                        "delta": boot_mean(sup["W"] - sup["C"], bidx)}
    # Quy tắc đọc (khóa progress_log 2026-10-25)
    dr = {rho: res["delta_W_minus_C"][f"r_{rho}"] for rho in RHOS[1:]}
    rho_star = max(dr, key=lambda r: dr[r]["point"])
    flat_ok = any(v["point"] >= 0.05 and v["ci95"][0] > 0 for v in dr.values())
    sp = boot_spearman(A["W"][f"r_{rho_star}"] - A["C"][f"r_{rho_star}"], sup["W"] - sup["C"], bidx)
    feat = {k: res["delta_W_minus_C"][f"T_dino_swin_l_{k}"] for k in range(4)}
    rep = res["delta_W_minus_C"]["A_enc"], res["delta_W_minus_C"]["cosd_dino_swin_l_enc0"]
    res["verdict"] = {
        "H_flat": "ỦNG HỘ" if flat_ok else "BÁC", "rho_star": rho_star, "spearman_dr_dsup": sp,
        "H_flat_predicts": bool(flat_ok and sp["ci95"][0] > 0),
        "H_feat": "ỦNG HỘ" if any(v["point"] >= 0.05 and v["ci95"][0] > 0 for v in feat.values()) else "BÁC",
        "H_repair": "ỦNG HỘ" if (rep[0]["ci95"][0] > 0 and rep[1]["point"] >= 0.01) else "BÁC"}
    # Loại lỗi (CPU)
    res["err"] = {}
    for m in ("swin_t", "yolox_s", "dino_swin_l"):
        cp, ch = profile(coco, load_dets(m, SRC["W"][0], "clean", SRC["W"][2]), img_ids)
        res["err"][m] = {"clean": cp}
        for X, (r, name, f) in SRC.items():
            res["err"][m][X], _ = profile(coco, load_dets(m, r, name, f), img_ids, ch)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("saved", args.out)
    D = res["delta_W_minus_C"]
    fmt = lambda v: f"{v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
    print("(a)", " | ".join(f"{q}: C {res['C'][q]:.3f} W {res['W'][q]:.3f} Δ {fmt(D[q])}"
                            for q in [f"r_{r}" for r in RHOS] + ["r_def", "G0", "G_r101"]))
    for key in models:
        print(f"(b) {key}:", " | ".join(f"D{k + 1} C {res['C'][f'D_{key}_bb{k}']:.3f} W {res['W'][f'D_{key}_bb{k}']:.3f}"
                                        f" ΔT {fmt(D[f'T_{key}_{k}'])}" for k in range(4)))
    print("(c) cosd DINO:", " | ".join(f"{q}: C {res['C'][q]:.3f} W {res['W'][q]:.3f}" for q in res["C"]
                                       if q.startswith("cosd_dino")), "| A_enc Δ", fmt(D["A_enc"]))
    print("(d)", " | ".join(f"{q}: C {res['C'][q]:.3f} W {res['W'][q]:.3f}" for q in
                            ("band_lo", "band_mid", "band_hi", "cos_S", "box_ratio", "l2")), "| cos_WC", fmt(res["cos_WC"]))
    print("sup Swin-L", res["sup_swinl"]["C"], res["sup_swinl"]["W"], "| verdict", json.dumps(res["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
