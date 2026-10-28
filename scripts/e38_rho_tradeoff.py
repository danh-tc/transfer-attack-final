#!/usr/bin/env python3
"""E38 (progress_log 2026-10-28): quét liều ρ của WP — đánh đổi "mượt ↔ tương đồng gradient" (S&P'24 "little robustness").
Điểm chung P1 = PNG ô ρ = 0 (C2-OCT-1V B36), mô tả P0 = ảnh sạch. Mỗi ảnh, mỗi ρ ∈ RHOS:
θ'_ρ = phòng thủ 1 bước (core.wp_update, ∇θ của L tại P); g_ρ = ∇ theo ảnh gốc của L tại θ'_ρ (L = L_OSFD k = 3 + 4·C2 cls,
feature sạch tại CÙNG θ', view đồng nhất);
  Sm(ρ)  = TB cos(g_ρ(P + ξ), g_ρ(P)), ξ Rademacher ±2 mức xám (CÙNG ξ mọi ρ)  — độ mượt
  Sim(ρ) = cos(g_ρ(P), g_T(P)), g_T = ∇ loss cls GT của target (targets.target_grads) — tương đồng gradient
Quy tắc ở progress_log entry E38. Chạy sau khi có PNG + generalization_e38_dev30.json:
  python scripts/e38_rho_tradeoff.py            (--limit n: smoke test, không đọc số)"""
import argparse
import json
import os
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import boot_mean, boot_weights
from attack.methods.core import wp_update
from attack.models import load_model
from attack.preprocess import resize_to_model
from attack.targets import load_any, target_dataset, target_grads
from scripts.e32_wp2_dissect import attack_loss, conv_params, cos, set_params

RUN = "dev300_B36_eps5_limit100"
ART = os.path.join(REPO_ROOT, "artifacts/runs", RUN)
CELLS = {0.0: "C2-OCT-1V", 0.001: "C2-OCT-1V-WP1", 0.002: "C2-OCT-1V-WP2", 0.005: "C2-OCT-1V-WP5", 0.01: "C2-OCT-1V-WP10"}
RHOS = tuple(CELLS)
TARGETS = ("dino_swin_l", "swin_t", "r101")  # dino_swin_l: quy tắc; còn lại mô tả
N_XI, XI_AMP = 2, 2.0


def grad_x(model, v0, ds, fc):
    v = v0.clone().requires_grad_(True)
    L, _ = attack_loss(model, resize_to_model(v, tuple(ds.img_shape)), ds, fc)
    return torch.autograd.grad(L, v)[0].detach()


def clean_fc(model, x, ds):
    with torch.no_grad():
        return attack_loss(model, x, ds, None)[1]


def sign_agree(a, b):
    return float((torch.sign(a) == torch.sign(b)).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/mechanism/e38_rho_tradeoff.json"))
    args = p.parse_args()
    dev = "cuda:0"
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:args.limit]

    # Pha 1: gradient target tại P0 / P1 (không phụ thuộc ρ), nạp từng target 1 lần, giữ CPU.
    gT = defaultdict(dict)  # gT[(target, point)][img_id]
    for key in TARGETS:
        model = load_any(key, dev)
        tds = target_dataset(key, img_ids)
        for j, img_id in enumerate(img_ids):
            orig = data[pos[img_id]]["orig_inputs"].to(dev).float()
            png = torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", CELLS[0.0], f"{img_id}.png"))).permute(2, 0, 1)
            for pt, v in (("P0", orig), ("P1", png.float().to(dev))):
                g, _, _ = target_grads(model, key, v, tds[j]["data_sample"], ("cls",))
                gT[(key, pt)][img_id] = g["cls"].cpu()
        del model
        torch.cuda.empty_cache()
        print(f"[e38] target grads {key} xong", flush=True)

    # Pha 2: R50, mỗi ρ.
    model = load_model("r50", dev)
    params = conv_params(model)
    base = [q.detach().clone() for q in params]
    rec = defaultdict(list)  # rec[(đại lượng, điểm, ρ)] = list theo ảnh
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds, orig = s["inputs"].to(dev).float(), s["data_sample"], s["orig_inputs"].to(dev).float()
        png = torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", CELLS[0.0], f"{img_id}.png"))).permute(2, 0, 1)
        gen = torch.Generator(device=dev).manual_seed(2026 + img_id)
        xis = [(torch.randint(0, 2, orig.shape, generator=gen, device=dev).float() * 2 - 1) * XI_AMP for _ in range(N_XI)]
        for pt, P in (("P0", orig), ("P1", png.float().to(dev))):
            # ∇θ tại θ0 (hướng phòng thủ chung cho mọi ρ)
            set_params(params, base)
            fc0 = clean_fc(model, x, ds)
            L, _ = attack_loss(model, resize_to_model(P, tuple(ds.img_shape)), ds, fc0)
            gth = torch.autograd.grad(L, params)
            for rho in RHOS:
                set_params(params, base if rho == 0 else [wp_update(p0, p0, g, rho) for p0, g in zip(base, gth)])
                fc = clean_fc(model, x, ds)
                g = grad_x(model, P, ds, fc)
                rec[("Sm", pt, rho)].append(np.mean([cos(grad_x(model, torch.clamp(P + xi, 0, 255), ds, fc), g) for xi in xis]))
                for key in TARGETS:
                    t = gT[(key, pt)][img_id].to(dev)
                    rec[(f"Sim_{key}", pt, rho)].append(cos(g, t))
                    rec[(f"Sgn_{key}", pt, rho)].append(sign_agree(g, t))
        set_params(params, base)
        if (j + 1) % 5 == 0:
            print(f"[e38] {j + 1}/{len(img_ids)}  P1 Sm " + " ".join(f"{r}:{np.mean(rec[('Sm', 'P1', r)]):.3f}" for r in RHOS)
                  + " | Sim_DINO " + " ".join(f"{r}:{np.mean(rec[('Sim_dino_swin_l', 'P1', r)]):.4f}" for r in RHOS), flush=True)
    assert all(torch.equal(q, b) for q, b in zip(params, base)), "θ0 không được khôi phục"

    n = len(img_ids)
    bidx = boot_weights(n, 1000, 2026)
    A = {k: np.array(v, dtype=float) for k, v in rec.items()}
    res = {"n_images": n, "rhos": list(RHOS), "mean": {}, "diff": {}}
    for (q, pt, rho), v in A.items():
        res["mean"][f"{q}|{pt}|{rho}"] = float(np.nanmean(v))
    D = lambda q, pt, a, b: boot_mean(A[(q, pt, a)] - A[(q, pt, b)], bidx)
    for pt in ("P0", "P1"):
        for q in ["Sm"] + [f"Sim_{k}" for k in TARGETS]:
            res["diff"][f"{q}|{pt}|0.01-0"] = D(q, pt, 0.01, 0.0)
            res["diff"][f"{q}|{pt}|0.01-0.002"] = D(q, pt, 0.01, 0.002)
            res["diff"][f"{q}|{pt}|0.002-0"] = D(q, pt, 0.002, 0.0)

    # Quy tắc (khóa progress_log 2026-10-28, tại P1)
    sm_means = [res["mean"][f"Sm|P1|{r}"] for r in RHOS]
    sp_sm = float(spearmanr(RHOS, sm_means)[0])
    d_sm = res["diff"]["Sm|P1|0.01-0"]
    d_sim_hi = res["diff"]["Sim_dino_swin_l|P1|0.01-0.002"]
    d_sim_wp2 = res["diff"]["Sim_dino_swin_l|P1|0.002-0"]
    gfile = os.path.join(REPO_ROOT, "results/runs", RUN, "generalization_e38_dev30.json")
    drop = None
    if os.path.exists(gfile):
        g = json.load(open(gfile))["relative_ap_drop"]["dino_swin_l"]
        drop = {r: g[m]["point"] for r, m in CELLS.items() if m in g}
    c1 = d_sm["point"] > 0 and d_sm["ci95"][0] > 0 and sp_sm >= 0.9
    c2 = d_sim_hi["point"] < 0 and d_sim_hi["ci95"][1] < 0
    c3 = drop is not None and len(drop) == len(RHOS) and max(drop, key=drop.get) in (0.001, 0.002, 0.005)
    if c1 and c2 and c3:
        v = "ỦNG HỘ"
    elif not c1 or d_sim_hi["point"] >= 0:
        v = "BÁC"
    else:
        v = "XÁM"
    sim0 = res["mean"]["Sim_dino_swin_l|P1|0.0"]
    res["verdict"] = {"H_tradeoff": v, "c1_smooth_up": c1, "spearman_rho_Sm": sp_sm, "c2_sim_drops_high": c2,
                      "c3_peak_in_mid": c3, "drop_dino_swin_l": drop,
                      "H_sim": "ỦNG HỘ" if (d_sim_wp2["ci95"][0] > 0 and d_sim_wp2["point"] >= 0.1 * abs(sim0)) else "BÁC"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("saved", args.out)
    fmt = lambda d: f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
    for pt in ("P1", "P0"):
        for q in ["Sm"] + [f"Sim_{k}" for k in TARGETS] + [f"Sgn_{k}" for k in TARGETS]:
            print(f"{pt} {q}: " + " ".join(f"{r}:{res['mean'][f'{q}|{pt}|{r}']:.4f}" for r in RHOS))
        for k, d in res["diff"].items():
            if k.split("|")[1] == pt:
                print(f"  Δ {k}: {fmt(d)}")
    print("verdict", json.dumps(res["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
