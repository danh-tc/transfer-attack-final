#!/usr/bin/env python3
"""E19 — oracle-δ dissection (quy tắc khóa: progress_log 2026-10-23 "E19"): trên CHÍNH R50, cách ước lượng hướng nào
(chỉ dùng R50) trỏ về δ_S = δ của C2-OCT-1V surrogate Swin-T (B41)? δ_S chỉ là oracle để phân tích.
δ_W = PNG C2-OCT-1V-WP2 B36 (surrogate R50) − ảnh gốc. Mọi δ / gradient ở không gian ảnh GỐC (như `noise` của attack).
Loss C2 = Σ_stage MSE(3·F_sạch_view, F_adv_view) + 4·(loss_rpn_cls + loss_cls), skip_mask, fp32; 8 view OCT cố định / ảnh
(ln s ~ U(±ln2/2), pha ph8, nhiễu σ = 6 cố định theo view, GT co-transform, RoI sampler seed theo view).
  A_S(h) = cos(sign h, δ_S), A_W(h) = cos(sign h, δ_W) tại P0 = ảnh sạch (dev100) và P1 = x + δ_W (dev30, mô tả).
  Độ phẳng: r(δ) = [TB_j L̄(δ + ξ_j) − L̄(0)] / Γ(δ), ξ Rademacher ±2 px (không kẹp ε), δ_S so t*·δ_W cùng mức Γ.
  Hồ sơ loss: p_c = ΔL_c(δ_S) / ΔL_c(δ_W).
  python scripts/e19_oracle_dissect.py --check        # sức khỏe trên 2 ảnh -> artifacts/e19/check.json
  python scripts/e19_oracle_dissect.py                # đo dev100 (resume) -> artifacts/e19/raw.json + phân tích
  python scripts/e19_oracle_dissect.py --analyze-only
  python scripts/e19_oracle_dissect.py --h1           # mô tả: AP drop R50 / DINO-Swin-L của δ_S (như E16)
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import bootstrap_indices, summarize
from attack.methods.box_transforms import get_gt
from attack.methods.core import wp_update
from attack.methods.elem import linear_homography, shift_h
from attack.methods.pia import gt_with_keep, warp, warp_boxes, warp_grid
from attack.preprocess import extract_features, resize_to_model, sum_loss_dict, to_batch

ART = os.path.join(REPO_ROOT, "artifacts/runs")
S_DIR = os.path.join(ART, "dev300_B41_eps5_limit100_sur-swin_t/adv/C2-OCT-1V")
W_DIR = os.path.join(ART, "dev300_B36_eps5_limit100/adv/C2-OCT-1V-WP2")
RAW = os.path.join(REPO_ROOT, "artifacts/e19/raw.json")
CHECK = os.path.join(REPO_ROOT, "artifacts/e19/check.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e19_oracle_dissect.json")
OUT_H1 = os.path.join(REPO_ROOT, "results/mechanism/e19_h1.json")
SEED, BOOT_SEED, N_BOOT, EPS, K, CLS_W, RHO = 0, 2026, 1000, 5.0, 3.0, 4.0, 0.002
N_VIEW, SIGMA, A_OCT = 8, 6.0, math.log(2) / 2
IG_T, IG_BLUR, PGN_N, PGN_ZETA, XI, N_XI, CONTENT_SIGMA = 8, 8.0, 4, 3.0, 2.0, 4, 2.0
T_GRID = [round(0.1 * i, 1) for i in range(1, 11)]
COMP = ("s0", "s1", "s2", "s3", "rpn", "roi")
ESTIMATORS = ("WP4", "IG-0", "IG-blur", "PGN4")
COMPONENTS = ("C-OSFD", "C-rpn", "C-roi", "C-OSFD-C2", "C-OSFD-C3", "C-OSFD-C4", "C-OSFD-C5")
NULLS = ("dW", "RAND", "CONTENT")


def nrm(g):
    return g / (g.abs().mean() + 1e-12)


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def gauss(img, s):
    """Gauss tách được, biên reflect, bán kính ⌈3σ⌉ ([C, H, W])."""
    r = int(math.ceil(3 * s))
    x = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-x ** 2 / (2 * s ** 2))
    k = k / k.sum()
    c = img.shape[0]
    t = F.conv2d(F.pad(img[None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)[0]


def read_png(path, dev):
    return torch.from_numpy(mmcv.imread(path)).permute(2, 0, 1).float().to(dev)


class Probe:
    """1 ảnh: 8 view OCT cố định (lưới, GT view, nhiễu, feature sạch của view, seed RoI sampler)."""

    def __init__(self, model, smp, seed_index, dev):
        self.model, self.dev = model, dev
        self.orig = smp["orig_inputs"].to(dev)
        ds = smp["data_sample"]
        if "masks" in ds.gt_instances:
            del ds.gt_instances.masks
        self.ds = ds
        self.size = tuple(ds.img_shape)
        h, w = self.size
        self.clean = resize_to_model(self.orig, self.size)
        self.params = [m.weight for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad]
        rng = random.Random(SEED + seed_index)
        boxes, _ = get_gt(ds, dev)
        self.views = []
        for v in range(N_VIEW):
            u, dx, dy = rng.uniform(-A_OCT, A_OCT), rng.randrange(8), rng.randrange(8)
            H = linear_homography(torch.tensor([[math.exp(u), 0.0], [0.0, math.exp(u)]], dtype=torch.float64), h, w) @ shift_h(dx, dy)
            grid = warp_grid(H, h, w, dev)
            nb, keep = warp_boxes(boxes, H, h, w)
            g = torch.Generator(device=dev).manual_seed(BOOT_SEED * 100000 + seed_index * 10 + v)
            nz = torch.randn((3, h, w), generator=g, device=dev) * SIGMA
            self.views.append((grid, gt_with_keep(ds, nb, None, keep), nz, self.clean_feats(grid), SEED + seed_index * 100 + v))

    @torch.no_grad()
    def clean_feats(self, grid):
        return tuple(f.detach() for f in extract_features(self.model, warp(self.clean, grid), self.ds))

    def parts(self, img, v, fc=None):
        """Các thành phần loss C2 của view v tại ảnh (không gian gốc) img."""
        grid, dsv, nz, fcv, seed = self.views[v]
        fc = fcv if fc is None else fc
        view = torch.clamp(warp(resize_to_model(torch.clamp(img, 0, 255), self.size), grid) + nz, 0, 255)
        cap = {}
        hk = self.model.backbone.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o))
        torch.manual_seed(seed)
        try:
            bi, bds = to_batch(self.model, [view], [dsv])
            losses = self.model.loss(bi, bds)
        finally:
            hk.remove()
        out = {f"s{s}": F.mse_loss(K * a, b) for s, (a, b) in enumerate(zip(fc, cap["f"]))}
        out["rpn"] = CLS_W * sum_loss_dict({"loss_rpn_cls": losses["loss_rpn_cls"]})
        out["roi"] = CLS_W * sum_loss_dict({"loss_cls": losses["loss_cls"]})
        return out

    def grad(self, img, v, fc=None, comps=False):
        x = img.detach().clone().requires_grad_(True)
        p = self.parts(x, v, fc)
        if not comps:
            return torch.autograd.grad(sum(p.values()), x)[0]
        gs = {}
        for j, k in enumerate(COMP):
            g = torch.autograd.grad(p[k], x, retain_graph=j < len(COMP) - 1, allow_unused=True)[0]
            gs[k] = torch.zeros_like(x) if g is None else g
        return gs

    def estimators(self, x, img_id):
        """Mọi ước lượng hướng tại điểm x (không gian gốc); feature sạch luôn của ảnh sạch (như attack)."""
        H, g8, osfd = {}, 0, 0
        comp = {k: 0 for k in COMP}
        for v in range(N_VIEW):
            gs = self.grad(x, v, comps=True)
            tot = sum(gs.values())
            if v == 0:
                H["G1"] = tot
            g8 = g8 + nrm(tot)
            osfd = osfd + nrm(sum(gs[f"s{s}"] for s in range(4)))
            for k in COMP:
                comp[k] = comp[k] + nrm(gs[k])
        H["G8"], H["C-OSFD"], H["C-rpn"], H["C-roi"] = g8, osfd, comp["rpn"], comp["roi"]
        for s in range(4):
            H[f"C-OSFD-C{s + 2}"] = comp[f"s{s}"]
        # WP4: θ' = cập nhật WP2 từ ∇θ của CHÍNH view (core.wp_update), ∇x dưới θ' (feature sạch cũng trên θ').
        base = [p.detach().clone() for p in self.params]
        h = 0
        try:
            for v in range(N_VIEW // 2):
                gp = torch.autograd.grad(sum(self.parts(x, v).values()), self.params)
                with torch.no_grad():
                    for p, p0, g in zip(self.params, base, gp):
                        p.copy_(wp_update(p, p0, g, RHO))
                h = h + nrm(self.grad(x, v, fc=self.clean_feats(self.views[v][0])))
                with torch.no_grad():
                    for p, p0 in zip(self.params, base):
                        p.copy_(p0)
        finally:
            with torch.no_grad():
                for p, p0 in zip(self.params, base):
                    p.copy_(p0)
        H["WP4"] = h
        for name, b in (("IG-0", torch.zeros_like(x)), ("IG-blur", gauss(self.orig, IG_BLUR))):
            acc = 0
            for k in range(1, IG_T + 1):
                acc = acc + nrm(self.grad(b + (k / IG_T) * (x - b), k - 1))
            H[name] = (x - b) * acc
        g = torch.Generator(device=self.dev).manual_seed(BOOT_SEED + img_id)
        h = 0
        for i in range(PGN_N):
            xp = torch.clamp(x + (2 * torch.rand(x.shape, generator=g, device=self.dev) - 1) * PGN_ZETA * EPS, 0, 255)
            g1 = self.grad(xp, i)
            g2 = self.grad(torch.clamp(xp - EPS * torch.sign(g1), 0, 255), i)
            h = h + 0.5 * nrm(g1) + 0.5 * nrm(g2)
        H["PGN4"] = h
        return H

    @torch.no_grad()
    def lbar(self, delta):
        """-> (loss tổng từng view [8], TB thành phần qua view) tại ảnh gốc + δ."""
        tots, comps = [], {k: 0.0 for k in COMP}
        for v in range(N_VIEW):
            p = {k: float(t) for k, t in self.parts(self.orig + delta, v).items()}
            tots.append(sum(p.values()))
            for k in COMP:
                comps[k] += p[k] / N_VIEW
        return tots, comps


def align(H, dS, dW):
    return {k: {"S": cos(torch.sign(h), dS), "W": cos(torch.sign(h), dW), "S_raw": cos(h, dS), "W_raw": cos(h, dW)}
            for k, h in H.items()}


def nulls(pr, img_id, dW):
    g = torch.Generator(device=pr.dev).manual_seed(BOOT_SEED + img_id)
    rand = EPS * (2 * torch.randint(0, 2, pr.orig.shape, generator=g, device=pr.dev).float() - 1)
    return {"dW": dW, "RAND": rand, "CONTENT": pr.orig - gauss(pr.orig, CONTENT_SIGMA)}


def measure(pr, img_id, dS, dW, with_p1):
    t0 = time.time()
    rec = {"img_id": img_id, "dev30": with_p1}
    H = pr.estimators(pr.orig, img_id)
    H.update(nulls(pr, img_id, dW))
    rec["P0"] = align(H, dS, dW)
    del H
    if with_p1:
        H = pr.estimators(pr.orig + dW, img_id)
        rec["P1"] = align(H, dS, dW)
        del H
    L0, c0 = pr.lbar(torch.zeros_like(dS))
    LS, cS = pr.lbar(dS)
    LW = {t: pr.lbar(t * dW) for t in T_GRID}
    gS = np.mean(LS) - np.mean(L0)
    t_star = min(T_GRID, key=lambda t: abs(np.mean(LW[t][0]) - np.mean(L0) - gS))
    g = torch.Generator(device=pr.dev).manual_seed(BOOT_SEED + 7 + img_id)
    xis = [XI * (2 * torch.randint(0, 2, dS.shape, generator=g, device=pr.dev).float() - 1) for _ in range(N_XI)]
    rec["flat"] = {"L0": L0, "LS": LS, "LW": {str(t): LW[t][0] for t in T_GRID}, "t_star": t_star,
                   "LSxi": [float(np.mean(pr.lbar(dS + xi)[0])) for xi in xis],
                   "LWxi": [float(np.mean(pr.lbar(t_star * dW + xi)[0])) for xi in xis]}
    rec["comp"] = {"0": c0, "S": cS, "W": LW[1.0][1]}
    with torch.no_grad():
        f0 = extract_features(pr.model, pr.clean, pr.ds)
        rec["feat_rel"] = {}
        for name, d in (("S", dS), ("W", dW)):
            fd = extract_features(pr.model, resize_to_model(torch.clamp(pr.orig + d, 0, 255), pr.size), pr.ds)
            rec["feat_rel"][name] = [float((b - a).norm() / (a.norm() + 1e-12)) for a, b in zip(f0, fd)]
    rec["sec"] = time.time() - t0
    return rec


def setup(dev):
    from attack.models import load_surrogate
    model = load_surrogate(dev)
    model.roi_head.mask_head = None  # skip_mask: loss C2 không dùng mask head
    data = build_attack_dataset("dev300")
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    pos = {i: k for k, i in enumerate(data.image_ids)}
    ids = [i for i in data.image_ids if i in keep]
    assert len(ids) == 100
    return model, data, pos, ids


def deltas(pr, img_id):
    dS = read_png(os.path.join(S_DIR, f"{img_id}.png"), pr.dev) - pr.orig
    dW = read_png(os.path.join(W_DIR, f"{img_id}.png"), pr.dev) - pr.orig
    assert dS.abs().max() <= EPS and dW.abs().max() <= EPS, img_id
    return dS, dW


def check(n=2):
    """Sức khỏe (khóa): ∇ tổng trực tiếp ≡ Σ ∇ thành phần (cos ≥ 0.999); cos(δ_S, δ_S) = 1; lặp ∇ cùng view (cos ≥ 0.999)."""
    dev = "cuda:0"
    model, data, pos, ids = setup(dev)
    out = []
    for img_id in ids[:n]:
        t0 = time.time()
        pr = Probe(model, data[pos[img_id]], pos[img_id], dev)
        dS, _ = deltas(pr, img_id)
        gs = pr.grad(pr.orig, 0, comps=True)
        direct, again = pr.grad(pr.orig, 0), pr.grad(pr.orig, 0)
        r = {"img_id": img_id, "cos_direct_vs_sum_comp": cos(direct, sum(gs.values())), "cos_repeat": cos(direct, again),
             "cos_dS_dS": cos(dS, dS), "grad_abs_mean": float(direct.abs().mean()),
             "comp_abs_mean": {k: float(g.abs().mean()) for k, g in gs.items()}, "sec_3grad": time.time() - t0}
        r["pass"] = r["cos_direct_vs_sum_comp"] >= 0.999 and r["cos_repeat"] >= 0.999 and abs(r["cos_dS_dS"] - 1) < 1e-6
        print(json.dumps(r), flush=True)
        out.append(r)
    t0 = time.time()
    rec = measure(pr, ids[n - 1], *deltas(pr, ids[n - 1]), with_p1=False)
    print(f"[E19 check] 1 ảnh đo đầy đủ (P0 + phẳng + hồ sơ): {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(CHECK), exist_ok=True)
    with open(CHECK, "w") as f:
        json.dump({"images": out, "pass": all(r["pass"] for r in out), "sec_full_image": rec["sec"]}, f, indent=2)
    print("CHECK", "QUA" if all(r["pass"] for r in out) else "KHÔNG QUA", flush=True)


def run(limit=None):
    dev = "cuda:0"
    model, data, pos, ids = setup(dev)
    ids = ids[:limit] if limit else ids
    dev30 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    raw = json.load(open(RAW)) if os.path.exists(RAW) else {}
    t0, done = time.time(), 0
    for n, img_id in enumerate(ids):
        if str(img_id) in raw:
            continue
        pr = Probe(model, data[pos[img_id]], pos[img_id], dev)
        raw[str(img_id)] = measure(pr, img_id, *deltas(pr, img_id), with_p1=img_id in dev30)
        del pr
        done += 1
        with open(RAW, "w") as f:  # lưu sau mỗi ảnh -> resume được
            json.dump(raw, f)
        el = (time.time() - t0) / done
        print(f"[E19] {n + 1}/{len(ids)}  {el:.0f}s/ảnh  còn ~{(len(ids) - n - 1) * el / 60:.0f} phút", flush=True)


def analyze(limit=None):
    raw = json.load(open(RAW))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))
           if str(i) in raw and i in set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))]
    ids = ids[:limit] if limit else ids
    R = [raw[str(i)] for i in ids]
    n = len(R)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    S = lambda a: summarize(a[0], a[1:])

    def wm(a, m=None):
        a = np.asarray(a, dtype=np.float64)
        m = np.ones(n, bool) if m is None else m
        return (W * m) @ np.where(m, a, 0.0) / np.maximum((W * m).sum(1), 1e-12)

    A = lambda pt, h, k="S": np.array([r[pt][h][k] for r in R])
    out = {"n_images": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "P0": {}}
    for h in R[0]["P0"]:
        out["P0"][h] = {k: S(wm(A("P0", h, k))) for k in ("S", "W", "S_raw", "W_raw")}
    aG, aC = wm(A("P0", "G8")), wm(A("P0", "CONTENT"))
    ck = json.load(open(CHECK)) if os.path.exists(CHECK) else {"pass": False}
    health = {"check_2img": bool(ck["pass"]), "rand_abs_lt_0.005": abs(out["P0"]["RAND"]["S"]["point"]) < 0.005,
              "G8_minus_RAND": S(aG - wm(A("P0", "RAND")))}
    health["G8_gt_RAND"] = health["G8_minus_RAND"]["ci95"][0] > 0
    health["pass"] = health["check_2img"] and health["rand_abs_lt_0.005"] and health["G8_gt_RAND"]
    out["health"] = health

    sep = {}
    for h in ESTIMATORS + COMPONENTS:
        a = wm(A("P0", h))
        d, dc = S(a - aG), S(a - aC)
        ok = a[0] >= 1.5 * aG[0] and d["point"] >= 0.01 and d["ci95"][0] > 0 and dc["ci95"][0] > 0
        sep[h] = {"A_S": S(a), "ratio_to_G8": float(a[0] / aG[0]) if aG[0] else float("nan"),
                  "delta_vs_G8": d, "delta_vs_CONTENT": dc, "TACH_DUOC": bool(ok)}
    out["separable"] = sep

    fl = [r["flat"] for r in R]
    m0 = lambda L: float(np.mean(L))
    gS = np.array([m0(f["LS"]) - m0(f["L0"]) for f in fl])
    gW = np.array([m0(f["LW"]["1.0"]) - m0(f["L0"]) for f in fl])
    gT = np.array([m0(f["LW"][str(f["t_star"])]) - m0(f["L0"]) for f in fl])
    rS = np.array([(np.mean(f["LSxi"]) - m0(f["L0"])) for f in fl]) / np.where(gS != 0, gS, np.nan)
    rW = np.array([(np.mean(f["LWxi"]) - m0(f["L0"])) for f in fl]) / np.where(gT != 0, gT, np.nan)
    inc = (gS > 0.05 * gW) & (gT > 0) & np.isfinite(rS) & np.isfinite(rW)
    d = S(wm(rS - rW, inc))
    spread = lambda L0, L: float(np.std(np.subtract(L, L0)) / (abs(np.mean(np.subtract(L, L0))) + 1e-12))
    out["flatness"] = {"n_included": int(inc.sum()), "r_S": S(wm(rS, inc)), "r_tW": S(wm(rW, inc)),
                       "r_S_minus_r_tW": d, "PHANG_HON": bool(d["point"] >= 0.10 and d["ci95"][0] > 0),
                       "gain_S": S(wm(gS)), "gain_W": S(wm(gW)), "t_star_median": float(np.median([f["t_star"] for f in fl])),
                       "view_spread_S": float(np.median([spread(f["L0"], f["LS"]) for f in fl])),
                       "view_spread_W": float(np.median([spread(f["L0"], f["LW"]["1.0"]) for f in fl]))}

    prof = {}
    for c in COMP + ("tot",):
        get = (lambda r, s: sum(r["comp"][s].values())) if c == "tot" else (lambda r, s, c=c: r["comp"][s][c])
        num = wm([get(r, "S") - get(r, "0") for r in R])
        den = wm([get(r, "W") - get(r, "0") for r in R])
        prof[c] = S(num / np.where(den != 0, den, np.nan))
    khac = [c for c in COMP if prof[c]["point"] >= 0.8 and prof[c]["ci95"][0] >= 0.6]
    out["profile"] = {"p": prof, "KHAC_HO_SO": bool(khac and prof["tot"]["point"] <= 0.5), "components_hit": khac,
                      "feat_rel_S": np.mean([r["feat_rel"]["S"] for r in R], 0).tolist(),
                      "feat_rel_W": np.mean([r["feat_rel"]["W"] for r in R], 0).tolist()}

    p1 = [r for r in R if "P1" in r]
    if p1:
        out["P1_dev30"] = {"n_images": len(p1), **{h: {k: float(np.mean([r["P1"][h][k] for r in p1])) for k in ("S", "W")}
                                                   for h in p1[0]["P1"]}}

    est_ok = [h for h in ESTIMATORS if sep[h]["TACH_DUOC"]]
    comp_ok = [h for h in COMPONENTS if sep[h]["TACH_DUOC"]]
    if not health["pass"]:
        row = "KHÔNG ĐO ĐƯỢC"
    elif est_ok:
        row = f"(1) ước lượng tách được: {max(est_ok, key=lambda h: sep[h]['delta_vs_G8']['point'])}"
    elif out["flatness"]["PHANG_HON"]:
        row = "(2) dual-flat"
    elif comp_ok or out["profile"]["KHAC_HO_SO"]:
        row = f"(3) đặt lại trọng số loss: {comp_ok + khac}"
    else:
        row = "(4) DỪNG — không ước lượng chỉ-R50 nào trỏ về δ_S"
    out["verdict"] = {"estimators_separable": est_ok, "components_separable": comp_ok, "row": row}
    out["mean_sec_per_image"] = float(np.mean([r["sec"] for r in R]))
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    f3 = lambda s: f"{s['point']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    print(f"\n=== E19 @ n = {n} — P0 (ảnh sạch): A_S = cos(sign h, δ_S) | A_W = cos(sign h, δ_W) ===")
    for h, v in out["P0"].items():
        extra = ""
        if h in sep:
            extra = f"  Δ vs G8 {f3(sep[h]['delta_vs_G8'])}  ×{sep[h]['ratio_to_G8']:.2f}  {'TÁCH' if sep[h]['TACH_DUOC'] else ''}"
        print(f"{h:<11} A_S {f3(v['S'])}  A_W {v['W']['point']:+.4f}{extra}")
    print("health", {k: v for k, v in health.items() if k != "G8_minus_RAND"}, "G8 − RAND", f3(health["G8_minus_RAND"]))
    fz = out["flatness"]
    print(f"flat: n {fz['n_included']}  r_S {f3(fz['r_S'])}  r_t*W {f3(fz['r_tW'])}  Δ {f3(fz['r_S_minus_r_tW'])}  "
          f"PHẲNG HƠN={fz['PHANG_HON']}  t* trung vị {fz['t_star_median']}")
    print("profile p_c:", {c: round(v["point"], 3) for c, v in prof.items()}, "KHÁC HỒ SƠ =", out["profile"]["KHAC_HO_SO"])
    if p1:
        print("P1 (dev30, mô tả):", {h: (round(v["S"], 4), round(v["W"], 4)) for h, v in out["P1_dev30"].items() if h != "n_images"})
    print("VERDICT", out["verdict"])


def rand_profile(n=30):
    """Đối chứng (progress_log 2026-10-23, khóa trước): p_c(X) = ΔL_c(X) / ΔL_c(δ_W), X ∈ {δ_S, RAND}, dev30."""
    dev = "cuda:0"
    model, data, pos, ids = setup(dev)
    rows = []
    for img_id in ids[:n]:
        pr = Probe(model, data[pos[img_id]], pos[img_id], dev)
        dS, dW = deltas(pr, img_id)
        c = {k: pr.lbar(d)[1] for k, d in (("0", torch.zeros_like(dS)), ("S", dS), ("W", dW),
                                           ("R", nulls(pr, img_id, dW)["RAND"]))}
        rows.append(c)
        del pr
    nb = len(rows)
    W = np.vstack([np.ones((1, nb)), bootstrap_indices(nb, N_BOOT, BOOT_SEED)])
    wm = lambda a: W @ np.asarray(a, dtype=np.float64) / W.sum(1)
    tot = lambda r, s: sum(r[s].values())
    p = {}
    for c in COMP + ("tot",):
        get = (lambda r, s: tot(r, s)) if c == "tot" else (lambda r, s, c=c: r[s][c])
        den = wm([get(r, "W") - get(r, "0") for r in rows])
        p[c] = {x: wm([get(r, x) - get(r, "0") for r in rows]) / den for x in ("S", "R")}
    d = summarize(*(lambda a: (a[0], a[1:]))(p["s0"]["S"] - p["s0"]["R"]))
    out = {"n_images": nb, "p": {c: {x: summarize(v[0], v[1:]) for x, v in pc.items()} for c, pc in p.items()},
           "d_C2_S_minus_RAND": d, "C2_THAT": bool(d["point"] >= 0.3 and d["ci95"][0] > 0)}
    with open(os.path.join(REPO_ROOT, "results/mechanism/e19_rand_profile_dev30.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for c, pc in out["p"].items():
        print(f"{c:<4} p(δ_S) {pc['S']['point']:.3f}  p(RAND) {pc['R']['point']:.3f}")
    print(f"d = p_C2(δ_S) − p_C2(RAND) {d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  C2 THẬT = {out['C2_THAT']}")


def h1():
    """Mô tả: relative AP drop R50 / DINO-Swin-L của δ_S mới (inference_detector trên file, như E16; E16: 78.7 / 58.0)."""
    from mmdet.apis import inference_detector, init_detector
    from pycocotools.coco import COCO
    from attack.evaluation import CocoImageEval, to_coco_dets
    from attack.models import CONTROLLED_PANEL
    from scripts.e16_delta_band import ANN_FILE, CKPT, IMG_DIR, MODELS
    coco = COCO(ANN_FILE)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), N_BOOT, BOOT_SEED)])
    cat_ids = sorted(coco.getCatIds())
    out = {}
    for key in ("r50", "dino_swin_l"):
        cfg, pat = MODELS[key]
        ckpt = CONTROLLED_PANEL[key].checkpoint_path if pat is None else glob.glob(os.path.join(CKPT, pat + ".pth"))[0]
        model = init_detector(cfg, ckpt, device="cuda:0")
        dets = {"clean": [], "S": []}
        for img_id in ids:
            dets["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])), img_id, cat_ids)
            dets["S"] += to_coco_dets(inference_detector(model, os.path.join(S_DIR, f"{img_id}.png")), img_id, cat_ids)
        del model
        torch.cuda.empty_cache()
        ev = {c: CocoImageEval(coco, d, ids) for c, d in dets.items()}
        ap = {c: np.array([ev[c].ap(w) for w in W])[:, 0] * 100 for c in dets}
        out[key] = summarize(*(lambda a: (a[0], a[1:]))(100 * (ap["clean"] - ap["S"]) / ap["clean"]))
        print(f"[E19 H1] {key}: drop δ_S {out[key]['point']:.1f} [{out[key]['ci95'][0]:.1f}, {out[key]['ci95'][1]:.1f}]", flush=True)
    with open(OUT_H1, "w") as f:
        json.dump(out, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--h1", action="store_true")
    p.add_argument("--rand-profile", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.check:
        return check()
    if args.rand_profile:
        return rand_profile()
    if args.h1:
        return h1()
    if not args.analyze_only:
        run(args.limit)
    analyze(args.limit)


if __name__ == "__main__":
    main()
