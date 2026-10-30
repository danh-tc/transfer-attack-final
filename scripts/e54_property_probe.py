#!/usr/bin/env python3
"""E54 (progress_log 2026-10-30): dò tính chất CHỈ-R50 tách hướng chung δ_S khỏi hướng R50 δ_RD, dọc trục C -> RD -> S
(E53: WP là bước làm phẳng một phần trên trục này). Mỗi ảnh dev30, mỗi δ_X (PNG - ảnh gốc) X in {C, RD, S}, ở khung model
(resize_to_model); L_OSFD k=3 + 4*C2 cls (e32_wp2_dissect.attack_loss); D_s = ||F_s(x+δ)-F_s(x)||^2 / ||F_s(x)||^2 (C2..C5):
  P1 deep     = D_C5 / sum_s D_s                              (WP dời distortion về stage sâu, E32)
  P2 r_def    = G(θ') / G(θ0), θ'=WP phòng thủ 1 bước ρ=0.002 chống CHÍNH δ (E32); G = L(x+δ)-L(x)
  P3 view_rob = TB_v Dfeat(view(x+δ) vs view(x)) / Dfeat(id)  (v: co 0.8, giãn 1.25, dịch vòng 4px) — độ bền OCT
  P4 scramble = Dfeat(δ xáo ô 16px) / Dfeat(δ)                 (phụ thuộc mẫu cục bộ, kiểu E7)
  P5 boxconc  = (năng lượng distortion C4 trong box GT) / (tỉ lệ token trong box)
  P6 clsfrac  = G_cls / G                                      (G_cls = 4*ΔC2 cls)
Một tính chất là ứng viên regularizer nếu ĐƠN ĐIỆU C->RD->S và bước RD->S > bước C->RD (CI loại 0).
  python scripts/e54_property_probe.py [--limit n]"""
import argparse
import json
import os
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import boot_weights
from attack.methods.core import wp_update
from attack.methods.repair import _box_mask, cls_term
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch
from scripts.e32_wp2_dissect import K, LAM, RHO_DEF, attack_loss, conv_params, set_params

ART = os.path.join(REPO_ROOT, "artifacts/runs")
SRC = {"C": "dev300_B36_eps5_limit100/adv/C2-OCT-1V", "RD": "dev300_B36_eps5_limit100/adv/C2-OCT-1V-WP2-RD",
       "S": "dev300_B41_eps5_limit100_sur-swin_t/adv/C2-OCT-1V"}
XS = ("C", "RD", "S")
PROPS = ("deep", "r_def", "view_rob", "scramble", "boxconc", "clsfrac")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e54_property_probe.json")


@torch.no_grad()
def bb_feats(model, pix, ds):
    bi, _ = to_batch(model, [pix], [ds])
    return model.backbone(bi)


def dist_stages(fc, fa):
    return [float(((a.double() - c.double()) ** 2).sum() / (c.double() ** 2).sum()) for c, a in zip(fc, fa)]


@torch.no_grad()
def dfeat(model, xr, ar, ds):
    """Σ_s ||F_s(ar) - F_s(xr)||^2 / ||F_s(xr)||^2 (distortion feature toàn phần)."""
    return float(sum(dist_stages(bb_feats(model, xr, ds), bb_feats(model, ar, ds))))


def scramble(d, tile=16):
    """Xáo δ theo ô tile×tile (hoán vị cố định theo ảnh), giữ nội dung trong mỗi ô."""
    c, h, w = d.shape
    hh, ww = h // tile, w // tile
    if hh < 2 or ww < 2:
        return d.clone()
    core = d[:, :hh * tile, :ww * tile].reshape(c, hh, tile, ww, tile).permute(0, 1, 3, 2, 4).reshape(c, hh * ww, tile, tile)
    g = torch.Generator().manual_seed(2026)
    perm = torch.randperm(hh * ww, generator=g)
    out = d.clone()
    s = core[:, perm].reshape(c, hh, ww, tile, tile).permute(0, 1, 3, 2, 4).reshape(c, hh * tile, ww * tile)
    out[:, :hh * tile, :ww * tile] = s
    return out


def view(img, name):
    """Biến đổi 1 ảnh (ở khung model); áp y hệt cho clean và adv -> distortion feature đo được."""
    if name == "id":
        return img
    if name == "shift":
        return torch.roll(img, shifts=(4, 4), dims=(-2, -1))
    return F.interpolate(img[None], scale_factor={"shrink": 0.8, "grow": 1.25}[name], mode="bilinear", align_corners=False)[0]


VIEWS = ("shrink", "grow", "shift")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    dev = "cuda:0"
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:args.limit]
    model = load_model("r50", dev)
    params = conv_params(model)
    base = [q.detach().clone() for q in params]
    rec = defaultdict(lambda: defaultdict(list))  # rec[prop][X] = list theo ảnh

    for img_id in img_ids:
        s = data[pos[img_id]]
        ds, orig = s["data_sample"], s["orig_inputs"].to(dev).float()
        xr = resize_to_model(orig, tuple(ds.img_shape))
        fc_clean = tuple(f.detach() for f in bb_feats(model, xr, ds))
        for X, path in SRC.items():
            d = torch.from_numpy(mmcv.imread(os.path.join(ART, path, f"{img_id}.png"))).permute(2, 0, 1).float().to(dev) - orig
            d = torch.clamp(orig + d, 0, 255) - orig
            ar = resize_to_model(orig + d, tuple(ds.img_shape))
            # G, G_cls tại θ0
            set_params(params, base)
            lc, _ = attack_loss(model, xr, ds, fc_clean)
            la, fa = attack_loss(model, ar, ds, fc_clean)
            G = float(la - lc)
            torch.manual_seed(0); bic, bdc = to_batch(model, [xr], [ds]); gc = LAM * float(cls_term(model, bic, bdc, "c2"))
            torch.manual_seed(0); bia, bda = to_batch(model, [ar], [ds]); ga = LAM * float(cls_term(model, bia, bda, "c2"))
            fa_adv = [f.detach() for f in bb_feats(model, ar, ds)]  # feature adv thật (attack_loss trả fc_clean đã truyền)
            Ds = dist_stages(fc_clean, fa_adv)
            rec["deep"][X].append(Ds[3] / (sum(Ds) + 1e-12))
            rec["clsfrac"][X].append((ga - gc) / (G + 1e-12))
            # P5 boxconc: C4 (stage idx 2, stride 16) distortion trong box / tỉ lệ token
            fcc, faa = fc_clean[2].double()[0], fa_adv[2].double()[0]
            e = ((faa - fcc) ** 2).sum(0)
            m = _box_mask(fa[2], 2, ds).to(e.device)
            tf = float(m.float().mean())
            rec["boxconc"][X].append(float(e[m].sum() / (e.sum() + 1e-12)) / tf if 0 < tf < 1 else np.nan)
            # P3 view_rob, P4 scramble (distortion feature); view áp y hệt cho clean + adv
            d_id = dfeat(model, xr, ar, ds)
            vr = np.mean([dfeat(model, view(xr, nm), view(ar, nm), ds) / (d_id + 1e-12) for nm in VIEWS])
            rec["view_rob"][X].append(float(vr))
            ar_sc = resize_to_model(orig + scramble(d), tuple(ds.img_shape))
            rec["scramble"][X].append(dfeat(model, xr, ar_sc, ds) / (d_id + 1e-12))
            # P2 r_def: WP phòng thủ 1 bước chống chính δ (như E32)
            _, fcd = attack_loss(model, xr, ds, None)
            Ld, _ = attack_loss(model, ar, ds, tuple(f.detach() for f in fcd))
            gth = torch.autograd.grad(Ld, params)
            set_params(params, [wp_update(p0, p0, g, RHO_DEF) for p0, g in zip(base, gth)])
            lc2, fc2 = attack_loss(model, xr, ds, None)
            la2, _ = attack_loss(model, ar, ds, tuple(f.detach() for f in fc2))
            rec["r_def"][X].append(float((la2 - lc2) / (G + 1e-12)))
            set_params(params, base)
    assert all(torch.equal(q, b) for q, b in zip(params, base)), "θ0 không khôi phục"

    n = len(img_ids)
    bidx = boot_weights(n, 1000, 2026)
    res = {"n_images": n, "mean": {}, "step": {}, "verdict": {}}
    for prop in PROPS:
        A = {X: np.array(rec[prop][X], dtype=float) for X in XS}
        for X in XS:
            res["mean"][f"{prop}|{X}"] = float(np.nanmean(A[X]))
        def step(a, b):
            dd = A[a] - A[b]
            bs = np.array([np.nanmean(dd[w]) for w in bidx])
            return {"point": float(np.nanmean(dd)), "ci95": [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]}
        s_crd, s_rds = step("RD", "C"), step("S", "RD")
        res["step"][f"{prop}|RD-C"] = s_crd
        res["step"][f"{prop}|S-RD"] = s_rds
        mono = (res["mean"][f"{prop}|C"] <= res["mean"][f"{prop}|RD"] <= res["mean"][f"{prop}|S"]) or \
               (res["mean"][f"{prop}|C"] >= res["mean"][f"{prop}|RD"] >= res["mean"][f"{prop}|S"])
        big = abs(s_rds["point"]) > abs(s_crd["point"]) and (s_rds["ci95"][0] > 0 or s_rds["ci95"][1] < 0)
        res["verdict"][prop] = "ỨNG VIÊN" if (mono and big) else ("đơn điệu" if mono else "-")
    if not args.limit:
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    for prop in PROPS:
        print(f"{prop:9s} C {res['mean'][f'{prop}|C']:+.4f}  RD {res['mean'][f'{prop}|RD']:+.4f}  S {res['mean'][f'{prop}|S']:+.4f}"
              f"  | RD-C {res['step'][f'{prop}|RD-C']['point']:+.4f}  S-RD {res['step'][f'{prop}|S-RD']['point']:+.4f}"
              f" [{res['step'][f'{prop}|S-RD']['ci95'][0]:+.3f},{res['step'][f'{prop}|S-RD']['ci95'][1]:+.3f}]  -> {res['verdict'][prop]}")


if __name__ == "__main__":
    main()
