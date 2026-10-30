#!/usr/bin/env python3
"""E53 (progress_log 2026-10-30): giả thuyết "SAI THỨ TỰ ĐỘ DỐC" — tại ảnh sạch, hướng chung (δ_S, sinh trên Swin-T) có
THOẢI trên R50 nhưng DỐC trên DINO-Swin-L, còn hướng R50 (δ_RD) ngược lại?
Mỗi ảnh dev30, mỗi model M ∈ {R50 surrogate, DINO-Swin-L (chỉ phân tích)}, g_M = ∇ theo ảnh gốc của loss M tại ảnh sạch:
  r_M(X) = g_M · δ_X / (ε‖g_M‖₁)  — tỉ lệ lợi bậc 1 của δ_X so hướng dốc nhất trong hộp L∞ (ε·sign g_M ⇒ r = 1)
  G_M(X) = L_M(x + δ_X) − L_M(x)  — lợi thật (phi tuyến)
R50: L = L_OSFD k = 3 + 4·C2 cls, view đồng nhất (e32_wp2_dissect.attack_loss, như E32 / E38). DINO-Swin-L: loss cls không-dn
(targets.target_grads 'cls', như E1 / E38 / E51). δ_X (PNG − ảnh gốc): S = C2-OCT-1V sur Swin-T B41; RD = RD B36 (E46);
C = C2-OCT-1V B36 (E48); W = WB-FULL DINO-Swin-L (E51); RAND = Rademacher ±5 (seed 2026 + img_id + 1, như RAND5).
  python scripts/e53_steepness.py [--limit n]"""
import argparse
import json
import os
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import boot_weights
from attack.models import load_model
from attack.preprocess import resize_to_model, sum_loss_dict, to_batch
from attack.targets import _loss_mode, kind, load_any, select_terms, target_dataset, target_grads
from scripts.e32_wp2_dissect import attack_loss

ART = os.path.join(REPO_ROOT, "artifacts")
SRC = {"S": "runs/dev300_B41_eps5_limit100_sur-swin_t/adv/C2-OCT-1V", "RD": "runs/dev300_B36_eps5_limit100/adv/C2-OCT-1V-WP2-RD",
       "C": "runs/dev300_B36_eps5_limit100/adv/C2-OCT-1V", "W": "e51/WB-FULL"}
EPS = 5.0
OUT = os.path.join(REPO_ROOT, "results/mechanism/e53_steepness.json")


def deltas(orig, img_id):
    d = {k: torch.from_numpy(mmcv.imread(os.path.join(ART, p, f"{img_id}.png"))).permute(2, 0, 1).float().to(orig) - orig
         for k, p in SRC.items()}
    g = torch.Generator().manual_seed(2026 + img_id + 1)
    d["RAND"] = (torch.randint(0, 2, orig.shape, generator=g).float() * 2 - 1).to(orig) * EPS
    return {k: torch.clamp(orig + v, 0, 255) - orig for k, v in d.items()}


@torch.no_grad()
def target_loss(model, key, img, ds):
    _loss_mode(model, kind(key))
    try:
        torch.manual_seed(0)
        inputs, dss = to_batch(model, [resize_to_model(img, tuple(ds.img_shape))], [ds])
        return float(sum_loss_dict(select_terms(model.loss(inputs, dss), kind(key), "cls")))
    finally:
        model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    dev = "cuda:0"
    data = build_attack_dataset("dev300")
    pos = {i: j for j, i in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:args.limit]
    r, G = defaultdict(list), defaultdict(list)  # [(model, X)] theo ảnh

    model = load_model("r50", dev)
    for img_id in img_ids:
        s = data[pos[img_id]]
        x, ds, orig = s["inputs"].to(dev).float(), s["data_sample"], s["orig_inputs"].to(dev).float()
        with torch.no_grad():
            l0, fc = attack_loss(model, resize_to_model(orig, tuple(ds.img_shape)), ds, None)
        v = orig.clone().requires_grad_(True)
        L, _ = attack_loss(model, resize_to_model(v, tuple(ds.img_shape)), ds, fc)
        g = torch.autograd.grad(L, v)[0]
        for X, d in deltas(orig, img_id).items():
            r[("r50", X)].append(float((g * d).sum() / (EPS * g.abs().sum())))
            with torch.no_grad():
                G[("r50", X)].append(float(attack_loss(model, resize_to_model(orig + d, tuple(ds.img_shape)), ds, fc)[0] - l0))
    del model
    torch.cuda.empty_cache()
    print("[e53] r50 xong", flush=True)

    model = load_any("dino_swin_l", dev)
    tds = target_dataset("dino_swin_l", img_ids)
    for j, img_id in enumerate(img_ids):
        orig, ds = data[pos[img_id]]["orig_inputs"].to(dev).float(), tds[j]["data_sample"]
        g = target_grads(model, "dino_swin_l", orig, ds, ("cls",))[0]["cls"]
        l0 = target_loss(model, "dino_swin_l", orig, ds)
        for X, d in deltas(orig, img_id).items():
            r[("dino_swin_l", X)].append(float((g * d).sum() / (EPS * g.abs().sum())))
            G[("dino_swin_l", X)].append(target_loss(model, "dino_swin_l", orig + d, ds) - l0)
    print("[e53] dino_swin_l xong", flush=True)

    n = len(img_ids)
    W = boot_weights(n, 1000, 2026)
    A = lambda q, k: np.array(q[k])
    def ratio(q, m, a, b):  # TB(a) / TB(b), bootstrap theo ảnh
        pa, pb = A(q, (m, a)), A(q, (m, b))
        pt = pa.mean() / pb.mean()
        bs = pa[W].mean(1) / pb[W].mean(1)
        return {"point": float(pt), "ci95": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
    res = {"n_images": n, "r_mean": {f"{m}|{X}": float(np.mean(v)) for (m, X), v in r.items()},
           "G_mean": {f"{m}|{X}": float(np.mean(v)) for (m, X), v in G.items()},
           "ratio": {"r50 r(RD)/r(S)": ratio(r, "r50", "RD", "S"), "dino r(S)/r(RD)": ratio(r, "dino_swin_l", "S", "RD"),
                     "r50 G(RD)/G(S)": ratio(G, "r50", "RD", "S"), "dino G(S)/G(RD)": ratio(G, "dino_swin_l", "S", "RD")}}
    a, b = res["ratio"]["r50 G(RD)/G(S)"], res["ratio"]["dino G(S)/G(RD)"]  # quy tắc trên lợi thật G (r: mô tả)
    if a["point"] >= 1.5 and a["ci95"][0] > 1 and b["point"] >= 1.5 and b["ci95"][0] > 1:
        v = "ỦNG HỘ"
    elif res["G_mean"]["r50|S"] >= res["G_mean"]["r50|RD"]:
        v = "BÁC"
    else:
        v = "XÁM"
    res["verdict_H_steep"] = v
    if not args.limit:
        json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    for m in ("r50", "dino_swin_l"):
        print(m, "r:", " ".join(f"{X} {res['r_mean'][f'{m}|{X}']:+.4f}" for X in ("S", "RD", "C", "W", "RAND")),
              "| G:", " ".join(f"{X} {res['G_mean'][f'{m}|{X}']:+.3f}" for X in ("S", "RD", "C", "W", "RAND")))
    for k, d in res["ratio"].items():
        print(f"{k}: {d['point']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]")
    print("H_steep:", v)


if __name__ == "__main__":
    main()
