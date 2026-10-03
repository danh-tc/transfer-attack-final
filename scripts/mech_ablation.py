#!/usr/bin/env python3
"""Bước 2 của docs/plan_sota.md (LOCKED): suy ra cơ chế từ ablation OSFD — chỉ forward trên PNG có sẵn.

100 ảnh dev100 (chung cho mọi điều kiện). Điều kiện chính: OSFD + 7 ô ablation; mô tả thêm: KG-E1/E2,
SGM-0.2/0.5/mix (cùng seed, có sẵn PNG trên dev100).

Mô tả cơ chế mỗi (điều kiện, ảnh):
- D_k(m) = ‖f_k(adv) − f_k(clean)‖ / ‖f_k(clean)‖, stage 1–4, backbone R50 / R101 / ConvNeXt-T / Swin-T /
  DINO-Swin-L (cùng Resize 1333×800 + normalize -> cùng tensor đầu vào); định nghĩa như A2.
- Phổ δ (δ = PNG − ảnh gốc, cỡ gốc): tỉ lệ năng lượng dải thấp [0, 1/6), trung [1/6, 1/3), cao [1/3, ∞)
  cycles/pixel (như A4).
- Tập trung không gian: năng lượng δ trong GT box / tỉ lệ diện tích box (như A4).
Transfer: relative AP drop dev100 từng model + CrossAvg₃; per-ảnh retained confidence (định nghĩa A2/A3).

Phân tích:
1. Giữa điều kiện: Spearman(mô tả, CrossAvg₃) trên điều kiện chính (8) và trên tất cả (13); CI bằng
   bootstrap ảnh (mô tả và AP cùng trọng số).
2. Trong OSFD, theo ảnh: Spearman(mô tả, retained confidence) ở ConvNeXt-T, Swin-T, YOLOX-S.
3. Nguyên liệu (RRB, xoay, MI) đổi mô tả nào: Δ mô tả = OSFD − ô, CI.

Chạy: python scripts/mech_ablation.py   -> results/ablation/mech/mech.json
"""
import glob
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices
from attack.mechanism import per_image_transfer
from attack.models import CONTROLLED_PANEL, load_model
from attack.preprocess import pipeline_resize, to_batch

RUN = "dev300_B50_eps5"
ART = os.path.join(REPO_ROOT, "artifacts/runs", RUN)
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
BOOT_SEED, N_BOOT = 2026, 1000
MAIN = ["OSFD", "ABL-task-loss", "ABL-no-RRB", "ABL-RRB-no-noise", "ABL-RRB-no-rotate",
        "ABL-RRB-no-resize", "ABL-no-MI", "ABL-k5"]
EXTRA = ["OSFD-KG-E1", "OSFD-KG-E2", "OSFD-SGM-0.2", "OSFD-SGM-0.5", "OSFD-SGM-mix"]
CONDS = MAIN + EXTRA
DETS_FILE = {**{c: "dets_abl_dev100.json" for c in MAIN}, "OSFD-KG-E1": "dets_kg.json",
             "OSFD-KG-E2": "dets_kg.json", **{c: "dets_sgm.json" for c in EXTRA[2:]}}
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
GEN = ["yolox_s", "dino_swin_l"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]
FEAT_MODELS = CTRL + ["dino_swin_l"]
INGREDIENTS = {"RRB": "ABL-no-RRB", "xoay": "ABL-RRB-no-rotate", "MI": "ABL-no-MI"}


@torch.no_grad()
def feats(model, pixels, ds):
    return model.backbone(to_batch(model, [pixels], [ds])[0])[:4]


def spectrum(delta):
    """δ [C,H,W] -> tỉ lệ năng lượng 3 dải (cycles/pixel, bán kính chuẩn hóa theo trục)."""
    f = np.abs(np.fft.fft2(delta, axes=(-2, -1))) ** 2
    e = f.sum(0)
    h, w = e.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    tot = e.sum()
    return [float(e[r < 1 / 6].sum() / tot), float(e[(r >= 1 / 6) & (r < 1 / 3)].sum() / tot),
            float(e[r >= 1 / 3].sum() / tot)]


def box_ratio(delta, anns, h, w):
    m = np.zeros((h, w), bool)
    for a in anns:
        x, y, bw, bh = a["bbox"]
        m[int(y):int(np.ceil(y + bh)), int(x):int(np.ceil(x + bw))] = True
    if not m.any():
        return np.nan
    e = (delta ** 2).sum(0)
    return float((e[m].sum() / e.sum()) / m.mean())


def main():
    dev = "cuda:0"
    from mmdet.apis import init_detector
    models = {k: load_model(k, dev) for k in CTRL}
    ck = glob.glob(os.path.join(REPO_ROOT, "checkpoints/dino-5scale_swin-l*.pth"))
    models["dino_swin_l"] = init_detector(os.path.join(CFG, "dino/dino-5scale_swin-l_8xb2-36e_coco.py"), ck[0], device=dev)
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    img_ids = [i for i in data.image_ids if i in set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))]
    n = len(img_ids)
    coco = COCO(ANN_FILE)

    D = {c: {m: np.zeros((n, 4)) for m in FEAT_MODELS} for c in CONDS}
    spec = {c: np.zeros((n, 3)) for c in CONDS}
    boxr = {c: np.zeros(n) for c in CONDS}
    t0 = time.time()
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds, orig = s["inputs"].to(dev), s["data_sample"], s["orig_inputs"].numpy()
        clean = {m: feats(models[m], x, ds) for m in FEAT_MODELS}
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        for c in CONDS:
            adv = mmcv.imread(os.path.join(ART, "adv", c, f"{img_id}.png"))
            delta = adv.transpose(2, 0, 1).astype(np.float64) - orig
            spec[c][j] = spectrum(delta)
            boxr[c][j] = box_ratio(delta, anns, *orig.shape[1:])
            a = pipeline_resize(torch.from_numpy(adv).permute(2, 0, 1).float().to(dev), x.shape[-2:])
            for m in FEAT_MODELS:
                fa = feats(models[m], a, ds)
                D[c][m][j] = [float((u - v).norm() / v.norm()) for u, v in zip(fa, clean[m])]
        if (j + 1) % 10 == 0:
            print(f"[mech] {j + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    # Transfer dev100: AP với cùng trọng số bootstrap.
    dets = {}
    cache = {}
    for c in CONDS:
        fname = DETS_FILE[c]
        if fname not in cache:
            with open(os.path.join(ART, fname)) as f:
                cache[fname] = json.load(f)
        for k in CTRL:
            dets.setdefault(k, {})[c] = cache[fname][c][k]
            dets[k].setdefault("clean", cache[fname]["clean"][k])
    for k in GEN:
        with open(os.path.join(ART, "gen_dets", f"{k}.json")) as f:
            g = json.load(f)
        dets[k] = {c: g[c] for c in ["clean"] + CONDS}
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    ap = {k: {c: np.array([ev.ap(w)[0] for w in W]) * 100
              for c, ev in ((c, CocoImageEval(coco, dets[k][c], img_ids)) for c in ["clean"] + CONDS)}
          for k in CTRL + GEN}
    drop = {c: {k: 100 * (ap[k]["clean"] - ap[k][c]) / ap[k]["clean"] for k in ap} for c in CONDS}
    cross3 = {c: np.mean([drop[c][k] for k in CROSS3], axis=0) for c in CONDS}

    # Mô tả mức điều kiện, theo từng mẫu bootstrap (cùng trọng số ảnh).
    Wn = W / W.sum(1, keepdims=True)
    desc = {}
    for c in CONDS:
        d = {}
        for m in FEAT_MODELS:
            for k in range(4):
                d[f"D{k + 1}_{m}"] = Wn @ D[c][m][:, k]
            d[f"D4/D1_{m}"] = Wn @ (D[c][m][:, 3] / D[c][m][:, 0])
        for b, name in enumerate(["low", "mid", "high"]):
            d[f"spec_{name}"] = Wn @ spec[c][:, b]
        ok = ~np.isnan(boxr[c])
        d["box_ratio"] = (W[:, ok] @ boxr[c][ok]) / W[:, ok].sum(1)
        desc[c] = d
    names = list(desc[CONDS[0]])

    def corr(conds):
        out = {}
        for nm in names:
            r = np.array([spearmanr([desc[c][nm][b] for c in conds], [cross3[c][b] for c in conds])[0]
                          for b in range(len(W))])
            lo, hi = np.nanpercentile(r[1:], [2.5, 97.5])
            out[nm] = {"rho": float(r[0]), "ci95": [float(lo), float(hi)]}
        return out
    corr_main, corr_all = corr(MAIN), corr(CONDS)

    # Theo ảnh trong OSFD: mô tả ảnh vs retained confidence ở target khác họ.
    per_img = {}
    for k in CROSS3:
        ret = per_image_transfer(coco, dets[k]["clean"], dets[k]["OSFD"], img_ids)["retained"]
        row = {}
        for m in FEAT_MODELS:
            for st in range(4):
                v = D["OSFD"][m][:, st]
                ok = ~np.isnan(ret)
                row[f"D{st + 1}_{m}"] = float(spearmanr(v[ok], ret[ok])[0])
        for b, name in enumerate(["low", "mid", "high"]):
            ok = ~np.isnan(ret)
            row[f"spec_{name}"] = float(spearmanr(spec["OSFD"][ok, b], ret[ok])[0])
        per_img[k] = row

    # Nguyên liệu đổi mô tả nào: OSFD − ô.
    ing = {g: {nm: {"point": float(desc["OSFD"][nm][0] - desc[c][nm][0]),
                    "ci95": [float(v) for v in np.percentile(desc["OSFD"][nm][1:] - desc[c][nm][1:], [2.5, 97.5])]}
               for nm in names} for g, c in INGREDIENTS.items()}

    out = {"rule": "docs/plan_sota.md bước 2 (mô tả)", "n_images": n, "conds_main": MAIN, "conds_extra": EXTRA,
           "cross_avg3_dev100": {c: float(cross3[c][0]) for c in CONDS},
           "relative_ap_drop_dev100": {c: {k: float(drop[c][k][0]) for k in ap} for c in CONDS},
           "descriptors": {c: {nm: float(desc[c][nm][0]) for nm in names} for c in CONDS},
           "spearman_vs_cross3_main": corr_main, "spearman_vs_cross3_all": corr_all,
           "per_image_osfd_spearman_vs_retained": per_img, "ingredient_effect_on_descriptors": ing}
    path = os.path.join(REPO_ROOT, "results/ablation/mech/mech.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    top = sorted(names, key=lambda nm: -abs(corr_all[nm]["rho"]))[:15]
    print("\n=== Spearman(mô tả, CrossAvg₃) giữa điều kiện: 8 chính | 13 tất cả ===")
    for nm in top:
        a, b = corr_main[nm], corr_all[nm]
        print(f"{nm:<22} {a['rho']:+.2f} [{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}] | {b['rho']:+.2f} [{b['ci95'][0]:+.2f},{b['ci95'][1]:+.2f}]")
    print("\n=== mô tả theo điều kiện (D4/D1, D4, phổ thấp) ===")
    for c in CONDS:
        d = out["descriptors"][c]
        print(f"{c:<20} Cross3 {cross3[c][0]:5.1f}  " + "  ".join(f"D4 {m} {d[f'D4_{m}']:.2f}" for m in FEAT_MODELS)
              + f"  low {d['spec_low']:.2f}")


if __name__ == "__main__":
    main()
