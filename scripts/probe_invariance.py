#!/usr/bin/env python3
"""Probe hướng 2 (distortion bất biến) — mô tả, không phải tiêu chí. Chỉ forward, 100 ảnh dev100.

Câu hỏi: distortion mà δ gây ra trên surrogate R50 gồm bao nhiêu phần CHUNG giữa các view xoay và
bao nhiêu phần ĐẶC THÙ view; phần chung có dự đoán transfer tốt hơn tổng distortion không?

Mỗi (điều kiện, ảnh): T = 8 góc xoay θ_t ∈ {0} ∪ U(−7°, 7°) quanh tâm ảnh (khung model), áp CÙNG góc cho
ảnh sạch và ảnh adv (PNG có sẵn). Feature backbone R50 stage k, xoay NGƯỢC về khung gốc (feature map,
cùng tâm), chỉ lấy vùng hợp lệ chung (mask xoay đi xoay về). Δ_t = F_k(adv_t) − F_k(clean_t) (đã căn).
  E = mean_t ‖Δ_t‖²  (tổng),  S = ‖mean_t Δ_t‖²  (chung),  r = S / E  (tỉ lệ chung, ∈ [0, 1]).
Phân tích: (a) giữa điều kiện: mean r vs CrossAvg₃ dev100 (từ results/ablation/mech/mech.json);
(b) theo ảnh trong OSFD: Spearman(S, retained) vs Spearman(E, retained) ở ConvNeXt-T, Swin-T, YOLOX-S.

Chạy: python scripts/probe_invariance.py   -> results/ablation/mech/invariance_probe.json
"""
import json
import os
import random
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr
from torchvision.transforms.functional import rotate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.mechanism import per_image_transfer
from attack.models import SURROGATE_KEY, load_model
from attack.preprocess import pipeline_resize, to_batch

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
MECH = os.path.join(REPO_ROOT, "results/ablation/mech/mech.json")
T, THETA, SEED = 8, 7.0, 2026
CONDS = ["OSFD", "ABL-task-loss", "ABL-no-RRB", "ABL-RRB-no-noise", "ABL-RRB-no-rotate",
         "ABL-RRB-no-resize", "ABL-no-MI", "ABL-k5", "OSFD-KG-E1", "OSFD-KG-E2",
         "OSFD-SGM-0.2", "OSFD-SGM-0.5", "OSFD-SGM-mix"]
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]


@torch.no_grad()
def feats(model, pix, ds):
    return model.backbone(to_batch(model, [pix], [ds])[0])


def main():
    dev = "cuda:0"
    model = load_model(SURROGATE_KEY, dev)
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    img_ids = [i for i in data.image_ids if i in keep]
    rng = random.Random(SEED)
    angles = {img_id: [0.0] + [rng.uniform(-THETA, THETA) for _ in range(T - 1)] for img_id in img_ids}
    E = {c: np.zeros((len(img_ids), 4)) for c in CONDS}
    S = {c: np.zeros((len(img_ids), 4)) for c in CONDS}
    for j, img_id in enumerate(img_ids):
        s = data[pos[img_id]]
        x, ds = s["inputs"].to(dev), s["data_sample"]
        h, w = x.shape[-2:]
        center = [w / 2, h / 2]
        clean_f = [feats(model, rotate(x[None], a, center=center)[0], ds) for a in angles[img_id]]
        for c in CONDS:
            adv = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", c, f"{img_id}.png")))
                                  .permute(2, 0, 1).float().to(dev), (h, w))
            deltas = [[] for _ in range(4)]
            masks = [[] for _ in range(4)]
            for a, cf in zip(angles[img_id], clean_f):
                af = feats(model, rotate(adv[None], a, center=center)[0], ds)
                for k in range(4):
                    d = af[k] - cf[k]  # [1,C,h,w], ở khung đã xoay
                    # Tâm ảnh ở khung feature stage k (stride 4/8/16/32; canvas pad ở phải/dưới
                    # nên gốc tọa độ không đổi).
                    cfeat = [center[0] / (2 ** (k + 2)), center[1] / (2 ** (k + 2))]
                    back = rotate(d, -a, center=cfeat)
                    ones = torch.ones_like(d[:, :1])
                    valid = rotate(rotate(ones, a, center=cfeat), -a, center=cfeat) > 0.99
                    deltas[k].append(back)
                    masks[k].append(valid)
            for k in range(4):
                m = torch.stack(masks[k]).all(0).float()          # [1,1,h,w] vùng hợp lệ chung
                D = torch.stack(deltas[k]) * m                      # [T,1,C,h,w]
                E[c][j, k] = float(D.pow(2).sum(dim=(1, 2, 3, 4)).mean())
                S[c][j, k] = float(D.mean(0).pow(2).sum())
        if (j + 1) % 20 == 0:
            print(f"[probe] {j + 1}/{len(img_ids)}", flush=True)

    mech = json.load(open(MECH))
    cross3 = mech["cross_avg3_dev100"]
    r = {c: S[c] / E[c] for c in CONDS}
    out = {"n_images": len(img_ids), "T": T, "theta": THETA,
           "share_ratio_mean": {c: r[c].mean(0).round(4).tolist() for c in CONDS},
           "cross_avg3_dev100": {c: cross3[c] for c in CONDS}, "between_conditions": {}, "per_image_osfd": {}}
    for k in range(4):
        vals = [r[c][:, k].mean() for c in CONDS]
        out["between_conditions"][f"stage{k + 1}"] = float(spearmanr(vals, [cross3[c] for c in CONDS])[0])

    coco = COCO(ANN_FILE)
    with open(os.path.join(ART, "dets_abl_dev100.json")) as f:
        ctrl = json.load(f)
    dets = {k: {c: ctrl[c][k] for c in ["clean", "OSFD"]} for k in ["convnext_t", "swin_t"]}
    with open(os.path.join(ART, "gen_dets", "yolox_s.json")) as f:
        g = json.load(f)
    dets["yolox_s"] = {c: g[c] for c in ["clean", "OSFD"]}
    for t in CROSS3:
        ret = per_image_transfer(coco, dets[t]["clean"], dets[t]["OSFD"], img_ids)["retained"]
        ok = ~np.isnan(ret)
        out["per_image_osfd"][t] = {f"stage{k + 1}": {
            "rho_total_E": float(spearmanr(E["OSFD"][ok, k], ret[ok])[0]),
            "rho_shared_S": float(spearmanr(S["OSFD"][ok, k], ret[ok])[0]),
            "rho_ratio_r": float(spearmanr(r["OSFD"][ok, k], ret[ok])[0])} for k in range(4)}
    path = os.path.join(REPO_ROOT, "results/ablation/mech/invariance_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== tỉ lệ chung r = S/E theo stage (mean 100 ảnh) | CrossAvg₃ ===")
    for c in CONDS:
        print(f"{c:<20} " + "  ".join(f"{v:.3f}" for v in out["share_ratio_mean"][c]) + f"   | {cross3[c]:.1f}")
    print("Spearman giữa điều kiện (r vs CrossAvg₃):", {k: round(v, 2) for k, v in out["between_conditions"].items()})
    print("\n=== theo ảnh (OSFD): Spearman với retained (âm = dự đoán phá tốt) — tổng E | chung S | tỉ lệ r ===")
    for t, row in out["per_image_osfd"].items():
        print(t, "  ".join(f"{st}: {v['rho_total_E']:+.2f} | {v['rho_shared_S']:+.2f} | {v['rho_ratio_r']:+.2f}" for st, v in row.items()))


if __name__ == "__main__":
    main()
