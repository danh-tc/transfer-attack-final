#!/usr/bin/env python3
"""OTSV (E21, progress_log 2026-10-24) — cổng sức khỏe target-free, chọn β ∈ {0.25, 0.5, 1.0}. CHỈ R50, không target nào.
(H1) AP sạch R50 khi forward ảo OTSV (box GT khung model) trên dev30 ≥ 90% R50 thường.
(H3a) forward ảo thật sự đổi C4: trung bình ‖F' − F‖ / ‖F‖ ≥ 0.05 (ảnh 1–4 dev300, ảnh sạch).
(H3b) gradient ảnh tại δ = 0 của loss attack (L_OSFD k = 3 + 4·(rpn_cls + cls), feature sạch cùng chế độ) khác forward gốc:
      trung bình cos(g_OTSV, g_gốc) ≤ 0.95 (ảnh 1–4 dev300).
(H2, white-box từ 0) đo riêng bằng scripts/tx_calib.py.
Chạy: python scripts/otsv_health.py"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import AttackDataset, load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.methods.pia import otsv_hook
from attack.methods.repair import make_view_loss
from attack.models import load_model
from attack.preprocess import extract_features, predict

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")


def gt_boxes(ds):
    b = ds.gt_instances.bboxes
    return b.tensor if hasattr(b, "tensor") else b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results/otsv/health.json"))
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    coco = COCO(ANN_FILE)
    cat_ids = sorted(coco.getCatIds())
    model = load_model("r50", "cuda:0")
    ds30 = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ds4 = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv"))[1:5])
    state = {"on": False, "boxes": None}
    res = {}
    for b in [0.0] + args.betas:
        r = {}
        with otsv_hook(model, b, state):
            dets = []
            for i in range(len(ds30)):
                s = ds30[i]
                x, d = s["inputs"].cuda(), s["data_sample"]
                state.update(on=b > 0, boxes=gt_boxes(d))
                dets += to_coco_dets(predict(model, x, d), ds30.image_ids[i], cat_ids)
            r["ap"] = float(CocoImageEval(coco, dets, ds30.image_ids).ap(np.ones(len(ds30)))[0])
            if b > 0:
                rel, cos = [], []
                for i in range(len(ds4)):
                    s = ds4[i]
                    x, d = s["inputs"].cuda(), s["data_sample"]
                    state["boxes"] = gt_boxes(d)
                    g = {}
                    for on in (False, True):
                        state["on"] = on
                        with torch.no_grad():
                            fc = tuple(f.detach() for f in extract_features(model, x, d))
                        xv = x.clone().requires_grad_(True)
                        torch.manual_seed(0)  # cùng mẫu RoI giữa 2 chế độ
                        g[on] = torch.autograd.grad(make_view_loss(fc, 3.0, "c2", 4.0)(model, xv, d), xv)[0].flatten()
                        if on:
                            with torch.no_grad():
                                state["on"] = False
                                f0 = extract_features(model, x, d)[2]
                            rel.append(float((fc[2] - f0).norm() / f0.norm()))
                    cos.append(float(torch.nn.functional.cosine_similarity(g[True], g[False], dim=0)))
                r.update(rel_change_c4=float(np.mean(rel)), grad_cos=float(np.mean(cos)))
        res[str(b)] = r
        base = res["0.0"]["ap"]
        msg = f"β = {b}: AP sạch {100 * r['ap']:.1f} ({100 * r['ap'] / base:.0f}%)"
        if b > 0:
            ok = r["ap"] >= 0.9 * base and r["rel_change_c4"] >= 0.05 and r["grad_cos"] <= 0.95
            r["pass_h1_h3"] = bool(ok)
            msg += f" | ΔC4 {r['rel_change_c4']:.3f} | cos(g) {r['grad_cos']:.3f} -> H1+H3 {'QUA' if ok else 'TRƯỢT'}"
        print(msg, flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
