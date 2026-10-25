#!/usr/bin/env python3
"""E36 Gate 1 (forward-only, progress_log 2026-10-25): REL = divergence ma trận affinity token–token (C4/C5, GT-box) giữa
δ và sạch, trên R50 θ0 cố định. Hỏi: REL có xếp δ theo mức phá DINO-Swin-L không (cross-δ VÀ dọc quỹ đạo CAP-WP-IS)?
Swin-L drop = suppression per-image (DINO inference trên PNG). Chạy: python scripts/e36_rel_probe.py"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import AttackDataset, load_image_ids
from attack.evaluation import to_coco_dets
from attack.mechanism import per_image_transfer
from attack.models import load_model
from attack.preprocess import extract_features, pipeline_resize, resize_to_model
from attack.targets import load_any
from mmdet.apis import inference_detector

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG = os.path.join(REPO_ROOT, "data/coco/val2017")
TAU = 0.1
# (nhãn, run, method) — oracle + C2/WP2 + biến thể + quỹ đạo CAP-WP-IS
DELTAS = [
    ("oracle δ_SwinT", "dev300_B41_eps5_limit100_sur-swin_t", "C2-OCT-1V"),
    ("C2-OCT-1V", "dev300_B41_eps5_limit100", "C2-OCT-1V"),
    ("WP2", "dev300_B36_eps5_limit100", "C2-OCT-1V-WP2"),
    ("WPA", "dev300_B36_eps5_limit100", "C2-OCT-1V-WPA"),
    ("noCT", "dev300_B41_eps5_limit100", "C2-OCT-1V-noCT"),
    ("RaPA", "dev300_B41_eps5_limit100", "C2-OCT-1V-RaPA"),
    ("LA1", "dev300_B41_eps5_limit100", "C2-OCT-1V-LA1"),
    ("FIA", "dev300_B41_eps5_limit100", "C2-OCT-1V-FIA"),
    ("RJ30", "dev300_B41_eps5_limit100", "C2-OCT-1V-RJ30"),
    ("WP2-XBG", "dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-XBG"),
]
TRAJ = [("IS-B4", "dev300_B4_eps5_limit100", "C2-OCT-1V-CAPWP-IS"),
        ("IS-B12", "dev300_B12_eps5_limit100", "C2-OCT-1V-CAPWP-IS"),
        ("IS-B24", "dev300_B24_eps5_limit100", "C2-OCT-1V-CAPWP-IS")]


def rel_div(feat_a, feat_c, boxes_model, tau=TAU):
    """Σ_{stage∈{C4,C5}} Σ_box symKL(A_adv, A_clean); A = softmax(F̂F̂ᵀ/τ), token = vị trí trong box, chuẩn hóa channel."""
    D, nb = 0.0, 0
    for fi, stride in ((2, 16), (3, 32)):
        Fa, Fc = feat_a[fi][0], feat_c[fi][0]  # [C,H,W]
        C, H, W = Fa.shape
        for (x1, y1, x2, y2) in boxes_model:
            xs, xe = int(x1 / stride), max(int(x1 / stride) + 1, int(np.ceil(x2 / stride)))
            ys, ye = int(y1 / stride), max(int(y1 / stride) + 1, int(np.ceil(y2 / stride)))
            xe, ye = min(xe, W), min(ye, H)
            ta = Fa[:, ys:ye, xs:xe].reshape(C, -1).T  # [N,C]
            tc = Fc[:, ys:ye, xs:xe].reshape(C, -1).T
            if ta.shape[0] < 2:
                continue
            ta = F.normalize(ta.float(), dim=1); tc = F.normalize(tc.float(), dim=1)
            Aa = F.softmax(ta @ ta.T / tau, dim=1); Ac = F.softmax(tc @ tc.T / tau, dim=1)
            kl = lambda p, q: (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(1).mean()
            D += float(0.5 * (kl(Aa, Ac) + kl(Ac, Aa))); nb += 1
    return D / max(nb, 1)


def main():
    dev = "cuda:0"
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    data = AttackDataset(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")))
    pos = {i: j for j, i in enumerate(data.image_ids)}
    r50 = load_model("r50", dev)
    dino = load_any("dino_swin_l", dev)
    A = os.path.join(REPO_ROOT, "artifacts/runs")

    # clean cache: R50 feature + DINO clean dets + GT boxes (model frame)
    cf, cdets, gtbox = {}, [], {}
    for i in ids:
        s = data[pos[i]]; ds = s["data_sample"]
        xc = resize_to_model(s["orig_inputs"].to(dev).float(), tuple(ds.img_shape))
        with torch.no_grad():
            cf[i] = tuple(f.detach() for f in extract_features(r50, xc, ds))
        gb = ds.gt_instances.bboxes
        gtbox[i] = (gb.tensor if hasattr(gb, "tensor") else gb).float().cpu().numpy().tolist()
        cdets += to_coco_dets(inference_detector(dino, os.path.join(IMG, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)

    rows = []
    for label, run, method in DELTAS + TRAJ:
        rel_v, adets = [], []
        for i in ids:
            ds = data[pos[i]]["data_sample"]
            png = os.path.join(A, run, "adv", method, f"{i}.png")
            adv = pipeline_resize(torch.from_numpy(mmcv.imread(png)).permute(2, 0, 1).float().to(dev), tuple(ds.img_shape))
            with torch.no_grad():
                fa = tuple(f.detach() for f in extract_features(r50, adv, ds))
            rel_v.append(rel_div(fa, cf[i], gtbox[i]))
            adets += to_coco_dets(inference_detector(dino, png), i, cat_ids)
        sup = per_image_transfer(coco, cdets, adets, ids)["suppression"]
        rows.append({"label": label, "rel": float(np.mean(rel_v)), "swinl_drop": float(np.nanmean(sup) * 100),
                     "rel_per_img": rel_v, "sup_per_img": sup.tolist()})
        print(f"{label:<16} REL {np.mean(rel_v):.4f}  Swin-L drop {np.nanmean(sup) * 100:.1f}", flush=True)

    # Gate 1a: Spearman(REL, Swin-L drop) qua DELTAS (không gồm quỹ đạo)
    dd = rows[:len(DELTAS)]
    rho_d, p_d = spearmanr([r["rel"] for r in dd], [r["swinl_drop"] for r in dd])
    orc = dd[0]["rel"]; wp2 = next(r["rel"] for r in dd if r["label"] == "WP2")
    # Gate 1b: REL dọc quỹ đạo (Swin-L sụp) — Spearman(REL, Swin-L drop) phải DƯƠNG (REL giảm khi drop giảm)
    tr = rows[len(DELTAS):]
    rho_t, _ = spearmanr([r["rel"] for r in tr], [r["swinl_drop"] for r in tr])
    out = {"deltas": rows, "gate1a_spearman_rel_swinl": float(rho_d), "gate1a_p": float(p_d),
           "oracle_rel": orc, "wp2_rel": wp2, "oracle_gt_wp2": bool(orc > wp2),
           "gate1b_traj_spearman": float(rho_t)}
    # verdict
    a_ok = rho_d > 0.3 and orc > wp2
    b_ok = rho_t > 0  # REL cùng chiều Swin-L drop dọc quỹ đạo (cả hai giảm)
    out["verdict"] = "QUA" if (a_ok and b_ok) else "DỪNG"
    json.dump(out, open(os.path.join(REPO_ROOT, "results/mechanism/e36_rel_probe.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\nGate1a: ρ(REL,Swin-L drop) qua {len(dd)} δ = {rho_d:+.3f} (p={p_d:.3f}); oracle REL {orc:.4f} vs WP2 {wp2:.4f} → oracle>WP2={orc>wp2}")
    print(f"Gate1b: quỹ đạo CAP-WP-IS ρ(REL, Swin-L drop) = {rho_t:+.3f} (dương = REL giảm khi Swin-L sụp)")
    print(f"VERDICT: {out['verdict']} (cần ρ_cross>0.3 & oracle>WP2 & ρ_traj>0)")


if __name__ == "__main__":
    main()
