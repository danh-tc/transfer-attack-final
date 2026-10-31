#!/usr/bin/env python3
"""E64 (progress_log 2026-10-31): vì sao δ từ surrogate ConvNeXt-T yếu trên họ ResNet? Mô tả, chỉ forward trên PNG dev30.
δ_R = RD-R50, δ_C = RD-CNX, δ_O = OSFD (E62 / E63). Đo:
  (1) vân tay δ: 3 dải tần A4, cos + đồng thuận dấu δ_C vs δ_R;
  (2) D_k = ||F_k(adv) - F_k(clean)|| / ||F_k(clean)|| theo stage, qua hook backbone trong inference_detector;
  (3) AP drop các biến thể dải: LP2, HP2 (như E16, co về L∞ = 5) + MIX = LP2(δ_C) + HP2(δ_R), kẹp ±5;
  (4) recall@0.3 theo cỡ GT, FP / ảnh trên R50.
  python scripts/e64_cnx_drop.py   -> results/mechanism/e64_cnx_drop.json"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.mechanism import _iou_xywh
from scripts.e16_delta_band import ANN_FILE, EPS, IMG_DIR, fingerprint, lp, variant
from scripts.eval_generalization import GEN_PANEL
from attack.models import CONTROLLED_PANEL

ART = os.path.join(REPO_ROOT, "artifacts/e64")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e64_cnx_drop.json")
RUNS = os.path.join(REPO_ROOT, "artifacts/runs")
SRC = {"R": ("dev300_B36_eps5_limit100", "C2-OCT-1V-WP2-RD"),
       "C": ("dev300_B36_eps5_limit100_sur-convnext_t", "C2-OCT-1V-WP2-RD-CNX"),
       "O": ("dev300_B50_eps5_limit100", "OSFD")}
BOOT_SEED, N_BOOT = 2026, 1000
FULL = ["R", "C", "O"]
BAND = ["R-LP2", "R-HP2", "C-LP2", "C-HP2", "MIX"]


def adv_path(s, img_id):
    run, name = SRC[s]
    return os.path.join(RUNS, run, "adv", name, f"{img_id}.png")


def cond_path(coco, c, img_id):
    if c == "clean":
        return os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])
    return adv_path(c, img_id) if c in FULL else os.path.join(ART, "adv", c, f"{img_id}.png")


def build(coco, ids):
    """PNG biến thể dải + vân tay δ."""
    fp = {"R": [], "C": [], "O": [], "cos_RC": [], "sign_RC": []}
    for img_id in ids:
        x = cv2.imread(cond_path(coco, "clean", img_id)).astype(np.float32)
        boxes = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]
        d = {s: cv2.imread(adv_path(s, img_id)).astype(np.float32) - x for s in FULL}
        for s in FULL:
            fp[s].append(fingerprint(d[s], boxes)["band"])
        a, b = d["R"].ravel(), d["C"].ravel()
        fp["cos_RC"].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
        fp["sign_RC"].append(float(np.mean(np.sign(a[(a != 0) & (b != 0)]) == np.sign(b[(a != 0) & (b != 0)]))))
        out = {f"{s}-{v}": variant(d[s], v) for s in "RC" for v in ("LP2", "HP2")}
        out["MIX"] = np.clip(lp(d["C"], 2.0) + (d["R"] - lp(d["R"], 2.0)), -EPS, EPS)
        for c, f in out.items():
            os.makedirs(os.path.join(ART, "adv", c), exist_ok=True)
            cv2.imwrite(os.path.join(ART, "adv", c, f"{img_id}.png"), np.clip(np.round(x + f), 0, 255).astype(np.uint8))
    return {k: (np.mean(v, 0).tolist() if k in FULL else float(np.mean(v))) for k, v in fp.items()}


def load(key):
    from mmdet.apis import init_detector
    if key in CONTROLLED_PANEL:
        p = CONTROLLED_PANEL[key]
        return init_detector(p.config_path, p.checkpoint_path, device="cuda:0")
    cfg, pat = GEN_PANEL[key]
    ckpt = glob.glob(os.path.join(REPO_ROOT, "checkpoints", pat + ".pth"))[0]
    return init_detector(os.path.join(REPO_ROOT, "third_party/mmdetection/configs", cfg), ckpt, device="cuda:0")


def run_model(key, coco, ids, conds):
    """dets từng điều kiện + D_k (chỉ điều kiện FULL)."""
    from mmdet.apis import inference_detector
    model = load(key)
    grab = {}
    h = model.backbone.register_forward_hook(lambda m, i, o: grab.__setitem__("f", [f.detach().float() for f in o]))
    cat_ids = sorted(coco.getCatIds())
    dets, dist = {c: [] for c in conds}, {s: [] for s in FULL}
    with torch.no_grad():
        for img_id in ids:
            dets["clean"] += to_coco_dets(inference_detector(model, cond_path(coco, "clean", img_id)), img_id, cat_ids)
            fc = grab["f"]
            for c in conds[1:]:
                dets[c] += to_coco_dets(inference_detector(model, cond_path(coco, c, img_id)), img_id, cat_ids)
                if c in FULL:
                    dist[c].append([float((a - b).norm() / b.norm()) for a, b in zip(grab["f"], fc)])
    h.remove()
    del model
    torch.cuda.empty_cache()
    return dets, {s: np.mean(v, 0).tolist() for s, v in dist.items()}


def by_size(coco, dets, ids):
    """recall@0.3 đúng lớp IoU ≥ 0.5 theo cỡ COCO, FP / ảnh (score ≥ 0.3)."""
    by = {}
    for d in dets:
        if d["score"] >= 0.3:
            by.setdefault(d["image_id"], []).append(d)
    hit, tot, fp = np.zeros(3), np.zeros(3), 0
    for i in ids:
        ds = by.get(i, [])
        b = np.array([d["bbox"] for d in ds]).reshape(-1, 4)
        c = np.array([d["category_id"] for d in ds])
        used = np.zeros(len(ds), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            s = 0 if a["area"] < 32 ** 2 else 1 if a["area"] < 96 ** 2 else 2
            tot[s] += 1
            if len(ds):
                ok = (_iou_xywh(np.array(a["bbox"]), b) >= 0.5) & (c == a["category_id"])
                used |= ok
                hit[s] += ok.any()
        fp += (~used).sum()
    return {"recall_s_m_l": (hit / np.maximum(tot, 1) * 100).round(1).tolist(), "n_s_m_l": tot.astype(int).tolist(), "fp_per_img": round(fp / len(ids), 2)}


def main():
    coco = COCO(ANN_FILE)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    res = {"fingerprint": build(coco, ids), "drop": {}, "D_k": {}, "size_r50": {}}
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    conds = ["clean"] + FULL + BAND
    for key in ("r50", "r101", "convnext_t", "swin_t", "dino_swin_l"):
        dets, res["D_k"][key] = run_model(key, coco, ids, conds)
        ap = {}
        for c in conds:
            e = CocoImageEval(coco, dets[c], ids)
            ap[c] = np.array([e.ap(w) for w in W])[:, 0]
        res["drop"][key] = {c: (lambda a: summarize(a[0], a[1:]))(100 * (ap["clean"] - ap[c]) / ap["clean"]) for c in conds[1:]}
        print(key, {c: round(res["drop"][key][c]["point"], 1) for c in conds[1:]}, flush=True)
        if key == "r50":
            res["size_r50"] = {c: by_size(coco, dets[c], ids) for c in ["clean"] + FULL}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fingerprint", "D_k", "size_r50")}, indent=0))


if __name__ == "__main__":
    main()
