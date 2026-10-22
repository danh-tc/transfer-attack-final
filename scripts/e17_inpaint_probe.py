#!/usr/bin/env python3
"""E17 (progress_log 2026-10-22): hướng NGỮ NGHĨA "xóa object" trong không gian pixel có trung lập họ backbone không?

x_bg = ảnh gốc inpaint (OpenCV Telea, bán kính 5) trong mọi GT box không-crowd (không model học -> không thêm surrogate).
Điều kiện (PNG, ảnh gốc cỡ gốc, kẹp [0, 255]):
  INP    δ = 5·sign(x_bg − x) trong GT box, 0 ngoài box (nhích pixel object về phía nền đúng ±5)
  RNDBOX δ = 5·(±1 ngẫu nhiên) trong GT box (đối chứng: cùng vùng, cùng L∞, hướng ngẫu nhiên; seed 2026 + img_id)
  WP2    PNG C2-OCT-1V-WP2 B36 dev100 có sẵn (transfer từ R50, mốc)
Đo relative AP drop (inference_detector trên file, cùng cách cho clean) trên 6 model, paired bootstrap.
Chạy: python scripts/e17_inpaint_probe.py   (resume: dets mỗi model lưu artifacts/e17/dets_<model>.json)
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.models import CONTROLLED_PANEL

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CKPT = os.path.join(REPO_ROOT, "checkpoints")
ART = os.path.join(REPO_ROOT, "artifacts/e17")
WP2 = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B36_eps5_limit100/adv/C2-OCT-1V-WP2")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e17_inpaint_probe.json")
BOOT_SEED, N_BOOT, EPS = 2026, 1000, 5
CONDS = ["clean", "INP", "RNDBOX", "WP2"]
MODELS = {k: (CONTROLLED_PANEL[k].config_path, None) for k in ("r50", "r101", "convnext_t", "swin_t")}
MODELS["yolox_s"] = (os.path.join(CFG, "yolox/yolox_s_8xb8-300e_coco.py"), "yolox_s_8x8*")
MODELS["dino_swin_l"] = (os.path.join(CFG, "dino/dino-5scale_swin-l_8xb2-36e_coco.py"), "dino-5scale_swin-l*")


def path(coco, img_id, cond):
    if cond == "clean":
        return os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])
    if cond == "WP2":
        return os.path.join(WP2, f"{img_id}.png")
    return os.path.join(ART, "adv", cond, f"{img_id}.png")


def build(coco, img_ids):
    """PNG INP / RNDBOX (resume) + mô tả: tỉ lệ pixel trong box, tỉ lệ pixel INP có δ ≠ 0."""
    stat = {}
    for img_id in img_ids:
        x = cv2.imread(path(coco, img_id, "clean"))
        h, w = x.shape[:2]
        m = np.zeros((h, w), np.uint8)
        for bx, by, bw, bh in (a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))):
            m[int(by):int(np.ceil(by + bh)), int(bx):int(np.ceil(bx + bw))] = 1
        xbg = cv2.inpaint(x, m, 5, cv2.INPAINT_TELEA)
        mb = m.astype(bool)[..., None]
        d_inp = EPS * np.sign(xbg.astype(int) - x.astype(int)) * mb
        rng = np.random.default_rng(2026 + img_id)
        d_rnd = EPS * rng.choice([-1, 1], size=x.shape) * mb
        for cond, d in (("INP", d_inp), ("RNDBOX", d_rnd)):
            p = path(coco, img_id, cond)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                cv2.imwrite(p, np.clip(x.astype(int) + d, 0, 255).astype(np.uint8))
        stat[img_id] = {"box_frac": float(m.mean()), "inp_nonzero_in_box": float((d_inp != 0).sum() / max(1, 3 * m.sum()))}
    return stat


def predict(key, coco, img_ids):
    out = os.path.join(ART, f"dets_{key}.json")
    dets = json.load(open(out)) if os.path.exists(out) else {}
    todo = [c for c in CONDS if c not in dets]
    if not todo:
        return dets
    from mmdet.apis import inference_detector, init_detector
    cfg, pat = MODELS[key]
    ckpt = CONTROLLED_PANEL[key].checkpoint_path if pat is None else glob.glob(os.path.join(CKPT, pat + ".pth"))[0]
    model = init_detector(cfg, ckpt, device="cuda:0")
    cat_ids = sorted(coco.getCatIds())
    for c in todo:
        t0 = time.time()
        dets[c] = []
        for img_id in img_ids:
            dets[c] += to_coco_dets(inference_detector(model, path(coco, img_id, c)), img_id, cat_ids)
        print(f"[e17 {key}] {c} ({time.time() - t0:.0f}s)", flush=True)
        with open(out, "w") as f:  # lưu sau mỗi điều kiện -> resume được
            json.dump(dets, f)
    del model
    return dets


def main():
    coco = COCO(ANN_FILE)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    img_ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(img_ids) == 100
    stat = build(coco, img_ids)
    print("[e17] PNG xong", flush=True)
    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    S = lambda a: summarize(a[0], a[1:])
    drop = {}
    for key in MODELS:
        dets = predict(key, coco, img_ids)
        ev = {c: CocoImageEval(coco, dets[c], img_ids) for c in CONDS}
        ap = {c: np.array([ev[c].ap(w) for w in W])[:, 0] * 100 for c in CONDS}
        drop[key] = {c: 100 * (ap["clean"] - ap[c]) / ap["clean"] for c in CONDS[1:]}
    cnn = lambda c: np.mean([drop[k][c] for k in ("r101", "convnext_t", "yolox_s")], axis=0)
    res = {"n_images": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "relative_ap_drop": {k: {c: S(v) for c, v in d.items()} for k, d in drop.items()},
           "swinl_over_cnn3": {c: float(drop["dino_swin_l"][c][0] / max(cnn(c)[0], 1e-8)) for c in CONDS[1:]},
           "swinl_INP_minus_RNDBOX": S(drop["dino_swin_l"]["INP"] - drop["dino_swin_l"]["RNDBOX"]),
           "cnn3_INP_minus_RNDBOX": S(cnn("INP") - cnn("RNDBOX")),
           "box_frac_mean": float(np.mean([s["box_frac"] for s in stat.values()])),
           "inp_nonzero_in_box_mean": float(np.mean([s["inp_nonzero_in_box"] for s in stat.values()])),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    print(f"\n=== relative AP drop % (n = {n}) ===\n{'':<8}" + "".join(f"{k:>20}" for k in MODELS) + "   SwinL/CNN3")
    for c in CONDS[1:]:
        print(f"{c:<8}" + "".join(f"{fmt(res['relative_ap_drop'][k][c]):>20}" for k in MODELS) + f"   {res['swinl_over_cnn3'][c]:.2f}")
    print("Swin-L INP − RNDBOX:", fmt(res["swinl_INP_minus_RNDBOX"]), "| CNN3 INP − RNDBOX:", fmt(res["cnn3_INP_minus_RNDBOX"]))
    print(f"tỉ lệ pixel trong box {res['box_frac_mean']:.2f}, INP δ ≠ 0 trong box {res['inp_nonzero_in_box_mean']:.2f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
