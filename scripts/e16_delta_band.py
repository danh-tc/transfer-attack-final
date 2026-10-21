#!/usr/bin/env python3
"""E16 (progress_log 2026-10-21): phần nào của δ_SwinT mang transfer sang DINO-Swin-L?

δ_X = PNG adv − ảnh gốc (C2-OCT-1V dev100 B41; X = R50 | SwinT surrogate). Biến thể mỗi kênh riêng:
LP_σ (Gauss σ ∈ {1, 2, 4}), HP_2 = δ − LP_2, GRAY = trung bình 3 kênh; co giãn lại L∞ = 5, làm tròn,
kẹp [0, 255]. FULL = PNG gốc. Đo relative AP drop (inference_detector trên file, cùng cách cho clean)
trên DINO-Swin-L + Mask R-CNN R50 / Swin-T, paired bootstrap; vân tay δ theo ảnh (CPU).

Chạy: python scripts/e16_delta_band.py   (resume: dets mỗi model lưu artifacts/e16/dets_<model>.json)
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
ART = os.path.join(REPO_ROOT, "artifacts/e16")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e16_delta_band.json")
BOOT_SEED, N_BOOT, EPS = 2026, 1000, 5.0
SOURCES = {"R50": "dev300_B41_eps5_limit100", "SwinT": "dev300_B41_eps5_limit100_sur-swin_t"}
VARIANTS = ["FULL", "LP1", "LP2", "LP4", "HP2", "GRAY"]
MODELS = {"dino_swin_l": (os.path.join(CFG, "dino/dino-5scale_swin-l_8xb2-36e_coco.py"), "dino-5scale_swin-l*"),
          "r50": (CONTROLLED_PANEL["r50"].config_path, None), "swin_t": (CONTROLLED_PANEL["swin_t"].config_path, None)}


def lp(d, s):
    return np.stack([cv2.GaussianBlur(d[..., c], (0, 0), s, borderType=cv2.BORDER_REFLECT) for c in range(3)], -1)


def variant(d, v):
    """d float32 HxWx3 (BGR, đơn vị /255) -> biến thể co giãn L∞ = EPS."""
    if v == "GRAY":
        f = np.repeat(d.mean(-1, keepdims=True), 3, -1)
    elif v == "HP2":
        f = d - lp(d, 2.0)
    else:
        f = lp(d, float(v[2:]))
    return f * (EPS / max(np.abs(f).max(), 1e-8))


def fingerprint(d, boxes):
    """Vân tay δ (năng lượng = tổng bình phương): 3 dải tần A4, tỉ lệ màu, mật độ năng lượng trong GT box."""
    h, w = d.shape[:2]
    P = sum(np.abs(np.fft.fft2(d[..., c])) ** 2 for c in range(3))
    fr = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    bands = [P[fr < 1 / 6].sum(), P[(fr >= 1 / 6) & (fr < 1 / 3)].sum(), P[fr >= 1 / 3].sum()]
    e = (d ** 2).sum(-1)
    chroma = ((d - d.mean(-1, keepdims=True)) ** 2).sum() / max(e.sum(), 1e-8)
    m = np.zeros((h, w), bool)
    for x, y, bw, bh in boxes:
        m[int(y):int(np.ceil(y + bh)), int(x):int(np.ceil(x + bw))] = True
    in_box = e[m].sum() / e.sum() / m.mean() if m.any() and m.mean() < 1 else np.nan
    return {"band": (np.array(bands) / sum(bands)).tolist(), "chroma": float(chroma), "in_box_density": float(in_box)}


def build(coco, img_ids):
    """PNG biến thể + vân tay (resume: PNG đã có thì không ghi lại)."""
    fp = {}
    for img_id in img_ids:
        x = cv2.imread(os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])).astype(np.float32)
        boxes = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]
        ds = {}
        for src, run in SOURCES.items():
            adv = cv2.imread(os.path.join(REPO_ROOT, "artifacts/runs", run, "adv/C2-OCT-1V", f"{img_id}.png")).astype(np.float32)
            assert adv.shape == x.shape and np.abs(adv - x).max() <= EPS, (src, img_id)
            ds[src] = adv - x
            for v in VARIANTS[1:]:
                p = os.path.join(ART, "adv", f"{src}-{v}", f"{img_id}.png")
                if not os.path.exists(p):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    cv2.imwrite(p, np.clip(np.round(x + variant(ds[src], v)), 0, 255).astype(np.uint8))
        fp[img_id] = {src: fingerprint(d, boxes) for src, d in ds.items()}
        a, b = ds["R50"].ravel(), ds["SwinT"].ravel()
        fp[img_id]["cos_R50_SwinT"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return fp


def path(coco, img_id, cond):
    if cond == "clean":
        return os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]["file_name"])
    src, v = cond.split("-")
    if v == "FULL":
        return os.path.join(REPO_ROOT, "artifacts/runs", SOURCES[src], "adv/C2-OCT-1V", f"{img_id}.png")
    return os.path.join(ART, "adv", cond, f"{img_id}.png")


def predict(key, coco, img_ids, conds):
    out = os.path.join(ART, f"dets_{key}.json")
    dets = json.load(open(out)) if os.path.exists(out) else {}
    todo = [c for c in conds if c not in dets]
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
        print(f"[e16 {key}] {c} ({time.time() - t0:.0f}s)", flush=True)
        with open(out, "w") as f:  # lưu sau mỗi điều kiện -> resume được
            json.dump(dets, f)
    del model
    return dets


def main():
    coco = COCO(ANN_FILE)
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    img_ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(img_ids) == 100
    fp = build(coco, img_ids)
    print("[e16] PNG biến thể + vân tay xong", flush=True)
    conds = ["clean"] + [f"{s}-{v}" for s in SOURCES for v in VARIANTS]

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    S = lambda a: summarize(a[0], a[1:])
    drop = {}
    for key in MODELS:
        dets = predict(key, coco, img_ids, conds)
        ev = {c: CocoImageEval(coco, dets[c], img_ids) for c in conds}
        ap = {c: np.array([ev[c].ap(w) for w in W])[:, 0] * 100 for c in conds}
        drop[key] = {c: 100 * (ap["clean"] - ap[c]) / ap["clean"] for c in conds[1:]}

    sw = drop["dino_swin_l"]
    base = sw["R50-FULL"]
    G = sw["SwinT-FULL"] - base
    res = {"n_images": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "relative_ap_drop": {k: {c: S(v) for c, v in d.items()} for k, d in drop.items()},
           "dino_vs_R50_FULL": {c: S(sw[c] - base) for c in conds[1:]},
           "retain_frac_of_G": {c: float((sw[c][0] - base[0]) / G[0]) for c in conds[1:]},
           "G": S(G)}
    lp_gain = {f"LP{s}": sw[f"R50-LP{s}"] - base for s in (1, 2, 4)}
    best = max(lp_gain, key=lambda k: lp_gain[k][0])
    A = lp_gain[best][0] >= 0.5 * G[0] and np.percentile(lp_gain[best][1:], 2.5) > 0
    B = res["retain_frac_of_G"]["SwinT-LP2"] >= 0.7
    C = res["retain_frac_of_G"]["SwinT-HP2"] >= 0.7
    res["reading"] = {"A_r50_lowpass_closes": bool(A), "best_r50_lp": best, "B_swint_low": bool(B), "C_swint_high": bool(C),
                      "verdict": "A" if A else ("B" if B else ("C" if C else "XAM"))}

    fpk = lambda src, f: np.array([fp[i][src][f] for i in img_ids], dtype=np.float64)
    wmean = lambda a: S(np.concatenate([[np.nanmean(a)], [np.nansum(a * w) / np.sum(w * ~np.isnan(a)) for w in W[1:]]]))
    res["fingerprint"] = {src: {"band_low": wmean(fpk(src, "band")[:, 0]), "band_mid": wmean(fpk(src, "band")[:, 1]),
                                "band_high": wmean(fpk(src, "band")[:, 2]), "chroma": wmean(fpk(src, "chroma")),
                                "in_box_density": wmean(fpk(src, "in_box_density"))} for src in SOURCES}
    res["fingerprint"]["cos_R50_SwinT"] = wmean(np.array([fp[i]["cos_R50_SwinT"] for i in img_ids]))
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    fmt = lambda s: f"{s['point']:5.1f} [{s['ci95'][0]:5.1f},{s['ci95'][1]:5.1f}]"
    print(f"\n=== relative AP drop % (n = {n}) ===\n{'':<12}" + "".join(f"{k:>21}" for k in MODELS) + f"{'Δ Swin-L vs R50-FULL':>24}  giữ/G")
    for c in conds[1:]:
        print(f"{c:<12}" + "".join(f"{fmt(res['relative_ap_drop'][k][c]):>21}" for k in MODELS)
              + f"{fmt(res['dino_vs_R50_FULL'][c]):>24}  {res['retain_frac_of_G'][c]:5.2f}")
    print("\nG =", fmt(res["G"]), " đọc:", res["reading"])
    for src in SOURCES:
        print(src, {k: round(v["point"], 3) for k, v in res["fingerprint"][src].items()})
    print("cos(δ_R50, δ_SwinT) =", round(res["fingerprint"]["cos_R50_SwinT"]["point"], 3))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
