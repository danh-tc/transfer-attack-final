#!/usr/bin/env python3
"""E10a (quy tắc khóa: progress_log 2026-10-07): δ ĐÃ đánh được Swin-L (white-box WB-OSFD, E8) có thành phần QUY MÔ LỚN
không, so với δ OSFD từ R50? Chỉ forward, dev30, PNG có sẵn.
Điều kiện trên δ = PNG − ảnh gốc: ID; phép E7 trong GT box (BLUR02/05, SHUF3/6, đúng `corrupt` của E6);
LP toàn ảnh σ ∈ {1, 2, 4} px. Ảnh sạch predict lại cùng script (dev30 không thuộc n300 nên không dùng dets E6).
  python scripts/e10a_wb_dissect.py                 -> artifacts/e10/dets.json + results/mechanism/e10a_wb_dissect.json
  python scripts/e10a_wb_dissect.py --analyze-only
"""
import argparse
import json
import os
import sys
import time

import cv2
import mmcv
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from scripts.e4_target_side import SIZES, matched
from scripts.e6_shape_texture import CONDS as BOX_CONDS, IMG_DIR, OTHERS, TARGETS, corrupt
from scripts.mech_a4_spectral import band_fracs, spatial

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SRC = {"WB-OSFD": os.path.join(REPO_ROOT, "artifacts/runs/e8_wbosfd_dev30/WB-OSFD"),
       "OSFD": os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv/OSFD")}
LP = {"LP1": 1.0, "LP2": 2.0, "LP4": 4.0}
CONDS = ["ID"] + list(BOX_CONDS) + list(LP)
OUT = os.path.join(REPO_ROOT, "results/mechanism")
DETS = os.path.join(REPO_ROOT, "artifacts/e10/dets.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def transform(d, anns, c, img_id):
    if c == "ID":
        return d
    if c in LP:
        return cv2.GaussianBlur(d, (0, 0), LP[c])
    kind, p = BOX_CONDS[c]
    return corrupt(d, anns, kind, p, np.random.default_rng(2026 + img_id))


def run(coco, ids):
    from mmdet.apis import inference_detector
    from attack.targets import load_any
    import torch
    cat_ids = sorted(coco.getCatIds())
    keys = ["clean"] + [f"{s}|{c}" for s in SRC for c in CONDS]
    dets = {t: {k: [] for k in keys} for t in TARGETS}
    for t in TARGETS:
        model = load_any(t)
        t0 = time.time()
        for n, i in enumerate(ids):
            x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
            dets[t]["clean"] += to_coco_dets(inference_detector(model, x.astype(np.uint8)), i, cat_ids)
            for s, root in SRC.items():
                d = mmcv.imread(os.path.join(root, f"{i}.png")).astype(np.float32) - x
                for c in CONDS:
                    im = np.clip(np.round(x + transform(d, anns, c, i)), 0, 255).astype(np.uint8)
                    dets[t][f"{s}|{c}"] += to_coco_dets(inference_detector(model, im), i, cat_ids)
            if (n + 1) % 10 == 0:
                print(f"[E10a {t}] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump(dets, f)


def spectra(coco, ids, W):
    """Mô tả: 3 dải năng lượng của δ + năng lượng trong GT box / diện tích box, theo nguồn."""
    out = {}
    for s, root in SRC.items():
        bf, ib = [], []
        for i in ids:
            x = mmcv.imread(os.path.join(IMG_DIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)
            d = (mmcv.imread(os.path.join(root, f"{i}.png")).astype(np.float32) - x).transpose(2, 0, 1)
            bf.append(band_fracs(d))
            ib.append(spatial(d, [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))])[1])
        bf, ib = np.array(bf), np.array(ib)
        mean = lambda v: (W @ np.nan_to_num(v)) / W.sum(1)
        out[s] = {"band_low_mid_high": [float(bf[:, b].mean()) for b in range(3)],
                  "band_low_ci95": [float(x) for x in np.percentile(mean(bf[:, 0])[1:], [2.5, 97.5])],
                  "in_box_ratio": float(np.nanmean(ib))}
    return out


def analyze(coco, ids):
    dets = json.load(open(DETS))
    pos = {i: k for k, i in enumerate(ids)}
    area = {a["id"]: a["area"] for a in coco.loadAnns(coco.getAnnIds(imgIds=ids, iscrowd=False))}
    zone = lambda ar: next(n for n, lo, hi in SIZES if lo <= ar < hi)
    zs = [n for n, _, _ in SIZES]
    rng = np.random.default_rng(BOOT_SEED)
    W = np.vstack([np.ones((1, len(ids))),
                   [np.bincount(r, minlength=len(ids)) for r in rng.integers(0, len(ids), (N_BOOT, len(ids)))]])
    A, SUP, N = {}, {}, {}
    for t in TARGETS:
        cl = matched(coco, dets[t]["clean"], ids)
        for k in dets[t]:
            if k == "clean":
                continue
            cc = matched(coco, dets[t][k], ids)
            rec = [(pos[i], zone(area[a]), sa / sc) for i in ids
                   for a, sc, sa in zip(cl[i][0], cl[i][1], cc[i][1]) if sc >= SCORE]
            for z in zs:
                r = [x for x in rec if x[1] == z]
                img = np.array([x[0] for x in r], dtype=int); y = np.array([x[2] for x in r]); w = W[:, img]
                den = np.maximum(w.sum(1), 1e-9)
                A[(t, k, z)] = 1 - (w * y).sum(1) / den
                SUP[(t, k, z)] = (w * (y < SCORE)).sum(1) / den  # nhất quán E8: suppression = score khớp < 0.3·s_clean
                N[(t, z)] = len(r)
    F = lambda t, s, c, z: A[(t, f"{s}|{c}", z)] / A[(t, f"{s}|ID", z)]
    ci = lambda v: [float(x) for x in np.nanpercentile(v[1:], [2.5, 97.5])]
    out = {"n_images": len(ids), "n_gt": {t: {z: N[(t, z)] for z in zs} for t in TARGETS},
           "A": {s: {t: {c: {z: float(A[(t, f"{s}|{c}", z)][0]) for z in zs} for c in CONDS} for t in TARGETS} for s in SRC},
           "F": {s: {t: {c: {z: {"point": float(F(t, s, c, z)[0]), "ci95": ci(F(t, s, c, z))} for z in zs}
                         for c in CONDS[1:]} for t in TARGETS} for s in SRC}}
    sw = "dino_swin_l"
    diff = F(sw, "WB-OSFD", "LP2", "large") - F(sw, "OSFD", "LP2", "large")
    f_wb = F(sw, "WB-OSFD", "LP2", "large")
    out["rule"] = {"F_WB_LP2_large": {"point": float(f_wb[0]), "ci95": ci(f_wb)},
                   "F_WB_minus_F_OSFD_LP2_large": {"point": float(diff[0]), "ci95": ci(diff)}}
    if f_wb[0] >= 0.5 and ci(diff)[0] > 0:
        verdict = "(i) ỦNG HỘ: δ đánh được Swin-L mang thành phần QUY MÔ LỚN"
    elif f_wb[0] <= 0.2:
        verdict = "(ii): Swin-L cũng bị đánh bằng mẫu tần cao khớp nội dung"
    else:
        verdict = "XÁM"
    out["rule"]["verdict"] = verdict
    # Sanity so E8 (AP drop 84.6, sup large 0.57): đọc qua cùng dets ID của script này
    ap = lambda k: CocoImageEval(coco, dets[sw][k], ids).ap(np.ones(len(ids)))[0]
    ap_clean = ap("clean")
    out["sanity_swinl"] = {s: {"ap_drop": float(100 * (ap_clean - ap(f"{s}|ID")) / ap_clean),
                               "sup": {z: float(SUP[(sw, f"{s}|ID", z)][0]) for z in zs}} for s in SRC}
    out["spectra"] = spectra(coco, ids, W)
    with open(os.path.join(OUT, "e10a_wb_dissect.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=== Sanity Swin-L (E8: WB-OSFD AP drop 84.6, sup large 0.57; OSFD 27.2, 0.25) ===")
    for s, v in out["sanity_swinl"].items():
        print(f"{s:<8} AP drop {v['ap_drop']:5.1f}  sup " + " / ".join(f"{v['sup'][z]:.2f}" for z in zs))
    for s in SRC:
        print(f"\n=== {s}: A(ID) nhỏ/vừa/lớn | F lớn theo điều kiện ===")
        print(f"{'':<12}{'A(ID)':<17}" + "".join(f"{c:>8}" for c in CONDS[1:]))
        for t in TARGETS:
            a = out["A"][s][t]["ID"]
            print(f"{t:<12}{a['small']:.2f}/{a['medium']:.2f}/{a['large']:.2f}   "
                  + "".join(f"{out['F'][s][t][c]['large']['point']:8.2f}" for c in CONDS[1:]))
    print("\n=== Swin-L, F theo kích thước (nhỏ / vừa / lớn), LP1 / LP2 / LP4 ===")
    for s in SRC:
        for c in LP:
            print(f"{s:<8} {c}: " + " / ".join(f"{out['F'][s][sw][c][z]['point']:.2f}" for z in zs))
    print("\n=== Phổ δ (thấp / trung / cao) | năng lượng trong box / diện tích ===")
    for s, v in out["spectra"].items():
        print(f"{s:<8} " + " / ".join(f"{x:.3f}" for x in v["band_low_mid_high"]) + f"  | in-box {v['in_box_ratio']:.2f}")
    r = out["rule"]
    print(f"\nQUY TẮC: F_WB(LP2, lớn) = {r['F_WB_LP2_large']['point']:.2f} {r['F_WB_LP2_large']['ci95']}; "
          f"F_WB − F_OSFD = {r['F_WB_minus_F_OSFD_LP2_large']['point']:+.2f} {r['F_WB_minus_F_OSFD_LP2_large']['ci95']}")
    print("KẾT LUẬN:", verdict)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(coco, ids)
    analyze(coco, ids)


if __name__ == "__main__":
    main()
