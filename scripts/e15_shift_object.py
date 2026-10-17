#!/usr/bin/env python3
"""E15 (quy tắc khóa: progress_log 2026-10-17): kênh "DỜI / TÁCH object" trên DINO-Swin-L — dời nội dung token object
ở bb3 (stride 16) có làm DINO mất box đúng (và đặt box ở chỗ mới) không? Chỉ forward, khung `noise_3` của E14 (mọi stage
sạch, riêng bb3 sửa), dev100.

Thao tác trên c3 (token SẠCH bb3, [C, h, w]); object = hình chữ nhật token của GT box, dời ρ = 0.5 × bề rộng box (token,
≥ 1) về phía còn chỗ (phải nếu đủ chỗ, không thì trái) — IoU box dời vs box gốc ≈ 1/3 < 0.5:
- MOVE : mọi token object (gán GT, như E14c) ← μ_nền; dán nội dung chữ nhật SẠCH của từng GT ở vị trí dời (lớn → nhỏ).
- DUP  : giữ nguyên, dán thêm bản dời (object xuất hiện 2 chỗ — thử khử trùng lặp của DINO).
- SPLIT: token object ← μ_nền; nửa trái dời trái ρ/2·w, nửa phải dời phải ρ/2·w (2 mảnh tách rời).
Mỗi thao tác X: `X` = đầy đủ (trần: DINO có "đi theo" nội dung token không) và `X_T` = c3 + Δ_X·‖Δ_in^OSFD‖/‖Δ_X‖
(cùng độ lớn phần trong-object mà δ OSFD R50 thật tạo ra — mức "với tới được"). Mốc: `OSFD_in` = c3 + Δ_in^OSFD
(PNG OSFD B50 dev100 trên V100), `bg_T` = kéo token object về μ_nền ở độ lớn OSFD (synthT của E14c).
Chỉ số: D = 1 − tỉ lệ GT (phát hiện ở sạch: cùng lớp, IoU ≥ 0.5, score ≥ 0.3) còn score ≥ 0.3 (bootstrap ảnh 1000,
BOOT_SEED 2026); F = tỉ lệ GT đó có detection cùng lớp, score ≥ 0.3, IoU ≥ 0.5 với BOX ĐÃ DỜI (MOVE / DUP: box dời;
SPLIT: 1 trong 2 nửa); n_det = số detection ≥ 0.3 / ảnh.
  python scripts/e15_shift_object.py [--limit N]   -> artifacts/e15/dets.json + results/mechanism/e15_shift_object.json
  python scripts/e15_shift_object.py --analyze-only
"""
import argparse
import glob
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pycocotools.coco import COCO  # noqa: E402

from attack.data import build_attack_dataset, load_image_ids  # noqa: E402
from attack.preprocess import pipeline_resize, to_batch  # noqa: E402
from scripts.e4_target_side import matched  # noqa: E402
from scripts.e14_patching import ANN_FILE, CFG, head, to_coco  # noqa: E402
from scripts.e14c_bgshift import assign, bg_dirs  # noqa: E402

STRIDE, RHO, SCORE = 16, 0.5, 0.3
OSFD_DIR = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5_limit100/adv/OSFD")
MANIP = ["MOVE", "DUP", "SPLIT"]
CONDS = MANIP + [f"{m}_T" for m in MANIP] + ["OSFD_in", "bg_T"]
DETS = os.path.join(REPO_ROOT, "artifacts/e15/dets.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e15_shift_object.json")
BOOT_SEED, N_BOOT = 2026, 1000


def gt_rects(ds, h, w):
    """[(g, x1, y1, x2, y2) theo token (x2, y2 loại trừ), box khung model], lớn → nhỏ."""
    gi = ds.gt_instances
    b = gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes
    keep = ~gi.ignore_flags.bool() if "ignore_flags" in gi else torch.ones(len(b), dtype=torch.bool)
    out = []
    for g, (x1, y1, x2, y2) in enumerate(b[keep].float().tolist()):
        tx1, ty1 = min(int(x1 // STRIDE), w - 1), min(int(y1 // STRIDE), h - 1)
        tx2, ty2 = max(tx1 + 1, min(int(np.ceil(x2 / STRIDE)), w)), max(ty1 + 1, min(int(np.ceil(y2 / STRIDE)), h))
        out.append((g, tx1, ty1, tx2, ty2, (x2 - x1) * (y2 - y1)))
    return sorted(out, key=lambda r: -r[5])


def shift_amount(tx1, tx2, w):
    """Dời (token, có dấu): ρ·bề rộng, ≥ 1, về phía còn chỗ; bị kẹp trong ảnh."""
    d = max(1, int(round(RHO * (tx2 - tx1))))
    if tx2 + d <= w:
        return d
    if tx1 - d >= 0:
        return -d
    return (w - tx2) if (w - tx2) >= tx1 else -tx1


def paste(out, src, x1, y1, x2, y2, dx):
    """Dán src[:, y1:y2, x1:x2] vào out ở cột dời dx (cắt phần ra ngoài)."""
    w = out.shape[2]
    a, b = max(0, x1 + dx), min(w, x2 + dx)
    if b > a:
        out[:, y1:y2, a:b] = src[:, y1:y2, a - dx:b - dx]


def manipulate(c3, lab, rects):
    """-> {MOVE, DUP, SPLIT: c3 đã sửa}, {tên: [(g, dx_token) | (g, dx_trái, dx_phải)]}."""
    h, w = c3.shape[1:]
    bg = lab < 0
    mu = c3[:, bg].mean(1) if bg.any() else c3.reshape(c3.shape[0], -1).mean(1)
    obj = lab >= 0
    erased = c3.clone()
    erased[:, obj] = mu[:, None]
    res = {"MOVE": erased.clone(), "DUP": c3.clone(), "SPLIT": erased.clone()}
    shifts = {"MOVE": [], "DUP": [], "SPLIT": []}
    for g, x1, y1, x2, y2, _ in rects:
        d = shift_amount(x1, x2, w)
        paste(res["MOVE"], c3, x1, y1, x2, y2, d)
        paste(res["DUP"], c3, x1, y1, x2, y2, d)
        shifts["MOVE"].append((g, d)); shifts["DUP"].append((g, d))
        xm = (x1 + x2) // 2
        s = max(1, int(round(RHO / 2 * (x2 - x1))))
        if xm > x1:
            paste(res["SPLIT"], c3, x1, y1, xm, y2, -s)
        paste(res["SPLIT"], c3, xm, y1, x2, y2, s)
        shifts["SPLIT"].append((g, -s, s, xm))
    return res, shifts


def follow(pred, boxes_orig, labels, clean_hit, shifts, kind, scale_x):
    """Số GT (đã phát hiện ở sạch) có detection cùng lớp, score ≥ 0.3, IoU ≥ 0.5 với box đã dời (khung gốc)."""
    pb, ps, pl = pred
    keep = ps >= SCORE
    pb, pl = pb[keep], pl[keep]

    def hit(box, lab):
        if len(pb) == 0:
            return False
        ix1, iy1 = np.maximum(pb[:, 0], box[0]), np.maximum(pb[:, 1], box[1])
        ix2, iy2 = np.minimum(pb[:, 2], box[2]), np.minimum(pb[:, 3], box[3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
        return bool(((inter / np.maximum(area, 1e-9) >= 0.5) & (pl == lab)).any())

    n = 0
    for sh in shifts:
        g = sh[0]
        if not clean_hit[g]:
            continue
        b, lab = boxes_orig[g], labels[g]
        if kind == "SPLIT":
            _, sl, sr, xm = sh
            xm_o = xm * STRIDE / scale_x
            left = np.array([b[0] + sl * STRIDE / scale_x, b[1], min(xm_o, b[2]) + sl * STRIDE / scale_x, b[3]])
            right = np.array([max(xm_o, b[0]) + sr * STRIDE / scale_x, b[1], b[2] + sr * STRIDE / scale_x, b[3]])
            n += hit(left, lab) or hit(right, lab)
        else:
            dx = sh[1] * STRIDE / scale_x
            n += hit(np.array([b[0] + dx, b[1], b[2] + dx, b[3]]), lab)
    return n


@torch.no_grad()
def run(coco, ids):
    from mmdet.apis import init_detector
    dev = "cuda:0"
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    cat_ids = sorted(coco.getCatIds())
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    dets = {c: [] for c in ["clean"] + CONDS}
    fol = {c: [0, 0] for c in MANIP + [f"{m}_T" for m in MANIP]}  # [đi theo, GT đã phát hiện ở sạch]
    t0 = time.time()
    for n, img_id in enumerate(ids):
        smp = data[pos[img_id]]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        xb, dss = to_batch(model, [x], [ds])
        c = list(model.backbone(xb))
        pc = head(model, c, dss)
        dets["clean"] += to_coco(pc, img_id, cat_ids)
        c3 = c[2][0]
        lab = assign(ds, *c3.shape[1:], dev)
        obj = lab >= 0
        rects = gt_rects(ds, *c3.shape[1:])
        # GT khung gốc + đã phát hiện ở sạch (để đo "đi theo")
        gi = ds.gt_instances
        gb = gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes
        keep = ~gi.ignore_flags.bool() if "ignore_flags" in gi else torch.ones(len(gb), dtype=torch.bool)
        sx, sy = float(ds.scale_factor[0]), float(ds.scale_factor[1])
        boxes_orig = (gb[keep].float().cpu().numpy() / np.array([sx, sy, sx, sy]))
        labels = gi.labels[keep].cpu().numpy()
        clean_hit = [False] * len(boxes_orig)
        for g in range(len(boxes_orig)):
            clean_hit[g] = follow(pc, boxes_orig, labels, [True] * len(boxes_orig), [(g, 0)], "MOVE", sx) > 0
        # mốc OSFD (phần trong-object) + nền ở độ lớn OSFD
        a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(OSFD_DIR, f"{img_id}.png"))).permute(2, 0, 1).float().to(dev),
                            x.shape[-2:])
        d_in = (model.backbone(to_batch(model, [a], [ds])[0])[2][0] - c3) * obj
        tgt = float(d_in.norm())
        U, _ = bg_dirs(c3, lab)
        feats = {"OSFD_in": c3 + d_in, "bg_T": c3 + U * (tgt / max(float(U.norm()), 1e-12))}
        res, shifts = manipulate(c3, lab, rects)
        for m in MANIP:
            dlt = res[m] - c3
            feats[m] = res[m]
            feats[f"{m}_T"] = c3 + dlt * (tgt / max(float(dlt.norm()), 1e-12))
        for cond, f3 in feats.items():
            pr = head(model, [c[0], c[1], f3[None], c[3]], dss)
            dets[cond] += to_coco(pr, img_id, cat_ids)
            if cond in fol:
                kind = cond.split("_")[0]
                fol[cond][0] += follow(pr, boxes_orig, labels, clean_hit, shifts[kind], kind, sx)
                fol[cond][1] += sum(clean_hit)
        if (n + 1) % 10 == 0:
            print(f"[E15] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump({"dets": dets, "follow": fol}, f)


def analyze(coco, ids):
    blob = json.load(open(DETS))
    dets, fol = blob["dets"], blob["follow"]
    n = len(ids)
    rng = np.random.default_rng(BOOT_SEED)
    W = np.vstack([np.ones((1, n)), [np.bincount(r, minlength=n) for r in rng.integers(0, n, (N_BOOT, n))]])
    cl = matched(coco, dets["clean"], ids)
    rows = [(k, j) for k, i in enumerate(ids) for j, sc in enumerate(cl[i][1]) if sc >= SCORE]
    w = W[:, np.array([r[0] for r in rows])]

    def D(key):
        mm = matched(coco, dets[key], ids)
        y = np.array([mm[ids[k]][1][j] < SCORE for k, j in rows], dtype=float)
        return (w * y).sum(1) / w.sum(1)

    ci = lambda v: [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
    ndet = lambda key: sum(d["score"] >= SCORE for d in dets[key]) / n
    Ds = {k: D(k) for k in CONDS}
    out = {"n_images": n, "n_gt_detected_clean": len(rows), "n_det_clean": ndet("clean"),
           "D": {k: {"point": float(v[0]), "ci95": ci(v)} for k, v in Ds.items()},
           "D_minus_OSFD_in": {k: {"point": float(v[0] - Ds["OSFD_in"][0]), "ci95": ci(v - Ds["OSFD_in"])}
                               for k, v in Ds.items() if k != "OSFD_in"},
           "follow": {k: {"rate": v[0] / max(v[1], 1), "n": v[1]} for k, v in fol.items()},
           "n_det": {k: ndet(k) for k in CONDS}}
    p = lambda k: out["D"][k]["point"]
    best_T = max(MANIP, key=lambda m: p(f"{m}_T"))
    out["verdict"] = {
        "K1_channel_exists": {m: bool(p(m) >= 0.5 and out["follow"][m]["rate"] >= 0.3) for m in MANIP},
        "best_T": best_T,
        "K2_reachable": bool(p(f"{best_T}_T") >= 2 * p("OSFD_in")
                             and out["D_minus_OSFD_in"][f"{best_T}_T"]["ci95"][0] > 0),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"GT phát hiện ở sạch: {len(rows)}   n_det sạch / ảnh: {out['n_det_clean']:.1f}")
    print(f"{'điều kiện':<10}{'D':>8}{'CI95':>18}{'D−OSFD_in':>12}{'theo dời':>10}{'n_det/ảnh':>11}")
    for k in CONDS:
        v, dd = out["D"][k], out["D_minus_OSFD_in"].get(k)
        fr = out["follow"].get(k)
        dd_s = f"{dd['point']:+12.3f}" if dd else " " * 12
        fr_s = f"{fr['rate']:10.2f}" if fr else " " * 10
        print(f"{k:<10}{v['point']:8.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]{dd_s}{fr_s}{out['n_det'][k]:11.1f}")
    print("\nVERDICT", out["verdict"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(coco, ids)
    analyze(coco, ids)


if __name__ == "__main__":
    main()
