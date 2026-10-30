#!/usr/bin/env python3
"""Bảng panel CHỐT (progress_log 2026-10-28): relative AP drop + ASR trên Controlled (r50 / r101 / convnext_t / swin_t, từ dets của
run_baselines) + PANEL_GEN (gen_dets của eval_generalization), cho các ô ở run khác nhau; paired bootstrap so 1 ô mốc.
ASR = trong các GT target detect được ở ảnh sạch (đúng lớp, IoU ≥ 0.5, score ≥ 0.3), tỉ lệ không còn detect (per_image_transfer).
T8 = 8 target hộp đen (bỏ r50 white-box). Ô dạng TÊN@RUN:FILE_DETS_CONTROLLED.
Thêm (progress_log 2026-10-28, theo benchmark 2602.16494 §3.1.1, IoU 0.5), mọi chỉ số dạng "% giảm so ảnh sạch" trừ ASR:
  aploc   = AP50 với MỌI lớp gộp thành 1 lớp (định vị / "thấy có object")
  csr0, csr3 = recall micro đúng lớp: mỗi GT lấy detection score cao nhất có IoU ≥ 0.5 (mọi lớp), TP nếu đúng lớp;
           csr0 không ngưỡng score (đúng chữ paper), csr3 chỉ xét detection score ≥ 0.3
  asr_micro = như ASR nhưng gộp theo object (Σ object mất / Σ object detect được ở ảnh sạch)
  python scripts/panel_table.py --subset dev30 --ref C2-OCT-1V-WP2-RD \
      --cells OSFD@dev300_B50_eps5_limit100:dets_e40_dev30.json C2-OCT-1V-WP2-RD@dev300_B36_eps5_limit100:dets_e45_dev30.json ...
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices
from attack.mechanism import DETECT_SCORE, MATCH_IOU, _iou_xywh, _matched_scores, per_image_transfer
from scripts.eval_generalization import PANEL_GEN

ART = os.path.join(REPO_ROOT, "artifacts/runs")
ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CTRL = ["r50", "r101", "convnext_t", "swin_t"]
T8 = ["r101", "convnext_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]
BOOT_SEED, N_BOOT = 2026, 1000


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def agnostic_coco(coco):
    """Bản GT gộp mọi lớp thành 1 lớp (APloc)."""
    ds = dict(coco.dataset)
    ds["annotations"] = [dict(a, category_id=1) for a in coco.dataset["annotations"]]
    ds["categories"] = [{"id": 1, "name": "object"}]
    g = COCO()
    g.dataset = ds
    with quiet():
        g.createIndex()
    return g


def csr_counts(coco, dets, ids, thr):
    """[n_img, 2]: (số GT không-crowd được detect ĐÚNG lớp, số GT) — detection score cao nhất IoU ≥ 0.5 mọi lớp."""
    by = {}
    for d in dets:
        if d["score"] >= thr:
            by.setdefault(d["image_id"], []).append(d)
    out = np.zeros((len(ids), 2))
    for i, img_id in enumerate(ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        ds = by.get(img_id, [])
        tp = 0
        if ds:
            boxes = np.array([d["bbox"] for d in ds])
            cats = np.array([d["category_id"] for d in ds])
            sc = np.array([d["score"] for d in ds])
            for a in anns:
                ok = _iou_xywh(np.array(a["bbox"]), boxes) >= MATCH_IOU
                if ok.any():
                    tp += int(cats[ok][np.argmax(sc[ok])] == a["category_id"])
        out[i] = tp, len(anns)
    return out


def asr_counts(coco, clean, adv, ids):
    """[n_img, 2]: (số object mất, số object detect được ở ảnh sạch) — gộp theo object."""
    sc, sa = _matched_scores(coco, clean, ids), _matched_scores(coco, adv, ids)
    out = np.zeros((len(ids), 2))
    for i, img_id in enumerate(ids):
        keep = sc[img_id] >= DETECT_SCORE
        out[i] = np.sum(sa[img_id][keep] < DETECT_SCORE), keep.sum()
    return out


def ratio(cnt, rows):
    """Tỉ lệ micro Σ cột 0 / Σ cột 1 theo các tập ảnh (rows: [R, n] chỉ số) -> [R]."""
    return np.array([cnt[r, 0].sum() / max(cnt[r, 1].sum(), 1e-9) for r in rows])



METRICS = ["drop", "asr", "asr_micro", "aploc", "csr0", "csr3"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="dev30")
    p.add_argument("--cells", nargs="+", required=True)
    p.add_argument("--ref", required=True, help="tên ô mốc cho Δ (paired bootstrap)")
    p.add_argument("--out", default=None)
    p.add_argument("--gen-models", nargs="+", default=None, help="giới hạn PANEL_GEN (vd dino_swin_l yolox_l) — bản gọn")
    args = p.parse_args()
    with quiet():
        coco = COCO(ANN_FILE)
    coco1 = agnostic_coco(coco)
    ids = load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.subset}.csv"))
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    rows = [np.arange(n)] + list(np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n)))
    models = CTRL + (args.gen_models if args.gen_models else PANEL_GEN)

    def ap(g, d, j):
        with quiet():
            ev = CocoImageEval(g, d, ids)  # khởi tạo 1 lần, tái dùng mọi trọng số bootstrap
        return np.array([ev.ap(w) for w in W])[:, j] * 100

    def pct_drop(c, a):
        return 100 * (c - a) / c

    # met[tên][chỉ số][model] = mảng [1 + N_BOOT] (phần tử 0 = điểm ước lượng)
    met = {}
    for cell in args.cells:
        name, rest = cell.split("@")
        run, dfile = rest.split(":")
        ctrl = json.load(open(os.path.join(ART, run, dfile)))
        met[name] = {m: {} for m in METRICS}
        for k in models:
            if k in CTRL:
                c, a = ctrl["clean"][k], ctrl[name][k]
            else:
                g = json.load(open(os.path.join(ART, run, "gen_dets", f"{k}.json")))
                c, a = g["clean"], g[name]
            met[name]["drop"][k] = pct_drop(ap(coco, c, 0), ap(coco, a, 0))
            agn = lambda ds: [dict(d, category_id=1) for d in ds]
            met[name]["aploc"][k] = pct_drop(ap(coco1, agn(c), 1), ap(coco1, agn(a), 1))
            with quiet():
                s = 100 * per_image_transfer(coco, c, a, ids)["suppression"]
            met[name]["asr"][k] = np.array([np.nanmean(s[r]) for r in rows])
            met[name]["asr_micro"][k] = 100 * ratio(asr_counts(coco, c, a, ids), rows)
            for key, thr in (("csr0", 0.0), ("csr3", DETECT_SCORE)):
                met[name][key][k] = pct_drop(ratio(csr_counts(coco, c, ids, thr), rows),
                                             ratio(csr_counts(coco, a, ids, thr), rows))
        for m in METRICS:
            met[name][m]["T8"] = np.nanmean([met[name][m][k] for k in T8 if k in met[name][m]], axis=0)

    cols = models + ["T8"]
    r2 = lambda x: round(float(x), 2)
    res = {"subset": args.subset, "n_images": n, "ref": args.ref, "metrics": METRICS,
           "note": "drop / aploc / csr0 / csr3 = % giảm so ảnh sạch; asr / asr_micro = % object mất (macro theo ảnh / micro theo object)",
           "table": {}, "delta_vs_ref": {}}
    for name in met:
        res["table"][name] = {k: {m: r2(met[name][m][k][0]) for m in METRICS} for k in cols}
        if name == args.ref:
            continue
        res["delta_vs_ref"][name] = {k: {m: [r2(d[0]), *[r2(v) for v in np.nanpercentile(d[1:], [2.5, 97.5])]]
                                         for m in METRICS for d in [met[name][m][k] - met[args.ref][m][k]]}
                                     for k in cols}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    for m in METRICS:
        print(f"\n[{m}]".ljust(21), " ".join(c[:11].rjust(11) for c in cols))
        for name, row in res["table"].items():
            print(name[:20].ljust(20), " ".join(f"{row[c][m]:11.1f}" for c in cols))
        for name, row in res["delta_vs_ref"].items():
            print(f"Δ {name[:18]}".ljust(20), " ".join(f"{row[c][m][0]:+11.1f}" for c in cols))
        for name, row in res["delta_vs_ref"].items():
            print(f"  CI {name[:15]}".ljust(20), " ".join(f"[{row[c][m][1]:+.0f},{row[c][m][2]:+.0f}]".rjust(11) for c in cols))


if __name__ == "__main__":
    main()
