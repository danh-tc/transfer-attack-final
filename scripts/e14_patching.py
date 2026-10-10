#!/usr/bin/env python3
"""E14 (quy tắc khóa: progress_log 2026-10-10): DINO-Swin-L trung hòa δ transfer ở đâu — ghép feature backbone
(activation patching), độ lớn hay hướng? Chỉ forward, dev30, PNG có sẵn.

Backbone 4 stage cho ảnh sạch c_s và adv a_s; feature ghép -> neck -> transformer -> bbox_head.predict (rescale).
Điều kiện: full, noise_s / denoise_s (s = 1..4), noise_12, noise_34; transfer: up (khớp ‖·‖ WB từng stage), x2, x4;
WB: down_<T> (hướng WB, độ lớn T). D = 1 − tỉ lệ GT (phát hiện ở sạch, score ≥ 0.3) còn score ≥ 0.3.
  python scripts/e14_patching.py                -> artifacts/e14/dets.json + results/mechanism/e14_patching.json
  python scripts/e14_patching.py --analyze-only
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
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import CocoImageEval
from attack.preprocess import pipeline_resize, to_batch
from scripts.e4_target_side import matched

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py")
WB = "WB-OSFD"
SRC = {WB: os.path.join(REPO_ROOT, "artifacts/runs/e8_wbosfd_dev30/WB-OSFD"),
       "OSFD": os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv/OSFD"),
       "OCT": os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv/OCT"),
       "C2-PIA": os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5/adv/C2-PIA")}
TRANSFER = [s for s in SRC if s != WB]
DETS = os.path.join(REPO_ROOT, "artifacts/e14/dets.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e14_patching.json")
BOOT_SEED, N_BOOT, SCORE = 2026, 1000, 0.3


def conds_for(src):
    c = ["full"] + [f"noise_{s}" for s in range(1, 5)] + [f"denoise_{s}" for s in range(1, 5)] + ["noise_12", "noise_34"]
    return c + (["up", "x2", "x4"] if src != WB else [f"down_{t}" for t in TRANSFER])


@torch.no_grad()
def head(model, feats, dss):
    """Feature backbone (4 stage) -> list detection (khung ảnh gốc)."""
    hi = model.forward_transformer(model.neck(tuple(feats)), dss)
    r = model.bbox_head.predict(**hi, rescale=True, batch_data_samples=dss)[0]
    return r.bboxes.cpu().numpy(), r.scores.cpu().numpy(), r.labels.cpu().numpy()


def to_coco(pred, img_id, cat_ids):
    b, s, l = pred
    return [{"image_id": img_id, "category_id": cat_ids[int(li)],
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(si)}
            for (x1, y1, x2, y2), si, li in zip(b, s, l)]


@torch.no_grad()
def run(coco, ids):
    from mmdet.apis import init_detector
    dev = "cuda:0"
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    cat_ids = sorted(coco.getCatIds())
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    dets = {"clean": []}
    dets.update({f"{s}|{c}": [] for s in SRC for c in conds_for(s)})
    dist = {s: [] for s in SRC}     # mỗi ảnh: [4, 3] (cosd, rel, ‖a − c‖)
    sanity = {"predict_maxabs_score": 0.0}
    t0 = time.time()
    for n, img_id in enumerate(ids):
        smp = data[pos[img_id]]
        x, ds = smp["inputs"].to(dev), smp["data_sample"]
        xb, dss = to_batch(model, [x], [ds])
        c = [f for f in model.backbone(xb)]
        pc = head(model, c, dss)
        dets["clean"] += to_coco(pc, img_id, cat_ids)
        if n < 3:  # đường ghép (backbone -> head) trên sạch phải trùng model.predict
            pr = model.predict(xb, dss, rescale=True)[0].pred_instances
            sanity["predict_maxabs_score"] = max(sanity["predict_maxabs_score"],
                                                        float(np.abs(pr.scores.cpu().numpy() - pc[1]).max()))
        A = {}
        for s, root in SRC.items():
            a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(root, f"{img_id}.png")))
                                .permute(2, 0, 1).float().to(dev), x.shape[-2:])
            ab, _ = to_batch(model, [a], [ds])
            A[s] = [f for f in model.backbone(ab)]
            row = []
            for k in range(4):
                fa, fc = A[s][k].flatten().double(), c[k].flatten().double()
                nd = float((fa - fc).norm())
                row.append([float(1 - fa @ fc / (fa.norm() * fc.norm())), nd / float(fc.norm()), nd])
            dist[s].append(row)
        for s in SRC:
            a = A[s]
            nrm = lambda src, k: float((A[src][k] - c[k]).double().norm())
            for cond in conds_for(s):
                if cond == "full":
                    f = a
                elif cond.startswith("noise_"):
                    on = {int(ch) - 1 for ch in cond[6:]}
                    f = [a[k] if k in on else c[k] for k in range(4)]
                elif cond.startswith("denoise_"):
                    k0 = int(cond[-1]) - 1
                    f = [c[k] if k == k0 else a[k] for k in range(4)]
                elif cond == "up":
                    f = [c[k] + (nrm(WB, k) / max(nrm(s, k), 1e-12)) * (a[k] - c[k]) for k in range(4)]
                elif cond in ("x2", "x4"):
                    f = [c[k] + float(cond[1]) * (a[k] - c[k]) for k in range(4)]
                else:  # down_<T>
                    t = cond[5:]
                    f = [c[k] + (nrm(t, k) / max(nrm(WB, k), 1e-12)) * (a[k] - c[k]) for k in range(4)]
                dets[f"{s}|{cond}"] += to_coco(head(model, f, dss), img_id, cat_ids)
        if (n + 1) % 5 == 0:
            print(f"[E14] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(DETS), exist_ok=True)
    with open(DETS, "w") as f:
        json.dump({"dets": dets, "dist": dist, "sanity": sanity}, f)


def analyze(coco, ids):
    blob = json.load(open(DETS))
    dets, dist, sanity = blob["dets"], blob["dist"], blob["sanity"]
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ids)
    W = np.vstack([np.ones((1, n)), [np.bincount(r, minlength=n) for r in rng.integers(0, n, (N_BOOT, n))]])
    cl = matched(coco, dets["clean"], ids)
    rows = [(k, j) for k, i in enumerate(ids) for j, sc in enumerate(cl[i][1]) if sc >= SCORE]
    img = np.array([r[0] for r in rows])
    w = W[:, img]

    def D(key):
        m = matched(coco, dets[key], ids)
        y = np.array([m[ids[k]][1][j] < SCORE for k, j in rows], dtype=float)
        return (w * y).sum(1) / w.sum(1)

    ev_clean = CocoImageEval(coco, dets["clean"], ids).ap(np.ones(n))[0]
    apd = lambda key: float(100 * (ev_clean - CocoImageEval(coco, dets[key], ids).ap(np.ones(n))[0]) / ev_clean)
    Ds = {k: D(k) for k in dets if k != "clean"}
    ci = lambda v: [float(x) for x in np.percentile(v[1:], [2.5, 97.5])]
    dwb = Ds[f"{WB}|full"]
    out = {"n_images": n, "n_gt_detected_clean": len(rows), "sanity": sanity,
           "D": {k: {"point": float(v[0]), "ci95": ci(v), "r_vs_WB": float(v[0] / dwb[0]),
                     "r_ci95": ci(v / dwb)} for k, v in Ds.items()},
           "ap_drop": {k: apd(k) for k in dets if k != "clean"},
           "dist": {s: {f"bb{k + 1}": {"cosd": float(np.mean([r[k][0] for r in dist[s]])),
                                        "rel": float(np.mean([r[k][1] for r in dist[s]]))} for k in range(4)} for s in SRC},
           "alpha_up_median": {t: [float(np.median([dist[WB][i][k][2] / dist[t][i][k][2] for i in range(n)]))
                                   for k in range(4)] for t in TRANSFER}}
    out["sanity"]["ap_drop_full"] = {s: out["ap_drop"][f"{s}|full"] for s in SRC}
    # Quy tắc
    p = lambda k: out["D"][k]["point"]
    full_wb = p(f"{WB}|full")
    out["R1_stages_carry_WB"] = [s for s in range(1, 5) if p(f"{WB}|noise_{s}") >= 0.5 * full_wb]
    rel = lambda s: np.mean([out["dist"][s]["bb1"]["rel"], out["dist"][s]["bb2"]["rel"]])
    out["R2_shallow"] = bool(p(f"{WB}|noise_12") >= 0.5 * full_wb and rel(WB) >= 2 * rel("OSFD"))
    out["R3"] = {}
    for t in TRANSFER:
        r_T, r_up, r_down = p(f"{t}|full") / full_wb, p(f"{t}|up") / full_wb, p(f"{WB}|down_{t}") / full_wb
        lab = ("MAG" if r_up >= 0.7 and r_down <= r_T + 0.2 else
               "DIR" if r_up <= r_T + 0.2 and r_down >= 0.6 else "XAM")
        out["R3"][t] = {"r_T": r_T, "r_up": r_up, "r_down": r_down, "label": lab}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("SANITY", json.dumps(out["sanity"], ensure_ascii=False), "| GT phát hiện ở sạch:", len(rows))
    print("\n=== distortion backbone Swin-L: cosd / rel theo stage ===")
    for s in SRC:
        print(f"{s:<8} " + "  ".join(f"bb{k}: {out['dist'][s][f'bb{k}']['cosd']:.3f}/{out['dist'][s][f'bb{k}']['rel']:.3f}"
                                     for k in range(1, 5)))
    print("α_up trung vị theo stage:", {t: [round(a, 2) for a in v] for t, v in out["alpha_up_median"].items()})
    print(f"\n{'điều kiện':<22}{'D':>7} {'CI95':>16} {'r/WB':>7} {'APdrop':>8}")
    for s in SRC:
        for cond in conds_for(s):
            k = f"{s}|{cond}"
            v = out["D"][k]
            print(f"{k:<22}{v['point']:7.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] {v['r_vs_WB']:7.2f} {out['ap_drop'][k]:8.1f}")
        print()
    print("R1 stage gánh thiệt hại WB:", out["R1_stages_carry_WB"], "| R2 nút thắt nông:", out["R2_shallow"])
    for t, v in out["R3"].items():
        print(f"R3 {t:<7} r_T {v['r_T']:.2f}  r_up {v['r_up']:.2f}  r_down {v['r_down']:.2f}  -> {v['label']}")


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
