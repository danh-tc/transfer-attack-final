#!/usr/bin/env python3
"""E4 (quy tắc khóa: progress_log 2026-10-06): tổn thương theo GT của từng target có do thuộc tính object / độ tự tin của
chính target quyết định không. Chỉ đọc dets có sẵn (n300), không GPU.
  python scripts/e4_target_side.py -> results/mechanism/e4_target_side.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.mechanism import DETECT_SCORE, MATCH_IOU, _iou_xywh

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
BOOT_SEED, N_BOOT = 2026, 1000
FEATS = ["s_clean", "s_clean_r50", "log_area", "rel_area", "n_gt", "max_iou_other"]


def matched(coco, dets, img_ids):
    """{img_id: (ann_ids, score khớp tốt nhất mỗi GT)} — như mechanism._matched_scores nhưng giữ ann_id."""
    by = defaultdict(list)
    for d in dets:
        by[d["image_id"]].append(d)
    out = {}
    for i in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        ds, sc = by.get(i, []), np.zeros(len(anns))
        if ds:
            b = np.array([d["bbox"] for d in ds]); c = np.array([d["category_id"] for d in ds]); s = np.array([d["score"] for d in ds])
            for j, a in enumerate(anns):
                ok = (c == a["category_id"]) & (_iou_xywh(np.array(a["bbox"]), b) >= MATCH_IOU)
                sc[j] = s[ok].max() if ok.any() else 0.0
        out[i] = ([a["id"] for a in anns], sc)
    return out


def gt_table(coco, img_ids):
    rows = {}
    for i in img_ids:
        info = coco.loadImgs(i)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        boxes = np.array([a["bbox"] for a in anns]) if anns else np.zeros((0, 4))
        for j, a in enumerate(anns):
            others = np.delete(boxes, j, 0)
            rows[a["id"]] = {"img_id": i, "cat": a["category_id"], "log_area": float(np.log(max(a["area"], 1.0))),
                             "area": a["area"], "rel_area": a["area"] / (info["width"] * info["height"]),
                             "n_gt": len(anns),
                             "max_iou_other": float(_iou_xywh(np.array(a["bbox"]), others).max()) if len(others) else 0.0}
    return rows


def boot_rho(x, y, img, idx_imgs, groups):
    """Spearman theo GT, bootstrap cụm theo ảnh."""
    pt = spearmanr(x, y)[0]
    bs = []
    for r in idx_imgs:
        ii = np.concatenate([groups[k] for k in r])
        bs.append(spearmanr(x[ii], y[ii])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"rho": float(pt), "ci95": [float(lo), float(hi)]}


SIZES = (("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf")))


def delta_by_size(out, args, img_ids, idx_imgs):
    """E5: Δ suppression (ô − base) theo kích thước GT, CÙNG tập GT (target phát hiện ở ảnh sạch — không phụ thuộc ô);
    kèm large − small. Bootstrap cụm theo ảnh."""
    pos = {i: k for k, i in enumerate(img_ids)}
    res = {}
    for t in args.targets:
        base = {(g[0], g[1]): g for g in out["results"][f"{args.base}/{t}"]["_per_gt"]}
        res[t] = {}
        for m in args.methods:
            if m == args.base:
                continue
            cur = {(g[0], g[1]): g for g in out["results"][f"{m}/{t}"]["_per_gt"]}
            keys = sorted(base)
            img = np.array([pos[k[0]] for k in keys]); area = np.array([base[k][2] for k in keys])
            d = np.array([cur[k][3] - base[k][3] for k in keys])
            cls = {n: (area >= lo) & (area < hi) for n, lo, hi in SIZES}

            def stat(w):
                v = {n: float((d * w * c).sum() / max((w * c).sum(), 1e-9)) for n, c in cls.items()}
                v["large_minus_small"] = v["large"] - v["small"]
                return v
            pt = stat(np.ones(len(d)))
            bs = []
            for r in idx_imgs:
                cnt = np.bincount(r, minlength=len(img_ids))
                bs.append(stat(cnt[img].astype(float)))
            res[t][m] = {n: {"point": pt[n], "ci95": [float(x) for x in np.percentile([b[n] for b in bs], [2.5, 97.5])]}
                         for n in pt}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="n300_B50_eps5")
    p.add_argument("--methods", nargs="+", default=["MI-FGSM", "M-DI2-FGSM", "OSFD"])
    p.add_argument("--targets", nargs="+", default=["dino_swin_l", "swin_t", "convnext_t", "r101"])
    p.add_argument("--subset", default=None, help="vd dev30 trong run dev300 (E5)")
    p.add_argument("--dets", default="dets.json", help="dets Controlled Panel của run")
    p.add_argument("--tag", default="", help="hậu tố file kết quả, vd _elem_dev30")
    p.add_argument("--base", default=None, help="E5: ô đối chứng (vd NOISE) -> Δ suppression theo kích thước, bootstrap ảnh")
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.run.split('_')[0]}.csv"))
    if args.subset:
        keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.subset}.csv")))
        img_ids = [i for i in img_ids if i in keep]
    art = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    ctrl = json.load(open(os.path.join(art, args.dets)))

    def dets_of(t, c):
        g = os.path.join(art, "gen_dets", f"{t}.json")
        return json.load(open(g))[c] if t not in ctrl["clean"] else ctrl[c][t]

    gt = gt_table(coco, img_ids)
    r50_clean = matched(coco, dets_of("r50", "clean"), img_ids)
    s_r50 = {a: s for i in img_ids for a, s in zip(*r50_clean[i])}
    rng = np.random.default_rng(BOOT_SEED)
    idx_imgs = rng.integers(0, len(img_ids), size=(N_BOOT, len(img_ids)))
    pos = {i: k for k, i in enumerate(img_ids)}
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    out = {"n_images": len(img_ids), "rule": "progress_log 2026-10-06 E4", "results": {}}

    for t in args.targets:
        clean = matched(coco, dets_of(t, "clean"), img_ids)
        for m in args.methods:
            adv = matched(coco, dets_of(t, m), img_ids)
            recs = []
            for i in img_ids:
                for a, sc, sa in zip(clean[i][0], clean[i][1], adv[i][1]):
                    if sc >= DETECT_SCORE:
                        recs.append(dict(gt[a], ann=a, s_clean=sc, s_clean_r50=s_r50[a], retained=sa / sc,
                                         suppressed=float(sa < DETECT_SCORE)))
            groups = defaultdict(list)
            for k, r in enumerate(recs):
                groups[pos[r["img_id"]]].append(k)
            groups = [np.array(groups.get(k, []), dtype=int) for k in range(len(img_ids))]
            y = np.array([r["retained"] for r in recs])
            res = {"n_gt": len(recs), "mean_retained": float(y.mean()), "suppression": float(np.mean([r["suppressed"] for r in recs])),
                   "spearman_retained": {f: boot_rho(np.array([r[f] for r in recs]), y, None, idx_imgs, groups) for f in FEATS}}
            # bin: COCO size, tứ phân vị s_clean
            area = np.array([r["area"] for r in recs]); s = np.array([r["s_clean"] for r in recs])
            sup = np.array([r["suppressed"] for r in recs])
            res["by_size"] = {n: {"n": int(k.sum()), "retained": float(y[k].mean()) if k.any() else None,
                                  "suppression": float(sup[k].mean()) if k.any() else None}
                              for n, k in (("small", area < 32 ** 2), ("medium", (area >= 32 ** 2) & (area < 96 ** 2)),
                                           ("large", area >= 96 ** 2))}
            qs = np.quantile(s, [0.25, 0.5, 0.75])
            b = np.digitize(s, qs)
            res["by_s_clean_quartile"] = {f"Q{q + 1}": {"s_range": [float(s[b == q].min()), float(s[b == q].max())],
                                                        "retained": float(y[b == q].mean()), "suppression": float(sup[b == q].mean())}
                                          for q in range(4)}
            cats = defaultdict(list)
            for r in recs:
                cats[cat_name[r["cat"]]].append(r["retained"])
            res["by_category_top"] = {c: {"n": len(v), "retained": float(np.mean(v))}
                                      for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))[:12]}
            res["_per_gt"] = [(r["img_id"], r["ann"], r["area"], r["suppressed"], r["retained"]) for r in recs]
            out["results"][f"{m}/{t}"] = res

    if args.base:
        out["delta_vs_base"] = delta_by_size(out, args, img_ids, idx_imgs)
    os.makedirs(os.path.join(REPO_ROOT, "results/mechanism"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, f"results/mechanism/e4_target_side{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for key, r in out["results"].items():
        sp = r["spearman_retained"]
        print(f"{key:<26} nGT={r['n_gt']:<5} ret={r['mean_retained']:.2f} sup={r['suppression']:.2f}  " + "  ".join(
            f"{f}:{sp[f]['rho']:+.2f}[{sp[f]['ci95'][0]:+.2f},{sp[f]['ci95'][1]:+.2f}]" for f in FEATS))
    for t, rows in out.get("delta_vs_base", {}).items():
        print(f"\n=== Δ suppression vs {args.base} theo kích thước — {t} ===")
        for m, v in rows.items():
            print(f"{m:<10}" + "  ".join(f"{n}: {x['point']:+.2f}[{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]" for n, x in v.items()))
    for key in [k for k in out["results"] if k.startswith("OSFD")]:
        r = out["results"][key]
        print(f"\n{key}  size: " + "  ".join(f"{n} n={v['n']} ret={v['retained']:.2f} sup={v['suppression']:.2f}"
                                          for n, v in r["by_size"].items()))
        print(f"{'':<14}s_clean Q: " + "  ".join(f"{q} [{v['s_range'][0]:.2f},{v['s_range'][1]:.2f}] ret={v['retained']:.2f} sup={v['suppression']:.2f}"
                                                 for q, v in r["by_s_clean_quartile"].items()))


if __name__ == "__main__":
    main()
