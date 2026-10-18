#!/usr/bin/env python3
"""Vẽ detection sạch vs adv trên từng model (mô tả, không GPU, dùng dets có sẵn).

Mỗi ảnh 1 hình: hàng = model, cột = sạch | mỗi method. Box score >= 0.3. Box adv so với box SẠCH của CHÍNH
model đó (không so GT) để thấy attack thêm / bớt / làm lệch gì:
- xanh lá  = giữ   (cùng lớp, IoU >= 0.5 với 1 box sạch)
- cam      = lệch  (cùng lớp, 0.1 <= IoU < 0.5)
- tím      = đổi lớp (khác lớp, IoU >= 0.5)
- đỏ       = THÊM  (box mới, không khớp box sạch nào theo các luật trên)
- vàng nét đứt = MẤT (box sạch không còn box adv nào giữ)
Cột sạch: box sạch (xanh dương) + GT (trắng nét đứt).

Chạy: python scripts/viz_dets.py --run dev300_B50_eps5_limit5 --dets dets_oct_dev5.json --methods OSFD OCT
      -> results/figs/<tag>/<img_id>.png + summary.json
Method dạng TÊN@RUN (progress_log 2026-10-18): lấy PNG + dets của TÊN từ run khác (cùng tên --dets), sạch từ --run.
--draw N: chỉ vẽ N ảnh đầu, thống kê vẫn trên mọi ảnh. Thống kê thêm score_ratio = TB score_adv / score_sạch của box "giữ".
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from scripts.probe_error_types import iou_mat  # noqa: E402
from attack.data import load_image_ids  # noqa: E402

ANN_FILE = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMG_DIR = os.path.join(REPO_ROOT, "data/coco/val2017")
MODELS = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
LABEL = {"r50": "R50 (surrogate)", "r101": "R101", "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T",
         "yolox_s": "YOLOX-S", "dino_swin_l": "DINO-Swin-L"}
SCORE = 0.3
COLORS = {"keep": "lime", "loc": "orange", "cls": "magenta", "add": "red", "lost": "yellow"}


def by_image(dets):
    out = defaultdict(list)
    for d in dets:
        if d["score"] >= SCORE:
            out[d["image_id"]].append(d)
    return out


def compare(clean, adv):
    """Gán nhãn box adv theo box sạch; trả (nhãn mỗi box adv, chỉ số box sạch bị mất)."""
    c = np.array([d["bbox"] for d in clean]).reshape(-1, 4)
    a = np.array([d["bbox"] for d in adv]).reshape(-1, 4)
    cc = np.array([d["category_id"] for d in clean])
    ac = np.array([d["category_id"] for d in adv])
    M = iou_mat(a, c)
    order = np.argsort([-d["score"] for d in adv])
    used, tags, pairs = set(), [None] * len(adv), []
    for i in order:  # khớp tham lam theo score, 1 box sạch giữ tối đa 1 box adv
        same = [(M[i, j], j) for j in range(len(clean)) if cc[j] == ac[i] and j not in used and M[i, j] >= 0.5]
        if same:
            j = max(same)[1]
            used.add(j); tags[i] = "keep"; pairs.append((i, j))
    for i in range(len(adv)):
        if tags[i]:
            continue
        if len(clean) and any(cc[j] == ac[i] and 0.1 <= M[i, j] < 0.5 for j in range(len(clean))):
            tags[i] = "loc"
        elif len(clean) and any(cc[j] != ac[i] and M[i, j] >= 0.5 for j in range(len(clean))):
            tags[i] = "cls"
        else:
            tags[i] = "add"
    lost = [j for j in range(len(clean)) if j not in used]
    compare.pairs = pairs
    return tags, lost


def draw(ax, img, boxes, color, dashed=False, lw=1.5):
    ax.imshow(img)
    for b in boxes:
        x, y, w, h = b
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=lw,
                                       linestyle="--" if dashed else "-"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dets", required=True, help="dets Controlled Panel (artifacts/runs/<run>/)")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--draw", type=int, default=None, help="chỉ vẽ N ảnh đầu (thống kê vẫn trên mọi ảnh)")
    args = p.parse_args()
    art = os.path.join(REPO_ROOT, "artifacts/runs", args.run)
    tag = args.tag or f"viz_{args.run}"
    out_dir = os.path.join(REPO_ROOT, "results/figs", tag)
    os.makedirs(out_dir, exist_ok=True)

    run_of = lambda c: c.split("@")[1] if "@" in c else args.run
    name_of = lambda c: c.split("@")[0]
    art_of = lambda c: os.path.join(REPO_ROOT, "artifacts/runs", run_of(c))
    dets = {}  # dets[cond][model] -> {img_id: [det]}
    for cond in ["clean"] + args.methods:
        a = art_of(cond)
        ctrl = json.load(open(os.path.join(a, args.dets)))
        dets[cond] = {m: by_image(ctrl[name_of(cond)][m]) for m in ctrl[name_of(cond)]}
        for m in ("yolox_s", "dino_swin_l"):
            g = json.load(open(os.path.join(a, "gen_dets", f"{m}.json")))
            dets[cond][m] = by_image(g[name_of(cond)])
    models = [m for m in MODELS if m in dets["clean"]]
    sets = [{int(f[:-4]) for f in os.listdir(os.path.join(art_of(c), "adv", name_of(c)))} for c in args.methods]
    common = set.intersection(*sets)
    split_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists", f"{args.run.split('_')[0]}.csv"))
    img_ids = [i for i in split_ids if i in common]  # thứ tự split (vd 10 ảnh đầu = dev10)

    coco = COCO(ANN_FILE)
    cat = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    summary = {}
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        clean_img = mmcv.imread(os.path.join(IMG_DIR, info["file_name"]), channel_order="rgb")
        draw_it = args.draw is None or img_ids.index(img_id) < args.draw
        adv_imgs = {c: mmcv.imread(os.path.join(art_of(c), "adv", name_of(c), f"{img_id}.png"), channel_order="rgb")
                    for c in args.methods} if draw_it else {}
        gt = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]
        ncol = 1 + len(args.methods)
        h, w = clean_img.shape[:2]
        if draw_it:
            fig, axes = plt.subplots(len(models), ncol, figsize=(4.2 * ncol, 4.2 * len(models) * h / w + 0.4))
        summary[img_id] = {}
        for r, m in enumerate(models):
            clean = dets["clean"][m].get(img_id, [])
            if draw_it:
                ax = axes[r, 0]
                draw(ax, clean_img, gt, "white", dashed=True, lw=1.0)
                draw(ax, clean_img, [d["bbox"] for d in clean], "deepskyblue")
                ax.set_title(f"{LABEL[m]} | sạch: {len(clean)} box (GT {len(gt)})", fontsize=9)
            summary[img_id][m] = {"clean": len(clean)}
            for k, cond in enumerate(args.methods):
                adv = dets[cond][m].get(img_id, [])
                tags, lost = compare(clean, adv)
                cnt = {t: tags.count(t) for t in ("keep", "loc", "cls", "add")}
                cnt["lost"] = len(lost)
                cnt["score_ratio_sum"] = float(sum(adv[i]["score"] / clean[j]["score"] for i, j in compare.pairs))
                summary[img_id][m][cond] = cnt
                if not draw_it:
                    continue
                ax = axes[r, k + 1]
                ax.imshow(adv_imgs[cond])
                for d, t in zip(adv, tags):
                    x, y, bw, bh = d["bbox"]
                    ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, edgecolor=COLORS[t], lw=1.5))
                    if t in ("add", "cls"):
                        ax.text(x, y, cat[d["category_id"]], color=COLORS[t], fontsize=6, va="bottom")
                for j in lost:
                    x, y, bw, bh = clean[j]["bbox"]
                    ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, edgecolor=COLORS["lost"],
                                                   lw=1.5, linestyle="--"))
                ax.set_title(f"{name_of(cond)}: giữ {cnt['keep']} | mất {cnt['lost']} | thêm {cnt['add']} | "
                             f"lệch {cnt['loc']} | đổi lớp {cnt['cls']}", fontsize=8)
        if not draw_it:
            continue
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"ảnh {img_id} — xanh lá giữ · vàng đứt mất · đỏ thêm · cam lệch · tím đổi lớp (score ≥ {SCORE})",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{img_id}.png"), dpi=110)
        plt.close(fig)
        print(f"[viz] {img_id}", flush=True)

    # Tổng 5 ảnh theo model × method
    tot = {m: {c: defaultdict(int) for c in ["clean"] + args.methods} for m in models}
    for img_id, s in summary.items():
        for m in models:
            tot[m]["clean"]["n"] += s[m]["clean"]
            for c in args.methods:
                for t, v in s[m][c].items():
                    tot[m][c][t] += v
    print(f"\n{len(summary)} ảnh. Mỗi method: giữ / mất / thêm / lệch / đổi lớp | score_adv/score_sạch TB của box giữ")
    for m in models:
        row = "  ".join(f"{name_of(c)} {tot[m][c]['keep']}/{tot[m][c]['lost']}/{tot[m][c]['add']}/{tot[m][c]['loc']}/{tot[m][c]['cls']}"
                        f" | {tot[m][c]['score_ratio_sum'] / max(1, tot[m][c]['keep']):.2f}" for c in args.methods)
        print(f"{LABEL[m]:<14} sạch {tot[m]['clean']['n']:>5}  {row}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"per_image": summary, "total": tot}, f, indent=1)


if __name__ == "__main__":
    main()
