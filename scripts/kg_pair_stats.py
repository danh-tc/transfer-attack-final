#!/usr/bin/env python3
"""Thống kê cặp object cho pilot KG TRƯỚC khi chạy attack (docs/progress_log.md 2026-10-03,
mục "Cặp hợp lệ, ảnh có thông tin"). Chỉ dùng annotation COCO + ma trận KG, không đo transfer.

Box lấy từ annotation ở khung ảnh gốc (non-crowd, w,h >= 1). Tỉ lệ diện tích union đã nở
margin bất biến với Resize keep_ratio (scale đều 2 chiều), nên khớp với khung model.
Các đại lượng "kỳ vọng" tính giải tích từ phân bố chọn cặp đã khóa:
  q ~ p ∝ S (E2); E1 ~ đều trên M(q) = {r : |log(A_r/A_q)| <= log 2}.
- P(E1 ≠ E2) = 1 − Σ_q p(q)/|M(q)|  (tương ứng sanity S2, ngưỡng 0.5)
- E|log(A_E1/A_E2)|                 (tương ứng S3, ngưỡng log 1.5; ở đây là trung bình)
- TV(p_E2, p_E1): khoảng cách phân bố chọn cặp — KG có làm phân bố khác random khớp
  hình học không (mô tả, không thuộc quy tắc).

Chạy: python scripts/kg_pair_stats.py   -> results/kg_pilot/pair_stats.json
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MARGIN = 0.15
AREA_RATIO = 2.0
SETS = ("dev300", "n300")


def union_areas(boxes, img_w, img_h):
    n = len(boxes)
    i, j = np.triu_indices(n, 1)
    x1 = np.minimum(boxes[i, 0], boxes[j, 0]); y1 = np.minimum(boxes[i, 1], boxes[j, 1])
    x2 = np.maximum(boxes[i, 2], boxes[j, 2]); y2 = np.maximum(boxes[i, 3], boxes[j, 3])
    uw, uh = x2 - x1, y2 - y1
    w = np.clip(x2 + MARGIN * uw, 0, img_w) - np.clip(x1 - MARGIN * uw, 0, img_w)
    h = np.clip(y2 + MARGIN * uh, 0, img_h) - np.clip(y1 - MARGIN * uh, 0, img_h)
    return i, j, w * h


def main():
    S = np.load(REPO_ROOT / "data/kg/coco80_vg_relation_matrix.npy")
    classes = json.load(open(REPO_ROOT / "data/kg/coco80_vg_relation_metadata.json"))["classes"]
    coco = json.load(open(REPO_ROOT / "data/coco/annotations/instances_val2017.json"))
    cat2idx = {c["id"]: classes.index(c["name"]) for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    anns = collections.defaultdict(list)
    for a in coco["annotations"]:
        if not a["iscrowd"] and a["bbox"][2] >= 1 and a["bbox"][3] >= 1:
            anns[a["image_id"]].append(a)

    out = {"margin": MARGIN, "area_ratio": AREA_RATIO}
    for name in SETS:
        with open(REPO_ROOT / f"data/image_lists/{name}.csv", newline="") as f:
            ids = [int(r["image_id"]) for r in csv.DictReader(f)]
        cnt = collections.Counter()
        p_diff, log_ratio, tv, same_kg, same_unif = [], [], [], [], []
        for img_id in ids:
            A = anns[img_id]
            n_pairs = len(A) * (len(A) - 1) // 2
            if n_pairs < 2:
                cnt[f"{n_pairs}_pair"] += 1
                continue
            cnt["informative"] += 1
            boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in A)])
            cls = np.array([cat2idx[a["category_id"]] for a in A])
            i, j, area = union_areas(boxes, imgs[img_id]["width"], imgs[img_id]["height"])
            s = S[cls[i], cls[j]]
            if s.max() == 0:
                cnt["kg_degenerate"] += 1
                p = np.full(len(s), 1 / len(s))
            else:
                p = s / s.sum()
                if np.allclose(s, s[0]):
                    cnt["kg_uniform"] += 1
            la = np.log(area)
            dist = np.abs(la[:, None] - la[None, :])
            match = (dist <= np.log(AREA_RATIO)).astype(float)
            msize = match.sum(1)
            p_diff.append(1 - np.sum(p / msize))
            log_ratio.append(np.sum(p * (match * dist).sum(1) / msize))
            p1 = (p[:, None] * match / msize[:, None]).sum(0)
            tv.append(0.5 * np.abs(p - p1).sum())
            same = cls[i] == cls[j]
            same_kg.append(np.sum(p * same)); same_unif.append(same.mean())

        def summ(x):
            x = np.asarray(x)
            return {"mean": float(x.mean()), "median": float(np.median(x)),
                    "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}

        out[name] = {
            "n_images": len(ids),
            "n_0_pair": cnt["0_pair"], "n_1_pair": cnt["1_pair"],
            "n_informative": cnt["informative"],
            "n_kg_degenerate": cnt["kg_degenerate"],
            "n_kg_uniform_nonzero": cnt["kg_uniform"],
            "informative_expected_p_E1_ne_E2": summ(p_diff),
            "informative_expected_abs_log_area_ratio": summ(log_ratio),
            "informative_tv_E2_vs_E1": summ(tv),
            "informative_p_same_class_E2": float(np.mean(same_kg)),
            "informative_p_same_class_uniform": float(np.mean(same_unif)),
        }
    path = REPO_ROOT / "results/kg_pilot/pair_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
