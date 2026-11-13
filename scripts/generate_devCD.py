#!/usr/bin/env python3
"""Chốt devC300 + devD200 — PROTOCOL AMENDMENT 2026-11-13 (mục tiêu B, docs/plan_paper_lock.md AMENDMENT B.1).

devC300: phát triển method mới (tối đa 3 candidate × 2 cấu hình). devD200: xác nhận ĐÚNG 1 lần cho 1 candidate cuối.
Quy tắc (khóa + commit TRƯỚC mọi thí nghiệm B):
- Pool = ảnh val2017 có >= 1 instance annotation \\ (n1000 ∪ dev300 ∪ devB100).
  Kiểm tường minh: n300 ⊂ n1000; dev5/10/30/100 ⊂ dev300; giao = ∅ với MỌI tập cũ.
- devC300 = random.Random(20261114).sample(sorted(pool), 300)
- devD200 = random.Random(20261115).sample(sorted(pool − devC300), 200)
- CSV theo image_id tăng dần. Chỉ chạy 1 lần — có devCD_meta.json thì bỏ qua.
Chạy: python scripts/generate_devCD.py
"""
import csv
import json
import random
from pathlib import Path

SEED_C, N_C, SEED_D, N_D = 20261114, 300, 20261115, 200
REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_FILE = REPO_ROOT / "data/coco/annotations/instances_val2017.json"
OUT_DIR = REPO_ROOT / "data/image_lists"
OLD = ["n300", "n1000", "dev5", "dev10", "dev30", "dev100", "dev300", "devB100"]


def read_ids(name):
    with open(OUT_DIR / f"{name}.csv", newline="") as f:
        return {int(r["image_id"]) for r in csv.DictReader(f)}


def write(name, ids, images_by_id):
    with open(OUT_DIR / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name"])
        for i in sorted(ids):
            w.writerow([i, images_by_id[i]])


def main():
    if (OUT_DIR / "devCD_meta.json").exists():
        print(f"{OUT_DIR}/devCD_meta.json đã tồn tại — devC/devD đã khóa, bỏ qua.")
        return
    with open(ANN_FILE) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    ids_with_ann = {ann["image_id"] for ann in coco["annotations"]} & set(images_by_id)
    old = {k: read_ids(k) for k in OLD}
    assert old["n300"] <= old["n1000"]
    assert all(old[k] <= old["dev300"] for k in ("dev5", "dev10", "dev30", "dev100"))
    assert not (old["dev300"] & old["n1000"]) and not (old["devB100"] & (old["dev300"] | old["n1000"]))
    used = set().union(*old.values())
    pool = sorted(ids_with_ann - used)
    dev_c = random.Random(SEED_C).sample(pool, N_C)
    dev_d = random.Random(SEED_D).sample(sorted(set(pool) - set(dev_c)), N_D)
    assert not (set(dev_c) & set(dev_d)) and not ((set(dev_c) | set(dev_d)) & used)
    write("devC300", dev_c, images_by_id)
    write("devD200", dev_d, images_by_id)
    meta = {"seed_devC": SEED_C, "n_devC": N_C, "seed_devD": SEED_D, "n_devD": N_D, "pool_size": len(pool),
            "pool": "val2017 có >= 1 annotation \\ (n1000 ∪ dev300 ∪ devB100)",
            "intersect_old": {k: 0 for k in OLD}, "intersect_devC_devD": 0,
            "role": "AMENDMENT 2026-11-13 mục tiêu B: devC300 phát triển (≤ 3 candidate × 2 cấu hình); "
                    "devD200 xác nhận ĐÚNG 1 lần cho 1 candidate cuối; n1000 không đổi"}
    json.dump(meta, open(OUT_DIR / "devCD_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"devC300 / devD200 từ pool {len(pool)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
