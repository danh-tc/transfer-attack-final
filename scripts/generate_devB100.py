#!/usr/bin/env python3
"""Chốt devB100 — PROTOCOL AMENDMENT 2026-11-13 (progress_log, E87; user duyệt phương án A).

Vì sao: dev300 đã dùng hết cho việc HÌNH THÀNH giả thuyết stage/C2 (dev100: E83/E85; x200: E84/E86). E87 kiểm
giả thuyết hậu kỳ (E86) trên ảnh chưa từng dùng. devB100 KHÔNG phải test final; n1000 không đổi; KHÔNG dùng devB100
để tune α sau E87.

Quy tắc (khóa + commit TRƯỚC khi sinh):
- Pool = ảnh val2017 có >= 1 instance annotation (cùng tiêu chí n1000 / dev300) trừ (n1000 ∪ dev300).
- devB100 = random.Random(20261113).sample(sorted(pool), 100); ghi CSV theo image_id tăng dần (như dev100/dev300).
Chỉ chạy 1 lần — có devB100_meta.json thì bỏ qua.

Chạy: python scripts/generate_devB100.py
"""
import csv
import json
import random
from pathlib import Path

SEED = 20261113
N = 100
REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_FILE = REPO_ROOT / "data/coco/annotations/instances_val2017.json"
OUT_DIR = REPO_ROOT / "data/image_lists"


def read_ids(name):
    with open(OUT_DIR / name, newline="") as f:
        return {int(r["image_id"]) for r in csv.DictReader(f)}


def main():
    if (OUT_DIR / "devB100_meta.json").exists():
        print(f"{OUT_DIR}/devB100_meta.json đã tồn tại — devB100 đã khóa, bỏ qua.")
        return
    with open(ANN_FILE) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    ids_with_ann = {ann["image_id"] for ann in coco["annotations"]} & set(images_by_id)
    n1000, dev300 = read_ids("n1000.csv"), read_ids("dev300.csv")
    assert len(n1000) == 1000 and len(dev300) == 300 and not (n1000 & dev300)
    pool = sorted(ids_with_ann - n1000 - dev300)
    dev = random.Random(SEED).sample(pool, N)
    assert not (set(dev) & (n1000 | dev300))
    with open(OUT_DIR / "devB100.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name"])
        for i in sorted(dev):
            w.writerow([i, images_by_id[i]])
    meta = {"seed": SEED, "n": N, "pool_size": len(pool),
            "pool": "val2017 có >= 1 annotation \\ (n1000 ∪ dev300)",
            "role": "protocol amendment 2026-11-13 (E87) — dev mới, KHÔNG phải test final; không tune α sau E87",
            "intersect_n1000": 0, "intersect_dev300": 0}
    json.dump(meta, open(OUT_DIR / "devB100_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"devB100: {N} ảnh từ pool {len(pool)} -> {OUT_DIR}/devB100.csv")


if __name__ == "__main__":
    main()
