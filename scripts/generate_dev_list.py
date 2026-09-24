#!/usr/bin/env python3
"""Chốt tập dev cho pilot method (docs/progress_log.md 2026-09-24): dev100 (pilot A′1), rồi
dev300 = dev100 ∪ 200 ảnh mới (pilot P2) — tập development DUY NHẤT cho RQ3 từ đây.

Tập dev là nơi DUY NHẤT được sàng lọc/chọn cấu hình method — n300/n1000 không được
dùng để tune (idea.md §4). Quy tắc (khóa trong progress_log trước khi sinh):
- Pool = ảnh val2017 có >=1 instance annotation (cùng tiêu chí n1000) trừ n1000.
- dev100 = random.Random(20260924).sample(sorted(pool), 100); Dev100 ∩ N1000 = ∅.
- dev300 = dev100 ∪ random.Random(20260925).sample(sorted(pool − dev100), 200).
Chỉ chạy 1 lần mỗi tập — kết quả commit vào git; có file meta thì bỏ qua.

Chạy: python scripts/generate_dev_list.py
"""
import csv
import json
import random
from pathlib import Path

SEED = 20260924
N_DEV = 100
SEED_EXT = 20260925
N_EXT = 200

REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_FILE = REPO_ROOT / "data/coco/annotations/instances_val2017.json"
OUT_DIR = REPO_ROOT / "data/image_lists"


def main():
    with open(ANN_FILE) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    ids_with_ann = {ann["image_id"] for ann in coco["annotations"]} & set(images_by_id)
    with open(OUT_DIR / "n1000.csv", newline="") as f:
        n1000 = {int(r["image_id"]) for r in csv.DictReader(f)}
    assert n1000 <= ids_with_ann

    pool = sorted(ids_with_ann - n1000)
    if (OUT_DIR / "dev_meta.json").exists():
        print(f"{OUT_DIR}/dev_meta.json đã tồn tại — dev100 đã khóa, bỏ qua.")
    else:
        write_dev100(pool, images_by_id, n1000)
    if (OUT_DIR / "dev300_meta.json").exists():
        print(f"{OUT_DIR}/dev300_meta.json đã tồn tại — dev300 đã khóa, bỏ qua.")
    else:
        write_dev300(pool, images_by_id, n1000)


def write_csv(path, ids, images_by_id):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name"])
        for i in sorted(ids):
            w.writerow([i, images_by_id[i]])


def write_dev300(pool, images_by_id, n1000):
    with open(OUT_DIR / "dev100.csv", newline="") as f:
        dev100 = {int(r["image_id"]) for r in csv.DictReader(f)}
    assert dev100 == set(random.Random(SEED).sample(pool, N_DEV))
    ext = random.Random(SEED_EXT).sample(sorted(set(pool) - dev100), N_EXT)
    dev300 = dev100 | set(ext)
    assert len(dev300) == N_DEV + N_EXT and not dev300 & n1000
    write_csv(OUT_DIR / "dev300.csv", dev300, images_by_id)
    meta = {"seed_dev100": SEED, "seed_extension": SEED_EXT, "n_dev": len(dev300),
            "definition": "dev100 ∪ Random(seed_extension).sample(sorted(pool − dev100), 200)",
            "source_pool": "coco val2017, images with >=1 instance annotation, minus n1000",
            "source_pool_size": len(pool), "disjoint_from": "n1000 (hence n300)",
            "purpose": "the ONLY development set for RQ3 — see docs/progress_log.md 2026-09-24"}
    with open(OUT_DIR / "dev300_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"đã ghi {OUT_DIR}/dev300.csv ({len(dev300)} ảnh), dev300_meta.json")


def write_dev100(pool, images_by_id, n1000):
    dev_ids = random.Random(SEED).sample(pool, N_DEV)
    assert len(set(dev_ids)) == N_DEV and not set(dev_ids) & n1000

    write_csv(OUT_DIR / "dev100.csv", dev_ids, images_by_id)

    meta = {
        "seed": SEED,
        "n_dev": N_DEV,
        "source_pool": "coco val2017, images with >=1 instance annotation, minus n1000",
        "source_pool_size": len(pool),
        "disjoint_from": "n1000 (hence n300)",
        "purpose": "pilot method screening only — see docs/progress_log.md 2026-09-24",
    }
    with open(OUT_DIR / "dev_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"pool {len(pool)} ảnh -> đã ghi {OUT_DIR}/dev100.csv ({N_DEV} ảnh), dev_meta.json")


if __name__ == "__main__":
    main()
