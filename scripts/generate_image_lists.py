#!/usr/bin/env python3
"""Chốt cố định danh sách ảnh n=300 (Confirmation) và n=1000 (Final) từ COCO val2017.

Theo idea.md §4: không được tune trên n=1000, cả 2 tập phải cố định (seed) để tái lập
chính xác giữa các lần thuê máy GPU khác nhau. Script này chỉ chạy MỘT LẦN — kết quả
(data/image_lists/n300.csv, n1000.csv, meta.json) được commit vào git (nhẹ, chỉ là
danh sách id) để không bao giờ đổi lại giữa các phiên sau.

Quy tắc chọn (khóa tại đây, đổi phải ghi lý do vào docs/progress_log.md):
- Nguồn: toàn bộ ảnh val2017 (5000 ảnh) có annotations/instances_val2017.json.
- Lọc: chỉ giữ ảnh có ít nhất 1 instance annotation (cần GT thật cho GT-assisted
  threat model, idea.md §3).
- n=300 là TẬP CON của n=1000 (lấy 300 phần tử đầu của danh sách n=1000 đã sample) —
  đảm bảo Confirmation Stage không dùng ảnh nằm ngoài Final Stage, tránh 2 nguồn nhiễu
  khác nhau giữa 2 phase.
- Sample bằng random.Random(SEED).sample() trên danh sách image_id đã sort tăng dần
  (thứ tự input cố định trước khi sample => kết quả tái lập được trên mọi máy/mọi lần
  chạy, không phụ thuộc thứ tự file trả về từ pycocotools).

Chạy: python scripts/generate_image_lists.py (cần data/coco/annotations/instances_val2017.json
đã tải qua scripts/download_dataset.sh).
"""
import csv
import json
import random
from pathlib import Path

SEED = 42
N_FINAL = 1000
N_CONFIRMATION = 300

REPO_ROOT = Path(__file__).resolve().parent.parent
ANN_FILE = REPO_ROOT / "data/coco/annotations/instances_val2017.json"
OUT_DIR = REPO_ROOT / "data/image_lists"


def main():
    if (OUT_DIR / "meta.json").exists():
        print(
            f"{OUT_DIR}/meta.json đã tồn tại — danh sách coi như đã khóa, bỏ qua "
            "(script tất định 100% với cùng seed/pool nên chạy lại cũng ra kết quả "
            "giống hệt, nhưng không tự ghi đè để tránh lẫn lộn nếu ai đó sửa script "
            "sau này; xóa data/image_lists/ thủ công nếu thật sự muốn tạo lại)."
        )
        return

    if not ANN_FILE.exists():
        raise SystemExit(
            f"Không thấy {ANN_FILE} — chạy scripts/download_dataset.sh trước."
        )

    with open(ANN_FILE) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img["file_name"] for img in coco["images"]}
    ids_with_ann = sorted({ann["image_id"] for ann in coco["annotations"]})
    ids_with_ann = [i for i in ids_with_ann if i in images_by_id]

    print(f"Tổng ảnh val2017: {len(images_by_id)}")
    print(f"Ảnh có >=1 instance annotation: {len(ids_with_ann)}")

    if len(ids_with_ann) < N_FINAL:
        raise SystemExit(
            f"Pool ({len(ids_with_ann)}) nhỏ hơn N_FINAL ({N_FINAL}) — không đủ ảnh."
        )

    rng = random.Random(SEED)
    n1000_ids = rng.sample(ids_with_ann, N_FINAL)  # sample từ pool đã sort => tái lập được
    n300_ids = n1000_ids[:N_CONFIRMATION]  # n=300 là tập con của n=1000

    assert set(n300_ids) <= set(n1000_ids)
    assert len(set(n1000_ids)) == N_FINAL
    assert len(set(n300_ids)) == N_CONFIRMATION

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_csv(path, ids):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "file_name"])
            for i in sorted(ids):
                w.writerow([i, images_by_id[i]])

    write_csv(OUT_DIR / "n300.csv", n300_ids)
    write_csv(OUT_DIR / "n1000.csv", n1000_ids)

    meta = {
        "seed": SEED,
        "n_confirmation": N_CONFIRMATION,
        "n_final": N_FINAL,
        "source_pool": "coco val2017, images with >=1 instance annotation",
        "source_pool_size": len(ids_with_ann),
        "nesting": "n300 is a subset of n1000 (first 300 of the seeded sample of n1000)",
        "annotation_file": "data/coco/annotations/instances_val2017.json",
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Đã ghi {OUT_DIR}/n300.csv ({N_CONFIRMATION} ảnh), n1000.csv ({N_FINAL} ảnh), meta.json")


if __name__ == "__main__":
    main()
