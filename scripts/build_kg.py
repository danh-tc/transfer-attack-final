#!/usr/bin/env python3
"""Dựng Knowledge Graph COCO-80 từ quan hệ Visual Genome cho pilot KG
(quy tắc LOCKED: docs/progress_log.md 2026-10-03).

- Nguồn: VG v1.4 `relationships.json` + `image_data.json` trong data/vg/ (gitignored;
  tải từ https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/).
- Chống rò rỉ: bỏ mọi ảnh VG có coco_id ∈ dev300 ∪ n300 ∪ n1000.
- Map tên object VG -> COCO-80: KHỚP TÊN CHÍNH XÁC sau chuẩn hóa (lowercase, gộp khoảng
  trắng, bỏ dấu câu cuối) với NAME_TABLE viết tay (tên COCO + đồng nghĩa + số nhiều).
  KHÔNG dùng synset của VG: kiểm trước khi viết bảng cho thấy synset nhiễu nặng
  (dog.n.01 gồm "hot dog" 3.5k lần, child.n.01 gồm "fries"/"french fries" ~1.5k,
  accountant.n.01 = "controller"), ~17% object không có synset. Tên không map -> bỏ.
- C_ij = số relationship (subject ∈ i, object ∈ j; bỏ quan hệ của 1 object với chính nó);
  C̃ = C + Cᵀ (gồm i = j).
- S = PPMI, tổng biên trên toàn bộ C̃; sau đó C̃_ij < MIN_SUPPORT -> S_ij = 0.
Chỉ chạy 1 lần — có output thì bỏ qua (KG khóa trước mọi lần chạy attack).

Chạy: python scripts/build_kg.py
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VG_DIR = REPO_ROOT / "data/vg"
LIST_DIR = REPO_ROOT / "data/image_lists"
OUT_DIR = REPO_ROOT / "data/kg"
OUT_MATRIX = OUT_DIR / "coco80_vg_relation_matrix.npy"
OUT_META = OUT_DIR / "coco80_vg_relation_metadata.json"

MIN_SUPPORT = 20
EXCLUDE_LISTS = ("dev300", "n300", "n1000")

# Thứ tự = mmdet CocoDataset.METAINFO['classes'] (= nhãn gt_instances.labels).
COCO80 = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush')

# Tên VG (đã chuẩn hóa) -> class COCO. Viết tay, khóa trước khi tính S. Tên mơ hồ bị loại
# có chủ đích: bag, glass, glove, plant, monitor, desk, van, calf, pitcher, female/male, crowd.
NAME_TABLE = {
    'person': ['person', 'persons', 'man', 'men', 'woman', 'women', 'people', 'boy', 'boys',
               'girl', 'girls', 'child', 'children', 'kid', 'kids', 'lady', 'ladies', 'guy',
               'guys', 'baby', 'babies', 'player', 'players', 'skier', 'skiers', 'surfer',
               'surfers', 'skateboarder', 'skateboarders', 'snowboarder', 'snowboarders',
               'batter', 'catcher', 'umpire', 'tennis player', 'baseball player',
               'baseball players', 'rider', 'riders', 'spectator', 'spectators', 'passenger',
               'passengers', 'adult', 'adults'],
    'bicycle': ['bicycle', 'bicycles', 'bike', 'bikes'],
    'car': ['car', 'cars', 'automobile', 'suv'],
    'motorcycle': ['motorcycle', 'motorcycles', 'motorbike', 'motorbikes'],
    'airplane': ['airplane', 'airplanes', 'plane', 'planes', 'aeroplane', 'jet', 'jets',
                 'airliner'],
    'bus': ['bus', 'buses', 'busses'],
    'train': ['train', 'trains', 'locomotive'],
    'truck': ['truck', 'trucks', 'pickup truck'],
    'boat': ['boat', 'boats', 'ship', 'ships', 'sailboat', 'sail boat', 'canoe', 'kayak'],
    'traffic light': ['traffic light', 'traffic lights', 'stoplight', 'stop light',
                      'traffic signal'],
    'fire hydrant': ['fire hydrant', 'hydrant'],
    'stop sign': ['stop sign'],
    'parking meter': ['parking meter', 'parking meters'],
    'bench': ['bench', 'benches'],
    'bird': ['bird', 'birds', 'duck', 'ducks', 'seagull', 'pigeon'],
    'cat': ['cat', 'cats', 'kitten', 'kitty'],
    'dog': ['dog', 'dogs', 'puppy'],
    'horse': ['horse', 'horses', 'pony'],
    'sheep': ['sheep', 'lamb', 'lambs'],
    'cow': ['cow', 'cows', 'cattle', 'bull'],
    'elephant': ['elephant', 'elephants'],
    'bear': ['bear', 'bears', 'polar bear'],
    'zebra': ['zebra', 'zebras'],
    'giraffe': ['giraffe', 'giraffes'],
    'backpack': ['backpack', 'backpacks'],
    'umbrella': ['umbrella', 'umbrellas'],
    'handbag': ['handbag', 'handbags', 'purse', 'purses'],
    'tie': ['tie', 'ties', 'necktie'],
    'suitcase': ['suitcase', 'suitcases', 'luggage'],
    'frisbee': ['frisbee', 'frisbees'],
    'skis': ['skis', 'ski'],
    'snowboard': ['snowboard', 'snow board'],
    'sports ball': ['ball', 'balls', 'tennis ball', 'baseball', 'soccer ball', 'football',
                    'volleyball', 'basketball'],
    'kite': ['kite', 'kites'],
    'baseball bat': ['baseball bat', 'bat'],
    'baseball glove': ['baseball glove', 'baseball mitt', 'mitt', "catcher's mitt"],
    'skateboard': ['skateboard', 'skateboards'],
    'surfboard': ['surfboard', 'surfboards', 'surf board'],
    'tennis racket': ['tennis racket', 'tennis racquet', 'racket', 'racquet'],
    'bottle': ['bottle', 'bottles'],
    'wine glass': ['wine glass', 'wineglass', 'wine glasses'],
    'cup': ['cup', 'cups', 'mug'],
    'fork': ['fork', 'forks'],
    'knife': ['knife', 'knives'],
    'spoon': ['spoon', 'spoons'],
    'bowl': ['bowl', 'bowls'],
    'banana': ['banana', 'bananas'],
    'apple': ['apple', 'apples'],
    'sandwich': ['sandwich', 'sandwiches'],
    'orange': ['orange', 'oranges'],
    'broccoli': ['broccoli'],
    'carrot': ['carrot', 'carrots'],
    'hot dog': ['hot dog', 'hot dogs', 'hotdog', 'hotdogs'],
    'pizza': ['pizza', 'pizzas'],
    'donut': ['donut', 'donuts', 'doughnut', 'doughnuts'],
    'cake': ['cake', 'cakes'],
    'chair': ['chair', 'chairs'],
    'couch': ['couch', 'couches', 'sofa'],
    'potted plant': ['potted plant', 'houseplant', 'pot plant'],
    'bed': ['bed', 'beds'],
    'dining table': ['dining table', 'table'],
    'toilet': ['toilet'],
    'tv': ['tv', 'television', 'tv set', 'television set'],
    'laptop': ['laptop', 'laptops'],
    'mouse': ['mouse', 'computer mouse'],
    'remote': ['remote', 'remotes', 'remote control'],
    'keyboard': ['keyboard'],
    'cell phone': ['cell phone', 'cellphone', 'phone', 'mobile phone', 'smartphone',
                   'smart phone'],
    'microwave': ['microwave', 'microwave oven'],
    'oven': ['oven'],
    'toaster': ['toaster'],
    'sink': ['sink'],
    'refrigerator': ['refrigerator', 'fridge'],
    'book': ['book', 'books'],
    'clock': ['clock', 'clocks'],
    'vase': ['vase'],
    'scissors': ['scissors'],
    'teddy bear': ['teddy bear', 'teddy bears', 'teddy'],
    'hair drier': ['hair drier', 'hair dryer', 'blow dryer'],
    'toothbrush': ['toothbrush', 'toothbrushes'],
}


def normalize(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().lower())
    return name.rstrip(" .,;:!\"'")


def obj_name(o) -> str:
    if "name" in o:
        return o["name"]
    return o["names"][0] if o.get("names") else ""


def load_ids(name):
    with open(LIST_DIR / f"{name}.csv", newline="") as f:
        return {int(r["image_id"]) for r in csv.DictReader(f)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT_MATRIX.exists() or OUT_META.exists():
        print(f"{OUT_DIR} đã có output — KG đã khóa, bỏ qua.")
        return

    assert set(NAME_TABLE) == set(COCO80)
    name2cls = {}
    for cls, names in NAME_TABLE.items():
        for n in names:
            assert normalize(n) == n and n not in name2cls, n
            name2cls[n] = COCO80.index(cls)

    excluded_coco = set().union(*(load_ids(n) for n in EXCLUDE_LISTS))
    image_data = json.load(open(VG_DIR / "image_data.json"))
    excluded_vg = {im["image_id"] for im in image_data
                   if im.get("coco_id") is not None and im["coco_id"] in excluded_coco}
    relationships = json.load(open(VG_DIR / "relationships.json"))

    C = np.zeros((80, 80), dtype=np.int64)
    n_rel = n_rel_kept_images = n_mapped = n_self = 0
    for entry in relationships:
        rels = entry["relationships"]
        n_rel += len(rels)
        if entry["image_id"] in excluded_vg:
            continue
        n_rel_kept_images += len(rels)
        for r in rels:
            s, o = r["subject"], r["object"]
            ci = name2cls.get(normalize(obj_name(s)))
            cj = name2cls.get(normalize(obj_name(o)))
            if ci is None or cj is None:
                continue
            if s.get("object_id") is not None and s.get("object_id") == o.get("object_id"):
                n_self += 1
                continue
            C[ci, cj] += 1
            n_mapped += 1

    Ct = C + C.T
    total = Ct.sum()
    row = Ct.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(Ct * total / np.outer(row, row))
    S = np.where(Ct > 0, np.maximum(pmi, 0.0), 0.0)
    below = (Ct > 0) & (Ct < MIN_SUPPORT)
    S[below] = 0.0
    assert np.allclose(S, S.T) and np.isfinite(S).all()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_MATRIX, S.astype(np.float64))
    upper = np.triu_indices(80)
    meta = {
        "source": "Visual Genome v1.4 (relationships.json, image_data.json)",
        "source_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/",
        "source_sha256": {f: sha256(VG_DIR / f) for f in
                          ("relationships.json.zip", "image_data.json.zip")
                          if (VG_DIR / f).exists()},
        "classes": list(COCO80),
        "name_table": NAME_TABLE,
        "name_matching": "exact match after normalize() (lowercase, collapse whitespace, "
                         "strip trailing punctuation); VG synsets NOT used (noisy)",
        "normalization": "PPMI on symmetrized counts C~ = C + C^T (marginals over full C~); "
                         f"then S_ij = 0 where C~_ij < {MIN_SUPPORT}",
        "min_support": MIN_SUPPORT,
        "leakage_excluded_lists": list(EXCLUDE_LISTS),
        "n_coco_ids_excluded": len(excluded_coco),
        "n_vg_images_total": len(image_data),
        "n_vg_images_excluded": len(excluded_vg),
        "n_relationships_total": n_rel,
        "n_relationships_after_exclusion": n_rel_kept_images,
        "n_relationships_mapped": n_mapped,
        "n_self_relations_dropped": n_self,
        "n_cells_upper_triangle": len(upper[0]),
        "n_cells_count_positive": int((Ct[upper] > 0).sum()),
        "n_cells_removed_by_min_support": int(below[upper].sum()),
        "n_cells_S_positive": int((S[upper] > 0).sum()),
        "frac_cells_S_positive": float((S[upper] > 0).mean()),
    }
    json.dump(meta, open(OUT_META, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("name_table", "classes")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
