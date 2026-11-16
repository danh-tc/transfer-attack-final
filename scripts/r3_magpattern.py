"""R3 (progress_log 2026-11-17 02:10, KHÓA): mẫu độ lớn δ ở cùng ngân sách — uniform vs dropout vs randmag. Forward-only."""
import json, os, sys
import mmcv, numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.r1_region import ADV, ATT, IMGDIR
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
TARGETS = ["r101", "dino_swin_l"]; ATKS = ["RD-CNX@50", "OSFD-PUB"]; C = 0.5

def variants(d, rng):
    full = d.copy()
    uni = np.round(C * d).astype(np.int32)
    drop = np.where(rng.random(d.shape) < C, d, 0)
    randmag = np.round(d * (2 * C) * rng.random(d.shape)).astype(np.int32)
    return {"full": full, "uniform": uni, "dropout": drop, "randmag": randmag}

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector
    coco = COCO(ANN); cat_ids = sorted(coco.getCatIds())
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    delta = {i: {a: mmcv.imread(os.path.join(ADV, ATT[a][0], ATT[a][1], f"{i}.png")).astype(np.int32) - clean[i] for a in ATKS} for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    res = {}
    for key in TARGETS:
        model = load(key); res[key] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat_ids)).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for a in ATKS:
                rng = np.random.default_rng(42); cells = {}
                for i in ids:
                    for v, dd in variants(delta[i][a], rng).items():
                        img = np.clip(clean[i] + dd, 0, 255).astype(np.uint8)
                        g = group(to_coco_dets(inference_detector(model, img), i, cat_ids)).get(i, [])
                        cells.setdefault(v, 0)
                        if tp[i]:
                            cells[v] += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                row = {v: round(100 * cells[v] / max(ntp, 1)) for v in cells}
                res[key][a] = row
                print(f"{key:12} {a:10} full={row['full']} | c0.5: uniform={row['uniform']} dropout={row['dropout']} randmag={row['randmag']}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump({"c": C, "kill": res}, open(os.path.join(REPO_ROOT, "results/mechanism/r3_magpattern_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
