"""R2 (progress_log 2026-11-17 01:40, KHÓA): thu nhỏ δ (×c) kiểm ±ε có tối ưu. Forward-only."""
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
TARGETS = ["r101", "swin_t", "dino_swin_l"]; SC = [0.25, 0.5, 0.75, 1.0]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector
    coco = COCO(ANN); cat_ids = sorted(coco.getCatIds())
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    clean, delta = {}, {}
    for i in ids:
        c = mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32); clean[i] = c
        delta[i] = {a: mmcv.imread(os.path.join(ADV, ATT[a][0], ATT[a][1], f"{i}.png")).astype(np.int32) - c for a in ATT}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    res = {}
    for key in TARGETS:
        model = load(key)
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat_ids)).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            res[key] = {}
            for a in ATT:
                row = []
                for c in SC:
                    k = 0
                    for i in ids:
                        if not tp[i]: continue
                        img = np.clip(clean[i] + np.round(c * delta[i][a]).astype(np.int32), 0, 255).astype(np.uint8)
                        g = group(to_coco_dets(inference_detector(model, img), i, cat_ids)).get(i, [])
                        k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                    row.append(round(100 * k / max(ntp, 1)))
                res[key][a] = row
                print(f"{key:12} {a:10} kill% theo c{SC}: {row}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump({"scales": SC, "kill": res}, open(os.path.join(REPO_ROOT, "results/mechanism/r2_scale_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
