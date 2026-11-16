"""G2 (progress_log 2026-11-17 03:10, KHÓA): hiệu suất transfer của δ sớm vs muộn dọc quỹ đạo RD-CNX. Forward sau sinh."""
import json, os, sys
import mmcv, numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.preprocess import to_adv_image
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.e88_ens_baselines import C, seed_all
from scripts.run_baselines import METHODS
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs/g2_earlytraj/adv")
SNAPS = [5, 10, 25, 50]
MODELS = ["convnext_t", "r101", "fcos_r50", "dino_r50", "dino_swin_l"]
RES3 = ["r101", "fcos_r50", "dino_r50"]

def png(s, i): return os.path.join(ART, f"s{s}", f"{i}.png")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    # sinh snapshot
    if not all(os.path.exists(png(s, i)) for s in SNAPS for i in ids):
        cnx = load_model(C, "cuda:0"); ds = {s["img_id"]: s for s in target_dataset(C, ids)}
        fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            snaps = {}
            seed_all(1000 * (n + 1))
            d = fn(cnx, orig, dsm, steps=50, epsilon=5.0, on_step=lambda k, nz: snaps.__setitem__(k, nz.detach().clone()) if k in SNAPS else None)
            snaps[50] = d.detach().clone()
            for s in SNAPS:
                a = to_adv_image(orig, snaps[s]); os.makedirs(os.path.dirname(png(s, i)), exist_ok=True)
                mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png(s, i))
            print(f"[g2 gen] {n+1}/{len(ids)}", flush=True)
        del cnx; torch.cuda.empty_cache()
    coco = COCO(ANN); cat_ids = sorted(coco.getCatIds())
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    kill = {}
    for key in MODELS:
        model = load(key)
        with torch.no_grad():
            tp = {}
            for i in ids:
                g = group(to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values())
            for s in SNAPS:
                k = 0
                for i in ids:
                    if not tp[i]: continue
                    g = group(to_coco_dets(inference_detector(model, png(s, i)), i, cat_ids)).get(i, [])
                    k += sum(fate(a["bbox"], a["category_id"], g)[0] != "survived" for a in tp[i])
                kill[(key, s)] = k / max(ntp, 1)
        del model; torch.cuda.empty_cache()
        print(f"{key:12} kill% " + " ".join(f"s{s}={kill[(key,s)]*100:.0f}" for s in SNAPS), flush=True)
    res3 = {s: float(np.mean([kill[(m, s)] for m in RES3])) for s in SNAPS}
    print("\nHiệu suất transfer E = kill_target / kill_CNX(WB):")
    print(f"{'bước':6} {'WB_CNX':>8} {'SwinL':>8} {'E_SwinL':>9} {'RES3':>8} {'E_RES3':>9}")
    out = {"kill": {f"{m}|{s}": kill[(m, s)] for m in MODELS for s in SNAPS}, "E": {}}
    for s in SNAPS:
        wb = kill[("convnext_t", s)]
        esl = kill[("dino_swin_l", s)] / (wb + 1e-9); er3 = res3[s] / (wb + 1e-9)
        out["E"][s] = {"WB": wb, "SwinL": kill[("dino_swin_l", s)], "E_SwinL": esl, "RES3": res3[s], "E_RES3": er3}
        print(f"{s:6} {wb*100:8.0f} {kill[('dino_swin_l',s)]*100:8.0f} {esl:9.3f} {res3[s]*100:8.0f} {er3:9.3f}")
    json.dump(out, open(os.path.join(REPO_ROOT, "results/mechanism/g2_earlytraj_dev30.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
