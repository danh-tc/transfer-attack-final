"""G4 (progress_log 2026-11-17 04:10, KHÓA): surrogate Swin-T, đồng thuận dấu vs WB DINO-Swin-L + transfer. Chẩn đoán."""
import json, os, sys
import mmcv, numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.preprocess import to_adv_image
from attack.models import load_model
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.e88_ens_baselines import seed_all
from scripts.run_baselines import METHODS
from scripts.r1_region import region_masks, ADV, ATT, IMGDIR
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
WB = os.path.join(REPO_ROOT, "artifacts/runs/r1_wb_swinl/adv/WB")
OUTDIR = os.path.join(REPO_ROOT, "artifacts/runs/g4_swint/adv/SWT"); os.makedirs(OUTDIR, exist_ok=True)
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    # sinh swin_t RD
    if not all(os.path.exists(os.path.join(OUTDIR, f"{i}.png")) for i in ids):
        m = load_model("swin_t", "cuda:0"); ds = {s["img_id"]: s for s in target_dataset("swin_t", ids)}
        fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
        for n, i in enumerate(ids):
            orig = ds[i]["orig_inputs"].cuda().float()
            seed_all(1000 * (n + 1))
            d = fn(m, orig, ds[i]["data_sample"], steps=50, epsilon=5.0)
            a = to_adv_image(orig, d)
            mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), os.path.join(OUTDIR, f"{i}.png"))
            print(f"[g4 gen] {n+1}/10", flush=True)
        del m; torch.cuda.empty_cache()
    # sign agreement in IN vs WB
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    M = {i: region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), *clean[i].shape[:2]) for i in ids}
    def sign_in(path_fn):
        return np.concatenate([np.sign(path_fn(i) - clean[i])[M[i]["IN"].astype(bool)] for i in ids])
    swb = sign_in(lambda i: mmcv.imread(os.path.join(WB, f"{i}.png")).astype(np.int32))
    srcs = {"SWT": lambda i: mmcv.imread(os.path.join(OUTDIR, f"{i}.png")).astype(np.int32)}
    for a in ATT:
        srcs[a] = (lambda a: lambda i: mmcv.imread(os.path.join(ADV, ATT[a][0], ATT[a][1], f"{i}.png")).astype(np.int32))(a)
    print("Đồng thuận dấu trong IN vs WB DINO-Swin-L:")
    agree = {}
    for k, pf in srcs.items():
        s = sign_in(pf); nz = (swb != 0) & (s != 0); agree[k] = float((swb[nz] == s[nz]).mean())
        print(f"  {k:10} {agree[k]:.3f}")
    # transfer kill + AP của SWT trên dino_swin_l + RES3
    res = {"sign_agree_vs_wb": agree, "transfer": {}}
    for key in ["dino_swin_l", "r101", "fcos_r50", "dino_r50"]:
        model = load(key)
        with torch.no_grad():
            cl = []; tp = {}
            for i in ids:
                det = inference_detector(model, clean[i].astype(np.uint8)); cl += to_coco_dets(det, i, cat)
                g = group(to_coco_dets(det, i, cat)).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values()); apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]
            ad = []; k = 0
            for i in ids:
                det = inference_detector(model, os.path.join(OUTDIR, f"{i}.png")); ad += to_coco_dets(det, i, cat)
                g = group(to_coco_dets(det, i, cat)).get(i, [])
                k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
            apa = CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]
            res["transfer"][key] = {"kill": 100 * k / max(ntp, 1), "drop": 100 * (apc - apa) / apc, "clean_ap": apc * 100}
            print(f"  SWT->{key:12} kill {res['transfer'][key]['kill']:.0f}%  AP {apc*100:.0f}->{apa*100:.0f} (drop {res['transfer'][key]['drop']:.0f}%)")
        del model; torch.cuda.empty_cache()
    v = "HỌ BACKBONE GIÚP tầng dấu" if agree["SWT"] > 0.52 else "Swin-T cũng ~0.5: rào do HEAD DINO, cộng Swin vô ích"
    res["verdict"] = v; print("G4 VERDICT:", v)
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g4_swint_dev10.json"), "w"), indent=1, ensure_ascii=False)
if __name__ == "__main__":
    main()
