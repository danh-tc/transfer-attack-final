"""G9 (progress_log 2026-11-17 07:10, KHÓA): control — R²(v_cls) target dễ (dino_r50) vs khó (dino_swin_l) trong cùng span."""
import json, os, sys
import numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.targets import load_any, target_dataset, target_grads
from scripts.r1_region import region_masks
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SUR = ["r50", "r101", "convnext_t", "swin_t"]; TGT = ["dino_r50", "dino_swin_l"]
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN); ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    ds0 = {s["img_id"]: s for s in target_dataset("r50", ids)}
    orig = {i: ds0[i]["orig_inputs"].float() for i in ids}
    inm = {i: torch.tensor(region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), *orig[i].shape[1:])["IN"], dtype=torch.bool) for i in ids}
    def clsgrad(key):
        m = load_any(key); ds = {s["img_id"]: s for s in target_dataset(key, ids)}
        out = {i: target_grads(m, key, orig[i].cuda(), ds[i]["data_sample"], objectives=("cls",))[0]["cls"][:, inm[i]].reshape(-1).cpu().numpy() for i in ids}
        del m; torch.cuda.empty_cache(); print(f"[g9] {key} xong", flush=True); return out
    G = {k: clsgrad(k) for k in SUR}
    res = {}
    for t in TGT:
        V = clsgrad(t)
        r2 = []
        for i in ids:
            A = np.stack([G[k][i] for k in SUR], 1); v = V[i]
            coef, *_ = np.linalg.lstsq(A, v, rcond=None); proj = A @ coef
            r2.append(float((proj @ proj) / (v @ v + 1e-12)))
        res[t] = {"R2_mean": float(np.mean(r2)), "R2_per_img": [round(x, 3) for x in r2]}
        print(f"R²(v_cls {t}) trong span 4 surrogate = {np.mean(r2):.3f}  (per-img {[round(x,2) for x in r2]})", flush=True)
    ratio = res["dino_r50"]["R2_mean"] / (res["dino_swin_l"]["R2_mean"] + 1e-9)
    res["ratio_easy_over_hard"] = ratio
    res["verdict"] = ("THƯỚC ĐO PHÂN BIỆT: span-deficiency Swin-L THẬT" if res["dino_r50"]["R2_mean"] >= 0.1 and ratio >= 5
                      else "THƯỚC ĐO VÔ NGHĨA (trực giao chiều cao): RÚT LẠI kết luận span")
    print(f"ratio dễ/khó = {ratio:.1f}x -> {res['verdict']}")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g9_control_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
