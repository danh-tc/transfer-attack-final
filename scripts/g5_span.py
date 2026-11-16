"""G5 (progress_log 2026-11-17 04:55, KHÓA): v_t (WB DINO-Swin-L input-grad cls) có trong span gradient surrogate không.
Chiếu + R², tập surrogate lồng nhau, ảnh sạch. Load tuần tự."""
import json, os, sys
import numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.targets import load_any, target_dataset, target_grads
from scripts.r1_region import region_masks
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SUR = ["r50", "r101", "convnext_t", "swin_t", "dino_r50"]
SETS = {"r50": ["r50"], "CNN2": ["r50", "r101"], "+cnx": ["r50", "r101", "convnext_t"],
        "+swinT(4bb)": ["r50", "r101", "convnext_t", "swin_t"],
        "+dinoR50(head)": ["r50", "r101", "convnext_t", "swin_t", "dino_r50"]}

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN); ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    # orig + IN mask mỗi ảnh (dùng ds của r50 để lấy orig; orig = ảnh gốc, chung mọi model)
    ds0 = {s["img_id"]: s for s in target_dataset("r50", ids)}
    orig = {i: ds0[i]["orig_inputs"].float() for i in ids}
    inm = {}
    for i in ids:
        h, w = orig[i].shape[1:]
        inm[i] = torch.tensor(region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), h, w)["IN"], dtype=torch.bool)
    def grads(key):
        m = load_any(key); ds = {s["img_id"]: s for s in target_dataset(key, ids)}
        out = {}
        for i in ids:
            g = target_grads(m, key, orig[i].cuda(), ds[i]["data_sample"], objectives=("cls",))[0]["cls"].cpu()
            out[i] = g[:, inm[i]].reshape(-1).numpy()   # [3*n_in]
        del m; torch.cuda.empty_cache(); print(f"[g5] {key} xong", flush=True)
        return out
    G = {k: grads(k) for k in SUR}
    V = grads("dino_swin_l")
    # chiếu per ảnh
    res = {"sets": {}, "cos2_single": {}}
    for name, keys in SETS.items():
        r2s = []
        for i in ids:
            A = np.stack([G[k][i] for k in keys], 1)   # [d,k]
            v = V[i]
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            proj = A @ coef
            r2s.append(float((proj @ proj) / (v @ v + 1e-12)))
        res["sets"][name] = {"R2_mean": float(np.mean(r2s)), "R2_per_img": [round(x, 3) for x in r2s]}
        print(f"{name:16} R²={np.mean(r2s):.3f}  (per-img {[round(x,2) for x in r2s]})", flush=True)
    for k in SUR:
        c = [float((G[k][i] @ V[i]) ** 2 / ((G[k][i] @ G[k][i]) * (V[i] @ V[i]) + 1e-12)) for i in ids]
        res["cos2_single"][k] = float(np.mean(c))
    print("cos² đơn (phần năng lượng 1 surrogate bắt được):")
    for k in SUR: print(f"  {k:12} {res['cos2_single'][k]:.4f}")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g5_span_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
