"""G6 (progress_log 2026-11-17 05:35, KHÓA): hướng feature-backbone (KHÔNG head) của Swin-L có trong span surrogate không.
Tách vai trò head DINO vs backbone. Load tuần tự, ảnh sạch, trong IN."""
import json, os, sys
import numpy as np, torch
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.preprocess import resize_to_model, to_batch
from attack.targets import load_any, target_dataset, target_grads
from scripts.r1_region import region_masks
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SUR = ["r50", "r101", "convnext_t", "swin_t"]; K = 3.0

def feat_grad(model, key, orig, dsm):
    """input-grad của L_OSFD trên model.backbone (4 stage, k=3) tại δ=0."""
    from attack.targets import _loss_mode
    _loss_mode(model, "mrcnn" if key != "dino_swin_l" else "dino")  # eval/train mode đúng
    model.eval()
    x = orig.clone().requires_grad_(True)
    inp, _ = to_batch(model, [resize_to_model(x, tuple(dsm.img_shape))], [dsm])
    feats = model.backbone(inp)
    with torch.no_grad():
        fc = [f.detach() for f in feats]
    loss = sum(((K * c - f) ** 2).mean() for c, f in zip(fc, feats))
    loss.backward()
    return x.grad.detach().cpu()

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    coco = COCO(ANN); ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    ds0 = {s["img_id"]: s for s in target_dataset("r50", ids)}
    orig = {i: ds0[i]["orig_inputs"].float() for i in ids}
    inm = {i: torch.tensor(region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), *orig[i].shape[1:])["IN"], dtype=torch.bool) for i in ids}
    def gfeat(key):
        for p in (True,):
            m = load_any(key); ds = {s["img_id"]: s for s in target_dataset(key, ids)}
            for q in m.parameters(): q.requires_grad_(False)
            out = {}
            for i in ids:
                g = feat_grad(m, key, orig[i].cuda(), ds[i]["data_sample"])
                out[i] = g[:, inm[i]].reshape(-1).numpy()
            del m; torch.cuda.empty_cache(); print(f"[g6] {key} feat xong", flush=True)
            return out
    Gf = {k: gfeat(k) for k in SUR}
    Vf = gfeat("dino_swin_l")
    # v_cls swinL (để cos head-vs-feat)
    mc = load_any("dino_swin_l"); dsc = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    Vc = {}
    for i in ids:
        Vc[i] = target_grads(mc, "dino_swin_l", orig[i].cuda(), dsc[i]["data_sample"], objectives=("cls",))[0]["cls"][:, inm[i]].reshape(-1).cpu().numpy()
    del mc; torch.cuda.empty_cache()
    r2, cos_hf, cos2 = [], [], {k: [] for k in SUR}
    for i in ids:
        A = np.stack([Gf[k][i] for k in SUR], 1); v = Vf[i]
        coef, *_ = np.linalg.lstsq(A, v, rcond=None); proj = A @ coef
        r2.append(float((proj @ proj) / (v @ v + 1e-12)))
        cos_hf.append(float((Vc[i] @ Vf[i]) / (np.linalg.norm(Vc[i]) * np.linalg.norm(Vf[i]) + 1e-12)))
        for k in SUR:
            cos2[k].append(float((Gf[k][i] @ v) ** 2 / ((Gf[k][i] @ Gf[k][i]) * (v @ v) + 1e-12)))
    res = {"R2_feat_swinL_in_surrogate_span": float(np.mean(r2)), "R2_feat_per_img": [round(x, 3) for x in r2],
           "R2_cls_ref_G5_4bb": 0.016, "cos_vcls_vfeat_swinL": float(np.mean(cos_hf)),
           "cos2_feat_single": {k: float(np.mean(cos2[k])) for k in SUR}}
    print(f"\nR²_feat (hướng feature-backbone Swin-L trong span 4 surrogate): {res['R2_feat_swinL_in_surrogate_span']:.3f}")
    print(f"R²_cls tham chiếu (G5, 4bb): 0.016")
    print(f"cos(v_cls, v_feat) trên Swin-L: {res['cos_vcls_vfeat_swinL']:.3f}  (head đổi hướng bao nhiêu)")
    print("cos²_feat đơn:", {k: round(v, 4) for k, v in res["cos2_feat_single"].items()})
    v = ("BACKBONE với tới, HEAD là span-deficiency" if res["R2_feat_swinL_in_surrogate_span"] >= max(0.1, 3 * 0.016)
         else "Backbone Swin-L cũng ngoài span")
    res["verdict"] = v; print("G6 VERDICT:", v)
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g6_head_vs_bb_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
