"""G7 (progress_log 2026-11-17 06:10, KHÓA): ConvNeXt-Large IN-21K backbone có bắc được hướng feature DINO-Swin-L không."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.preprocess import resize_to_model, to_batch
from attack.targets import load_any, target_dataset
from scripts.r1_region import region_masks
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
SUR = ["r50", "r101", "convnext_t", "swin_t"]; K = 3.0
CKPT_L = os.path.join(REPO_ROOT, "checkpoints/convnext-large_in21k.pth")
# ImageNet norm (mmpretrain, RGB, thang 0-255)
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1); STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

def feat_grad_det(model, key, orig, dsm):
    from attack.targets import _loss_mode
    _loss_mode(model, "mrcnn" if key != "dino_swin_l" else "dino"); model.eval()
    x = orig.clone().requires_grad_(True)
    inp, _ = to_batch(model, [resize_to_model(x, tuple(dsm.img_shape))], [dsm])
    feats = model.backbone(inp)
    fc = [f.detach() for f in feats]
    sum(((K * c - f) ** 2).mean() for c, f in zip(fc, feats)).backward()
    return x.grad.detach().cpu()

def feat_grad_cnxL(bb, orig):
    """orig BGR 0-255 [C,h,w] -> RGB, ImageNet norm, resize 640, ConvNeXt-L stages, OSFD feat grad."""
    x = orig.clone().requires_grad_(True)
    rgb = x.flip(0)  # BGR->RGB
    xn = (rgb - MEAN.to(x.device)) / STD.to(x.device)
    xr = F.interpolate(xn[None], size=(640, 640), mode="bilinear", align_corners=False)
    feats = bb(xr)
    fc = [f.detach() for f in feats]
    sum(((K * c - f) ** 2).mean() for c, f in zip(fc, feats)).backward()
    return x.grad.detach().cpu()

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmpretrain.models import ConvNeXt
    coco = COCO(ANN); ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    ds0 = {s["img_id"]: s for s in target_dataset("r50", ids)}
    orig = {i: ds0[i]["orig_inputs"].float() for i in ids}
    inm = {i: torch.tensor(region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), *orig[i].shape[1:])["IN"], dtype=torch.bool) for i in ids}
    def gdet(key):
        m = load_any(key)
        for q in m.parameters(): q.requires_grad_(False)
        ds = {s["img_id"]: s for s in target_dataset(key, ids)}
        out = {i: feat_grad_det(m, key, orig[i].cuda(), ds[i]["data_sample"])[:, inm[i]].reshape(-1).numpy() for i in ids}
        del m; torch.cuda.empty_cache(); print(f"[g7] {key} xong", flush=True); return out
    G = {k: gdet(k) for k in SUR}
    Vf = gdet("dino_swin_l")
    # ConvNeXt-L
    bb = ConvNeXt(arch='large', out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
    sd = torch.load(CKPT_L, map_location="cpu")["state_dict"]
    bbsd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    miss = bb.load_state_dict(bbsd, strict=False)
    print(f"[g7] CNX-L load: missing {len(miss.missing_keys)} unexpected {len(miss.unexpected_keys)}", flush=True)
    bb = bb.cuda(); bb.eval()
    for q in bb.parameters(): q.requires_grad_(False)
    GL = {i: feat_grad_cnxL(bb, orig[i].cuda())[:, inm[i]].reshape(-1).numpy() for i in ids}
    del bb; torch.cuda.empty_cache()
    def r2(keys, Gd):
        out = []
        for i in ids:
            A = np.stack([Gd[k][i] for k in keys], 1); v = Vf[i]
            coef, *_ = np.linalg.lstsq(A, v, rcond=None); proj = A @ coef
            out.append(float((proj @ proj) / (v @ v + 1e-12)))
        return float(np.mean(out))
    Gall = {**G, "convnext_large": GL}
    cos2 = lambda g: float(np.mean([(g[i] @ Vf[i]) ** 2 / ((g[i] @ g[i]) * (Vf[i] @ Vf[i]) + 1e-12) for i in ids]))
    res = {"R2_4bb": r2(SUR, G), "R2_4bb_plus_cnxL": r2(SUR + ["convnext_large"], Gall),
           "cos2_convnext_large": cos2(GL), "cos2_convnext_t": cos2(G["convnext_t"]), "cos2_swin_t": cos2(G["swin_t"])}
    print(f"\nR²(4 backbone nhỏ): {res['R2_4bb']:.3f}")
    print(f"R²(4 + ConvNeXt-L IN21K): {res['R2_4bb_plus_cnxL']:.3f}")
    print(f"cos²_single: ConvNeXt-L {res['cos2_convnext_large']:.4f} | ConvNeXt-T {res['cos2_convnext_t']:.4f} | Swin-T {res['cos2_swin_t']:.4f}")
    res["verdict"] = ("Backbone lớn IN-21K bắc được" if res["cos2_convnext_large"] > 0.05 or res["R2_4bb_plus_cnxL"] - res["R2_4bb"] > 0.1
                      else "Backbone lớn CNN vẫn không bắc được hướng feature Swin-L")
    print("G7 VERDICT:", res["verdict"])
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g7_bigbb_dev10.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
