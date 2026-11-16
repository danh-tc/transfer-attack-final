"""G10 (progress_log 2026-11-17 07:50, KHÓA): feature-dispersion attack từ backbone phân loại IN-22K (Swin-L vs ConvNeXt-L),
đo transfer mAP drop trên DINO-Swin-L + controls. Behavioral."""
import json, os, sys
import mmcv, numpy as np, torch, torch.nn.functional as F
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json"); IMG = os.path.join(REPO_ROOT, "data/coco/val2017")
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).cuda(); STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).cuda()
EPS, STEPS, RES, K = 5.0, 50, 640, 3.0
BBS = {"swin_large_cls": ("swin", "checkpoints/swin-large_22kto1k.pth"),
       "convnext_large_cls": ("convnext", "checkpoints/convnext-large_in21k.pth")}
TGT = ["dino_swin_l", "r101", "dino_r50", "convnext_t"]
ART = os.path.join(REPO_ROOT, "artifacts/runs/g10_bbattack/adv")

def build(kind, ckpt):
    from mmpretrain.models import ConvNeXt, SwinTransformer
    if kind == "swin":
        bb = SwinTransformer(arch='large', out_indices=(0, 1, 2, 3), drop_path_rate=0.0, pad_small_map=True)
    else:
        bb = ConvNeXt(arch='large', out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
    ck = torch.load(os.path.join(REPO_ROOT, ckpt), map_location="cpu"); ck = ck.get("state_dict", ck)
    bb.load_state_dict({k[9:]: v for k, v in ck.items() if k.startswith("backbone.")}, strict=False)
    bb = bb.cuda(); bb.eval()
    for q in bb.parameters(): q.requires_grad_(False)
    return bb

def feats(bb, x):
    rgb = x.flip(0); xn = (rgb - MEAN) / STD
    xr = F.interpolate(xn[None], size=(RES, RES), mode="bilinear", align_corners=False)
    return bb(xr)

def attack(bb, orig):
    delta = torch.zeros_like(orig); g = torch.zeros_like(orig)
    with torch.no_grad():
        fc = [f.detach() for f in feats(bb, orig)]
    for _ in range(STEPS):
        x = torch.clamp(orig + delta, 0, 255).requires_grad_(True)
        fa = feats(bb, x)
        loss = sum(((K * c - a) ** 2).mean() for a, c in zip(fc, fa))  # OSFD: k·F_clean target, gradient != 0 tại δ=0
        gr = torch.autograd.grad(loss, x)[0]
        g = g + gr / (gr.abs().mean() + 1e-12)
        delta = torch.clamp(delta - torch.sign(g) * 1.0, -EPS, EPS)  # descent (OSFD minimize)
        delta = torch.clamp(orig + delta, 0, 255) - orig
    return delta.detach()

def png(name, i): return os.path.join(ART, name, f"{i}.png")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from mmdet.apis import inference_detector
    coco = COCO(ANN); cat = sorted(coco.getCatIds()); ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    ds = {s["img_id"]: s for s in target_dataset("r50", ids)}
    orig = {i: ds[i]["orig_inputs"].float() for i in ids}
    for name, (kind, ckpt) in BBS.items():
        if all(os.path.exists(png(name, i)) for i in ids): continue
        bb = build(kind, ckpt); os.makedirs(os.path.join(ART, name), exist_ok=True)
        for n, i in enumerate(ids):
            torch.manual_seed(1000 * (n + 1))
            d = attack(bb, orig[i].cuda())
            a = to_adv_image(orig[i].cuda(), d)
            mmcv.imwrite(a.permute(1, 2, 0).cpu().numpy().astype(np.uint8), png(name, i))
            print(f"[g10 {name}] {n+1}/10", flush=True)
        del bb; torch.cuda.empty_cache()
    res = {}
    for key in TGT:
        m = load(key)
        with torch.no_grad():
            cl = []; tp = {}
            for i in ids:
                p = os.path.join(IMG, coco.loadImgs(i)[0]["file_name"]); det = inference_detector(m, p); cl += to_coco_dets(det, i, cat)
                gg = group(to_coco_dets(det, i, cat)).get(i, []); tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) if fate(a["bbox"], a["category_id"], gg)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values()); apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]
            res[key] = {"clean_ap": round(apc * 100, 1)}
            for name in BBS:
                ad = []; k = 0
                for i in ids:
                    det = inference_detector(m, png(name, i)); ad += to_coco_dets(det, i, cat)
                    gg = group(to_coco_dets(det, i, cat)).get(i, []); k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                apa = CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]
                res[key][name] = {"drop": round(100 * (apc - apa) / apc, 1), "kill": round(100 * k / max(ntp, 1), 1)}
                print(f"{name:20}->{key:12} AP {apc*100:.0f}->{apa*100:.0f} drop {res[key][name]['drop']}% kill {res[key][name]['kill']}%", flush=True)
        del m; torch.cuda.empty_cache()
    d = res["dino_swin_l"]["swin_large_cls"]["drop"] - res["dino_swin_l"]["convnext_large_cls"]["drop"]
    res["swinL_minus_cnxL_on_SwinL"] = d
    res["verdict"] = "KIẾN TRÚC Swin bắc cầu behavioral" if d >= 10 else ("backbone Swin-L cũng không giúp" if d <= 3 else "KHÔNG RÕ")
    print(f"\nΔdrop SwinL (Swin-L-bb − ConvNeXt-L-bb) trên DINO-Swin-L = {d:+.1f} -> {res['verdict']}")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/g10_bbattack_dev10.json"), "w"), indent=1, ensure_ascii=False)
if __name__ == "__main__":
    main()
