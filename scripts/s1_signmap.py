"""S1 (progress_log 2026-11-17 09:50, KHÓA): bản đồ dấu theo thành phần loss — feature vs detector, 2 method, CÙNG
surrogate ConvNeXt-T, + WB DINO-Swin-L nhiều seed + dấu ngẫu nhiên. Quick test 10 ảnh (dev100[:10], như R1-WB/G3).

Ô sinh (surrogate Mask R-CNN ConvNeXt-T, B50, ε5, seed 1000·(n+1)):
  RD-{ALL,FEAT,DET}   : OCT 1 view, amp, WP ρ0.002 weight_linear (RD-ALL = METHODS["C2-OCT-1V-WP2-RD-CNX"] y hệt)
                        FEAT = L_OSFD C3–C5 (feat_stages=(1,2,3)), DET = chỉ loss cls RoI (feature=False)
  OSFD-{ALL,FEAT,DET} : geo_osfd_attack geo="rrb" (2 view/backward, co-transform), C2–C5, amp, không WP
                        ALL/DET thêm loss cls RoI λ0.2 (λ của CNX theo rd_calib, KHÔNG tune lại cho RRB)
  WB-0 / WB-r1 / WB-r2: white-box DINO-Swin-L, loss cls (không-dn), MI μ1 sign, B20, ε5 (như R1-WB);
                        WB-0 khởi tạo δ=0, WB-r{k} khởi tạo δ~U(−ε,ε) seed k
  RAND                : dấu ±ε ngẫu nhiên đều mọi pixel-kênh (seed 42) — mốc sàn
Đo: (A) đồng thuận dấu từng cặp ô (IN / OUT, gộp pixel-kênh 10 ảnh, bỏ pixel δ=0);
    (B) relative AP drop + kill trên dino_swin_l, swin_t, r101, fcos_r50, dino_r50;
    (C) dấu vs drop: theo ô (Spearman 6 ô attack) và theo ảnh (drop Swin-L từng ảnh, nửa cao vs nửa thấp).
  python scripts/s1_signmap.py            # sinh + eval + phân tích
  python scripts/s1_signmap.py --n 1 --tag smoke
"""
import argparse
import functools
import json
import os
import random
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.r1_region import region_masks
from scripts.run_baselines import METHODS, _C2OCT

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
SUR, EPS, STEPS = "convnext_t", 5.0, 50
WB_STEPS, WB_ALPHA, WB_MU = 20, 1.0, 1.0
WP = dict(weight_rho=0.002, weight_linear=True)
RRB = dict(geo="rrb", amp=True)
CLS = dict(cls_weight=0.2, cls_mode="roi", skip_mask=True)
ATTACK = {
    "RD-ALL": METHODS["C2-OCT-1V-WP2-RD-CNX"][0],
    "RD-FEAT": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=0.0, skip_mask=False,
                                                         feat_stages=(1, 2, 3)), **WP),
    "RD-DET": functools.partial(geo_osfd_attack, **dict(_C2OCT, cls_weight=0.2), cls_mode="roi", feature=False, **WP),
    "OSFD-ALL": functools.partial(geo_osfd_attack, **RRB, **CLS),
    "OSFD-FEAT": functools.partial(geo_osfd_attack, **RRB),
    "OSFD-DET": functools.partial(geo_osfd_attack, **RRB, **CLS, feature=False),
}
WB = {"WB-0": None, "WB-r1": 1, "WB-r2": 2}
CELLS = list(ATTACK) + list(WB) + ["RAND"]
TARGETS = ["dino_swin_l", "swin_t", "r101", "fcos_r50", "dino_r50"]


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def save(adv, p):
    mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), p + ".tmp.png")
    os.replace(p + ".tmp.png", p)


def gen_attack(ids, cell, out):
    if all(os.path.exists(os.path.join(out, f"{i}.png")) for i in ids):
        return
    os.makedirs(out, exist_ok=True)
    model = load_model(SUR, "cuda:0"); ds = {s["img_id"]: s for s in target_dataset(SUR, ids)}
    t0 = time.time()
    for n, i in enumerate(ids):
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        orig = ds[i]["orig_inputs"].cuda().float()
        seed_all(1000 * (n + 1))
        d = ATTACK[cell](model, orig, ds[i]["data_sample"], steps=STEPS, epsilon=EPS)
        adv = to_adv_image(orig, d)
        assert float((adv - orig).abs().max()) <= EPS + 1e-4
        save(adv, p)
        print(f"[s1 {cell}] {n + 1}/{len(ids)} {(time.time() - t0) / (n + 1):.1f}s/ảnh", flush=True)
    del model; torch.cuda.empty_cache()


def gen_wb(ids, outs):
    if all(os.path.exists(os.path.join(outs[c], f"{i}.png")) for c in WB for i in ids):
        return
    model = load_any("dino_swin_l"); ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    for cell, sd in WB.items():
        os.makedirs(outs[cell], exist_ok=True)
        for n, i in enumerate(ids):
            p = os.path.join(outs[cell], f"{i}.png")
            if os.path.exists(p):
                continue
            torch.manual_seed(1000 * (n + 1))
            orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
            if sd is None:
                delta = torch.zeros_like(orig)
            else:
                gen = torch.Generator(device="cpu").manual_seed(sd * 100000 + i % 100000)
                delta = ((torch.rand(orig.shape, generator=gen) * 2 - 1) * EPS).to(orig.device)
                delta = torch.clamp(orig + delta, 0, 255) - orig
            g = torch.zeros_like(orig)
            for _ in range(WB_STEPS):
                x = torch.clamp(orig + delta, 0, 255)
                grad = target_grads(model, "dino_swin_l", x, dsm, objectives=("cls",))[0]["cls"]
                g = WB_MU * g + grad / (grad.abs().mean() + 1e-12)
                delta = torch.clamp(delta + WB_ALPHA * torch.sign(g), -EPS, EPS)
                delta = torch.clamp(orig + delta, 0, 255) - orig
            save(to_adv_image(orig, delta), p)
            print(f"[s1 {cell}] {n + 1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()


def gen_rand(ids, out, clean):
    os.makedirs(out, exist_ok=True)
    for i in ids:
        p = os.path.join(out, f"{i}.png")
        if os.path.exists(p):
            continue
        rng = np.random.default_rng(42 + i)
        s = rng.choice([-1, 1], size=clean[i].shape).astype(np.int32)
        mmcv.imwrite(np.clip(clean[i] + int(EPS) * s, 0, 255).astype(np.uint8), p)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    gpu = torch.cuda.get_device_name(0)
    run = os.path.join(REPO_ROOT, f"artifacts/runs/s1_signmap{sfx}/adv")
    outs = {c: os.path.join(run, c) for c in CELLS}
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/s1_signmap_dev{args.n}{sfx}.json")
    print(f"[s1] GPU {gpu} | {len(ids)} ảnh | ô {CELLS}", flush=True)

    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    IN = {i: region_masks(gts[i], *clean[i].shape[:2])["IN"].astype(bool) for i in ids}
    for c in ATTACK:
        gen_attack(ids, c, outs[c])
    gen_wb(ids, outs)
    gen_rand(ids, outs["RAND"], clean)

    # ---- (A) bản đồ dấu ----
    S = {c: {i: np.sign(mmcv.imread(os.path.join(outs[c], f"{i}.png")).astype(np.int32) - clean[i]) for i in ids}
         for c in CELLS}

    def agree(a, b, region, imgs=ids, mask_fn=None):
        num = den = 0
        for i in imgs:
            r = IN[i] if region == "IN" else ~IN[i] if region == "OUT" else np.ones_like(IN[i])
            sa, sb = S[a][i][r], S[b][i][r]
            nz = (sa != 0) & (sb != 0)
            if mask_fn is not None:
                nz &= mask_fn(i)[r]
            num += int((sa[nz] == sb[nz]).sum()); den += int(nz.sum())
        return num / den if den else float("nan")

    # pixel-kênh mà 3 WB cùng dấu (dấu WB "ổn định")
    wb_cons = {i: (S["WB-0"][i] == S["WB-r1"][i]) & (S["WB-r1"][i] == S["WB-r2"][i]) & (S["WB-0"][i] != 0)
               for i in ids}
    res = {"gpu": gpu, "n": len(ids), "ids": ids, "agree": {}, "agree_wbcons_IN": {}, "drop": {}, "kill": {}}
    for reg in ("IN", "OUT"):
        res["agree"][reg] = {a: {b: agree(a, b, reg) for b in CELLS} for a in CELLS}
    cons_frac = float(np.mean(np.concatenate([wb_cons[i][IN[i]] for i in ids])))
    for c in CELLS:
        res["agree_wbcons_IN"][c] = agree(c, "WB-0", "IN", mask_fn=lambda i: wb_cons[i])
    res["wb_consensus_frac_IN"] = cons_frac
    print(f"\n(A) Đồng thuận dấu trong IN (hàng × cột) — n={len(ids)}")
    print(f"{'':10}" + "".join(f"{c[:9]:>10}" for c in CELLS))
    for a in CELLS:
        print(f"{a:10}" + "".join(f"{res['agree']['IN'][a][b]:10.3f}" for b in CELLS))
    print(f"\nTỉ lệ pixel-kênh IN mà 3 WB cùng dấu: {cons_frac:.3f}")
    print("Khớp dấu vs WB-0 (IN / OUT / chỉ pixel WB-đồng-thuận IN):")
    for c in CELLS:
        print(f"  {c:10} {res['agree']['IN'][c]['WB-0']:.3f} / {res['agree']['OUT'][c]['WB-0']:.3f} / "
              f"{res['agree_wbcons_IN'][c]:.3f}")

    # ---- (B) drop / kill ----
    from mmdet.apis import inference_detector
    per_img = {}
    for key in TARGETS:
        model = load(key)
        with torch.no_grad():
            cl, tp = [], {}
            for i in ids:
                d = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat); cl += d
                g = group(d).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            evc = CocoImageEval(coco, cl, ids); apc = evc.ap(np.ones(len(ids)))[0]
            oh = np.eye(len(ids))
            apc_i = np.array([evc.ap(oh[k])[0] for k in range(len(ids))])
            ntp = sum(len(v) for v in tp.values())
            res["drop"][key], res["kill"][key] = {}, {}
            for c in CELLS:
                ad, k = [], 0
                for i in ids:
                    d = to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(outs[c], f"{i}.png"))), i, cat)
                    ad += d
                    g = group(d).get(i, [])
                    k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                eva = CocoImageEval(coco, ad, ids)
                res["drop"][key][c] = 100 * (apc - eva.ap(np.ones(len(ids)))[0]) / max(apc, 1e-9)
                res["kill"][key][c] = 100 * k / max(ntp, 1)
                if key == "dino_swin_l":
                    apa_i = np.array([eva.ap(oh[j])[0] for j in range(len(ids))])
                    per_img[c] = np.where(apc_i > 0, 100 * (apc_i - apa_i) / np.maximum(apc_i, 1e-9), np.nan)
        del model; torch.cuda.empty_cache()
    print(f"\n(B) relative AP drop % (kill %)")
    print(f"{'':10}" + "".join(f"{t[:11]:>14}" for t in TARGETS))
    for c in CELLS:
        print(f"{c:10}" + "".join(f"{res['drop'][t][c]:8.1f} ({res['kill'][t][c]:3.0f})" for t in TARGETS))

    # ---- (C) dấu vs drop ----
    att = list(ATTACK)
    a_cell = [np.mean([res["agree"]["IN"][c][w] for w in WB]) for c in att]
    d_cell = [res["drop"]["dino_swin_l"][c] for c in att]
    rho_cell = spearmanr(a_cell, d_cell).correlation
    pts = []
    for c in att:
        for k, i in enumerate(ids):
            ag = np.mean([agree(c, w, "IN", imgs=[i]) for w in WB])
            pts.append((c, i, ag, float(per_img[c][k])))
    pts = [p for p in pts if np.isfinite(p[3])]
    ag_v = np.array([p[2] for p in pts]); dr_v = np.array([p[3] for p in pts])
    med = np.median(dr_v)
    hi, lo = ag_v[dr_v > med].mean(), ag_v[dr_v <= med].mean()
    rho_img = spearmanr(ag_v, dr_v).correlation
    res["sign_vs_drop"] = {"cell_agree_wb_IN": dict(zip(att, a_cell)), "cell_drop_swinl": dict(zip(att, d_cell)),
                           "spearman_cell": rho_cell, "spearman_img": rho_img, "median_drop": float(med),
                           "agree_hi_drop": float(hi), "agree_lo_drop": float(lo),
                           "points": [{"cell": p[0], "img": p[1], "agree": p[2], "drop": p[3]} for p in pts]}
    print(f"\n(C) dấu vs drop Swin-L: Spearman theo ô (6) {rho_cell:+.2f} | theo (ô, ảnh) ({len(pts)}) {rho_img:+.2f}")
    print(f"    khớp WB trong IN: nửa drop CAO {hi:.3f} vs nửa drop THẤP {lo:.3f} (Δ {hi - lo:+.3f})")

    # ---- phán quyết (quy tắc khóa) ----
    A = res["agree"]["IN"]
    wb_pair = np.mean([A["WB-0"]["WB-r1"], A["WB-0"]["WB-r2"], A["WB-r1"]["WB-r2"]])
    wb_drop = min(res["drop"]["dino_swin_l"][w] for w in WB)
    v1 = ("ỔN ĐỊNH: bản đồ dấu WB gần duy nhất -> 'khớp WB' là thước đo dùng được" if wb_pair >= 0.8 else
          "NHIỀU NGHIỆM: dấu WB không duy nhất -> 'khớp WB' là thước đo SAI" if wb_pair <= 0.6 and wb_drop >= 80 else
          "KHÔNG RÕ")
    comp = {}
    for m in ("RD", "OSFD"):
        af, ad_, fd = A[f"{m}-ALL"][f"{m}-FEAT"], A[f"{m}-ALL"][f"{m}-DET"], A[f"{m}-FEAT"][f"{m}-DET"]
        dom = "FEAT" if af - ad_ >= 0.05 else "DET" if ad_ - af >= 0.05 else "CÂN"
        comp[m] = {"all_feat": af, "all_det": ad_, "feat_det": fd, "dominant": dom}
    v3 = ("DẤU PHÂN BIỆT drop" if (hi - lo) >= 0.03 and rho_img > 0 else "DẤU KHÔNG phân biệt drop")
    res["verdict"] = {"wb_pairwise_IN": wb_pair, "wb_min_drop": wb_drop, "wb": v1, "component": comp, "sign_drop": v3}
    print(f"\n=> WB đôi một (IN) {wb_pair:.3f}, drop WB min {wb_drop:.0f}% -> {v1}")
    for m, d in comp.items():
        print(f"=> {m}: ALL~FEAT {d['all_feat']:.3f}  ALL~DET {d['all_det']:.3f}  FEAT~DET {d['feat_det']:.3f} "
              f"-> dấu cuối do {d['dominant']}")
    print(f"=> {v3}")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(f"[s1] -> {out_json}")


if __name__ == "__main__":
    main()
