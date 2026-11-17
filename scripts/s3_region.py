"""S3 (progress_log 2026-11-17 10:25, KHÓA): vùng pixel ĐỊNH NGHĨA LẠI theo mask instance + kết luận phải VỮNG QUA NHIỀU
BẢN ĐỒ DẤU (5 seed WB + 6 ô attack), KHÔNG so dấu giữa 2 δ khác nhau — chỉ thao tác trên CHÍNH δ đó + độ nhạy target.
10 ảnh dev100[:10], PNG S1 (+ sinh WB-r3, WB-r4 cùng công thức S1).

Vùng (rời nhau, phủ ảnh): MASK = hợp mask GT (kể cả crowd) | BOXBG = hợp box GT \\ MASK | RING = box nở 25% mỗi phía \\ box
                         | FAR = phần còn lại.
(a) Hành vi: mỗi δ ∈ {6 attack, 5 WB}, biến thể full / keep_R / rem_R (R ∈ 4 vùng) / rnd_R (đảo dấu ngẫu nhiên, giữ |δ|,
    R ∈ 4 vùng + BOX = MASK∪BOXBG + ALL) / flip25_BOX (đảo đúng 25% pixel-kênh trong BOX). Eval dino_swin_l + r101:
    kill rate object clean-TP + relative AP drop (10 ảnh). Hiệu suất vùng eff_R = share_keep_R / diện tích_R.
(b) Độ nhạy: g_t = ∇ₓ L_cls DINO-Swin-L tại x + t·δ (t ∈ {0, 0.5, 1}); theo vùng: mass_R = ‖g_R‖₁/‖g‖₁, mật độ
    mass_R/area_R, A_R = Σ_R sign δ·g / Σ_R |g|. g lưu fp16 ra đĩa (artifacts/runs/s3_region/grads).
  python scripts/s3_region.py
"""
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.s1_signmap as s1
from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, to_coco_dets
from attack.targets import load_any, target_dataset, target_grads
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ADV = os.path.join(REPO_ROOT, "artifacts/runs/s1_signmap/adv")
GDIR = os.path.join(REPO_ROOT, "artifacts/runs/s3_region/grads")
OUT = os.path.join(REPO_ROOT, "results/mechanism/s3_region_dev10.json")
ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
WBS = ["WB-0", "WB-r1", "WB-r2", "WB-r3", "WB-r4"]
CELLS = ATT + WBS
REG = ["MASK", "BOXBG", "RING", "FAR"]
VARS = (["full"] + [f"keep_{r}" for r in REG] + [f"rem_{r}" for r in REG]
        + [f"rnd_{r}" for r in REG + ["BOX", "ALL"]] + ["flip25_BOX"])
TARGETS = ["dino_swin_l", "r101"]
TS = (0.0, 0.5, 1.0)


def regions(coco, i, h, w):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
    mask = np.zeros((h, w), bool); box = np.zeros((h, w), bool); big = np.zeros((h, w), bool)
    for a in anns:
        mask |= coco.annToMask(a) > 0
        x, y, bw, bh = a["bbox"]
        box[max(int(y), 0):int(np.ceil(y + bh)), max(int(x), 0):int(np.ceil(x + bw))] = True
        ex, ey = 0.25 * bw, 0.25 * bh
        big[max(int(y - ey), 0):int(np.ceil(y + bh + ey)), max(int(x - ex), 0):int(np.ceil(x + bw + ex))] = True
    box |= mask; big |= box
    R = {"MASK": mask, "BOXBG": box & ~mask, "RING": big & ~box, "FAR": ~big}
    assert sum(r.astype(int) for r in R.values()).max() == 1 and sum(r.astype(int) for r in R.values()).min() == 1
    R["BOX"] = box; R["ALL"] = np.ones((h, w), bool)
    return R


def variant(clean, d, v, R, rng):
    """clean [H,W,3] int, d [H,W,3] int -> ảnh uint8."""
    if v == "full":
        dd = d
    elif v.startswith("keep_"):
        dd = d * R[v[5:]][:, :, None]
    elif v.startswith("rem_"):
        dd = d * (~R[v[4:]])[:, :, None]
    elif v.startswith("rnd_"):
        m = R[v[4:]][:, :, None]
        s = rng.choice([-1, 1], size=d.shape)
        dd = np.where(m, np.abs(d) * s, d)
    else:  # flip25_BOX: đảo đúng 25% pixel-kênh (δ≠0) trong BOX
        m = np.broadcast_to(R["BOX"][:, :, None], d.shape) & (d != 0)
        idx = np.flatnonzero(m)
        flip = rng.choice(idx, size=len(idx) // 4, replace=False)
        dd = d.copy().reshape(-1); dd[flip] = -dd[flip]; dd = dd.reshape(d.shape)
    return np.clip(clean + dd, 0, 255).astype(np.uint8)


def robust(vals, cond):
    return int(sum(bool(cond(v)) for v in vals))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global OUT
    if args.tag:
        OUT = OUT.replace(".json", f"_{args.tag}.json")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    s1.WB = {"WB-r3": 3, "WB-r4": 4}
    s1.gen_wb(ids, {c: os.path.join(ADV, c) for c in s1.WB})
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    R = {i: regions(coco, i, *clean[i].shape[:2]) for i in ids}
    D = {c: {i: mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i] for i in ids} for c in CELLS}
    gts = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    area = {r: float(np.mean([R[i][r].mean() for i in ids])) for r in REG}
    res = {"n": len(ids), "ids": ids, "area": area, "kill": {}, "drop": {}, "sens": {}}
    print("[s3] diện tích TB: " + "  ".join(f"{r} {area[r]:.2f}" for r in REG), flush=True)

    # ---- (a) hành vi ----
    from mmdet.apis import inference_detector
    for key in TARGETS:
        model = load(key)
        res["kill"][key], res["drop"][key] = {}, {}
        with torch.no_grad():
            cl, tp = [], {}
            for i in ids:
                d = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat); cl += d
                g = group(d).get(i, [])
                tp[i] = [a for a in gts[i] if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            apc = CocoImageEval(coco, cl, ids).ap(np.ones(len(ids)))[0]
            ntp = sum(len(v) for v in tp.values())
            for c in CELLS:
                res["kill"][key][c], res["drop"][key][c] = {}, {}
                for v in VARS:
                    ad, k = [], 0
                    for i in ids:
                        rng = np.random.default_rng(42 + i)
                        img = variant(clean[i], D[c][i], v, R[i], rng)
                        d = to_coco_dets(inference_detector(model, img), i, cat); ad += d
                        g = group(d).get(i, [])
                        k += sum(fate(x["bbox"], x["category_id"], g)[0] != "survived" for x in tp[i])
                    res["kill"][key][c][v] = 100 * k / max(ntp, 1)
                    res["drop"][key][c][v] = 100 * (apc - CocoImageEval(coco, ad, ids).ap(np.ones(len(ids)))[0]) / apc
                print(f"[s3 {key}] {c}: full kill {res['kill'][key][c]['full']:.0f}", flush=True)
        del model; torch.cuda.empty_cache()

    # ---- (b) độ nhạy target ----
    os.makedirs(GDIR, exist_ok=True)
    model = load_any("dino_swin_l")
    ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    acc = {c: {t: {r: [0.0, 0.0] for r in REG} for t in TS} for c in CELLS + ["RAND"]}   # [Σ sign·g, Σ|g|]
    D["RAND"] = {i: (np.random.default_rng(42 + i).choice([-1, 1], size=clean[i].shape) * 5).astype(np.int32) for i in ids}
    for n, i in enumerate(ids):
        orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]

        def grad_at(x, name):
            p = os.path.join(GDIR, f"{i}_{name}.npy")
            if os.path.exists(p):
                return np.load(p).astype(np.float32)
            torch.manual_seed(1000 * (n + 1))
            g = target_grads(model, "dino_swin_l", x, dsm, objectives=("cls",))[0]["cls"].float().cpu().numpy()
            np.save(p, g.astype(np.float16))
            return g
        g0 = grad_at(orig, "clean")
        for c in CELLS + ["RAND"]:
            dt = torch.from_numpy(D[c][i].transpose(2, 0, 1)).float().cuda()
            sd = np.sign(D[c][i].transpose(2, 0, 1))
            for t in TS:
                g = g0 if t == 0.0 else grad_at(torch.clamp(orig + t * dt, 0, 255), f"{c}_t{t}")
                g = g / (np.abs(g).sum() + 1e-30)          # mỗi ảnh đóng góp ngang nhau
                for r in REG:
                    m = np.broadcast_to(R[i][r][None], g.shape)
                    acc[c][t][r][0] += float((sd * g)[m].sum()); acc[c][t][r][1] += float(np.abs(g)[m].sum())
        print(f"[s3 grad] {n + 1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()
    for c in CELLS + ["RAND"]:
        res["sens"][c] = {}
        for t in TS:
            tot = sum(acc[c][t][r][1] for r in REG)
            res["sens"][c][str(t)] = {r: {"mass": acc[c][t][r][1] / tot, "density": acc[c][t][r][1] / tot / area[r],
                                          "A": acc[c][t][r][0] / max(acc[c][t][r][1], 1e-30)} for r in REG}

    # ---- in + phán quyết ----
    K = res["kill"]["dino_swin_l"]
    share = {c: {r: (K[c][f"keep_{r}"] / K[c]["full"] if K[c]["full"] else float("nan")) for r in REG} for c in CELLS}
    eff = {c: {r: share[c][r] / area[r] for r in REG} for c in CELLS}
    res["share_keep"], res["eff"] = share, eff
    print(f"\n(a) DINO-Swin-L kill % (10 ảnh)")
    print(f"{'':10}" + "".join(f"{v[:11]:>12}" for v in VARS))
    for c in CELLS:
        print(f"{c:10}" + "".join(f"{K[c][v]:12.0f}" for v in VARS))
    print("\nshare keep_R (= kill keep_R / kill full) | hiệu suất share/diện tích")
    for c in CELLS:
        print(f"  {c:10} " + "  ".join(f"{r} {share[c][r]:.2f}/{eff[c][r]:.1f}" for r in REG))
    print("\n(b) độ nhạy |g_t| Swin-L ảnh sạch: " + "  ".join(
        f"{r} mass {res['sens']['RD-ALL']['0.0'][r]['mass']:.2f} (mật độ {res['sens']['RD-ALL']['0.0'][r]['density']:.1f})"
        for r in REG))
    print("A_R TB t∈{0.5,1} (khớp có trọng số trong vùng):")
    Am = {c: {r: float(np.mean([res["sens"][c][str(t)][r]["A"] for t in (0.5, 1.0)])) for r in REG} for c in CELLS + ["RAND"]}
    res["A_mid"] = Am
    for c in CELLS + ["RAND"]:
        print(f"  {c:10} " + "  ".join(f"{r} {Am[c][r]:+.4f}" for r in REG))

    W, Aa = WBS, ATT
    v = {}
    n_mask_main_wb = robust([share[c] for c in W], lambda s: s["MASK"] >= 0.5)
    n_mask_main_att = robust([share[c] for c in Aa], lambda s: s["MASK"] >= 0.5)
    n_eff_wb = robust([eff[c] for c in W], lambda e: e["MASK"] > e["BOXBG"])
    n_eff_att = robust([eff[c] for c in Aa], lambda e: e["MASK"] > e["BOXBG"])
    n_ctx_wb = robust([share[c] for c in W], lambda s: s["BOXBG"] + s["RING"] >= s["MASK"])
    n_ctx_att = robust([share[c] for c in Aa], lambda s: s["BOXBG"] + s["RING"] >= s["MASK"])
    v["1_region"] = {"mask_share>=0.5": [n_mask_main_wb, n_mask_main_att], "eff_mask>eff_boxbg": [n_eff_wb, n_eff_att],
                     "context>=mask": [n_ctx_wb, n_ctx_att],
                     "verdict": ("MASK là vùng chính (vững)" if n_mask_main_wb >= 4 and n_mask_main_att >= 5 and n_eff_wb >= 4 and n_eff_att >= 5
                                 else "NGỮ CẢNH (BOXBG+RING) ≥ MASK (vững)" if n_ctx_wb >= 4 and n_ctx_att >= 5
                                 else "KHÔNG VỮNG qua các bản đồ dấu")}
    dsg = {c: {r: K[c]["full"] - K[c][f"rnd_{r}"] for r in REG} for c in CELLS}
    top = {c: max(REG, key=lambda r: dsg[c][r]) for c in CELLS}
    res["sign_loss_by_region"] = dsg
    cnt = {r: [sum(top[c] == r for c in W), sum(top[c] == r for c in Aa)] for r in REG}
    best = max(REG, key=lambda r: cnt[r][0] + cnt[r][1])
    v["2_sign_region"] = {"top_region_count[wb,att]": cnt,
                          "verdict": (f"dấu quan trọng nhất ở {best} (vững)" if cnt[best][0] >= 4 and cnt[best][1] >= 5
                                      else "vùng dấu quan trọng KHÔNG vững")}
    rb = [K[c]["rnd_BOX"] for c in W]
    v["3_g3"] = {"wb_rnd_BOX_kill": rb, "wb_flip25_BOX_kill": [K[c]["flip25_BOX"] for c in W],
                 "verdict": ("G3 sàn SAI (rnd trong box ≈ mất hết), R1 đúng" if max(rb) <= 8 else
                             "R1 rnd_IN SAI (sàn G3 tái lập)" if min(rb) >= 15 else "KHÔNG RÕ")}
    defi = {r: [Am[w][r] - np.mean([Am[a][r] for a in Aa]) for w in W] for r in REG}
    big = {w: max(REG, key=lambda r: (Am[w][r] - np.mean([Am[a][r] for a in Aa])) * res["sens"][w]["0.5"][r]["mass"])
           for w in W}
    cntd = {r: sum(big[w] == r for w in W) for r in REG}
    bd = max(REG, key=lambda r: cntd[r])
    v["4_deficit"] = {"A_wb_minus_att": defi, "largest_weighted_deficit_count": cntd,
                      "verdict": (f"thiếu hụt khớp (×khối nhạy) lớn nhất ở {bd} (vững)" if cntd[bd] >= 4
                                  else "vùng thiếu hụt KHÔNG vững")}
    res["verdict"] = v
    print("\n=> 1 vùng (hành vi):", v["1_region"])
    print("=> 2 dấu theo vùng:", v["2_sign_region"])
    print("=> 3 G3:", v["3_g3"])
    print("=> 4 thiếu hụt:", v["4_deficit"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"[s3] -> {OUT}")


if __name__ == "__main__":
    main()
