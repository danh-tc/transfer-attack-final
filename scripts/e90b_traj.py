"""E90b — trajectory object-level consensus-high vs low trên surrogate ConvNeXt (RD-CNX đóng băng), devC300 20 ảnh đầu.
Quy tắc khóa: docs/progress_log.md entry "E90b" 2026-11-14. Chẩn đoán, KHÔNG tuning; target chỉ dùng OFFLINE để phân tích.

Luồng (1 tiến trình, 1 GPU):
 1. Ảnh SẠCH -> dets của CONS = {fcos_r50, dino_r50, yolov3_d53} (consensus), TGT = {yolox_l, dino_swin_l}, surrogate CNX.
 2. Ghép cặp (chỉ thông tin SẠCH, trước attack): object = GT không-crowd, CNX detect sạch >= 0.3; C = TB score sạch matched
    trên CONS; cặp (hi, lo) trong cùng ảnh: C_hi - C_lo >= 0.2 VÀ |logit s_cnx(hi) - logit s_cnx(lo)| <= 0.5; tham lam theo
    |Δlogit s_cnx| tăng dần (hòa: ΔC lớn trước, rồi chỉ số ann), rời nhau, tối đa PAIRS_PER_IMG cặp/ảnh.
 3. Attack RD-CNX (METHODS["C2-OCT-1V-WP2-RD-CNX"], steps 50, ε 5, seed 1000·(n+1)); hook on_step CHỈ chép δ tại {5,10,25,50}
    (trong lúc attack trọng số đang là θ' của WP -> không tính gì trong hook). Kiểm checksum trọng số trước/sau.
 4. Đo SAU attack, trọng số gốc, eval mode, fp32, trên ảnh adv THẬT (round, clamp) tại mốc t ∈ {0,5,10,25,50}:
    z_i(t) = logit p_y(RoI = box GT i) của RoI head CNX; score detection matched (CNX, TGT).
    Gradient (chỉ object trong cặp, t ∈ {0, 50}): ‖∇ₓ z_i‖₁ (L∞ attack: mức giảm tuyến tính tối đa / đơn vị α), ‖·‖₂;
    hiệu suất hướng sớm e_i = -∇z_i(0)·δ_5 / (5·‖∇z_i(0)‖₁) ∈ [-1, 1].
  OMP_NUM_THREADS=4 python scripts/e90b_traj.py [--n 20] [--pairs 2] [--no-grad] [--smoke]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from attack.mechanism import DETECT_SCORE, _matched_scores
from attack.models import load_model
from attack.preprocess import resize_to_model, to_adv_image, to_batch
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.run_baselines import METHODS

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
CONS = ["fcos_r50", "dino_r50", "yolov3_d53"]
TGT = ["yolox_l", "dino_swin_l"]
SUR = "convnext_t"
CKPTS = [0, 5, 10, 25, 50]
GRAD_T = [0, 50]
EPS, FLOOR, BOOT_SEED, NB = 5.0, 0.01, 2026, 1000
DC_MIN = float(os.environ.get("E90B_DC", 0.2))
DS_MAX = float(os.environ.get("E90B_DS", 0.5))
MIN_PAIRS, MIN_IMGS = 10, 6


def seed_all(s):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)


def lgc(p):
    p = np.clip(np.asarray(p, float), FLOOR, 0.999)
    return np.log(p / (1 - p))


def wsum(model):
    return float(sum(p.detach().double().sum() for p in model.parameters()))


def adv_np(orig, delta):
    return to_adv_image(orig, delta).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # BGR HWC


def dets_on(model, img_np, img_id, cat_ids):
    from mmdet.apis import inference_detector
    return to_coco_dets(inference_detector(model, img_np), img_id, cat_ids)


def gt_logits(model, x, ds, boxes_xyxy0, labels, need_grad=False):
    """x [C,h0,w0] float (ảnh adv thật, không gian gốc). -> z [N] = logit p_y của RoI head trên box GT."""
    from mmdet.structures.bbox import bbox2roi
    H, W = ds.img_shape
    h0, w0 = x.shape[1:]
    sc = torch.tensor([W / w0, H / h0, W / w0, H / h0], device=x.device)
    inputs, _ = to_batch(model, [resize_to_model(x, (H, W))], [ds])
    feats = model.extract_feat(inputs)
    rois = bbox2roi([boxes_xyxy0 * sc])
    cls = model.roi_head._bbox_forward(feats, rois)["cls_score"].float()
    lp = torch.log_softmax(cls, -1)[torch.arange(len(labels)), labels]
    return lp - torch.log1p(-torch.exp(lp).clamp(max=1 - 1e-6))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 ảnh đầu, kiểm hook + thời gian; KHÔNG đọc số")
    ap.add_argument("--pairs-only", action="store_true", help="chỉ dets sạch + ghép cặp (không attack, không outcome)")
    args = ap.parse_args()
    T0 = time.time()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devC300.csv"))[: (2 if args.smoke else args.n)]
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    lab_of = {c: j for j, c in enumerate(cat_ids)}
    anns = {i: coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in ids}
    tm = {}

    # ---- 1. dets sạch CONS + TGT ----
    clean_np = {}
    sur_ds = {s["img_id"]: s for s in target_dataset(SUR, ids)}
    for i in ids:
        clean_np[i] = sur_ds[i]["orig_inputs"].permute(1, 2, 0).numpy().astype(np.uint8)
    clean_s = {}
    t = time.time()
    tgt_models = {}
    for key in CONS + TGT:
        m = load(key)
        dets = sum((dets_on(m, clean_np[i], i, cat_ids) for i in ids), [])
        clean_s[key] = _matched_scores(coco, dets, ids)
        if key in TGT:
            tgt_models[key] = m
        else:
            del m
        torch.cuda.empty_cache()
    tm["clean_dets"] = time.time() - t

    # ---- surrogate ----
    model = load_model(SUR, "cuda:0")
    model.eval()
    t = time.time()
    dets = sum((dets_on(model, clean_np[i], i, cat_ids) for i in ids), [])
    clean_s[SUR] = _matched_scores(coco, dets, ids)

    # ---- 2. ghép cặp (chỉ sạch) ----
    objs, pairs = {}, []
    for k, i in enumerate(ids):
        sc = clean_s[SUR][i]
        el = [j for j in range(len(anns[i])) if sc[j] >= DETECT_SCORE]
        C = {j: float(np.mean([clean_s[c][i][j] for c in CONS])) for j in el}
        objs[i] = {"eligible": el, "C": C}
        cand = []
        for a in el:
            for b in el:
                if a != b and C[a] - C[b] >= DC_MIN and abs(lgc(sc[a]) - lgc(sc[b])) <= DS_MAX:
                    cand.append((abs(lgc(sc[a]) - lgc(sc[b])), -(C[a] - C[b]), a, b))
        used, got = set(), 0
        for _, _, a, b in sorted(cand):
            if got >= args.pairs:
                break
            if a in used or b in used:
                continue
            pairs.append((k, i, a, b)); used |= {a, b}; got += 1

    if args.pairs_only:
        print(f"[pairs-only] n_images {len(ids)}  n_eligible {sum(len(o['eligible']) for o in objs.values())}  "
              f"n_pairs {len(pairs)} từ {len({p[0] for p in pairs})} ảnh  (cap {args.pairs}/ảnh)")
        return

    # ---- 3. attack + 4. đo ----
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    rec = {}
    w0 = wsum(model)
    t_att = t_diag = 0.0
    hook_check = None
    for k, i in enumerate(ids):
        s = sur_ds[i]
        orig = s["orig_inputs"].cuda().float()
        snaps = {0: torch.zeros_like(orig)}
        seed_all(1000 * (k + 1))
        t = time.time()
        d50 = fn(model, orig, s["data_sample"], steps=50, epsilon=EPS,
                 on_step=lambda st, d: snaps.__setitem__(st, d.clone()) if st in CKPTS else None)
        t_att += time.time() - t
        assert torch.equal(d50, snaps[50]) and abs(wsum(model) - w0) < 1e-6 * max(1.0, abs(w0)), "hook/weights"
        if args.smoke and k == 0:   # unit test: chạy lại KHÔNG hook -> δ phải trùng tuyệt đối
            seed_all(1000 * (k + 1))
            d_ref = fn(model, orig, s["data_sample"], steps=50, epsilon=EPS)
            hook_check = {"max_abs_diff": float((d_ref - d50).abs().max()), "frac_diff": float((d_ref != d50).float().mean())}
            assert abs(wsum(model) - w0) < 1e-6 * max(1.0, abs(w0))
        t = time.time()
        A = anns[i]
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                              for a in A], device="cuda").float()
        labels = torch.tensor([lab_of[a["category_id"]] for a in A], device="cuda")
        mine = sorted({j for (kk, _, a, b) in pairs if kk == k for j in (a, b)})
        r = {"z": {}, "s_sur": {}, "s_tgt": {key: {} for key in TGT}, "g1": {}, "g2": {}, "eff": {}}
        for tt in CKPTS:
            xa = to_adv_image(orig, snaps[tt])
            img = adv_np(orig, snaps[tt])
            with torch.no_grad():
                r["z"][tt] = gt_logits(model, xa, s["data_sample"], boxes, labels).cpu().numpy().tolist()
            r["s_sur"][tt] = _matched_scores(coco, dets_on(model, img, i, cat_ids), [i])[i].tolist()
            for key, m in tgt_models.items():
                r["s_tgt"][key][tt] = _matched_scores(coco, dets_on(m, img, i, cat_ids), [i])[i].tolist()
            if tt in GRAD_T and not args.no_grad and mine:
                x = xa.clone().requires_grad_(True)
                z = gt_logits(model, x, s["data_sample"], boxes, labels)
                g1, g2, eff = {}, {}, {}
                for j in mine:
                    g = torch.autograd.grad(z[j], x, retain_graph=True)[0]
                    g1[j], g2[j] = float(g.abs().sum()), float(g.norm())
                    if tt == 0:
                        eff[j] = float(-(g * snaps[5]).sum() / (5.0 * g.abs().sum() + 1e-12))
                r["g1"][tt], r["g2"][tt] = g1, g2
                if tt == 0:
                    r["eff"] = eff
                del z, x
        rec[i] = r
        t_diag += time.time() - t
        print(f"[e90b] {k + 1}/{len(ids)}  att {t_att / (k + 1):.1f}s/ảnh  diag {t_diag / (k + 1):.1f}s/ảnh  "
              f"tổng {time.time() - T0:.0f}s", flush=True)
    tm.update(attack=t_att, diag=t_diag, total=time.time() - T0)

    if args.smoke:
        n_obj = sum(len(o["eligible"]) for o in objs.values())
        print(f"[smoke] hook_check {hook_check}  timing {json.dumps({k: round(v, 1) for k, v in tm.items()})}  "
              f"n_eligible {n_obj}  n_pairs {len(pairs)} (KHÔNG đọc số đo)")
        return

    # ---- phân tích ----
    def drop(sc0, sct):
        return float(lgc(sc0) - lgc(max(sct, FLOOR)))

    P = []
    for (k, i, a, b) in pairs:
        r = rec[i]
        row = {"k": k, "img": i, "hi": a, "lo": b, "C_hi": objs[i]["C"][a], "C_lo": objs[i]["C"][b]}
        for tt in CKPTS[1:]:
            row[f"dz_{tt}"] = (r["z"][0][a] - r["z"][tt][a]) - (r["z"][0][b] - r["z"][tt][b])
            row[f"dsur_{tt}"] = drop(r["s_sur"][0][a], r["s_sur"][tt][a]) - drop(r["s_sur"][0][b], r["s_sur"][tt][b])
        for key in TGT:
            s0 = r["s_tgt"][key][0]
            if s0[a] >= DETECT_SCORE and s0[b] >= DETECT_SCORE:
                for tt in CKPTS[1:]:
                    st = r["s_tgt"][key][tt]
                    row[f"dtgt_{key}_{tt}"] = drop(s0[a], st[a]) - drop(s0[b], st[b])
        if not args.no_grad and r["g1"]:
            for tt in GRAD_T:
                row[f"RG_{tt}"] = float(np.log(r["g1"][tt][a] + 1e-12) - np.log(r["g1"][tt][b] + 1e-12))
                row[f"RG2_{tt}"] = float(np.log(r["g2"][tt][a] + 1e-12) - np.log(r["g2"][tt][b] + 1e-12))
            row["E"] = r["eff"][a] - r["eff"][b]
        # mô tả: tỉ phần Δz(50) đạt được tới bước 10 (hi, lo)
        for nm, j in (("hi", a), ("lo", b)):
            tot = r["z"][0][j] - r["z"][50][j]
            row[f"sat10_{nm}"] = (r["z"][0][j] - r["z"][10][j]) / tot if abs(tot) > 1e-6 else None
            row[f"z0_{nm}"] = r["z"][0][j]
        P.append(row)

    rng = np.random.default_rng(BOOT_SEED)
    img_k = sorted({p["k"] for p in P})
    boots = [rng.choice(img_k, len(img_k)) for _ in range(NB)]

    def stat(key, rows=None):
        rows = P if rows is None else rows
        vals = {kk: [p[key] for p in rows if p["k"] == kk and p.get(key) is not None] for kk in img_k}
        allv = [v for vv in vals.values() for v in vv]
        if len(allv) == 0:
            return None
        bs = []
        for b in boots:
            vv = [v for kk in b for v in vals[kk]]
            if vv:
                bs.append(np.mean(vv))
        return {"mean": round(float(np.mean(allv)), 4), "median": round(float(np.median(allv)), 4),
                "frac_neg": round(float(np.mean(np.array(allv) < 0)), 3), "n": len(allv),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}

    # dtgt gộp 2 target: mỗi (cặp, target) một giá trị
    for p in P:
        p["dtgt_pool_50"] = None
    PT = []
    for p in P:
        for key in TGT:
            if f"dtgt_{key}_50" in p:
                PT.append({"k": p["k"], "dtgt_pool_50": p[f"dtgt_{key}_50"]})
    keys = [f"dz_{t}" for t in CKPTS[1:]] + [f"dsur_{t}" for t in CKPTS[1:]] + \
           [f"dtgt_{key}_{t}" for key in TGT for t in CKPTS[1:]]
    if not args.no_grad:
        keys += [f"RG_{t}" for t in GRAD_T] + [f"RG2_{t}" for t in GRAD_T] + ["E"]
    res = {"note": "E90b PILOT chẩn đoán, devC300 first-n; không claim; target chỉ phân tích offline",
           "ids": ids, "n_images": len(ids), "n_eligible": sum(len(o["eligible"]) for o in objs.values()),
           "n_pairs": len(P), "n_pair_images": len(img_k), "pairs_per_img_cap": args.pairs, "grad": not args.no_grad,
           "timing_s": {k: round(v, 1) for k, v in tm.items()},
           "stats_hi_minus_lo": {k: stat(k) for k in keys}}
    res["stats_hi_minus_lo"]["dtgt_pool_50"] = stat("dtgt_pool_50", PT) if PT else None
    res["descr"] = {nm: {"sat10": stat(f"sat10_{nm}"), "z0": stat(f"z0_{nm}")} for nm in ("hi", "lo")}

    # ---- quy tắc khóa ----
    S = res["stats_hi_minus_lo"]
    clear_neg = lambda st: st is not None and st["ci95"][1] < 0
    enough = len(P) >= MIN_PAIRS and len(img_k) >= MIN_IMGS
    Sm, Tm = clear_neg(S["dz_50"]), clear_neg(S["dtgt_pool_50"])
    Gm = (not args.no_grad) and clear_neg(S.get("RG_0"))
    Em = (not args.no_grad) and clear_neg(S.get("E"))
    if not enough:
        case, verdict = "insufficient_pairs", "INCONCLUSIVE"
    elif Sm and Gm:
        case, verdict = "H1_gradient_starvation", "GO"
    elif Sm and Em:
        case, verdict = "H1_direction_misalignment", "GO"
    elif Sm and not args.no_grad:
        case, verdict = "surrogate_inefficiency_unexplained (grad không thấp, hướng không lệch)", "REPORT"
    elif Sm:
        case, verdict = "H1_compatible_no_grad_measured", "REPORT"
    elif Tm:
        case, verdict = "H2_transfer_limitation", "REPORT"
    else:
        case, verdict = "no_distinguishing_signal", "NO-GO"
    res["flags"] = {"enough": enough, "S-": Sm, "T-": Tm, "G-": Gm, "E-": Em}
    res["case"], res["verdict"] = case, verdict
    res["pairs"] = P
    res["raw"] = {str(i): rec[i] for i in ids}
    out = os.path.join(REPO_ROOT, "results/mechanism/e90b_traj_devC20.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"\nn_images {len(ids)}  eligible {res['n_eligible']}  pairs {len(P)} từ {len(img_k)} ảnh  timing {res['timing_s']}")
    for k2, v in S.items():
        if v:
            print(f"  {k2:24} mean {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  median {v['median']:+.3f}  "
                  f"neg {v['frac_neg']:.2f}  n {v['n']}")
    print(f"  descr {json.dumps(res['descr'])}")
    print(f"\nflags {res['flags']}\ncase {case}  ->  {verdict}\n[e90b] -> {out}")


if __name__ == "__main__":
    main()
