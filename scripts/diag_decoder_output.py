#!/usr/bin/env python3
"""Đầu ra decoder DINO-Swin-L: C2-PIA phá Swin-L qua phân loại hay định vị? (quy tắc khóa: progress_log 2026-10-04
"Phân tích đầu ra decoder DINO-Swin-L"). Chỉ forward, PNG có sẵn, dev30.

Primary "forced": decoder trên adv dùng ĐÚNG tập + thứ tự token encoder chọn trên ảnh sạch (slot k = cùng token,
cùng embedding nội dung). Query q*_i chọn trên ảnh sạch (argmax IoU, IoU ≥ 0.5). Chỉ số lớp cuối, logit thô:
s = z_{q*,c}, margin m = z_{q*,c} − max_{c'≠c} z, rank percentile r trong 900 query theo lớp c, I = IoU(box q*, GT).
Secondary "natural": token bị loại khỏi top-900, gán lại (s_best), phân rã lỗi, FP/ảnh.

  python scripts/diag_decoder_output.py   -> results/ablation/mech/decoder_output.json
"""
import argparse
import glob
import json
import os
import sys

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.evaluation import bootstrap_indices
from attack.preprocess import pipeline_resize, to_batch

ART = os.path.join(REPO_ROOT, "artifacts/runs/dev300_B50_eps5")
CFG = os.path.join(REPO_ROOT, "third_party/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py")
C2 = "PROBE-osfd+cls"
CONDS = ["OSFD", "GEO-C0", "PIA", "PIA-aff", C2, "C2-PIA"]
PAIRS = {"P1": ("C2-PIA", C2), "P2": ("PIA", "GEO-C0")}
SCORE = 0.3
BOOT_SEED, N_BOOT = 2026, 1000


@torch.no_grad()
def decode(model, pix, ds, forced_idx=None):
    """-> (z [Q, C] logit lớp cuối, boxes [Q,4] xyxy khung model, topk token idx [Q]). Tái tạo DINO.pre_decoder
    (chế độ test); forced_idx: dùng tập token này thay vì top-k của chính ảnh."""
    x, dss = to_batch(model, [pix], [ds])
    enc_in, dec_in = model.pre_transformer(model.neck(model.backbone(x)), dss)
    enc = model.forward_encoder(**enc_in)
    memory = enc["memory"]
    out_mem, props = model.gen_encoder_output_proposals(memory, enc["memory_mask"], enc["spatial_shapes"])
    L = model.decoder.num_layers
    cls_e = model.bbox_head.cls_branches[L](out_mem)
    coord_e = model.bbox_head.reg_branches[L](out_mem) + props
    idx = torch.topk(cls_e.max(-1)[0], k=model.num_queries, dim=1)[1] if forced_idx is None else forced_idx[None]
    ref = torch.gather(coord_e, 1, idx.unsqueeze(-1).repeat(1, 1, 4)).detach().sigmoid()
    query = model.query_embedding.weight[None]
    dec_in.update(query=query, memory=memory, reference_points=ref, dn_mask=None)
    dec = model.forward_decoder(**dec_in)
    cls, coords = model.bbox_head(dec["hidden_states"], dec["references"])
    z, c = cls[-1][0], coords[-1][0]
    h, w = dss[0].img_shape
    boxes = torch.stack([(c[:, 0] - c[:, 2] / 2) * w, (c[:, 1] - c[:, 3] / 2) * h,
                         (c[:, 0] + c[:, 2] / 2) * w, (c[:, 1] + c[:, 3] / 2) * h], 1)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w); boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return z, boxes, idx[0], (x, dss)


def iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda t: (t[:, 2:] - t[:, :2]).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


MAX_PER_IMG = 300  # = test_cfg.max_per_img của DINO (gán lại từ model trong main)


def top100(z, boxes):
    """Detection như DINO predict: top-MAX_PER_IMG theo sigmoid trên Q × C."""
    sc, ind = z.sigmoid().flatten().topk(MAX_PER_IMG)
    return boxes[ind // z.shape[1]], ind % z.shape[1], sc


def query_metrics(z, boxes, q, c, gt):
    """Mỗi GT: (s, margin, rank percentile, IoU) tại query q (tensor [G])."""
    zq = z[q]                                                     # [G, C]
    s = zq.gather(1, c[:, None])[:, 0]
    other = zq.clone(); other.scatter_(1, c[:, None], -float("inf"))
    m = s - other.max(1)[0]
    R = (z[:, c].T > s[:, None]).sum(1).float()                   # số query điểm lớp c cao hơn -> hạng − 1
    r = R / (z.shape[0] - 1)
    I = iou(boxes[q], gt).diagonal()
    return torch.stack([s, m, r, I], 1)


def error_decomp(z, boxes, gt, c, det_clean):
    """Phân rã trên GT phát hiện được ở ảnh sạch. -> list nhãn mỗi GT (None nếu không phát hiện ở sạch)."""
    b, l, sc = top100(z, boxes)
    M = iou(b, gt)                                                # [100, G]
    pres = ((M >= 0.5) & (l[:, None] == c[None]) & (sc[:, None] >= SCORE)).any(0)
    Mq = iou(boxes, gt)                                           # [Q, G]
    zc = z.sigmoid()[:, c]                                        # [Q, G]
    out = []
    for g in range(len(gt)):
        if not det_clean[g]:
            out.append(None)
        elif pres[g]:
            out.append("preserved")
        elif (Mq[:, g] >= 0.5).any():
            loc = Mq[:, g] >= 0.5
            p = z.sigmoid()[loc]
            wrong = bool(((p.max(1)[0] >= SCORE) & (p.argmax(1) != c[g])).any())
            out.append("conf_wrong_class" if wrong else "conf_rank")
        elif ((zc[:, g] >= SCORE) & (Mq[:, g] >= 0.1)).any():
            out.append("loc")
        else:
            out.append("lost")
    return out, pres


def fp_stats(z, boxes, gt, c):
    b, l, sc = top100(z, boxes)
    keep = sc >= SCORE
    b, l, sc = b[keep], l[keep], sc[keep]
    res = {"background": 0, "localization": 0, "class_confusion": 0, "other": 0}
    scores = []
    if len(b) == 0:
        return res, scores
    M = iou(b, gt) if len(gt) else torch.zeros(len(b), 0, device=b.device)
    same = (l[:, None] == c[None]) if len(gt) else torch.zeros(len(b), 0, dtype=torch.bool, device=b.device)
    for k in range(len(b)):
        if len(gt) and ((M[k] >= 0.5) & same[k]).any():
            continue
        scores.append(float(sc[k]))
        if len(gt) == 0 or M[k].max() < 0.1:
            res["background"] += 1
        elif (same[k] & (M[k] >= 0.1)).any():
            res["localization"] += 1
        elif ((~same[k]) & (M[k] >= 0.5)).any():
            res["class_confusion"] += 1
        else:
            res["other"] += 1
    return res, scores


def wmean(vals, W):
    """vals: list ảnh -> array [n_gt_i, K]; trung bình theo GT có trọng số ảnh. -> [1 + n_boot, K]."""
    K = next(v.shape[1] for v in vals if len(v))
    S = np.stack([v.sum(0) if len(v) else np.zeros(K) for v in vals])   # [n, K]
    N = np.array([len(v) for v in vals], dtype=float)
    return (W @ S) / (W @ N)[:, None]


def ci(x):
    return {"point": float(x[0]), "ci95": [float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["iou", "maxlogit"], default="iou",
                    help="maxlogit = dự phòng khai báo trước (trung vị score sạch của q* < 0.3)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ để thử sanity")
    args = ap.parse_args()
    dev = "cuda:0"
    from mmdet.apis import init_detector
    model = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    global MAX_PER_IMG
    MAX_PER_IMG = model.bbox_head.test_cfg.get("max_per_img", 100)
    data = build_attack_dataset("dev300")
    pos = {img_id: i for i, img_id in enumerate(data.image_ids)}
    img_ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv"))[:args.limit]

    forced = {c: [] for c in CONDS}           # mỗi ảnh: [G_sel, 4] Δ(s, m, r, I)
    clean_sel = []                            # mỗi ảnh: [G_sel, 4] giá trị sạch
    natural = {c: [] for c in CONDS}          # mỗi ảnh: [G_sel, 6] (deselected, Δm, Δr, ΔI | tracked; Δs_best, IoU_best)
    errs = {c: [] for c in ["clean"] + CONDS}
    fps = {c: [] for c in ["clean"] + CONDS}
    sanity = {"forced_eq_natural_clean_maxabs": 0.0, "predict_score_maxabs": 0.0, "predict_box_maxabs": 0.0,
              "n_gt": 0, "n_gt_sel": 0, "clean_sel_score_sigmoid": []}
    for j, img_id in enumerate(img_ids):
        s0 = data[pos[img_id]]
        x, ds = s0["inputs"].to(dev), s0["data_sample"]
        gi = ds.gt_instances
        gtb = gi.bboxes.tensor if hasattr(gi.bboxes, "tensor") else gi.bboxes
        keepg = ~gi.ignore_flags.bool() if "ignore_flags" in gi else torch.ones(len(gtb), dtype=torch.bool)
        gt = gtb[keepg].float().to(dev)
        gc = gi.labels[keepg].to(dev)

        z0, b0, idx0, (xb, dss) = decode(model, x, ds)
        zf, _, _, _ = decode(model, x, ds, forced_idx=idx0)
        sanity["forced_eq_natural_clean_maxabs"] = max(sanity["forced_eq_natural_clean_maxabs"], float((zf - z0).abs().max()))
        if j < 3:
            with torch.no_grad():
                pred = model.predict(xb, dss, rescale=False)[0].pred_instances
            pb, _, ps = top100(z0, b0)
            sanity["predict_score_maxabs"] = max(sanity["predict_score_maxabs"], float((pred.scores - ps).abs().max()))
            sanity["predict_box_maxabs"] = max(sanity["predict_box_maxabs"], float((pred.bboxes - pb).abs().max()))

        Mq = iou(b0, gt) if len(gt) else torch.zeros(len(b0), 0, device=dev)
        sanity["n_gt"] += len(gt)
        if len(gt):
            best_iou, q = Mq.max(0)
            sel = best_iou >= 0.5
            if args.select == "maxlogit":
                zc = torch.where(Mq >= 0.5, z0[:, gc], torch.full_like(Mq, -float("inf")))
                q = zc.argmax(0)
        else:
            q = torch.zeros(0, dtype=torch.long, device=dev); sel = torch.zeros(0, dtype=torch.bool, device=dev)
        q, gs, cs = q[sel], gt[sel], gc[sel]
        sanity["n_gt_sel"] += int(sel.sum())
        mc = query_metrics(z0, b0, q, cs, gs) if len(q) else torch.zeros(0, 4, device=dev)
        sanity["clean_sel_score_sigmoid"] += mc[:, 0].sigmoid().tolist()
        clean_sel.append(mc.cpu().numpy())
        tok = idx0[q]

        e0, det_clean = error_decomp(z0, b0, gt, gc, torch.ones(len(gt), dtype=torch.bool))
        errs["clean"].append(e0)
        fps["clean"].append(fp_stats(z0, b0, gt, gc))
        for cond in CONDS:
            a = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(ART, "adv", cond, f"{img_id}.png")))
                                .permute(2, 0, 1).float().to(dev), x.shape[-2:])
            za, ba, _, _ = decode(model, a, ds, forced_idx=idx0)
            ma = query_metrics(za, ba, q, cs, gs) if len(q) else torch.zeros(0, 4, device=dev)
            forced[cond].append((ma - mc).cpu().numpy())
            zn, bn, idxn, _ = decode(model, a, ds)
            rows = []
            for g in range(len(q)):
                hit = (idxn == tok[g]).nonzero()
                if len(hit):
                    mt = query_metrics(zn, bn, hit[0], cs[g:g + 1], gs[g:g + 1])[0]
                    d = [0.0, float(mt[1] - mc[g, 1]), float(mt[2] - mc[g, 2]), float(mt[3] - mc[g, 3])]
                else:
                    d = [1.0, np.nan, np.nan, np.nan]
                zb = zn[:, cs[g]]
                qb = int(zb.argmax())
                d += [float(zb[qb] - z0[:, cs[g]].max()), float(iou(bn[qb:qb + 1], gs[g:g + 1])[0, 0])]
                rows.append(d)
            natural[cond].append(np.array(rows).reshape(-1, 6))
            errs[cond].append(error_decomp(zn, bn, gt, gc, det_clean)[0])
            fps[cond].append(fp_stats(zn, bn, gt, gc))
        if (j + 1) % 10 == 0:
            print(f"[decoder] {j + 1}/{len(img_ids)}", flush=True)

    n = len(img_ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, N_BOOT, BOOT_SEED)])
    sc_sel = np.array(sanity.pop("clean_sel_score_sigmoid"))
    sanity.update({"frac_gt_sel": sanity["n_gt_sel"] / sanity["n_gt"], "clean_sel_score_median": float(np.median(sc_sel)),
                   "clean_sel_score_frac_ge_0.3": float(np.mean(sc_sel >= SCORE))})
    sanity["fallback_triggered"] = sanity["clean_sel_score_median"] < SCORE
    names = ["d_s", "d_margin", "d_rank_pct", "d_iou"]
    out = {"select": args.select, "n_images": n, "conds": CONDS, "pairs": {k: list(v) for k, v in PAIRS.items()}, "sanity": sanity,
           "clean_selected": {k: float(v) for k, v in zip(["s", "margin", "rank_pct", "iou"], wmean(clean_sel, W)[0])},
           "forced": {}, "natural": {}, "error_decomp": {}, "fp_per_img": {}, "pair_tests": {}, "verdict": {}}
    for cond in CONDS:
        wm = wmean(forced[cond], W)
        out["forced"][cond] = {k: ci(wm[:, i]) for i, k in enumerate(names)}
        allr = np.concatenate(natural[cond])
        tr = allr[allr[:, 0] == 0]
        out["natural"][cond] = {"frac_deselected": float(allr[:, 0].mean()),
                                "tracked_d_margin": float(np.mean(tr[:, 1])), "tracked_d_rank_pct": float(np.mean(tr[:, 2])),
                                "tracked_d_iou": float(np.mean(tr[:, 3])), "d_s_best": float(np.mean(allr[:, 4])),
                                "iou_best": float(np.mean(allr[:, 5]))}
    for cond in ["clean"] + CONDS:
        labs = [l for e in errs[cond] for l in e if l is not None]
        out["error_decomp"][cond] = {k: labs.count(k) / max(1, len(labs)) for k in
                                     ("preserved", "conf_rank", "conf_wrong_class", "loc", "lost")}
        out["error_decomp"][cond]["n_gt_detected_clean"] = len(labs)
        tot = {k: sum(f[0][k] for f in fps[cond]) / n for k in ("background", "localization", "class_confusion", "other")}
        s_all = [s for f in fps[cond] for s in f[1]]
        out["fp_per_img"][cond] = dict(tot, total=sum(tot.values()),
                                       fp_score_median=float(np.median(s_all)) if s_all else None)
    for p, (a, b) in PAIRS.items():
        diff = [fa - fb for fa, fb in zip(forced[a], forced[b])]
        wm = wmean(diff, W)
        t = {k: ci(wm[:, i]) for i, k in enumerate(names)}
        out["pair_tests"][p] = t
        out["verdict"][p] = {"CLS": bool(t["d_margin"]["ci95"][1] < 0 and t["d_rank_pct"]["ci95"][0] > 0),
                             "LOC": bool(t["d_iou"]["ci95"][1] < 0)}
    v1, v2 = out["verdict"]["P1"], out["verdict"]["P2"]
    out["verdict"]["conclusion"] = (
        "P1 CLS + P2 CLS -> hình học PIA tự tạo bất ổn phân loại query" if v1["CLS"] and v2["CLS"] else
        "P1 CLS, P2 không -> tương tác loss × hình học" if v1["CLS"] else
        "P1 không CLS nhưng LOC -> lợi ở định vị / gán query" if v1["LOC"] else
        "P1 không CLS, không LOC" + (" (P2 CLS)" if v2["CLS"] else " (P2 LOC)" if v2["LOC"] else
                                     "; P2 cũng không -> giả thuyết mức query bị bác"))

    path = os.path.join(REPO_ROOT, "results/ablation/mech", "decoder_output.json" if args.limit is None else "decoder_output_try.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SANITY", json.dumps(sanity, ensure_ascii=False))
    print("clean q*:", {k: round(v, 3) for k, v in out["clean_selected"].items()})
    print(f"\n{'forced':<16}" + "".join(f"{k:>22}" for k in names))
    for cond in CONDS:
        print(f"{cond:<16}" + "".join(f"{v['point']:+8.3f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" for v in out["forced"][cond].values()))
    print(f"\n{'natural':<16}" + "".join(f"{k:>20}" for k in out["natural"][CONDS[0]]))
    for cond in CONDS:
        print(f"{cond:<16}" + "".join(f"{v:20.3f}" for v in out["natural"][cond].values()))
    print(f"\n{'lỗi':<16}" + "".join(f"{k:>17}" for k in out["error_decomp"]["clean"]))
    for cond in ["clean"] + CONDS:
        print(f"{cond:<16}" + "".join(f"{v:17.3f}" for v in out["error_decomp"][cond].values()))
    print(f"\n{'FP/ảnh':<16}" + "".join(f"{k:>16}" for k in out["fp_per_img"]["clean"]))
    for cond in ["clean"] + CONDS:
        print(f"{cond:<16}" + "".join(f"{(v if v is not None else float('nan')):16.3f}" for v in out["fp_per_img"][cond].values()))
    for p, t in out["pair_tests"].items():
        print(f"\n{p} {PAIRS[p][0]} − {PAIRS[p][1]}: " + "  ".join(
            f"{k} {v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in t.items()), out["verdict"][p])
    print("\nKẾT LUẬN:", out["verdict"]["conclusion"])


if __name__ == "__main__":
    main()
