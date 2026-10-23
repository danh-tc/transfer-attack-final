#!/usr/bin/env python3
"""E18 — Gate A của RSE (quy tắc khóa: progress_log 2026-10-23): mức "xóa quan hệ object–nền" do CÙNG một δ gây ra trên
R50 (C4, stride 16) có đồng biến với trên Swin-L của DINO (bb3, stride 16) không, và có đi cùng việc DINO mất object không.
Chỉ forward, PNG có sẵn. Mỗi object (GT không-crowd, thứ tự = coco.getAnnIds như e4_target_side.matched):
  O = token có tâm trong box; B = token có tâm trong box nở 1.5× quanh tâm, KHÔNG nằm trong box GT nào; cần ≥ 4 token mỗi
  tập; lấy ≤ 64 token mỗi tập (RNG seed 2026 + ann_id, cùng chỉ số mọi điều kiện).
  z = LN_kênh(F) / ‖·‖; k(a, b) = exp((cos − 1) / τ), τ = trung vị (1 − cos) trên cặp token sạch O ∪ B (cố định theo object,
  dùng chung mọi điều kiện); MMD² dạng V (lệch, ≥ 0).
  r = MMD²(z_O^adv, z_B^sạch) / MMD²(z_O^sạch, z_B^sạch); Δr = 1 − r (dương = object giống nền sạch hơn).
DINO xóa object = score khớp sạch ≥ 0.3 và adv < 0.3 (như E14).
  python scripts/e18_rse_gate.py                 -> artifacts/e18/raw.json + results/mechanism/e18_rse_gate.json
  python scripts/e18_rse_gate.py --analyze-only
"""
import argparse
import glob
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import build_attack_dataset, load_image_ids
from attack.preprocess import pipeline_resize, to_batch
from scripts.e4_target_side import matched
from scripts.e14_patching import ANN_FILE, CFG, head, to_coco

ART = os.path.join(REPO_ROOT, "artifacts/runs")
SRC = {"WP2": os.path.join(ART, "dev300_B36_eps5_limit100/adv/C2-OCT-1V-WP2"),
       "XBG": os.path.join(ART, "dev300_B30_eps5_limit100/adv/C2-OCT-1V-WP2-XBG"),
       "XBGS": os.path.join(ART, "dev300_B30_eps5_limit100/adv/C2-OCT-1V-WP2-XBGS")}
DEV30_ONLY = {"XBG", "XBGS"}
RAW = os.path.join(REPO_ROOT, "artifacts/e18/raw.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/e18_rse_gate.json")
BOOT_SEED, N_BOOT, SCORE, STRIDE, LEVEL, MIN_TOK, MAX_TOK, RING, EPS = 2026, 1000, 0.3, 16, 2, 4, 64, 1.5, 5


def token_sets(boxes, h, w, dev):
    """-> list (O_idx, B_idx | None) chỉ số phẳng trên lưới [h, w] stride 16, theo thứ tự boxes (khung model, xyxy)."""
    yy = (torch.arange(h, device=dev).float() + 0.5) * STRIDE
    xx = (torch.arange(w, device=dev).float() + 0.5) * STRIDE
    inside = lambda x1, y1, x2, y2: (((yy >= y1) & (yy < y2))[:, None] & ((xx >= x1) & (xx < x2))[None]).flatten()
    any_obj = torch.zeros(h * w, dtype=torch.bool, device=dev)
    ins = []
    for b in boxes:
        m = inside(*b)
        ins.append(m)
        any_obj |= m
    out = []
    for (x1, y1, x2, y2), m in zip(boxes, ins):
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, RING * (x2 - x1) / 2, RING * (y2 - y1) / 2
        ring = inside(cx - hw, cy - hh, cx + hw, cy + hh) & ~any_obj
        O, B = m.nonzero()[:, 0], ring.nonzero()[:, 0]
        out.append((O, B) if len(O) >= MIN_TOK and len(B) >= MIN_TOK else None)
    return out


def z_tokens(f):
    """[C, h, w] -> [h·w, C]: LN theo kênh, rồi chuẩn hóa L2."""
    t = f.flatten(1).T.float()
    return F.normalize(F.layer_norm(t, t.shape[1:]), dim=1)


def mmd2(a, b, tau):
    k = lambda x, y: torch.exp((x @ y.T - 1) / tau)
    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


@torch.no_grad()
def run(coco, ids):
    from mmdet.apis import init_detector
    from attack.models import load_surrogate
    dev = "cuda:0"
    dino = init_detector(CFG, glob.glob(os.path.join(REPO_ROOT, "checkpoints", "dino-5scale_swin-l*.pth"))[0], device=dev)
    r50 = load_surrogate(dev)
    cat_ids = sorted(coco.getCatIds())
    data = build_attack_dataset("dev300")
    pos = {i: k for k, i in enumerate(data.image_ids)}
    dev30 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    conds = ["NOISE"] + list(SRC)
    dets = {c: [] for c in ["clean"] + conds}
    objs = []  # mỗi object: img_id, ann_id, area, {model: {cond: Δr}}
    t0 = time.time()
    for n, img_id in enumerate(ids):
        smp = data[pos[img_id]]
        x, ds, orig = smp["inputs"].to(dev), smp["data_sample"], smp["orig_inputs"].to(dev)
        sx, sy = ds.metainfo["scale_factor"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes = [(a["bbox"][0] * sx, a["bbox"][1] * sy, (a["bbox"][0] + a["bbox"][2]) * sx, (a["bbox"][1] + a["bbox"][3]) * sy)
                 for a in anns]
        g = torch.Generator(device=orig.device).manual_seed(BOOT_SEED + img_id)
        noise = EPS * (2 * torch.randint(0, 2, orig.shape, generator=g, device=orig.device).float() - 1)
        imgs = {"NOISE": pipeline_resize(torch.clamp(orig + noise, 0, 255), x.shape[-2:])}
        for c, root in SRC.items():
            if c in DEV30_ONLY and img_id not in dev30:
                continue
            imgs[c] = pipeline_resize(torch.from_numpy(mmcv.imread(os.path.join(root, f"{img_id}.png")))
                                      .permute(2, 0, 1).float().to(dev), x.shape[-2:])
        xb, dss = to_batch(dino, [x], [ds])
        cs = list(dino.backbone(xb))
        dets["clean"] += to_coco(head(dino, cs, dss), img_id, cat_ids)
        feats = {"swinl": {"clean": cs[LEVEL][0]}, "r50": {"clean": r50.backbone(to_batch(r50, [x], [ds])[0])[LEVEL][0]}}
        for c, a in imgs.items():
            fa = list(dino.backbone(to_batch(dino, [a], [ds])[0]))
            dets[c] += to_coco(head(dino, fa, dss), img_id, cat_ids)
            feats["swinl"][c] = fa[LEVEL][0]
            feats["r50"][c] = r50.backbone(to_batch(r50, [a], [ds])[0])[LEVEL][0]
        z = {m: {c: z_tokens(f) for c, f in fs.items()} for m, fs in feats.items()}
        sets = {m: token_sets(boxes, *feats[m]["clean"].shape[1:], dev) for m in feats}
        for j, a in enumerate(anns):
            rec = {"img_id": img_id, "j": j, "ann_id": a["id"], "area": a["area"], "dr": {}}
            for m in feats:
                if sets[m][j] is None:
                    continue
                O, B = sets[m][j]
                gs = torch.Generator().manual_seed(BOOT_SEED + a["id"])
                O = O[torch.randperm(len(O), generator=gs)[:MAX_TOK].to(dev)]
                B = B[torch.randperm(len(B), generator=gs)[:MAX_TOK].to(dev)]
                zc = z[m]["clean"]
                cos = torch.cat([zc[O], zc[B]]) @ torch.cat([zc[O], zc[B]]).T
                tau = float(torch.median(1 - cos[torch.triu_indices(len(cos), len(cos), 1).unbind()]).clamp_min(1e-4))
                base = mmd2(zc[O], zc[B], tau)
                if base <= 1e-9:
                    continue
                rec["dr"][m] = {c: 1 - mmd2(z[m][c][O], zc[B], tau) / base for c in imgs}
            objs.append(rec)
        if (n + 1) % 10 == 0:
            print(f"[E18] {n + 1}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    with open(RAW, "w") as f:
        json.dump({"dets": dets, "objs": objs}, f)


def analyze(coco, ids):
    blob = json.load(open(RAW))
    dets, objs = blob["dets"], blob["objs"]
    dev30 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    cl = matched(coco, dets["clean"], ids)
    mt = {c: matched(coco, d, ids) for c, d in dets.items() if c != "clean"}
    img_pos = {i: k for k, i in enumerate(ids)}
    rng = np.random.default_rng(BOOT_SEED)
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]

    def table(cond, sub=None):
        """Mảng theo object có Δr ở cả 2 model cho cond: img_k, dr_r50, dr_swinl, deleted (nan nếu DINO sạch không thấy)."""
        rows = []
        for o in objs:
            if sub is not None and o["img_id"] not in sub:
                continue
            if not all(m in o["dr"] and cond in o["dr"][m] for m in ("r50", "swinl")):
                continue
            sc = cl[o["img_id"]][1][o["j"]]
            dl = float(mt[cond][o["img_id"]][1][o["j"]] < SCORE) if sc >= SCORE else np.nan
            rows.append((img_pos[o["img_id"]], o["dr"]["r50"][cond], o["dr"]["swinl"][cond], dl, o["area"]))
        return np.array(rows, dtype=float).reshape(-1, 5)

    def rho(t, a, b):
        ok = ~np.isnan(t[:, b])
        return float(spearmanr(t[ok, a], t[ok, b])[0]) if ok.sum() > 5 else float("nan")

    def with_ci(t, a, b):
        point = rho(t, a, b)
        by = {k: np.where(t[:, 0] == k)[0] for k in np.unique(t[:, 0])}
        bs = []
        for s in boot:
            idx = np.concatenate([by[k] for k in s if k in by] or [np.zeros(0, int)])
            bs.append(rho(t[idx], a, b))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return {"rho": point, "ci95": [float(lo), float(hi)], "n_obj": int((~np.isnan(t[:, b])).sum())}

    out = {"n_images": len(ids), "conds": {}}
    for cond in mt:
        sub = dev30 if cond in DEV30_ONLY else None
        t = table(cond, sub)
        out["conds"][cond] = {"n_obj_both": len(t), "mean_dr_r50": float(t[:, 1].mean()), "mean_dr_swinl": float(t[:, 2].mean()),
                              "del_rate_dino": float(np.nanmean(t[:, 3])),
                              "S1_r50_vs_swinl": with_ci(t, 1, 2), "S2_swinl_vs_del": with_ci(t, 2, 3),
                              "S3_r50_vs_del": with_ci(t, 1, 3)}
    t = table("WP2")
    q = np.quantile(t[:, 4], [1 / 3, 2 / 3]) if len(t) else [0, 0]
    out["WP2_by_size"] = {name: {"S1": with_ci(t[m], 1, 2), "S2": with_ci(t[m], 2, 3)}
                          for name, m in (("small", t[:, 4] < 32 ** 2), ("medium", (t[:, 4] >= 32 ** 2) & (t[:, 4] < 96 ** 2)),
                                          ("large", t[:, 4] >= 96 ** 2))}
    new = set(ids) - dev30
    t70 = table("WP2", new)
    out["WP2_new70"] = {"S1": with_ci(t70, 1, 2), "S2": with_ci(t70, 2, 3)}
    w = out["conds"]["WP2"]
    out["verdict"] = {"S1_pass": bool(w["S1_r50_vs_swinl"]["rho"] > 0 and w["S1_r50_vs_swinl"]["ci95"][0] > 0),
                      "S2_pass": bool(w["S2_swinl_vs_del"]["rho"] > 0 and w["S2_swinl_vs_del"]["ci95"][0] > 0)}
    out["verdict"]["GATE_A"] = "QUA" if all(out["verdict"].values()) else "KHÔNG QUA"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    f3 = lambda d: f"{d['rho']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] n={d['n_obj']}"
    for c, v in out["conds"].items():
        print(f"{c:<6} n_obj {v['n_obj_both']:4d} | Δr R50 {v['mean_dr_r50']:+.3f} Swin-L {v['mean_dr_swinl']:+.3f} | "
              f"DINO xóa {v['del_rate_dino']:.3f}\n   S1 (Δr R50 ~ Δr Swin-L) {f3(v['S1_r50_vs_swinl'])}\n"
              f"   S2 (Δr Swin-L ~ xóa) {f3(v['S2_swinl_vs_del'])}\n   S3 (Δr R50 ~ xóa) {f3(v['S3_r50_vs_del'])}")
    for k, v in out["WP2_by_size"].items():
        print(f"WP2 {k:<7} S1 {f3(v['S1'])} | S2 {f3(v['S2'])}")
    print(f"WP2 70 ảnh mới  S1 {f3(out['WP2_new70']['S1'])} | S2 {f3(out['WP2_new70']['S2'])}")
    print("VERDICT", out["verdict"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    coco = COCO(ANN_FILE)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    ids = ids[:args.limit] if args.limit else ids
    if not args.analyze_only:
        run(coco, ids)
    analyze(coco, ids)


if __name__ == "__main__":
    main()
