"""S6 phan tich vung con theo object / class (progress_log KHOA S6, quy tac 1-4). Mac dinh dev30.
Theo object (instance khong-crowd, >= 30 pixel, clean-TP tren DINO-Swin-L), g_t cache:
  A[o,c,X]  = TB t in {0.5,1} cua Sum_X sign(delta_c)*g / Sum_X |g|   (c = 6 o attack)
  dens[o,X] = (Sum_X |g_clean| / Sum_MASK |g_clean|) / (|X| / |MASK|)
  harm[o,c] = 1 - score_adv/score_clean
  rho[o,X]  = Spearman qua 6 o cua (A[o,c,X], harm[o,c])
WB (5 seed): dong gop vung X = TB_o cua (Sum_X sign*g / Sum_MASK |g|) (TB t). CI bootstrap 1000 theo ANH, seed 2026.
  python scripts/s6_analysis.py [--list dev30]
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.s3_region import ADV, ANN, GDIR, IMGDIR
from scripts.s5_dev30 import match_score
from scripts.s6_common import BANDS, PARTS, VERT, inst_subregions

ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
WBS = ["WB-0", "WB-r1", "WB-r2", "WB-r3", "WB-r4"]
TS = ("0.5", "1.0")
NBOOT, SEED = 1000, 2026


def boot_img(vals, imgs, rng, n=NBOOT):
    """TB co CI bootstrap theo anh. vals/imgs: list song song (gia tri theo object, anh chua object)."""
    vals = np.asarray(vals, float); imgs = np.asarray(imgs)
    ok = np.isfinite(vals); vals, imgs = vals[ok], imgs[ok]
    if len(vals) == 0:
        return float("nan"), [float("nan"), float("nan")]
    u = np.unique(imgs); by = {i: vals[imgs == i] for i in u}
    bs = [np.concatenate([by[i] for i in rng.choice(u, len(u))]).mean() for _ in range(n)]
    return float(vals.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def diff_ci(va, vb, imgs, rng, n=NBOOT):
    """CI paired cua hieu (va - vb) TB, bootstrap theo anh."""
    va, vb, imgs = np.asarray(va, float), np.asarray(vb, float), np.asarray(imgs)
    ok = np.isfinite(va) & np.isfinite(vb); va, vb, imgs = va[ok], vb[ok], imgs[ok]
    if len(va) == 0:
        return float("nan"), [float("nan"), float("nan")]
    u = np.unique(imgs); ba = {i: va[imgs == i] for i in u}; bb = {i: vb[imgs == i] for i in u}
    bs = []
    for _ in range(n):
        s = rng.choice(u, len(u))
        bs.append(np.concatenate([ba[i] for i in s]).mean() - np.concatenate([bb[i] for i in s]).mean())
    return float((va - vb).mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--list", default="dev30"); args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, f"data/image_lists/{args.list}.csv"))
    out_json = os.path.join(REPO_ROOT, f"results/mechanism/s6_analysis_{args.list}.json")
    coco = COCO(ANN); cat = sorted(coco.getCatIds()); cname = {c["id"]: c["name"] for c in coco.loadCats(cat)}
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}

    # ---- harm theo object (DINO-Swin-L) ----
    from mmdet.apis import inference_detector
    model = load("dino_swin_l"); objs = []
    with torch.no_grad():
        for i in ids:
            d0 = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat)
            dc = {c: to_coco_dets(inference_detector(model, mmcv.imread(os.path.join(ADV, c, f"{i}.png"))), i, cat)
                  for c in ATT}
            h, w = clean[i].shape[:2]
            for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
                s0 = match_score(a["bbox"], a["category_id"], d0)
                if s0 < 0.3:
                    continue
                R = inst_subregions(coco, a, h, w)
                if R is None:
                    continue
                objs.append({"img": i, "cls": cname[a["category_id"]], "area": float(a["area"]), "R": R,
                             "harm": {c: 1 - match_score(a["bbox"], a["category_id"], dc[c]) / s0 for c in ATT}})
    del model; torch.cuda.empty_cache()
    print(f"[s6] {len(objs)} object clean-TP ({args.list})", flush=True)

    # ---- sign cache + g_t ----
    SD = {c: {i: np.sign(mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
              for i in ids} for c in ATT + WBS}
    G = {}
    for i in ids:
        G[(i, "clean")] = np.load(os.path.join(GDIR, f"{i}_clean.npy")).astype(np.float32)
        for c in ATT + WBS:
            for t in TS:
                G[(i, c, t)] = np.load(os.path.join(GDIR, f"{i}_{c}_t{t}.npy")).astype(np.float32)
    SUBS = ["MASK"] + BANDS + VERT + PARTS

    def A_val(o, c, X):
        if X not in o["R"]:
            return np.nan
        i = o["img"]; m = np.broadcast_to(o["R"][X][None], SD[c][i].shape)
        vs = []
        for t in TS:
            g = G[(i, c, t)]
            den = np.abs(g)[m].sum()
            vs.append((SD[c][i] * g)[m].sum() / den if den > 0 else np.nan)
        return float(np.mean(vs))

    for o in objs:
        o["A"] = {c: {X: A_val(o, c, X) for X in SUBS} for c in ATT}
        g = np.abs(G[(o["img"], "clean")])
        mass_mask = g[np.broadcast_to(o["R"]["MASK"][None], g.shape)].sum()
        o["dens"] = {}
        for X in SUBS:
            if X in o["R"]:
                mx = np.broadcast_to(o["R"][X][None], g.shape)
                area = o["R"][X].sum() / o["R"]["MASK"].sum()
                o["dens"][X] = float(g[mx].sum() / mass_mask / area) if mass_mask > 0 and area > 0 else np.nan
        o["rho"] = {}
        for X in SUBS:
            av = [o["A"][c][X] for c in ATT]; hv = [o["harm"][c] for c in ATT]
            o["rho"][X] = spearmanr(av, hv).correlation if np.all(np.isfinite(av)) and np.std(hv) > 0 else np.nan

    rng = np.random.default_rng(SEED)
    res = {"list": args.list, "n_obj": len(objs), "n_img": len(set(o["img"] for o in objs))}

    # ---- quy tac 1: rho theo vung con ----
    def gather(key_fn, sub):
        return [key_fn(o) for o in sub], [o["img"] for o in sub]
    res["rule1_rho_subregion"] = {}
    print("\n[1] rho(A,harm) TB qua object theo vung con [CI95 boot anh]:")
    for X in SUBS:
        v, im = gather(lambda o, X=X: o["rho"].get(X, np.nan), objs)
        pt, ci = boot_img(v, im, rng)
        res["rule1_rho_subregion"][X] = {"mean": pt, "ci95": ci, "n": int(np.isfinite(v).sum())}
        print(f"   {X:6} {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  (n={int(np.isfinite(v).sum())})")
    for name, P in (("BANDS", BANDS), ("VERT", VERT)):
        order = sorted(P, key=lambda X: -res["rule1_rho_subregion"][X]["mean"])
        top, sec = order[0], order[1]
        dv_a, dv_b, im = [], [], []
        for o in objs:
            dv_a.append(o["rho"].get(top, np.nan)); dv_b.append(o["rho"].get(sec, np.nan)); im.append(o["img"])
        pt, ci = diff_ci(dv_a, dv_b, im, rng)
        v = (f"{top} > {sec}" if ci[0] > 0 else f"{sec} > {top}" if ci[1] < 0 else "khong phan biet")
        res["rule1_rho_subregion"][f"{name}_top_diff"] = {"top": top, "sec": sec, "diff": pt, "ci95": ci, "verdict": v}
        print(f"   {name}: {top}-{sec} = {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -> {v}")

    # ---- quy tac 2: mat do |g_t| theo vung con ----
    res["rule2_density"] = {}
    print("\n[2] mat do |g_t clean| TB theo vung con (>1 = nhay hon trung binh mask):")
    for X in BANDS + VERT + PARTS:
        v, im = gather(lambda o, X=X: o["dens"].get(X, np.nan), objs)
        pt, ci = boot_img(v, im, rng)
        res["rule2_density"][X] = {"mean": pt, "ci95": ci, "n": int(np.isfinite(v).sum())}
        print(f"   {X:6} {pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  (n={int(np.isfinite(v).sum())})")
    for name, P in (("BANDS", BANDS), ("VERT", VERT)):
        order = sorted(P, key=lambda X: -res["rule2_density"][X]["mean"]); top, sec = order[0], order[1]
        va, vb, im = [], [], []
        for o in objs:
            va.append(o["dens"].get(top, np.nan)); vb.append(o["dens"].get(sec, np.nan)); im.append(o["img"])
        pt, ci = diff_ci(va, vb, im, rng)
        v = (f"{top} > {sec}" if ci[0] > 0 else f"{sec} > {top}" if ci[1] < 0 else "khong phan biet")
        res["rule2_density"][f"{name}_top_diff"] = {"top": top, "sec": sec, "diff": pt, "ci95": ci, "verdict": v}
        print(f"   {name}: {top}-{sec} = {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -> {v}")

    # dong gop WB theo vung con (vung, 5 seed) - "vung vung qua nhieu ban do dau"
    res["rule2_wb_contrib"] = {}
    print("\n[2b] dong gop dau WB (Sum sign*g_t / Sum|g_t| trong mask) theo vung con, 5 seed:")
    for w in WBS:
        row = {}
        for X in BANDS + VERT:
            num = den = 0.0
            for o in objs:
                if X not in o["R"]:
                    continue
                i = o["img"]; mx = np.broadcast_to(o["R"][X][None], SD[w][i].shape)
                mm = np.broadcast_to(o["R"]["MASK"][None], SD[w][i].shape)
                for t in TS:
                    g = G[(i, w, t)]
                    num += (SD[w][i] * g)[mx].sum(); den += np.abs(g)[mm].sum()
            row[X] = float(num / den) if den > 0 else np.nan
        res["rule2_wb_contrib"][w] = row
        print(f"   {w:7} " + "  ".join(f"{X} {row[X]:+.3f}" for X in BANDS + VERT))

    # ---- quy tac 3: theo class ----
    from collections import Counter
    cnt = Counter(o["cls"] for o in objs)
    imc = {cl: len(set(o["img"] for o in objs if o["cls"] == cl)) for cl in cnt}
    classes = [cl for cl in cnt if cnt[cl] >= 8 and imc[cl] >= 3]
    res["rule3_class"] = {"counts": dict(cnt), "classes_reported": classes}
    print(f"\n[3] class du (>=8 obj, >=3 anh): {[(cl, cnt[cl]) for cl in classes]}")
    for cl in classes:
        sub = [o for o in objs if o["cls"] == cl]
        prof = {}
        for X in BANDS + VERT:
            v, im = gather(lambda o, X=X: o["dens"].get(X, np.nan), sub)
            prof[X] = boot_img(v, im, rng)[0]
        res["rule3_class"][cl] = {"n": cnt[cl], "density": prof}
        print(f"   {cl:12} mat do " + "  ".join(f"{X} {prof[X]:.2f}" for X in BANDS + VERT))
    # claim person vs phan con lai (hieu mat do EDGE), CI boot anh
    if "person" in classes:
        pe = [o["dens"].get("EDGE", np.nan) for o in objs if o["cls"] == "person"]
        pim = [o["img"] for o in objs if o["cls"] == "person"]
        oe = [o["dens"].get("EDGE", np.nan) for o in objs if o["cls"] != "person"]
        oim = [o["img"] for o in objs if o["cls"] != "person"]
        pt_p, _ = boot_img(pe, pim, rng); pt_o, _ = boot_img(oe, oim, rng)
        res["rule3_class"]["person_vs_rest_EDGE"] = {"person": pt_p, "rest": pt_o, "n_rest": int(np.isfinite(oe).sum())}
        print(f"   person EDGE {pt_p:.2f} vs phan con lai {pt_o:.2f} (n_rest={int(np.isfinite(oe).sum())}; mo ta)")

    # ---- quy tac 4: PARTS person ----
    persons = [o for o in objs if "HEAD" in o["R"] or "TORSO" in o["R"]]
    res["rule4_parts"] = {"n_person_with_parts": len(persons)}
    if persons:
        print(f"\n[4] PARTS person ({len(persons)} object co keypoint):")
        for X in PARTS:
            dv, im = gather(lambda o, X=X: o["dens"].get(X, np.nan), persons)
            rv, _ = gather(lambda o, X=X: o["rho"].get(X, np.nan), persons)
            pt, ci = boot_img(dv, im, rng)
            res["rule4_parts"][X] = {"density": pt, "density_ci95": ci,
                                     "rho_mean": float(np.nanmean(rv)), "n": int(np.isfinite(dv).sum())}
            print(f"   {X:6} mat do {pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  rho {np.nanmean(rv):+.2f}  (n={int(np.isfinite(dv).sum())})")
        order = sorted([X for X in PARTS if res["rule4_parts"].get(X)], key=lambda X: -res["rule4_parts"][X]["density"])
        if len(order) >= 2:
            top, sec = order[0], order[1]
            va = [o["dens"].get(top, np.nan) for o in persons]; vb = [o["dens"].get(sec, np.nan) for o in persons]
            pt, ci = diff_ci(va, vb, [o["img"] for o in persons], rng)
            v = (f"{top} > {sec}" if ci[0] > 0 else f"{sec} > {top}" if ci[1] < 0 else "khong phan biet")
            res["rule4_parts"]["top_diff"] = {"top": top, "sec": sec, "diff": pt, "ci95": ci, "verdict": v}
            print(f"   top: {top}-{sec} = {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -> {v}")

    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"\n[s6 analysis] -> {out_json}")


if __name__ == "__main__":
    main()
