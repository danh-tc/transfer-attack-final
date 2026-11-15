"""ENS-WP α=2 (progress_log 2026-11-15): kiểm step size α=2 có tổng quát sang ENS-WP không + gap ENS − RD-CNX ở α đã tune.
devC[140:260] (cùng 120 ảnh, cùng 3090 với a_alpha.py). Ô sinh: ENS-WP B36 α=1 (= e88.ens_wp_attack) và α=2 (chỉ đổi α,
ở cả hai surrogate). RD-CNX B50 α=1 / α=2: DÙNG LẠI dets đã lưu của a_alpha (không sinh lại); AP sạch = dets sạch của a_alpha.
  python scripts/a_ens_alpha.py --check    # ENS α=1 truyền tường minh ≡ ens_wp_attack (1 ảnh)
  python scripts/a_ens_alpha.py
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.pia import geo_osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e81_ens_alloc_rho import COMMON, kw_for
from scripts.e88_ens_baselines import R, C, RHO, ens_wp_attack, seed_all
from scripts.f1b_r1a import ANN, ART, IMGDIR, BOOT_SEED, PANEL, GROUPS, save_png
from scripts.a_alpha import ids_devC, png

RUN = "a_alpha_devC_eps5_3090"
GEN = {"ENS-WP-B36": 1.0, "ENS-WP-A2-B36": 2.0}
NEW = {"ENS1": "ENS-WP-B36", "ENS2": "ENS-WP-A2-B36"}
OLD = ["B50", "A2"]                                 # từ dets a_alpha
OUT = os.path.join(REPO_ROOT, "results/mechanism/a_ens_alpha_devC.json")


def ens(models, ds, i, n, alpha):
    orig = ds[R][i]["orig_inputs"].cuda().float()
    delta, mom = None, None
    for b in range(36 // 2):
        key = (R, C)[b % 2]
        seed_all(1000 * (n + 1) + b)
        delta, mom = geo_osfd_attack(models[key], orig, ds[key][i]["data_sample"], steps=2, init_noise=delta,
                                     mom_in=mom, return_state=True, alpha=alpha, **kw_for(key, RHO), **COMMON)
    return orig, delta


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ids = ids_devC()
    assert torch.cuda.get_device_name(0).endswith("3090")
    models = {k: load_model(k, "cuda:0") for k in (R, C)}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in (R, C)}
    if args.check:
        i = ids[0]
        o, a = ens_wp_attack(models, ds, i, 0)
        _, b = ens(models, ds, i, 0, 1.0)
        _, c = ens_wp_attack(models, ds, i, 0)
        q = lambda d: torch.round(to_adv_image(o, d))
        r = {"img_id": i, "frac_diff_explicit_vs_ref": float((q(a) != q(b)).float().mean()),
             "frac_diff_ref_vs_ref": float((q(a) != q(c)).float().mean())}
        ok = r["frac_diff_explicit_vs_ref"] <= max(1.5 * r["frac_diff_ref_vs_ref"], 1e-3)
        print(r, "PASS" if ok else "FAIL", flush=True)
        json.dump({"pass": ok, **r}, open(os.path.join(REPO_ROOT, "results/mechanism/a_ens_alpha_check.json"), "w"), indent=1)
        return
    tpath = os.path.join(REPO_ROOT, "results/mechanism/a_ens_alpha_timing.json")
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for cell, alpha in GEN.items():
        t_all = timing.setdefault(cell, {})
        for n, i in enumerate(ids):
            p = png(RUN, cell, "final", i)
            if os.path.exists(p):
                continue
            torch.cuda.synchronize(); t0 = time.time()
            orig, delta = ens(models, ds, i, n, alpha)
            torch.cuda.synchronize(); t_all[str(i)] = time.time() - t0
            save_png(orig, delta, p)
            print(f"[ens-α {cell}] {n + 1}/{len(ids)}  {np.mean(list(t_all.values())):.2f}s/ảnh", flush=True)
        json.dump(timing, open(tpath, "w"), indent=1)
    del models
    torch.cuda.empty_cache()

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    nimg = len(ids)
    W = np.vstack([np.ones((1, nimg)), bootstrap_indices(nimg, 1000, BOOT_SEED)])
    npz = OUT.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
    from mmdet.apis import inference_detector
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in [*OLD, *NEW]):
            continue
        model = load(key)
        old = json.load(gzip.open(os.path.join(ART, RUN, "dets", key, "all.json.gz"), "rt"))
        dets = {c: [] for c in NEW}
        with torch.no_grad():
            for i in ids:
                for c, g in NEW.items():
                    dets[c] += to_coco_dets(inference_detector(model, png(RUN, g, "final", i)), i, cat_ids)
        with gzip.open(os.path.join(ART, RUN, "dets", key, "ens_alpha.json.gz"), "wt") as f:
            json.dump(dets, f)
        clean = np.array([CocoImageEval(coco, old["clean"], ids).ap(w)[0] for w in W]) * 100
        for c in [*OLD, *NEW]:
            ev = CocoImageEval(coco, old[c] if c in OLD else dets[c], ids)
            S[f"{key}|{c}"] = 100 * (clean - np.array([ev.ap(w)[0] for w in W]) * 100) / clean
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in [*OLD, *NEW]), flush=True)

    ci = lambda v: summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    grp = lambda g, c: np.mean([S[f"{m}|{c}"] for m in GROUPS[g]], axis=0) if g in GROUPS else S[f"{g}|{c}"]
    cells = [*OLD, *NEW]
    show = PANEL + list(GROUPS)
    res = {"n_images": nimg, "timing_s_per_img": {k: float(np.mean(list(v.values()))) for k, v in timing.items() if v},
           "drop": {g: {c: ci(grp(g, c)) for c in cells} for g in show}, "contrast": {}}
    print(f"\n=== ENS-WP α @ devC[140:260] — relative AP drop % ===")
    print(f"{'cell':6}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in cells:
        print(f"{c:6}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    for name, (a, b) in {"ENS2 − ENS1": ("ENS2", "ENS1"), "ENS1 − B50 (gap α=1)": ("ENS1", "B50"),
                         "ENS2 − A2 (gap α=2)": ("ENS2", "A2"), "best-ENS − A2": (None, "A2")}.items():
        if a is None:  # mốc ENS khóa: α=2 nếu T7 point ≥ α=1, ngược lại α=1
            a = "ENS2" if grp("T7", "ENS2")[0] >= grp("T7", "ENS1")[0] else "ENS1"
            res["ens_reference"] = a
        d = {g: ci(grp(g, a) - grp(g, b)) for g in ("RES3", "dino_swin_l", "T7")}
        res["contrast"][name] = d
        print(f"{name:22} " + "  ".join(f"{g} {fmt(v)}" for g, v in d.items()))
    g2 = res["contrast"]["ENS2 − ENS1"]["T7"]
    res["verdict_alpha_generalizes"] = ("α=2 TỔNG QUÁT sang ENS-WP" if g2["ci95"][0] > 0 else
                                        "α=2 HẠI ENS-WP" if g2["ci95"][1] < 0 else "KHÔNG RÕ trên ENS-WP")
    print(f"VERDICT: {res['verdict_alpha_generalizes']} | mốc ENS cho method mới: {res['ens_reference']}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[ens-α] -> {OUT}")


if __name__ == "__main__":
    main()
