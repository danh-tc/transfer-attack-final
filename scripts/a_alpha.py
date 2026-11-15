"""α-falsifier (progress_log 2026-11-15): RD-CNX B50 có yếu vì phân bổ độ lớn bước (α) chứ không phải thiếu bước?
Screening, KHÔNG phải candidate. devC[140:260] (120 ảnh), RTX 3090, ε 5, recipe RD-CNX đóng băng, chỉ đổi α.

Ô sinh (seed_all như E88/F1b):
  RD-CNX-B100  α=1, steps=100, snapshot δ@50 (= baseline B50, snapshot ≡ chạy riêng: f1b_snapcheck PASS) — seed 1000·(n+1)
  RD-CNX-B50s2 α=1, steps=50, seed 1000·(n+1)+7  (seed-null + mốc wall-clock B50 chạy riêng)
  A2-B50       α=2 hằng, steps=50, seed 1000·(n+1)
  AS-B50       α_t = 2 − 1.5·t/49, t=0..49 (2 → 0.5 tuyến tính), steps=50, seed 1000·(n+1)
Eval: 9 model, relative AP drop, boot 1000 seed 2026, contrast paired trong tiến trình. Quy tắc khóa trong progress_log.
  python scripts/a_alpha.py --check     # unit test: alpha_sched=[1]*50 ≡ đường mặc định (2 ảnh)
  python scripts/a_alpha.py --gen-only
  python scripts/a_alpha.py
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

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e88_ens_baselines import C, seed_all
from scripts.f1b_r1a import ANN, ART, IMGDIR, EPS, BOOT_SEED, PANEL, GROUPS, save_png
from scripts.run_baselines import METHODS

GEN = ["RD-CNX-B100", "RD-CNX-B50s2", "A2-B50", "AS-B50"]
AS_SCHED = [2.0 - 1.5 * t / 49 for t in range(50)]
CELLS = {"B50": ("RD-CNX-B100", "snap50"), "B100": ("RD-CNX-B100", "final"), "B50s2": ("RD-CNX-B50s2", "final"),
         "A2": ("A2-B50", "final"), "AS": ("AS-B50", "final")}
OUT = os.path.join(REPO_ROOT, "results/mechanism/a_alpha_devC.json")


def ids_devC():
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/devC300.csv"))
    assert len(ids) == 300
    return ids[140:260]


def png(run_dir, cell, sub, i):
    return os.path.join(ART, run_dir, "adv", cell, sub, f"{i}.png")


def attack(model, orig, dsm, cell, n, on_step=None):
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    kw = {"on_step": on_step} if on_step is not None else {}
    if cell == "RD-CNX-B100":
        seed_all(1000 * (n + 1)); return fn(model, orig, dsm, steps=100, epsilon=EPS, **kw)
    if cell == "RD-CNX-B50s2":
        seed_all(1000 * (n + 1) + 7); return fn(model, orig, dsm, steps=50, epsilon=EPS)
    if cell == "A2-B50":
        seed_all(1000 * (n + 1)); return fn(model, orig, dsm, steps=50, epsilon=EPS, alpha=2.0)
    if cell == "AS-B50":
        seed_all(1000 * (n + 1)); return fn(model, orig, dsm, steps=50, epsilon=EPS, alpha_sched=AS_SCHED)
    raise KeyError(cell)


def check(ids):
    """alpha_sched=[1]*50 phải ≡ đường mặc định α=1 (cùng seed), so với nền B50-vs-B50 (RD-CNX không tất định bit trên 3090)."""
    model = load_model(C, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX"]
    rep = []
    for n, i in enumerate(ids):
        orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
        q = lambda d: torch.round(to_adv_image(orig, d))
        seed_all(1000 * (n + 1)); a = q(fn(model, orig, dsm, steps=50, epsilon=EPS))
        seed_all(1000 * (n + 1)); b = q(fn(model, orig, dsm, steps=50, epsilon=EPS))
        seed_all(1000 * (n + 1)); s = q(fn(model, orig, dsm, steps=50, epsilon=EPS, alpha_sched=[1.0] * 50))
        r = {"img_id": i, "frac_diff_sched1_vs_default": float((s != a).float().mean()),
             "frac_diff_default_vs_default": float((a != b).float().mean())}
        rep.append(r); print(r, flush=True)
    ok = all(r["frac_diff_sched1_vs_default"] <= max(1.5 * r["frac_diff_default_vs_default"], 1e-3) for r in rep)
    print(f"[a_alpha check] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok, rep


def generate(ids, cell, run_dir, timing):
    subs = ["snap50", "final"] if cell == "RD-CNX-B100" else ["final"]
    if all(os.path.exists(png(run_dir, cell, s, i)) for i in ids for s in subs):
        return
    model = load_model(C, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}
    t_all = timing.setdefault(cell, {})
    for n, i in enumerate(ids):
        if all(os.path.exists(png(run_dir, cell, s, i)) for s in subs):
            continue
        orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
        snaps = {}
        on_step = (lambda kk, nz: snaps.__setitem__(kk, nz.detach().clone()) if kk == 50 else None) if cell == "RD-CNX-B100" else None
        torch.cuda.synchronize(); t0 = time.time()
        delta = attack(model, orig, dsm, cell, n, on_step)
        torch.cuda.synchronize()
        t_all[str(i)] = time.time() - t0
        if cell == "RD-CNX-B100":
            save_png(orig, snaps[50], png(run_dir, cell, "snap50", i))
        save_png(orig, delta, png(run_dir, cell, "final", i))
        print(f"[a_alpha {cell}] {n + 1}/{len(ids)}  {np.mean(list(t_all.values())):.2f}s/ảnh", flush=True)
        if (n + 1) % 10 == 0:
            json.dump(timing, open(os.path.join(REPO_ROOT, "results/mechanism/a_alpha_timing.json"), "w"), indent=1)
    del model
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gen-only", action="store_true")
    args = ap.parse_args()
    ids = ids_devC()
    gpu = torch.cuda.get_device_name(0)
    run_dir = f"a_alpha_devC_eps5_{gpu.split()[-1].lower()}"
    print(f"[a_alpha] GPU {gpu} | {run_dir} | n={len(ids)}", flush=True)
    if args.check:
        ok, rep = check(ids[:2])
        json.dump({"gpu": gpu, "pass": ok, "images": rep}, open(os.path.join(REPO_ROOT, "results/mechanism/a_alpha_check.json"), "w"), indent=1)
        return
    tpath = os.path.join(REPO_ROOT, "results/mechanism/a_alpha_timing.json")
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for g in GEN:
        generate(ids, g, run_dir, timing)
        json.dump(timing, open(tpath, "w"), indent=1)
    if args.gen_only:
        return

    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, 1000, BOOT_SEED)])
    npz = OUT.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
    from mmdet.apis import inference_detector
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in CELLS):
            continue
        model = load(key)
        dets = {c: [] for c in ["clean", *CELLS]}
        with torch.no_grad():
            for i in ids:
                dets["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)
                for c, (g, sub) in CELLS.items():
                    dets[c] += to_coco_dets(inference_detector(model, png(run_dir, g, sub, i)), i, cat_ids)
        os.makedirs(os.path.join(ART, run_dir, "dets", key), exist_ok=True)
        with gzip.open(os.path.join(ART, run_dir, "dets", key, "all.json.gz"), "wt") as f:
            json.dump(dets, f)
        clean = np.array([CocoImageEval(coco, dets["clean"], ids).ap(w)[0] for w in W]) * 100
        for c in CELLS:
            ev = CocoImageEval(coco, dets[c], ids)
            S[f"{key}|{c}"] = 100 * (clean - np.array([ev.ap(w)[0] for w in W]) * 100) / clean
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in CELLS), flush=True)
    report(S, W, gpu, run_dir, timing)


def report(S, W, gpu, run_dir, timing):
    ci = lambda v: summarize(v[0], v[1:])
    fmt = lambda s: f"{s['point']:+6.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    grp = lambda g, c: np.mean([S[f"{m}|{c}"] for m in GROUPS[g]], axis=0) if g in GROUPS else S[f"{g}|{c}"]
    t = {k: float(np.mean(list(v.values()))) for k, v in timing.items() if v}
    res = {"gpu": gpu, "run_dir": run_dir, "n_images": 120, "timing_s_per_img": t,
           "drop": {g: {c: ci(grp(g, c)) for c in CELLS} for g in PANEL + list(GROUPS)}, "contrast": {}, "verdict": {}}
    show = PANEL + list(GROUPS)
    print(f"\n=== α-falsifier @ devC[140:260], {gpu} — relative AP drop % ===")
    print(f"{'cell':7}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in CELLS:
        print(f"{c:7}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))
    sn = ci(grp("RES3", "B50s2") - grp("RES3", "B50"))
    gain100 = {g: grp(g, "B100") - grp(g, "B50") for g in ("RES3", "T7", "dino_swin_l")}
    res["contrast"]["seed_null_RES3"] = sn
    res["contrast"]["seed_null_SwinL"] = ci(grp("dino_swin_l", "B50s2") - grp("dino_swin_l", "B50"))
    res["contrast"]["B100-B50"] = {g: ci(v) for g, v in gain100.items()}
    print(f"seed-null RES3 {fmt(sn)} | SwinL {fmt(res['contrast']['seed_null_SwinL'])}")
    print("B100 − B50     " + "  ".join(f"{g} {fmt(ci(v))}" for g, v in gain100.items()))
    t_ref = t.get("RD-CNX-B50s2")
    for v, cell in (("A2", "A2-B50"), ("AS", "AS-B50")):
        d = {g: ci(grp(g, v) - grp(g, "B50")) for g in ("RES3", "dino_swin_l", "T7")}
        ratio = t.get(cell, float("nan")) / t_ref if t_ref else float("nan")
        dR, dS = d["RES3"], d["dino_swin_l"]
        if (dR["point"] >= 2 and dR["ci95"][0] > 0 and dS["point"] >= -2 and dR["point"] > abs(sn["point"]) and ratio <= 1.1):
            verdict = "PASS"
        elif dR["ci95"][1] < 2 or dS["ci95"][1] < -2:
            verdict = "NEGATIVE"
        else:
            verdict = "INCONCLUSIVE"
        rec = {}
        for g in ("RES3", "T7"):  # recovery fraction (mô tả): bootstrap tỉ số trên cùng resample
            num, den = grp(g, v) - grp(g, "B50"), gain100[g]
            rec[g] = ci(num / np.where(np.abs(den) < 1e-9, np.nan, den)) if gain100[g][1:].mean() > 0 else None
        res["contrast"][v] = {"delta_vs_B50": d, "wall_ratio_vs_B50run": ratio, "recovery": rec}
        res["verdict"][v] = verdict
        print(f"{v} − B50  RES3 {fmt(dR)}  SwinL {fmt(dS)}  T7 {fmt(d['T7'])}  wall×{ratio:.2f}  -> {verdict}")
        for g, r in rec.items():
            if r is not None:
                print(f"    recovery {g}: {r['point']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[a_alpha] -> {OUT}")


if __name__ == "__main__":
    main()
