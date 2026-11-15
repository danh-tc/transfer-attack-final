"""F1b + R1a (progress_log 2026-11-15): chẩn đoán hội tụ RD-CNX theo B + khoảng cách distortion feature sâu trên ResNet.
KHÔNG phải method candidate (amendment 2026-11-15). Toàn bộ ô sinh LẠI trên cùng 1 máy (3090), dev100, ε 5.

Ô sinh (seed torch + python random + numpy, quy tắc E88):
  RD-CNX-B100   METHODS["C2-OCT-1V-WP2-RD-CNX"] (đóng băng) steps=100, snapshot δ ở 36/50/75 -> ô RD-CNX@{36,50,75,100}
                seed 1000·(n+1)  (= seed RD-CNX của E88)
  RD-CNX-B50s2  y hệt, steps=50, seed 1000·(n+1) + 7  (seed-null, cùng máy)
  RD-R50-B36    METHODS["C2-OCT-1V-WP2-RD"] (đóng băng, = E73 RD-R50) steps=36, seed 1000·(n+1)
  ENS-WP-B36    e88.ens_wp_attack (đóng băng), seed 1000·(n+1) + b
  [OSFD-PUB-B50 chỉ khi user duyệt: osfd_attack θ7 ρ0.8, seed 1000·(n+1)]
Snapshot ≡ chạy riêng: kiểm bằng --snapcheck (2 ảnh) TRƯỚC khi sinh B100 (quy tắc khóa trong progress_log).

Eval (1 lượt forward/ảnh/ô/model, inference_detector trên PNG): relative AP drop, panel 9 model, boot 1000 seed 2026.
R1a trong CÙNG lượt forward: hook backbone, D_k = ‖F_k(adv) − F_k(clean)‖ / ‖F_k(clean)‖ (đúng định nghĩa E64), theo ảnh;
P_k = D_{k+1} / (D_k + 0.01) theo ảnh. Lưu toàn bộ mảng theo ảnh -> mọi contrast paired tính lại được.

  python scripts/f1b_r1a.py --snapcheck              # unit test snapshot (2 ảnh)
  python scripts/f1b_r1a.py --gen-only               # sinh mọi ô (resume theo PNG)
  python scripts/f1b_r1a.py                          # sinh (nếu thiếu) + eval + phán quyết
"""
import argparse
import gzip
import json
import os
import sys
import time

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from attack.methods.baselines import osfd_attack
from attack.models import load_model
from attack.preprocess import to_adv_image
from attack.targets import target_dataset
from scripts.e64_cnx_drop import load
from scripts.e88_ens_baselines import OSFD_PUB, R, C, ens_wp_attack, seed_all
from scripts.run_baselines import METHODS

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ART = os.path.join(REPO_ROOT, "artifacts/runs")
EPS, BOOT_SEED, ETA = 5.0, 2026, 0.01
SNAPS = (36, 50, 75)
GEN = ["RD-CNX-B100", "RD-CNX-B50s2", "RD-R50-B36", "ENS-WP-B36", "OSFD-PUB-B50"]
PANEL = ["r50", "r101", "convnext_t", "swin_t", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
GROUPS = {"RES3": ["r101", "fcos_r50", "dino_r50"],
          "T7": ["r101", "fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "swin_t", "dino_swin_l"]}
DK_MODELS = PANEL  # D_k đo trên mọi model của panel (backbone hook); gate R1a chỉ dùng RES3, Swin-L là đối chứng


def eval_cells(gen):
    """ô eval -> (ô sinh, thư mục con)."""
    out = {f"RD-CNX@{b}": ("RD-CNX-B100", f"snap{b}") for b in SNAPS}
    out["RD-CNX@100"] = ("RD-CNX-B100", "final")
    out["RD-CNX@50s2"] = ("RD-CNX-B50s2", "final")
    out["RD-R50"] = ("RD-R50-B36", "final")
    out["ENS-WP"] = ("ENS-WP-B36", "final")
    if "OSFD-PUB-B50" in gen:
        out["OSFD-PUB"] = ("OSFD-PUB-B50", "final")
    return out


def png_path(run_dir, cell, sub, i):
    return os.path.join(ART, run_dir, "adv", cell, sub, f"{i}.png")


def save_png(orig, delta, path):
    adv = to_adv_image(orig, delta)
    assert float((adv - orig).abs().max()) <= EPS + 1e-4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mmcv.imwrite(adv.permute(1, 2, 0).cpu().numpy().astype(np.uint8), path + ".tmp.png")
    os.replace(path + ".tmp.png", path)


def rd(key, model, orig, dsm, steps, seed, on_step=None):
    seed_all(seed)
    fn, _ = METHODS["C2-OCT-1V-WP2-RD-CNX" if key == C else "C2-OCT-1V-WP2-RD"]
    kw = {"on_step": on_step} if on_step is not None else {}
    return fn(model, orig, dsm, steps=steps, epsilon=EPS, **kw)


def generate(ids, cell, run_dir, timing):
    subs = [f"snap{b}" for b in SNAPS] + ["final"] if cell == "RD-CNX-B100" else ["final"]
    if all(os.path.exists(png_path(run_dir, cell, s, i)) for i in ids for s in subs):
        print(f"[f1b] {cell}: PNG đủ, bỏ qua", flush=True)
        return
    keys = {"RD-CNX-B100": [C], "RD-CNX-B50s2": [C], "RD-R50-B36": [R], "ENS-WP-B36": [R, C], "OSFD-PUB-B50": [R]}[cell]
    models = {k: load_model(k, "cuda:0") for k in keys}
    ds = {k: {s["img_id"]: s for s in target_dataset(k, ids)} for k in keys}
    t_all = timing.setdefault(cell, {})
    for n, i in enumerate(ids):
        if all(os.path.exists(png_path(run_dir, cell, s, i)) for s in subs):
            continue
        k0 = keys[0]
        orig = ds[k0][i]["orig_inputs"].cuda().float()
        dsm = ds[k0][i]["data_sample"]
        torch.cuda.synchronize(); t0 = time.time()
        if cell == "RD-CNX-B100":
            snaps = {}

            def on_step(kk, nz):
                if kk in SNAPS:
                    snaps[kk] = nz.detach().clone()
            delta = rd(C, models[C], orig, dsm, 100, 1000 * (n + 1), on_step)
            for b in SNAPS:
                save_png(orig, snaps[b], png_path(run_dir, cell, f"snap{b}", i))
        elif cell == "RD-CNX-B50s2":
            delta = rd(C, models[C], orig, dsm, 50, 1000 * (n + 1) + 7)
        elif cell == "RD-R50-B36":
            delta = rd(R, models[R], orig, dsm, 36, 1000 * (n + 1))
        elif cell == "ENS-WP-B36":
            orig, delta = ens_wp_attack(models, ds, i, n)
        else:
            seed_all(1000 * (n + 1))
            delta = osfd_attack(models[R], orig, dsm, steps=50, epsilon=EPS, **OSFD_PUB)
        torch.cuda.synchronize()
        t_all[str(i)] = time.time() - t0
        save_png(orig, delta, png_path(run_dir, cell, "final", i))
        print(f"[f1b {cell}] {n + 1}/{len(ids)}  {np.mean(list(t_all.values())):.2f}s/ảnh", flush=True)
    del models
    torch.cuda.empty_cache()


def snapcheck(ids):
    """Unit test: snapshot@50 của lượt B100 vs lượt B50 riêng (cùng ảnh, cùng seed), so với nền tất định B50 vs B50."""
    model = load_model(C, "cuda:0")
    ds = {s["img_id"]: s for s in target_dataset(C, ids)}
    rep = []
    for n, i in enumerate(ids):
        orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
        snap = {}
        rd(C, model, orig, dsm, 100, 1000 * (n + 1), lambda kk, nz: snap.__setitem__(kk, nz.clone()) if kk == 50 else None)
        a = rd(C, model, orig, dsm, 50, 1000 * (n + 1))
        b = rd(C, model, orig, dsm, 50, 1000 * (n + 1))
        q = lambda d: torch.round(to_adv_image(orig, d))
        s50, qa, qb = q(snap[50]), q(a), q(b)
        r = {"img_id": i, "frac_diff_snap_vs_B50": float((s50 != qa).float().mean()),
             "frac_diff_B50_vs_B50": float((qa != qb).float().mean()),
             "mean_abs_snap_vs_B50": float((s50 - qa).abs().mean()), "mean_abs_B50_vs_B50": float((qa - qb).abs().mean()),
             "max_abs_delta_snap": float(snap[50].abs().max()), "max_abs_delta_B50": float(a.abs().max())}
        rep.append(r)
        print(r, flush=True)
    ok = all(r["frac_diff_snap_vs_B50"] <= max(1.5 * r["frac_diff_B50_vs_B50"], 1e-3) for r in rep)
    print(f"[snapcheck] {'PASS' if ok else 'FAIL'} (snapshot ≤ max(1.5 × nền B50-vs-B50, 0.1%) pixel khác)", flush=True)
    return ok, rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapcheck", action="store_true")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--with-osfd", action="store_true", help="thêm ô OSFD-PUB-B50 (chỉ khi user duyệt)")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    assert len(ids) == 100
    gpu = torch.cuda.get_device_name(0)
    mtag = gpu.split()[-1].replace("-", "").lower()
    run_dir = f"f1b_r1a_eps5_{mtag}"
    out_json = os.path.join(REPO_ROOT, "results/mechanism/f1b_r1a_dev100.json")
    print(f"[f1b] GPU = {gpu} | run dir {run_dir}", flush=True)
    if args.snapcheck:
        ok, rep = snapcheck(ids[:2])
        json.dump({"gpu": gpu, "pass": ok, "images": rep},
                  open(os.path.join(REPO_ROOT, "results/mechanism/f1b_snapcheck.json"), "w"), indent=1)
        return

    gen = [g for g in GEN if g != "OSFD-PUB-B50" or args.with_osfd]
    tpath = os.path.join(REPO_ROOT, "results/mechanism/f1b_r1a_timing.json")
    timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for g in gen:
        generate(ids, g, run_dir, timing)
        json.dump(timing, open(tpath, "w"), indent=1)
    if args.gen_only:
        return

    cells = eval_cells(gen)
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    n = len(ids)
    W = np.vstack([np.ones((1, n)), bootstrap_indices(n, args.boot, BOOT_SEED)])
    npz = out_json.replace(".json", "_samples.npz")
    S = dict(np.load(npz)) if os.path.exists(npz) else {}
    from mmdet.apis import inference_detector
    dets_dir = os.path.join(ART, run_dir, "dets")
    for key in PANEL:
        if all(f"{key}|{c}" in S for c in cells):
            continue
        model = load(key)
        grab = {}
        h = model.backbone.register_forward_hook(lambda m, i, o: grab.__setitem__("f", [f.detach().float() for f in o]))
        stages = [f"C{j + 2}" for j in model.backbone.out_indices]  # ResNet/ConvNeXt/Swin: 0..3 -> C2..C5; DINO-R50 (1,2,3)
        dets = {c: [] for c in ["clean", *cells]}
        dk = {c: [] for c in cells}
        with torch.no_grad():
            for i in ids:
                dets["clean"] += to_coco_dets(inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i, cat_ids)
                fc = grab["f"]
                for c, (g, sub) in cells.items():
                    dets[c] += to_coco_dets(inference_detector(model, png_path(run_dir, g, sub, i)), i, cat_ids)
                    dk[c].append([float((a - b).norm() / b.norm()) for a, b in zip(grab["f"], fc)])
        h.remove()
        os.makedirs(os.path.join(dets_dir, key), exist_ok=True)
        with gzip.open(os.path.join(dets_dir, key, "all.json.gz"), "wt") as f:
            json.dump(dets, f)
        clean = np.array([CocoImageEval(coco, dets["clean"], ids).ap(w)[0] for w in W]) * 100
        for c in cells:
            ev = CocoImageEval(coco, dets[c], ids)
            adv = np.array([ev.ap(w)[0] for w in W]) * 100
            S[f"{key}|{c}"] = (100 * (clean - adv) / clean).astype(np.float64)
            S[f"Dk|{key}|{c}"] = np.array(dk[c], dtype=np.float64)       # [n_img, n_stage]
        S[f"stages|{key}"] = np.array(stages)
        del model
        torch.cuda.empty_cache()
        np.savez(npz, **S)
        print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{c}'][0]:.1f}" for c in cells), flush=True)

    report(S, cells, W, n, gpu, run_dir, out_json, timing)


def ci(v):
    return summarize(v[0], v[1:])


def report(S, cells, W, n, gpu, run_dir, out_json, timing):
    fmt = lambda s: f"{s['point']:+6.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    grp = lambda g, c: np.mean([S[f"{m}|{c}"] for m in GROUPS[g]], axis=0) if g in GROUPS else S[f"{g}|{c}"]
    res = {"n_images": n, "n_boot": len(W) - 1, "gpu": gpu, "run_dir": run_dir, "drop": {}, "f1b": {}, "r1a": {},
           "timing_s_per_img": {k: float(np.mean(list(v.values()))) for k, v in timing.items() if v}}
    for g in PANEL + list(GROUPS):
        res["drop"][g] = {c: ci(grp(g, c)) for c in cells}
    print(f"\n=== F1b/R1a @ dev100, {gpu} — relative AP drop % ===")
    show = PANEL + list(GROUPS)
    print(f"{'cell':12}" + "".join(f"{g[:10]:>11}" for g in show))
    for c in cells:
        print(f"{c:12}" + "".join(f"{res['drop'][g][c]['point']:11.1f}" for g in show))

    # ---- F1b (quy tắc khóa progress_log 2026-11-15) ----
    GR = grp("RES3", "RD-CNX@100") - grp("RES3", "RD-CNX@50")
    GS = grp("dino_swin_l", "RD-CNX@100") - grp("dino_swin_l", "RD-CNX@50")
    HR = grp("RES3", "ENS-WP") - grp("RES3", "RD-CNX@100")
    SN = grp("RES3", "RD-CNX@50s2") - grp("RES3", "RD-CNX@50")
    SNs = grp("dino_swin_l", "RD-CNX@50s2") - grp("dino_swin_l", "RD-CNX@50")
    f = {k: ci(v) for k, v in {"G_R": GR, "G_S": GS, "H_R": HR, "seed_RES3": SN, "seed_SwinL": SNs}.items()}
    for b1, b2 in ((36, 50), (50, 75), (75, 100)):
        f[f"RES3 {b1}->{b2}"] = ci(grp("RES3", f"RD-CNX@{b2}") - grp("RES3", f"RD-CNX@{b1}"))
        f[f"SwinL {b1}->{b2}"] = ci(grp("dino_swin_l", f"RD-CNX@{b2}") - grp("dino_swin_l", f"RD-CNX@{b1}"))
    g, gs, h = f["G_R"], f["G_S"], f["H_R"]
    if g["point"] >= 3 and g["ci95"][0] > 0 and gs["point"] >= -2 and g["point"] > abs(f["seed_RES3"]["point"]):
        verdict = "CONTINUED GAIN"
    elif g["point"] < 1.5 and g["ci95"][1] < 3 and h["point"] >= 5:
        verdict = "PRACTICAL PLATEAU"
    else:
        verdict = "INCONCLUSIVE"
    f["verdict"] = verdict
    res["f1b"] = f
    print("\n--- F1b ---")
    for k, v in f.items():
        if k != "verdict":
            print(f"{k:16} {fmt(v)}")
    print(f"F1b VERDICT: {verdict}")

    # ---- R1a ----
    pairs = {"ENS-WP − RD-CNX": ("ENS-WP", "RD-CNX@50"), "RD-R50 − RD-CNX": ("RD-R50", "RD-CNX@50")}
    if "OSFD-PUB" in cells:
        pairs["OSFD-PUB − RD-R50"] = ("OSFD-PUB", "RD-R50")
    r1 = {}
    for key in PANEL:
        st = list(S[f"stages|{key}"])
        r1[key] = {"stages": st, "D": {}, "P": {}, "dD": {}, "dP": {}, "dD_std_eff": {}, "dP_std_eff": {}}
        for c in cells:
            D = S[f"Dk|{key}|{c}"]
            r1[key]["D"][c] = {s: ci(W @ D[:, j] / n) for j, s in enumerate(st)}
            P = D[:, 1:] / (D[:, :-1] + ETA)
            r1[key]["P"][c] = {f"{st[j]}->{st[j + 1]}": ci(W @ P[:, j] / n) for j in range(len(st) - 1)}
        for name, (a, b) in pairs.items():
            Da, Db = S[f"Dk|{key}|{a}"], S[f"Dk|{key}|{b}"]
            dD = Da - Db
            dP = Da[:, 1:] / (Da[:, :-1] + ETA) - Db[:, 1:] / (Db[:, :-1] + ETA)
            r1[key]["dD"][name] = {s: ci(W @ dD[:, j] / n) for j, s in enumerate(st)}
            r1[key]["dP"][name] = {f"{st[j]}->{st[j + 1]}": ci(W @ dP[:, j] / n) for j in range(len(st) - 1)}
            r1[key]["dD_std_eff"][name] = {s: float(dD[:, j].mean() / (dD[:, j].std(ddof=1) + 1e-12)) for j, s in enumerate(st)}
            r1[key]["dP_std_eff"][name] = {f"{st[j]}->{st[j + 1]}": float(dP[:, j].mean() / (dP[:, j].std(ddof=1) + 1e-12))
                                           for j in range(len(st) - 1)}
    # Gate (khóa): feature gap ⇔ dD > 0 với cận dưới CI > 0 ở CẢ C4 và C5, cho CẢ ENS−CNX và R50−CNX, trên ≥ 2/3 RES3
    def gap_ok(key):
        return all(r1[key]["dD"][nm][s]["ci95"][0] > 0 for nm in ("ENS-WP − RD-CNX", "RD-R50 − RD-CNX") for s in ("C4", "C5"))
    n_gap = sum(gap_ok(k) for k in GROUPS["RES3"])
    ap_gap = ci(grp("RES3", "ENS-WP") - grp("RES3", "RD-CNX@50"))
    # P vs D (khóa): so hiệu ứng chuẩn hóa |mean/sd| của ΔP (C3->C4, C4->C5) với ΔD (C4, C5), cặp ENS − CNX, trên RES3
    pd = {}
    for key in GROUPS["RES3"]:
        eD = max(abs(r1[key]["dD_std_eff"]["ENS-WP − RD-CNX"][s]) for s in ("C4", "C5"))
        eP = max(abs(r1[key]["dP_std_eff"]["ENS-WP − RD-CNX"][p]) for p in ("C3->C4", "C4->C5"))
        pd[key] = {"eff_D": eD, "eff_P": eP, "P_better": eP > eD}
    n_pbetter = sum(v["P_better"] for v in pd.values())
    if n_gap >= 2:
        r1v = ("FEATURE GAP + ưu tiên cơ chế TRUYỀN (P phân biệt tốt hơn D ở ≥ 2/3)" if n_pbetter >= 2
               else "FEATURE GAP (chỉ distortion phân biệt: deep-feature vulnerability, KHÔNG claim propagation)")
    elif ap_gap["ci95"][0] > 0:
        r1v = "METRIC INCONCLUSIVE (AP khác rõ nhưng distortion không phân biệt ở ≥ 2/3 RES3)"
    else:
        r1v = "INCONCLUSIVE (AP gap ENS − CNX trên RES3 không loại 0)"
    res["r1a"] = {"per_model": r1, "n_gap_RES3": n_gap, "gap_models": [k for k in GROUPS["RES3"] if gap_ok(k)],
                  "AP_gap_RES3_ENS_minus_CNX": ap_gap, "P_vs_D": pd, "verdict": r1v}
    print("\n--- R1a: ΔD_k (paired) ---")
    for key in GROUPS["RES3"] + ["dino_swin_l", "convnext_t"]:
        for nm in pairs:
            print(f"{key:12} {nm:18} " + " ".join(f"{s} {fmt(v)}" for s, v in r1[key]["dD"][nm].items()))
    print("--- R1a: ΔP_k (paired) ---")
    for key in GROUPS["RES3"] + ["dino_swin_l"]:
        for nm in pairs:
            print(f"{key:12} {nm:18} " + " ".join(f"{s} {fmt(v)}" for s, v in r1[key]["dP"][nm].items()))
    print(f"P vs D: {pd}")
    print(f"R1a VERDICT: {r1v} | gap trên {n_gap}/3 RES3 | AP gap RES3 ENS−CNX {fmt(ap_gap)}")
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[f1b] -> {out_json}")


if __name__ == "__main__":
    main()
