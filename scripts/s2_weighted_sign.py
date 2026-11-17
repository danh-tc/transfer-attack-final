"""S2 (progress_log 2026-11-17 10:20, KHÓA): khớp dấu CÓ TRỌNG SỐ độ nhạy target — chẩn đoán (dùng gradient target).
Dùng lại PNG của S1 (artifacts/runs/s1_signmap/adv, 10 ô × dev100[:10]). g_t = ∇ₓ L_cls DINO-Swin-L (như WB S1) tại
x_t = x + t·δ, t ∈ {0, 0.5, 1}. Theo pixel-kênh (bỏ δ = 0 ở tử; mẫu số = ‖g‖₁ toàn vùng):
  A(R)      = Σ_R sign(δ)·g / Σ_R |g|                (−1..1, ngẫu nhiên ≈ 0); R ∈ {ALL, IN, OUT}
  A_path    = TB của A(ALL) qua 3 t  (≈ mức tăng loss bậc 1 chuẩn hóa dọc đường x → x+δ)
  mass_IN   = ‖g_IN‖₁ / ‖g‖₁ ;  top10 = phần ‖g‖₁ trong 10% pixel-kênh |g| lớn nhất
  A_top10 / A_rest = A trên 10% |g| lớn nhất / phần còn lại;  contrib_IN = phần tử số Σ sign·g nằm trong IN
Liên hệ drop Swin-L: drop theo ô (S1 json) và theo (ô, ảnh) (S1 json "points").
  python scripts/s2_weighted_sign.py
"""
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
from attack.targets import load_any, target_dataset, target_grads
from scripts.r1_region import region_masks

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
ADV = os.path.join(REPO_ROOT, "artifacts/runs/s1_signmap/adv")
S1 = os.path.join(REPO_ROOT, "results/mechanism/s1_signmap_dev10.json")
OUT = os.path.join(REPO_ROOT, "results/mechanism/s2_weighted_sign_dev10.json")
ATT = ["RD-ALL", "RD-FEAT", "RD-DET", "OSFD-ALL", "OSFD-FEAT", "OSFD-DET"]
CELLS = ATT + ["WB-0", "WB-r1", "WB-r2", "RAND"]
TS = (0.0, 0.5, 1.0)


def stats(sd, g, inm):
    """sd: sign δ [3,H,W] int; g: [3,H,W] float; inm: [H,W] bool -> dict tử/mẫu theo vùng (để gộp qua ảnh)."""
    a = np.abs(g); s = sd * g
    inm3 = np.broadcast_to(inm, g.shape)
    thr = np.quantile(a, 0.9); top = a >= thr
    d = {}
    for name, m in (("ALL", np.ones_like(inm3)), ("IN", inm3), ("OUT", ~inm3), ("top10", top), ("rest", ~top)):
        d[name] = (float(s[m].sum()), float(a[m].sum()))
    d["top10_mass"] = float(a[top].sum() / a.sum())
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s1 = json.load(open(S1, encoding="utf-8"))
    ids = s1["ids"]
    assert ids == load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:10]
    coco = COCO(ANN)
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    IN = {i: region_masks(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)), *clean[i].shape[:2])["IN"] > 0
          for i in ids}
    D = {c: {i: (mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]).transpose(2, 0, 1)
             for i in ids} for c in CELLS}

    model = load_any("dino_swin_l")
    ds = {s["img_id"]: s for s in target_dataset("dino_swin_l", ids)}
    R = {c: {t: {} for t in TS} for c in CELLS}          # R[c][t][i] = stats
    for n, i in enumerate(ids):
        orig = ds[i]["orig_inputs"].cuda().float(); dsm = ds[i]["data_sample"]
        assert np.abs(orig.cpu().numpy() - clean[i].transpose(2, 0, 1)).max() == 0

        def grad_at(x):
            torch.manual_seed(1000 * (n + 1))
            return target_grads(model, "dino_swin_l", x, dsm, objectives=("cls",))[0]["cls"].float().cpu().numpy()
        g0 = grad_at(orig)
        for c in CELLS:
            d = torch.from_numpy(D[c][i]).float().cuda()
            sd = np.sign(D[c][i])
            for t in TS:
                g = g0 if t == 0.0 else grad_at(torch.clamp(orig + t * d, 0, 255))
                R[c][t][i] = stats(sd, g, IN[i])
        print(f"[s2] {n + 1}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()

    def A(c, t, reg, imgs=ids):
        num = sum(R[c][t][i][reg][0] for i in imgs); den = sum(R[c][t][i][reg][1] for i in imgs)
        return num / den

    res = {"n": len(ids), "ts": TS, "cell": {}, "points": []}
    drop = s1["drop"]["dino_swin_l"]
    print(f"\n{'ô':10}{'drop':>7}" + "".join(f"{'A t=' + str(t):>10}" for t in TS) +
          f"{'A_path':>9}{'A_IN':>8}{'A_OUT':>8}{'A_top10':>9}{'A_rest':>8}{'ctbIN':>7}")
    for c in CELLS:
        a_t = {t: A(c, t, "ALL") for t in TS}
        path = float(np.mean(list(a_t.values())))
        num_in = np.mean([sum(R[c][t][i]["IN"][0] for i in ids) for t in TS])
        num_all = np.mean([sum(R[c][t][i]["ALL"][0] for i in ids) for t in TS])
        r = {"drop_swinl": drop[c], "A_t": {str(t): a_t[t] for t in TS}, "A_path": path,
             **{f"A_{k}": float(np.mean([A(c, t, k) for t in TS])) for k in ("IN", "OUT", "top10", "rest")},
             "contrib_IN": float(num_in / num_all) if num_all else float("nan")}
        res["cell"][c] = r
        print(f"{c:10}{drop[c]:7.1f}" + "".join(f"{a_t[t]:10.4f}" for t in TS) +
              f"{path:9.4f}{r['A_IN']:8.4f}{r['A_OUT']:8.4f}{r['A_top10']:9.4f}{r['A_rest']:8.4f}{r['contrib_IN']:7.2f}")
    mass_in = np.mean([R["RD-ALL"][0.0][i]["IN"][1] / R["RD-ALL"][0.0][i]["ALL"][1] for i in ids])
    top10 = np.mean([R["RD-ALL"][0.0][i]["top10_mass"] for i in ids])
    print(f"\n|g_t| ảnh sạch: mass_IN {mass_in:.2f} (diện tích IN {np.mean([IN[i].mean() for i in ids]):.2f}) | "
          f"top10% pixel-kênh giữ {top10:.2f} ‖g‖₁")
    res["g_clean"] = {"mass_IN": float(mass_in), "area_IN": float(np.mean([IN[i].mean() for i in ids])),
                      "top10_mass": float(top10)}

    # liên hệ drop
    pts = {(p["cell"], p["img"]): p["drop"] for p in s1["sign_vs_drop"]["points"]}
    out = {}
    for key in ["0.0", "0.5", "1.0", "path"]:
        def a_img(c, i):
            ts = TS if key == "path" else (float(key),)
            return np.mean([R[c][t][i]["ALL"][0] / R[c][t][i]["ALL"][1] for t in ts])
        xs = [res["cell"][c]["A_path"] if key == "path" else res["cell"][c]["A_t"][key] for c in ATT]
        rho_c = spearmanr(xs, [drop[c] for c in ATT]).correlation
        P = [(a_img(c, i), pts[(c, i)]) for c in ATT for i in ids if (c, i) in pts]
        av, dv = np.array([p[0] for p in P]), np.array([p[1] for p in P])
        rho_i = spearmanr(av, dv).correlation
        med = np.median(dv)
        out[key] = {"spearman_cell": rho_c, "spearman_img": rho_i, "n_img_pts": len(P),
                    "A_hi_drop": float(av[dv > med].mean()), "A_lo_drop": float(av[dv <= med].mean())}
        print(f"A({key:4}) vs drop Swin-L: Spearman ô(6) {rho_c:+.2f} | (ô,ảnh)({len(P)}) {rho_i:+.2f} | "
              f"A nửa drop cao {out[key]['A_hi_drop']:.4f} vs thấp {out[key]['A_lo_drop']:.4f}")
    res["vs_drop"] = out

    # phán quyết (quy tắc khóa)
    p = out["path"]
    v1 = ("A PHÂN BIỆT drop" if p["spearman_cell"] >= 0.6 and p["spearman_img"] >= 0.3 else
           "A KHÔNG phân biệt drop" if p["spearman_cell"] <= 0.1 and p["spearman_img"] <= 0.1 else "KHÔNG RÕ")
    C = res["cell"]
    v4 = ("A GIẢI nghịch lý S1 (DET > FEAT cả 2 method, khớp drop)"
          if C["RD-DET"]["A_path"] > C["RD-FEAT"]["A_path"] and C["OSFD-DET"]["A_path"] > C["OSFD-FEAT"]["A_path"]
          else "A KHÔNG giải nghịch lý S1")
    v3 = "|g_t| TẬP TRUNG" if top10 >= 0.5 else "|g_t| KHÔNG tập trung mạnh"
    res["verdict"] = {"discriminate": v1, "paradox": v4, "concentration": v3}
    print(f"\n=> {v1}\n=> {v4}\n=> {v3} (top10 {top10:.2f}); mass_IN {mass_in:.2f}")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[s2] -> {OUT}")


if __name__ == "__main__":
    main()
