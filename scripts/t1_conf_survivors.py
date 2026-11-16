"""T1 (progress_log 2026-11-16 22:35, KHÓA): survivor transfer (RD-CNX@50) có phải object CNX white-box còn tự tin?"""
import gzip, json, os, sys
import numpy as np
from pycocotools.coco import COCO
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
from scripts.e77_dfc import fate, group
from scripts.f1b_r1a import ANN, ART

RUN, SEED, THR = "f1b_r1a_eps5_3090", 2026, 0.9
OUT = os.path.join(REPO_ROOT, "results/mechanism/t1_conf_survivors_dev100.json")
GROUPS = {"RES3": ["r101", "fcos_r50", "dino_r50"], "SwinL": ["dino_swin_l"]}


def main():
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    ms = ["convnext_t", *GROUPS["RES3"], *GROUPS["SwinL"]]
    D = {m: json.load(gzip.open(os.path.join(ART, RUN, "dets", m, "all.json.gz"), "rt")) for m in ms}
    G = {m: {c: group(D[m][c]) for c in ("clean", "RD-CNX@50")} for m in ms}
    st = lambda m, c, k, a: fate(a["bbox"], a["category_id"], G[m][c].get(ids[k], []))
    rows = []
    for k, i in enumerate(ids):
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            if st("convnext_t", "clean", k, a)[0] != "survived":
                continue
            f, s = st("convnext_t", "RD-CNX@50", k, a)
            cs = s if f == "survived" else 0.0
            for g, mm in GROUPS.items():
                for m in mm:
                    if st(m, "clean", k, a)[0] != "survived":
                        continue
                    alive = st(m, "RD-CNX@50", k, a)[0] == "survived"
                    rows.append({"img": k, "grp": g, "model": m, "alive": alive, "cnx": cs})
    rng = np.random.default_rng(SEED)
    B = [np.arange(len(ids))] + [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    img = np.array([r["img"] for r in rows]); cnx = np.array([r["cnx"] for r in rows])
    alive = np.array([r["alive"] for r in rows]); grp = np.array([r["grp"] for r in rows])
    res = {}
    for g in GROUPS:
        out = {}
        for name, sel in (("survivors", (grp == g) & alive), ("killed_ref", (grp == g) & ~alive)):
            by = [np.where((img == k) & sel)[0] for k in range(len(ids))]
            v = np.array([np.mean(cnx[np.concatenate([by[k] for k in b])] >= THR) for b in B])
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            c = cnx[sel]
            out[name] = {"n": int(sel.sum()), "F_conf": float(v[0]), "ci95": [float(lo), float(hi)],
                         "frac_cnx_killed": float(np.mean(c == 0)), "frac_0.3-0.9": float(np.mean((c > 0) & (c < THR))),
                         "frac_ge_0.9": float(np.mean(c >= THR))}
            print(f"{g:6} {name:11} n={out[name]['n']:4d}  F_conf {v[0]:.2f} [{lo:.2f}, {hi:.2f}]  "
                  f"CNX chết {out[name]['frac_cnx_killed']:.2f} | sống 0.3–0.9 {out[name]['frac_0.3-0.9']:.2f} | ≥0.9 {out[name]['frac_ge_0.9']:.2f}")
        res[g] = out
    ok = res["RES3"]["survivors"]["ci95"][0] >= 0.25
    res["verdict"] = "MỞ pilot logit-margin (RES3)" if ok else "KHÔNG mở (survivor RES3 không chủ yếu là object CNX tự tin)"
    print("T1 VERDICT (khóa):", res["verdict"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
