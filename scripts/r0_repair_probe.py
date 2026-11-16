"""R0 (progress_log 2026-11-16 19:50, KHÓA): kiểm giả thuyết REPAIR trên DINO-Swin-L bằng can thiệp feature
phía target (chỉ forward, chẩn đoán — không sinh/tune attack).

Hook đầu ra backbone (trước neck). Mỗi ảnh: cache F_c (sạch), F_a (adv); các ô lai thay TOÀN BỘ đầu ra backbone
bằng tổ hợp từ cache (input forward = ảnh adv, không ảnh hưởng vì output bị thay).
  python scripts/r0_repair_probe.py            # dev30, RD-CNX@50 + RD-R50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import CocoImageEval, bootstrap_indices, summarize, to_coco_dets
from scripts.e64_cnx_drop import load

ANN = os.path.join(REPO_ROOT, "data/coco/annotations/instances_val2017.json")
IMGDIR = os.path.join(REPO_ROOT, "data/coco/val2017")
RUN = os.path.join(REPO_ROOT, "artifacts/runs/f1b_r1a_eps5_3090/adv")
ATTACKS = {"RD-CNX@50": ("RD-CNX-B100", "snap50"), "RD-R50": ("RD-R50-B36", "final")}
TARGETS = ["dino_swin_l", "dino_r50", "swin_t"]
BOOT_SEED = 2026
OUT = os.path.join(REPO_ROOT, "results/mechanism/r0_repair_dev30.json")


def noise_like(f, rho, gen):
    n = torch.randn(f.shape, generator=gen, device="cpu").to(f.device, f.dtype)
    return n * (rho * f.norm() / (n.norm() + 1e-12))


def run_target(key, coco, ids, cat_ids):
    from mmdet.apis import inference_detector
    model = load(key)
    st = {"mode": None}

    def hook(m, i, o):
        if st["mode"] is None:
            st["last"] = tuple(f.detach() for f in o)
            return o
        return tuple(st["mode"])

    h = model.backbone.register_forward_hook(hook)
    stages = [f"C{j + 2}" for j in model.backbone.out_indices]
    shallow = [j for j, s in enumerate(stages) if s in ("C2", "C3")]
    s3 = [stages.index("C3")]
    deep = [j for j in range(len(stages)) if j not in shallow]
    print(f"[r0] {key}: stages {stages} | shallow {[stages[j] for j in shallow]} | deep {[stages[j] for j in deep]}", flush=True)
    dets, rhos = {}, {}
    add = lambda c, r, i: dets.setdefault(c, []).extend(to_coco_dets(r, i, cat_ids))
    with torch.no_grad():
        for i in ids:
            st["mode"] = None
            add("clean", inference_detector(model, os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])), i)
            Fc = st["last"]
            for a, (cell, sub) in ATTACKS.items():
                png = os.path.join(RUN, cell, sub, f"{i}.png")
                st["mode"] = None
                add(f"{a}|adv", inference_detector(model, png), i)
                Fa = st["last"]
                D = [float((x - y).float().norm() / y.float().norm()) for x, y in zip(Fa, Fc)]
                rho = float(np.mean([D[j] for j in deep]))
                rhos.setdefault(a, []).append(D)
                gen = torch.Generator().manual_seed(int(i))
                N = [noise_like(f, rho, gen) for f in Fc]

                def mk(deep_adv, noise_idx):
                    out = []
                    for j in range(len(stages)):
                        if j in deep:
                            f = Fa[j] if deep_adv else Fc[j]
                        else:
                            f = Fc[j]
                        if j in noise_idx:
                            f = f + N[j]
                        out.append(f)
                    return out

                cells = {"DaSc": mk(True, []), "DaSn": mk(True, shallow), "DcSn": mk(False, shallow),
                         "DaSn3": mk(True, s3), "DcSn3": mk(False, s3)}
                for j, s in enumerate(stages):
                    cells[f"rest_{s}"] = [Fc[k] if k == j else Fa[k] for k in range(len(stages))]
                for c, feats in cells.items():
                    st["mode"] = feats
                    add(f"{a}|{c}", inference_detector(model, png), i)
            print(f"[r0] {key} img {ids.index(i) + 1}/{len(ids)}", flush=True)
    h.remove()
    del model
    torch.cuda.empty_cache()
    return dets, stages, {a: np.array(v).mean(0).tolist() for a, v in rhos.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:args.n]
    coco = COCO(ANN)
    cat_ids = sorted(coco.getCatIds())
    W = np.vstack([np.ones((1, len(ids))), bootstrap_indices(len(ids), args.boot, BOOT_SEED)])
    S, meta = {}, {}
    for key in TARGETS:
        cache = os.path.join(REPO_ROOT, f"results/r0/dets_{key}_n{len(ids)}.json")
        if os.path.exists(cache):
            c_ = json.load(open(cache))
            dets, stages, D = c_["dets"], c_["stages"], c_["D"]
        else:
            dets, stages, D = run_target(key, coco, ids, cat_ids)
            json.dump({"dets": dets, "stages": stages, "D": D}, open(cache, "w"))
        meta[key] = {"stages": stages, "mean_D_adv": D}
        ev = CocoImageEval(coco, dets["clean"], ids)   # dựng 1 lần, bootstrap qua trọng số
        clean = np.array([ev.ap(w)[0] for w in W])
        for c in dets:
            if c == "clean":
                continue
            ev = CocoImageEval(coco, dets[c], ids)
            adv = np.array([ev.ap(w)[0] for w in W])
            S[f"{key}|{c}"] = 100 * (clean - adv) / clean
        meta[key]["clean_AP"] = float(clean[0] * 100)
    res = {"n": len(ids), "boot": args.boot, "gpu": torch.cuda.get_device_name(0), "meta": meta, "drop": {}, "I": {}}
    fmt = lambda s: f"{s['point']:+6.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    ci = lambda v: summarize(v[0], v[1:])
    for a in ATTACKS:
        print(f"\n=== {a} — relative AP drop % (dev{len(ids)}) ===")
        for key in TARGETS:
            cs = [c.split("|", 2)[2] for c in S if c.startswith(f"{key}|{a}|")]
            res["drop"].setdefault(a, {})[key] = {c: ci(S[f"{key}|{a}|{c}"]) for c in cs}
            print(f"{key:12} " + "  ".join(f"{c} {S[f'{key}|{a}|{c}'][0]:.1f}" for c in cs))
            print(f"{'':12} D_k adv {[round(x, 3) for x in meta[key]['mean_D_adv'][a]]}")
        I = {}
        for key in TARGETS:
            g = lambda c: S[f"{key}|{a}|{c}"]
            I[key] = {"I_rep": g("DaSn") - g("DaSc") - g("DcSn"), "I_rep3": g("DaSn3") - g("DaSc") - g("DcSn3")}
        out = {k: {m: ci(v) for m, v in d.items()} for k, d in I.items()}
        out["SwinL-R50 I_rep3"] = ci(I["dino_swin_l"]["I_rep3"] - I["dino_r50"]["I_rep3"])
        out["SwinL-SwinT I_rep3"] = ci(I["dino_swin_l"]["I_rep3"] - I["swin_t"]["I_rep3"])
        res["I"][a] = out
        for k, v in out.items():
            print(f"{k:20} " + ("  ".join(f"{m} {fmt(s)}" for m, s in v.items()) if "point" not in v else fmt(v)))
    p1, p2 = res["I"]["RD-CNX@50"]["SwinL-R50 I_rep3"]["point"], res["I"]["RD-CNX@50"]["SwinL-SwinT I_rep3"]["point"]
    res["verdict"] = "H_rep ỦNG HỘ (xu hướng) -> khóa xác nhận dev100" if p1 >= 5 and p2 >= 5 else "KHÔNG ủng hộ -> DỪNG hướng repair"
    print(f"\nR0 VERDICT (khóa): {res['verdict']}  (SwinL−R50 {p1:+.1f}, SwinL−SwinT {p2:+.1f}; ngưỡng +5 cả hai)")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[r0] -> {OUT}")


if __name__ == "__main__":
    main()
