"""E86 — PILOT 3c (thăm dò, KHÔNG làm bằng chứng): alignment gradient có theo hiệu ứng stage của E84/E85 không?
20 ảnh đầu của x200 (dev300 \\ dev100; KHÔNG dùng dev100 để giữ sạch cho 3c thật). Ảnh SẠCH, không aug.
Surrogate {r50, convnext_t, r101, swin_t}: ∇ₓ L_OSFD (k=3, backbone) với stage C2–C5 | C3–C5 | chỉ C2.
Target T5C: ∇ₓ L_cls (attack.targets.target_grads, như E68). cos theo ảnh gốc, TB qua ảnh, bootstrap 1000 (seed 2026).
Dự đoán thăm dò: Δcos(C35 − C25) lớn nhất ở CNX, ≈ 0 ở R50/R101 (khớp stage effect E85: CNX +9.8, SWT +2.8, R50/R101 ≈ 0).
  python scripts/e86_pilot_align.py  -> results/mechanism/e86_pilot_align.json"""
import json, os, sys
import numpy as np, torch
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from attack.data import load_image_ids
import attack.targets as _tg
from attack.targets import load_any, target_dataset, target_grads
_tg.GEN.setdefault("fcos_r50", ("fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py", "fcos_center-normbbox*"))
_tg.GEN.setdefault("yolox_l", ("yolox/yolox_l_8xb8-300e_coco.py", "yolox_l_8x8*"))   # = scripts/eval_generalization.py
from attack.losses.osfd import make_osfd_loss_fn
from attack.preprocess import resize_to_model

SURS = {"r50": "R50", "convnext_t": "CNX", "r101": "R101", "swin_t": "SWT"}
STG = {"C25": None, "C35": [1, 2, 3], "C2": [0]}
T5C = ["fcos_r50", "dino_r50", "yolov3_d53", "yolox_l", "dino_swin_l"]
N, BOOT_SEED = int(os.environ.get("E86_N", 20)), 2026


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d100 = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i not in d100][:N]
    G = {}
    for m in T5C:
        model = load_any(m, "cuda:0"); ds = {s["img_id"]: s for s in target_dataset(m, ids)}
        for i in ids:
            g, _, _ = target_grads(model, m, ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"], objectives=("cls",))
            G[(m, i)] = g["cls"].cpu()
        del model; torch.cuda.empty_cache(); print(f"[e86] target {m} xong", flush=True)
    for s in SURS:
        model = load_any(s, "cuda:0"); model.eval(); ds = {x["img_id"]: x for x in target_dataset(s, ids)}
        for p in model.parameters(): p.requires_grad_(False)
        for i in ids:
            orig, dsm = ds[i]["orig_inputs"].cuda().float(), ds[i]["data_sample"]
            clean = resize_to_model(orig, tuple(dsm.img_shape))
            for st, stages in STG.items():
                lf = make_osfd_loss_fn(model, clean, dsm, k=3.0, stages=stages)
                x = orig.clone().requires_grad_(True)
                lf(model, resize_to_model(x, tuple(dsm.img_shape)), dsm).backward()
                G[(s, st, i)] = x.grad.detach().cpu()
        del model; torch.cuda.empty_cache(); print(f"[e86] surrogate {s} xong", flush=True)
    # cos[sur][st][target] = vector theo ảnh
    C = {s: {st: {t: np.array([cos(G[(s, st, i)], G[(t, i)]) for i in ids]) for t in T5C} for st in STG} for s in SURS}
    boot = np.random.default_rng(BOOT_SEED).integers(0, N, size=(1000, N))
    summ = lambda v: [round(float(v.mean()), 4), round(float(np.percentile(v[boot].mean(1), 2.5)), 4),
                      round(float(np.percentile(v[boot].mean(1), 97.5)), 4)]
    res = {"n_images": N, "images": "x200[:N]", "note": "[point, lo, hi]; cos(∇ surrogate L_OSFD(stage), ∇ target L_cls), ảnh sạch"}
    print(f"\n=== E86 pilot @ x200[:{N}] — cos TB qua T5C [CI95]; Δ = C35 − C25 ===")
    for s, p in SURS.items():
        avg = {st: np.mean([C[s][st][t] for t in T5C], 0) for st in STG}
        res[p] = {"T5C": {st: summ(avg[st]) for st in STG}, "delta_C35_C25_T5C": summ(avg["C35"] - avg["C25"]),
                  "dino_swin_l": {st: summ(C[s][st]["dino_swin_l"]) for st in STG},
                  "delta_C35_C25_dino_swin_l": summ(C[s]["C35"]["dino_swin_l"] - C[s]["C25"]["dino_swin_l"]),
                  "per_target_delta": {t: summ(C[s]["C35"][t] - C[s]["C25"][t]) for t in T5C},
                  "norm_ratio_C2_over_C25": summ(np.array([float(G[(s, "C2", i)].norm() / (G[(s, "C25", i)].norm() + 1e-12)) for i in ids])),
                  "cos_C2_C35": summ(np.array([cos(G[(s, "C2", i)], G[(s, "C35", i)]) for i in ids]))}
        f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
        print(f"{p:5} C25 {f(res[p]['T5C']['C25'])}  C35 {f(res[p]['T5C']['C35'])}  C2 {f(res[p]['T5C']['C2'])}  "
              f"Δ {f(res[p]['delta_C35_C25_T5C'])} | SwinL Δ {f(res[p]['delta_C35_C25_dino_swin_l'])} | "
              f"‖∇C2‖/‖∇C25‖ {f(res[p]['norm_ratio_C2_over_C25'])} cos(∇C2,∇C35) {f(res[p]['cos_C2_C35'])}")
    out = os.path.join(REPO_ROOT, "results/mechanism/e86_pilot_align.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False); print(f"[e86] -> {out}")


if __name__ == "__main__":
    main()
