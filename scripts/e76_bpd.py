#!/usr/bin/env python3
"""E76-BPD (progress_log 2026-10-31): Backward Path Diversity — DIAGNOSTIC cho DC-BPD, gate-first, KHÔNG dựng method.
RQ: từ MỘT detector pretrained, đổi CÁCH lan truyền backward (không teacher/target/fine-tune) có tạo DIRECTIONAL DIVERSITY
hữu ích — tức một backward-variant/tổ hợp phủ hướng-dễ-transfer của họ backbone YẾU tốt hơn gradient chuẩn không?
Khác E75 (chỉ reweight objective-gradient, đã NO-GO): E76 thêm backward-behavior MỚI = gating gradient theo STAGE backbone
(shallow/deep emphasis, kiểu SGM nhưng arch-agnostic qua tensor-grad-hook trên output từng stage). Đo CẢ HAI neo single-surrogate:
  - r50     -> target yếu: swin_t, dino_swin_l, convnext_t   (kiểm lại trần R50->Transformer)
  - convnext_t -> target yếu: r101, dino_r50                 (hướng hứa hẹn: phục hồi ResNet, CNN-generic)
Oracle = cls-grad target (CHỈ phân tích, như E68/E75). dev30, ảnh sạch x0, không sinh attack.
  python scripts/e76_bpd.py   -> results/mechanism/e76_bpd.json"""
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch
from attack.targets import kind, load_any, select_terms, target_dataset, target_grads

OUT = os.path.join(REPO_ROOT, "results/mechanism/e76_bpd.json")
ANCHORS = {"r50": ["swin_t", "dino_swin_l", "convnext_t"],
           "convnext_t": ["r101", "dino_r50"]}
ALL_TGT = ["swin_t", "dino_swin_l", "convnext_t", "r101", "dino_r50"]
VARIANTS = ["std", "rpn", "roi", "shallow", "deep"]   # std=task-sum; rpn/roi=head-path; shallow/deep=stage-gated backward
BOOT_SEED, N_BOOT = 2026, 1000


def unit(a):
    return a / (a.norm() + 1e-12)


def cos(a, b):
    return float(unit(a.double()) @ unit(b.double()))


def _loss_on(model, x, ds, stage_scale=None):
    """Forward task loss trên x (requires_grad); nếu stage_scale (list) -> hook nhân grad output từng stage backbone (SGM-arch-agnostic)."""
    hs = []
    if stage_scale is not None:
        def fhook(m, i, o):
            outs = o if isinstance(o, (list, tuple)) else [o]
            for k, f in enumerate(outs):
                if isinstance(f, torch.Tensor) and f.requires_grad and k < len(stage_scale):
                    f.register_hook(lambda g, s=float(stage_scale[k]): g * s)
        hs.append(model.backbone.register_forward_hook(fhook))
    inputs, dss = to_batch(model, [resize_to_model(x, tuple(ds["data_sample"].img_shape))], [ds["data_sample"]])
    losses = model.loss(inputs, dss)
    for h in hs:
        h.remove()
    return losses


def anchor_variants(model, key, orig, ds):
    """{variant: ∇_x0 [D] double} cho 1 anchor. rng reset mỗi backward. param đóng băng."""
    k = kind(key)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    st = torch.get_rng_state(), torch.cuda.get_rng_state()
    nstage = len(model.backbone(to_batch(model, [resize_to_model(orig, tuple(ds["data_sample"].img_shape))],
                                          [ds["data_sample"]])[0]))
    SH = [2.0, 1.0, 0.5, 0.25][:nstage]        # shallow emphasis
    DP = [0.25, 0.5, 1.0, 2.0][-nstage:]       # deep emphasis
    specs = {"std": (("task",), None), "rpn": (None, None), "roi": (None, None),
             "shallow": (("task",), SH), "deep": (("task",), DP)}
    out = {}
    for v, (_, scale) in specs.items():
        torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
        x = orig.clone().requires_grad_(True)
        losses = _loss_on(model, x, ds, stage_scale=scale)
        if v == "rpn":
            sel = {n: losses[n] for n in losses if "loss" in n and ("rpn" in n or "obj" in n)}
            sel = sel or select_terms(losses, k, "cls")
        elif v == "roi":
            sel = {n: losses[n] for n in losses if "loss" in n and "rpn" not in n and "obj" not in n and "mask" not in n}
        else:
            sel = {n: losses[n] for n in losses if "loss" in n}
        tot = None
        for val in sel.values():
            val = sum(val) if isinstance(val, (list, tuple)) else val
            tot = val if tot is None else tot + val
        tot.backward()
        out[v] = x.grad.detach().flatten().double()
        del losses
    return out


def coverage(A, g):
    AAt = A @ A.t()
    w = torch.linalg.solve(AAt + 1e-9 * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), A @ g)
    return float((A.t() @ w).norm() / (g.norm() + 1e-12))


def main():
    dev = "cuda:0"
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    ds_a = {a: {s["img_id"]: s for s in target_dataset(a, ids)} for a in ANCHORS}
    ds_t = {m: {s["img_id"]: s for s in target_dataset(m, ids)} for m in ALL_TGT}

    # rec[anchor][target][variant] = [cos per image]; + coverage, pairwise diversity
    rec = {a: {t: {v: [] for v in VARIANTS} for t in ALL_TGT} for a in ANCHORS}
    cov = {a: {t: [] for t in ANCHORS[a]} for a in ANCHORS}
    div = {a: [] for a in ANCHORS}  # mean pairwise cos giữa các variant (thấp = đa dạng)
    for n, iid in enumerate(ids):
        gvar = {}
        for a in ANCHORS:
            m = load_model(a, dev) if a in ("r50", "convnext_t") else load_any(a, dev)
            s = ds_a[a][iid]; orig = s["orig_inputs"].to(dev).float()
            gvar[a] = anchor_variants(m, a, orig, s)
            del m; torch.cuda.empty_cache()
            pv = [cos(gvar[a][VARIANTS[i]], gvar[a][VARIANTS[j]]) for i in range(len(VARIANTS)) for j in range(i + 1, len(VARIANTS))]
            div[a].append(float(np.mean(pv)))
        for t in ALL_TGT:
            model = load_any(t, dev); sm = ds_t[t][iid]; om = sm["orig_inputs"].to(dev).float()
            grads, _, _ = target_grads(model, t, om, sm["data_sample"], objectives=("cls",))
            gT = grads["cls"].flatten().double().to(dev)
            del model; torch.cuda.empty_cache()
            for a in ANCHORS:
                for v in VARIANTS:
                    rec[a][t][v].append(cos(gvar[a][v], gT))
                if t in ANCHORS[a]:
                    A = torch.stack([gvar[a][v] for v in VARIANTS])
                    cov[a][t].append(coverage(A, gT))
        print(f"[e76] {n + 1}/30", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, 30, size=(N_BOOT, 30))

    def summ(v):
        v = np.array(v); bs = v[boot].mean(1)
        return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    res = {"n_images": 30, "variants": VARIANTS, "oracle": "target cls-grad @ x0 (analysis-only)", "anchors": {}}
    for a in ANCHORS:
        res["anchors"][a] = {"pairwise_cos_variants": summ(div[a]), "targets": {}}
        for t in ANCHORS[a]:
            d = {v: summ(rec[a][t][v]) for v in VARIANTS}
            res["anchors"][a]["targets"][t] = {"cos": d, "coverage": summ(cov[a][t])}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    print("\n=== E76 BPD ===")
    for a in ANCHORS:
        pdiv = res["anchors"][a]["pairwise_cos_variants"][0]
        print(f"\nANCHOR {a}  | pairwise-cos variants = {pdiv:.3f} ({'ĐA DẠNG' if pdiv < 0.8 else 'COLLINEAR -> ít đa dạng'})")
        print(f"  {'target':12}" + "".join(v.rjust(8) for v in VARIANTS) + f"{'COVER':>8}{'best-std':>9}")
        for t in ANCHORS[a]:
            tg = res["anchors"][a]["targets"][t]
            cstd = tg["cos"]["std"][0]; cov0 = tg["coverage"][0]
            bestv = max(tg["cos"][v][0] for v in VARIANTS)
            print(f"  {t:12}" + "".join(f"{tg['cos'][v][0]:8.3f}" for v in VARIANTS) +
                  f"{cov0:8.3f}{cov0 - cstd:+9.3f}")
    print("\n--- GO/NO-GO (khóa trước khi đọc) ---")
    go = []
    for a in ANCHORS:
        pdiv = res["anchors"][a]["pairwise_cos_variants"][0]
        diverse = pdiv < 0.80
        for t in ANCHORS[a]:
            tg = res["anchors"][a]["targets"][t]
            hr = tg["coverage"][0] - tg["cos"]["std"][0]
            bestvar = max((tg["cos"][v][0] for v in VARIANTS if v != "std"), default=0) - tg["cos"]["std"][0]
            g = diverse and (hr >= 0.10)
            print(f"{a:11}->{t:12} diverse={diverse} headroom(cover-std)={hr:+.3f} best-variant-gain={bestvar:+.3f} GO={g}")
            if g:
                go.append(f"{a}->{t}")
    print("\nVERDICT:",
          "E76 GO @ " + ",".join(go) + " — backward-path diversification tạo được directional diversity HỮU ÍCH; đáng xây DC-BPD (routing)"
          if go else
          "E76 NO-GO — đổi backward-path KHÔNG tạo hướng phủ target yếu tốt hơn gradient chuẩn (dù oracle); DC-BPD routing không có đòn bẩy -> dừng, không chỉnh heuristic")


if __name__ == "__main__":
    main()
