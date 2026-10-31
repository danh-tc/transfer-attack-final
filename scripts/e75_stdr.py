#!/usr/bin/env python3
"""E75-STDR (progress_log 2026-10-31): Single-Surrogate Transfer Direction Recovery — DIAGNOSTIC 2 cổng (accessibility +
recoverability), KHÔNG dựng method. Giả thuyết MỚI (E68 CHƯA kiểm): trong tập input-gradient theo TỪNG objective của R50
{loss_rpn_cls, loss_rpn_bbox, loss_cls, loss_bbox} (tại ảnh SẠCH x0, khung ảnh gốc), có TỔ HỢP TUYẾN TÍNH nào phủ hướng-dễ-
transfer của target họ khác (ConvNeXt/Swin/Swin-L) TỐT HƠN hướng R50 chuẩn (task-sum) không? (gate-1 accessibility, dùng oracle
= gradient cls của target CHỈ để phân tích) — và chọn được tổ hợp đó CHỈ bằng thông tin R50 (không target) không? (gate-2
recoverability). Oracle KHÔNG bao giờ vào việc chọn hyperparameter/method. dev30, forward+backward, không sinh attack.
  python scripts/e75_stdr.py   -> results/mechanism/e75_stdr.json"""
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
from attack.targets import load_any, target_dataset, target_grads

OUT = os.path.join(REPO_ROOT, "results/mechanism/e75_stdr.json")
TERMS = ["loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox"]   # 4 objective-gradient của R50
TARGETS = ["convnext_t", "swin_t", "dino_swin_l", "r101", "dino_r50"]  # 3 bridge/transformer + 2 ResNet-control
TRANS = ["convnext_t", "swin_t", "dino_swin_l"]
BOOT_SEED, N_BOOT = 2026, 1000


def unit(a):
    return a / (a.norm() + 1e-12)


def cos(a, b):
    return float(unit(a.double()) @ unit(b.double()))


def r50_term_grads(model, orig, ds):
    """{term: ∇_x0 term} theo ảnh gốc [D] (double), + g_task = tổng mọi term. 1 forward/term, param đóng băng, rng reset."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    st = torch.get_rng_state(), torch.cuda.get_rng_state()
    out = {}
    for t in TERMS:
        torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
        x = orig.clone().requires_grad_(True)
        inputs, dss = to_batch(model, [resize_to_model(x, tuple(ds["data_sample"].img_shape))], [ds["data_sample"]])
        losses = model.loss(inputs, dss)
        v = losses[t]
        v = sum(v) if isinstance(v, (list, tuple)) else v
        v.backward()
        out[t] = x.grad.detach().flatten().double()
        del losses, inputs
    out["task"] = sum(out[t] for t in TERMS)   # hướng R50 chuẩn (gộp objective)
    return out


def coverage_and_w(A, g):
    """Phủ của g lên span các hàng A (mỗi hàng 1 gradient): coverage=||proj||/||g||, w*=hệ số chuẩn-hoá L2.
    Giải bình phương tối thiểu w = (A A^T)^-1 A g (hệ 4x4), proj = A^T w."""
    AAt = A @ A.t()
    Ag = A @ g
    w = torch.linalg.solve(AAt + 1e-9 * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), Ag)
    proj = A.t() @ w
    cover = float(proj.norm() / (g.norm() + 1e-12))
    wn = (w / (w.norm() + 1e-12)).cpu().numpy()
    return cover, wn


def main():
    dev = "cuda:0"
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    ds_r50 = {s["img_id"]: s for s in target_dataset("r50", ids)}
    ds_tgt = {m: {s["img_id"]: s for s in target_dataset(m, ids)} for m in TARGETS}

    rec = {m: {"cos_task": [], "cos_clsterm": [], "cos_bestsingle": [], "coverage": [],
               "cos_normsel": [], "w": []} for m in TARGETS}
    for n, iid in enumerate(ids):
        # --- gradient R50 theo từng term (ảnh sạch) ---
        r50 = load_model("r50", dev)
        s = ds_r50[iid]
        orig = s["orig_inputs"].to(dev).float()
        gR = r50_term_grads(r50, orig, s)
        del r50; torch.cuda.empty_cache()
        A = torch.stack([gR[t] for t in TERMS])          # 4 x D
        g_task = gR["task"]; g_clsterm = gR["loss_cls"]
        norms = torch.stack([gR[t].norm() for t in TERMS])
        g_normsel = (A * (norms / norms.sum()).unsqueeze(1)).sum(0)  # selector CHỈ-R50: trọng số ∝ ||g_k||, không target
        # --- oracle: gradient cls của target (CHỈ để phân tích) ---
        for m in TARGETS:
            model = load_any(m, dev)
            sm = ds_tgt[m][iid]
            om = sm["orig_inputs"].to(dev).float()
            grads, _, _ = target_grads(model, m, om, sm["data_sample"], objectives=("cls",))
            gT = grads["cls"].flatten().double().to(dev)
            del model; torch.cuda.empty_cache()
            cover, w = coverage_and_w(A, gT)
            singles = [cos(gR[t], gT) for t in TERMS]
            rec[m]["cos_task"].append(cos(g_task, gT))
            rec[m]["cos_clsterm"].append(cos(g_clsterm, gT))
            rec[m]["cos_bestsingle"].append(max(singles))
            rec[m]["coverage"].append(cover)
            rec[m]["cos_normsel"].append(cos(g_normsel, gT))
            rec[m]["w"].append(w.tolist())
        print(f"[e75] {n + 1}/30", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, 30, size=(N_BOOT, 30))

    def summ(v):
        v = np.array(v); bs = v[boot].mean(1)
        return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    res = {"n_images": 30, "terms": TERMS, "oracle": "target cls-grad @ x0 (analysis-only)", "drop": {}}
    for m in TARGETS:
        W = np.array(rec[m]["w"])  # 30 x 4 (chuẩn-hoá L2 mỗi ảnh)
        res["drop"][m] = {
            "cos_task": summ(rec[m]["cos_task"]),
            "cos_clsterm": summ(rec[m]["cos_clsterm"]),
            "cos_bestsingle": summ(rec[m]["cos_bestsingle"]),
            "coverage": summ(rec[m]["coverage"]),
            "cos_normsel_R50only": summ(rec[m]["cos_normsel"]),
            "w_mean": [round(x, 3) for x in W.mean(0)],
            "w_std": [round(x, 3) for x in W.std(0)],
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    # --- in bảng + GO/NO-GO (khóa trước) ---
    print(f"\n{'target':12}{'cos_task':>9}{'cos_cls':>9}{'best1':>8}{'COVER':>8}{'headroom':>9}{'normsel':>9}")
    for m in TARGETS:
        d = res["drop"][m]
        base = max(d["cos_task"][0], d["cos_bestsingle"][0])
        hr = d["coverage"][0] - base
        print(f"{m:12}{d['cos_task'][0]:9.3f}{d['cos_clsterm'][0]:9.3f}{d['cos_bestsingle'][0]:8.3f}"
              f"{d['coverage'][0]:8.3f}{hr:+9.3f}{d['cos_normsel_R50only'][0]:9.3f}")
        print(f"{'  w_mean':12}{str(d['w_mean'])}  w_std {d['w_std']}")

    print("\n--- GO/NO-GO (khóa trước khi đọc) ---")
    go1 = go2 = []
    go_trans = []
    for m in TRANS:
        d = res["drop"][m]
        base = max(d["cos_task"][0], d["cos_bestsingle"][0])
        hr = d["coverage"][0] - base
        g1 = (hr >= 0.10) and (d["coverage"][0] >= 0.30)     # gate-1 accessibility
        rec_frac = (d["cos_normsel_R50only"][0] - base) / hr if hr > 1e-6 else 0.0
        g2 = g1 and (rec_frac >= 0.50)                        # gate-2 recoverability (selector CHỈ-R50)
        print(f"{m:12} gate1(cover≥base+.10 & ≥.30)={g1}  headroom {hr:+.3f}  recover_frac {rec_frac:+.2f}  gate2={g2}")
        if g1 and g2:
            go_trans.append(m)
    print("\nVERDICT:",
          "STDR GO @ " + ",".join(go_trans) + " — có hướng transfer TRONG subspace objective R50 VÀ chọn được không-oracle"
          if go_trans else
          "STDR NO-GO — hướng ConvNeXt/Swin KHÔNG nằm (đủ/ổn định/chọn-được) trong subspace objective-gradient R50; "
          "single-surrogate recovery không khả thi từ objective-gradients -> cần ensemble / distillation")


if __name__ == "__main__":
    main()
