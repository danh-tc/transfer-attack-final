"""B3+ (progress_log 2026-11-15): kiểm TIỀN ĐỀ truy xuất — hướng gradient R50 "partner" có chưng cất được vào bộ nhớ
offline đánh index theo feature CNX không? CHẨN ĐOÁN tại x0, dev100, chỉ đo cos, KHÔNG sinh attack.
Vấn đề hình học: gradient từng ảnh ở không gian pixel KHÁC kích thước -> đưa tất cả về lưới chuẩn S=512 (resize bilinear)
rồi mới cos/truy xuất. Khóa truy xuất = embedding CNX (C5 global-avg-pool, 768-d, bất biến kích thước) của ẢNH SẠCH
(không rò rỉ target). Bộ nhớ = {embedding_j -> unit(grad R50_j)}, leave-one-out (bỏ chính ảnh i).
Partner (4 cách): live = R50 thật ảnh i (positive control, trần); gmean = mean toàn cục unit(R50); cmean = mean theo lớp
(lớp = category object GT lớn nhất); knn = mean k láng giềng theo embedding CNX (k=10 khóa; báo thêm 5/20).
combo(p) = cos(unit(g_cnx_std)+unit(partner), g_T) ở lưới chuẩn; gain = combo − base. RES3 = TB{r101, dino_r50}.
GO(knn) (khóa, bootstrap ảnh 1000 seed 2026): pos-control gain_RES3(live) CI>0; VÀ gain_RES3(knn) CI>0 VÀ
  gain_RES3(knn) >= 0.5·gain_RES3(live) VÀ combo_SwinL(knn) >= combo_SwinL(live). Báo thêm cos(partner, R50_true).
  python scripts/b3_retrieval.py -> results/mechanism/b3_retrieval_dev100.json"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pycocotools.coco import COCO
from attack.data import load_image_ids
from attack.models import load_model
from attack.preprocess import resize_to_model, to_batch
from attack.targets import load_any, target_dataset, target_grads
from scripts.e76_bpd import _loss_on, unit
from scripts.f1b_r1a import ANN

OUT = os.path.join(REPO_ROOT, "results/mechanism/b3_retrieval_dev100.json")
TARGETS = ["r101", "dino_r50", "dino_swin_l"]
RES3 = ["r101", "dino_r50"]
S, KS, K_LOCK, BOOT_SEED, N_BOOT = 512, (5, 10, 20), 10, 2026, 1000


def canon(g):
    """grad [C,h,w] -> vector lưới chuẩn [3*S*S] double cpu."""
    return F.interpolate(g[None].float(), size=(S, S), mode="bilinear", align_corners=False).flatten().double().cpu()


def cos(a, b):
    return float(unit(a) @ unit(b))


def cnx_task_grad_and_emb(model, orig, ds, state):
    torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    inp = to_batch(model, [resize_to_model(orig, tuple(ds["data_sample"].img_shape))], [ds["data_sample"]])[0]
    with torch.no_grad():
        emb = model.backbone(inp)[-1].mean((-1, -2)).flatten().double().cpu()
    x = orig.clone().requires_grad_(True)
    losses = _loss_on(model, x, ds)
    tot = None
    for n, val in losses.items():
        if "loss" in n:
            val = sum(val) if isinstance(val, (list, tuple)) else val
            tot = val if tot is None else tot + val
    tot.backward()
    return canon(x.grad.detach()), emb


def r50_task_grad(model, orig, ds, state):
    torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    x = orig.clone().requires_grad_(True)
    losses = _loss_on(model, x, ds)
    tot = None
    for n, val in losses.items():
        if "loss" in n:
            val = sum(val) if isinstance(val, (list, tuple)) else val
            tot = val if tot is None else tot + val
    tot.backward()
    return canon(x.grad.detach())


def main():
    dev = "cuda:0"
    coco = COCO(ANN)
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))
    n = len(ids)
    ds_c = {s["img_id"]: s for s in target_dataset("convnext_t", ids)}
    ds_r = {s["img_id"]: s for s in target_dataset("r50", ids)}
    ds_t = {t: {s["img_id"]: s for s in target_dataset(t, ids)} for t in TARGETS}
    # lớp đại diện = category object GT lớn nhất
    cls_of = {}
    for i in ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
        cls_of[i] = max(anns, key=lambda a: a["area"])["category_id"] if anns else -1

    Gc, Gr, E = [], [], []
    cnx = load_model("convnext_t", dev)
    for i in ids:
        s = ds_c[i]; o = s["orig_inputs"].to(dev).float(); st = (torch.get_rng_state(), torch.cuda.get_rng_state())
        g, e = cnx_task_grad_and_emb(cnx, o, s, st)
        Gc.append(g); E.append(e); torch.cuda.empty_cache()
    del cnx; torch.cuda.empty_cache()
    r50 = load_model("r50", dev)
    for i in ids:
        s = ds_r[i]; o = s["orig_inputs"].to(dev).float(); st = (torch.get_rng_state(), torch.cuda.get_rng_state())
        Gr.append(r50_task_grad(r50, o, s, st)); torch.cuda.empty_cache()
    del r50; torch.cuda.empty_cache()
    print("[b3] pass A xong (grad+emb)", flush=True)

    Uc = [unit(g) for g in Gc]; Ur = [unit(g) for g in Gr]
    Emb = torch.stack([unit(e) for e in E])                         # [n,768]
    sim = Emb @ Emb.t(); sim.fill_diagonal_(-2.0)                    # loại chính ảnh
    nn_idx = {k: sim.topk(k, dim=1).indices for k in KS}
    cls = np.array([cls_of[i] for i in ids])

    def partner(kind, i, k=K_LOCK):
        if kind == "live":
            return Ur[i]
        if kind == "gmean":
            return unit(sum(Ur[j] for j in range(n) if j != i))
        if kind == "cmean":
            idx = [j for j in range(n) if j != i and cls[j] == cls[i]]
            return unit(sum(Ur[j] for j in idx)) if idx else unit(sum(Ur[j] for j in range(n) if j != i))
        if kind.startswith("knn"):
            k = int(kind[3:]); idx = nn_idx[k][i].tolist()
            return unit(sum(Ur[j] for j in idx))
        raise KeyError(kind)

    PA = ["live", "gmean", "cmean", *[f"knn{k}" for k in KS]]
    P = {kind: [partner(kind, i) for i in range(n)] for kind in PA}
    cos_true = {kind: [cos(P[kind][i], Ur[i]) for i in range(n)] for kind in PA}  # tái tạo hướng R50 thật

    base = {t: [] for t in TARGETS}
    combo = {t: {p: [] for p in PA} for t in TARGETS}
    for t in TARGETS:
        model = load_any(t, dev)
        for k, i in enumerate(ids):
            s = ds_t[t][i]; om = s["orig_inputs"].to(dev).float()
            gT = canon(target_grads(model, t, om, s["data_sample"], objectives=("cls",))[0]["cls"])
            base[t].append(cos(Gc[k], gT))
            for p in PA:
                combo[t][p].append(cos(Uc[k] + P[p][k], gT))
        del model; torch.cuda.empty_cache()
        print(f"[b3] target {t} xong", flush=True)

    rng = np.random.default_rng(BOOT_SEED); B = rng.integers(0, n, size=(N_BOOT, n))

    def summ(a):
        v = np.array(a, float); bs = v[B].mean(1)
        return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    grp = lambda d: np.mean([np.array(d[t]) for t in RES3], axis=0)
    res = {"n_images": n, "grid": S, "k_lock": K_LOCK, "cos_partner_vs_R50true": {p: summ(cos_true[p]) for p in PA},
           "res3": {}, "swinl_combo": {}}
    bR = grp(base)
    res["res3"]["base"] = summ(bR)
    for p in PA:
        res["res3"][f"gain_{p}"] = summ(grp({t: combo[t][p] for t in RES3}) - bR)
        res["swinl_combo"][p] = summ(combo["dino_swin_l"][p])

    print("\n=== B3+ retrieval cos @ dev100 (lưới 512, x0) ===")
    print(f"RES3 base cos = {res['res3']['base'][0]:+.3f}")
    print(f"{'partner':8}{'gain_RES3':>22}{'cos(partner,R50true)':>24}{'combo_SwinL':>20}")
    for p in PA:
        g = res["res3"][f"gain_{p}"]; c = res["cos_partner_vs_R50true"][p]; sw = res["swinl_combo"][p]
        print(f"{p:8}{g[0]:+7.3f} [{g[1]:+.3f},{g[2]:+.3f}]  {c[0]:+7.3f} [{c[1]:+.3f},{c[2]:+.3f}]  {sw[0]:+6.3f}")

    gl = res["res3"]["gain_live"]; gk = res["res3"][f"gain_knn{K_LOCK}"]
    pc = gl[1] > 0
    go = pc and gk[1] > 0 and gk[0] >= 0.5 * gl[0] and res["swinl_combo"][f"knn{K_LOCK}"][0] >= res["swinl_combo"]["live"][0]
    res["positive_control_live"] = bool(pc); res["GO_knn"] = bool(go)
    print(f"\npositive control (R50 live): {pc} | gain_live_RES3 {gl[0]:+.3f}  gain_knn{K_LOCK}_RES3 {gk[0]:+.3f} "
          f"(= {gk[0] / gl[0] * 100:.0f}% nếu live>0)")
    print("VERDICT:", f"B3+ GO — bộ nhớ kNN (theo feature CNX) tái tạo đủ hướng R50; đáng dựng retrieval-attack"
          if go else "B3+ NO-GO — truy xuất kNN KHÔNG lấy lại >=50% lợi ích R50; hướng R50 là per-image, không chưng cất được")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[b3] -> {OUT}")


if __name__ == "__main__":
    main()
