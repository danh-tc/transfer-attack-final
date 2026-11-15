"""B2 (progress_log 2026-11-15): chẩn đoán "inductive-bias morphing". CHỈ đo hướng gradient tại ảnh sạch x0 (dev30),
KHÔNG sinh perturbation, KHÔNG lưu ảnh. Dùng lại công cụ E76 (task-grad surrogate vs cls-grad target làm oracle phân tích).
Câu hỏi: biến thể ConvNeXt-T (không fine-tune) mang thiên kiến ResNet có đóng vai PARTNER hướng gradient cho RES3 giống
vai trò R50 trong ENS-WP không? Nếu KHÔNG -> morphing không có đòn bẩy, dừng; nếu CÓ -> đáng dựng attack.

Biến thể CNX (cố định, không train, chỉ đổi forward/backward cục bộ trong lúc đo):
  std       gốc
  k3 / k5   crop kernel depthwise 7x7 về tâm 3x3 / 5x5 (ngoài = 0), rescale theo kênh giữ chuẩn L2 (đổi forward)
  lnfix     LayerNorm2d chuẩn hoá bằng mean/var DETACH (forward trùng số, backward như chuẩn-hoá cố-định kiểu BN-eval)
  relugrad  GELU forward giữ nguyên, backward = 1[x>0]  (đối chứng prior art LinBP/BPA)
Đo/ảnh, partner p: base = cos(g_std, g_T); combo(p) = cos(unit(g_std)+unit(g_p), g_T); gain(p) = combo(p) - base.
Partner thật = R50 task-grad (positive control, tái lập vai trò ENS-WP). RES3 = TB {r101, fcos_r50, dino_r50}.
GO(v) (khóa, bootstrap ảnh 1000 seed 2026):
  pos-control gain_RES3(R50) CI>0; VÀ gain_RES3(v) CI>0 VÀ gain_RES3(v) >= 0.5*gain_RES3(R50) VÀ combo_SwinL(v) >= combo_SwinL(R50).
  python scripts/b2_morph_cos.py -> results/mechanism/b2_morph_cos.json"""
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.models import load_model
from attack.targets import load_any, target_dataset, target_grads
from scripts.e76_bpd import _loss_on, cos, unit

OUT = os.path.join(REPO_ROOT, "results/mechanism/b2_morph_cos.json")
MORPHS = ["k3", "k5", "lnfix", "relugrad"]        # std là mốc, R50 là partner positive-control
PARTNERS = [*MORPHS, "r50"]
TARGETS = ["r101", "dino_r50", "dino_swin_l"]
RES3 = ["r101", "dino_r50"]  # fcos_r50 không nối grad; gate cos dùng 2 target ResNet có grad
BOOT_SEED, N_BOOT = 2026, 1000


class _ReluGradGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.gelu(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x > 0).to(g.dtype)


class _ReluGELUMod(torch.nn.Module):
    def forward(self, x):
        return _ReluGradGELU.apply(x)


def _lnfix_forward(mod):
    """LayerNorm2d.forward nhưng mean/var bị DETACH (backward không qua thống kê chuẩn-hoá)."""
    def fwd(x, data_format="channel_first"):
        cf = data_format == "channel_first"
        if cf:
            x = x.permute(0, 2, 3, 1)
        mu = x.mean(-1, keepdim=True).detach()
        var = x.var(-1, unbiased=False, keepdim=True).detach()
        x = (x - mu) / torch.sqrt(var + mod.eps) * mod.weight + mod.bias
        if cf:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
    return fwd


@contextlib.contextmanager
def morph(model, v):
    from mmpretrain.models.utils.norm import LayerNorm2d
    bb = model.backbone
    blocks = [b for st in bb.stages for b in st]
    undo = []
    try:
        if v in ("k3", "k5"):
            k = int(v[1])
            for b in blocks:
                w = b.depthwise_conv.weight                     # [C,1,7,7]
                old = w.data.clone(); undo.append(lambda w=w, old=old: w.data.copy_(old))
                ksz = w.shape[-1]; o = (ksz - k) // 2
                m = torch.zeros_like(w); m[..., o:o + k, o:o + k] = 1.0
                cropped = w.data * m
                n0 = w.data.flatten(1).norm(dim=1); n1 = cropped.flatten(1).norm(dim=1).clamp_min(1e-12)
                w.data = cropped * (n0 / n1).view(-1, 1, 1, 1)
        elif v == "relugrad":
            for b in blocks:
                orig = b.act; undo.append(lambda b=b, orig=orig: setattr(b, "act", orig))
                b.act = _ReluGELUMod()
        elif v == "lnfix":
            for mdl in bb.modules():
                if isinstance(mdl, LayerNorm2d):
                    orig = mdl.forward; undo.append(lambda mdl=mdl, orig=orig: setattr(mdl, "forward", orig))
                    mdl.forward = _lnfix_forward(mdl)
        elif v != "std":
            raise KeyError(v)
        yield
    finally:
        for fn in reversed(undo):
            fn()


def task_grad(model, orig, ds, state):
    torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    x = orig.clone().requires_grad_(True)
    losses = _loss_on(model, x, ds)
    tot = None
    for n, val in losses.items():
        if "loss" not in n:
            continue
        val = sum(val) if isinstance(val, (list, tuple)) else val
        tot = val if tot is None else tot + val
    tot.backward()
    g = x.grad.detach().flatten().double().cpu()
    del losses
    return g


def main():
    dev = "cuda:0"
    keep = set(load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev30.csv")))
    ids = [i for i in load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev300.csv")) if i in keep]
    assert len(ids) == 30
    ds_c = {s["img_id"]: s for s in target_dataset("convnext_t", ids)}
    ds_r = {s["img_id"]: s for s in target_dataset("r50", ids)}
    ds_t = {t: {s["img_id"]: s for s in target_dataset(t, ids)} for t in TARGETS}

    # Pass A: grad surrogate (CNX std + morphs, R50) theo ảnh -> CPU
    G = {i: {} for i in ids}
    cnx = load_model("convnext_t", dev)
    for i in ids:
        s = ds_c[i]; orig = s["orig_inputs"].to(dev).float(); st = (torch.get_rng_state(), torch.cuda.get_rng_state())
        for v in ["std", *MORPHS]:
            with morph(cnx, v):
                G[i][v] = task_grad(cnx, orig, s, st)
        torch.cuda.empty_cache()
    del cnx; torch.cuda.empty_cache()
    r50 = load_model("r50", dev)
    for i in ids:
        s = ds_r[i]; orig = s["orig_inputs"].to(dev).float(); st = (torch.get_rng_state(), torch.cuda.get_rng_state())
        G[i]["r50"] = task_grad(r50, orig, s, st)
    del r50; torch.cuda.empty_cache()
    print("[b2] pass A xong (grad surrogate)", flush=True)

    # Pass B: mỗi target -> cls-grad oracle theo ảnh, cos ngay
    base = {t: [] for t in TARGETS}
    combo = {t: {p: [] for p in PARTNERS} for t in TARGETS}
    for t in TARGETS:
        model = load_any(t, dev)
        for i in ids:
            s = ds_t[t][i]; om = s["orig_inputs"].to(dev).float()
            grads, _, _ = target_grads(model, t, om, s["data_sample"], objectives=("cls",))
            gT = grads["cls"].flatten().double().cpu()
            gstd = G[i]["std"]; ustd = unit(gstd)
            base[t].append(cos(gstd, gT))
            for p in PARTNERS:
                combo[t][p].append(cos(ustd + unit(G[i][p]), gT))
        del model; torch.cuda.empty_cache()
        print(f"[b2] target {t} xong", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    B = rng.integers(0, 30, size=(N_BOOT, 30))

    def summ(arr):
        v = np.array(arr); bs = v[B].mean(1)
        return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    grp = lambda d: np.mean([np.array(d[t]) for t in RES3], axis=0)
    res = {"n_images": 30, "morphs": MORPHS, "base": {}, "combo": {}, "gain": {}, "res3": {}}
    for t in TARGETS:
        res["base"][t] = summ(base[t])
        res["combo"][t] = {p: summ(combo[t][p]) for p in PARTNERS}
        res["gain"][t] = {p: summ(np.array(combo[t][p]) - np.array(base[t])) for p in PARTNERS}
    base_r3 = grp(base)
    res["res3"]["base"] = summ(base_r3)
    res["res3"]["gain"] = {p: summ(grp({t: combo[t][p] for t in RES3}) - base_r3) for p in PARTNERS}
    res["res3"]["combo"] = {p: summ(grp({t: combo[t][p] for t in RES3})) for p in PARTNERS}

    print("\n=== B2 morph cos @ dev30 (x0) ===")
    print(f"RES3 base cos = {res['res3']['base'][0]:+.3f}")
    print(f"{'partner':9}{'gain_RES3':>22}{'combo_SwinL':>22}")
    gR50 = res["res3"]["gain"]["r50"]
    for p in PARTNERS:
        gp = res["res3"]["gain"][p]; cs = res["combo"]["dino_swin_l"][p]
        print(f"{p:9}{gp[0]:+7.3f} [{gp[1]:+.3f},{gp[2]:+.3f}]  {cs[0]:+7.3f} [{cs[1]:+.3f},{cs[2]:+.3f}]")

    pc = gR50[1] > 0
    cs_r50 = res["combo"]["dino_swin_l"]["r50"][0]
    go = []
    for v in MORPHS:
        gv = res["res3"]["gain"][v]; csv = res["combo"]["dino_swin_l"][v][0]
        ok = pc and gv[1] > 0 and gv[0] >= 0.5 * gR50[0] and csv >= cs_r50
        if ok:
            go.append(v)
    res["positive_control_r50"] = pc
    res["GO"] = go
    print(f"\npositive control (R50 partner giúp RES3): {pc}")
    print("VERDICT:", f"B2 GO @ {','.join(go)} — morph tạo partner hữu ích, đáng dựng attack"
          if go else "B2 NO-GO — không morph nào đạt >=0.5 lợi ích R50 trên RES3 mà giữ SwinL; morphing không có đòn bẩy")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[b2] -> {OUT}")


if __name__ == "__main__":
    main()
