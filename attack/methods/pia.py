"""Pilot "đa dạng hình học kiểu PIA" @ dev30 (quy tắc khóa: docs/progress_log.md entry 2026-10-04).

PIA (Liang, Cao, Xiao — "Perspective-Invariant Attack with Enhanced Transferability of Adversarial
Examples", arXiv 2608.15115, IEEE TIFS 2026), port theo mô tả paper (paper không kèm code):
- 4 góc ảnh RA (TL, TR, BR, BL) ánh xạ về 4 điểm NGUỒN trong ảnh vào; điểm nguồn = góc +
  (int(ξ_x·δ_max·w/2), int(ξ_y·δ_max·h/2)), ξ ~ U(−1, 1), được phép ra ngoài ảnh.
- Multi-DOF vertex sampling, mode rút theo π = (π₄, π₃, π₂, π₁) mỗi view:
  mode 4: 4 góc độc lập (8-DOF); mode 3: 3 góc độc lập, góc thiếu (chọn đều) hoàn thiện hình bình
  hành (6-DOF); mode 2: TL, BR độc lập, TR/BL dựng hình chữ nhật trục (4-DOF); mode 1: 1 vector
  dịch chung (2-DOF).
- Homography 8 ẩn giải bằng float64; warp bằng grid_sample bilinear (khả vi), ngoài ảnh nguồn = 0.

Nền chung mọi ô GEO (khác OSFD ĐÚNG ở phép biến đổi + co-transform feature sạch): loss feature OSFD
(backbone, 4 stage, k = 3), MI, nhiễu σ = 6 cộng lên view adv; feature sạch của MỖI view = backbone
(T(x_clean)) cùng tham số hình học, không nhiễu, no_grad (forward thêm không tính B).
GEO-C0: T = RRB gốc (`rrb_views(companion=)`), đối chứng riêng hiệu ứng co-transform.
"""
import contextlib
import copy
import random

import torch
import torch.nn.functional as F
from mmdet.structures.mask import BitmapMasks
from torch.nn.utils.stateless import _reparametrize_module

from attack.methods.box_transforms import get_gt
from attack.methods.core import run_iterative_attack
from attack.methods.repair import make_view_loss
from attack.methods.rrb import gaussian_blur, rrb_views
from attack.preprocess import extract_features, resize_to_model

PI_PAPER = (0.4, 0.3, 0.2, 0.1)  # (π₄, π₃, π₂, π₁) — paper, mục Implementation Details
DELTA_PAPER = 0.6
N_PAPER = 20


def sample_source_corners(h: int, w: int, delta_max: float, pi=PI_PAPER):
    """Trả về (mode, src [4,2] float (x, y)) theo thứ tự TL, TR, BR, BL. RNG: `random` của Python."""
    base = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
    r, acc, mode = random.random(), 0.0, 1
    for m, p in zip((4, 3, 2, 1), pi):
        acc += p
        if r < acc:
            mode = m
            break

    def offset():
        return (float(int(random.uniform(-1, 1) * delta_max * w / 2)),
                float(int(random.uniform(-1, 1) * delta_max * h / 2)))

    def moved(i, o):
        return (base[i][0] + o[0], base[i][1] + o[1])

    if mode == 4:
        src = [moved(i, offset()) for i in range(4)]
    elif mode == 3:
        miss = random.randrange(4)
        src = [None] * 4
        for i in range(4):
            if i != miss:
                src[i] = moved(i, offset())
        prv, nxt, opp = src[(miss - 1) % 4], src[(miss + 1) % 4], src[(miss + 2) % 4]
        src[miss] = (prv[0] + nxt[0] - opp[0], prv[1] + nxt[1] - opp[1])
    elif mode == 2:
        tl, br = moved(0, offset()), moved(2, offset())
        src = [tl, (br[0], tl[1]), br, (tl[0], br[1])]
    else:
        o = offset()
        src = [moved(i, o) for i in range(4)]
    return mode, torch.tensor(src, dtype=torch.float64)


def homography(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """H [3,3] float64 sao cho src ~ H·dst (dst, src: [4,2] (x, y))."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst.tolist(), src.tolist()):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v]); b.append(v)
    h = torch.linalg.solve(torch.tensor(A, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
    return torch.cat([h, h.new_ones(1)]).view(3, 3)


FACTORS = ("T", "R", "S", "K", "P")  # tịnh tiến, xoay, co giãn đẳng hướng, lệch tỉ lệ + trượt, phối cảnh


def _center(h: int, w: int) -> torch.Tensor:
    return torch.tensor([[1, 0, (w - 1) / 2], [0, 1, (h - 1) / 2], [0, 0, 1]], dtype=torch.float64)


def decompose_h(H: torch.Tensor, h: int, w: int) -> dict:
    """Giai đoạn A (progress_log 2026-10-04): H (pixel ra -> nguồn) ở tọa độ tâm ảnh, H' = C⁻¹ H C = H_S·H_A·H_P.
    -> {t [2], theta, s, k1, k2, v [2]} (float64)."""
    C = _center(h, w)
    Hc = torch.linalg.inv(C) @ H.double() @ C
    Hc = Hc / Hc[2, 2]
    A, t, v = Hc[:2, :2], Hc[:2, 2], Hc[2, :2]
    Q, U = torch.linalg.qr(A - torch.outer(t, v))
    D = torch.diag(torch.sign(torch.diagonal(U)))
    Q, U = Q @ D, D @ U  # đường chéo U dương
    assert torch.det(Q) > 0, "homography có phản xạ"
    sc = torch.sqrt(torch.det(U))
    K = U / sc
    return {"t": t, "theta": torch.atan2(Q[1, 0], Q[0, 0]), "s": sc, "k1": K[0, 0], "k2": K[0, 1], "v": v}


def compose_h(f: dict, h: int, w: int) -> torch.Tensor:
    c, sn = torch.cos(f["theta"]), torch.sin(f["theta"])
    R = torch.stack([torch.stack([c, -sn]), torch.stack([sn, c])])
    K = torch.stack([torch.stack([f["k1"], f["k2"]]), torch.stack([torch.zeros_like(f["k1"]), 1 / f["k1"]])])
    M = f["s"] * R @ K
    Hc = torch.eye(3, dtype=torch.float64)
    Hc[:2, :2] = M + torch.outer(f["t"], f["v"])
    Hc[:2, 2] = f["t"]
    Hc[2, :2] = f["v"]
    C = _center(h, w)
    return C @ Hc @ torch.linalg.inv(C)


IDENTITY = {"T": {"t": torch.zeros(2, dtype=torch.float64)}, "R": {"theta": torch.tensor(0.0, dtype=torch.float64)},
            "S": {"s": torch.tensor(1.0, dtype=torch.float64)},
            "K": {"k1": torch.tensor(1.0, dtype=torch.float64), "k2": torch.tensor(0.0, dtype=torch.float64)},
            "P": {"v": torch.zeros(2, dtype=torch.float64)}}


def drop_factor(H: torch.Tensor, factor: str, h: int, w: int):
    """-> (H với thành phần `factor` đặt về đồng nhất, các thành phần đã tách của H gốc)."""
    f = decompose_h(H, h, w)
    return compose_h(dict(f, **IDENTITY[factor]), h, w), f


def warp_grid(H: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """Lưới grid_sample [1,h,w,2] (align_corners=True): pixel ra (x, y) lấy mẫu ở H·(x, y, 1)."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64),
                            indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) @ H.T
    u, v = pts[..., 0] / pts[..., 2], pts[..., 1] / pts[..., 2]
    grid = torch.stack([u / (w - 1) * 2 - 1, v / (h - 1) * 2 - 1], dim=-1)
    return grid.float().unsqueeze(0).to(device)


def warp(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """img [C,H,W] -> [C,H,W], bilinear, ngoài ảnh nguồn = 0."""
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True).squeeze(0)


def warp_boxes(boxes: torch.Tensor, H: torch.Tensor, h: int, w: int):
    """GT box (khung vào) -> khung ra qua H⁻¹ (H: ra -> nguồn): 4 góc -> AABB -> clip.
    Trả về (box [M,4], keep [N] bool): bỏ box rộng/cao < 1 px sau clip hoặc mẫu số ≤ 0 ở góc."""
    if len(boxes) == 0:
        return boxes, torch.zeros(0, dtype=torch.bool, device=boxes.device)
    Hi = torch.linalg.inv(H).to(boxes.device)
    b = boxes.double()
    xs = torch.stack([b[:, 0], b[:, 2], b[:, 2], b[:, 0]], 1)
    ys = torch.stack([b[:, 1], b[:, 1], b[:, 3], b[:, 3]], 1)
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Hi.T  # [N,4,3]
    den = pts[..., 2]
    ok = (den > 0).all(1)
    u, v = pts[..., 0] / den.clamp_min(1e-12), pts[..., 1] / den.clamp_min(1e-12)
    out = torch.stack([u.min(1).values.clamp(0, w), v.min(1).values.clamp(0, h),
                       u.max(1).values.clamp(0, w), v.max(1).values.clamp(0, h)], 1).float()
    keep = ok & (out[:, 2] - out[:, 0] >= 1) & (out[:, 3] - out[:, 1] >= 1)
    return out[keep], keep


def gt_with_keep(data_sample, boxes, masks, keep):
    """data_sample mới: chỉ giữ instance `keep`, box/mask đã warp (mask threshold 0.5 như with_gt)."""
    ds = copy.deepcopy(data_sample)
    ds.gt_instances = ds.gt_instances[keep.cpu()]
    ds.gt_instances.bboxes = boxes
    if masks is not None:
        mh, mw = masks.shape[-2:]
        ds.gt_instances.masks = BitmapMasks((masks[keep] > 0.5).to(torch.uint8).cpu().numpy(), mh, mw)
    return ds


def pia_views(adv_pixels: torch.Tensor, n: int, delta_max: float = DELTA_PAPER, pi=PI_PAPER,
              sigma: float = 6.0, companion=None, log=None, data_sample=None, gt_log=None, drop=None,
              factor_log=None):
    """n view PIA của adv (khung model [C,H,W]); companion (ảnh sạch) nhận CÙNG phép warp, không nhiễu.
    Trả về [(view_nhiễu, companion_warp, data_sample_view)]; data_sample_view = None nếu không truyền
    data_sample, ngược lại GT co-transform (box qua H⁻¹, mask cùng lưới). Rút tham số tuần tự từng view
    (cụm 5 × 4 lần = 20 lần 1); co-transform GT không tiêu thụ RNG.
    drop ∈ FACTORS (Giai đoạn A): bỏ 1 thành phần của H sau khi rút (không đổi RNG); factor_log nhận
    (thành phần của H gốc, |u| trung bình px của H dùng thật)."""
    h, w = adv_pixels.shape[-2:]
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    gt = get_gt(data_sample, adv_pixels.device) if data_sample is not None else None
    out = []
    for _ in range(n):
        mode, src = sample_source_corners(h, w, delta_max, pi)
        if log is not None:
            log.append(mode)
        H = homography(dst, src)
        if drop is not None:
            H, f = drop_factor(H, drop, h, w)
        grid = warp_grid(H, h, w, adv_pixels.device)
        if factor_log is not None:
            g = grid[0].double()
            disp = float(torch.sqrt(((g[..., 0] + 1) / 2 * (w - 1) - torch.arange(w, device=g.device)) ** 2
                                    + ((g[..., 1] + 1) / 2 * (h - 1) - torch.arange(h, device=g.device)[:, None]) ** 2).mean())
            ff = f if drop is not None else decompose_h(homography(dst, src), h, w)
            factor_log.append({"t": float(ff["t"].norm()), "theta_deg": float(ff["theta"].abs() * 180 / torch.pi),
                               "abs_ln_s": float(ff["s"].log().abs()), "abs_ln_k1": float(ff["k1"].log().abs()),
                               "abs_k2": float(ff["k2"].abs()), "v_px": float(ff["v"].norm() * max(h, w)),
                               "disp_px": disp})
        view = gaussian_blur(warp(adv_pixels, grid), sigma=sigma) if sigma > 0 else warp(adv_pixels, grid)
        dsv = None
        if gt is not None:
            boxes, masks = gt
            nb, keep = warp_boxes(boxes, H, h, w)
            m = warp(masks, grid) if masks is not None and len(masks) else masks
            dsv = gt_with_keep(data_sample, nb, m, keep)
            if gt_log is not None:
                gt_log.append((int(keep.sum()), len(keep)))
        out.append((view, warp(companion, grid) if companion is not None else None, dsv))
    return out


def _view_loss(feats_clean, k: float, stages=None):
    """stages: chỉ số stage backbone dùng trong loss (None = cả 4, như OSFD; OCT-C4 / OCT-C34, progress_log 2026-10-10)."""
    def loss_fn(model, pixel_tensor, data_sample):
        feats_adv = extract_features(model, pixel_tensor, data_sample)
        return sum(F.mse_loss(k * fc, fa) for j, (fc, fa) in enumerate(zip(feats_clean, feats_adv))
                   if stages is None or j in stages)
    return loss_fn


def _obj_mask(f, boxes, level: int = 2, stride: int = 16):
    """Mặt nạ token object [h, w] ở stage `level`: ô stride `stride` có tâm nằm trong ≥ 1 GT box (khung view)."""
    a = f[level][0]
    h, w = a.shape[-2:]
    yy = (torch.arange(h, device=a.device).float() + 0.5) * stride
    xx = (torch.arange(w, device=a.device).float() + 0.5) * stride
    obj = torch.zeros(h, w, dtype=torch.bool, device=a.device)
    for x1, y1, x2, y2 in boxes:
        obj |= ((yy >= y1) & (yy < y2))[:, None] & ((xx >= x1) & (xx < x2))[None]
    return obj, (yy, xx)


def _bg_loss(a, c, obj, valid):
    """Loss "object → nền" (E14c, progress_log 2026-10-10): −mean_obj ‖a_t − μ_nền‖² / mean_obj ‖c_t − μ_nền‖², μ_nền =
    trung bình token SẠCH ngoài mọi box (chỉ token có tâm trong ảnh view, bỏ pad). Ascent -> kéo token object về
    μ_nền. 0 nếu thiếu object / nền."""
    bg = valid & ~obj
    ob = valid & obj
    if not bg.any() or not ob.any():
        return a.sum() * 0.0
    mu = c[:, bg].mean(1, keepdim=True)
    return -((a[:, ob] - mu) ** 2).sum(0).mean() / ((c[:, ob] - mu) ** 2).sum(0).mean().clamp_min(1e-12)


def _view_loss_bg(feats_clean, k: float, boxes, bg_weight: float, feat_weight: float, level: int = 2, stride: int = 16):
    """feat_weight·L_OSFD + bg_weight·L_bg (OCT+BG / BG-OCT, progress_log 2026-10-10)."""
    def loss_fn(model, pixel_tensor, data_sample):
        feats_adv = extract_features(model, pixel_tensor, data_sample)
        ph, pw = pixel_tensor.shape[-2:]
        obj, (yy, xx) = _obj_mask(feats_adv, boxes, level, stride)
        valid = (yy < ph)[:, None] & (xx < pw)[None]
        lb = _bg_loss(feats_adv[level][0].float(), feats_clean[level][0].float(), obj, valid)
        if feat_weight == 0.0:
            return bg_weight * lb
        return feat_weight * sum(F.mse_loss(k * fc, fa) for fc, fa in zip(feats_clean, feats_adv)) + bg_weight * lb
    return loss_fn


def rapa_sampler(model, p: float, device):
    """RaPA (Su et al., CVPR 2026; progress_log 2026-10-04 MORGD-RaPA): DropConnect lên weight + bias của mọi
    BatchNorm2d trong backbone, M ~ Bernoulli(1 − p), không nhân bù. Trả về hàm () -> context manager thay TẠM
    tham số backbone bằng tensor masked (dùng nhất quán cả forward lẫn backward — khác code chính thức, vốn đổi
    `.data` rồi khôi phục trước backward). RNG riêng, seed = torch.initial_seed() của ảnh."""
    params = [(f"{n}.{a}", getattr(m, a)) for n, m in model.backbone.named_modules()
              if isinstance(m, torch.nn.BatchNorm2d) for a in ("weight", "bias")]
    gen = torch.Generator(device=device)
    gen.manual_seed(torch.initial_seed())
    # Bộ đếm "masked" giữ trên GPU (progress_log 2026-10-18): int() mỗi tensor = 106 lần đồng bộ CPU–GPU / view (~14 ms).
    stats = {"masked": torch.zeros((), dtype=torch.long, device=device), "total": 0}

    def sample():
        masked = {}
        for name, t in params:
            m = torch.bernoulli(torch.full_like(t, 1.0 - p), generator=gen)
            stats["masked"] += (m == 0).sum(); stats["total"] += m.numel()
            masked[name] = t.detach() * m
        return lambda: _reparametrize_module(model.backbone, masked)
    return sample, stats


def rapa_cls_sampler(model, p: float, device, scope: str = "all"):
    """RaPA trên ĐƯỜNG PHÂN LOẠI (progress_log 2026-10-08, ý user): DropConnect weight + bias của mọi BatchNorm2d backbone
    + Linear của RoI bbox head trên nhánh phân loại (shared_fcs.*, fc_cls); M ~ Bernoulli(1 − p), không nhân bù; tham số
    masked nhất quán forward + backward (`_reparametrize_module` trên model gốc). RNG riêng, seed = torch.initial_seed()."""
    assert scope in ("all", "backbone", "head")  # định vị (progress_log 2026-10-08): chỉ backbone BN | chỉ RoI head
    bh = model.roi_head.bbox_head
    names = [] if scope == "head" else [f"backbone.{n}.{a}" for n, m in model.backbone.named_modules()
                                        if isinstance(m, torch.nn.BatchNorm2d) for a in ("weight", "bias")]
    names += [] if scope == "backbone" else [f"roi_head.bbox_head.{n}.{a}" for n, m in bh.named_modules()
              if isinstance(m, torch.nn.Linear) and (n.startswith("shared_fcs") or n == "fc_cls") for a in ("weight", "bias")]
    params = [(n, model.get_parameter(n)) for n in names]
    gen = torch.Generator(device=device)
    gen.manual_seed(torch.initial_seed())
    stats = {"masked": torch.zeros((), dtype=torch.long, device=device), "total": 0}  # như rapa_sampler: đếm trên GPU

    def sample():
        masked = {}
        for name, t in params:
            m = torch.bernoulli(torch.full_like(t, 1.0 - p), generator=gen)
            stats["masked"] += (m == 0).sum(); stats["total"] += m.numel()
            masked[name] = t.detach() * m
        return lambda: _reparametrize_module(model, masked)
    return sample, stats


def calib_cls_weight(model, orig_pixels, clean_pixels, data_sample, k: float, cls_mode: str, clip=(1.0, 16.0)):
    """λ theo ảnh (progress_log 2026-10-18): mean|∇ₓL_feat| / mean|∇ₓL_cls| tại δ = 0, ảnh không biến đổi — đúng quy tắc
    chọn λ = 4 cũ (median trên 5 ảnh) nhưng đo riêng từng ảnh. 1 forward + 2 lần autograd.grad. RNG torch (CPU + CUDA)
    khôi phục sau đo -> chuỗi view / RoI sampling của attack trùng ô λ cố định. Trả (λ đã clip, tỉ lệ thô)."""
    from attack.methods.repair import cls_term
    from attack.preprocess import to_batch
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
    with torch.no_grad():
        fc = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))
    n = torch.zeros_like(orig_pixels, requires_grad=True)
    adv = resize_to_model(orig_pixels + n, tuple(data_sample.img_shape))
    cap = {}
    h = model.backbone.register_forward_hook(lambda m, i, o: cap.update(f=o))
    try:
        bi, bds = to_batch(model, [adv], [data_sample])
        c = cls_term(model, bi, bds, cls_mode)
    finally:
        h.remove()
    feat = sum(F.mse_loss(k * a, b) for a, b in zip(fc, cap["f"]))
    gf = torch.autograd.grad(feat, n, retain_graph=True)[0]
    gc = torch.autograd.grad(c, n)[0]
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    raw = float(gf.abs().mean() / (gc.abs().mean() + 1e-12))
    return min(max(raw, clip[0]), clip[1]), raw


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [N,4], b [M,4] xyxy -> IoU [N,M]."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda t: (t[:, 2:] - t[:, :2]).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def fia_channel_weights(model, clean_pixels, data_sample, n: int, elem: dict, sigma: float, mix: float = 0.5):
    """C2-OCT-1V-FIA (progress_log 2026-10-18; kiểu FIA / NAA, dạng mềm): độ quan trọng kênh của từng stage backbone =
    mean_{n view OCT của ảnh SẠCH (nhiễu σ, GT co-transform)} mean_hw |∂L_cls / ∂F_l| (L_cls = rpn_cls + cls, như C2).
    Theo KÊNH (bất biến với warp — gradient không gian của các view lệch nhau). w = (1 − mix) + mix · g / mean_c g (TB 1).
    RNG Python + torch (CPU, CUDA) khôi phục -> chuỗi view của attack trùng ô gốc. n forward + n backward (tới feature)."""
    from attack.methods.elem import elem_views
    from attack.methods.repair import cls_term
    from attack.preprocess import to_batch
    rng = (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state())
    acc = None
    for v, _, dsv in elem_views(clean_pixels, n, elem, sigma, data_sample=data_sample):
        cap = {}
        h = model.backbone.register_forward_hook(lambda m, i, o: cap.update(f=o))
        try:
            # input cần grad: stem + layer1 của R50 đóng băng (frozen_stages = 1) -> C2 không có grad nếu input không có
            bi, bds = to_batch(model, [v.detach().requires_grad_(True)], [dsv])
            c = cls_term(model, bi, bds, "c2")
        finally:
            h.remove()
        gs = torch.autograd.grad(c, cap["f"], allow_unused=True)
        g = [(x.float().abs().mean(dim=(0, 2, 3)) if x is not None else torch.zeros(f.shape[1], device=f.device))
             for x, f in zip(gs, cap["f"])]
        acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    random.setstate(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state(rng[2])
    return [((1 - mix) + mix * a / (a.mean() + 1e-12)).detach() for a in acc]


def geo_osfd_attack(model, orig_pixels, data_sample, steps: int, epsilon: float = 5.0,
                    alpha: float = 1.0, k: float = 3.0, momentum: float = 1.0, geo: str = "pia",
                    n_views: int = 2, backward_per_step: int = 1, delta_max: float = DELTA_PAPER,
                    pi=PI_PAPER, sigma: float = 6.0, info=None, on_step=None,
                    co_transform: bool = True, cls_weight: float = 0.0, rapa_p=None, pia_drop=None,
                    cam_theta: float = 15.0, cam_kappa: float = 0.0, elem=None,
                    rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8, rrb_s_max: float = 1.1,
                    skip_mask: bool = False, amp: bool = False, feature: bool = True,
                    cls_mode: str = "c2", rapa_cls_p=None, rapa_scope: str = "all", cache_views: int = 0,
                    bg_weight: float = 0.0, bg_only: bool = False, feat_stages=None, strat: bool = False,
                    season_sigma=None, cls_auto=None, ilpd_gamma=None, fia_views: int = 0, roi_pos: bool = False,
                    roi_cap=None, fab_k: int = 0, fab_weight: float = 0.0, obj_weight: float = 0.0, dual=None,
                    margin=None, margin_scale: float = 1.0):
    """MI + (RRB | PIA) + loss feature OSFD với feature sạch co-transform theo từng view.
    geo="rrb": GEO-C0 (luôn 2 view). geo="pia": n_views view/backward, backward_per_step backward/step.
    co_transform=False (PIA-noCT): feature sạch của ảnh sạch CHƯA biến đổi, như OSFD gốc.
    geo="morgd" | "smooth": trường dịch chuyển mượt neo theo GT object | neo ngẫu nhiên (pilot MORGD).
    geo="cam": Giai đoạn B — xoay camera ảo (cam_theta độ, yaw/pitch) + méo ống kính (cam_kappa), camgeo.py.
    pia_drop ∈ FACTORS (geo="pia"): Giai đoạn A — bỏ 1 thành phần homography (T, R, S, K, P) mỗi view.
    geo="pia_morgd2" | "pia_smooth2" | "morgd2": MORGD-v2 (phân vùng theo box) ∘ PIA | đối chứng tâm ngẫu nhiên ∘ PIA |
    MORGD-v2 không PIA (morgd2.py); RNG (r, a) = Random(seed + 1), tâm ngẫu nhiên = Random(seed + 2).
    geo="grgd" | "grgd_random": T_relative ∘ PIA, neo tại tâm GT | tâm ngẫu nhiên (pilot GRGD-v0, attack/methods/grgd.py);
    ξ từ random.Random(torch.initial_seed() + 1), tâm ngẫu nhiên từ random.Random(torch.initial_seed() + 2).
    geo="elem": phép biến đổi cơ bản quanh tâm (attack/methods/elem.py), elem = {loại: liều}.
    cls_weight > 0 (C2-PIA / C2-MORGD / C2-SMOOTH, không cho geo="rrb"): + cls_weight·(loss_rpn_cls + loss_cls) với GT co-transform,
    1 forward/view (repair.make_view_loss, không sửa = loss của C2).
    skip_mask (tối ưu tốc độ, progress_log 2026-10-08): loss C2 chỉ dùng loss_rpn_cls + loss_cls -> tắt mask head
    (forward + backward) và bỏ co-transform mask GT; RoI sampling / RNG không đổi -> cùng gradient.
    rapa_cls_p (chỉ với feature=False): RaPA trên backbone BN + RoI head phân loại, mask mới mỗi view (rapa_cls_sampler).
    cache_views=N (geo="elem", loss feature thuần; progress_log 2026-10-08): tập CỐ ĐỊNH N phép biến đổi (N = 64: đủ 8×8 pha
    dịch, mỗi pha 1 mức co giãn phân tầng trong elem["scale"]); feature sạch của N view tính 1 lần / ảnh (fp16), mỗi view
    rút chỉ số đều -> không forward sạch trong vòng lặp.
    cls_mode="rpn": số hạng cls chỉ là loss_rpn_cls (bỏ sinh proposal + RoI head), repair.cls_term.
    feature=False (CLS-OCT): bỏ loss feature OSFD, chỉ cls_weight·cls (không forward feature sạch co-transform).
    amp: autocast fp16 cho forward / backward surrogate (δ, momentum, bước sign giữ fp32) — đổi số học, phải kiểm bằng eval.
    bg_weight > 0 (geo="elem", loss feature, progress_log 2026-10-10): + bg_weight·L_bg (kéo token object C4 / stride 16 về
    μ_nền sạch của view, box co-transform); bg_only: bỏ L_OSFD (BG-OCT).
    feat_stages (loss feature thuần, có / không cache): chỉ các stage backbone này trong L_OSFD (OCT-C4 = (2,), OCT-C34 = (1, 2)).
    strat (geo="elem", elem = {scale, shift: ph8}; progress_log 2026-10-17): co giãn ∘ pha theo lịch phủ có hệ thống
    (elem.strat_schedule, steps × n_views view) thay cho rút độc lập.
    cls_auto = c (progress_log 2026-10-18): thay cls_weight bằng c · λ_ảnh (calib_cls_weight, đo tại δ = 0; +2 backward / ảnh).
    ilpd_gamma = γ (progress_log 2026-10-18): ILPD trên đường head ở stage 3 (stride 16), repair.make_view_loss; cần cls_weight > 0."""
    assert ilpd_gamma is None or (cls_weight > 0 and feature and co_transform)
    # fia_views = n (progress_log 2026-10-18): L_OSFD trọng số kênh 50% đều + 50% |∂L_cls/∂F| (fia_channel_weights, +n backward / ảnh).
    # roi_pos / roi_cap: loss_cls RoI chỉ trên RoI dương | + bỏ RoI dương có p(lớp đúng) < roi_cap (repair.roi_cls_filter).
    assert not fia_views or (geo == "elem" and cls_weight > 0 and feature and co_transform)
    assert roi_cap is None or roi_pos
    assert obj_weight == 0.0 or (cls_weight > 0 and feature and co_transform)  # C2-OCT-1V-OBJ: embedding RoI mức object
    # dual (progress_log 2026-10-20): momentum riêng cho L_feat và L_cls (core.run_iterative_attack). "exact": cả 2 số hạng
    # mỗi bước; "alt": bước chẵn chỉ L_feat (chỉ backbone), bước lẻ chỉ L_cls (bỏ forward sạch co-transform).
    assert dual is None or (geo == "elem" and cls_weight > 0 and feature and co_transform and rapa_cls_p is None
                            and not (fia_views or roi_pos or fab_k or obj_weight or ilpd_gamma is not None or cls_auto))
    assert not roi_pos or (cls_weight > 0 and cls_mode == "c2")
    assert geo in ("rrb", "pia", "morgd", "smooth", "grgd", "grgd_random",
                   "pia_morgd2", "pia_smooth2", "morgd2", "cam", "noise", "flip", "pool", "elem")  # morgd/smooth: morgd.py; grgd*: grgd.py; *2: morgd2.py
    assert cls_weight == 0.0 or geo != "rrb"
    assert co_transform or geo in ("pia", "elem")  # elem: C2-OCT-1V-noCT (progress_log 2026-10-18) — box GT vẫn co-transform
    assert rapa_p is None or (co_transform and cls_weight == 0.0)  # RaPA: chỉ khung loss feature + co-transform
    assert bg_weight == 0.0 or (geo == "elem" and cls_weight == 0.0 and co_transform and not cache_views and rapa_p is None)
    assert not bg_only or bg_weight > 0
    assert feat_stages is None or (cls_weight == 0.0 and bg_weight == 0.0 and rapa_p is None and feature)
    clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
    mask_head = None
    if skip_mask:
        assert cls_weight > 0
        data_sample = copy.deepcopy(data_sample)
        if "masks" in data_sample.gt_instances:
            del data_sample.gt_instances.masks
        mask_head, model.roi_head.mask_head = model.roi_head.mask_head, None
    log = {"forward_no_grad": 0, "modes": [], "gt_kept": [], "disp": [], "grgd": [], "morgd2": [], "pia_factors": []}
    if cls_auto is not None:
        assert cls_weight > 0 and feature and co_transform and cls_mode == "c2"
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lam, raw = calib_cls_weight(model, orig_pixels, clean_pixels, data_sample, k, cls_mode)
        cls_weight = cls_auto * lam
        log.update(cls_lambda=cls_weight, cls_ratio_raw=raw)
    if fab_k:
        # Box mồi (progress_log 2026-10-18): fab_k box cố định / ảnh, khung model; cỡ = cỡ 1 GT thật ngẫu nhiên (không GT ->
        # 0.2 × cạnh ảnh); vị trí đều trong ảnh, tránh IoU ≥ 0.1 với GT thật + mồi trước (thử 50 lần); nhãn đều 80 lớp.
        # RNG riêng seed + 5 -> chuỗi view trùng ô gốc. Gộp vào gt_instances (nhãn + offset) để co-transform cùng GT.
        assert skip_mask and cls_weight > 0 and geo == "elem" and fab_weight > 0
        from mmengine.structures import InstanceData
        from attack.methods.repair import FAB_LABEL_OFFSET
        from attack.methods.box_transforms import get_gt
        rng_f = random.Random(torch.initial_seed() + 5)
        real, _ = get_gt(data_sample, clean_pixels.device)
        h, w = data_sample.img_shape
        sizes = [(float(b[2] - b[0]), float(b[3] - b[1])) for b in real] or [(0.2 * w, 0.2 * h)]
        taken, fab = [b.tolist() for b in real], []
        for _ in range(fab_k):
            bw, bh = rng_f.choice(sizes)
            bw, bh = min(bw, w - 1), min(bh, h - 1)
            for _try in range(50):
                x1, y1 = rng_f.uniform(0, w - bw), rng_f.uniform(0, h - bh)
                cand = [x1, y1, x1 + bw, y1 + bh]
                if not taken or float(_iou_xyxy(torch.tensor([cand]), torch.tensor(taken)).max()) < 0.1:
                    break
            taken.append(cand); fab.append(cand)
        labels = torch.tensor([rng_f.randrange(80) + FAB_LABEL_OFFSET for _ in fab], device=clean_pixels.device)
        gi = data_sample.gt_instances
        data_sample.gt_instances = InstanceData(
            bboxes=torch.cat([real, torch.tensor(fab, device=real.device)]),
            labels=torch.cat([gi.labels.to(clean_pixels.device), labels]))
        sf = data_sample.scale_factor  # (w, h): khung model -> ảnh gốc chia cho sf
        log["fab_boxes_orig"] = [[b[0] / sf[0], b[1] / sf[1], b[2] / sf[0], b[3] / sf[1]] for b in fab]
        log["fab_labels"] = [int(v) - FAB_LABEL_OFFSET for v in labels]
    feat_chw = None
    if fia_views:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            feat_chw = fia_channel_weights(model, clean_pixels, data_sample, fia_views, elem, sigma)
        log["fia_w_max"] = [float(w.max()) for w in feat_chw]
    roi_stats = {"kept": 0, "pos": 0, "total": 0}
    roi_ctx = None
    if roi_pos:
        from attack.methods.repair import roi_cls_filter
        roi_ctx = lambda: roi_cls_filter(model, roi_cap, roi_stats)
    if margin is not None:  # C2-OCT-1V-MRG / -MRGF (progress_log 2026-10-20): loss_cls RoI -> margin CW trên RoI dương
        assert not roi_pos and cls_weight > 0 and cls_mode == "c2" and dual is None
        from attack.methods.repair import roi_margin
        roi_ctx = lambda: roi_margin(model, margin, margin_scale)
    if geo.startswith("grgd") or geo.endswith("2"):
        rng_xi = random.Random(torch.initial_seed() + 1)
        rng_c = random.Random(torch.initial_seed() + 2)
    rrb_kw = dict(theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho, s_max=rrb_s_max, sigma=sigma)
    feats_fixed = None
    if not co_transform:
        with torch.no_grad():
            feats_fixed = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))

    rapa = rapa_sampler(model, rapa_p, clean_pixels.device) if rapa_p is not None else None
    # feature=True (C2-OCT-1V-RaPA, progress_log 2026-10-18): chỉ scope="backbone", CÙNG mask cho feature sạch và adv của view.
    assert rapa_cls_p is None or not feature or (rapa_scope == "backbone" and cls_weight > 0 and co_transform)
    rapa_cls = rapa_cls_sampler(model, rapa_cls_p, clean_pixels.device, rapa_scope) if rapa_cls_p is not None else None

    sched = None
    if strat:
        assert geo == "elem" and set(elem) == {"scale", "shift"} and elem["shift"] == "ph8"
        assert backward_per_step == 1 and not cache_views
        from attack.methods.elem import strat_schedule
        sched = strat_schedule(steps * n_views, elem["scale"])

    cache = None
    if cache_views:
        assert geo == "elem" and cls_weight == 0.0 and co_transform and set(elem) == {"scale", "shift"} and elem["shift"] == "ph8"
        assert cache_views == 64
        import math
        from attack.methods.elem import linear_homography, shift_h
        rng = random.Random(torch.initial_seed() + 3)
        a = elem["scale"]
        us = [-a + (j + rng.random()) * 2 * a / cache_views for j in range(cache_views)]
        rng.shuffle(us)
        h, w = clean_pixels.shape[-2:]
        cache = []
        with torch.no_grad():
            for j, (tx, ty) in enumerate([(x, y) for x in range(8) for y in range(8)]):
                sc = math.exp(us[j])
                H = linear_homography(torch.tensor([[sc, 0.0], [0.0, sc]], dtype=torch.float64), h, w) @ shift_h(tx, ty)
                grid = warp_grid(H, h, w, clean_pixels.device)
                fc = tuple(f.detach().half() for f in extract_features(model, warp(clean_pixels, grid), data_sample))
                cache.append((grid, fc))
        log["forward_no_grad"] += cache_views

    def views_fn(img, ds, step_idx, total):
        if cache is not None:
            out = []
            for _ in range(n_views):
                grid, fc = cache[random.randrange(len(cache))]
                v = warp(img, grid)
                out.append((gaussian_blur(v, sigma=sigma) if sigma > 0 else v, ds,
                            _view_loss(tuple(f.float() for f in fc), k, feat_stages)))
            return out
        comp = clean_pixels if co_transform else None
        if geo == "rrb":
            triples = [(v, c, ds) for v, _, c in rrb_views(img, ds, with_gt=True, companion=comp, **rrb_kw)]
        elif geo == "pia":
            triples = pia_views(img, n_views, delta_max, pi, sigma, companion=comp, log=log["modes"],
                                data_sample=ds if cls_weight > 0 else None, gt_log=log["gt_kept"], drop=pia_drop,
                                factor_log=log["pia_factors"] if pia_drop is not None else None)
        elif geo in ("noise", "flip", "pool"):
            assert cls_weight == 0.0
            from attack.methods.geopool import flip_views, pool_views  # tránh import vòng
            if geo == "pool":
                triples = pool_views(img, n_views, sigma, companion=comp, log=log["disp"])
            else:
                triples = flip_views(img, n_views, 0.5 if geo == "flip" else 0.0, sigma, companion=comp)
        elif geo == "elem":
            from attack.methods.elem import elem_views  # tránh import vòng
            triples = elem_views(img, n_views, elem, sigma, companion=comp, log=log["disp"],
                                 data_sample=ds if cls_weight > 0 or bg_weight > 0 else None, gt_log=log["gt_kept"],
                                 fixed=sched[step_idx * n_views:(step_idx + 1) * n_views] if sched is not None else None)
        elif geo == "cam":
            from attack.methods.camgeo import cam_views  # tránh import vòng
            triples = cam_views(img, n_views, cam_theta, cam_kappa, sigma, companion=comp,
                                data_sample=ds if cls_weight > 0 else None, log=log["disp"], gt_log=log["gt_kept"])
        elif geo.endswith("2"):
            from attack.methods.morgd2 import morgd2_views  # tránh import vòng
            triples = morgd2_views(img, n_views, ds, rng_xi, rng_c, anchors="random" if geo == "pia_smooth2" else "object",
                                   use_pia=geo != "morgd2", delta_max=delta_max, pi=pi, sigma=sigma, companion=comp,
                                   with_gt=cls_weight > 0, log=log["modes"], gt_log=log["gt_kept"], stats=log["morgd2"])
        elif geo.startswith("grgd"):
            from attack.methods.grgd import grgd_views  # tránh import vòng
            triples = grgd_views(img, n_views, ds, rng_xi, rng_c, anchors="object" if geo == "grgd" else "random",
                                 delta_max=delta_max, pi=pi, sigma=sigma, companion=comp, with_gt=cls_weight > 0,
                                 log=log["modes"], gt_log=log["gt_kept"], stats=log["grgd"])
        else:
            from attack.methods.morgd import morgd_views  # tránh import vòng (morgd dùng pia.warp)
            triples = morgd_views(img, n_views, ds, anchors="object" if geo == "morgd" else "random",
                                  sigma=sigma, companion=comp, with_gt=cls_weight > 0, log=log["disp"],
                                  gt_log=log["gt_kept"])
        out = []
        for v, c, dsv in triples:
            ctx = rapa[0]() if rapa is not None else contextlib.nullcontext  # CÙNG mask cho sạch và adv của view
            if rapa_cls is not None and feature:
                ctx = rapa_cls[0]()
            if not feature:
                assert cls_weight > 0
                lf = make_view_loss(None, k, cls_mode, cls_weight, feat_weight=0.0)
                out.append((v, dsv, _with_ctx(lf, rapa_cls[0]()) if rapa_cls is not None else lf))
                continue
            if dual == "alt" and step_idx % 2 == 1:
                lc = make_view_loss(None, k, cls_mode, cls_weight, feat_weight=0.0)
                out.append((v, dsv, lambda m, p, d, lc=lc: (None, lc(m, p, d))))
                continue
            if co_transform:
                with torch.no_grad(), ctx():
                    fc = tuple(f.detach() for f in extract_features(model, c, ds))
                log["forward_no_grad"] += 1
            else:
                fc = feats_fixed
            if bg_weight > 0:
                gtb = dsv.gt_instances.bboxes
                gtb = gtb.tensor if hasattr(gtb, "tensor") else gtb
                out.append((v, ds, _view_loss_bg(fc, k, gtb.float(), bg_weight, 0.0 if bg_only else 1.0)))
            elif dual == "alt":
                lfe = _view_loss(fc, k)
                out.append((v, dsv, lambda m, p, d, lfe=lfe: (lfe(m, p, d), None)))
            elif cls_weight > 0:
                lf = make_view_loss(fc, k, cls_mode, cls_weight, ilpd_gamma=ilpd_gamma, feat_chw=feat_chw, roi_ctx=roi_ctx,
                                    fab_weight=fab_weight, obj_weight=obj_weight, split=dual == "exact")
                out.append((v, dsv, _with_ctx(lf, ctx) if rapa_cls is not None else lf))
            elif rapa is not None:
                out.append((v, ds, _with_ctx(_view_loss(fc, k), ctx)))
            else:
                out.append((v, ds, _view_loss(fc, k, feat_stages)))
        return out

    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            noise = run_iterative_attack(model, orig_pixels, data_sample, loss_fn=None, steps=steps,
                                         epsilon=epsilon, alpha=alpha, momentum=momentum, views_fn=views_fn,
                                         on_step=on_step, backward_per_step=backward_per_step,
                                         season_sigma=season_sigma, dual=dual)
    finally:
        if mask_head is not None:
            model.roi_head.mask_head = mask_head
    if rapa is not None:
        log["rapa_masked_frac"] = int(rapa[1]["masked"]) / max(1, rapa[1]["total"])
    if rapa_cls is not None:
        log["rapa_masked_frac"] = int(rapa_cls[1]["masked"]) / max(1, rapa_cls[1]["total"])
    if roi_pos:
        log["roi_kept_frac_of_pos"] = roi_stats["kept"] / max(1, roi_stats["pos"])
        log["roi_pos_frac"] = roi_stats["pos"] / max(1, roi_stats["total"])
    if info is not None:
        info.update(log)
    return noise


def _with_ctx(loss_fn, ctx):
    def wrapped(model, pixel_tensor, data_sample):
        with ctx():
            return loss_fn(model, pixel_tensor, data_sample)
    return wrapped
