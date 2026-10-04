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
              sigma: float = 6.0, companion=None, log=None, data_sample=None, gt_log=None):
    """n view PIA của adv (khung model [C,H,W]); companion (ảnh sạch) nhận CÙNG phép warp, không nhiễu.
    Trả về [(view_nhiễu, companion_warp, data_sample_view)]; data_sample_view = None nếu không truyền
    data_sample, ngược lại GT co-transform (box qua H⁻¹, mask cùng lưới). Rút tham số tuần tự từng view
    (cụm 5 × 4 lần = 20 lần 1); co-transform GT không tiêu thụ RNG."""
    h, w = adv_pixels.shape[-2:]
    dst = torch.tensor([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=torch.float64)
    gt = get_gt(data_sample, adv_pixels.device) if data_sample is not None else None
    out = []
    for _ in range(n):
        mode, src = sample_source_corners(h, w, delta_max, pi)
        if log is not None:
            log.append(mode)
        H = homography(dst, src)
        grid = warp_grid(H, h, w, adv_pixels.device)
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


def _view_loss(feats_clean, k: float):
    def loss_fn(model, pixel_tensor, data_sample):
        feats_adv = extract_features(model, pixel_tensor, data_sample)
        return sum(F.mse_loss(k * fc, fa) for fc, fa in zip(feats_clean, feats_adv))
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
    stats = {"masked": 0, "total": 0}

    def sample():
        masked = {}
        for name, t in params:
            m = torch.bernoulli(torch.full_like(t, 1.0 - p), generator=gen)
            stats["masked"] += int((m == 0).sum()); stats["total"] += m.numel()
            masked[name] = t.detach() * m
        return lambda: _reparametrize_module(model.backbone, masked)
    return sample, stats


def geo_osfd_attack(model, orig_pixels, data_sample, steps: int, epsilon: float = 5.0,
                    alpha: float = 1.0, k: float = 3.0, momentum: float = 1.0, geo: str = "pia",
                    n_views: int = 2, backward_per_step: int = 1, delta_max: float = DELTA_PAPER,
                    pi=PI_PAPER, sigma: float = 6.0, info=None, on_step=None,
                    co_transform: bool = True, cls_weight: float = 0.0, rapa_p=None,
                    rrb_theta: float = 7.0, rrb_l_s: int = 10, rrb_rho: float = 0.8, rrb_s_max: float = 1.1):
    """MI + (RRB | PIA) + loss feature OSFD với feature sạch co-transform theo từng view.
    geo="rrb": GEO-C0 (luôn 2 view). geo="pia": n_views view/backward, backward_per_step backward/step.
    co_transform=False (PIA-noCT): feature sạch của ảnh sạch CHƯA biến đổi, như OSFD gốc.
    geo="morgd" | "smooth": trường dịch chuyển mượt neo theo GT object | neo ngẫu nhiên (pilot MORGD).
    geo="pia_morgd2" | "pia_smooth2" | "morgd2": MORGD-v2 (phân vùng theo box) ∘ PIA | đối chứng tâm ngẫu nhiên ∘ PIA |
    MORGD-v2 không PIA (morgd2.py); RNG (r, a) = Random(seed + 1), tâm ngẫu nhiên = Random(seed + 2).
    geo="grgd" | "grgd_random": T_relative ∘ PIA, neo tại tâm GT | tâm ngẫu nhiên (pilot GRGD-v0, attack/methods/grgd.py);
    ξ từ random.Random(torch.initial_seed() + 1), tâm ngẫu nhiên từ random.Random(torch.initial_seed() + 2).
    cls_weight > 0 (C2-PIA / C2-MORGD / C2-SMOOTH, không cho geo="rrb"): + cls_weight·(loss_rpn_cls + loss_cls) với GT co-transform,
    1 forward/view (repair.make_view_loss, không sửa = loss của C2)."""
    assert geo in ("rrb", "pia", "morgd", "smooth", "grgd", "grgd_random",
                   "pia_morgd2", "pia_smooth2", "morgd2")  # morgd/smooth: morgd.py; grgd*: grgd.py; *2: morgd2.py
    assert cls_weight == 0.0 or geo != "rrb"
    assert co_transform or geo == "pia"
    assert rapa_p is None or (co_transform and cls_weight == 0.0)  # RaPA: chỉ khung loss feature + co-transform
    clean_pixels = resize_to_model(orig_pixels, tuple(data_sample.img_shape))
    log = {"forward_no_grad": 0, "modes": [], "gt_kept": [], "disp": [], "grgd": [], "morgd2": []}
    if geo.startswith("grgd") or geo.endswith("2"):
        rng_xi = random.Random(torch.initial_seed() + 1)
        rng_c = random.Random(torch.initial_seed() + 2)
    rrb_kw = dict(theta=rrb_theta, l_s=rrb_l_s, rho=rrb_rho, s_max=rrb_s_max, sigma=sigma)
    feats_fixed = None
    if not co_transform:
        with torch.no_grad():
            feats_fixed = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))

    rapa = rapa_sampler(model, rapa_p, clean_pixels.device) if rapa_p is not None else None

    def views_fn(img, ds, step_idx, total):
        comp = clean_pixels if co_transform else None
        if geo == "rrb":
            triples = [(v, c, ds) for v, _, c in rrb_views(img, ds, with_gt=True, companion=comp, **rrb_kw)]
        elif geo == "pia":
            triples = pia_views(img, n_views, delta_max, pi, sigma, companion=comp, log=log["modes"],
                                data_sample=ds if cls_weight > 0 else None, gt_log=log["gt_kept"])
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
            if co_transform:
                with torch.no_grad(), ctx():
                    fc = tuple(f.detach() for f in extract_features(model, c, ds))
                log["forward_no_grad"] += 1
            else:
                fc = feats_fixed
            if cls_weight > 0:
                out.append((v, dsv, make_view_loss(fc, k, "c2", cls_weight)))
            elif rapa is not None:
                out.append((v, ds, _with_ctx(_view_loss(fc, k), ctx)))
            else:
                out.append((v, ds, _view_loss(fc, k)))
        return out

    noise = run_iterative_attack(model, orig_pixels, data_sample, loss_fn=None, steps=steps,
                                 epsilon=epsilon, alpha=alpha, momentum=momentum, views_fn=views_fn,
                                 on_step=on_step, backward_per_step=backward_per_step)
    if rapa is not None:
        log["rapa_masked_frac"] = rapa[1]["masked"] / max(1, rapa[1]["total"])
    if info is not None:
        info.update(log)
    return noise


def _with_ctx(loss_fn, ctx):
    def wrapped(model, pixel_tensor, data_sample):
        with ctx():
            return loss_fn(model, pixel_tensor, data_sample)
    return wrapped
