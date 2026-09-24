"""OSFD (feature-disruption transfer attack) — port từ ref-repo/OSFD-main/attack/ours/OSFD.py.

Không cần GT (khớp idea.md §3 secondary threat model: surrogate-prediction-only
vẫn dùng được, vì loss chỉ phụ thuộc backbone feature, không phụ thuộc
box/label nào cả).

Loss = tổng MSE(k * feat_clean, feat_adv) trên từng stage BACKBONE (không qua
neck/FPN — khớp `model.backbone(...)` của bản gốc, xem attack/preprocess.py).
Vì sao ASCENT (tăng loss này) lại đúng hướng "làm feature lệch khỏi clean":
tại bước đầu adv≈clean nên (feat_adv - k*feat_clean) ≈ (1-k)*feat_clean; với
k=3 hướng gradient-ascent xấp xỉ đẩy feat_adv theo hướng -2*feat_clean, tức
RA XA feat_clean — không cần viết riêng 1 loss "descent", giữ được quy ước
ascent-only thống nhất với MI-FGSM/DI-FGSM trong attack/methods/core.py.

Pilot A′1 (docs/progress_log.md 2026-09-24): `spatial_weight` đặt trọng số không gian
W_l BÊN TRONG loss (trước reduction) — đổi hướng gradient, khác với nhân W vào gradient
trước sign() (vô tác dụng khi W > 0):
  L = Σ_l Σ_hw W_l ‖k·F_cln − F_adv‖²₂ / (C_l · Σ_hw W_l)
Chia C_l để W ≡ 1 trùng tuyệt đối F.mse_loss của OSFD gốc.

Pilot P2 (progress_log 2026-09-24): `channel_concave` gộp LÕM theo channel —
  d_{l,c} = mean_hw (k·F_cln − F_adv)²,  L = Σ_l mean_c τ_l·log(1 + d_{l,c}/τ_l)
τ_l = s·median_c d⁰_{l,c} (d⁰ tại δ=0, không RRB = (1−k)²·mean_hw F_cln²), cố định mỗi ảnh.
φ(d) = d trùng tuyệt đối F.mse_loss (Σ_l mean_c d_{l,c}).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample

from attack.preprocess import extract_features, to_batch

# Cấu hình W khóa trong progress_log 2026-09-24 — không thêm cấu hình khác.
SPATIAL_WEIGHTS = {
    "box": dict(ring=0.0, min_ring_px=0.0),       # W1: trong GT box
    "box_ring": dict(ring=0.25, min_ring_px=16.0),  # W2: box nở mỗi phía 0.25×cạnh, >= 16 px
}
BG_WEIGHT = 0.1
# Cấu hình P2 khóa trong progress_log 2026-09-24: chỉ đổi s (τ = s·median), giữ log1p.
CHANNEL_CONCAVE = {"median": 1.0, "2median": 2.0}  # P2a, P2b


def spatial_weight_map(data_sample: DetDataSample, canvas_hw, ring: float, min_ring_px: float,
                       device) -> torch.Tensor:
    """M [H_pad,W_pad] ở khung input của model (canvas đã pad): 1 trong hợp GT box (đã nở
    theo ring), BG_WEIGHT ở mọi chỗ khác. GT box của data_sample ở khung đã resize (không
    gồm crowd — PackDetInputs tách sang ignored_instances). Không có box -> W ≡ 1."""
    boxes = data_sample.gt_instances.bboxes
    boxes = (boxes.tensor if hasattr(boxes, "tensor") else boxes).float()
    if len(boxes) == 0:
        return torch.ones(canvas_hw, device=device)
    img_h, img_w = data_sample.img_shape
    m = torch.full(canvas_hw, BG_WEIGHT, device=device)
    for x1, y1, x2, y2 in boxes.tolist():
        dx = max(ring * (x2 - x1), min_ring_px) if ring > 0 else 0.0
        dy = max(ring * (y2 - y1), min_ring_px) if ring > 0 else 0.0
        x1, x2 = int(max(x1 - dx, 0.0)), int(min(x2 + dx, img_w) + 0.999)
        y1, y2 = int(max(y1 - dy, 0.0)), int(min(y2 + dy, img_h) + 0.999)
        m[y1:y2, x1:x2] = 1.0
    return m


def weighted_feature_mse(feat_cln: torch.Tensor, feat_adv: torch.Tensor,
                         w: torch.Tensor) -> torch.Tensor:
    """feat [1,C,H,W], w [H,W] -> Σ_hw w·‖feat_cln − feat_adv‖²₂ / (C·Σ_hw w)."""
    sq = (feat_cln - feat_adv).pow(2).sum(dim=1)[0]
    return (w * sq).sum() / (feat_cln.shape[1] * w.sum())


def channel_distortion(feat_cln: torch.Tensor, feat_adv: torch.Tensor) -> torch.Tensor:
    """feat [1,C,H,W] -> d [C] = mean_hw (feat_cln − feat_adv)²."""
    return (feat_cln - feat_adv).pow(2).mean(dim=(0, 2, 3))


def concave_channel_loss(d: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """mean_c τ·log(1 + d_c/τ)."""
    return (tau * torch.log1p(d / tau)).mean()


def make_osfd_loss_fn(model: torch.nn.Module, clean_pixels: torch.Tensor,
                      data_sample: DetDataSample, k: float = 3.0, stages=None,
                      spatial_weight: Optional[str] = None,
                      channel_concave: Optional[str] = None):
    """Cache feature sạch 1 lần (không cần tính lại mỗi step attack), trả về
    loss_fn tương thích attack.methods.core.run_iterative_attack.

    stages: None = mọi stage backbone (OSFD gốc); list chỉ số 0-based (vd [1]) = chỉ
    tấn công các stage đó (Mechanism A3, docs/mechanism_plan.md).
    spatial_weight: None = OSFD gốc; key của SPATIAL_WEIGHTS = pilot A′1. W_l = M
    downsample kiểu area (adaptive avg pool) về lưới stage l; KHÔNG co-transform theo
    view RRB (feature sạch của OSFD gốc cũng không).
    channel_concave: None = OSFD gốc; key của CHANNEL_CONCAVE = pilot P2."""
    assert spatial_weight is None or channel_concave is None
    with torch.no_grad():
        feats_clean = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))
        weights = None
        if spatial_weight is not None:
            canvas_hw = tuple(to_batch(model, [clean_pixels], [data_sample])[0].shape[-2:])
            m = spatial_weight_map(data_sample, canvas_hw, device=clean_pixels.device,
                                   **SPATIAL_WEIGHTS[spatial_weight])
            weights = [F.adaptive_avg_pool2d(m[None, None], f.shape[-2:])[0, 0] for f in feats_clean]
        taus = None
        if channel_concave is not None:
            s = CHANNEL_CONCAVE[channel_concave]
            taus = [(s * channel_distortion(k * f, f).median()).clamp_min(1e-12) for f in feats_clean]

    def loss_fn(model, adv_pixels, data_sample):
        feats_adv = extract_features(model, adv_pixels, data_sample)
        loss = None
        for s, (feat_cln, feat_adv) in enumerate(zip(feats_clean, feats_adv)):
            if stages is not None and s not in stages:
                continue
            if weights is not None:
                term = weighted_feature_mse(k * feat_cln, feat_adv, weights[s])
            elif taus is not None:
                term = concave_channel_loss(channel_distortion(k * feat_cln, feat_adv), taus[s])
            else:
                term = F.mse_loss(k * feat_cln, feat_adv)
            loss = term if loss is None else loss + term
        return loss

    return loss_fn
