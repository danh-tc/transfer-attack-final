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
"""
import torch
import torch.nn.functional as F
from mmdet.structures import DetDataSample

from attack.preprocess import extract_features


def make_osfd_loss_fn(model: torch.nn.Module, clean_pixels: torch.Tensor,
                      data_sample: DetDataSample, k: float = 3.0):
    """Cache feature sạch 1 lần (không cần tính lại mỗi step attack), trả về
    loss_fn tương thích attack.methods.core.run_iterative_attack."""
    with torch.no_grad():
        feats_clean = tuple(f.detach() for f in extract_features(model, clean_pixels, data_sample))

    def loss_fn(model, adv_pixels, data_sample):
        feats_adv = extract_features(model, adv_pixels, data_sample)
        loss = None
        for feat_cln, feat_adv in zip(feats_clean, feats_adv):
            term = F.mse_loss(k * feat_cln, feat_adv)
            loss = term if loss is None else loss + term
        return loss

    return loss_fn
