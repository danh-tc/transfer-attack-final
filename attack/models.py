"""Load Controlled Panel models (mmdet v3 API).

Nguồn sự thật cho config/checkpoint: docs/protocol_lock.md. Không hardcode
đường dẫn checkpoint URL ở đây — checkpoint đã tải qua scripts/download_checkpoints.sh
vào checkpoints/, chỉ trỏ tới file đã có sẵn.
"""
import os
from dataclasses import dataclass

from mmdet.apis import init_detector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMDET_CONFIGS_ROOT = os.path.join(REPO_ROOT, "third_party/mmdetection/configs")
CHECKPOINTS_ROOT = os.path.join(REPO_ROOT, "checkpoints")


@dataclass(frozen=True)
class ModelSpec:
    role: str  # surrogate | same_family | cross_cnn | cnn_to_transformer
    backbone_family: str
    config_path: str
    checkpoint_path: str


# Khớp chính xác docs/protocol_lock.md §Controlled Panel — sửa ở đó trước, rồi mới sửa ở đây.
CONTROLLED_PANEL = {
    "r50": ModelSpec(
        role="surrogate",
        backbone_family="cnn_resnet",
        config_path=os.path.join(
            MMDET_CONFIGS_ROOT, "mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_coco.py"),
        checkpoint_path=os.path.join(
            CHECKPOINTS_ROOT,
            "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"),
    ),
    "r101": ModelSpec(
        role="same_family_target",
        backbone_family="cnn_resnet",
        config_path=os.path.join(
            MMDET_CONFIGS_ROOT, "mask_rcnn/mask-rcnn_r101_fpn_ms-poly-3x_coco.py"),
        checkpoint_path=os.path.join(
            CHECKPOINTS_ROOT,
            "mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth"),
    ),
    "convnext_t": ModelSpec(
        role="cross_cnn_target",
        backbone_family="cnn_other",
        config_path=os.path.join(
            MMDET_CONFIGS_ROOT,
            "convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py"),
        checkpoint_path=os.path.join(
            CHECKPOINTS_ROOT,
            "mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth"),
    ),
    "swin_t": ModelSpec(
        role="cnn_to_transformer_target",
        backbone_family="transformer",
        config_path=os.path.join(
            MMDET_CONFIGS_ROOT,
            "swin/mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py"),
        checkpoint_path=os.path.join(
            CHECKPOINTS_ROOT,
            "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth"),
    ),
}

SURROGATE_KEY = "r50"


def load_model(key: str, device: str = "cuda:0"):
    """Load 1 model của Controlled Panel theo key trong CONTROLLED_PANEL.

    Model trả về ở chế độ eval() (init_detector đã tự làm), KHÔNG bao giờ gọi
    model.train() trong pipeline tấn công — kể cả khi tính GT-loss cho
    MI-FGSM/DI-FGSM (model.loss(...)), giữ BatchNorm/Dropout ở eval mode là
    chủ đích (batch_size=1 khi tấn công, batch-stat BN sẽ suy biến).
    """
    if key not in CONTROLLED_PANEL:
        raise KeyError(f"Không có model '{key}' trong CONTROLLED_PANEL: {list(CONTROLLED_PANEL)}")
    spec = CONTROLLED_PANEL[key]
    if not os.path.exists(spec.config_path):
        raise FileNotFoundError(spec.config_path)
    if not os.path.exists(spec.checkpoint_path):
        raise FileNotFoundError(
            f"{spec.checkpoint_path} — chạy scripts/download_checkpoints.sh trước.")
    model = init_detector(spec.config_path, spec.checkpoint_path, device=device)
    return model


def load_surrogate(device: str = "cuda:0"):
    return load_model(SURROGATE_KEY, device=device)


def load_all_targets(device: str = "cuda:0"):
    """Trả về dict {key: model} cho 3 target (không gồm surrogate)."""
    return {k: load_model(k, device=device) for k in CONTROLLED_PANEL if k != SURROGATE_KEY}
