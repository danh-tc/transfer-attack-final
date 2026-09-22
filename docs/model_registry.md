# Model Registry

Chỉ ghi thông tin đã xác minh — cột "Ngày verify" trống nghĩa là số liệu lấy từ README chính thức của mmdetection, **chưa tự load checkpoint + eval lại trên GPU thật**. Việc đầu tiên khi có máy GPU là verify từng dòng, điền ngày, và sửa lại nếu số liệu lệch.

## Controlled Panel

| Model | Vai trò | Detector | Backbone | Họ backbone | Config | Checkpoint (mim) | Box AP (README) | Ngày verify |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R50 | Surrogate | Mask R-CNN | ResNet-50 | CNN — ResNet | `mask-rcnn_r50_fpn_ms-poly-3x_coco.py` | `mask_rcnn_r50_fpn_mstrain-poly_3x_coco` | 40.9 | _chưa_ |
| Mask R-CNN R101 | Same-family target | Mask R-CNN | ResNet-101 | CNN — ResNet | `mask-rcnn_r101_fpn_ms-poly-3x_coco.py` | `mask_rcnn_r101_fpn_mstrain-poly_3x_coco` | 42.7 | _chưa_ |
| Mask R-CNN ConvNeXt-T | Cross-CNN target | Mask R-CNN | ConvNeXt-T | CNN — khác (ConvNeXt) | `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco` | 46.2 | _chưa_ |
| Mask R-CNN Swin-T | CNN→Transformer target | Mask R-CNN | Swin-T | Transformer | `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask_rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco` | 46.0 | _chưa_ |

## Generalization Panel (chạy sau khi Controlled Panel confirm gap)

| Model | Detector | Backbone | Họ backbone | Config | Box AP (README) | Ngày verify | Ghi chú |
|---|---|---|---|---|---|---|---|
| FCOS-R50 | FCOS | ResNet-50 | CNN — ResNet | `fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py` | 38.7 | _chưa_ | biến thể "improved" |
| DETR-R50 | DETR | ResNet-50 | CNN — ResNet | `detr_r50_8xb2-150e_coco.py` | 42.0 | _chưa_ | |
| YOLOX-CSP | YOLOX | CSPDarknet | CNN — khác (CSP) | `yolox_s_8xb8-300e_coco.py` | 40.5 | _chưa_ | size S vs L chưa chốt, xem protocol_lock.md |
| DINO-Swin-L | DINO | Swin-L | Transformer | `dino-5scale_swin-l_8xb2-36e_coco.py` | 58.4 | _chưa_ | thay Swin-T (không có checkpoint chính thức) — lệch capacity, xem caveat protocol_lock.md |

## Tham chiếu họ backbone

- **CNN — ResNet**: ResNet-50, ResNet-101
- **CNN — khác**: DarkNet, CSPNet/CSPDarknet, ConvNeXt
- **Transformer**: Swin Transformer, PVT, ViT

Lưu ý (idea.md, research_plan gốc): kiến trúc detector head không đồng nghĩa họ backbone — ví dụ DETR/DINO đều có thể chạy với backbone CNN (ResNet) hoặc Transformer (Swin), phải đọc đúng tên backbone trong config, không suy đoán từ tên detector.

## Ghi chú / câu hỏi còn mở

- Chưa verify bằng cách chạy thật trên GPU — toàn bộ AP trong bảng lấy từ README GitHub của mmdetection (tra cứu 2026-09-22).
- DINO+Swin-T: xác nhận không tồn tại trong model zoo chính thức mmdet v3 (tra README `configs/dino/README.md`, chỉ có DINO-R50 và DINO-Swin-L).
