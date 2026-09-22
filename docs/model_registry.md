# Model Registry

Chỉ ghi thông tin đã xác minh — cột "Ngày verify" trống nghĩa là số liệu lấy từ README chính thức của mmdetection, **chưa tự load checkpoint + eval lại trên GPU thật**. Việc đầu tiên khi có máy GPU là verify từng dòng, điền ngày, và sửa lại nếu số liệu lệch.

Cột "Checkpoint (mim)" là identifier mim dùng để tải (`mim download mmdet --config <identifier>`) — **khớp đúng tên file config** (bỏ `.py`), không phải tên file `.pth` kiểu cũ (v2). Xem `docs/progress_log.md` entry 2026-09-22 (phiên verify GPU) để biết vì sao tên cũ trong bảng trước đây sai.

## Controlled Panel

| Model | Vai trò | Detector | Backbone | Họ backbone | Config | Checkpoint (mim) | Box AP (README) | Box AP (verify GPU thật, full val2017) | Ngày verify |
|---|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R50 | Surrogate | Mask R-CNN | ResNet-50 | CNN — ResNet | `mask-rcnn_r50_fpn_ms-poly-3x_coco.py` | `mask-rcnn_r50_fpn_mstrain-poly_3x_coco` | 40.9 | **40.9** (khớp) | 2026-09-22 |
| Mask R-CNN R101 | Same-family target | Mask R-CNN | ResNet-101 | CNN — ResNet | `mask-rcnn_r101_fpn_ms-poly-3x_coco.py` | `mask-rcnn_r101_fpn_ms-poly-3x_coco` | 42.7 | **42.7** (khớp) | 2026-09-22 |
| Mask R-CNN ConvNeXt-T | Cross-CNN target | Mask R-CNN | ConvNeXt-T | CNN — khác (ConvNeXt) | `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco` | 46.2 | **46.2** (khớp) | 2026-09-22 |
| Mask R-CNN Swin-T | CNN→Transformer target | Mask R-CNN | Swin-T | Transformer | `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco` | 46.0 | **46.0** (khớp) | 2026-09-22 |

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

- **Controlled Panel: đã verify đủ 4/4 model trên GPU thật (RTX 3090, 2026-09-22)** — eval clean AP full COCO val2017 (5000 ảnh), khớp chính xác README cho cả 4 model. Xem `results/eval_clean/{r50,r101,convnext-t,swin-t}/` (log + metrics, git-tracked vì nhẹ) và `docs/progress_log.md`.
- Generalization Panel (FCOS/DETR/YOLOX/DINO-Swin-L): chưa verify, chưa tải checkpoint — chỉ làm sau khi Controlled Panel confirm transfer gap (idea.md §6).
- DINO+Swin-T: xác nhận không tồn tại trong model zoo chính thức mmdet v3 (tra README `configs/dino/README.md`, chỉ có DINO-R50 và DINO-Swin-L).
