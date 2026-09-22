# Protocol Lock — Exact Config

idea.md kết thúc bằng: "việc nên làm tiếp ngay bây giờ là khóa exact config của 4 baseline + model checkpoints + gradient budget cho run 300 ảnh." Đây là file đó. Mọi số liệu/hyperparameter dùng trong code phải khớp file này; nếu cần đổi, sửa ở đây trước rồi mới sửa code, và ghi lý do vào `progress_log.md`.

## Framework

**mmdetection v3.3.0** (release 2024-01-05, bản v3.x mới nhất tại thời điểm khóa), clone + editable install, không dùng nhánh v2.x.

Lý do: v3 có sẵn config DINO/ConvNeXt/Swin/DETR trong model zoo; v2 (bản OSFD reference repo dùng) thiếu DINO hoàn toàn, cần cho Generalization Panel (idea.md §6). Code attack loop tham khảo từ `ref-repo/OSFD-main` (mmdet v2.28.2 API: `attack/base/{IFGSM,MI,DI,RRB}.py`, `attack/ours/OSFD.py`) phải được **port sang v3 API**, không copy trực tiếp.

## Controlled Panel (idea.md §5) — bắt buộc cho Baseline-First Stage

Chọn toàn bộ lịch train **3x + multi-scale**, không lẫn 1x/3x hay caffe/pytorch style trong cùng bộ, để clean AP giữa 4 model tương đối gần nhau (40.9–46.2) — tránh nhiễu "gap cross-family" chỉ vì model target vốn đã yếu/mạnh hơn hẳn surrogate.

| Vai trò | Config | Checkpoint (mim / openmmlab) | Box AP | Verify GPU thật |
|---|---|---|---|---|
| **Surrogate** | `mask-rcnn_r50_fpn_ms-poly-3x_coco.py` | `mask-rcnn_r50_fpn_mstrain-poly_3x_coco` | 40.9 | **40.9** ✓ (2026-09-22) |
| Same-family target | `mask-rcnn_r101_fpn_ms-poly-3x_coco.py` | `mask-rcnn_r101_fpn_ms-poly-3x_coco` | 42.7 | **42.7** ✓ (2026-09-22) |
| Cross-CNN target | `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco` | 46.2 | **46.2** ✓ (2026-09-22) |
| CNN→Transformer target | `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco` | 46.0 | **46.0** ✓ (2026-09-22) |

Ghi chú: đã verify cả 4 dòng bằng cách load checkpoint thật + eval clean AP trên full COCO val2017 (5000 ảnh) trên GPU thật (RTX 3090) — khớp chính xác AP README, không lệch. Chi tiết quá trình (bao gồm 1 checkpoint Swin-T tải lần đầu bị đứt file, phải tải lại) xem `docs/progress_log.md` entry 2026-09-22.

Checkpoint tải qua `mim download mmdet --config <identifier> --dest checkpoints/` — **identifier phải khớp đúng tên file config** (bỏ `.py`, giữ nguyên gạch nối kiểu `mask-rcnn_...`), không phải tên file `.pth` kiểu v2 cũ (`mask_rcnn_...`) — 2 kiểu tên khác nhau, dùng nhầm sẽ báo lỗi `mim` không tìm thấy config trong index. Không hardcode URL trực tiếp trừ khi `mim download` lỗi giữa chừng (file `.pth` bị đứt) — khi đó tải thẳng từ URL trong `configs/*/metafile.yml` (field `Weights`) như một fallback.

## Generalization Panel (idea.md §6) — chạy sau khi Controlled Panel confirm gap

| Model | Config | Box AP | Ghi chú |
|---|---|---|---|
| FCOS-R50 | `fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py` | 38.7 | biến thể "improved" (center+normbbox+giou), không phải FCOS gốc trần |
| DETR-R50 | `detr_r50_8xb2-150e_coco.py` | 42.0 | |
| YOLOX-CSP | `yolox_s_8xb8-300e_coco.py` | 40.5 | size **chưa chốt** — có thể đổi sang `yolox_l` (49.4 AP) nếu cần model mạnh hơn để so sánh cùng tầm AP với panel còn lại |
| DINO-Swin | `dino-5scale_swin-l_8xb2-36e_coco.py` | 58.4 | ⚠️ **không có checkpoint DINO+Swin-T chính thức trong mmdet v3** — dùng Swin-**L** thay thế, quyết định chấp nhận lệch capacity (model to hơn nhiều so với Swin-T dùng ở Controlled Panel) vì đây là generalization panel chứ không phải controlled panel. Phải ghi caveat này trong paper. |

## Gradient-evaluation budget B (idea.md §7)

**Định nghĩa: B = tổng số lần gọi `.backward()` qua surrogate** (không phải FLOPs, không phải số ảnh xử lý). Batch-doubling kiểu RRB (concat 2 augmented view rồi backward 1 lần trên batch gộp) tính là B=1 cho bước đó theo định nghĩa này.

- **MI-FGSM, DI-FGSM, OSFD**: 1 backward/iteration → `iterations = B` trực tiếp.
- **AugTrans**: dùng EOT với `N_EOT=10` sample/iteration (mỗi sample forward-backward riêng rồi average gradient, theo Algorithm 2 trong paper gốc) → `iterations = B / N_EOT` để tổng backward khớp 3 baseline kia.
- **Primary budget set: B ∈ {50, 200}.** B=10 bị loại khỏi so sánh chéo method vì ở B=10, AugTrans chỉ được 1 iteration (10/10) — không đủ để curriculum/EOT của nó hoạt động có ý nghĩa (paper gốc cần ~40+ iteration mới đạt phần lớn performance drop). B=10 vẫn có thể dùng làm stress-test riêng cho MI-FGSM/DI-FGSM/OSFD (không AugTrans).

## Danh sách ảnh n=300 / n=1000 (idea.md §4)

Đã chốt bằng `scripts/generate_image_lists.py`, seed cố định **SEED=42**, chạy 1 lần duy nhất
(2026-09-22) — kết quả (`data/image_lists/n300.csv`, `n1000.csv`, `meta.json`) đã commit vào git.
**Không chạy lại script này để tạo danh sách khác** — nếu thật sự cần đổi (chỉ khi có lý do
phương pháp luận rõ ràng, không phải để "thử seed khác cho đẹp"), phải ghi quyết định + lý do
vào `docs/progress_log.md` trước, rồi mới xóa `data/image_lists/` và chạy lại.

Quy tắc chọn (khóa):
- Nguồn: COCO val2017 (5000 ảnh), lọc còn 4952 ảnh có ít nhất 1 instance annotation (cần GT
  thật cho GT-assisted threat model, idea.md §3).
- `n1000_ids = random.Random(42).sample(sorted(pool_ids), 1000)`.
- `n300_ids = n1000_ids[:300]` — **n=300 là tập con của n=1000**, đảm bảo Confirmation Stage
  không dùng ảnh nằm ngoài Final Stage (tránh 2 nguồn nhiễu khác nhau giữa 2 phase).

## Attack epsilon (idea.md §7)

Primary: `epsilon = 5/255`. Secondary: `epsilon = 8/255`. Không đổi giữa các method trong cùng 1 lần so sánh.

## Nguồn tham khảo trong repo

- `papers/06244-AAAI24.DingX.pdf` — OSFD paper gốc (AAAI'24, Ding et al.)
- `papers/AugTrans.pdf` — AugTrans paper (EOT + augmentation curriculum, K_max=160, N_EOT=10, eps=5/255, step=0.0004 trong config mặc định của họ)
- `papers/2602.16494v1.pdf` — paper benchmark/taxonomy, nguồn của threat-model taxonomy ở idea.md §3
- `ref-repo/OSFD-main/` — code OSFD chính thức (mmdet v2.28.2), chỉ tham khảo logic, không chạy trực tiếp vì đã chốt dùng mmdet v3

## Còn mở, chưa chốt

- Size YOLOX (S vs L) cho Generalization Panel.
- Method direction ("stage-aware backward regularization", idea.md §12) — chỉ giữ nếu Mechanism Stage (idea.md §11) support, chưa làm gì ở giai đoạn hiện tại.
