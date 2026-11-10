# Rà prior art + protocol — Bước 1 của `docs/plan_paper_lock.md` (2026-11-10)

Mức xác minh: **[PDF]** đã đọc full text trong `papers/`; **[HTML]** đã đọc full text online; **[ABS]** chỉ có abstract hoặc tóm tắt từ công cụ tìm kiếm; **[CHƯA]** không tìm được nguồn.

## 1. Bảng tổng hợp

| Công trình | Mức xác minh | Đóng góp | Chạm vào claim nào của mình | Code | Xếp lớp (plan §4 bước 4) |
|---|---|---|---|---|---|
| OSFD (AAAI'24) | [PDF] | Feature loss + RRB, transfer xuyên detector | Baseline chính, Track S | Có (`ref-repo/OSFD-main`, mmdet v2) | A (sau fidelity check) |
| Benchmark 2602.16494 | [PDF] | Đo transfer nhiều attack; nêu gap CNN→Transformer; DINO-Swin-L bền | Gap CNN→Transformer không còn "mới" | Code metric "sẽ công bố" | C (cross-protocol) |
| CWA (ICLR'24) | [HTML] | Ensemble; flatness theo **input**; CSE: cập nhật x **tuần tự từng model**, chuẩn hóa gradient, tối đa cosine giữa gradient các model | **ENS-WP**: luân phiên surrogate trùng một phần ý tưởng CSE; WP (weight space) thì khác | Có (`huanranchen/AdversarialAttacks`) | Detection = universal patch → C; bản full-image adapted → B |
| Why Does Little Robustness Help? (S&P'24) | [ABS] | Transferability bị chi phối bởi model smoothness × **gradient similarity** surrogate–target | **C2**: "gradient alignment giải thích transfer" ĐÃ CÓ ở classification | Có | Related work |
| FUGEA (Neural Networks 2026) | [ABS] | Ensemble gradient: Uncertainty Weight + Gradient Agreement Mapping, sau đó refine; CNN↔ViT, classification | Ensemble gradient-agreement cho cross-architecture | ? | Related work |
| CTSTA (SSRN, ~cuối 2025) | [ABS] | Dùng segmentation làm task phụ; cross-task alignment; trung bình gradient qua nhiều model và nhiều bước | Multi-surrogate gradient averaging cho detection | ? | ? — cần full text |
| SVul-Attack (online 02/11/2026?) | **[CHƯA]** | Theo mô tả user: task + feature + frequency, 22 detector, COCO 1000 ảnh | Đụng lõi RD (feature + task loss) | ? | ? |

## 2. Chi tiết theo từng nguồn

### 2.1 Benchmark 2602.16494 [PDF]

**Protocol:**
- Detector train trên COCO (mmdet), test trên **toàn bộ VOC2007 test**.
- mAP@0.5, báo **relative drop so với benign mAP**; có thêm APloc và CSR.
- OSFD chạy tham số mặc định của tác giả: ε = 5, k = 3. Theo chính paper OSFD, ε = 5 cho L∞ thực tế ≈ 7.

**Surrogate OSFD:** YOLOv3-D53, **FRC-R50**, **MRC-Swin-T**. Target chỉ black-box: DETR, FCOS, YOLOX-L, DINO-Swin-L.

**⚠️ SỬA một số đã ghi trong CLAUDE.md / progress_log.** Con số "OSFD 40.2 trên DINO-Swin-L" là khi surrogate là **Mask R-CNN Swin-T** (đã có Swin). Bảng 5, relative drop (%) trên DINO-Swin-L:

| Surrogate | DINO-Swin-L | YOLOX-L | MRC-Swin-T |
|---|---|---|---|
| YOLOv3 | 27.3 | 84.3 | 85.5 |
| **FRC-R50** | **15.4** | 70.2 | 77.6 |
| MRC-Swin-T | 40.2 | 76.1 | 99.4 (WB) |

→ Với surrogate CNN họ ResNet (gần setting của mình nhất), OSFD chỉ đạt **15.4** trên DINO-Swin-L. Đó mới là số tham chiếu đúng.

**Thời gian:** OSFD 44 s/ảnh (TITAN RTX).

**Khả năng chạy lại (plan bước 6): KHẢ THI.**
- Dữ liệu = full VOC2007 test (công khai, không cần danh sách ảnh riêng).
- Detector là mmdet COCO. Cần đối chiếu config: MRC-Swin-T và DINO-Swin-L nhiều khả năng trùng với của mình; YOLOX-L trùng.
- Chưa có code metric của họ → phải tự tính mAP@0.5 trên 20 lớp VOC.
- Chi phí: khoảng 4952 ảnh × 4–6 s ≈ 6–8 h GPU cho mỗi method.

### 2.2 OSFD (AAAI'24) [PDF] — số tham chiếu cho fidelity check

**Protocol của họ:**
- 2000 ảnh ngẫu nhiên từ VOC2012 trainval; danh sách có trong `ref-repo/OSFD-main/data/image_index.txt` (1999 dòng — cần kiểm dòng cuối).
- Resize 800×800, pad 0.
- ε = 5/255, k = 3, **200 bước**, MI + RRB (θ = 7°, l_s = 10, ρ = 0.8, s_max = 1.1, σ = 6).
- Metric: **mAP@[.5:.95] tuyệt đối**.

**Checkpoint trùng với panel của mình:** MRCNN Swin-T (`mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x`), YOLOX-L, DETR-R50, YOLOv3-D53 (`ref-repo/OSFD-main/ummdet/checkpoints/eval_cfg/`).

**Đề xuất fidelity check:**
- Surrogate MRCNN Swin-T, chạy OSFD_RRB theo đúng protocol trên (800×800 pad, 200 bước), so với Bảng 1 dòng MRCNN / OSFD_RRB:

| Target | Benign mAP | mAP sau attack |
|---|---|---|
| YOLOv3 | 0.521 | 0.039 |
| YOLOX-L | 0.717 | 0.113 |
| DETR | 0.623 | 0.032 |
| MRCNN (white-box) | 0.656 | 0.001 |

- Kiểm trước clean mAP của từng target khớp hàng "Benign mAP".
- Lý do chọn surrogate này: là surrogate duy nhất trong Bảng 1 có checkpoint trùng với panel mình (FRCNN của họ là R101-caffe, mình không có).
- Lưu ý: bản port của mình tối ưu δ ở ảnh cỡ gốc, không phải 800×800 pad → fidelity check phải dùng đúng preprocessing của họ (thêm chế độ riêng, không đổi đường mặc định).

### 2.3 CWA (ICLR'24) [HTML: arxiv.org/html/2303.09105v2]

- **Flatness theo input x**, trong quả cầu ℓ∞ (SAM trên x: bước ascent trong rồi descent ngoài). **KHÁC WP** (nhiễu trọng số surrogate) → điểm khác biệt của ENS-WP còn giữ được.
- **CSE: "sequentially updates x using each model in turn, normalizes each gradient"**, tối đa cosine giữa gradient các model. → **Cơ chế luân phiên surrogate của ENS-WP (E70/E71) TRÙNG MỘT PHẦN với CSE.**
  - Không được claim "momentum chung + luân phiên" là đóng góp mới của method.
  - Điểm còn lại của ENS-WP: (a) cặp surrogate được chọn theo cơ chế (2 CNN gần trực giao, E68); (b) WP weight space; (c) objective feat + cls RoI. Phải định vị theo hướng này.
- **Detection:**
  - Universal patch trên INRIA (person), surrogate YOLOv3 + YOLOv5.
  - Target: YOLOv2, YOLOv3-T, YOLOv4, YOLOv4-T, FasterRCNN, SSD. Metric mAP. Adam-CWA trung bình 9.85.
  - → Không cùng setting (patch, khác dataset, khác detector) → lớp C.
- **Hệ quả cho bảng lớp B:** CWA full-image adapted trên R50 + CNX trở thành đối chứng **quan trọng nhất** của Track M, vì nó cùng luân phiên surrogate. Ens-RD (trung bình loss) vẫn bắt buộc.
- Còn 40k ký tự phụ lục chưa đọc (có thể có thêm thí nghiệm detection).

### 2.4 Why Does Little Robustness Help? (Zhang et al., S&P'24) [ABS]

- Transferability = đánh đổi giữa **độ mượt của surrogate** và **độ tương đồng gradient surrogate–target**.
- **Hệ quả cho C2:** "gradient alignment liên quan đến transfer" đã được xác lập ở classification → không phải phát hiện mới.
- Phần còn mới của mình:
  - (a) Trong detection, alignment phụ thuộc OBJECTIVE (feat vs cls RoI) — interaction I.
  - (b) Alignment tách khỏi CKA feature (E67 + E68).
  - (c) Kiểm định 3c (vượt CKA, LOSO).
- Cũng liên quan WP: họ dùng SAM khi *huấn luyện* surrogate. WP của mình là nhiễu trọng số *lúc attack* — gần với họ "flat surrogate", phải trích dẫn.

### 2.5 FUGEA [ABS: sciencedirect S0893608026003874, PubMed 41967317]

- Hai giai đoạn: Rapid Convergence Engine (Uncertainty Weight + Gradient Agreement Mapping làm trọng số gradient ensemble), rồi Precision Refinement Gradient.
- Đích: CNN↔ViT, classification.
- **Chưa xác nhận** ensemble có trộn CNN + ViT không — trang ScienceDirect và PubMed chặn truy cập tự động. → **Cần user tải PDF vào `papers/`.**

### 2.6 CTSTA [ABS: SSRN 5733227, "Taming Semantic Segmentation for Effective Transferable Attack Against Object Detection", Wang et al.]

- Segmentation làm task phụ; module cross-task alignment; iterative cross-model gradient averaging (qua bước và qua model).
- Chưa rõ số surrogate, kiểu perturbation, dataset, code. SSRN chặn truy cập tự động. → **Cần PDF.**

### 2.7 SVul-Attack [CHƯA]

- Hai lần tìm (standard + extended) không ra paper nào tên SVul / SVul-Attack.
- → **Cần user cung cấp title đầy đủ / DOI / PDF.** Đây là đối thủ gần nhất của RD-CNX (task + feature loss), không bỏ qua được.

## 3. Hệ quả cho plan và câu chữ novelty

1. **C2** (`plan_paper_lock.md` §2.2): thêm trích dẫn Zhang et al. S&P'24. Phần mới chỉ còn: alignment **phụ thuộc objective trong detection** + **vượt CKA** (3c). Câu claim 2.2 giữ nguyên ("additional explanatory evidence"), vốn đã đủ dè dặt.
2. **ENS-WP** (Track M): luân phiên surrogate không còn là điểm mới (CWA-CSE). Định vị lại: cặp surrogate chọn theo cơ chế + WP weight space + objective detection. **CWA adapted full-image là đối chứng bắt buộc lớp B** (nâng từ "nên có").
3. **Số tham chiếu benchmark:** sửa "OSFD 40.2 trên DINO-Swin-L" thành **15.4 với surrogate FRC-R50** (40.2 là với surrogate Swin-T). Bảng cross-protocol (bước 6) khả thi trên full VOC2007 test.
4. **Fidelity check OSFD:** surrogate MRCNN Swin-T, protocol VOC2012-2000 / 800×800 / 200 bước / mAP@[.5:.95], đối chiếu Bảng 1 dòng MRCNN / OSFD_RRB.
5. **Câu hỏi về "ConvNeXt surrogate → ViT" ở classification:** tìm nhanh chưa thấy bài so sánh trực tiếp ConvNeXt vs ResNet làm surrogate → chưa có bằng chứng tiền lệ. Rủi ro R1 vẫn mở. Cần tìm thêm trong các benchmark transfer lớn (ví dụ TransferAttackEval, Mao et al. "Transfer Attacks Revisited").

## 4. Còn thiếu để đóng bước 1

- [ ] PDF: FUGEA, CTSTA, SVul-Attack (user cung cấp, hoặc title/DOI cho SVul).
- [ ] Đọc nốt phụ lục CWA (offset 100000 của bản HTML).
- [ ] Tìm thêm literature "surrogate architecture → ViT transfer" (R1).
- [ ] Đối chiếu config DINO-Swin-L / MRC-Swin-T của benchmark với của mình.
