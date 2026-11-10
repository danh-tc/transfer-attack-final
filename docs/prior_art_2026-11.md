# Rà prior art + protocol — Bước 1 của `docs/plan_paper_lock.md` (2026-11-10)

**Mức xác minh:**
- **[PDF]** đã đọc full text trong `papers/`.
- **[ABS]** chỉ có abstract hoặc tóm tắt.

Mọi bài trong bảng dưới đều đã đọc full text, trừ một bài ghi [ABS].

## 1. Bảng tổng hợp

| Công trình | Venue / ngày online | Setting | Số surrogate | Chạm vào claim nào của mình | Code | Xếp lớp |
|---|---|---|---|---|---|---|
| **OSFD** [PDF] | AAAI'24 | Untargeted, full-image, ε 5, VOC12 2000 ảnh, mAP@[.5:.95] | 1 | Baseline chính, Track S | Có (`ref-repo/OSFD-main`, mmdet v2) | A (sau fidelity check) |
| **Benchmark 2602.16494** [PDF] | arXiv 02/2026 | Đo nhiều attack, VOC07 test, mAP@0.5, relative drop | — | Gap CNN→Transformer đã được nêu | Code metric "sẽ công bố" | C (cross-protocol) |
| **SVul-Attack** [PDF] | Information Sciences 766 (2027), **online 02/11/2026** | Untargeted, full-image, **ε 8**, T 20, COCO val **1000 ảnh ngẫu nhiên**, **mAP@0.5 tuyệt đối**, 22 target | **4** (Faster R-CNN, YOLOv3, FCOS, DETR) | RD (task + feature loss); câu "architecture không ngăn transfer"; CKA làm cơ chế | **Chưa có** (GitHub `LimingZhang719/SVul-Attack` trả 404, 2026-11-10) | Track M; B (tự cài) hoặc C (cross-protocol) |
| **CWA** [PDF] | ICLR'24 | Classification per-image; **detection = universal patch** (INRIA, YOLOv3 + v5) | ≥2 | ENS-WP: **CSE cập nhật x tuần tự từng model** | Có (`huanranchen/AdversarialAttacks`) | Patch → C; full-image adapted → B |
| **CTSTA** [PDF] | Expert Syst. Appl. 334 (2027), **online 18/09/2026** | **Targeted** (ASR), ε 10/20/30, VOC + COCO; surrogate được **huấn luyện thêm** (segmentation phụ + distillation) | 2 (FRCNN + RetinaNet) | ENS-WP: **luân phiên theo khối Q bước giữa các surrogate** + chiếu gradient | ? | Related work (khác threat model) |
| **FUGEA** [PDF] | Neural Networks 201 (2026) | Classification ImageNet, ASR | Ensemble, **có cấu hình CNN-only** (Res-18, Inc-v3, BiT-M-R50, Inc-v4) → target ViT/Swin | ENS-WP: "ensemble CNN tới được Transformer" | ? | Related work |
| **Why Does Little Robustness Help?** [ABS] | S&P'24 | Classification | — | **C2:** gradient similarity surrogate–target chi phối transfer | Có | Related work |

## 2. Chi tiết theo từng bài

### 2.1 Benchmark 2602.16494 [PDF]

**Protocol:**
- Detector train trên COCO (mmdet), test trên **toàn bộ VOC2007 test**.
- mAP@0.5, báo **relative drop**; kèm APloc, CSR.
- OSFD chạy mặc định: ε 5, k 3 (L∞ thực tế ≈ 7, theo chính paper OSFD).

**⚠️ ĐÃ SỬA trong CLAUDE.md:** con số "OSFD 40.2 trên DINO-Swin-L" là khi surrogate là **Mask R-CNN Swin-T**. Bảng 5, relative drop (%):

| Surrogate | DINO-Swin-L | YOLOX-L | MRC-Swin-T |
|---|---|---|---|
| YOLOv3 | 27.3 | 84.3 | 85.5 |
| **FRC-R50** | **15.4** | 70.2 | 77.6 |
| MRC-Swin-T | 40.2 | 76.1 | 99.4 (WB) |

**Chạy lại (bước 6): KHẢ THI.**
- Dữ liệu = full VOC07 test (công khai); detector mmdet COCO.
- Phải tự tính mAP@0.5 trên 20 lớp VOC.
- Chi phí khoảng 6–8 h GPU cho mỗi method.

### 2.2 OSFD [PDF] — số tham chiếu cho fidelity check

**Protocol của họ:**
- VOC12 trainval 2000 ảnh (`ref-repo/OSFD-main/data/image_index.txt`, 1999 dòng — kiểm dòng cuối).
- Resize 800×800, pad 0.
- ε 5/255, k 3, **200 bước**, MI + RRB.
- Metric **mAP@[.5:.95] tuyệt đối**.

**Đề xuất fidelity check:**
- Surrogate **MRCNN Swin-T** (checkpoint trùng panel mình), đúng protocol trên. So với Bảng 1, dòng MRCNN / OSFD_RRB:

| Target | Benign mAP | Sau attack |
|---|---|---|
| YOLOv3 | 0.521 | 0.039 |
| YOLOX-L | 0.717 | 0.113 |
| DETR | 0.623 | 0.032 |
| MRCNN (white-box) | 0.656 | 0.001 |

- Trước tiên phải khớp hàng benign mAP.
- Bản port của mình tối ưu ở cỡ ảnh gốc → thêm chế độ 800×800-pad riêng cho fidelity check, không đổi đường mặc định.

### 2.3 SVul-Attack [PDF] — đối thủ gần nhất

**Method:**
- Loss = Σ_k α_k [task loss + γ · feature loss tần cao (high-freq của feature trung gian)], α_k theo clean AP của từng surrogate.
- Cân λ thích nghi giữa 2 miền.
- **MSNS:** lấy mẫu lân cận đa tỉ lệ, N = M = 3 → 9 lượt/bước → **17.5 s/ảnh**, so với OSFD 3.4 s (Bảng 10, 2× RTX 4090).
- ε = 8/255, T = 20.

**Target:** 22 detector mmdet.
- Paper **không ghi backbone**; tên model đều là mmdet mặc định (Libra/Grid/Cascade R-CNN, RetinaNet, ATSS, YOLOF, FoveaBox, CenterNet, RepPoints, VFNet, Deformable/Conditional/DAB-DETR, DINO…).
- **Không có Swin, ConvNeXt hay ViT** ở bất kỳ đâu trong paper → "Transformer-Based" là **đầu dò kiểu DETR**, nhiều khả năng backbone ResNet. Đây là suy luận, cần ghi đúng mức này khi trích dẫn.
- Clean mAP@0.5 của họ thấp hơn số chính thức khá nhiều (FCOS 28.4, DINO 42.9, ATSS 26.4) → pipeline eval khác → **không so trực tiếp số tuyệt đối**.

**Kết quả:**
- Trung bình 18 target chưa thấy: mAP@0.5 **4.12** (OSFD 4.94, cùng 4 surrogate).
- Bảng 8: số lượng và **thành phần** surrogate quan trọng (1 surrogate trung bình 12.10, nhưng riêng FRCNN 4.17).
- Bảng 9: ensemble đa dạng > đồng nhất.
- Bảng 6: CKA tần cao giữa các detector tăng dưới attack → họ dùng CKA làm **cơ chế**.

**Hệ quả cho mình:**
1. **Móc định vị mạnh.** SVul kết luận "architectural differences alone do not necessarily prevent cross-model adversarial transfer", nhưng đó là khác **đầu dò** trên cùng (khả năng cao) backbone ResNet. Controlled panel của mình cho thấy **họ backbone** mới là rào cản (OSFD: DINO-R50 97% vs DINO-Swin-L 25%; benchmark: FRC-R50 → DINO-Swin-L 15.4).
   → Câu chuyện: *"cross-detector transfer là dễ khi chung backbone; cross-backbone-family mới là bài toán mở"*. SVul + benchmark + controlled panel ghép lại thành một khoảng trống sạch.
2. **RD:** task + feature loss không còn mới (SVul). Phần riêng của mình: cls RoI + surrogate ConvNeXt + interaction, và **1 surrogate** (SVul dùng 4).
3. **Cơ chế:** SVul dùng **CKA feature** làm lời giải thích; E67 của mình cho thấy CKA **không** giải thích transfer cross-family → đối chiếu trực tiếp, có lợi cho C2.
4. **Baseline:**
   - Chưa có code → muốn đưa vào bảng phải tự cài + fidelity check. Nhưng họ không công bố danh sách ảnh, và pipeline eval lệch (clean AP thấp) → fidelity check khó.
   - Khả thi hơn: (a) **lớp B** "SVul-loss (our impl.)" trên R50 + CNX, ε 8; (b) **lớp C**: chạy RD-CNX / ENS-WP theo protocol của họ (COCO val 1000 ảnh, ε 8, mAP@0.5) và đặt cạnh số họ báo, ghi rõ khác danh sách ảnh và pipeline eval.
   - Theo dõi repo GitHub, có code thì nâng lên lớp A.

### 2.4 CWA [PDF]

- **Flatness theo input x** (SAM trên x). WP của mình là nhiễu **trọng số** → khác.
- **CSE: cập nhật x tuần tự qua từng model**, chuẩn hóa gradient, tối đa cosine giữa gradient các model → luân phiên surrogate của ENS-WP **không mới**.
- **Detection: chỉ có universal patch** (INRIA, surrogate YOLOv3 + v5; full PDF xác nhận, không có thí nghiệm detection theo từng ảnh) → lớp C.
- CWA full-image adapted trên R50 + CNX = **đối chứng bắt buộc lớp B** của Track M.

### 2.5 CTSTA [PDF]

- **Targeted** attack: chọn ngẫu nhiên một nhãn không có trong ảnh, đo ASR.
- Surrogate FRCNN + RetinaNet được **tăng cường bằng huấn luyện** (segmentation làm task phụ + distillation trên COCO train) → **vi phạm nguyên tắc không fine-tune surrogate** của dự án; khác threat model.
- **ICGA:** luân phiên giữa các surrogate theo khối Q bước con, kèm chiếu gradient khi xung đột → **tiền lệ trực tiếp cho luân phiên theo khối của ENS-WP (E71)**.
- ε 10/20/30.
- → Chỉ là related work; không so số.

### 2.6 FUGEA [PDF]

- Classification ImageNet; Uncertainty Weight + Gradient Agreement Mapping (trọng số ensemble), sau đó refine bằng neighbor sampling + bước dự đoán.
- **Có bảng ensemble toàn CNN** (Res-18, Inc-v3, BiT-M-R50, Inc-v4) tấn công target ViT-B / PiT-B / Visformer / Swin-B.
- → Điểm khác biệt "2 CNN vẫn tới được Transformer" **KHÔNG còn đứng được**, ít nhất ở classification. Chỉ còn: trong **detection**, và với **giải thích cơ chế** (E67 + E68).

### 2.7 Why Does Little Robustness Help? (S&P'24) [ABS]

- Transferability = đánh đổi giữa model smoothness và **gradient similarity** surrogate–target.
- → C2 không mới ở dạng tổng quát. Phần mới chỉ còn:
  - (a) alignment **phụ thuộc objective** trong detection (I);
  - (b) alignment tách khỏi CKA (E67 + E68);
  - (c) kiểm định 3c.
- Nên đọc full text: họ dùng SAM / input-gradient regularization khi *huấn luyện* surrogate, có thể liên quan WP.

## 3. Hệ quả cho plan (đề xuất, chờ user duyệt trước khi sửa `plan_paper_lock.md`)

1. **Định vị:** thêm một câu khoảng trống có trích dẫn:
   > "SVul-Attack (2026) và benchmark 2602.16494 cho thấy transfer xuyên **detector** đã gần bão hòa khi chung backbone, còn CNN → Transformer backbone vẫn mở (OSFD 15.4% trên DINO-Swin-L từ surrogate R50)."
   Controlled panel + interaction của mình nhắm đúng khoảng trống này.
2. **ENS-WP (Track M):**
   - Luân phiên surrogate (CWA-CSE, CTSTA-ICGA) và ensemble toàn CNN → ViT (FUGEA) đều đã có.
   - ENS-WP chỉ còn là **validation** của cơ chế (cặp surrogate chọn theo E68 + WP weight space + objective detection); **không claim novelty method**. Khớp vai trò "practical" trong plan, nhưng phải nói rõ hơn.
   - Đối chứng lớp B bắt buộc: CWA adapted, Ens-RD, **SVul-loss (our impl.)**.
3. **RD-CNX (Track S):** điểm mới thực nghiệm mạnh nhất — **1 surrogate**, cross-**backbone-family**, Swin-L 59% ở ε 5. SVul dùng 4 surrogate, ε 8 và không đo target backbone Transformer.
4. **C2:** thêm trích dẫn S&P'24; đối chiếu với SVul (CKA làm cơ chế) vs E67 (CKA không giải thích).
5. **Cross-protocol (bước 6):** thêm protocol SVul (COCO val 1000, ε 8, mAP@0.5 tuyệt đối), bên cạnh benchmark 2602.16494.

## 4. Còn mở

- [ ] R1: literature "ConvNeXt (CNN hiện đại) làm surrogate → ViT" trong classification — chưa thấy tiền lệ so sánh trực tiếp. Tìm thêm trong TransferAttackEval / Mao et al. "Transfer Attacks Revisited".
- [ ] Đọc full text S&P'24 (arXiv 2307.07873).
- [ ] Đối chiếu config DINO-Swin-L / MRC-Swin-T của benchmark với của mình.
- [ ] Theo dõi repo SVul-Attack.
