# Framework v0 — giải thích cơ chế (2026-11-14)

> **Trạng thái:** đây là **mechanistic explanation**, **KHÔNG phải phương pháp attack mới**. Framework hiện tại chưa chỉ ra quyết định thiết kế nào tốt hơn recipe đã đóng băng (RD-CNX B50, ENS-WP B36).
> Hai mục tiêu tách riêng:
> - **Scientific novelty:** E91 kiểm nguyên lý stage sensitivity có tổng quát hóa sang backbone chưa dùng không.
> - **Method novelty:** chỉ mở lại khi có limitation mới có headroom VÀ nguyên lý can thiệp khác prior art.
>
> Nguồn số liệu: `docs/progress_log.md` (entry E83–E90b). Relative AP drop %, ε = 5/255, bootstrap paired theo ảnh.

## 1. Kết quả đã xác lập

### 1.1 Dương (có cổng khóa trước, trên ảnh mới)

| # | Kết quả | Số | Dữ liệu |
|---|---|---|---|
| E84 | Tương tác surrogate × công thức feature (OCT-C35 vs RRB-C25) | I_recipe +6.9 [4.2, 9.2] trên TB(Swin-T, DINO-Swin-L) | x200 (dev300 \ dev100) |
| E84 (phụ) | Tương tác nằm ở **stage**, không ở augmentation | I_stage +7.5 [5.8, 8.7]; I_aug −0.6 [−2.3, 1.4] | x200 |
| E85 (mô tả) | Hiệu ứng stage (C35 − C25) theo surrogate, T5C | R50 +0.1, R101 +0.0, Swin-T +2.8 [1.9, 3.7], **ConvNeXt +9.8 [7.6, 11.6]** | dev100 |
| E87 | Can thiệp: giảm trọng số C2 trên ConvNeXt có lợi | Δ(α0.5 − α1) T5C **+4.8 [3.0, 6.3]**; liều đơn điệu 58.2 → 63.0 → 66.8 | devB100 (mới) |
| E87 | Đặc thù surrogate có gradient C2 lớn | I_stage +8.1 [5.8, 10.2]; đối chứng R50 α0 − α1 +0.4 [−0.5, 1.4] | devB100 |
| E88 | ENS-WP > ensemble đơn giản cùng 36 backward | T7 +7.2 [4.8, 10.1] (caveat: 36 vs 18 update) | dev100 |

**T5C** = {fcos_r50, dino_r50, yolov3_d53, yolox_l, dino_swin_l} (không target nào là G0 của surrogate nào).

### 1.2 Chẩn đoán cơ chế (mô tả, hậu kỳ)

- **Độ lớn gradient C2 xếp đúng thứ tự hiệu ứng stage ở 4/4 surrogate** (E86, 20 ảnh; E87-diag, 100 ảnh). ‖g₂‖/‖g₃₄₅‖: R101 ≈ R50 ≈ 0.003 ≪ Swin-T ≈ 0.08 ≪ ConvNeXt ≈ 0.51. Ở ConvNeXt, năng lượng C2 ≈ 21%.
- **Pha loãng do độ lớn, không phải xung đột dấu:** g₂ lật dấu ~17–20% toạ độ của g₃₄₅, nhưng ít hơn một vector ngẫu nhiên cùng norm (D − D_null = −0.046 [−0.049, −0.042]). Trên ảnh adv, cos(g₂, g₃₄₅) = +0.26.
- Hiệu ứng có cả ở white-box (G0, ConvNeXt +7.6). Nghĩa là C2 làm yếu attack nói chung trên ConvNeXt, không chỉ phần transfer.

### 1.3 Âm / bị bác (giữ làm ràng buộc)

| # | Giả thuyết | Kết quả |
|---|---|---|
| E83 | Tương tác surrogate × **loss cls** (claim 2.1 gốc) | I = −0.9 [−6.9, 1.8] → bác |
| E86 | Alignment gradient trên ảnh sạch giải thích hiệu ứng stage | \|cos\| < 0.01 ở mọi ô → bác |
| E87-diag | Sign conflict giữa C2 và tầng sâu | D − D_null < 0 → bác |
| E87-oracle | Còn headroom nếu chọn α theo ảnh (routing C2) | +0.53 [0.18, 0.95] ≤ tham chiếu R50 +0.70 → NO-GO |
| S0 | Recombination kết quả hiện có | +0.31 / +1.24 T7, ≪ ngưỡng +3 |
| S0c | Object conf cao / lớn là limitation riêng | conf: hiệu ứng ngưỡng (drop dương); size: đảo chiều sau kiểm soát → NO-GO |
| E90b | Object consensus-high bị gradient starvation / lệch hướng trên surrogate | log‖∇z‖₁ +0.03 [−0.28, 0.33]; e −0.03 [−0.10, 0.04] → không phát hiện; NO-GO |

**Giới hạn diễn giải của E90b:**
- Chỉ bác **phiên bản H1 đã kiểm**, tức gradient L1 thấp từ đầu và hướng update sớm lệch. Không bác mọi dạng surrogate-side limitation.
- CI chứa 0 không chứng minh hai nhóm tương đương, vì không có equivalence margin khóa trước.
- Đơn vị bootstrap là ảnh: 60 ảnh xử lý, 16 ảnh có cặp, 21 cặp (11 ảnh 1 cặp, 5 ảnh 2 cặp). Riêng target: yolox_l 12 cặp / 11 ảnh, dino_swin_l 16 cặp / 13 ảnh.

**Exploratory observation (KHÔNG phải contribution):** E90a (dev100, dữ liệu đã hình thành giả thuyết) cho thấy object có consensus cao giữa các detector khác họ sống sót nhiều hơn (+30 / +25 điểm sau kiểm soát). Ghép theo score sạch surrogate ở E90b (dữ liệu mới) chưa xác nhận: −0.58 [−1.9, 0.7]. Không đưa vào contribution chính.

## 2. Nguyên lý giải thích

> **Backbone-dependent feature-stage sensitivity:** với loss feature đa tầng (kiểu OSFD), mỗi stage đóng góp vào gradient một lượng phụ thuộc kiến trúc backbone. Khi một stage nông có đóng góp độ lớn lớn (ConvNeXt C2: stem patchify + stage 1) và gần trực giao với tầng sâu, nó pha loãng bước sign. Bỏ hoặc giảm stage đó làm attack mạnh hơn. Ở backbone mà stage nông gần như không đóng góp (ResNet), lựa chọn stage gần như không ảnh hưởng.

Câu claim đã duyệt (plan_paper_lock AMENDMENT 2026-11-13): *"Backbone-dependent shallow-feature gradient contributions influence the transferability of multi-stage feature attacks in object detection"* + "consistent with differences in shallow-stage gradient magnitude" (**không** viết "fully explained").

**Phạm vi:**
- Đã kiểm trên 4 surrogate Mask R-CNN (R50, R101, Swin-T, ConvNeXt-T). Can thiệp nhân quả mới chỉ trên ConvNeXt, với R50 làm đối chứng.
- Nguyên lý giải thích vì sao recipe RD-CNX (feature C3–C5, tương đương α = 0) hợp với ConvNeXt. Nó giải thích lựa chọn surrogate × objective trong E65 (OSFD-CNX 26.1 vs RD-CNX 62.8 trên Swin-L, dev30).
- **Không** claim: sign conflict; alignment gradient dự báo transfer; C2 chỉ chặn thông tin transferable.

**Overlap prior art:** GHFA (JKSU-CIS 2024: gradient scaling cân bằng feature map trong OD) và MFA (UAI 2023: trọng số layer theo attribution) đã có cân bằng giữa các tầng. Đóng góp khác của mình nằm ở chỗ: **đo và can thiệp cho thấy độ nhạy phụ thuộc họ backbone**, kèm đối chứng âm. Phải đọc full text GHFA/MFA trước khi chốt câu chữ.

## 3. Limitation chưa giải quyết

1. **Không có quyết định thiết kế mới.** RD-CNX đã dùng C3–C5. E87-oracle cho thấy không còn headroom trên trục C2. Một attack "stage weighting theo backbone" sẽ trùng GHFA về cơ chế, và trên ConvNeXt chỉ tái tạo α = 0 mà recipe đang dùng.
2. **Gap còn lại không có đòn bẩy đã biết.**
   - Thất bại của ENS-WP và RD-CNX chồng lấn mạnh (S0: RD-CNX chỉ phá 15% survivor của ENS-WP).
   - Mọi đòn bẩy ENS đã thử đều NO-GO (E80–E82).
   - Các nhánh object-level (S0c, E90) đều NO-GO.
3. **Tính tổng quát mới có 4 điểm**, và 4 điểm này cũng chính là dữ liệu sinh ra giả thuyết. Chưa có dự đoán nào được kiểm trên backbone ngoài mẫu.
4. Một seed, một họ detector (Mask R-CNN) cho surrogate. Có lệch phần cứng theo method (CLAUDE.md).

## 4. Dự đoán bác bỏ được — E91 (khóa riêng trước khi chạy)

**Câu hỏi:** chỉ từ thống kê gradient trên ảnh sạch, có dự đoán TRƯỚC được hiệu ứng stage (C35 − C25) trên một backbone chưa dùng không?

**Dự đoán (khóa ở dạng quy tắc, số cụ thể khóa trong entry E91):**
- Đo r = ‖g₂‖/‖g₃₄₅‖ (∇ₓ L_OSFD, k = 3, ảnh sạch, không aug, như E86/E87-diag) trên backbone mới.
- Phân lớp theo 4 điểm đã có (R50/R101 ≈ 0.003 → +0.0–0.1; Swin-T ≈ 0.08 → +2.8; ConvNeXt ≈ 0.5 → +9.8):
  - **r < 0.02 → hiệu ứng ≈ 0:** \|C35 − C25\| < 1.5 trên T5C.
  - **0.02 ≤ r < 0.2 → hiệu ứng dương nhỏ:** 0 < điểm < 6.
  - **r ≥ 0.2 → hiệu ứng lớn:** cận dưới CI > 0 VÀ điểm ≥ +4.
- Với ≥ 2 backbone mới: thứ tự hiệu ứng trùng thứ tự r.
- **Bác** nếu backbone mới nằm sai lớp, hoặc đảo thứ tự. Báo cả trường hợp đúng một phần.

**Ứng viên** (Mask R-CNN có sẵn trong `third_party/mmdetection/configs`, chưa từng dùng làm surrogate hay target): RegNetX (`regnet/`), ResNeSt (`resnest/`), Res2Net (`res2net/`), ResNeXt-101 (`mask_rcnn/mask-rcnn_x101-32x4d_*`), HRNet (`hrnet/`; đa nhánh, định nghĩa "stage" phải khóa riêng).
- Chọn backbone **trước** khi đo r, theo tiêu chí kiến trúc (nên có ít nhất một backbone có stem không phải ResNet).
- **Không dùng Swin-S**: đó là held-out target.

**Ràng buộc chi phí và độ mạnh (cần user quyết khi khóa):**
- Đo r: ≤ 5' GPU / backbone.
- Kiểm hiệu ứng stage: 2 ô (C25, C35) × N ảnh × eval T5C. Với hard stop 15', N chỉ khoảng 30–40 ảnh: CI rộng, đủ cho lớp "≈ 0" và "lớn" nhưng khó cho lớp "nhỏ". Tham chiếu: E85 CNX ±2 ở n = 100.
- Ảnh: devC300 phần chưa dùng (ảnh 61–300). devD200 / n1000 giữ nguyên.

## 5. Trạng thái các nhánh

| Nhánh | Trạng thái |
|---|---|
| Tối ưu / routing C2 | **Đóng** (E87-oracle) |
| Object-survivor conf/size | **Đóng** (S0c) |
| Consensus-robust | **Đóng** (E90b); giữ làm exploratory observation |
| Mục tiêu B (method mới) | **Tạm dừng**; chỉ mở khi có limitation có headroom + nguyên lý khác prior art |
| E91 predictive stage validation | **Chờ khóa** (mục 4) |
| Mục tiêu A (CWA adapted, OSFD fidelity, final n1000) | Chưa làm |
