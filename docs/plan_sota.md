# Plan: OSFD thắng nhờ cái gì → method SOTA

> **Trạng thái: LOCKED (user duyệt 2026-10-03).** Đổi plan = ghi lý do vào `docs/progress_log.md` TRƯỚC khi đổi.
>
> **Tiến độ (2026-10-03):** Bước 1 ✅ (nguyên liệu: RRB, xoay, MI) · Bước 2 ✅ (cơ chế + Swin-L + loại lỗi) · Bước 3 ✅ (C2 = OSFD + 4·cls QUA sàng lọc dev300; ứng viên mở rộng sang loại loss — lý do ở progress_log) · Bước 4: GO @ n300 cho C2 **tạm hoãn** (110/300, chưa eval).
>
> **Nhánh DINO-Swin-L (2026-10-04):** quy tắc "Repair-Robust Attack" (probe V, R-a chính, R-b dự phòng, cổng độ trung thực, sàng lọc so C2 + OSFD) khóa trong `docs/progress_log.md` entry 2026-10-04.

## Mục tiêu

Tìm method **vượt OSFD** (SOTA so được công bằng) trên target khác họ backbone. Cách làm: **chỉ phân tích OSFD** — bỏ/thay từng thành phần, đo tụt bao nhiêu → suy ra cơ chế → method khuếch đại đúng thành phần đó.

```
Bước 1  Ablation OSFD (7 ô)           dev100 → ô mơ hồ mới nối đủ dev300
Bước 2  Suy ra cơ chế                  chỉ forward trên PNG có sẵn
Bước 3  Method + sàng lọc              dev300
Bước 4  GO @ n300 (1 lần) → n1000
```

Mở lại RQ3 (entry R1 đã cam kết đóng): lý do — các pilot trước can thiệp dựa trên giả thuyết chưa kiểm; ablation OSFD có giá trị cho analysis paper kể cả khi không ra method. Ghi vào progress_log khi duyệt.

## Thiết lập cố định

- Surrogate Mask R-CNN R50; ε = 5/255; B = 50 backward; α = 1; noise init 0; seed ảnh = SEED + chỉ số **dev300** (như `run_baselines.py`); đánh giá qua PNG uint8 cỡ gốc.
- Đánh giá 6 model: R50 (white-box), R101 (cùng họ), ConvNeXt-T, Swin-T, YOLOX-S (khác họ), **DINO-Swin-L** (ca khó).
- Chỉ số chính **CrossAvg₃** = mean(drop ConvNeXt-T, Swin-T, YOLOX-S). Báo cáo kèm R101, DINO-Swin-L, adv AP / clean AP (ít bị trần hơn), runtime, số view forward.
- Paired bootstrap 1000 mẫu trên ảnh, BOOT_SEED 2026.
- dev300 cho bước 1–3; n300 chỉ bước 4, chạy 1 lần; n1000 chỉ bảng cuối.

## Bước 1 — Ablation OSFD

OSFD full recipe = **loss** `MSE(k·F_clean, F_adv)` trên 4 stage backbone, k = 3 + **MI** μ = 1 + **RRB** 2 view/step (view1 = xoay ±7° + nhiễu Gaussian σ = 6; view2 = view1 resize-pad ≤ 1.1× + nhiễu σ = 6), 1 backward/step. Mốc: OSFD trong run `dev300_B50_eps5` (có sẵn).

| # | Ô (OSFD bỏ/thay đúng 1 thứ) | Câu hỏi |
|---|---|---|
| A1 | loss → task loss GT (5 loss Mask R-CNN); giữ MI + RRB (RRB co-transform GT box + mask) | thắng nhờ **loss**? |
| A2 | bỏ RRB (1 view/step) | thắng nhờ **diversity**? |
| A3 | RRB bỏ nhiễu σ | phần nào của RRB gánh? |
| A4 | RRB bỏ xoay | |
| A5 | RRB bỏ resize (view2 = view1 rút lại xoay + nhiễu) | |
| A6 | bỏ MI (μ = 0) | momentum? |
| A7 | k = 5 | độ mạnh đẩy feature |

Ghi chú nghi vấn A3: sàn sign agreement giữa 2 lần chạy OSFD khác seed chỉ 0.52 → nhiễu σ = 6 gần như ngẫu nhiên hóa dấu gradient mỗi step, có thể là "nguyên liệu ngầm" (làm mịn gradient kiểu EOT). Chưa kiểm.

**Chạy 2 bậc:**
- **Bậc 1:** 7 ô trên 100 ảnh `dev100.csv` (seed theo chỉ số dev300 để nối được), ~40 phút. Δ = OSFD − ô.
  - cận dưới CI ≥ 2 → **nguyên liệu**;
  - cận trên CI < 2 → **không phải**;
  - còn lại → **mơ hồ**.
- **Bậc 2:** chỉ ô mơ hồ chạy thêm 200 ảnh còn lại của dev300, quyết lại 1 lần trên 300 ảnh.

**Cổng 1:** danh sách nguyên liệu. Rỗng → OSFD không tách được ở độ phân giải này → user quyết dừng hay không.

## Bước 2 — Suy ra cơ chế

Trên PNG của OSFD + 7 ô (100 ảnh chung), chỉ forward:
- D_k = ‖f_k(adv) − f_k(clean)‖ / ‖f_k(clean)‖ theo stage, trên backbone R50, R101, ConvNeXt-T, Swin-T **và DINO-Swin-L**.
- Phổ tần của δ (dải thấp / trung / cao), phân bố không gian của δ.

Câu hỏi: nguyên liệu ở bước 1 thay đổi **tính chất nào của δ / distortion**, và tính chất đó có đi cùng transfer khác họ không (giữa các ô và theo từng ảnh). Kết quả = một câu cơ chế cụ thể làm mục tiêu thiết kế cho bước 3.

### Câu hỏi phụ: vì sao DINO-Swin-L thấp (~25%)?

Đã biết (diễn giải đóng băng 2026-09-23): không do kiểu detector (DINO-R50 bị OSFD 97.3%); capacity trong họ Swin chỉ giải thích ~12% (Swin-T → S); phần lớn gắn với Swin-L + pretrain IN-22k/384, không tách thêm được bằng model zoo hiện có. **Không thêm model mới.**

Trong plan này chỉ làm 2 việc, đều miễn phí hoặc rất rẻ, đều là **mô tả**, không thuộc cổng:
1. DINO-Swin-L đã nằm trong eval của mọi ô → xem có thành phần OSFD nào **riêng** ảnh hưởng tới Swin-L (vd bỏ nhiễu làm Swin-L tụt nhiều hơn các target khác).
2. D_k trên backbone Swin-L (bước 2): nhiễu có vào được feature Swin-L không, và tắt ở stage nào — so với Swin-T.

## Bước 3 — Method + sàng lọc @ dev300

- Method phải **khuếch đại một nguyên liệu** (cổng 1) theo cơ chế ở bước 2; không thêm thành phần không có bằng chứng.
- Kiểm **liều–đáp ứng** trước: 2 mức liều của nguyên liệu trên dev100; chỉ đi tiếp nếu liều cao hơn → CrossAvg₃ cao hơn, R101 không vỡ.
- B = 50 giữ nguyên. Cần thêm view forward → so thêm với OSFD cùng số view forward.
- Tối đa **3 ứng viên**, khai báo đủ cấu hình trong progress_log trước khi chạy; không thêm ứng viên cứu vãn.

Sàng lọc @ dev300, QUA nếu đồng thời: ΔCrossAvg₃ (ứng viên − OSFD) ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S; drop R101 ≥ R101(OSFD) − 3. Nhiều cái qua → chọn Δ lớn nhất (chênh < 0.5 → cái đơn giản hơn), đóng băng. Không cái nào qua → dừng method.

## Bước 4 — Xác nhận

- **GO @ n300** (1 lần, OSFD = PNG `n300_B50_eps5` trên HF): ΔCrossAvg₃ ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả 3 target khác họ VÀ ≥ 2/3 target ≥ 3; drop R101 ≥ R101(OSFD) − 3.
- **n1000** (chỉ khi GO): MI, M-DI², OSFD, method; B ∈ {50, 200}; ε ∈ {5, 8}; Controlled + Generalization Panel; thêm ASR, LPIPS. Trước bước này rà literature 2024–2026 xem có baseline mới khớp threat model không.

## Việc cần code trước bước 1

- `run_baselines.py`: tùy chọn chạy tập con `dev100` trong run dev300, giữ seed theo chỉ số dev300.
- A1: RRB co-transform GT box + mask (dùng `attack/methods/box_transforms.py`).
- A3–A5: cờ tắt từng phần trong `rrb_views`; A6, A7: tham số có sẵn.
- Unit test (RRB đầy đủ qua cờ mặc định trùng tuyệt đối OSFD hiện tại), commit trước khi chạy.
