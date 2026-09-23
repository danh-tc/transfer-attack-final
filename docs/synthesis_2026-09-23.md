# Tổng hợp sau Baseline + Generalization + Mechanism A1–A4 (2026-09-23)

Mục đích: nhìn toàn cục trước khi chọn hướng method (idea.md §12). Mọi số liệu dưới đây lấy từ file kết quả đã commit — nguồn ghi ở từng mục. Chi tiết quá trình: `docs/progress_log.md`.

Thiết lập chung: surrogate Mask R-CNN R50, ε = 5/255, B = 50, n = 300 (A3: 100 ảnh đầu), đánh giá qua ảnh uint8 cỡ gốc, paired bootstrap 95% CI. Số là relative AP drop (%) trừ khi ghi khác.

---

## 1. Kết quả chắc chắn (đạt tiêu chí khóa hoặc CI rõ)

**1.1 Gap cùng họ > khác họ tồn tại ở cả 3 method — Hypothesis Pass @300 = PASS.** (`results/runs/n300_B50_eps5/metrics.json`)

| Method | R101 | ConvNeXt-T | Swin-T | Gap R101 − avg(cross) |
|---|---|---|---|---|
| MI-FGSM | 77.3 | 54.0 | 45.1 | 27.7 |
| M-DI²-FGSM | 93.0 | 78.0 | 67.1 | 20.4 |
| OSFD | 92.4 | 80.4 | 78.7 | 12.9 [10.2, 16.7] |

Gap co lại khi attack mạnh lên (một phần do R101 gần trần).

**1.2 Backbone quan trọng hơn detector.** (`generalization_metrics.json`, `generalization_diag_dino.json`)
- Mọi target backbone ResNet-50, bất kể detector (FCOS, DETR, DINO-R50): OSFD 92.5 / 96.3 / 97.3 — bằng hoặc hơn R101.
- Cùng detector DINO, đổi R50 → Swin-L: 97.3 → 24.6 (hiệu 72.7 [69.6, 76.6]). DETR-R50 vs DINO-R50: −1.0 (không khác).

**1.3 Generalization Panel: OSFD còn chỗ trống ở target khác họ.** Headroom OSFD (drop R101 − drop target): YOLOX-S 15.0 [12.1, 19.7], DINO-Swin-L 67.8 [63.6, 72.7] → quy tắc khóa: **ĐI TIẾP** method.

**1.4 DINO-Swin-L gần miễn nhiễm (OSFD 24.6, M-DI² 26.1), nhưng KHÔNG dùng làm bằng chứng nhân quả cho họ backbone** (diễn giải đóng băng): capacity trong họ Swin (T→S) chỉ giải thích ~12% (6.3 [3.8, 9.1] điểm); phần lớn gắn với Swin-L + pretrain IN-22k/384, không tách được. Bằng chứng backbone-family sạch = Controlled Panel + nhóm cùng R50.

**1.5 Hướng "stage-aware backward regularization" bị loại** (quy tắc khóa: cần A2 hoặc A3 có bằng chứng). A3: không cặp stage nào đảo thứ hạng giữa R101 và target khác họ; không stage đơn nào hơn OSFD gốc ở target khác họ (stage4 − all: ConvNeXt −0.1, Swin-T −4.4).

## 2. Mechanism Stage — trạng thái theo tiêu chí khóa

| | Câu hỏi | Kết quả |
|---|---|---|
| A1 | Gradient R50 cùng hướng với target? | ✗ — cùng họ > khác họ rõ (sign agreement .5205 / .5097 / .5066), nhưng cosine ngược chiều per-ảnh với OSFD |
| A2 | Distortion tách ở stage nào? | ◐ — tách từ stage 2, reproduce OSFD + M-DI²; không nhất quán độ lớn theo method |
| A3 | Can thiệp 1 stage | ✗ — không đảo thứ hạng |
| A4 | Lệch phổ tần? | ✗ — Swin không lệch phổ so với R101; tương quan khớp-phổ ↔ transfer ngược chiều |
| A5 | Overfit surrogate theo iteration? | chưa chạy (sweep OSFD n=30 không thấy dấu hiệu) |

**Chưa có cơ chế nào đạt đủ tiêu chí.** Đây là kết quả âm có giá trị cho RQ2, không phải thất bại quy trình.

## 3. Manh mối nhất quán nhưng CHƯA đạt tiêu chí (chỉ dùng để sinh giả thuyết)

1. **Nhiễu "vào" được target khác họ nhưng không được khuếch đại theo độ sâu** (A2): stage 1 ConvNeXt/Swin bị lệch ≥ R101, từ stage 2 R101 khuếch đại (OSFD D: 0.22 → 1.31) còn ConvNeXt/Swin thì không. Với target Transformer, tấn công stage 3 ≥ stage 4 (A3: Swin-T 72.1 vs 73.2, DINO-Swin-L 22.3 vs 18.7), R101 thì stage 4 tốt nhất.
2. **Hướng gradient pixel gần như ngẫu nhiên giữa các model** (A1: sign agreement chỉ trên 0.5 một-hai điểm %; A3: gradient OSFD-stage vs task target ≈ 0.500) — khác biệt transfer không nằm ở hướng gradient pixel.
3. **OSFD thiên tần thấp** (A4: 39% năng lượng dải thấp vs MI 23%, M-DI² 28%) và transfer tốt nhất — chỉ 3 method, không kết luận.
4. **Gradient tập trung mạnh vào vật thể, δ trải đều** (A4): top-5% pixel ≈ 88% năng lượng gradient; năng lượng trong GT box gấp 6.9× (R50/R101), 8.5× (ConvNeXt/Swin) tỉ lệ diện tích; δ sign-step ≈ 1.0× → phần lớn budget ε rơi vào nền.
5. **Cùng kiến trúc R50 (có thể cùng ImageNet init) > cùng họ khác độ sâu**: MI-FGSM R50-targets 79–94% vs R101 77% — chưa kiểm cùng init.

## 4. Chỗ trống còn lại cho method (OSFD, B=50)

| Target | OSFD | Trần tham chiếu (R101 92.4) | Chỗ trống |
|---|---|---|---|
| ConvNeXt-T | 80.4 | 92.4 | ~12 |
| Swin-T | 78.7 | 92.4 | ~14 |
| YOLOX-S | 77.4 | 92.4 | ~15 |
| DINO-Swin-L (ca khó, mô tả) | 24.6 | — | rất lớn, nhưng không quy nhân quả cho backbone |

Phát hiện được cải thiện ở n=300 cần ≈ 4 điểm CrossAvg (nửa độ rộng CI hiệu method ~3.7); ở n=1000 ≈ 2 điểm.

## 5. Hướng method khả dĩ (giả thuyết — cần pilot, chưa có bằng chứng cơ chế)

Không hướng nào dưới đây có bằng chứng Mechanism Stage đạt tiêu chí; đều suy từ manh mối mục 3. Đề xuất chọn qua **pilot chỗ trống rẻ, quy tắc chốt trước** (vd n=100, B=50, OSFD + X vs OSFD, đo ConvNeXt-T / Swin-T / YOLOX-S, DINO-Swin-L mô tả; ĐI TIẾP nếu CrossAvg tăng với cận dưới CI ≥ 3 điểm).

| # | Hướng | Dựa trên manh mối | Rủi ro / độ mới |
|---|---|---|---|
| M1 | **Phân bổ budget theo vật thể**: trọng số không gian cho loss feature/cập nhật δ tập trung vào vùng object (ví dụ từ GT box hoặc saliency của surrogate) thay vì sign-step trải đều | 4 | Cần kiểm literature gần OSFD ("object-aware"); ràng buộc L∞ vẫn cho phép δ đủ ở nền nên lợi ích có thể nhỏ |
| M2 | **Ưu tiên tần thấp / làm mịn cập nhật** (lọc thông thấp gradient, kiểu TI-FGSM) cộng OSFD | 3 | Độ mới thấp (TI đã có); bằng chứng A4 yếu; nhưng pilot rẻ và đo đúng chỗ trống Transformer |
| M3 | **Giảm lệ thuộc khuếch đại đặc thù ResNet ở stage sâu** (vd làm yếu tín hiệu từ nhánh residual/skip kiểu SGM, hoặc đa dạng hóa surrogate ở stage sâu) | 1 | A3 đã cho thấy chọn stage không giúp → cơ chế phải khác "chọn stage"; literature classification đã có SGM |

Nếu pilot không tạo ≥ 3 điểm ở hướng nào: cân nhắc định vị paper theo contribution 1–2 (bằng chứng có kiểm soát về gap + backbone > detector + kết quả âm có hệ thống của Mechanism Stage), method là phụ hoặc bỏ.

## 6. Việc mở / việc cho phiên sau

1. **Push** các commit của 2026-09-23 (máy không lưu credential GitHub).
2. ~~Upload HF ảnh adv của A3~~ — đã xong (`runs/a3_stage_n100_B50_eps5.tar`).
3. Chọn hướng: (a) chốt quy tắc pilot + chạy pilot M1/M2/M3, hoặc (b) chuyển trọng tâm sang analysis paper.
4. Khi tiện: B=200 bảng cuối (~3 h, tmux) + A5 gộp chung.
5. Verify clean AP full val2017 cho model Generalization Panel + chẩn đoán (model_registry.md còn trống).
6. Chưa tính: ASR, APloc, CSR, LPIPS; threat model phụ surrogate-prediction-only; ε = 8.
