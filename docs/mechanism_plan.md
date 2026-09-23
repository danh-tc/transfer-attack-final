# Mechanism Stage — Plan (idea.md §11)

**Trạng thái: APPROVED (2026-09-23)** sau 3 chỉnh của user (thứ tự chạy, vai trò A3, ceiling; wording A2). Từ đây, mọi định nghĩa chỉ số + quy tắc kết luận dưới đây coi như khóa (cùng tinh thần protocol_lock.md): không đổi chỉ số/ngưỡng sau khi thấy kết quả; nếu buộc phải đổi, ghi lý do vào progress_log trước.

## Bối cảnh

Hypothesis Pass @300 = PASS (progress_log 2026-09-23). Từ `results/runs/n300_B50_eps5/metrics.json`:
- Gap cùng họ > khác họ ở cả 3 method, CI > 0.
- Gap co lại khi attack mạnh hơn: MI 27.7 → M-DI² 20.4 → OSFD 12.9 (R101 gần trần ~92–93% với M-DI²/OSFD).
- MI/M-DI²: Swin khó hơn ConvNeXt rõ (~9–11 điểm). OSFD: ConvNeXt ≈ Swin (80.4 vs 78.7).

Câu hỏi của Mechanism Stage: **gap xuất hiện ở đâu (stage nào của mạng) và vì sao** — đủ cụ thể để quyết định giữ/bỏ hướng method "stage-aware backward regularization" (idea.md §12).

## Sự kiện kỹ thuật đã kiểm (2026-09-23, GPU)

- Cả 4 backbone ra 4 stage cùng lưới không gian (stride 4/8/16/32; vd ảnh 1196×800 → 304×200, 152×100, 76×50, 38×25). Kênh: ResNet 256/512/1024/2048, ConvNeXt-T/Swin-T 96/192/384/768 → so sánh stage-wise dùng chỉ số không cần cùng số kênh (CKA).
- Cả 4 model chung preprocessing + chung loss task (Mask R-CNN head giống nhau) → gradient đầu vào so sánh được trực tiếp, cùng không gian ảnh gốc (qua `resize_to_model`).
- Loss task có ngẫu nhiên (RPN/RCNN sampler) nhưng nhỏ: cos(grad seed0, seed1) = 0.991 / 0.997 / 0.999 / 0.997 (R50/R101/ConvNeXt/Swin, ảnh img_id=776) → 1 seed đủ; vẫn báo cáo "trần nhiễu" R50-vs-R50 khác seed làm mốc trên.
- Gradient của target CHỈ dùng để phân tích, không dùng sinh attack → không vi phạm threat model (idea.md §3).

## Chỉ số transfer theo từng ảnh (dùng chung cho mọi phân tích tương quan)

AP không ổn định ở mức 1 ảnh → per (ảnh, method, target), tính từ `dets.json` của run B=50 (không cần predict lại), trên các GT được target detect trên ảnh sạch (IoU ≥ 0.5, đúng lớp, score ≥ 0.3):
- **suppression rate**: tỉ lệ GT đó KHÔNG còn được detect trên ảnh adv (chỉ số chính, dễ đọc, nhưng bão hòa ở 1).
- **retained confidence**: trung bình score_adv / score_clean của GT-matched detection (0 nếu mất) — liên tục, ít bão hòa hơn; bắt buộc báo cáo song song vì R101 đã ~92–93% (hiệu ứng trần).
- Mức tập hợp: kèm **adv AP / clean AP** (từ metrics.json) — ít bị trần hơn relative drop khi so gap.

## Các phân tích

**Thứ tự chạy (chốt): A2 → A1 → A3 → A4 → A5.** A2 lên đầu chỉ vì phụ thuộc ảnh adv PNG sắp mất khi trả máy; sau đó theo chi phí. Tập ảnh: n300 cho A1/A2/A4; **100 ảnh đầu của n300** cho A3/A5 (tốn hơn). Mọi so sánh "cùng họ vs khác họ" dùng paired bootstrap trên ảnh (1000 mẫu, như harness baseline).

### A1 — Input-gradient alignment

- **Đo:** trên ảnh sạch, g_m = ∇ₓ L_task(m, x) cho 4 model (x = ảnh gốc). Chỉ số: cosine(g_R50, g_t) và **tỉ lệ trùng dấu** sign(g_R50) = sign(g_t) (attack dùng sign-step nên chỉ số này sát cơ chế hơn cosine). Mốc trên: R50 vs R50 khác seed.
- **Mở rộng:** đo lại tại vài điểm trên quỹ đạo attack (x_adv sau 10/25/50 step, M-DI² và OSFD) — alignment có giảm dần khi perturbation "chui sâu" vào surrogate không.
- **Ủng hộ giả thuyết nếu:** align(R50,R101) > align(R50,ConvNeXt) và > align(R50,Swin), CI hiệu số > 0; VÀ Spearman(alignment per-ảnh, suppression per-ảnh) có CI > 0 cho ít nhất 1 target khác họ.
- **Chi phí:** ~300 × 4 backward, vài phút.

### A2 — Forward feature similarity + lan truyền distortion

- **Đo (a) — tương đồng biểu diễn:** linear CKA giữa feature stage k của R50 và stage k của target (mẫu = vị trí không gian trong ảnh, trung bình qua ảnh), k = 1..4, trên ảnh sạch.
- **Đo (b) — distortion lan tới đâu:** với ảnh adv (PNG của run B=50), relative feature distortion D_k(m) = ‖f_k(x_adv) − f_k(x)‖ / ‖f_k(x)‖ tại từng stage k của từng model m. So đường D_k theo k giữa R101 / ConvNeXt / Swin: gap xuất hiện ở stage nào (đường tách nhau từ đâu).
- **Ủng hộ giả thuyết "divergence nảy sinh ở stage cụ thể" nếu:** có stage k* mà từ đó profile D_k của R101 TÁCH khỏi profile của target khác họ (hiệu số có CI loại trừ 0), trong khi trước k* không tách hoặc tách nhỏ hơn rõ; chiều + độ lớn tách nhau NHẤT QUÁN với thứ tự transfer đã quan sát (theo target: R101 vs ConvNeXt vs Swin; theo method: gap lớn hơn ↔ tách lớn hơn); và k* reproduce trên cả OSFD và M-DI². **Không** khóa trước dấu tuyệt đối kiểu D_k(R101) > D_k(cross) ("distortion lớn hơn = transfer tốt hơn"), vì scale/hình học feature của ResNet, ConvNeXt, Swin khác nhau kể cả khi dùng distortion tương đối. Kèm tương quan per-ảnh D_k với retained confidence trong từng target.
- A2 là bằng chứng **quan sát** (distortion tách ở đâu) — không đủ một mình để kết luận nhân quả, xem A3.
- **Chi phí:** chỉ forward, vài phút. **Cần ảnh adv B=50 hiện đang nằm trên máy** (`artifacts/runs/n300_B50_eps5/adv/`, chưa lưu ra ngoài) — nên chạy A2 trước khi trả máy; nếu mất phải sinh lại (~50 phút, không bit-identical với metrics.json đã commit do ROIAlign backward không deterministic).

### A3 — Stage-wise backward sensitivity (ablation theo stage) — thí nghiệm quyết định cho stage-aware

- **Đo:** attack OSFD-loss chỉ trên 1 stage k của R50 (k = 1..4; giữ MI + RRB, B=50, ε=5, cùng seed), đánh giá transfer lên 3 target. Kèm: alignment (A1) giữa ∇ₓ(OSFD-loss stage k) và g_t.
- **Ủng hộ "stage-aware" nếu:** thứ hạng stage theo cross-family transfer KHÁC thứ hạng theo same-family transfer (vd stage nông transfer sang Swin tốt hơn stage sâu, trong khi R101 ngược lại), với CI hiệu số > 0.
- A3 là **can thiệp** (chỉ tấn công stage k → quan sát transfer) nên là bằng chứng mạnh nhất cho/chống stage-aware; nặng ký hơn A2 khi hai bên mâu thuẫn.
- **Chi phí:** 4 stage × 100 ảnh × ~3 s ≈ 20 phút + eval.

### A4 — Gradient / perturbation concentration

- **Đo:** với δ cuối của từng method và với g_m của từng model: (a) phân bố năng lượng theo tần số (FFT, tỉ lệ năng lượng ở dải thấp/trung/cao, 3 dải cố định theo bán kính), (b) tập trung không gian (tỉ lệ năng lượng trong top-5% pixel; tỉ lệ năng lượng trong GT box so với diện tích box).
- **Ủng hộ "lệch phổ tần" nếu:** phổ của g_Swin/g_ConvNeXt khác g_R50 rõ hơn g_R101 (khoảng cách phổ, CI > 0) VÀ per-ảnh, δ có tỉ lệ năng lượng ở dải mà target nhạy nhiều hơn thì transfer tốt hơn (Spearman CI > 0).
- **Chi phí:** vài phút (dùng lại gradient của A1 + δ từ PNG).

### A5 — Iterative stability

- **Đo:** chạy 1 lần B=200 cho MI, M-DI², OSFD trên 100 ảnh, chụp δ ở B ∈ {10, 20, 50, 100, 200} (như `budget_sweep.py`), đo transfer từng mốc + cosine giữa hướng update liên tiếp.
- **Ủng hộ "overfit surrogate" nếu:** cross-family transfer bão hòa/giảm SỚM hơn same-family (mốc B đạt 95% giá trị cuối nhỏ hơn rõ), ở ≥ 2 method. Sweep OSFD n=30 trước đó KHÔNG thấy giảm → kỳ vọng tiên nghiệm: yếu.
- **Chi phí:** 3 method × 100 × ~12 s ≈ 1 h → chạy tmux, có thể gộp với việc chạy B=200 bảng cuối.

## Quy tắc kết luận (khóa khi duyệt)

- Một cơ chế được coi là **có bằng chứng** khi đạt đủ điều kiện "ủng hộ" của nó ở trên, trên **cả OSFD và M-DI²-FGSM** (giống tiêu chí §10.4).
- **Giữ hướng "stage-aware backward regularization"** (idea.md §12) chỉ khi A2(b) hoặc A3 có bằng chứng divergence phụ thuộc stage. A3 (can thiệp) được ưu tiên: A2 yếu nhưng A3 rõ → vẫn giữ; A2 rõ nhưng A3 không → ghi nhận là quan sát, KHÔNG đủ giữ. Nếu chỉ A1/A4 có bằng chứng (lệch gradient/phổ không gắn với stage) → ghi nhận, đề xuất hướng method khác phù hợp với cơ chế đó thay vì ép vào stage-aware.
- Không cơ chế nào đạt → vẫn báo cáo (kết quả âm có giá trị cho RQ2), không tự thêm phân tích mới để "tìm bằng được"; mọi phân tích bổ sung phải ghi lý do vào progress_log trước khi chạy.
- Không tune gì của attack/method dựa trên kết quả Mechanism Stage ở n300 ngoài quyết định giữ/bỏ hướng method.

## Còn mở (không chặn Mechanism Stage)

- **Hiệu ứng trần (chốt 2026-09-23):** CHƯA chạy ε=3. A1/A2/A3 báo cáo song song chỉ số ít bão hòa (retained confidence, adv AP/clean AP, distortion/alignment liên tục). **Điều kiện kích hoạt:** nếu kết luận cơ chế ĐỔI giữa suppression rate và chỉ số liên tục → chạy sensitivity check ε=3/255, B=50 (~50 phút).
- **B=200 bảng cuối** (protocol_lock.md): có thể gộp với A5.
- **Lưu ảnh adv ra ngoài** (HF dataset private): user tạm hoãn (2026-09-23).
