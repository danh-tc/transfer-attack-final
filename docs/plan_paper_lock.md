# Kế hoạch paper — BẢN KHÓA (2026-11-10)

> **Trạng thái: APPROVED FOR EXECUTION** (sau 5 vòng review, 2026-11-10). Repo lúc khóa: HEAD `f1789eb` (E81).
> Đổi bất cứ điều gì trong file này = ghi lý do vào `docs/progress_log.md` TRƯỚC khi đổi.
> Chỉ mở lại vòng review ý tưởng nếu rà prior art (bước 1) phát hiện công trình đã giải quyết trực tiếp interaction + cơ chế gradient này.
> Tài liệu lịch sử, KHÔNG phải protocol của plan này: `docs/mechanism_plan.md` (Mechanism Stage RQ2 cũ, dùng n300), `docs/plan_sota.md` (ablation OSFD).

## 0. Tóm tắt

- Hướng paper: **hiểu cơ chế trước, method sau.** Hai kết quả thực nghiệm ngang hàng, mỗi cái trong threat model riêng:
  - **Track S** (1 surrogate): **RD-CNX**.
  - **Track M** (2 surrogate, zero-query): **ENS-WP**.
- **Trụ cột khoa học (primary endpoint):** interaction surrogate × objective (I, mục 4 bước 3a), xác nhận trên nhiều target.
- **Claim cơ chế (secondary, có điều kiện):** gradient alignment — chỉ gọi "predictive" nếu vượt kiểm định 3c.
- Chỉ claim SOTA sau benchmark final, với baseline đã qua kiểm tra độ trung thực.
- Thứ tự thực hiện: prior art → strict-1 + WP → đóng băng recipe → cơ chế → baseline → final → cross-protocol → mở rộng.

## 1. Bằng chứng hiện có

Relative AP drop (%), COCO dev100, ε = 5/255, máy A4500, bootstrap 300.
**T7** = 7 target cố định `r101, fcos_r50, dino_r50, yolov3_d53, yolox_l, swin_t, dino_swin_l` (`scripts/e73_dev100.py:27`). KHÔNG chứa R50 hay ConvNeXt-T. Bộ này dùng chung cho mọi method (kể cả OSFD-R50) để có một panel cố định so sánh trực tiếp.

| Method | T7 | DINO-Swin-L |
|---|---|---|
| OSFD-R50 (B50) | 74.2 | 24.4 |
| RD-R50 (B36) | 77.6 | 34.8 |
| RD-CNX (B50) | 78.4 | 59.0 |
| ENS-WP (B36) | 82.6 | 55.3 |

**E65 (dev30, chưa có CI cho interaction):** Swin-L OSFD-R50 26.4 / RD-R50 33.8 / OSFD-CNX 26.1 / RD-CNX 62.8 → contrast ≈ +29.
Confound: OSFD và RD khác nhau cả ở augmentation (RRB vs OCT), stage feature, loss cls và WP. Riêng augmentation + stage ≈ +8.6 (FEAT-OCT-CNX 34.7 vs OSFD-CNX 26.1).

**Cơ chế** (dev30 trừ khi ghi khác):
- **E66:** trên CNX, thêm cls vào feat cho +18.2 Swin-L; feat phụ +9.8; WP +9.9 (gần như không mất gì ở CNN).
- **E67:** CKA feature sạch — ConvNeXt gần ResNet (0.66–0.70) hơn gần Swin.
- **E68:** cos(∇CNX, ∇Swin-L) gấp ~2.3× của R50, CI loại 0; cos(∇R50, ∇CNX) = 0.14.
- **E70/E71:** momentum chung 2 surrogate giữ được cả 2 họ — **bằng chứng can thiệp hỗ trợ**.
- **E64 MIX + A4:** **loại trừ giả thuyết tần số**. Không dùng làm bằng chứng ủng hộ alignment.
- **E77:** cơ cấu thất bại [cls, loc, supp] bất biến qua họ backbone.
- **E78:** ConvNeXt xoay quỹ đạo tối ưu (orth_frac 0.77).
- **E81:** ρ = 0.002 tốt hơn rõ các liều 0.005 và 0.010 đã thử (Swin-L 60.3 → 49.1 → 39.8, RES3 gần như phẳng). Kết quả gợi ý Swin-L rất nhạy với ρ. **Chưa quét dày hai phía nên chưa xác định được cực đại.**
- 8 gate single-R50 đều NO-GO: E64, E72, E74, E75, E76, E77, E78, E79.

**Điểm yếu đã biết:**
- E65/E66/E68/E78 mới ở dev30.
- Bản port OSFD chưa đối chiếu với số trong paper gốc.
- ENS-WP chưa tái lập bit-chính-xác (python-random chưa seed).
- Phần cứng gây lệch khác nhau theo method (RD lệch ~8 Swin-L giữa V100 và Ampere; ENS-WP ~0).

## 2. Claim

**2.1 Primary (trụ cột):**
> "Surrogate–objective interactions substantially shape cross-backbone adversarial transfer in object detection, beyond what clean feature similarity alone can explain."

Interaction I chỉ chứng minh một đặc trưng cố định theo backbone (như CKA) không đủ giải thích. I **chưa** chứng minh gradient alignment là thước đo tốt nhất.

**2.2 Mechanistic (có điều kiện):**
> "Objective-dependent gradient alignment provides additional explanatory evidence, and potentially predictive information, about cross-backbone transfer."

Chỉ bỏ chữ "potentially" nếu vượt kiểm định 3c. Ngay cả khi vượt, với 4 surrogate thì chỉ được nói là **robust trên các backbone đã thử**, không phải predictor tổng quát cho mọi backbone mới.

**2.3 Practical:** RD-CNX (Track S) và ENS-WP (Track M) cho transfer mạnh tới các detector chưa thấy, báo riêng theo threat model. Không bắt buộc cả hai đạt SOTA.

**2.4 Không claim:**
- "Lần đầu" cho: tương tác surrogate × objective, ensemble, flatness, task + feature loss, gap CNN–Transformer.
- "Detection-specific" (chưa có đối chứng classification) → viết "in object detection".
- "Impossibility".
- Gradient alignment là nguyên nhân duy nhất.
- E64/E70 là "causal proof".

## 3. Định nghĩa dùng chung

**3.1 Nhóm target** (taxonomy theo `docs/model_registry.md`):
- **G0** — white-box / surrogate đã dùng: không vào trung bình.
- **G1** — detector chưa thấy, CÙNG HỌ với ≥1 surrogate. Ví dụ với ENS-WP: R101, FCOS-R50, DINO-R50 là detector chưa thấy nhưng **không phải họ chưa thấy**.
- **G2** — detector chưa thấy, KHÁC HỌ với mọi surrogate.
  - CNN khác: YOLOv3-D53, YOLOX-L.
  - Transformer: Swin-T, DINO-Swin-L (báo tách riêng).
- Surrogate có Swin (Swin-T ở 3b): DINO-Swin-L thuộc G1 (cùng họ Swin), KHÔNG gọi là họ Transformer chưa thấy.
- Metric tổng: T7 + trung bình từng nhóm.

**3.2 Metric:**
- Relative AP drop và mAP drop tuyệt đối.
- AP50, ASR, APloc, CSR.
- Realized L∞, L2, LPIPS.
- B (số backward), số view forward, wall-clock trên cùng GPU.

**3.3 Outcome theo ảnh** (cho mô hình thống kê; AP không xác định ở mức ảnh). Dùng định nghĩa đã khóa trong `docs/mechanism_plan.md`:
- *suppression rate*: tỉ lệ GT mà target bắt được trên ảnh sạch (IoU ≥ 0.5, đúng lớp, score ≥ 0.3) nhưng mất trên ảnh adv.
- *retained confidence*: trung bình score_adv / score_clean.
- Phụ: phân loại thất bại theo object như E77.

Relative AP drop và I luôn tính trên **tập ảnh** bằng bootstrap, không đưa vào mô hình theo ảnh.

**3.4 Dữ liệu và lịch sử sử dụng:**

| Tập | Vai trò |
|---|---|
| dev5/10/30/100/300 | Phát triển + chọn method. Giao n1000 = 0 (đã kiểm 2026-11-10). **Toàn bộ thí nghiệm C1/C2 mới (bước 2–3) chỉ dùng dev.** |
| n300 (⊂ n1000) | Đã dùng cho RQ1 (gap baseline), Mechanism Stage cũ A1–A4, 1 lần GO C2 bỏ dở (110/300). Công bố trong data-use history. |
| n1000 | Test final, dùng 1 lần. |
| n700 = n1000 \ n300 | Sensitivity check, **không** phải replication độc lập. |

**3.5 Thống kê:**
- Paired bootstrap theo ảnh, 1000 mẫu, BOOT_SEED 2026.
- CI của hiệu số và interaction tính **trong cùng tiến trình** (cùng bộ trọng số).
- Mọi ô cần so với nhau sinh trên **cùng một máy**.

**3.6 Seed:**
- Mọi method dùng chung một quy tắc seed cố định theo chỉ số ảnh (Python + NumPy + Torch).
- Ở final: ENS-WP, RD-CNX, OSFD chạy ≥ 3 seed; method ngẫu nhiên khác chạy seed chung và ghi rõ.

## 4. Các bước và quy tắc

**Quy tắc chung (áp dụng mọi bước):**
- GO/NO-GO ghi vào progress_log + commit **TRƯỚC** khi chạy.
- Ô thắng ở dev30 phải xác nhận dev100.
- Không chọn bất cứ gì trên n300/n1000.
- Báo cáo mọi target, mọi kết quả, kể cả âm.
- Đếm số lần sửa recipe và báo trong paper.
- Các ngưỡng kiểu +4 / −3 là **quy tắc quyết định của dự án**, không phải ngưỡng thống kê phổ quát.

### Bước 1 — Rà prior art và protocol (không cần GPU)

- **FUGEA** (Neural Networks 2026, online 02/04/2026): ensemble có luôn trộn CNN + ViT không? (quyết định điểm khác biệt "2 CNN vẫn tới Transformer" của ENS-WP); setting; có code không.
- **CWA** (ICLR'24):
  - Detection có đúng là universal patch không?
  - Flatness ở input space (khác WP ở weight space)?
  - Code có chuyển sang full-image và thay surrogate được không?
- **SVul-Attack** (online 02/11/2026):
  - Số surrogate trong từng thí nghiệm → xếp Track S hay M.
  - Code, danh sách ảnh, checkpoint, metric (mAP tuyệt đối).
- **CTSTA:** ngày online-first, code, setting.
- **Literature classification:**
  - Surrogate ConvNeXt / CNN hiện đại → ViT.
  - Tương tác surrogate × objective (Transfer Attacks Revisited 2022; logit alignment 2026).
  - Gradient alignment như yếu tố dự báo transfer.
- **Benchmark 2602.16494** (`papers/2602.16494v1.pdf`): có công bố danh sách ảnh, checkpoint, metric để chạy lại không?
- **OSFD** (`papers/06244-AAAI24.DingX.pdf`): chọn 1 số tham chiếu cho fidelity check của bản port.
- **Đầu ra:** bảng prior art (đóng góp / overlap / code / setting / track) + câu chữ novelty sửa theo đó.

### Bước 2 — strict-1 + WP, rồi ĐÓNG BĂNG

**Code:** thêm `wp_in` / `wp_out` vào `run_iterative_attack` và `geo_osfd_attack`, để giữ θ' riêng cho từng surrogate qua các lần gọi. Hiện tại θ' bị caller khôi phục ngay ở `steps=1`, nên WP không có hiệu lực.

**Regression test (phải qua cả 4):**
1. Đường mặc định trùng tuyệt đối bản cũ.
2. Trọng số gốc của mỗi surrogate được khôi phục sau khi sinh xong.
3. θ' thực sự làm đổi gradient ở lượt kế tiếp của CHÍNH surrogate đó.
4. Không áp nhầm θ' của R50 lên ConvNeXt; số lần backward không đổi.

**Seed:** Python + NumPy + Torch theo từng khối (sửa luôn e80/e81 nếu tái dùng).

**Chạy:**
- dev30, panel gọn (`dino_swin_l` + `r101` / `fcos_r50` / `dino_r50`), boot 100, cùng máy.
- Sinh lại ENS-WP-B36 trên chính máy đó làm mốc.

**Quy tắc:**
- **GO** nếu ΔSwin-L ≥ +4 VÀ cận dưới CI paired > 0 VÀ RES3 ≥ mốc − 3 → xác nhận dev100 (panel 9 model).
- **NO-GO** → giữ ENS-WP hiện tại.

**ĐÓNG BĂNG** sau bước này, ghi đủ config vào progress_log:
- ENS-WP (hoặc bản strict-1 nếu GO ở dev100).
- RD-CNX B50 (cùng B với OSFD B50).

### Bước 3 — Kiểm chứng cơ chế (dev100, cùng máy)

**3a. Factorial giai đoạn 1 — PRIMARY**

- **Ô:** {R50, ConvNeXt-T} × {FEAT-only, CLS-only, FEAT+CLS}, tất cả là Mask R-CNN.
- **Giữ cố định:** OCT 1 view, feature C3–C5, cùng B (đề xuất 36 — khóa trước khi chạy), không WP, ε = 5, seed theo chỉ số ảnh. CLS-only = loss phân loại RoI của Mask R-CNN.
- **Primary contrast:** I = (CNX_CLS − CNX_FEAT) − (R50_CLS − R50_FEAT), đo trên **trung bình {Swin-T, DINO-Swin-L}**, CI paired cho chính I.
  - Ô một số hạng không phụ thuộc thang loss: bước MI chuẩn hóa mean|g| rồi lấy sign, nên không có λ.
- **Báo cáo:** I + CI cho MỌI target trong panel, bất kể kết quả.
- **Secondary:** ô FEAT+CLS (λ theo quy tắc `rd_calib`, không tune); sensitivity λ × 0.5 và × 2 (4 ô).
- **Ô nối với E65:** OSFD-R50, OSFD-CNX (recipe gốc), RD-R50, RD-CNX → tách phần do augmentation/stage và phần do objective.
- **Quy tắc:**
  - Cận dưới CI(I) > 0 → claim 2.1 giữ.
  - CI chứa 0 → **đánh giá lại toàn bộ central claim** (hạ xuống mức mô tả hiện tượng, viết lại định vị theo số thật). Không mặc định rằng C2 đủ cứu hướng mechanism-first.

**3b. Factorial giai đoạn 2 — BẮT BUỘC (vì 3c cần trục surrogate)**

- **Thêm surrogate:** **R101 và Swin-T** (cùng Mask R-CNN, cùng định nghĩa CLS-only). Ô FEAT-only và CLS-only.
- **DINO-R50 KHÔNG dùng ở đây** (không có RoI cls head → đổi cả kiến trúc detector lẫn định nghĩa loss). Chỉ dùng trong thí nghiệm tổng quát hóa riêng, nếu có objective tương thích.
- Mỗi surrogate loại chính checkpoint của nó khỏi target; với surrogate Swin-T, DINO-Swin-L thuộc G1 (mục 3.1).
- Không trình bày như kiểm định xác nhận độc lập của giai đoạn 1.

**3c. Gradient alignment vs CKA — điều kiện để dùng chữ "predictive"**

*Định nghĩa CKA (khóa):*
- Linear CKA giữa feature backbone của surrogate và target, trên ảnh SẠCH, cùng pixel đầu vào (như E67).
- Stage 3 và 4; báo trung bình hai stage.
- Mẫu = vị trí không gian. Nếu hai lưới khác cỡ, nội suy bilinear feature target về lưới surrogate.
- Spatial CKA tính theo từng ảnh, rồi tổng hợp thành **CKA cấp cặp** (surrogate, target) = trung bình qua ảnh. Biến thiên theo ảnh báo riêng, không gộp làm một loại tín hiệu.

*Định nghĩa alignment (khóa):*
- cos(∇ₓ L_obj qua surrogate, ∇ₓ L_cls qua target), theo ảnh gốc, trên ảnh SẠCH, như E68.
- Tổng hợp thành **alignment cấp (surrogate, objective, target)** = trung bình qua ảnh.

*Mô hình:*
- outcome theo ảnh (3.3) ~ alignment + CKA + (target) + (ảnh), kèm kiểm soát độ lớn loss của surrogate.
- Biến thiên giữa surrogate và target là nguồn bất định chính. CI bằng bootstrap theo ảnh, báo kèm uncertainty theo surrogate và theo target, không chỉ một p-value gộp.
- 100 ảnh **không** phải 100 kiến trúc độc lập (pseudoreplication).

*LOSO (khóa trước):*
- 4 fold, mỗi fold bỏ 1 surrogate.
- Dự báo outcome trung bình của từng ô (objective, target) của surrogate bị bỏ ra.
- So **mô hình CKA + alignment** với **baseline chỉ có CKA** (cùng biến kiểm soát).
- Metric: Spearman giữa dự báo và quan sát qua các ô, kèm MAE.

*PASS nếu:*
- Hệ số alignment > 0 với CI loại 0 sau khi kiểm soát CKA, VÀ
- Mô hình có alignment tốt hơn CKA-only ở **≥ 3/4 fold** theo Spearman, VÀ
- Trung bình ΔMAE < 0.

*FAIL hoặc không ổn định* → claim 2.2 chỉ là "explanatory evidence".
Ngay cả khi PASS: chỉ nói robust trên 4 backbone đã thử.

*Tiền lệ cần nêu:* A1 (cosine theo ảnh ngược chiều với transfer), E75 (cosine tại ảnh sạch là proxy yếu).

**3d. Can thiệp oracle — bỏ thành phần aligned (chỉ chẩn đoán)**

- δ lấy từ RD-CNX. g_t = ∇ₓ L_cls của DINO-Swin-L, **lấy tại ảnh SẠCH** (khớp E68).
- Treatment: δ⊥ = δ − ⟨δ, g_t⟩ / ‖g_t‖² · g_t.
- Ba nhánh, cùng một quy tắc chuẩn hóa, cuối cùng kẹp L∞ ≤ ε:
  - *treatment*: bỏ thành phần dọc g_t.
  - *random*: bỏ thành phần dọc một hướng ngẫu nhiên, với CÙNG độ lớn phần bị loại.
  - *sham*: δ đi qua cùng pipeline, phần bị loại = 0.
- Báo: realized L2, số pixel bão hòa, độ lớn phần bị loại — tất cả **sau** clipping.
- Đọc: hiệu ứng = (treatment − sham) so với (random − sham).
- Null (cos ~0.04 nên phần bị loại rất nhỏ) → "chưa cung cấp bằng chứng", KHÔNG phải "alignment vô vai trò".
- Dùng gradient của target → **chỉ phân tích cơ chế**, không bao giờ dùng cho bảng attack.

**3e.** Lặp lại E68 và E78 ở dev100 có CI (cho hình chính).

3c và 3d là kiểm định nâng claim cơ chế, **không** phải điều kiện để paper đi tiếp. Trụ cột là 3a.

### Bước 4 — Baseline ngoài

**Ba lớp:**
- **A — Direct comparable:** cùng kiểu perturbation (image-specific, full-image), cùng quyền truy cập surrogate, cùng eval. Code chính thức hoặc bản đã qua kiểm tra độ trung thực.
- **B — Controlled adaptation** (gắn nhãn "adapted" / "our impl."):
  - CWA chuyển sang full-image, cùng R50 + CNX.
  - **Ens-OSFD và Ens-RD** (trung bình loss) — **BẮT BUỘC**, để tách "có 2 model" khỏi momentum chung + luân phiên.
- **C — Cross-protocol / related work:** CWA bản patch gốc; số trích từ paper khác protocol.

**Checklist độ trung thực** (theo từng method, không dùng ngưỡng chung): objective, ràng buộc perturbation, preprocessing, checkpoint, clean AP, định nghĩa metric, tái lập 1 số tham chiếu trong setting gốc.
- Bản tự cài không qua → gắn **"unverified"**, KHÔNG dùng làm đối chứng chính và KHÔNG gọi là bằng chứng cross-protocol (lớp C chỉ dành cho số chính thức của tác giả). Tiền lệ: AugTrans.
- OSFD: fidelity check cho bản port v2 → v3.

**Siêu tham số:** theo khuyến nghị của tác giả. Nếu protocol lệch, cho một ngân sách tune nhỏ trên dev và ghi rõ.

**Ablation của mình** (bảng riêng): ENS-WP bỏ WP / bỏ momentum chung (warm-start E69) / từng nhánh đơn.

### Bước 5 — Benchmark final (1 máy duy nhất, ghi rõ GPU)

- n1000 + sensitivity n700; ε ∈ {5, 8}/255; B chính 50 (ENS-WP 36).
- **Compute:** single B72 vs ENS B36 (chẩn đoán) + wall-clock trên cùng GPU (backward ConvNeXt ≈ 1.4× R50, E62/E63).
- **Panel:** Controlled (R101, ConvNeXt-T, Swin-T) + A/B/C (FCOS-R50, DINO-R50, YOLOv3, YOLOX-L, Swin-T, DINO-Swin-L). Báo theo G1/G2 và T7.
- **Seed:** theo 3.6.

### Bước 6 — Cross-protocol

Chạy RD-CNX / ENS-WP trên đúng protocol của benchmark 2602.16494 và/hoặc SVul; so với số HỌ tự báo (ví dụ OSFD 40.2 trên DINO-Swin-L, VOC2007). Chỉ làm khi họ công bố đủ protocol (kiểm ở bước 1).

### Bước 7 — Tính tổng quát (nên có)

- VOC; backbone held-out Swin-S (chưa từng eval); DINO-R50 làm surrogate nếu có objective tương thích; thêm họ khác nếu còn tài nguyên.
- Mục tiêu: kiểm tra CƠ CHẾ còn đúng, không chỉ thêm số.

## 5. Bảng / hình trong paper

| | Nội dung |
|---|---|
| Bảng 1 | Lớp A, chia Track S / M (số surrogate, query, B, wall-clock) |
| Bảng 2 | Lớp B: adapted CWA, Ens-OSFD, Ens-RD, ENS-WP (cùng surrogate) |
| Bảng 3 | Factorial + interaction I có CI, mọi target |
| Bảng 4 | Alignment vs CKA (3c) + can thiệp (3d) |
| Bảng 5 | Ablation ENS-WP / RD-CNX |
| Bảng 6 | Cross-protocol |
| Hình | E67 vs E68, E78, E77, E81 |
| Phụ lục | Lịch sử dữ liệu (3.4), số lần sửa recipe, phần cứng, taxonomy họ |

## 6. Rủi ro và phương án

- **R1** — Classification đã có "ConvNeXt bắc cầu ViT" → điểm tựa chuyển sang bằng chứng trong detection: E67 + E68, E77, I trên nhiều detector.
- **R2** — CI(I) chứa 0 → đánh giá lại toàn bộ central claim (3a), không mặc định C2 cứu được.
- **R3** — 3c FAIL → "explanatory" thay vì "predictive".
- **R4** — Baseline không port được / "unverified" → dựa lớp B + số chính thức ở lớp C, ghi rõ.
- **R5** — ENS-WP thua adapted CWA → Track M claim "cạnh tranh"; nhấn Track S và cơ chế.
- **R6** — Phần cứng → final trên 1 máy; mốc luôn sinh lại trên cùng máy.

## 7. Cần quyết (người duyệt / thầy)

- **Q1:** Duyệt pivot threat model 2-surrogate cho Track M (idea.md §3).
- **Q2:** B final: giữ 50/36 hay chuẩn hóa theo wall-clock?
- **Q3:** Venue mục tiêu (quyết định bước 7 có bắt buộc không).
