# transfer-attack-final

Dự án nghiên cứu: transferable evasion attack cho object detection, single-surrogate, xuyên backbone-family (CNN→CNN khác họ, CNN→Transformer). Plan đầy đủ đã khóa: [idea.md](idea.md).

## Bootstrap khi bắt đầu session mới (đọc theo đúng thứ tự)

Máy chạy thí nghiệm là **GPU thuê (RTX 3090 hoặc RTX 4000 Ada), luôn setup lại từ đầu** — không có gì tồn tại ngoài repo này sau khi trả máy. Memory riêng của Claude (nếu phiên này chạy trên máy khác với máy đã lưu memory) **không tồn tại** trên máy GPU mới. Mọi context bắt buộc phải nằm trong repo, không phải trong memory.

1. [idea.md](idea.md) — research plan đã khóa: RQ, threat model, dataset, model panel, baseline, metrics, Go/No-Go. Chỉ sửa khi có pivot phương pháp luận có chủ đích, không sửa tùy tiện.
2. [docs/protocol_lock.md](docs/protocol_lock.md) — phần "exact config" mà idea.md yêu cầu khóa: config file mmdet chính xác, checkpoint, định nghĩa B. Đây là nguồn sự thật cho mọi con số hyperparameter.
3. [docs/progress_log.md](docs/progress_log.md) — đọc ít nhất entry mới nhất: lần trước dừng ở đâu, quyết định gì, chạy dở cái gì.
4. [docs/model_registry.md](docs/model_registry.md) — nếu việc sắp làm liên quan model/backbone, tin vào đây thay vì đoán từ tên model hoặc suy diễn lại.
5. [docs/environment_setup.md](docs/environment_setup.md) — nếu cần dựng lại môi trường hoặc troubleshoot cài đặt.

## Giai đoạn hiện tại

**⚠️ CẬP NHẬT 2026-11-13 10:20 — E87 XONG: PRIMARY HỖ TRỢ** Δ_CNX(α0.5−α1) T5C +4.8 [3.0, 6.3]; I_stage +8.1 [5.8, 10.2]; R50 control ≈ 0; chẩn đoán: pha loãng do độ lớn (năng lượng C2 CNX 21%), KHÔNG phải sign conflict (D−D_null < 0). Việc tiếp: baseline fidelity → final. Xem entry E87 KẾT QUẢ.
**(trước đó) E87 ĐÃ DUYỆT + KHÓA** (can thiệp trọng số C2: CNX α∈{1,0.5,0} + R50 α∈{1,0} control, devB100 mới — protocol amendment). PRIMARY Δ_CNX = α0.5 − α1 trên T5C, cận dưới CI > 0. Sau E87 (kể cả âm) **KHÔNG mở vòng tối ưu C2** → baseline fidelity (OSFD chạy cả θ10/ρ1.0 lẫn θ7/ρ0.8) + final benchmark; recipe vẫn đóng băng. Xem 2 entry 2026-11-13 trong progress_log.
**⚠️ CẬP NHẬT 2026-11-12 23:00 GMT+7 — PHIÊN KẾT THÚC, máy đã trả.** E86 pilot: alignment ảnh sạch KHÔNG theo hiệu ứng stage; hậu kỳ: độ lớn gradient C2 xếp đúng thứ tự 4/4 surrogate (CNX ‖g₂‖/‖g₃₄₅‖ ≈ 0.52, ~21% năng lượng; ResNet ≈ 0). **Việc tiếp = E87 can thiệp C2 (α ∈ {1, 0.5, 0} trên CNX)** — CHƯA khóa; cần user quyết devB100 (sửa protocol) và các câu hỏi mở. Đọc entry "KẾT THÚC PHIÊN 2026-11-12 23:00" trong progress_log TRƯỚC khi làm gì.
**⚠️ CẬP NHẬT 2026-11-12 22:31 GMT+7 — E85 (3b, mô tả) XONG @ dev100:** hiệu ứng stage (C35−C25) theo surrogate trên T5C: R50 +0.1, R101 +0.0, SWT +2.8 [1.9, 3.7], **CNX +9.8 [7.6, 11.6]** → đặc thù ConvNeXt, không phải CNN vs Transformer; aug ≈ 0 mọi surrogate. Chờ user duyệt 3c. Chi tiết: entry E85 kết quả trong progress_log.
**⚠️ CẬP NHẬT 2026-11-12 18:26 GMT+7 — E84 XÁC NHẬN:** I_recipe (sur × [OCT-C35 − RRB-C25]) trên TB(Swin-T, DINO-Swin-L) = **+6.9 [+4.2, +9.2]** @ x200 (dev300\dev100) → claim 2.1 viết lại thành tương tác surrogate × công thức feature; phân rã (secondary): gần như toàn bộ ở **stage** (I_stage +7.5 [5.8, 8.7]; I_aug ≈ 0). Chờ user duyệt 3b với trục mới. Chi tiết: entry E84 kết quả trong progress_log.
**⚠️ CẬP NHẬT 2026-11-12 — E83 (bước 3a) XONG: PRIMARY THẤT BẠI.** I = (CNX_CLS − CNX_FEAT) − (R50_CLS − R50_FEAT) trên TB(Swin-T, DINO-Swin-L) = **−0.9 [−6.9, +1.8]** @ dev100 → claim 2.1 KHÔNG giữ → theo plan 3a/R2 phải **đánh giá lại central claim, CHƯA chạy 3b**, chờ user quyết. Hậu kỳ (chưa xác nhận): tương tác E65 có thật (I_E65 +10.3 [5.6, 13.9]) nhưng nằm ở surrogate × công thức feature/augmentation (OCT+C3–C5 vs RRB+C2–C5), không ở loss cls. Chi tiết: entry E83 trong progress_log.

**Trạng thái (cập nhật 2026-11-10 tối): BƯỚC 1 + 2 XONG (E82 NO-GO, recipe đóng băng) → tiếp bước 3a. PLAN PAPER ĐÃ KHÓA — đọc [docs/plan_paper_lock.md](docs/plan_paper_lock.md) TRƯỚC mọi thứ khác (thay mục "VIỆC MAI" bên dưới).** Primary endpoint = interaction surrogate × objective (factorial dev100); gradient alignment là claim có điều kiện; RD-CNX B50 (Track S) và ENS-WP (Track M) là 2 kết quả ngang hàng. Việc tiếp: bước 1 rà prior art (không cần GPU) → bước 2 strict-1 + WP rồi đóng băng.

**(Lịch sử) QUYẾT ĐỊNH 2026-10-31: DỪNG tìm method single-surrogate trên R50.** Cơ sở: **8 gate single-R50 NO-GO** (E64, E72, E74–E79) + E67/E68. ⚠️ Định vị "impossibility" và tiêu đề "Orthogonal by Architecture…" đã **BỎ** ngày 2026-11-10: 8 gate chỉ là giới hạn thực nghiệm của R50, không phải bất khả thi (RD-CNX 1 surrogate đạt Swin-L 59.0).

**METHOD HIỆN TẠI (2 mức giả định, dev100 trên A4500, E73/E74):**
| | Method | T7 | DINO-Swin-L | Ghi chú |
|---|---|---|---|---|
| **Ensemble** (2 surrogate) | **ENS-WP** | **82.6** | 55.3 | R50 ⊕ ConvNeXt-T, momentum chung, luân phiên khối 2 bước, WP ρ0.002, B36. Hơn OSFD +8.5 T7 / +30.9 Swin-L |
| **1 surrogate** | **RD-CNX B50** | 78.4 | **59.0** | surrogate ConvNeXt-T, B50 |
| baseline | OSFD-R50 B50 | 74.2 | 24.4 | |
| baseline | RD-R50 B36 | 77.6 | 34.8 | |

**THREAT MODEL — tin tốt (đọc `papers/2602.16494v1.pdf`, verify 2026-11-09):** benchmark nguồn taxonomy CÓ 2 method nhiều surrogate — **CAA** (Faster R-CNN + YOLOv3, đúng 2 surrogate, transfer-based black-box) và **EBAD** (ensemble + QUERY victim → grey-box). → giả định 2-surrogate của ENS-WP **có tiền lệ** và **nhẹ hơn EBAD** (ENS-WP zero query). Pivot idea.md §3 vẫn cần duyệt chính thức nhưng KHÔNG nặng như progress_log trước nay giả định. Số họ trên DINO-Swin-L (VOC2007, relative drop): CAA 3.8, EBAD 4.5, OSFD **15.4 với surrogate FRC-R50** (40.2 là với surrogate Mask R-CNN Swin-T, đã có Swin — sửa 2026-11-10, xem `docs/prior_art_2026-11.md`). Câu kết của họ: *"no attacks ... capable of compromising modern detectors in a realistic black-box scenario"* → ENS-WP trả lời bài toán mở đó.

**CƠ CHẾ (lõi novelty, không nằm ở optimizer):**
- **E67:** ConvNeXt về FEATURE sạch giống ResNet (CKA sâu 0.66–0.70), KHÔNG giống Swin → bác câu chuyện hiển nhiên "ConvNeXt ≈ Swin nên bắc cầu".
- **E68:** NHƯNG ∇ₓ loss cls qua ConvNeXt gần hướng Swin-L **~2.3×** R50; ∇R50 ⊥ ∇ConvNeXt (cos 0.14) → "transferable direction" là tính chất **ĐƯỜNG GRADIENT**, không phải tương đồng biểu diễn.
- **E70/E71:** momentum-chung cộng 2 hướng gần-trực-giao vào CÙNG vector → phủ cả hai họ (bằng chứng nhân quả cho E68).
- **E78:** block ConvNeXt XOAY quỹ đạo (orth_frac 0.77, mag_ratio ≈1.0, ổn định early→late) — lượng hoá động-học.
- **E77:** composition thất bại [cls, loc, supp] ≈ [0.05–0.12, 0.33–0.45, 0.45–0.66] **BẤT BIẾN** qua mọi attack × target × họ → gap thuần là ĐỘ LỚN của cùng một kiểu hỏng.
- **E81 (mới):** ρ0.002 tốt hơn rõ các liều 0.005 / 0.010 đã thử, chỉ tác động hướng Transformer; họ ResNet gần như trơ với liều. ρ: 0 → 0.002 → 0.005 → 0.010 cho Swin-L ~50 → 60.3 → 49.1 → 39.8 (RES3 bẹt: ~91 → 91.0 → 90.7 → 88.0). Gợi ý Swin-L rất nhạy với ρ; chưa quét dày hai phía nên CHƯA xác định được cực đại.

**E82 strict-1 + WP (2026-11-10, dev30 V100-PCIE): NO-GO** — ΔSwin-L −0.6 CI [−2.1, 3.7], ΔRES3 −0.4. Lever Swin-L cuối đã đóng → **ĐÓNG BĂNG ENS-WP (khối 2) + RD-CNX B50**, config đầy đủ trong entry E82 của progress_log. Việc tiếp: **bước 3a factorial** (`docs/plan_paper_lock.md`). Máy V100-PCIE phiên này nghẽn CPU (~10 s/ảnh ô B36, ~2× V100-SXM2).

**⚠️ LỆCH PHẦN CỨNG — PHẢI KIỂM PER-METHOD (sửa 2026-11-09):** lệch ~8 điểm Swin-L giữa V100 và Ampere (E59/E60/E61) là **CỦA RD (1 surrogate R50), KHÔNG tổng quát**. E80 đo trực tiếp: **ENS-WP B36 Swin-L V100 60.3 vs A4500 60.3 = +0.0**; lệch nằm ở họ ResNet và chiều ÂM (r101 −1.8, fcos −2.5, dino_r50 −3.9). → **không suy lệch máy từ method này sang method khác; sinh cả ô mốc trên CÙNG máy khi so.** Bảng paper vẫn sinh toàn bộ trên một máy.

**QUY TRÌNH đã rút ra (áp dụng từ nay):**
- **CI hiệu số PAIRED phải tính TRONG cùng tiến trình** (`summarize` chỉ lưu point/ci95/p_le_0 → hậu kỳ KHÔNG tính lại được; đã vướng khi so RD-CNX-B50 vs ENS-WP, +3.7 không có CI nào đỡ). Xem `scripts/e8{0,1}_*.py` làm mẫu.
- **Panel gọn + boot 100 cho sàng lọc rẻ hơn ~10× bảng đầy đủ:** 4 model (dino_swin_l + RES3) × dev30 ≈ 3.5 ph eval, sinh ~2.5 ph/ô → **1 ô ≈ 6 ph**. Dùng mốc E74 (9 model, boot 300) để ước sẽ cao gấp 5 lần.

**VIỆC MAI (ưu tiên theo thứ tự):**
1. **Verify prior art** (chưa làm, user cắt tìm web): **SVul-Attack** (feature + task loss + frequency, 22 detectors — đụng lõi RD), **CWA ICLR'24** (ensemble + flatness — đụng ENS-WP), CTSTA, FUGEA. Phải xong trước khi chốt câu chữ novelty.
2. **strict-1 + WP** (lever Swin-L cuối) — sửa core + unit test đường mặc định không đổi, rồi dev30 panel gọn.
3. **Bảng chính:** n1000 full panel ε∈{5,8} CÙNG MÁY; **OSFD-ensemble baseline** (phản biện "chỉ vì 2 model"); compute-matched single B72 vs ENS B36; CI dev100 cho E68 + E78 (hiện dev30).
4. NÊN (cửa Q1): dataset thứ 2 (VOC), 1–2 held-out backbone (Swin-S).
5. KHÔNG cần: thêm gate single-surrogate; thêm trục không-sửa-core cho ENS.

**Panel CHỐT (10-28, giữ nguyên):** surrogate Mask R-CNN R50; A FCOS-R50 / DINO-R50 | B YOLOv3-D53 / YOLOX-L | C Mask R-CNN Swin-T / DINO-Swin-L; Controlled R101 / ConvNeXt-T / Swin-T; phụ YOLOX-S, DETR-R50; held-out Swin-S; metric AP drop + ASR (`scripts/panel_table.py --gen-models`). Nguyên tắc user: KHÔNG fine-tune surrogate; **khóa quy tắc + commit TRƯỚC khi chạy**; quick test dev30 trước (Swin-L phải xác nhận dev100 — bài học E56→E58: CWT +4 dev30 chết ở dev100); trả lời ngắn gọn; báo giờ GMT+7.

**⚠️ ARTIFACT + GIT:** phiên 11-09 KHÔNG có `.env`/HF_TOKEN và KHÔNG có git credential → PNG V100 (83M, `artifacts/runs/*_v100/`) **mất khi trả máy**, commit **chỉ nằm local, user phải tự push**. Artifact Ampere trên HF `congdanh99/transfer-attack` (private), tag `_a4500_dev30` và `_a4500_dev100` — tải: `set -a; . ./.env; set +a` rồi `python scripts/sync_artifacts.py download <run> --tag _a4500_dev100`. ENS-WP **không tái lập bit-chính-xác** (E78: OSFD elem dùng python-random, `e8{0,1}` chỉ seed torch).

Tóm tắt đường đi tới đây (chi tiết + số liệu trong progress_log):
- RQ1 PASS @ n300; Generalization Panel ĐI TIẾP; Mechanism A1–A4 không đạt tiêu chí; 5 pilot đoán-cơ-chế (A′1, P1, P2, KG, R1/SGM) đều NO-GO.
- **Ablation OSFD** (dev100 → dev300): nguyên liệu = **RRB (+32), xoay trong RRB (+5.5), MI (+3.7)**; KHÔNG phải: resize, k, nhiễu σ, **loss feature**.
- **E65 bảng 2×2 Swin-L** (dev30): OSFD-R50 26.4 / RD-R50 33.8 / OSFD-CNX 26.1 / **RD-CNX 62.8** → tương tác surrogate × method ≈ **+29**: bước nhảy chỉ có khi CÓ CẢ HAI. Đổi riêng surrogate cho OSFD không giúp gì.
- **E66:** cls RoI = **cầu chính** CNN→Transformer (+18.2 Swin-L), feat phụ (+9.8), WP +9.9 gần miễn phí. Giải thích vì sao OSFD (feature thuần) không hưởng.
- **Trần single-R50 Swin-L ~33–38** — vượt được bằng ĐỔI HỌ SURROGATE (ConvNeXt), không bằng tinh chỉnh loss trên R50.

Quy tắc cứng: dev300 là tập development DUY NHẤT (ngoại lệ duy nhất: **devB100**, protocol amendment 2026-11-13, chỉ cho E87, không tune gì trên nó); **không tune trên n300/n1000**; mọi quy tắc quyết định khóa + commit TRƯỚC khi chạy. Quét siêu tham số trên dev30 = forking path → ô thắng BẮT BUỘC xác nhận dev100, số cuối trên n300/n1000.

## Chạy môi trường

Máy GPU thuê mới hoàn toàn — chạy 1 lệnh duy nhất, làm hết mọi thứ (dựng venv, tải ĐỦ checkpoint Controlled +
Generalization Panel + chẩn đoán, tải dataset, chốt danh sách ảnh; có `HF_TOKEN` trong `.env` thì kéo luôn PNG baseline n300):

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

Idempotent — chạy lại an toàn nếu bị gián đoạn giữa chừng (mỗi bước tự bỏ qua phần đã xong).
`data/image_lists/{n300,n1000}.csv` chỉ tạo **một lần duy nhất** (đã commit vào git, seed=42,
xem `docs/protocol_lock.md`) — `bootstrap.sh` không bao giờ ghi đè lại danh sách này.

Chạy riêng từng bước (khi cần debug hoặc chỉ thiếu 1 phần): `scripts/setup_env.sh`,
`scripts/download_checkpoints.sh`, `scripts/download_dataset.sh`, `scripts/generate_image_lists.py`.

Chi tiết/troubleshooting: [docs/environment_setup.md](docs/environment_setup.md).

## Quy ước

- Không tối ưu/tune trên n=300, càng không trên n=1000 (idea.md §4).
- Không cài đặt method mới (idea.md §12) trước khi Baseline-First + Hypothesis Pass @300 (idea.md §10) xác nhận transfer gap.
- Mỗi lần chạy thí nghiệm: cùng ảnh, cùng surrogate, cùng preprocessing, cùng epsilon, cùng gradient-evaluation budget B — ghi rõ B dùng ở mức nào (docs/protocol_lock.md).
- Sau mỗi phiên GPU có kết quả hoặc quyết định mới: append entry vào `docs/progress_log.md`, commit + push trước khi trả máy. Artifact nặng (checkpoint, ảnh adversarial) không commit — xem `.gitignore` và mục "Persistent storage" trong docs/environment_setup.md.
- Toàn bộ docs trong dự án này viết bằng tiếng Việt.
