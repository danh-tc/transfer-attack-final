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

**Trạng thái (cập nhật 2026-10-22): TẠM DỪNG — đọc entry cuối `docs/progress_log.md` ("TẠM DỪNG 2026-10-22").** Máy phiên cuối **V100 16GB** (B cùng thời gian OSFD B50: C2-OCT-1V B41, WP2 B36; A4500: C2-OCT-1V B57). **Method tốt nhất: `C2-OCT-1V-WP2`** (C2-OCT-1V + nhiễu trọng số R50 ρ = 0.002) — GIÚP @ dev100: Swin-L +6.2 [3.1, 8.2] so C2-OCT-1V, +11.5 so OSFD; lợi chỉ ở DINO-Swin-L, không ở Swin-T; trần đo được ~38 (WP2 B100). **Mục tiêu user (10-22): tìm method "bước nhảy kiểu RRB", DINO-Swin-L > 50% ở ε = 5, 1 surrogate R50 — KHÔNG dừng ở định vị SOTA + cơ chế.** Đã bác hôm 10-22 (dev30, so WP2): cỡ object (SZ / R / SZR / SZC), xóa texture (TX), bất biến biên độ (LN), BN theo ảnh (IN), hướng pixel về ảnh xóa object (E17); ε = 8 gác lại. **Việc tiếp:** ý chưa chạy — WP2 + loss đích F_R50(ảnh đã xóa object), thành phần universal (đa dạng giữa ảnh); khóa quy tắc vào progress_log TRƯỚC khi chạy, ngưỡng "bước nhảy" ΔSwin-L ≥ +10. Nguyên tắc user: SOTA + novelty, target CHƯA THẤY; quick test dev30 trước; trả lời ngắn gọn; báo giờ theo GMT+7. Commit 10-22 CHƯA push, PNG 10-22 CHƯA upload HF (chờ user đồng ý). `.env` có `HF_TOKEN = ...`.

Tóm tắt đường đi tới đây (chi tiết + số liệu trong progress_log):
- RQ1 PASS @ n300; Generalization Panel ĐI TIẾP; Mechanism A1–A4 không đạt tiêu chí; 5 pilot đoán-cơ-chế (A′1, P1, P2, KG, R1/SGM) đều NO-GO → mở lại RQ3 theo plan mới: **phân tích SOTA (OSFD) trước, method sau.**
- **Ablation OSFD** (dev100 → dev300): nguyên liệu = **RRB (+32), xoay trong RRB (+5.5), MI (+3.7)**; KHÔNG phải: resize, k, nhiễu σ, **loss feature** (task loss + RRB ≥ OSFD).
- **Cơ chế:** RRB/xoay làm δ tần thấp hơn, bớt đặc thù surrogate (D₄ R50 giảm, D₄ target tăng); D₄ trên target dự đoán suppression theo ảnh. **DINO-Swin-L bền** vì attack chỉ lệch vừa feature Swin-L và encoder DINO "sửa" (cosd 0.42 → 0.07); object giả bị chặn; chỉ hạ confidence phân loại còn lọt. Phá định vị: có thật nhưng đóng góp nhỏ. Feature loss mạnh ở YOLOX/CNN, loss phân loại mạnh ở DINO-Swin-L → bổ trợ.
- **Method C2** (đóng băng) = MI + RRB (co-transform GT) + L_OSFD + 4·(loss_rpn_cls + loss_cls), 1 forward/view, B = 50, ε = 5 (`PROBE-osfd+cls` trong `scripts/run_baselines.py`). **Sàng lọc dev300 QUA:** CrossAvg₃ +4.7 [1.7, 6.0] so OSFD, cả 3 target khác họ > 0, DINO-Swin-L +7.0.

**GO @ n300 cho C2 CŨ (RRB; đã bị C2-OCT-1V thay — chỉ để tham khảo lệnh):** `run_baselines.py --split n300 --methods OSFD PROBE-osfd+cls --out metrics_go_c2.json` rồi `eval_generalization.py --run n300_B50_eps5 --models yolox_s dino_swin_l ... --dets dets_go_c2.json --out generalization_go_c2.json` (log `results/go_c2/run.log`; resume được — chạy lại đúng lệnh). Cần PNG baseline `artifacts/runs/n300_B50_eps5/` (HF, `scripts/sync_artifacts.py download n300_B50_eps5`, token ở `.env` — gitignored, KHÔNG in ra). Khi xong:
```bash
python scripts/ablation_screen.py --run n300_B50_eps5 --dets dets_go_c2.json --tag go_c2_n300 --cells PROBE-osfd+cls
python scripts/go_check.py --tag go_c2_n300 --cand PROBE-osfd+cls   # tiêu chí GO khóa, chạy 1 lần
```
GO → bước 4.2 (n1000, B ∈ {50, 200}, ε ∈ {5, 8}, rà literature baseline mới). NO-GO → ghi log, user quyết.

Quy tắc cứng: dev300 là tập development DUY NHẤT; không tune trên n300/n1000; mọi quy tắc quyết định khóa + commit TRƯỚC khi chạy. Artifact chưa upload HF: `artifacts/runs/dev300_B50_eps5/` (mọi ô ablation/pilot).

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
