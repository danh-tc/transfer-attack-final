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

**Trạng thái (cập nhật 2026-10-06): chuỗi CƠ CHẾ "vì sao đa dạng hình học giúp, vì sao DINO-Swin-L khó" (E1–E9).** Cách làm user chốt: dev30 xem xu hướng → mở rộng khi ổn; ưu tiên thí nghiệm rẻ (forward / gradient / PNG có sẵn); thí nghiệm tốn kém phải phân tích vì sao đáng chạy, user confirm rồi mới chạy; mọi quy tắc đọc khóa vào progress_log TRƯỚC khi chạy. Đọc entry "TỔNG HỢP E1–E8" (2026-10-06) trong `docs/progress_log.md` trước. Tóm tắt: đa dạng làm δ BỀN (không qua hướng gradient); sức bền Swin-L tập trung ở object LỚN; Swin-L nhận object lớn bằng HÌNH DẠNG, R50 + target khác bằng cấu trúc cục bộ; δ từ R50 là mẫu tần cao khớp nội dung; Swin-L KHÔNG bền cơ bản (white-box OSFD 84.6% vs transfer 27%) → nút thắt là chuyển giao. Hướng method (chưa khóa): loss / δ ở MỨC OBJECT (manh mối: loss cls của C2 / C2-PIA). E9 (C2, C2-PIA @ dev30 tách theo kích thước) XONG: lợi loss cls trên Swin-L ở object NHỎ, không ở object lớn (lực yếu). **TẠM DỪNG 2026-10-07 — đọc entry cuối `docs/progress_log.md` (tổng kết + việc mở).** Q1 (vì sao đa dạng hình học giúp) có câu trả lời ứng viên đã kiểm ở dev30: **lệch PHA LƯỚI stride của surrogate** (tịnh tiến lẻ pixel ≈ ROT15 ≈ OSFD; tịnh tiến bội 32 ≈ NOISE). Q2 (Swin-L ~30%) chưa có cơ chế đứng được — user dừng hướng dùng cos_HP. **Stride quyết định XONG 2026-10-08 (dev30): k\* = 4** — phần δ gắn lưới thuộc stride NÔNG (2/4, phần nào 8) của R50, không phải stride 32. SH2 (dịch chẵn) @ dev30: f = 0.86 → **stride 4 (maxpool → C2) quyết định**, không phải stride 2; Swin-L SH2 24.0. Việc mở: xác nhận dev100 (SH32/SH4/SH2/SHODD), method lấy mẫu pha có chủ đích (Q3, phủ residue mod 4/8). Không có tiến trình nào chạy dở. Bước 1 chuỗi elem XONG (chọn SCL25, CAM-R15, SHR30); bước 2 TẠM GÁC (user). Script mới: `attack/targets.py` (gradient target chỉ để phân tích), `scripts/e{1..8}_*.py`.

Tóm tắt đường đi tới đây (chi tiết + số liệu trong progress_log):
- RQ1 PASS @ n300; Generalization Panel ĐI TIẾP; Mechanism A1–A4 không đạt tiêu chí; 5 pilot đoán-cơ-chế (A′1, P1, P2, KG, R1/SGM) đều NO-GO → mở lại RQ3 theo plan mới: **phân tích SOTA (OSFD) trước, method sau.**
- **Ablation OSFD** (dev100 → dev300): nguyên liệu = **RRB (+32), xoay trong RRB (+5.5), MI (+3.7)**; KHÔNG phải: resize, k, nhiễu σ, **loss feature** (task loss + RRB ≥ OSFD).
- **Cơ chế:** RRB/xoay làm δ tần thấp hơn, bớt đặc thù surrogate (D₄ R50 giảm, D₄ target tăng); D₄ trên target dự đoán suppression theo ảnh. **DINO-Swin-L bền** vì attack chỉ lệch vừa feature Swin-L và encoder DINO "sửa" (cosd 0.42 → 0.07); object giả bị chặn; chỉ hạ confidence phân loại còn lọt. Phá định vị: có thật nhưng đóng góp nhỏ. Feature loss mạnh ở YOLOX/CNN, loss phân loại mạnh ở DINO-Swin-L → bổ trợ.
- **Method C2** (đóng băng) = MI + RRB (co-transform GT) + L_OSFD + 4·(loss_rpn_cls + loss_cls), 1 forward/view, B = 50, ε = 5 (`PROBE-osfd+cls` trong `scripts/run_baselines.py`). **Sàng lọc dev300 QUA:** CrossAvg₃ +4.7 [1.7, 6.0] so OSFD, cả 3 target khác họ > 0, DINO-Swin-L +7.0.

**GO @ n300 (tạm hoãn — chạy lại khi user yêu cầu, nối tiếp được):** `run_baselines.py --split n300 --methods OSFD PROBE-osfd+cls --out metrics_go_c2.json` rồi `eval_generalization.py --run n300_B50_eps5 --models yolox_s dino_swin_l ... --dets dets_go_c2.json --out generalization_go_c2.json` (log `results/go_c2/run.log`; resume được — chạy lại đúng lệnh). Cần PNG baseline `artifacts/runs/n300_B50_eps5/` (HF, `scripts/sync_artifacts.py download n300_B50_eps5`, token ở `.env` — gitignored, KHÔNG in ra). Khi xong:
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
