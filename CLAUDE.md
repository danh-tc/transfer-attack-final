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

**Trạng thái (cập nhật 2026-10-30 ~16:45 GMT+7, máy RTX 3090): TẠM DỪNG — đọc đuôi `docs/progress_log.md` (E46→E61).** Phiên nay chạy trên 3090 (KHÁC V100 phiên 10-28 — xem mục "lệch phần cứng" dưới).

**METHOD CHỐT = `C2-OCT-1V-WP2-RD` ("RD")** (L_OSFD C3–C5 + 6·loss_cls RoI, OCT 1V, MI, WP ρ = 0.002, B36 = cùng thời gian OSFD B50). **E58 XÁC NHẬN @ dev100 cùng máy 3090, CI loại 0: RD − OSFD trên DINO-Swin-L = +9.8 [5.1, 13.1]**, hơn ConvNeXt +6.6, Swin-T +5.4 — SOTA-competitive ở target khác họ backbone. (RD-CV = +0.3·vanishing: E46 nhỉnh drop/APloc nhưng ASR chưa qua ngưỡng; để phụ.)

**⚠️ LỆCH PHẦN CỨNG (quan trọng cho mọi so sánh Swin-L):** cùng code RD, V100 → Swin-L ~38, 3090 → ~31–34. Root cause ĐÃ CHỐT (E59/E60/E61): **kernel fp16 Volta lúc SINH δ** (loại seed, code, eval; δ V100 eval trên 3090 = 37.7). Mẫu dấu khác (cos 0.30) cùng mọi thống kê bậc thấp. Khai thác hậu kỳ (RD-GQ = flush gradient nhỏ) chỉ tái tạo ~1/3 (+2.4, đỉnh p=0.7). → **KHÔNG ghép số Swin-L giữa máy V100 và Ampere; số dev30 Swin-L ≤ ~4 coi là nhiễu tới khi dev100 xác nhận** (bài học E56→E58: CWT +4 dev30 chết ở dev100). Số Swin-L trung thực của RD trên 3090 = **33.8 dev100**.

**Trần Swin-L một-R50-chuẩn ~33–38** (3090 dev100 33.8; V100 37.7; δ cùng-họ Swin-T 54–59). **50 gần như = xóa gap RQ1 → cần đổi surrogate (robust-AT / cùng-họ), ngoài ràng buộc "KHÔNG fine-tune surrogate".** Các hướng single-R50 đã thử & ĐÓNG phiên này: RD-CV/RD-V (đánh đổi ASR↔drop), RD-SCR (regularizer bền-xáo, E55: scramble là chỉ báo không phải đòn bẩy), RD-CWT (E56 dev30 +4 → E58 dev100 +0.6, chết), kênh ngữ cảnh/nền (E50/E51/E52: oracle WB chỉ-nền 66% nhưng R50 không với tới), RD-GQ (flush fp16, +2.4 không đủ). Cơ chế cho RQ2: **gap = SAI THỨ TỰ ĐỘ DỐC** (E53: hướng R50 dốc hơn hướng chung ~1.9× trên loss R50; δ_S mạnh hơn ~2.1× trên Swin-L), δ_R50 overfit vị trí cục bộ (E54).

**Panel CHỐT (10-28, giữ nguyên):** surrogate Mask R-CNN R50; A FCOS-R50 / DINO-R50 | B YOLOv3-D53 / YOLOX-L | C Mask R-CNN Swin-T / DINO-Swin-L; Controlled R101 / ConvNeXt-T / Swin-T; phụ YOLOX-S, DETR-R50; held-out Swin-S; metric AP drop + ASR (`scripts/panel_table.py`, có `--gen-models` chạy gọn + APloc/CSR/ASR-micro). Nguyên tắc user: KHÔNG fine-tune surrogate; khóa quy tắc + commit TRƯỚC khi chạy; quick test dev30 trước (nhưng Swin-L phải xác nhận dev100); trả lời ngắn gọn; báo giờ GMT+7.

**VIỆC MAI (user quyết):** (1) **Chốt RD → dựng bảng paper**: full panel + OSFD + MI + M-DI² @ dev100/dev300 CÙNG MÁY (đừng trộn V100), rồi n300 GO → n1000 → ε=8; verify clean AP full val2017 panel A/B/C; kiểm surrogate FRC-R50. HOẶC (2) **nới ràng buộc surrogate** (robust-AT / cùng-họ) để đuổi 50 — đổi threat model, cần duyệt rõ. Code mới phiên nay: `pia.{_scramble,_cwt_spec/_cwt_apply,_with_scramble, params scr/cwt/pix_region/box_feat/gflush}`, `--seed-base`/`--no-tf32` trong run_baselines, `panel_table --gen-models`, `scripts/e5{3,4,9}_*.py`, `e51_wb_bgmask.py`, `e50_bg_mask.py`. Ô: `C2-OCT-1V-WP2-RD-{CV,V,SCR*,CWT*,GQ*}`, `RD-{OUT,RING}{F,O}`.

**⚠️ ARTIFACT 3090 phiên nay MẤT khi trả máy, CHƯA upload HF** (số tổng hợp + dets đã trong `results/` git). PNG tải HF tag `_1028v100` là của V100 (giải nén ra `artifacts/runs/dev300_B36_eps5_limit100/` — ⚠️ TRỘN với tên không tag; δ V100 ≠ 3090). Mai cần dev100/dev300 thì sinh lại trên máy mới (`results/e*/run.sh`). `.env` (HF_TOKEN) đã có nhưng gitignored → máy mới tạo lại.

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
