# Nhật ký tiến độ (Progress Log)

Chỉ thêm mới (append-only). Sau mỗi phiên làm việc có kết quả hoặc quyết định, thêm một entry mới có ngày ở cuối file. Không sửa lại entry cũ, trừ khi sửa lỗi sai sự kiện — nếu phân vân, gạch ngang phần cũ thay vì xóa.

---

## 2026-09-22 — Khóa protocol + dựng cấu trúc dự án

- `idea.md` (research plan đã khóa, baseline-first) đã có sẵn từ trước, kèm `papers/` (OSFD, AugTrans, benchmark taxonomy) và `ref-repo/OSFD-main` (code OSFD gốc, mmdet v2.28.2).
- Quyết định framework: **mmdetection v3.3.0** (không dùng v2 như OSFD ref-repo), vì v3 có sẵn config DINO/ConvNeXt/Swin cần cho Generalization Panel; code attack loop tham khảo OSFD ref-repo cần port sang v3 API.
- Khóa exact config Controlled Panel (surrogate R50 + 3 target R101/ConvNeXt-T/Swin-T, toàn bộ 3x+multi-scale) và Generalization Panel (FCOS-R50, DETR-R50, YOLOX-S, DINO-Swin-**L** thay Swin-T vì không có checkpoint chính thức) — chi tiết đầy đủ ở [protocol_lock.md](protocol_lock.md), [model_registry.md](model_registry.md).
- Khóa định nghĩa B (gradient-evaluation budget) = tổng số `.backward()` calls; AugTrans (dùng EOT, N_EOT=10) normalize bằng `iterations = B/N_EOT`. Loại B=10 khỏi so sánh chéo method vì AugTrans không đủ iteration ở mức này (chỉ 1 iteration, không đủ cho curriculum/EOT hoạt động). Primary budget set chốt: **B ∈ {50, 200}**.
- Dựng cấu trúc dự án theo khuôn mẫu từ `transfer-attack-new` (dự án nghiên cứu tương tự, thư mục khác): `CLAUDE.md` (bootstrap, auto-load mỗi session) + `docs/protocol_lock.md` + `docs/model_registry.md` + `docs/progress_log.md` (file này) + `docs/environment_setup.md` + `scripts/setup_env.sh`.
- Lý do tổ chức thế này: GPU thuê (RTX 3090 / RTX 4000 Ada), mỗi phiên là máy mới hoàn toàn, memory riêng của Claude không tồn tại trên máy GPU nếu phiên chạy trực tiếp trên máy thuê — toàn bộ context bắt buộc phải nằm trong repo (git-tracked), không phụ thuộc memory ngoài repo.
- Chưa cài môi trường thật, chưa tải checkpoint, chưa verify model nào bằng cách chạy thật (mọi AP trong model_registry.md lấy từ README GitHub, chưa tự eval lại).
- Bước tiếp theo: chạy `scripts/setup_env.sh` trên máy GPU thật lần đầu để verify script chạy được (chưa test), sau đó tải 4 checkpoint Controlled Panel + verify clean AP khớp bảng trong model_registry.md.
