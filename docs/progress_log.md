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

---

## 2026-09-22 — Chạy `setup_env.sh` lần đầu trên GPU thuê thật (RTX 3090) — 3 lỗi, đã sửa script

- Máy: RTX 3090, driver 570.211.01, 24GB VRAM (GPU thuê thật, không phải RTX 4000 Ada).
- `scripts/setup_env.sh` chạy lần đầu **fail ở bước cài mmdetection editable** (`set -e` dừng script giữa chừng, exit code thật bị pipe `tee` che mất — cần lưu ý lần sau không suy luận "exit code 0 của lệnh có tee" = script thành công, phải đọc log). 3 lỗi phát hiện, đã sửa trực tiếp vào `scripts/setup_env.sh`:
  1. **`pip install -e third_party/mmdetection`** (không có `--no-build-isolation`) → `ModuleNotFoundError: No module named 'torch'`. Nguyên nhân: `setup.py` của mmdet import `torch` ở top-level, nhưng build isolation mặc định của pip tạo venv tạm không có torch (dù venv chính đã cài). Fix: thêm `--no-build-isolation`.
  2. Sau khi thêm `--no-build-isolation`, gặp lỗi mới: `Project ... uses a build backend that is missing the 'build_editable' hook`. Nguyên nhân: `setuptools` mặc định trong venv là **60.2.0** (quá cũ), không hỗ trợ PEP 660 editable install cho package chỉ có `setup.py` (không có `pyproject.toml`) khi dùng với pip rất mới (26.2.1 trong môi trường này). Fix: `pip install -U "setuptools<81"` trước bước cài mmdet (đủ mới để hỗ trợ PEP 660, chưa nhảy sang major version có thể đổi hành vi khác — chưa test setuptools ≥81).
  3. Cài `opencv-python-headless` không pin version → kéo bản mới nhất (5.0.0.93) đòi `numpy>=2`, phá pin `numpy==1.26.4` (torch import lỗi `_ARRAY_API not found` vì mmcv/mmdet build với numpy 1.x ABI). Ngoài ra `mmcv`/`mmengine` tự kéo thêm package `opencv-python` (không phải headless) cũng bản mới nhất, cùng vấn đề. Fix: pin cả `opencv-python` lẫn `opencv-python-headless` về `4.10.0.84` (bản cuối cùng còn hỗ trợ numpy<2), cài cùng lúc với `numpy==1.26.4` để resolver không tự nâng lại, và pin lại lần nữa sau khi cài `pycocotools`/thư viện phụ trợ (phòng bị kéo lại).
  - Xung đột resolver **chấp nhận bỏ qua, không phải lỗi chặn**: `openxlab==0.1.3 requires setuptools~=60.2.0` (openxlab là dependency phụ của mmpretrain, không dùng trong pipeline attack/eval của dự án này) và `requests==2.28.2 requires urllib3<1.27` (không ảnh hưởng phần dùng trong dự án). Không cần xử lý trừ khi sau này thực sự import `openxlab` hoặc gặp lỗi liên quan `requests`.
- Sau khi sửa, verify sạch: `torch 2.1.2+cu118 cuda_available=True` (GPU: NVIDIA GeForce RTX 3090), `mmengine 0.10.7`, `mmcv 2.1.0`, `mmdet 3.3.0`, `mmpretrain 1.2.0`, `numpy 1.26.4`, `cv2 4.10.0` — không còn warning ABI numpy.
- Verify thêm (chưa nằm trong `setup_env.sh` gốc, làm thủ công để bám protocol_lock.md — "việc đầu tiên khi có GPU"): parse **và** build (`MODELS.build`) cả 4 config Controlled Panel bằng `mmdet.registry` + `init_default_scope("mmdet")` — tất cả OK, không lỗi registry ConvNeXt/Swin (mmpretrain resolve đúng):
  - `mask-rcnn_r50_fpn_ms-poly-3x_coco.py` — 44.4M params
  - `mask-rcnn_r101_fpn_ms-poly-3x_coco.py` — 63.4M params
  - `mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` — 48.1M params
  - `mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` — 47.8M params
  - Lưu ý: đây mới là build kiến trúc model (random init), **chưa tải checkpoint, chưa load weight thật, chưa eval clean AP** — cột "Ngày verify" trong `model_registry.md` vẫn để trống, chưa được điền ở bước này.
- `environment_report.txt` đã ghi lại (không commit, xem `.gitignore`).
- Bước tiếp theo: viết `scripts/download_checkpoints.sh` (tải 4 checkpoint Controlled Panel qua `mim download mmdet --config <name> --dest .` theo đúng tên config trong `protocol_lock.md`) + tải/chốt COCO subset n=300 ảnh (ghi `data/image_lists/`), sau đó chạy eval clean thật để điền cột "Ngày verify" trong `model_registry.md`.

---

## 2026-09-22 — Tải checkpoint + COCO val2017, verify 4/4 model Controlled Panel trên GPU thật

Tiếp tục phiên GPU cùng ngày (sau entry "Chạy `setup_env.sh` lần đầu..." ở trên — môi trường đã dựng xong trong entry đó).

- **Tải dataset**: `images.cocodataset.org` bị lỗi cert HTTPS phía server chính thức (SSL cert trả về là của `s3.amazonaws.com`, không match `images.cocodataset.org` — lỗi đã tồn tại lâu ở phía COCO, không phải do máy/mạng của mình). Workaround: tải qua `http://` thay vì `https://` (server vẫn phục vụ HTTP bình thường). Đã tải + giải nén `annotations_trainval2017.zip` (252MB) và `val2017.zip` (815MB, 5000 ảnh) vào `data/coco/` (không commit — đúng `.gitignore`). Cần cài thêm `unzip` (chưa có trong apt deps của `setup_env.sh`) — đã thêm vào script.
- **Viết `scripts/download_checkpoints.sh`** — tải 4 checkpoint Controlled Panel qua `mim download mmdet --config <identifier> --dest checkpoints/`. Phát hiện 2 lỗi, đã sửa:
  1. `mim download` báo lỗi `model-index.yml ... not found, please upgrade your mmdet` — nguyên nhân: mmdetection cài kiểu editable PEP 660 (meta path finder, không phải `.egg-link` như `setup.py develop` kiểu cũ) khiến `mim` (dùng `pkg_resources.get_distribution(...).location`) resolve sai đường dẫn cài đặt — trỏ vào `.venv/lib/.../site-packages/mmdet` (không tồn tại vật lý) thay vì `third_party/mmdetection/mmdet` (nơi code thật nằm, và nơi mình đã tự tạo symlink `.mim` → `configs/`, `tools/`, `model-index.yml` ở bước trước vì `add_mim_extension()` trong `setup.py` của mmdet chỉ chạy khi `'develop' in sys.argv`, không xảy ra với pip PEP 660). Fix: tạo thêm symlink `'.venv/lib/python3.10/site-packages/mmdet' -> 'third_party/mmdetection/mmdet'` — sau đó `mim` resolve đúng, `import mmdet` vẫn hoạt động bình thường (Python ưu tiên thư mục thật trong site-packages hơn finder do `.pth` đăng ký). **Chưa đưa fix này vào `setup_env.sh`** (làm thủ công trong phiên này) — cần thêm vào script cho lần thuê máy sau, xem mục "còn mở" cuối entry.
  2. Tên checkpoint identifier trong `protocol_lock.md`/`model_registry.md` ghi kiểu v2 cũ (`mask_rcnn_r50_fpn_mstrain-poly_3x_coco`, gạch dưới) — nhưng `mim` (model-index v3.3.0) chỉ nhận identifier **khớp đúng tên file config** (`mask-rcnn_r50_fpn_mstrain-poly_3x_coco`, gạch nối `mask-rcnn`). Đã sửa cả `scripts/download_checkpoints.sh` và 2 file docs cho khớp thực tế (xác nhận bằng cách đọc `configs/*/metafile.yml` field `Name`).
  - Checkpoint Swin-T lần tải đầu qua `mim` bị **đứt file** (94.9MB thay vì 191MB thật, load checkpoint báo `RuntimeError: unexpected EOF`) — không rõ nguyên nhân (mim không báo lỗi, exit code 0). Fix: tải lại trực tiếp từ URL trong `metafile.yml` (field `Weights`) bằng `curl`, ra đúng 191,464,891 bytes, load OK. Bài học: sau khi `mim download`, nên luôn `init_detector(...)` thử load thật trước khi tin checkpoint nguyên vẹn, không chỉ tin exit code hay tin kích thước file "trông hợp lý".
  - Cả 4 config `.py` cũng được `mim` copy vào `checkpoints/` nhưng bị thiếu 1 file (Swin) — cuối cùng quyết định **dùng thẳng config gốc trong `third_party/mmdetection/configs/.../*.py`** làm nguồn chính thức thay vì bản copy trong `checkpoints/`, để tránh 2 nguồn config lệch nhau.
- **Verify 4/4 checkpoint bằng eval clean AP thật trên full COCO val2017 (5000 ảnh)**, dùng `third_party/mmdetection/tools/test.py` — mỗi model mất ~5-7 phút trên RTX 3090. Kết quả khớp chính xác README/`model_registry.md`, không lệch:
  - Mask R-CNN R50 (surrogate): bbox AP **40.9**
  - Mask R-CNN R101 (same-family target): bbox AP **42.7**
  - Mask R-CNN ConvNeXt-T (cross-CNN target): bbox AP **46.2**
  - Mask R-CNN Swin-T (CNN→Transformer target): bbox AP **46.0**
  - Log + metrics đầy đủ (bbox + segm AP, AP50/75/S/M/L) nằm ở `results/eval_clean/{r50,r101,convnext-t,swin-t}/` — nhẹ, git-tracked được.
  - Đã cập nhật `docs/model_registry.md` (điền "Ngày verify" = 2026-09-22, thêm cột AP verify GPU thật) và `docs/protocol_lock.md` (sửa tên checkpoint identifier + ghi chú đã verify).
- **Quyết định chấp nhận, không phải lỗi chặn**: dùng `http://` thay `https://` cho `images.cocodataset.org` — vẫn xác thực nội dung đúng qua kích thước file + giải nén thành công + annotation JSON đọc được bình thường qua `pycocotools`, rủi ro MITM chấp nhận được vì đây là dataset public, không phải secret/credential.
- **Còn mở, cần làm ở phiên tiếp theo hoặc trước khi trả máy**:
  1. Đưa 2 fix vào `scripts/setup_env.sh`/`scripts/download_checkpoints.sh` cho tái lập được ở máy thuê tiếp theo: symlink `site-packages/mmdet` (mục 1 ở trên) hiện chưa nằm trong script, chỉ làm thủ công trong phiên này — **phải thêm vào `setup_env.sh` trước khi trả máy**, nếu không lần sau `mim download` sẽ lại lỗi.
  2. Chưa tạo `data/image_lists/` (danh sách cố định n=300/n=1000 image_id theo seed) — bắt buộc trước khi chạy Baseline-First Stage (idea.md §4, §8).
  3. Chưa viết code attack nào (MI-FGSM/DI-FGSM/OSFD/AugTrans port sang mmdet v3) — vẫn ở đúng giai đoạn "chưa có code" như CLAUDE.md mô tả, chỉ mới xong phần hạ tầng (env + checkpoint + dataset pool).
  4. Chưa commit + push các thay đổi trong phiên này (`scripts/setup_env.sh`, `scripts/download_checkpoints.sh`, `docs/*.md`, `results/eval_clean/`) — cần làm trước khi trả máy GPU, theo đúng quy ước trong CLAUDE.md.

---

## 2026-09-22 — Commit hạ tầng + viết bootstrap.sh gộp toàn bộ setup, chốt danh sách ảnh n=300/n=1000

Tiếp tục cùng phiên GPU (RTX 3090) — đã commit + push xong phần hạ tầng của 2 entry trước (env fix + checkpoint verify), giờ làm nốt phần còn thiếu #1 và #2 ở mục "còn mở" của entry trước.

- **Fix #1 (site-packages/mmdet symlink cho `mim download`) đã đưa vào `scripts/setup_env.sh`** (trước đó chỉ làm thủ công) — cùng với fix `mmdet/.mim` symlink, script giờ tự làm cả 2 fix khi chạy trên máy mới, không cần sửa tay nữa.
- **Viết `scripts/download_dataset.sh`** — tải + giải nén COCO val2017 (ảnh + annotations) vào `data/coco/`, idempotent (bỏ qua nếu đã có đủ 5000 ảnh / đã có annotations). Dùng lại workaround `http://` cho `images.cocodataset.org` (xem entry trước).
- **Viết `scripts/generate_image_lists.py`** — chốt danh sách ảnh n=300/n=1000 theo seed cố định. Quyết định (đã ghi vào `docs/protocol_lock.md`):
  - Pool: COCO val2017 lọc còn ảnh có ≥1 instance annotation (4952/5000 ảnh).
  - `SEED=42`, `n1000 = random.Random(42).sample(sorted(pool), 1000)`, `n300 = n1000[:300]` — **n=300 là tập con của n=1000** (quyết định có chủ đích: tránh Confirmation Stage và Final Stage dùng 2 nguồn ảnh không lồng nhau, dễ gây nhiễu khi so sánh kết quả giữa 2 phase).
  - Script có guard idempotent: nếu `data/image_lists/meta.json` đã tồn tại thì không chạy lại (danh sách coi như khóa vĩnh viễn sau lần đầu, dù script tất định 100% nên chạy lại cũng ra kết quả giống hệt — guard chỉ để tránh lẫn lộn nếu code bị sửa sau này).
  - Đã chạy thật lần đầu: `data/image_lists/n300.csv` (300 dòng), `n1000.csv` (1000 dòng), `meta.json` — đã verify n300 ⊂ n1000 bằng script kiểm tra riêng, đúng.
- **Viết `scripts/bootstrap.sh`** — gộp cả 4 bước (`setup_env.sh` → `download_checkpoints.sh` → `download_dataset.sh` → `generate_image_lists.py`) thành 1 lệnh duy nhất cho máy GPU thuê mới, đúng theo yêu cầu "thuê pod mới thì auto setup hết". Đã test từng script con trên máy hiện tại (idempotency check hoạt động đúng — bỏ qua phần đã có).
- Cập nhật `CLAUDE.md` và `docs/environment_setup.md`: entrypoint chính giờ là `bash scripts/bootstrap.sh` thay vì `scripts/setup_env.sh` đơn lẻ.
- **Việc còn lại trước khi bắt đầu code attack**: commit + push các file mới của entry này (`scripts/bootstrap.sh`, `scripts/download_dataset.sh`, `scripts/generate_image_lists.py`, `data/image_lists/{n300,n1000}.csv`, `data/image_lists/meta.json`, docs đã sửa). Sau đó mới bắt đầu port code attack (MI-FGSM/DI-FGSM/OSFD/AugTrans) từ `ref-repo/OSFD-main` sang mmdet v3 API — vẫn hoàn toàn chưa làm gì ở phần này.

---

## 2026-09-22 — Đọc kỹ ref-repo/OSFD-main/attack/, viết khung harness cho mmdet v3

Tiếp tục cùng phiên GPU (RTX 3090).

- **Đọc toàn bộ `ref-repo/OSFD-main/attack/`** để hiểu kiến trúc trước khi port. Phát hiện quan trọng:
  - Repo tách 2 loại module ghép qua registry: `base_attack` (`BASEATK`: IFGSM/MI/DI/RRB — cơ chế update noise/biến đổi input, không quan tâm loss từ đâu) và `transfer_attack` (`TSFATK`: chỉ có **OSFD** — định nghĩa loss nào dùng để lấy gradient, ở đây là feature-disruption `MSE(k·feat_clean, feat_adv)`, không cần GT).
  - `attack/comparing/__init__.py` **rỗng** — dù tên thư mục gợi ý "attack để so sánh", không có MI-FGSM/DI-FGSM task-loss chuẩn nào được đăng ký thật sự.
  - `ummdet/detectors/model_hook.py` có sẵn `ModelHook.forward_bottom()` (tính loss task thật qua GT, kiểu `forward_train` v2) nhưng **không được wire vào transfer_attack nào cả** — code chết.
  - Kết luận: repo gốc chỉ thực sự implement được OSFD. MI-FGSM/DI-FGSM baseline (task-loss chuẩn, theo idea.md §8) và AugTrans đều phải viết mới hoàn toàn, không có ref code dùng được trực tiếp.
- **Khảo sát API mmdet v3 thật** (đọc source `third_party/mmdetection/mmdet/apis/inference.py`, `mmdet/models/detectors/{base,two_stage}.py`, `mmengine/model/base_model/data_preprocessor.py`) để thiết kế harness:
  - Phát hiện quan trọng: normalize (mean/std) + pad + bgr2rgb trong v3 nằm trong `model.data_preprocessor` — một `nn.Module` thật, **không có `torch.no_grad()`/`.detach()` nào bên trong** (đã đọc source xác nhận). Nghĩa là có thể cộng noise thẳng vào tensor pixel [0,255] THÔ (trước data_preprocessor) và gradient lan truyền ngược tự nhiên — **đơn giản hơn nhiều so với v2** (OSFD phải tự denormalize/renormalize thủ công vì v2 normalize trong CPU pipeline, không khả vi).
  - Verify cả 4 config Controlled Panel dùng **chung hệt** `test_pipeline` (`Resize(scale=(1333,800), keep_ratio=True)`) và **chung hệt** `data_preprocessor` (mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], bgr_to_rgb=True, pad_size_divisor=32) — nghĩa là sinh 1 tensor pixel duy nhất cho mỗi ảnh (ở resolution đã resize theo surrogate) feed thẳng được vào cả 4 model mà không cần resize riêng noise cho từng target như v2 phải làm (`resizer = transforms.Resize(...)` trong `single_gpu_test` cũ). Đơn giản hóa đáng kể.
  - `model.loss(batch_inputs, batch_data_samples)` gọi thẳng được (định nghĩa trong `BaseDetector`/`TwoStageDetector`), không cần custom detector subclass như `MaskRCNNAdv`/`ModelHook` bên v2.
  - Test pipeline của tất cả config Controlled Panel **đã có sẵn `LoadAnnotations(with_bbox=True)`** — GT box tự động có trong `data_sample.gt_instances` khi build dataset qua registry chuẩn, không cần viết pipeline riêng.
- **Viết `attack/` package** (mmdet v3 harness, thay thế hoàn toàn `ref-repo/OSFD-main/attack/utils/mmdet.py`):
  - `attack/models.py` — `CONTROLLED_PANEL` registry (config+checkpoint path cho 4 model, khớp `protocol_lock.md`), `load_model()`/`load_surrogate()`/`load_all_targets()`.
  - `attack/data.py` — `load_image_ids()` (đọc `data/image_lists/*.csv`), `AttackDataset` (build qua `mmdet.registry.DATASETS` từ chính config surrogate, lọc+reorder theo danh sách n=300/n=1000 cố định, `serialize_data=False` để giữ `data_list` là list thường). Trả về `(img_id, inputs pixel-space chưa normalize, data_sample có gt_instances)`.
  - `attack/preprocess.py` — lớp glue khả vi: `to_batch()` (gọi `model.data_preprocessor` trực tiếp trên list tensor pixel), `compute_gt_loss()` (loss GT-assisted cho MI-FGSM/DI-FGSM sau này), `extract_features()` (nền cho OSFD), `predict()` (nền cho eval mAP).
- **Verify thật trên GPU** (`scripts/smoke_test_harness.py`, không phải attack method thật, chỉ verify harness):
  1. `AttackDataset('n300')` load đúng 300 ảnh, ảnh đầu tiên (img_id=776) có 4 GT box.
  2. Gradient lan truyền được: tạo noise=0 (`requires_grad=True`), forward qua `compute_gt_loss` → `loss.backward()` → `noise.grad` khác None, `grad_norm=0.0086` (khác 0) — xác nhận đồ thị tính toán không bị đứt qua `data_preprocessor`.
  3. `extract_features`: 5 stage FPN, shape đúng kỳ vọng cho ảnh 1196×800 (sau `Resize(1333,800,keep_ratio)` + pad `divisor=32`).
  4. `predict` trên ảnh sạch: 15 detection, top score 0.989 — model hoạt động đúng (không phải rác).
  5. Load cả 4 model qua `attack/models.py` (`load_all_targets()` cho 3 target + surrogate riêng) — không lỗi.
- Lưu ý kỹ thuật: script cần `sys.path.insert(0, REPO_ROOT)` để import được package `attack/` (không có `PYTHONPATH`/package install nào set up) — đã thêm vào `scripts/smoke_test_harness.py`, các script sau này dùng `attack/` cũng cần dòng này ở đầu.
- **Việc tiếp theo**: viết attack core thật (`attack/base/{ifgsm,mi,di}.py` port từ ref-repo — thuần tensor math, rủi ro thấp; `attack/losses/osfd.py` port OSFD; loss GT-based chuẩn cho MI-FGSM/DI-FGSM dùng `compute_gt_loss` đã có sẵn trong harness; AugTrans viết từ paper). Commit + push toàn bộ `attack/` + `scripts/smoke_test_harness.py` của entry này trước.

---

## 2026-09-22 — Viết attack core: IFGSM/MI/DI engine + OSFD loss port, verify thật trên GPU

Tiếp tục cùng phiên GPU (RTX 3090).

- **Viết `attack/methods/core.py`** — `run_iterative_attack()`: 1 hàm chung thay cho hệ registry+pipeline 4 giai đoạn của ref-repo (không cần abstraction đó vì chỉ có 3 biến thể cố định). `momentum=0` → IFGSM thuần, `momentum>0` → MI-FGSM (chuẩn hoá gradient bằng `mean(abs(.))` trước khi tích lũy, khớp `attack/base/MI.py` gốc), `diversity_fn` → DI-FGSM. Noise re-leaf mỗi step bằng `torch.autograd.grad` (không dùng `.grad`/`zero_grad` thủ công như ref-repo — an toàn hơn, tương đương về mặt toán học, đã verify bằng cách đọc kỹ luồng `@torch.no_grad()` trong `UpdateNoise.__call__` của ref-repo để xác nhận hành vi giống hệt).
- **Viết `attack/methods/diversity.py`** — port `DI.input_diversity`, **có sửa 1 lỗi trong ref-repo**: bản gốc chỉ dùng `imgs.shape[2]` (chiều cao) làm kích thước cho CẢ 2 chiều khi resize/pad → ép ảnh về hình vuông. Với ảnh classification (ImageNet crop vuông sẵn) không sao, nhưng ảnh detection sau `Resize(keep_ratio=True)` không vuông (vd COCO ảnh mẫu 1196×800) — ép vuông sẽ méo tỉ lệ nghiêm trọng, không đúng tinh thần "input diversity" (biến đổi scale nhẹ, không phá cấu trúc ảnh) của paper gốc DI-FGSM (Xie et al., CVPR'19). Đã sửa xử lý H/W độc lập, giống cách `attack/base/RRB.py`'s `adaptive_random_resizing` làm đúng trong cùng ref-repo.
- **Viết `attack/losses/osfd.py`** — port `attack/ours/OSFD.py`: cache feature sạch 1 lần (`torch.no_grad()`), loss = tổng `MSE(k·feat_clean, feat_adv)` trên từng stage. Ghi rõ trong code vì sao ASCENT (tăng loss này) vẫn đúng hướng "đẩy feature ra xa clean" dù nhìn qua tưởng cần descent: tại bước đầu adv≈clean nên `(feat_adv - k·feat_clean) ≈ (1-k)·feat_clean`, với k=3 hướng ascent xấp xỉ đẩy feat_adv theo `-2·feat_clean` — tức ra xa feat_clean thật. Nhờ vậy giữ được quy ước "toàn bộ ascent" thống nhất giữa 3 method, không cần loss riêng kiểu descent.
- **Viết `attack/methods/baselines.py`** — 3 hàm sẵn dùng khớp tên trong idea.md §8: `mi_fgsm_attack`, `di_fgsm_attack`, `osfd_attack`. Hyperparameter mặc định: epsilon=5.0 (=5/255 quy về pixel-space [0,255]), alpha=1.0, momentum=1.0 (MI, theo Dong et al. CVPR'18), prob=0.7/scale=1.1 (DI, theo `attack/base/DI.py` gốc), k=3.0 (OSFD, theo `protocol_lock.md`/`config/default.py` gốc).
  - **Quyết định CHƯA chốt, ghi rõ trong docstring của file**: `osfd_attack()` hiện chỉ ghép loss OSFD với update rule IFGSM thuần (momentum=0, không diversity) — CHƯA gồm RRB (`attack/base/RRB.py`, chưa port) hay MI, trong khi ref-repo's `base.yaml` định nghĩa "OSFD" đầy đủ = MI+RRB+OSFD-loss. Cần quyết định trước khi chạy baseline table chính thức: dùng OSFD-loss đơn lẻ (đơn giản hơn, tách bạch rõ "đóng góp của loss") hay full recipe của paper gốc (đúng fidelity nhưng cần port thêm RRB).
- **Verify thật trên GPU** (`scripts/smoke_test_attacks.py`, ảnh đầu tiên của n300, steps=20, epsilon=5.0):
  - Cả 3 method: `torch.isnan` sạch, noise cuối luôn nằm đúng trong `[-5,5]` (constraint epsilon đúng).
  - **Bài học quan trọng về cách đánh giá**: heuristic ban đầu "đếm tổng số detection giảm" SAI — cả 3 attack đều làm tổng số detection TĂNG mạnh (15→57-100), ban đầu tưởng là bug. Kiểm tra kỹ hơn bằng cách match GT-box (IoU>0.5 + đúng label) thì phát hiện: detection thật KHỚP GT vẫn còn đó nhưng confidence giảm rõ, đồng thời attack sinh rất nhiều false-positive tràn lan ở chỗ khác — đây là hiện tượng THẬT đã biết trong literature tấn công detector (maximize loss RPN/ROI khiến model "ảo giác" object khắp nơi, sập Precision/AP dù recall/count tăng), khớp đúng threat model "untargeted, random-output" của idea.md §3. Đã sửa lại `scripts/smoke_test_attacks.py` dùng đúng tín hiệu (GT-matched confidence) thay vì đếm box.
  - Kết quả (mean confidence drop trên 4 GT box, ảnh mẫu img_id=776): **MI-FGSM: -0.058, DI-FGSM: -0.188, OSFD: -0.563** (suppress hoàn toàn cả 4 GT box về confidence 0). OSFD mạnh nhất — khớp kỳ vọng literature (feature-disruption thường hiệu quả hơn task-loss trực tiếp, đặc biệt ở near white-box).
- **Việc tiếp theo**: (1) quyết định RRB có cần port cho OSFD baseline hay không (xem "chưa chốt" ở trên), (2) viết AugTrans (không có ref code, đọc từ `papers/AugTrans.pdf`), (3) viết harness đánh giá đầy đủ (loop qua n=300, sinh ảnh adv, eval AP thật qua cả 4 model Controlled Panel bằng pycocotools, ghi `results/runs/*/metrics.json` theo idea.md §8-9) — hiện mới test được 1 ảnh, chưa có baseline table thật nào. Commit + push `attack/methods/`, `attack/losses/`, `scripts/smoke_test_attacks.py` của entry này trước.

---

## 2026-09-22 — Port RRB, osfd_attack() lên full recipe (MI+RRB+OSFD-loss)

Tiếp tục cùng phiên GPU (RTX 3090) — trả lời câu hỏi "chưa chốt" ở entry trước.

- Quyết định: **"OSFD" baseline = full recipe của paper gốc** (base_attack=['MI','RRB'] + transfer_attack=OSFD, đúng theo `ref-repo/OSFD-main/config/base.yaml`), không phải OSFD-loss đơn lẻ — vì đó mới là method paper thực sự định nghĩa và report kết quả.
- **Viết `attack/methods/rrb.py`** — port `attack/base/RRB.py`: `random_axis_rotation` (xoay quanh tâm 1 GT box ngẫu nhiên có jitter, hoặc tâm ảnh), `adaptive_random_resizing` (resize+pad theo kích thước 1 GT box ngẫu nhiên, giữ H/W độc lập — không mắc lỗi ép-vuông như DI), `gaussian_blur`, và `rrb_views()` sinh đúng 2 view theo CHAIN của bản gốc (view1=rotate(adv), view2=resize(view1), cả 2 cùng blur). Sửa 1 lỗi nhỏ: điểm xoay fallback bản gốc dùng `[H//2, W//2]` như thể là (x,y) trong khi đúng ra là (cx,cy) — với ảnh không vuông (luôn đúng ở đây) làm lệch điểm xoay; đã sửa đúng thứ tự `(W//2, H//2)`.
- **Tổng quát hoá `attack/methods/core.py`**: đổi tham số `diversity_fn` (1 ảnh -> 1 ảnh) thành `views_fn` (1 ảnh -> list ảnh) để hỗ trợ RRB batch-doubling. Cách tính: gọi `loss_fn` riêng cho từng view rồi CỘNG loss lại trước 1 lần `torch.autograd.grad()` duy nhất — về mặt toán học tương đương ghép batch thật rồi backward 1 lần (autograd cộng gradient qua mọi đường), đơn giản hơn nhiều so với phải batch qua `model.data_preprocessor`, và vẫn khớp đúng định nghĩa B trong `protocol_lock.md` ("concat 2 view rồi backward 1 lần tính B=1").
- **Cập nhật `attack/methods/baselines.py`**: `osfd_attack()` giờ mặc định `use_rrb=True, momentum=1.0` (full recipe), hyperparameter RRB khớp `base.yaml` (theta=7, l_s=10, rho=0.8, s_max=1.1, sigma=6.0) chứ không phải default khiêm tốn hơn của từng class. Vẫn giữ đường tắt `use_rrb=False, momentum=0.0` cho ablation OSFD-loss đơn lẻ sau này nếu cần.
- **Bug phát hiện + sửa khi test thật**: `gt_boxes` lấy từ `data_sample.gt_instances` không tự động cùng device với ảnh — `AttackDataset` trả về `data_sample` nguyên bản từ pipeline (luôn ở CPU), chỉ tensor ảnh mới được code gọi `.to(device)` thủ công. RRB truy cập trực tiếp `gt_boxes` (khác với `attack/preprocess.py` vốn tự đồng bộ device qua `model.data_preprocessor`) nên bị lỗi `RuntimeError: ... cpu and cuda:0`. Đã sửa: `rrb_views()` tự `.to(adv_pixels.device)` cho gt_boxes.
- **Verify lại `scripts/smoke_test_attacks.py`** sau khi sửa: OSFD full recipe vẫn suppress hoàn toàn 4 GT box (confidence → 0.000 cả 4, giống kết quả OSFD-loss-đơn-lẻ trước đó). DI-FGSM vẫn giảm rõ (-0.170). MI-FGSM lần chạy này cho kết quả khác lần trước (delta≈-0.04 thay vì -0.058, tức gần như không đổi) — **không phải bug**: MI-FGSM không có randomness trong code, nhưng ROIAlign backward trên GPU (dùng trong roi_head của Mask R-CNN) vốn KHÔNG deterministic (atomicAdd trong CUDA kernel), sai số nhỏ này bị khuếch đại qua 20 bước cập nhật kiểu `sign()` (hàm bậc thang, nhạy với sai số nhỏ gần 0) → kết quả cuối khác nhau giữa các lần chạy dù cùng input/hyperparameter. Ghi chú quan trọng cho sau này: **đánh giá từng ảnh đơn lẻ có thể nhiễu**, phải luôn nhìn số liệu tổng hợp trên n=300 (khớp đúng lý do idea.md §10 dùng paired bootstrap 95% CI thay vì so sánh 1 ảnh).
- **Việc tiếp theo**: viết AugTrans (không có ref code, đọc từ `papers/AugTrans.pdf`), rồi viết harness đánh giá đầy đủ (loop n=300, sinh + lưu ảnh adv, eval AP thật qua cả 4 model bằng pycocotools, ghi `results/runs/*/metrics.json`). Commit + push `attack/methods/rrb.py`, `attack/methods/core.py`, `attack/methods/baselines.py` đã sửa của entry này trước.

---

## 2026-09-22 — Port AugTrans, phát hiện + sửa 2 bug thật, ghi nhận giới hạn còn mở

Tiếp tục cùng phiên GPU (RTX 3090). Đọc `papers/AugTrans.pdf` (Pandey et al., "AugTrans: Boosting Adversarial Transferability in Object Detection with a Dynamic, Object-Aware Augmentation Pipeline") — **không có ref code trong repo**, port trực tiếp từ Algorithm 1/2, Eq (2)-(6), bảng hyperparameter Section 4.1.

- **Viết `attack/methods/augtrans.py`** — 4 thành phần biến đổi tuần tự mỗi EOT sample (dynamic object-centric rotation với curriculum θ_max(k), multi-box aware resizing dựa trên K_obj=3 GT box lớn nhất, contextual crop/reflective-pad, composite Gaussian+salt-pepper noise) + loss đa thành phần Eq (6) (weighted, có nonlinear scaling γ=0.8 cho cls/objectness). N_EOT=10, K_max=B/10 theo đúng quy ước đã khoá trong `protocol_lock.md`.
- Ghi chú 1 câu mơ hồ trong paper (Section 3.3 mô tả pipeline cố định 4 bước, nhưng phần "Hyperparameter Configuration" lại viết "apply 2 sequential transformations, randomly sampling" — mâu thuẫn) — chọn theo Section 3.3 + Algorithm 2 (có pseudocode cụ thể) vì nhất quán hơn.
- **Bug #1 phát hiện qua verify thật (fairness alpha)**: `alpha=η=0.0004` (giá trị paper) chỉ cho `K_max*alpha` tối đa 0.51-2.04 ở `B∈{50,200}` đã khoá (paper dùng K_max=160, mình chỉ có K_max=5-20 do N_EOT=10) — noise KHÔNG BAO GIỜ chạm epsilon=5, khiến AugTrans thua thiệt giả tạo do quy đổi đơn vị B, không phải do method yếu. **Đã hỏi ý kiến, quyết định**: `alpha` tự tính = `2*epsilon/K_max` (margin 2x) thay vì dùng thẳng giá trị paper — đảm bảo AugTrans cũng tận dụng hết budget perturbation như 3 baseline kia.
- **Bug #2 phát hiện qua verify thật (box không co-transform)**: sau khi sửa alpha, `realized L_inf` đã đúng bằng epsilon nhưng confidence trên GT-matched box KHÔNG giảm (thậm chí tăng) — nghi ngờ do rotation/resize làm object DỊCH CHUYỂN trong ảnh nhưng `data_sample.gt_instances.bboxes` vẫn giữ toạ độ GỐC khi tính `model.loss()`, khiến loss tính sai vị trí hoàn toàn. Xác nhận đúng bằng thực nghiệm (đo điểm mốc qua `torchvision.transforms.functional.rotate` để suy ra đúng công thức xoay góc — quy ước: điểm lệch (dx,dy) quanh tâm chuyển thành `(dx·cosθ+dy·sinθ, -dx·sinθ+dy·cosθ)`, khác cả 2 chiều "chuẩn toán học y-lên" lẫn "trực giác visual y-xuống" nếu suy đoán không kiểm chứng). Đã sửa:
  - Viết `attack/methods/box_transforms.py` (`rotate_boxes`, `scale_shift_boxes`) dùng chung.
  - `attack/methods/augtrans.py`: `_object_centric_rotate`/`_resize_crop_pad` giờ trả về cả box đã biến đổi; `augtrans_transform` build `DetDataSample` MỚI (deepcopy + gán box mới) cho mỗi EOT view.
  - Tổng quát hoá `attack/methods/core.py`: `views_fn` đổi từ trả `List[Tensor]` sang `List[Tuple[Tensor, DetDataSample]]` — mỗi view mang data_sample riêng.
  - **Áp dụng luôn fix này cho DI-FGSM** (`attack/methods/diversity.py`, hàm mới `input_diversity_with_boxes`) — DI-FGSM ghép `input_diversity` (resize+pad) với `compute_gt_loss` (GT-based) nên mắc CÙNG LOẠI bug, dù không tự nhận ra khi viết ban đầu (ref-repo gốc không có bug này vì DI trong ref-repo chỉ từng ghép với loss OSFD, không phụ thuộc GT). Sau fix, DI-FGSM cải thiện rõ rệt (confidence drop trên ảnh mẫu: -0.046→-0.325), xác nhận bug này có thật và có ảnh hưởng đáng kể. RRB (dùng cho OSFD) KHÔNG cần fix vì loss OSFD (feature-disruption) không phụ thuộc GT.
- **Bug #3 phát hiện qua verify thật (thiếu loss_mask)**: sau khi sửa bug #2, AugTrans vẫn yếu. Debug bằng cách in từng thành phần loss trên ảnh sạch: `loss_cls=0.52, loss_bbox=0.43, loss_rpn_*≈0.001-0.03, loss_mask=1.68` — `loss_mask` chiếm ~62% tổng loss nhưng KHÔNG có trong Eq (6) của paper. Lý do: paper AugTrans dùng **Faster R-CNN** làm source model (Section 3.5.1), không có mask head, nên Eq (6) 4 số hạng đã là toàn bộ task loss của họ — nhưng surrogate của dự án này bị khoá là **Mask R-CNN** (`protocol_lock.md`, để giữ kiến trúc nhất quán Controlled Panel), có thêm mask loss lớn nhất mà Eq (6) port nguyên văn bỏ sót hoàn toàn. **Đã hỏi ý kiến, quyết định**: thêm số hạng thứ 5 `alpha_mask * loss_mask` (alpha_mask=1.0, không có exponent — cùng kiểu linear như loss_bbox/loss_rpn_bbox) vào `augtrans_loss()`, ghi rõ đây là ĐIỀU CHỈNH so với paper gốc (không phải Eq 6 nguyên văn), lý do và bằng chứng đầy đủ trong docstring.
- **Giới hạn còn mở, CHƯA giải quyết được**: sau khi sửa cả 3 vấn đề trên, AugTrans vẫn cho kết quả suppress YẾU HƠN rõ rệt so với MI-FGSM/DI-FGSM/OSFD trên ảnh mẫu (img_id=776) — kể cả khi test ở đúng K_max=160 (khớp scale paper gốc dùng, budget_B=1600) thay vì K_max=5-20 của protocol B∈{50,200}. Thử cô lập từng thành phần biến đổi (chỉ rotate / chỉ resize / chỉ noise / tổ hợp) cho kết quả KHÔNG NHẤT QUÁN (vd "chỉ noise" suppress khá tốt nhưng "full pipeline" lại yếu hơn cả — ngược trực giác) — dấu hiệu nhiễu thống kê cao trên phép đo 1-ảnh-1-lần-chạy hơn là 1 bug cụ thể còn sót (đã biết trước: ROIAlign backward không deterministic trên GPU, xem entry "Port RRB" — càng rõ hơn với AugTrans vì EOT lấy trung bình qua nhiều transform NGẪU NHIÊN mỗi step, biến thiên giữa các lần chạy vốn đã cao hơn 3 method kia).
  - **Quyết định (đã hỏi ý kiến)**: KHÔNG debug sâu thêm dựa trên 1 ảnh — khớp đúng lý do idea.md dùng paired bootstrap 95% CI trên n=300 thay vì tin số liệu từng ảnh đơn lẻ (idea.md §10). Code AugTrans coi như ĐÃ XONG về mặt kỹ thuật (không crash, gradient chảy đúng, epsilon đúng, 3 bug thật đã tìm và sửa có bằng chứng rõ ràng) nhưng **hiệu quả tương đối so với 3 baseline kia CHƯA kết luận được cho tới khi chạy thật trên n=300** và nhìn số liệu tổng hợp.
  - Nếu khi chạy Baseline-First Stage thật (n=300) mà AugTrans vẫn yếu hơn hẳn 1 cách nhất quán (không phải nhiễu ảnh đơn lẻ nữa), cần quay lại điều tra thêm — nghi vấn hàng đầu tiếp theo (chưa kiểm chứng): công thức content-adaptive resize (Eq 3-4) có thể tạo scale factor khá lớn (lên tới ~1.5-1.65x) với ảnh có object to, kết hợp crop ngẫu nhiên có thể đẩy phần lớn object ra ngoài vùng crop — cần trực quan hoá vài EOT view thật (lưu ảnh ra xem) để kiểm tra thay vì chỉ suy luận từ số liệu.

---

## 2026-09-23 — Sửa bug GT lệch khung ảnh (LoadAnnotations sau Resize) + co-transform GT mask

Phiên GPU mới (RTX 3090). Phát hiện khi đọc lại code, verify thật trên GPU.

- **Bug nghiêm trọng (ảnh hưởng MI-FGSM, DI-FGSM, AugTrans — mọi loss dựa trên GT):** test pipeline của config mmdet đặt `LoadAnnotations` **sau** `Resize` (vì lúc eval GT chỉ đọc từ file annotation, ở tọa độ gốc). `AttackDataset` dùng nguyên pipeline này nên `inputs` đã resize (vd 1196×800) nhưng `gt_instances.bboxes/masks` vẫn ở tọa độ ảnh **gốc** (vd 640×428, scale ~1.87) — box chỉ phủ ~góc trên-trái ảnh. Mọi `model.loss()` từ trước tới nay tính trên vị trí sai. OSFD không bị trực tiếp (loss feature), RRB chỉ lệch nhẹ (tâm xoay/scale theo box sai tỉ lệ).
  - Fix: `attack/data.py::_gt_before_resize()` đưa `LoadAnnotations` lên trước `Resize` (Resize tự co-transform box+mask) và bật `poly2mask=True` (mask dạng BitmapMasks, xem dưới). Eval không đổi: `predict(rescale=True)` vẫn dùng `scale_factor` về tọa độ gốc.
  - Hệ quả: các kết luận 1-ảnh trong entry trước ("MI-FGSM gần như không giảm confidence", "AugTrans yếu nghi do nhiễu") được đo với GT lệch — **không còn giá trị**.
- **GT mask chưa co-transform trong DI/AugTrans** (entry trước chỉ co-transform box): `loss_mask` dùng `gt_masks`, sau xoay/resize view thì mask vẫn ở vị trí cũ. Fix: `box_transforms.get_gt()/with_gt()` — mask lấy dạng tensor [N,H,W], đi qua CHÍNH phép biến đổi tensor của ảnh (rotate/interpolate/crop), riêng pad dùng 0 thay vì reflect (vùng reflect không có GT box tương ứng); threshold 0.5 rồi gói lại `BitmapMasks`.
- **Verify thật** (`scripts/smoke_test_attacks.py`, viết lại: 3 ảnh đầu n300, steps=20, eps=5, AugTrans B=50; so khớp trong khung ảnh đã resize, `predict(rescale=False)`):
  - GT frame: box max khớp kích thước ảnh resize, IoU(bbox của mask, box) 0.93–1.00, detection sạch khớp GT 75–100%.
  - Co-transform: IoU(bbox mask, box) sau biến đổi DI 0.93–1.00, AugTrans 0.84–0.92 (thấp hơn do box xoay lấy AABB — lỏng hơn mask, đúng kỳ vọng).
  - Mean GT-confidence drop (white-box surrogate): MI-FGSM **+0.716**, DI-FGSM **+0.716**, OSFD **+0.716** (cả 3 suppress 100% GT cả 3 ảnh), AugTrans **+0.503** (+0.110/+0.455/+0.943 — yếu hơn ở white-box với K_max=5 là dự kiến, EOT làm mịn gradient; quan trọng là transfer, chưa đo).
  - A/B tái tạo bug cũ (GT về tọa độ gốc): MI-FGSM **−0.099** (confidence còn TĂNG), DI-FGSM +0.163 — xác nhận bug có thật và ảnh hưởng rất lớn.
  - GT loss ảnh sạch (img 776): 0.79 (trước ~2.7 khi GT lệch). Loss breakdown trung bình 20 ảnh sạch: cls 0.20, bbox 0.24, rpn_cls 0.03, rpn_bbox 0.06, mask 0.29 (~36%, không phải ~62% như đo khi GT lệch) — lý do thêm `loss_mask` vào AugTrans vẫn đứng (thành phần lớn nhất), đã sửa số liệu trong docstring.
- **Còn mở:**
  1. Định nghĩa B chưa nhất quán: RRB 2 view/step tính B=1, AugTrans 10 view/step (cộng loss, 1 lần `autograd.grad`) lại tính B=10. `protocol_lock.md` ("B = số lần `.backward()`") và code AugTrans đang theo 2 cách hiểu khác nhau — cần chốt đơn vị (số backward vs số view forward-backward qua surrogate) trước khi chạy baseline table.
  2. Chưa có harness eval AP n=300 × 4 model.

---

## 2026-09-23 — Quick transfer check sơ bộ n=30 (không phải baseline table chính thức)

`scripts/quick_transfer_check.py 30 50`: 30 ảnh đầu n300, surrogate R50, eps=5, B=50 (MI/DI/OSFD 50 step; AugTrans K_max=5×N_EOT=10 — định nghĩa B vẫn chưa chốt), bbox AP pycocotools giới hạn 30 ảnh. Kết quả: `results/quick_transfer/n30_B50_20260923_055238.json`.

| Relative AP drop | R50 (white-box) | R101 | ConvNeXt-T | Swin-T |
|---|---|---|---|---|
| MI-FGSM | 100.0% | 81.2% | 56.4% | 38.6% |
| DI-FGSM | 100.0% | 92.2% | 73.6% | 55.2% |
| OSFD | 99.9% | 98.1% | 84.5% | 78.9% |
| AugTrans (tự cài lại) | 89.0% | 65.8% | 47.5% | 44.2% |

- Pattern same-family > cross-CNN > CNN→Transformer xuất hiện ở cả 4 method (AugTrans: ConvNeXt≈Swin). OSFD KHÔNG bão hòa ở cross-family (~15-20 điểm gap) — còn chỗ cho RQ1/RQ3.
- AugTrans (bản tự cài lại) yếu nhất, dưới cả DI-FGSM ở mọi target — trái với claim của paper gốc; cùng với việc code gốc không tồn tại (repo công bố trong paper trả 404 ngày 2026-09-23) → rủi ro cài đặt sai cao.
- Chỉ là tín hiệu sơ bộ: n=30, chưa có CI, không dùng để tune bất kỳ hyperparameter nào.

---

## 2026-09-23 — Bỏ AugTrans khỏi plan, rà MI/DI, phát hiện vấn đề đánh giá "tensor vs ảnh thật"

- **AugTrans tạm bỏ khỏi plan** (quyết định của user): không có code chính thức (repo công bố trong paper `github.com/sudhirpandey243/LLM-Model` trả 404), mô tả mâu thuẫn, bản tự cài lại lệch paper nhiều chỗ và yếu hơn DI-FGSM ở quick check n=30. Đã sửa idea.md §8/§12, protocol_lock.md (mục B), smoke test; `attack/methods/augtrans.py` giữ lại, có ghi chú. OSFD là strong baseline chính — benchmark 2602.16494 chỉ chọn OSFD/EBAD/CAA/PhantomSponges, trong đó chỉ OSFD khớp threat model (1 surrogate, image-specific, không query). **Còn mở**: tiêu chí idea.md §10.4 ("≥2 strong baselines") — đã đánh dấu, chưa sửa.
- **Rà MI-FGSM/DI-FGSM — không thấy bug thuật toán**: update MI khớp Dong et al. + `ref-repo/.../MI.py` (grad chuẩn hoá L1-mean, μ=1, sign step, clip ε); DI khớp `DI.py` (resize+pad, sửa ép vuông, co-transform box+mask đã verify IoU 0.93–1.00); loss GT = tổng 5 loss (rpn_cls, rpn_bbox, cls, bbox, mask) với GT đúng khung; gradient qua data_preprocessor OK, preprocessor không sửa data_sample gốc (cast_data tạo bản sao). Loss có ngẫu nhiên do RandomSampler (RPN 256 anchor, RCNN 512 RoI) của train_cfg — chuẩn với attack dùng train loss, cần seed để tái lập.
  - **Lựa chọn chưa chốt (không phải bug)**: (a) DI prob=0.7 (default class trong ref) vs 1.0 (`ref-repo/config/default.py`); (b) DI hiện KHÔNG có momentum (DI²-FGSM thuần) trong khi OSFD có MI — cân nhắc M-DI²-FGSM (MI+DI) cho so sánh công bằng hơn; (c) α=1 pixel cho mọi method (ref default), khác α=ε/T của MI-FGSM gốc — nhất quán giữa các method nên chấp nhận được.
- **Vấn đề đánh giá lớn (ảnh hưởng MỌI method)**: harness đang đưa adv dạng tensor float ở khung ảnh ĐÃ resize (vd 1196×800, upsample ~1.87× từ 640×428) thẳng vào target. Ảnh adversarial thực tế phải là ảnh ở kích thước gốc, uint8, rồi mỗi target tự resize (đúng lý do benchmark loại AFOG). Thử nghiệm: `to_image_space()` trong `scripts/quick_transfer_check.py` (thu delta về ori_shape bằng INTER_AREA, cộng vào ảnh gốc uint8, làm tròn, rồi Resize lại như pipeline; round-trip ảnh sạch khớp tuyệt đối). Nhiễu ngẫu nhiên ±5 ở khung upsample chỉ còn |δ| trung bình ~1.2–1.7 sau khi về ảnh thật.
  - Kết quả n=5, B=50, eps=5 (`results/quick_transfer/n5_B50_20260923_060441.json`), relative AP drop R50/R101/ConvNeXt-T/Swin-T:
    - tensor: MI 100/74.8/47.9/45.3 · DI 99.9/90.5/73.6/57.1 · OSFD 100/100/89.1/88.3
    - **ảnh thật**: MI 78.5/42.1/26.7/21.2 · DI 85.6/43.3/40.0/33.3 · OSFD 100/78.3/55.7/56.6
  - → Đánh giá dạng tensor thổi phồng transfer rất nhiều (OSFD cross-family ~89% → ~56%). Số n=30 ở entry trước (đánh giá dạng tensor) phải coi là KHÔNG hợp lệ cho kết luận. Gap cùng họ > khác họ vẫn thấy ở cả 2 cách (n=5, rất nhiễu).
- **Việc tiếp theo (đề xuất, chưa làm)**: sửa harness để tối ưu perturbation ngay ở không gian ảnh gốc (δ ở ori_shape, resize khả vi bên trong vòng lặp lên kích thước model) + đánh giá luôn qua ảnh uint8 — như vậy perturbation không mất khi lưu ảnh; rồi chạy lại quick check n=30. Sau đó chốt: đơn vị B (OSFD/RRB 2 view/step), DI có momentum không, §10.4.

---

## 2026-09-23 — Sửa harness: tối ưu δ ở không gian ảnh gốc + OSFD dùng feature backbone; quick check n=30 lại

Phiên GPU (RTX 3090). Rà lại toàn bộ harness, so từng dòng với `ref-repo/OSFD-main`, verify bằng số trên GPU.

- **Bug 1 — δ tối ưu ở khung đã resize (ảnh hưởng mọi method):** trước đây δ nằm trên tensor đã upsample (~1.87×, vd 640×428 → 1196×800); ảnh adversarial thật (cỡ gốc, uint8) mất phần lớn δ — đo: nhiễu sign ±5 chỉ còn mean|δ| **1.79** sau INTER_AREA về cỡ gốc (mất ~64%). Số `@img` n=5 ở entry trước là CẬN DƯỚI (attack không tối ưu cho ảnh thật), số tensor là thổi phồng — cả hai không dùng được.
  - Fix: `AttackDataset` trả thêm `orig_inputs` (ảnh gốc BGR, `mmcv.imread`); `core.run_iterative_attack` nhận ảnh gốc, δ cùng shape ảnh gốc, mỗi step `resize_to_model(clamp(orig+δ))` (F.interpolate bilinear, align_corners=False, khả vi) rồi mới vào surrogate. Đánh giá: `to_adv_image` = round/clamp uint8 cỡ gốc → `pipeline_resize` (mmcv.imresize cv2 — ĐÚNG Resize của test pipeline) → target.
  - Đã đo: resize torch vs cv2 pipeline lệch tối đa 1 mức xám, trung bình ~0.12 (fixed-point cv2) trên 5 ảnh — gradient đúng không gian. `pipeline_resize(orig) == inputs` tuyệt đối (assert trong smoke test + quick check). L_inf thực tế trên ảnh uint8 = 5.000.
- **Bug 2 — OSFD lấy feature sai tầng:** bản gốc dùng `model.backbone(...)` (ResNet C2–C5, 256/512/1024/2048 kênh — `pipelines.py:50`, `TransferAttack.py:18`), bản port dùng `model.extract_feat` (backbone + FPN, 5 mức 256 kênh). Fix: `preprocess.extract_features` trả `model.backbone(...)`.
- Ghi chú đính chính: `['MI','RRB']` của OSFD nằm trong `config/attack_*.yaml` (config tác giả chạy paper), không phải `base.yaml` (`base.yaml` là `['IFGSM','MI','DI','RRB']`) — code chọn đúng, chỉ entry cũ ghi sai nguồn.
- Lệch nhỏ còn lại so với ref (chưa sửa, ghi nhận): noise init = 0 (ref: randint [−2,2]); DI prob 0.7 (ref default.py: 1.0).
- Smoke test (3 ảnh, 20 step, đánh giá qua ảnh uint8): GT-conf drop surrogate MI +0.695, DI +0.683, OSFD +0.713; A/B GT lệch cũ vẫn ≈0 (đúng kỳ vọng).
- **Quick check n=30, B=50, eps=5, đánh giá ảnh thật** (`results/quick_transfer/n30_B50_20260923_122751.json`), relative AP drop R50 / R101 / ConvNeXt-T / Swin-T:
  - MI-FGSM: 100.0 / 76.6 / 57.2 / 42.8
  - DI-FGSM: 100.0 / 88.4 / 70.2 / 56.2
  - OSFD: 99.4 / 95.3 / 79.7 / 77.7
  - Pattern cùng họ > khác họ giữ nguyên ở cả 3 method; gap OSFD R101→cross-family ~16–18 điểm. Vẫn chỉ là tín hiệu sơ bộ (n=30, chưa CI), không tune gì dựa trên số này.
  - Runtime ~3–3.5 s/ảnh/method ở B=50.
- Cài thêm `tmux` vào apt deps của `scripts/setup_env.sh` (chạy n=300/1000 không bị ngắt khi mất kết nối).
- **Việc tiếp theo:** chốt đơn vị B (RRB 2 view/step), DI có momentum không + prob, §10.4; viết harness eval n=300 × 4 model ghi `results/runs/*/metrics.json` (có paired bootstrap CI).

---

## 2026-09-23 — Đường bão hòa OSFD theo B (n=30)

`scripts/budget_sweep.py 30 200`: chạy OSFD 1 lần B=200, chụp noise tại các mốc B (callback `on_step` mới trong `core.run_iterative_attack`), đánh giá ảnh thật trên 4 model. Kết quả: `results/budget_sweep/osfd_n30_Bmax200_20260923_124034.json`. Relative AP drop (%) R50 / R101 / ConvNeXt-T / Swin-T:

| B | R50 | R101 | ConvNeXt-T | Swin-T | cross-avg |
|---|---|---|---|---|---|
| 10 | 92.2 | 74.6 | 52.7 | 52.0 | 52.3 |
| 20 | 97.9 | 90.2 | 72.8 | 67.8 | 70.3 |
| 30 | 99.2 | 92.6 | 78.7 | 69.0 | 73.9 |
| 50 | 99.4 | 95.3 | 79.6 | 78.1 | 78.8 |
| 100 | 99.6 | 97.0 | 83.5 | 79.3 | 81.4 |
| 200 | 99.8 | 97.5 | 85.2 | 80.1 | 82.7 |

- OSFD gần bão hòa từ B≈50: B50→B200 chỉ thêm ~4 điểm cross-avg (~2 điểm R101), mức chênh cỡ nhiễu n=30. Mốc B=50 khớp quick check trước (79.7/77.7) — sanity OK.
- Gap R101 − cross-family ổn định ~15–17 điểm ở mọi B ≥ 20 → thêm budget KHÔNG thu hẹp gap (tốt cho RQ1).
- Runtime OSFD B=200: 12.1 s/ảnh → n=300 ≈ 1 h/method.
- Chưa đổi bộ B đã khóa ({50, 200}) — chỉ là số liệu lập kế hoạch; nếu đổi phải ghi quyết định vào protocol_lock.md.

---

## 2026-09-23 — Chốt các điểm mở trước baseline table

Quyết định của user:
- **Budget:** B=50 là budget chính cho Confirmation Stage @300; B=200 chạy 1 lần cho bảng cuối (dựa trên sweep bão hòa ở entry trước).
- **Đơn vị B = số backward** (OSFD/RRB 2 view/step vẫn tính B=1, đúng recipe paper); bảng kết quả báo cáo thêm runtime + số view forward.
- **DI-FGSM = M-DI²-FGSM** (MI μ=1.0 + DI), p=1.0 theo ref `config/default.py` — `di_fgsm_attack` đã sửa default (momentum=1.0, prob=1.0).
- **idea.md §10.4** định nghĩa lại: gap phải xuất hiện ở cả OSFD và M-DI²-FGSM; §10.5 giữ nguyên.
- Đã cập nhật idea.md §8/§10, protocol_lock.md (mục B + bảng hyperparameter baseline), CLAUDE.md (giai đoạn hiện tại).
- Việc tiếp theo: harness eval n=300 × 4 model (`results/runs/*/metrics.json`, paired bootstrap 95% CI), chạy baseline table B=50.

---

## 2026-09-23 — Baseline table chính thức n=300, B=50, ε=5 (Controlled Panel)

Harness mới: `attack/evaluation.py` (COCO AP có trọng số ảnh cho paired bootstrap — tự kiểm khớp pycocotools |Δ|<1e-9 mỗi lần chạy) + `scripts/run_baselines.py` (attack → PNG uint8 cỡ gốc ở `artifacts/runs/<run>/adv/`, 527 MB, gitignored, dùng lại được cho Generalization Panel; eval đọc PNG từ đĩa; resume được). Kết quả: `results/runs/n300_B50_eps5/metrics.json` (+ `attack_stats.jsonl`). Bootstrap 1000 mẫu, resample ảnh, paired giữa mọi điều kiện/model.

Relative AP drop % [95% CI], R50 (white-box) / R101 / ConvNeXt-T / Swin-T:
- MI-FGSM: 99.9 / 77.3 [71.6, 79.7] / 54.0 [47.2, 56.0] / 45.1 [39.3, 47.9]
- M-DI²-FGSM: 99.7 / 93.0 [89.7, 94.7] / 78.0 [72.7, 80.3] / 67.1 [61.5, 69.7]
- OSFD: 98.9 / 92.4 [88.4, 95.4] / 80.4 [75.7, 83.6] / 78.7 [73.9, 81.2]

TransferGap (drop R101 − drop target), điểm [95% CI], p(gap≤0) = 0 ở mọi ô:
- MI-FGSM: →ConvNeXt 23.3 [20.6, 28.1], →Swin 32.1 [28.3, 36.4], →avg 27.7
- M-DI²-FGSM: →ConvNeXt 15.0 [12.5, 18.8], →Swin 25.8 [22.4, 30.8], →avg 20.4
- OSFD: →ConvNeXt 12.0 [9.4, 15.9], →Swin 13.8 [11.0, 18.1], →avg 12.9 [10.2, 16.7]

CrossAvg: MI 49.6, M-DI² 72.5, OSFD 79.5. OSFD − M-DI² = +7.0 [3.9, 11.3]. Clean AP trên n300: 45.0 / 46.2 / 51.0 / 49.8. L_inf = 5 (uint8) mọi ảnh; runtime ~3.1–3.6 s/ảnh.

Đối chiếu Hypothesis Pass (idea.md §10): (1) same-family > cross-family ở cả 3 method, cả 2 target ✓; (3) CI của gap > 0 ở mọi ô ✓; (4) có ở OSFD và M-DI²-FGSM ✓; (5) không chỉ MI ✓; (2) "effect size đủ meaningful" chưa định lượng trong idea.md — gap nhỏ nhất 12.0 điểm (OSFD→ConvNeXt). **Chưa tuyên bố PASS chính thức** — chờ user xác nhận tiêu chí (2).

Quan sát (cho Mechanism Stage, chưa kết luận):
- Gap co lại khi attack mạnh hơn (MI 27.7 → M-DI² 20.4 → OSFD 12.9) — một phần có thể do trần: R101 đã ~92–93% với M-DI²/OSFD nên gap (hiệu số drop) bị nén; cân nhắc thêm chỉ số ít bị trần (vd tỉ số AP còn lại adv/clean, hoặc ε=3) nếu cần.
- Với MI/M-DI², Swin khó hơn ConvNeXt rõ (~9–11 điểm); với OSFD hai target khác họ gần bằng nhau (80.4 vs 78.7).
- Lưu ý thống kê: bootstrap mean của AP lệch lên ~1.2 điểm so với điểm gốc (category hiếm rơi khỏi mẫu → trung bình category đổi), CI percentile hơi lệch phải; với gap (hiệu số, paired) lệch chỉ +0.2–0.8 — không đổi kết luận. Nếu cần chặt hơn: BCa hoặc cố định tập category.
- Chưa tính: ASR, APloc, CSR, LPIPS (cần định nghĩa/cài `lpips`).

---

## 2026-09-23 — Chốt tiêu chí 2 của Hypothesis Pass → PASS @300

- **Quyết định (user):** tiêu chí 2 idea.md §10 ("effect size đủ meaningful") = **cận dưới 95% CI paired bootstrap của gap ≥ 5 điểm relative AP drop**, cho từng target khác họ (ConvNeXt-T, Swin-T), ở cả OSFD và M-DI²-FGSM.
- **Minh bạch:** ngưỡng chốt SAU khi đã thấy bảng n=300 (idea.md trước đó không định lượng). Đã chọn phương án chặt hơn "CI > 0" (tiêu chí 3) và chặt hơn "điểm ước lượng ≥ 5"; phương án chặt nhất đưa ra (cận dưới ≥ 10) sẽ fail đúng 1 ô (OSFD→ConvNeXt 9.4). Phải nêu điều này nếu báo cáo tiêu chí trong paper.
- Đối chiếu `results/runs/n300_B50_eps5/metrics.json`: cận dưới CI gap — M-DI²-FGSM →ConvNeXt 12.5, →Swin 22.4; OSFD →ConvNeXt 9.4, →Swin 11.0 — đều ≥ 5.
- **Hypothesis Pass @300: PASS** (tiêu chí 1–5 đều đạt, xem entry baseline table). Được phép chuyển sang Mechanism Stage (idea.md §11).

---

## 2026-09-23 — Duyệt plan Mechanism Stage

- `docs/mechanism_plan.md` APPROVED sau chỉnh của user: thứ tự chạy **A2 → A1 → A3 → A4 → A5** (A2 trước vì ảnh adv B=50 chỉ còn trên máy); A3 (can thiệp) là thí nghiệm quyết định cho stage-aware, nặng ký hơn A2 (quan sát); tiêu chí A2 không khóa dấu tuyệt đối D_k(R101) > D_k(cross) mà khóa "tách nhau theo stage, nhất quán với thứ tự transfer, reproduce trên OSFD + M-DI²"; ceiling: chưa chạy ε=3, báo cáo song song chỉ số ít bão hòa (retained confidence, adv AP/clean AP), chỉ chạy ε=3 nếu kết luận đổi theo chỉ số.

---

## 2026-09-23 — Mechanism A2: feature similarity + lan truyền distortion (n=300)

`scripts/mech_a2_features.py` (+ `attack/mechanism.py`: linear CKA — tự kiểm self=1, bất biến trực giao/scale, độc lập≈0.04; chỉ số transfer per-ảnh). Kết quả: `results/mechanism/a2_features.json`, `a2_per_image.json` (per-ảnh transfer dùng lại cho A1/A3/A4). Ảnh adv = đúng PNG của run `n300_B50_eps5`.

**(a) CKA(R50, target), ảnh sạch, stage 1→4:** R101 0.88 / 0.85 / 0.81 / 0.86; ConvNeXt 0.46 / 0.58 / 0.57 / 0.75; Swin 0.66 / 0.68 / 0.63 / 0.60 (CI rộng ~±0.005). Cùng họ giống R50 hơn hẳn ở MỌI stage. Thứ tự ConvNeXt vs Swin đảo theo độ sâu: stage 1 Swin > ConvNeXt, stage 4 ConvNeXt > Swin — khớp thứ tự transfer (ConvNeXt dễ hơn Swin với MI/M-DI²).

**(b) Distortion D_k (mean), stage 1→4:**
- R101 khuếch đại distortion theo độ sâu (OSFD 0.22→0.70→1.18→1.31; M-DI² 0.22→0.60→0.85→0.94).
- ConvNeXt bị distortion LỚN HƠN R101 ở stage 1 (0.46–0.48) rồi giảm ở stage 2 (0.31–0.41), tăng lại chậm; Swin tăng tới stage 3 rồi giảm ở stage 4.
- Separation D_k(R101) − D_k(cross): **âm ở stage 1** (−0.12 đến −0.26), **dương từ stage 2 và lớn dần** (stage 4: 0.37–0.59); separation_k − separation_1 có CI > 0 ở mọi ô, cả 3 method → **k* = 2, reproduce trên OSFD và M-DI²**.
- Tương quan per-ảnh D_k với transfer: Spearman(D_k, retained) âm, CI loại trừ 0 ở mọi (method, target, stage) (|ρ| 0.12–0.58, mạnh nhất stage 4 ở target khác họ). Suppression cho cùng kết luận (dấu ngược, CI loại 0 mọi ô) → **không kích hoạt ceiling check ε=3**.

**Đối chiếu tiêu chí A2 (khóa trong mechanism_plan.md):**
- ✓ Tách theo stage: k* = 2, trước đó (stage 1) không tách theo chiều R101 (thậm chí ngược), CI loại 0, reproduce OSFD + M-DI².
- ✓ Nhất quán thứ tự theo TARGET ở stage 4 (separation Swin > ConvNeXt, khớp Swin khó hơn); ✗ ở stage 2–3 (ConvNeXt > Swin, ngược thứ tự transfer).
- ✗ Nhất quán theo METHOD với hiệu số tuyệt đối (chỉ số đã khóa): gap MI 27.7 > M-DI² 20.4 > OSFD 12.9 nhưng separation stage 4 OSFD LỚN NHẤT (0.48/0.59) > M-DI² (0.37/0.49) ≈ MI (0.37/0.46) — hiệu số tuyệt đối lớn lên theo độ mạnh attack.
- **Kết luận A2 theo tiêu chí khóa: ĐẠT MỘT PHẦN — chưa đủ điều kiện "có bằng chứng".** Có bằng chứng quan sát rõ rằng divergence nảy sinh theo độ sâu (từ stage 2: R101 khuếch đại distortion, target khác họ suy giảm/không khuếch đại), nhưng điều kiện nhất quán độ lớn theo method không đạt với chỉ số đã khóa.
- **Exploratory (post-hoc, KHÔNG dùng để kết luận):** tỉ số D_4(cross)/D_4(R101) — ConvNeXt MI 0.54 / M-DI² 0.60 / OSFD 0.63; Swin 0.44 / 0.48 / 0.55 — thứ tự khớp gap theo method. Chỉ số chuẩn hóa scale này chỉ được nghĩ ra sau khi thấy dữ liệu → ghi nhận làm giả thuyết cho A3, không đổi tiêu chí A2.
- Theo quy tắc, A3 (can thiệp) là thí nghiệm quyết định cho stage-aware; A2 không chặn.

---

## 2026-09-23 — Mechanism A1: input-gradient alignment (n=300)

`scripts/mech_a1_gradients.py` (tiêu chí vận hành chốt + commit TRƯỚC khi chạy, `27ccccd`: phải đạt với CẢ cosine VÀ sign agreement). Kết quả: `results/mechanism/a1_gradients.json` (+ `_per_image.json`). Gradient ∇ₓ L_task ở không gian ảnh gốc, ảnh sạch.

- **Alignment với R50** (mean [CI]): cosine R101 0.201 / ConvNeXt 0.142 / Swin 0.109 (trần nhiễu R50 khác seed 0.973); sign agreement 0.5205 / 0.5097 / 0.5066 (trần 0.868, ngẫu nhiên 0.5).
- **R101 − cross, CI > 0 ở cả 2 chỉ số, cả 2 target** (cosine +0.059 / +0.092; sign +0.011 / +0.014). Thứ tự R101 > ConvNeXt > Swin khớp đúng thứ tự transfer của MI/M-DI².
- **Tương quan per-ảnh alignment ↔ suppression (target khác họ):**
  - sign agreement: dương, CI > 0 ở cả MI, M-DI², OSFD (ρ 0.14–0.25) ✓.
  - cosine: M-DI² chỉ Swin (0.136) ✓; **OSFD ngược chiều**: ConvNeXt −0.212 [−0.320, −0.091], Swin −0.083 (CI qua 0) ✗.
- Retained confidence cho cùng kết luận với suppression (dấu ngược, cùng ô đạt/trượt ở mức tiêu chí) → không kích hoạt ceiling check.
- **Kết luận theo tiêu chí khóa: A1 KHÔNG ĐẠT** (cosine trượt điều kiện tương quan per-ảnh với OSFD). Đạt một phần: khác biệt alignment cùng họ vs khác họ rõ và đúng thứ tự ở cả 2 chỉ số; liên hệ per-ảnh giữ với sign agreement ở cả 3 method.
- Ghi chú diễn giải (không đổi kết luận): OSFD tối ưu loss feature chứ không phải task loss, nên alignment của task-gradient không phải cơ chế trực tiếp của nó; cosine bị chi phối bởi vài pixel gradient lớn, sign agreement đếm đều mọi pixel. Mức sign agreement chỉ trên ngẫu nhiên 1–2 điểm % dù transfer chênh lớn → gợi ý khác biệt không nằm ở hướng gradient pixel mà ở cách mạng khuếch đại/triệt tiêu nhiễu theo độ sâu (khớp A2).
- Đang chạy mở rộng quỹ đạo (`--trajectory`, alignment tại step 10/25/50 của M-DI² và OSFD, mô tả, không thuộc tiêu chí).

---

## 2026-09-23 — Chuẩn bị Generalization Panel + chốt quy tắc khả thi (trước khi có kết quả)

- User quyết định: chạy Generalization Panel ngay để đánh giá chỗ trống của OSFD trước khi đầu tư method mới (có thể không phải hướng stage-aware).
- **Quy tắc quyết định (chốt TRƯỚC khi có kết quả):** headroom = drop OSFD @R101 − drop OSFD @model; ĐI TIẾP nếu cận dưới CI ≥ 10 ở ≥2/4 model gồm DINO-Swin-L; DỪNG/cân nhắc nếu điểm < 5 ở ≥3/4 model; giữa = vùng xám. Chi tiết: protocol_lock.md.
- YOLOX: S chính, L phụ (không tính vào quy tắc).
- Đã tải checkpoint (`scripts/download_checkpoints.sh --gen`; YOLOX mim id là `yolox_{s,l}_8x8_300e_coco`, khác tên file config). Chưa verify clean AP full val2017 cho 5 model này (model_registry.md vẫn trống ngày verify).
- `scripts/eval_generalization.py`: dùng lại PNG adv của `n300_B50_eps5`, `inference_detector` trên file (pipeline riêng từng model — YOLOX Resize 640+Pad 114, FCOS caffe normalize); đã thử YOLOX-S trên CPU: ảnh sạch/adv chạy đúng.
- **Sự cố GPU (~14:55):** tiến trình mới báo `No CUDA GPUs are available` / `nvidia-smi: Failed to initialize NVML: Unknown Error` dù `/dev/nvidia*` còn; tiến trình đã mở GPU trước đó (A1 trajectory) vẫn chạy. Dấu hiệu container mất quyền device cgroup phía host → cần restart container/pod từ nhà cung cấp. **Ảnh adv (`artifacts/`, 527 MB) chưa lưu ra ngoài** — nếu restart không giữ ổ /workspace thì mất.

---

## 2026-09-23 — A1 trajectory + lưu artifacts lên HF

- **A1 mở rộng quỹ đạo** (`results/mechanism/a1_gradients_traj.json`, mô tả, không thuộc tiêu chí): sign agreement R50 vs target tại step 10/25/50 — M-DI²: R101 0.515→0.512, ConvNeXt 0.508→0.506, Swin 0.505→0.504; OSFD: R101 0.525→0.522, ConvNeXt 0.512→0.510, Swin 0.508→0.506 (ảnh sạch: 0.5205 / 0.5097 / 0.5066). Thứ tự R101 > ConvNeXt > Swin giữ ở mọi step; alignment giảm nhẹ dần theo quỹ đạo, khoảng cách cùng họ − khác họ gần như không đổi → không thấy divergence về hướng gradient tăng dần khi attack "chui sâu" vào surrogate. Phần chính chạy lại trong cùng lần → A1_pass = False như trước.
- **Artifacts** `n300_B50_eps5` (900 PNG + dets.json, tar 552 MB, sha256 9eadf774…f8d590) đã upload lên HF dataset `congdanh99/transfer-attack`, repo chuyển **private** trước khi upload (theo quyết định user). Đã verify tải lại khớp sha256, đủ 900 PNG. Script `scripts/sync_artifacts.py` (upload/download, token qua env `HF_TOKEN`, từ chối nếu repo public); `huggingface_hub<1.0` thêm vào `setup_env.sh` (dry-run: không đổi gói đã pin).

---

## 2026-09-23 — Trước khi restart pod (sự cố GPU/NVML) — VIỆC CẦN LÀM NGAY SAU RESTART

Restart pod xóa ổ container (`/`), giữ volume `/workspace` (xfs riêng) → repo, `.venv`, `third_party/`, checkpoint (cả Generalization Panel), `data/coco`, `artifacts/` còn nguyên. Mất: gói apt, `/root` (memory Claude, session), `/tmp`. `.venv` trỏ `/usr/bin/python3.10` của image gốc (Ubuntu 22.04, python3.10-minimal) → dự kiến vẫn dùng được.

**KHÔNG chạy lại `scripts/setup_env.sh`** (nó xóa + dựng lại venv, mất ~20 phút). Chỉ cần:
1. `apt-get update && apt-get install -y tmux unzip libgl1 libglib2.0-0` (libgl1 cần cho `import cv2`).
2. Verify: `nvidia-smi` chạy được; `source .venv/bin/activate && python -c "import torch, cv2, mmdet; print(torch.cuda.is_available())"` → True.
3. Nếu `artifacts/runs/n300_B50_eps5/` mất vì lý do nào đó: `HF_TOKEN=... python scripts/sync_artifacts.py download n300_B50_eps5`.
4. Chạy Generalization Panel (tmux, ~20 phút): `python scripts/eval_generalization.py` → áp quy tắc khả thi đã khóa trong `protocol_lock.md` (headroom OSFD).
5. Verify clean AP full val2017 cho 5 model Generalization Panel (`third_party/mmdetection/tools/test.py`), điền `model_registry.md`.
6. Sau đó mới tới A3 (docs/mechanism_plan.md) — tùy kết quả bước 4.

---

## 2026-09-23 — Generalization Panel @300 (B=50, ε=5) + quyết định khả thi theo quy tắc khóa

- GPU dùng lại được sau khi reload cửa sổ VSCode — **không cần restart pod** (các bước "sau restart" ở entry trước không cần làm; tmux/apt còn nguyên).
- Thứ tự (user duyệt, pivot thứ tự chứ không đổi RQ): A2 → Generalization baselines → xác định vùng transfer khó → Mechanism → Method.
- `scripts/eval_generalization.py`, dùng lại PNG adv của `n300_B50_eps5` (surrogate Mask R-CNN R50), mỗi model đọc file qua pipeline riêng. Kết quả: `results/runs/n300_B50_eps5/generalization_metrics.json` (+ `gen_run.log`); detection thô `artifacts/runs/n300_B50_eps5/gen_dets/`.
- Clean AP trên n300: FCOS 41.9, DETR 44.3, YOLOX-S 43.9, YOLOX-L 53.5, DINO-Swin-L 62.4 — đều cao hơn README ~3–4 điểm, cùng mức lệch của R50 (45.0 vs 40.9) → nhất quán do tập con, checkpoint load đúng. Chưa verify full val2017.

Relative AP drop % [95% CI] (R101 = mốc cùng họ Controlled Panel):

| Method | R101 | FCOS-R50 | DETR-R50 | YOLOX-S | YOLOX-L (phụ) | DINO-Swin-L |
|---|---|---|---|---|---|---|
| MI-FGSM | 77.3 | 78.8 | 93.6 | 34.9 | 33.8 | 16.9 |
| M-DI²-FGSM | 93.0 | 93.1 | 96.1 | 58.0 | 60.5 | 26.1 |
| OSFD | 92.4 | 92.5 [89.7, 95.3] | 96.3 [93.9, 98.2] | 77.4 [71.8, 80.1] | 71.7 | 24.6 [20.4, 27.4] |

Headroom OSFD = drop(R101) − drop(model): FCOS −0.1 [−3.8, 3.3]; DETR −3.9 [−7.2, −1.5]; YOLOX-S 15.0 [12.1, 19.7]; DINO-Swin-L 67.8 [63.6, 72.7] (YOLOX-L 20.7, phụ).

**Áp quy tắc khóa (protocol_lock.md):** cận dưới CI headroom ≥ 10 ở YOLOX-S (12.1) và DINO-Swin-L (63.6) = 2/4 model, có DINO-Swin-L → **ĐI TIẾP** phát triển method. (Điều kiện DỪNG: headroom < 5 chỉ ở 2/4 — FCOS, DETR — không đạt ≥ 3/4.)

Quan sát (diễn giải, chưa kết luận):
- **Cùng backbone ResNet-50, khác detector (FCOS, DETR): transfer ≥ mức R101** (OSFD 92.5 / 96.3) → đổi detector head không làm yếu transfer khi backbone cùng họ; ủng hộ backbone-family effect. Caveat: FCOS/DETR khác surrogate cả về cách train (caffe 1x / 150e); và nhiều khả năng cùng khởi tạo từ ResNet-50 ImageNet-pretrained như surrogate → có thể chia sẻ feature cấp thấp — chưa kiểm.
- **YOLOX-S (CNN khác họ, CSPDarknet): OSFD 77.4 ≈ ConvNeXt 80.4** — nhất quán với cross-CNN ở Controlled Panel. OSFD vượt M-DI² rõ ở YOLOX (+19.4 S, +11.2 L).
- **DINO-Swin-L gần như miễn nhiễm**: OSFD 24.6, M-DI² 26.1, MI 16.9 (OSFD ≈ M-DI², hiệu −1.5 CI qua 0). Trong khi Mask R-CNN Swin-T (Controlled) bị OSFD 78.7 → **backbone Transformer đơn thuần KHÔNG giải thích được**; DINO-Swin-L lẫn 3 yếu tố: kiểu detector (DETR-family deformable, 5-scale), capacity/pretrain (Swin-L, ImageNet-22k, 384), và họ backbone. Panel hiện tại không tách được.
- Hệ quả cho RQ: vùng khó nhất (DINO-Swin-L) có thể KHÔNG phải do backbone-family → method nhắm vào đó có thể lệch khỏi RQ1–3. Đề xuất chẩn đoán (CHƯA chạy, cần user duyệt vì ngoài panel định trước): DINO-R50 (`dino-4scale_r50_8xb2-12e_coco`, có trong mmdet v3) để tách detector-paradigm khỏi backbone.

---

## 2026-09-23 — Chẩn đoán DINO-R50: quy tắc diễn giải khóa TRƯỚC khi chạy

> "DINO-R50 added post hoc as a diagnostic control to disentangle detector-paradigm effects from backbone-family effects observed for DINO-Swin-L; it is not used for target selection or method tuning." (user duyệt)

- Model: `dino-4scale_r50_8xb2-12e_coco` (AP README 49.0) — bản DINO-R50 DUY NHẤT có checkpoint chính thức trong mmdet v3 (24e/36e không có weights). Caveat: khác DINO-Swin-L cả ở lịch train (12e vs 36e) và số scale (4 vs 5), không chỉ backbone.
- Cùng ảnh adv `n300_B50_eps5`, cùng cách eval; kết quả ghi file riêng `results/runs/n300_B50_eps5/generalization_diag_dino.json` (không ghi đè bảng Generalization Panel).
- **Quy tắc diễn giải (user đề xuất; ngưỡng số do Claude chuẩn hóa từ ví dụ của user, chốt trước khi chạy)** — dựa trên relative AP drop của OSFD trên DINO-R50 (điểm ước lượng), kiểm chéo M-DI²:
  - **Case 1** — drop ≥ 80: kiểu detector DINO tự nó không giải thích failure → Swin-L / họ backbone / capacity-pretrain là nghi phạm chính; DINO-Swin-L phù hợp hướng cross-backbone.
  - **Case 2** — drop ≤ 40 (gần DINO-Swin-L 24.6): failure chủ yếu do kiểu detector DINO → KHÔNG dùng DINO-Swin-L làm bằng chứng backbone-gap; Controlled Panel vẫn là bằng chứng backbone sạch.
  - **Case 3** — 40 < drop < 80: cả detector lẫn backbone/capacity đều đóng góp → DINO-Swin-L chỉ là "generalization hard case", không quy nhân quả cho backbone.
  - **Contrast DETR vs DINO (cùng R50):** nếu cận dưới CI của drop(DETR-R50) − drop(DINO-R50) ≥ 10 → bằng chứng khác biệt nằm ở thiết kế detector giữa DETR và DINO.
- Sau run này mới quyết định có nhắm method vào DINO-Swin-L hay không.

---

## 2026-09-23 — Kết quả chẩn đoán DINO-R50 → Case 1

`results/runs/n300_B50_eps5/generalization_diag_dino.json` (+ `diag_dino_run.log`). Clean AP n300: DINO-R50 53.1 (README 49.0, cùng mức lệch tập con).

Relative AP drop % [95% CI] — DETR-R50 / DINO-Swin-L / **DINO-R50**:
- MI-FGSM: 93.6 / 16.9 / **92.8** [89.3, 94.7]
- M-DI²-FGSM: 96.1 / 26.1 / **95.4** [92.7, 96.5]
- OSFD: 96.3 / 24.6 / **97.3** [95.5, 98.5]

Contrast OSFD: DETR-R50 − DINO-R50 = −1.0 [−2.9, 0.9]; DINO-R50 − DINO-Swin-L = 72.7 [69.6, 76.6].

**Áp quy tắc khóa:** OSFD drop DINO-R50 = 97.3 ≥ 80 → **Case 1**: kiểu detector DINO tự nó KHÔNG giải thích failure; nghi phạm chính là backbone Swin-L (họ backbone và/hoặc capacity/pretrain). Contrast DETR−DINO (cận dưới −2.9) < 10 → không có bằng chứng khác biệt do thiết kế detector DETR vs DINO. DINO-Swin-L giữ được làm case cho hướng cross-backbone.

Quan sát (chưa kết luận):
- Cùng detector DINO, chỉ đổi R50 → Swin-L: drop 97.3 → 24.6 — hiệu ứng backbone cực lớn (hơn hẳn Mask R-CNN R101→Swin-T: 92.4 → 78.7).
- Trong Swin còn lẫn **capacity/pretrain**: Swin-T (Mask R-CNN, IN-1k, window 7) bị OSFD 78.7, Swin-L (DINO, IN-22k, 384, window 12) chỉ 24.6. Chưa tách được "họ Transformer" khỏi "Swin lớn + pretrain 22k". Chẩn đoán khả dĩ (CHƯA chạy, cần duyệt): Mask R-CNN Swin-S (`mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco`, cùng detector với Swin-T, lớn hơn) — đo độ dốc theo capacity trong cùng họ.
- Mọi target backbone ResNet-50 (FCOS, DETR, DINO) bị lừa ≥ R101 (MI-FGSM: 78.8 / 93.6 / 92.8 so với R101 77.3) → dấu hiệu "cùng kiến trúc R50 (có thể cùng ImageNet init)" transfer tốt hơn cả "cùng họ ResNet khác độ sâu". Chưa kiểm cùng init.

---

## 2026-09-23 — Chẩn đoán Mask R-CNN Swin-S: quy tắc khóa TRƯỚC khi chạy (chẩn đoán CUỐI)

> "Mask R-CNN Swin-S added as a diagnostic control to test whether model capacity within the Swin family explains the large transfer drop observed on DINO-Swin-L; not used for target selection or method tuning." (user duyệt)

- Model `mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco`: config giống hệt Swin-T (Controlled Panel), chỉ khác depths [2,2,18,2] vs [2,2,6,2]; cùng pretrain IN-1k 224, cùng lịch 3x → chỉ đổi capacity. KHÔNG tách được pretrain IN-22k/384 của Swin-L (mmdet không có Mask R-CNN Swin-L).
- Đo: d = drop_OSFD(Swin-T) − drop_OSFD(Swin-S), paired bootstrap trên cùng ảnh adv `n300_B50_eps5`; khoảng cần giải thích Swin-T → DINO-Swin-L = 78.7 − 24.6 = 54.1. Kiểm chéo M-DI².
- **Quy tắc (3 case của user; ngưỡng số do Claude chuẩn hóa, chốt trước khi chạy):**
  - **Case A — Swin-S ≈ Swin-T:** CI của d chứa 0, hoặc d < 5 (kể cả Swin-S bị lừa NHIỀU hơn) → capacity riêng không giải thích collapse ở Swin-L; nghi phạm còn lại: scale Swin-L + pretrain IN-22k / representation shift.
  - **Case B — capacity đóng góp lớn:** cận dưới CI d ≥ 5 VÀ d ≥ 27 (≥ nửa khoảng 54.1, tức drop Swin-S ≤ 51.7) → không được claim toàn bộ effect là do họ backbone.
  - **Case C — cộng dồn:** cận dưới CI d ≥ 5 VÀ d < 27 → family + capacity + pretraining cùng đóng góp; framing thận trọng.
- **Nguyên tắc (user): đây là chẩn đoán CUỐI.** Dù kết quả thế nào: đóng băng diễn giải → quay lại Mechanism Stage; không thêm chẩn đoán model nữa (tránh trôi sang "model robustness taxonomy").

---

## 2026-09-23 — Kết quả chẩn đoán Swin-S (cuối) — rơi vào khe hở của quy tắc

`results/runs/n300_B50_eps5/generalization_diag_swin_s.json` (+ `diag_swin_s_run.log`). Clean AP n300: Swin-S 52.2 (Swin-T 49.8).

Relative AP drop % — Swin-T / **Swin-S** / DINO-Swin-L: MI 45.1 / **38.1** / 16.9; M-DI² 67.1 / **58.1** / 26.1; OSFD 78.7 / **72.4** [67.3, 75.2] / 24.6.
OSFD: d = Swin-T − Swin-S = **6.3 [3.8, 9.1]**; Swin-S − DINO-Swin-L = 47.8 [43.1, 51.6]. d chiếm ~12% khoảng 54.1.

**Áp quy tắc khóa:** KHÔNG case nào khớp chặt — Case A cần CI chứa 0 hoặc d < 5 (CI [3.8, 9.1] không chứa 0, d = 6.3 ≥ 5); Case B/C cần cận dưới CI ≥ 5 (3.8 < 5). Khe hở do Claude chuẩn hóa ngưỡng: A dùng điểm ước lượng, B/C dùng cận dưới CI, không phủ vùng "d ≥ 5 nhưng cận dưới < 5". Không đổi ngưỡng sau khi thấy dữ liệu; ghi nhận nguyên trạng.

Mô tả trung tính: capacity trong họ Swin (T→S, cùng pretrain IN-1k) có hiệu ứng NHỎ nhưng khác 0 (CI loại 0; M-DI² 9.0, MI 7.0 cùng chiều), chỉ ~12% khoảng Swin-T → DINO-Swin-L; phần lớn (47.8) nằm giữa Swin-S và DINO-Swin-L — gắn với Swin-L scale + pretrain IN-22k/384 (+ tổ hợp với DINO), panel hiện tại không tách thêm được. **Diễn giải đóng băng: chờ user xác nhận câu chữ.** Theo nguyên tắc đã chốt: không thêm chẩn đoán model; quay lại Mechanism Stage.

**Chốt diễn giải Swin-S (user xác nhận, đóng băng):** capacity trong họ Swin có hiệu ứng nhỏ nhưng khác 0 (~12% mức tụt), không giải thích collapse ở DINO-Swin-L; phần lớn gắn với Swin-L scale + pretrain IN-22k/384, panel không tách thêm được → DINO-Swin-L là ca khó minh họa generalization, KHÔNG dùng làm bằng chứng nhân quả cho họ backbone. Bằng chứng backbone-family sạch: Controlled Panel + nhóm cùng R50 (FCOS/DETR/DINO-R50 ≥ 92%). **Dừng chẩn đoán model**, quay lại Mechanism Stage (A3). Khi đọc A3: lõi cơ chế = R101 vs ConvNeXt-T vs Swin-T; Swin-S/DINO-Swin-L chỉ là bằng chứng generalization (mô tả).

---

## 2026-09-23 — Mechanism A3: OSFD theo từng stage (n=100) → KHÔNG ĐẠT → bỏ hướng stage-aware

`scripts/mech_a3_stage_attack.py` (tiêu chí chốt + commit trước khi chạy, `18fe556`). Kết quả: `results/mechanism/a3_stage_attack.json` (+ `a3_run.log`); PNG `artifacts/runs/a3_stage_n100_B50_eps5/` (chưa upload HF).

Relative AP drop % (100 ảnh đầu n300), R101 / ConvNeXt-T / Swin-T | mô tả: Swin-S / DINO-Swin-L:
- all (OSFD gốc): 92.2 / 80.6 / 77.6 | 73.5 / 27.2
- stage1: 15.9 / 8.8 / 11.3 | 7.8 / 2.4
- stage2: 42.6 / 29.8 / 29.4 | 22.8 / 6.3
- stage3: 86.7 / 71.9 / 72.1 | 65.8 / 22.3
- stage4: 93.8 / 80.5 / 73.2 | 69.2 / 18.7

**Áp tiêu chí khóa:** không có cặp stage nào ĐẢO thứ hạng giữa R101 và target khác họ (reversals = [], A3_pass = False). Thứ tự stage1 < stage2 < stage3 ≤ stage4 giữ ở mọi target lõi.
- Có interaction khác 0 về ĐỘ LỚN (không đảo chiều): lợi ích của stage sâu nhỏ hơn trên Swin-T — vd stage3 vs stage4: R101 −7.1, Swin-T −1.1, interaction 6.0 [1.6, 11.0]; stage1 vs stage4 Swin 16.0 [9.5, 24.8]. Không đủ theo tiêu chí (cần đảo thứ hạng).
- Stage đơn so với OSFD gốc: không stage nào hơn "all" ở target khác họ (stage4 − all: ConvNeXt −0.1 [−3.4, 3.8], Swin-T −4.4 [−7.3, −2.1]; stage3 − all: −8.7 / −5.5) → chọn 1 stage KHÔNG tạo chỗ trống; OSFD gốc đã gần bằng stage tốt nhất.
- Mô tả: với Transformer (Swin-T, DINO-Swin-L) stage3 ≥ stage4 (DINO-Swin-L 22.3 vs 18.7), còn R101 stage4 tốt nhất — khớp quan sát A2 (distortion Swin giảm ở stage 4). Chỉ là quan sát.
- Alignment sign ∇(OSFD stage k) vs ∇ task target ≈ 0.500 ở mọi stage/target → OSFD không hoạt động qua căn hướng task-gradient (khớp A1: cosine OSFD ngược chiều).

**Quyết định theo quy tắc khóa (mechanism_plan.md):** giữ stage-aware chỉ khi A2(b) hoặc A3 có bằng chứng — A2 đạt một phần (không phải "có bằng chứng"), A3 không đạt → **BỎ hướng "stage-aware backward regularization"**. A1 cũng không đạt. Còn A4 (phổ tần), A5 (iterative stability) chưa chạy. Theo quy tắc: nếu chỉ A1/A4 có bằng chứng → đề xuất hướng method khác phù hợp cơ chế.

---

## 2026-09-23 — Mechanism A4: phổ tần + tập trung không gian (n=300) → KHÔNG ĐẠT

`scripts/mech_a4_spectral.py` (tiêu chí chốt + commit trước khi chạy, `14869e9`). Kết quả: `results/mechanism/a4_spectral.json` (+ `a4_run.log`). 3 dải cố định (cycles/pixel): thấp [0,1/6), trung [1/6,1/3), cao [1/3,∞).

- Tỉ lệ năng lượng thấp/trung/cao — gradient: R50 .236/.415/.349, R101 .270/.396/.334, ConvNeXt .290/.437/.274, Swin .253/.400/.347; δ: MI .228/.330/.442, M-DI² .279/.365/.356, **OSFD .390/.335/.275** (OSFD tần thấp hơn hẳn).
- (i) L1 phổ g_R50 vs g_t: R101 .095, ConvNeXt .172, Swin .094. ConvNeXt − R101 = +.077 [.069, .086] ✓; **Swin − R101 = −.001 [−.009, .007] ✗** → không đạt "cả 2 target khác họ".
- (ii) Spearman(match δ↔g_t, suppression) ở target khác họ **ÂM**, CI < 0 ở nhiều ô (OSFD: ConvNeXt −.177, Swin −.216; M-DI²: ConvNeXt −.186) — NGƯỢC chiều giả thuyết; retained cho cùng kết luận (dấu ngược) → không kích hoạt ceiling check. ✗
- **A4_pass = False.** Diễn giải (không đổi kết luận): tương quan âm per-ảnh nhiều khả năng do nội dung ảnh gây nhiễu (texture ảnh ảnh hưởng cả phổ δ lẫn độ dễ bị lừa), không phải cơ chế.
- Mô tả: gradient task tập trung mạnh theo không gian (top-5% pixel ≈ 87–89% năng lượng; năng lượng trong GT box gấp 6.9 lần tỉ lệ diện tích với R50/R101, 8.5 lần với ConvNeXt/Swin), còn δ (sign-step) gần như đều (top-5% ≈ 6%, trong box ≈ 1.0×). Giữa 3 method, method có δ tần thấp hơn (OSFD) transfer tốt hơn — chỉ 3 điểm dữ liệu, không kết luận.

**Tổng Mechanism Stage (A1–A4) theo tiêu chí khóa:** A1 ✗, A2 đạt một phần, A3 ✗, A4 ✗; A5 chưa chạy. Chưa có cơ chế nào "có bằng chứng"; hướng stage-aware đã bỏ (entry A3).

---

## 2026-09-23 — Tổng hợp cuối ngày

Viết `docs/synthesis_2026-09-23.md`: kết quả chắc chắn (gap PASS, backbone > detector, headroom Generalization → ĐI TIẾP, DINO-Swin-L chỉ là ca khó, stage-aware bị loại), trạng thái A1–A4, manh mối chưa đạt tiêu chí, chỗ trống theo target, 3 hướng method giả thuyết (M1 phân bổ budget theo vật thể, M2 ưu tiên tần thấp, M3 giảm lệ thuộc khuếch đại stage sâu) cần pilot với quy tắc chốt trước. CLAUDE.md trỏ tới file này. Dừng phiên, mai tiếp. Chưa push. Ảnh adv A3 đã upload HF (`runs/a3_stage_n100_B50_eps5.tar`, 282 MB, sha256 cba15cb5…b366c964, verify tải lại khớp, 400 PNG).

---

## 2026-09-24 — Đóng Mechanism Stage; chốt quy tắc pilot method A′1 (TRƯỚC khi code/chạy)

Máy mới (RTX 3090), `bootstrap.sh` xong; sanity: torch 2.1.2+cu118 `cuda_available=True`, mmcv 2.1.0 (ops OK), mmdet 3.3.0, 4 checkpoint Controlled Panel, COCO val2017 5000 ảnh.

**Đóng Mechanism Stage (user duyệt):** A1 ✗, A2 một phần, A3 ✗, A4 ✗, A5 để sau (prior yếu). Claim an toàn: transferability gắn với biểu diễn backbone, không chỉ kiến trúc detector; KHÔNG claim cơ chế cụ thể (gradient alignment / tần số / stage).

**Vì sao không dùng "W⊙g trước sign()" (đề xuất ban đầu của hướng M1):** với W > 0 mọi pixel, sign(W⊙g) = sign(g); kể cả với MI (chuẩn hóa g/mean|g| chỉ là vô hướng) nếu W cố định qua step thì sign(W⊙m) = sign(m) → δ giống hệt OSFD. Ngoài ra δ đã gần bão hòa L∞: RMS|δ| run `n300_B50_eps5` = 4.65 / 4.66 / 4.77 trên ε = 5 (MI / M-DI² / OSFD) → không thể "dồn thêm biên độ" vào vật thể; thứ còn thay đổi được là HƯỚNG (dấu) của δ. Vì vậy chọn A′1: trọng số không gian đặt BÊN TRONG loss OSFD (đổi chính gradient, không chỉ thang).

Động cơ (giả thuyết, KHÔNG phải cơ chế đã chứng minh): A4 thấy gradient task tập trung mạnh quanh vật thể trong khi δ sign-step gần đều. Wording cho phép: "motivated by the observed mismatch between spatial gradient concentration and nearly uniform sign-based perturbations".

### Method A′1 — OSFD với loss feature có trọng số không gian (khóa)

L = Σ_l [ Σ_{h,w} W_l(h,w) · ‖k·F_l^clean(h,w) − F_l^adv(h,w)‖²₂ ] / [ C_l · Σ_{h,w} W_l(h,w) ]

- F_l: feature BACKBONE stage l (như OSFD, không qua FPN), k = 3. Chia C_l để W ≡ 1 trùng TUYỆT ĐỐI `F.mse_loss` của OSFD gốc — kiểm bằng số trước khi chạy (sanity, không phải cấu hình).
- W_l: dựng map M ở khung input của model (canvas đã pad của data_preprocessor, GT box của `data_sample.gt_instances` — đã ở khung resize, không gồm crowd), rồi downsample kiểu area về lưới của stage l. Không co-transform W theo view RRB (feature sạch của OSFD gốc cũng không co-transform; xoay ≤ 7°, scale ≤ 1.1).
- Ảnh không có GT box: W ≡ 1 (= OSFD gốc).
- Mọi thứ khác GIỮ NGUYÊN OSFD full recipe (protocol_lock.md): MI μ = 1, RRB θ=7, l_s=10, ρ=0.8, s_max=1.1, σ=6, ε = 5, α = 1, B = 50 (1 backward / 2 view mỗi step), noise init 0, seed ảnh = SEED + i như `run_baselines.py`.
- Threat model: GT-assisted (primary). Biến thể box dự đoán của surrogate chỉ để validation sau, không thuộc Go/No-Go.

**Đúng 2 cấu hình W (khai báo trước, không thêm cấu hình cứu vãn):**
- **W1 "box":** M = 1 trong hợp các GT box, M = 0.1 ở mọi chỗ khác (kể cả vùng pad).
- **W2 "box+ring":** mỗi box nở mỗi phía 0.25 × (rộng box) theo chiều ngang, 0.25 × (cao box) theo chiều dọc, tối thiểu 16 px (khung model), clip vào ảnh; M = 1 trong hợp các box đã nở, 0.1 chỗ khác.

### Tập dev100 (tune/sàng lọc CHỈ ở đây)

- Pool = ảnh val2017 có ≥ 1 instance annotation (cùng tiêu chí n1000, 4952 ảnh) − n1000 = 3952 ảnh; `dev100 = random.Random(20260924).sample(sorted(pool), 100)`; assert Dev100 ∩ N1000 = ∅. Sinh 1 lần, commit `data/image_lists/dev100.csv` + `dev_meta.json`, không sinh lại.
- n300/n1000 KHÔNG được dùng để chọn W hay bất cứ tham số nào.

### Sàng lọc @ dev100 (điểm ước lượng)

Chạy OSFD, OSFD+W1, OSFD+W2 trên dev100 (cùng seed ảnh). Đánh giá: ConvNeXt-T, Swin-T, YOLOX-S (quyết định); R50, R101, DINO-Swin-L (chỉ báo cáo). Δ = relative AP drop(candidate) − drop(OSFD).

Một cấu hình QUA sàng lọc nếu đồng thời: ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 2; Δ_ConvNeXt-T ≥ 0; Δ_Swin-T ≥ 0; Δ_YOLOX-S ≥ 0.
- Cả 2 qua → chọn ΔCrossAvg lớn hơn (chênh < 0.5 → chọn W1, đơn giản hơn). Chỉ 1 cấu hình được sang n300.
- Không cấu hình nào qua → DỪNG A′1, ghi log, không thêm biến thể.

### GO @ n300 (chạy 1 lần, cấu hình đóng băng, không tune thêm)

So với OSFD của run `n300_B50_eps5` (PNG trên HF, cùng seed ảnh → paired bootstrap 1000 mẫu, BOOT_SEED 2026). GO nếu đồng thời:
- ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 3 VÀ cận dưới CI 95% > 0;
- Δ_ConvNeXt-T > 0 VÀ Δ_Swin-T > 0 (điểm ước lượng);
- Δ_YOLOX-S ≥ 5 VÀ cận dưới CI 95% > 0 (bootstrap per-model của `eval_generalization.py` hỗ trợ được).
- DINO-Swin-L: chỉ báo cáo, không thuộc Go/No-Go. Báo cáo kèm R101 (Δ cùng họ), runtime, số view forward (không đổi: 2×B).
- Không đạt → NO-GO A′1, ghi log.

---

## 2026-09-24 — Pilot A′1 sàng lọc @ dev100 → W1 QUA (sát ngưỡng), W2 KHÔNG QUA

Code `6feb7ff` (commit trước khi chạy). Sanity trước khi chạy: W ≡ 1 trùng `F.mse_loss` tuyệt đối; W_l khớp lưới 4 stage. **Kiểm cơ chế (không phải kết quả transfer), 3 ảnh dev đầu, step 0:** gradient W1/W2 so với OSFD có cosine 0.62–0.91 nhưng sign agreement 0.935–0.985 — hạ trọng số nền trong loss chủ yếu làm gradient nền NHỎ đi chứ ít đổi dấu → tác động lên δ sign-step nhỏ theo thiết kế.

Kết quả `results/runs/dev100_B50_eps5/` (`metrics.json`, `generalization_metrics.json`, `pilot_a1_screen.json`, log). Clean AP dev100: R101 46.6, YOLOX-S 46.5, DINO-Swin-L 64.0. Relative AP drop % (điểm), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L:
- OSFD: 97.0 / 91.1 / 76.0 / 74.6 / 71.5 / 23.3
- OSFD-W1: 96.8 / 92.8 / 78.7 / 76.3 / 71.9 / 23.7
- OSFD-W2: 96.8 / 92.3 / 78.7 / 75.8 / 71.5 / 22.3

Áp quy tắc khóa (`scripts/pilot_a1_screen.py`):
- **W1 − OSFD:** ΔCrossAvg +2.13 ✓ (≥ 2), ConvNeXt +2.6 ✓, Swin +1.6 ✓, YOLOX-S +0.4 ✓ → **QUA**.
- **W2 − OSFD:** ΔCrossAvg +1.96 ✗, ConvNeXt +2.7, Swin +1.2, YOLOX-S 0.0 → **KHÔNG QUA** (sát ngưỡng; không làm tròn, không nới).
- → **Sang n300 với OSFD-W1** (cấu hình đóng băng).

Mô tả (không thuộc quy tắc): ΔCrossAvg paired bootstrap trên dev100 — W1 +2.1 [0.5, 4.1], W2 +2.0 [0.9, 3.4]; W1 ≈ W2 (−0.2 [−1.6, 1.3]). R101 cũng tăng (+1.7 / +1.2) → chưa phải gain riêng cho khác họ. YOLOX-S +0.4 [−2.6, 3.1], DINO-Swin-L +0.4 [−1.6, 2.6] — xa ngưỡng GO YOLOX-S ≥ 5 → kỳ vọng tiên nghiệm cho GO @ n300 thấp; vẫn chạy 1 lần theo quy tắc. Runtime không đổi (~5.0 s/ảnh, cả 3).

---

## 2026-09-24 — Pilot A′1 GO @ n300 (OSFD-W1, chạy 1 lần, đóng băng) → **NO-GO**

`results/runs/n300_B50_eps5/{metrics_pilot_a1.json, generalization_pilot_a1.json, pilot_a1_go.json, pilot_a1_go_run.log}` (file riêng, không đè `metrics.json` / `generalization_metrics.json` của baseline). OSFD dùng lại PNG của run baseline (không attack lại); **số OSFD tái lập đúng bảng baseline đã commit** (98.9 / 92.4 / 80.4 / 78.7; YOLOX-S 77.4; DINO-Swin-L 24.6) → pipeline nhất quán.

Relative AP drop % [95% CI], R50 / R101 / ConvNeXt-T / Swin-T | YOLOX-S / DINO-Swin-L:
- OSFD: 98.9 / 92.4 / 80.4 / 78.7 | 77.4 / 24.6
- OSFD-W1: 99.0 / 92.1 [88.3, 95.2] / 82.0 [77.1, 84.8] / 78.2 [73.5, 81.4] | 78.2 [72.9, 80.9] / 26.2 [21.6, 28.8]

W1 − OSFD (`scripts/pilot_a1_go.py`): ConvNeXt +1.6, Swin −0.5, R101 −0.3; **ΔCrossAvg +0.5 [−0.5, 2.1]**; **YOLOX-S +0.8 [−0.3, 2.3]**; DINO-Swin-L (báo cáo) +1.6 [0.1, 2.5].

Áp quy tắc khóa: ΔCrossAvg ≥ 3 ✗, CI > 0 ✗, ConvNeXt > 0 ✓, Swin > 0 ✗, YOLOX-S ≥ 5 ✗, CI > 0 ✗ → **NO-GO A′1**. Không thêm biến thể (theo quy tắc).

Ghi chú:
- Mức cải thiện ở dev100 (+2.1) co về +0.5 ở n300 — khớp kiểm cơ chế trước khi chạy (W chỉ đổi dấu gradient ở ~2–6% pixel) và cho thấy sàng lọc bằng điểm ước lượng trên 100 ảnh nhiễu cỡ ±2.
- Runtime ghi 3.1 (OSFD, lấy từ stats phiên 2026-09-23) vs 4.9 s/ảnh (W1, máy hôm nay) — khác máy, không phải chi phí method (ở dev100 cùng máy: 5.0 vs 5.1).
- Hệ quả cho hướng M1: dưới L∞ với sign-step, phân bổ không gian qua trọng số loss không tạo chỗ trống đáng kể. Hướng M1 dạng này coi như đóng.
- Artifact mới (PNG OSFD-W1 n300, `dets_pilot_a1.json`, `gen_dets/`; run `dev100_B50_eps5`) CHƯA upload HF.

---

## 2026-09-24 — Chốt quy tắc pilot P2 (gộp lõm theo channel) + dev300 (TRƯỚC khi code/chạy)

**Bỏ P1 (drop channel ngẫu nhiên trong loss, p = 0.1/0.2) trước khi chạy** — kiểm cơ chế 5 ảnh dev100: mask chỉ chọn số hạng của loss (không đổi forward) nên g_M = Σ_c M_c g_c tuyến tính theo mask; cos(g_M, g_OSFD) 0.993–0.996 / 0.984–0.991, sign agreement 0.96–0.97 / 0.94–0.95; trung bình 16 mask cos 0.999+ với g_OSFD (momentum lấy đúng trung bình đó) → P1 ≈ OSFD. P1′ (dropout trong forward, kiểu Ghost Networks, Li et al. AAAI 2020) để dự phòng xa (rủi ro trùng lặp cao).

**Manh mối cho P2 (đính chính số đã nêu trong chat):** "top-10% channel chiếm 57–66% loss stage 4" đo trên d ≈ 4·F_clean² ở δ≈0 — chủ yếu là tập trung năng lượng feature SẠCH. Đo đúng trên distortion thật e_c = mean_hw (F_adv − F_clean)² của R50, 100 PNG OSFD dev100: top-10% share s1–s4 = 0.29 / 0.28 / 0.29 / **0.48 ± 0.03** (đều = 0.10); năng lượng sạch 0.32 / 0.27 / 0.37 / 0.59; loss OSFD 0.30 / 0.26 / 0.33 / 0.55. OSFD-W1 giống hệt. → distortion OSFD tập trung rõ ở stage 4, một phần kế thừa từ feature sạch.

Giả thuyết (chưa là cơ chế): để vài channel sâu thống trị distortion có thể là chuyên biệt hóa theo representation của R50; giảm lợi ích biên của channel đã bị phá mạnh có thể giúp transfer khác họ. Ngược hướng FIA/NAA (tăng trọng feature quan trọng). Chưa claim novelty trước khi rà literature đầy đủ.

### Method P2 (khóa)

d_{l,c} = mean_{h,w} (F^adv_{l,c,h,w} − k·F^clean_{l,c,h,w})², k = 3 (OSFD gốc = Σ_l mean_c d_{l,c}, đúng `F.mse_loss`).
**L_P2 = Σ_l mean_c φ_l(d_{l,c}),  φ_l(d) = τ_l · log(1 + d/τ_l)**  (φ′ = 1/(1 + d/τ_l)).
- τ_l = s · median_c d⁰_{l,c}, d⁰ = d tại δ = 0 KHÔNG qua RRB (= (1−k)²·mean_hw F_clean²), tính 1 lần mỗi ảnh, detach, cố định suốt 50 step; chặn dưới 1e-12.
- Loss cộng trên 2 view RRB như OSFD. Mọi thứ khác giữ OSFD full recipe (MI μ=1, RRB, ε=5, α=1, B=50, noise init 0, seed ảnh SEED+i).
- Sanity trước khi chạy: φ(d) = d trùng tuyệt đối `F.mse_loss` OSFD.
- **Đúng 2 cấu hình, chỉ đổi s (giữ log1p):** P2a s = 1 (median, lõm mạnh); P2b s = 2 (nhẹ hơn). Không thử φ khác (sqrt/Huber/clip...).

### dev300 (tập development DUY NHẤT cho RQ3 từ đây)

dev300 = dev100 (giữ nguyên) ∪ `random.Random(20260925).sample(sorted(pool − dev100), 200)`, pool = ảnh val2017 có ≥ 1 instance − n1000; assert dev300 ∩ n1000 = ∅. Sinh 1 lần, commit `data/image_lists/dev300.csv`. **Không mở thêm dev500/dev700** dù P2 fail.

### Quy trình (một đường duy nhất)

1. **Mechanism gate** — 20 ảnh đầu của `dev100.csv`, mỗi cấu hình, cả hai điều kiện:
   - (G1) sign agreement(∇ₓL_P2, ∇ₓL_OSFD) tại δ = 0, không RRB, trung bình 20 ảnh **< 0.95**;
   - (G2) sau 10 step full recipe (MI + RRB, cùng seed ảnh), top-10% share của distortion thật e_{4,c} ở stage 4 (R50) trung bình 20 ảnh: **C10₄(P2) ≤ C10₄(OSFD) − 0.05**.
   Chỉ cấu hình qua gate mới vào dev300; không cấu hình nào qua → DỪNG P2.
2. **Sàng lọc @ dev300** (điểm ước lượng): ΔCrossAvg(ConvNeXt-T, Swin-T) ≥ 3; Δ_ConvNeXt-T > 0; Δ_Swin-T > 0; Δ_YOLOX-S ≥ 2. R50, R101, DINO-Swin-L chỉ báo cáo. Cả 2 qua → chọn ΔCrossAvg lớn hơn; chênh < 0.5 → chọn P2b (gần OSFD hơn).
3. **GO @ n300** (1 lần, đóng băng; so OSFD của `n300_B50_eps5`): ΔCrossAvg ≥ 3 VÀ cận dưới CI 95% > 0; Δ_ConvNeXt-T > 0 VÀ Δ_Swin-T > 0; Δ_YOLOX-S ≥ 5 VÀ cận dưới CI > 0. DINO-Swin-L chỉ báo cáo.
4. **Điều kiện cho diễn giải cơ chế (không phải điều kiện GO):** nếu GO, phải kèm C10₄(P2) < C10₄(OSFD) trên ảnh adv n300 (paired CI < 0) và báo cáo Spearman per-ảnh(ΔC10₄, Δ suppression) ở target khác họ. Không đạt → method có thể hữu ích nhưng KHÔNG viết theo câu chuyện nhân quả "giảm tập trung channel".
5. **Nếu P2 NO-GO:** không làm tiếp P3/P4 kiểu feature engineering; RQ3 pivot sang nguyên lý khác hẳn hoặc chấp nhận kết luận "các can thiệp đã thử không thu hẹp được phần gap còn lại một cách hệ thống".

---

## 2026-09-24 — Pilot P2: mechanism gate → KHÔNG cấu hình nào qua → DỪNG P2 (không chạy dev300)

Code `f78d785` (commit trước khi chạy). `scripts/pilot_p2_gate.py`, kết quả `results/runs/dev300_B50_eps5/p2_gate.json`. 20 ảnh đầu dev100.

- Sanity: φ(d) = d trùng `F.mse_loss` (sai số tương đối tối đa 2.2e-7) ✓.
- **G1 (đổi hướng gradient):** sign agreement với OSFD tại δ = 0 — P2a 0.837, P2b 0.859 (< 0.95 ✓; khoảng 0.72–0.93 theo ảnh). P2 đổi gradient thật, mạnh hơn nhiều so với W1 (0.935–0.985) và P1 (0.94–0.97).
- **G2 (giảm tập trung distortion stage 4 sau 10 step):** C10₄ OSFD 0.483; P2a 0.476 (Δ −0.008), P2b 0.477 (Δ −0.006) — cần ≤ −0.05 ✗. Theo ảnh, Δ nằm trong khoảng −0.03 đến +0.03; 3/20 ảnh P2 còn tập trung HƠN OSFD.
- **Áp quy tắc khóa:** cả 2 cấu hình trượt G2 → **DỪNG P2**, không chạy dev300, không thử φ/s khác.

Mô tả (không đổi kết luận): dù objective đổi hướng gradient đáng kể (~15% pixel đổi dấu), mức tập trung theo channel của distortion thật ở stage 4 gần như bất biến (0.48 ± 0.01). Gợi ý: sự tập trung này là thuộc tính của representation stage 4 của R50 (channel nào phản ứng với nhiễu cỡ ε = 5) chứ không phải do hàm gộp tuyến tính của OSFD — can thiệp ở mức loss khó phân bổ lại. Đây là quan sát, không phải kết luận nhân quả.

**Theo nguyên tắc đã chốt (entry quy tắc P2, mục 5):** không làm tiếp P3/P4 kiểu feature engineering. Chuỗi can thiệp mức loss trên OSFD đã thử: A′1 (không gian, NO-GO @ n300), P1 (bỏ trước khi chạy — ≈ OSFD), P2 (dừng ở gate). RQ3 phải pivot sang nguyên lý khác hẳn, hoặc chấp nhận kết luận "các can thiệp đã thử không thu hẹp được phần gap còn lại một cách hệ thống". Chờ user quyết.

---

## 2026-09-24 — Tổng kết cuối ngày (dừng phiên, mai tiếp)

Phiên hôm nay (máy mới RTX 3090, `bootstrap.sh` + `download_checkpoints.sh --gen`): đóng Mechanism Stage; tạo dev100 rồi dev300 (tập development DUY NHẤT cho RQ3); pilot A′1 → NO-GO @ n300; P1 bỏ trước khi chạy; P2 dừng ở mechanism gate. Chi tiết + số liệu: các entry 2026-09-24 ở trên.

**Trạng thái:** RQ3 chờ user quyết — (1) pivot sang nguyên lý khác hẳn (không phải đổi hàm loss trên feature R50; ví dụ can thiệp đường tính forward kiểu P1′/Ghost Networks), quy tắc chốt trước, chỉ dùng dev300; hoặc (2) chấp nhận kết luận âm, chuyển trọng tâm sang analysis paper (còn thiếu cho bảng cuối: B=200 + A5, ε=8, clean AP full val2017 cho Generalization Panel).

**Việc mở cho phiên sau:**
1. Push các commit hôm nay (máy không có credential GitHub).
2. Artifact CHƯA upload HF (mất nếu trả máy): `artifacts/runs/dev100_B50_eps5/` (211 MB: PNG OSFD/W1/W2, dets, gen_dets) và phần mới trong `artifacts/runs/n300_B50_eps5/` (PNG OSFD-W1, `dets_pilot_a1.json`, `gen_dets/`). Upload `n300_B50_eps5` sẽ GHI ĐÈ tar cũ trên HF (chỉ thêm nội dung) — cân nhắc upload phần mới dưới tên riêng.
3. Token HF đã dùng hôm nay bị lộ trong chat — revoke, tạo token mới.
4. Nếu máy mới: `bash scripts/bootstrap.sh` + `bash scripts/download_checkpoints.sh --gen`; ảnh adv baseline: `python scripts/sync_artifacts.py download n300_B50_eps5`.

---

## 2026-10-03 — Chốt quy tắc pilot KG (chọn cặp object theo Knowledge Graph + OSFD) — TRƯỚC khi code/chạy

> **Trạng thái: LOCKED (user duyệt 2026-10-03).** Mọi giá trị dưới đây (kể cả các giá trị Claude đề xuất ban đầu: PPMI, margin 15%, khớp diện tích ≤ 2×, S2 ≥ 0.5, S3 ≤ 1.5×, bảng đồng nghĩa) đã được user chấp nhận; user thêm min support = 20 và ghi rõ tập loại trừ rò rỉ. Không đổi sau khi thấy kết quả.

**Pivot RQ3 (user quyết):** sau P2, chuyển sang nguyên lý khác hẳn — can thiệp ở **view đầu vào** (thay 1 view của RRB bằng crop vùng hợp của 1 cặp object), không đổi hàm loss feature.

Giả thuyết (chưa là cơ chế): view tập trung vào vùng hợp của 2 object có quan hệ ngữ nghĩa (theo Visual Genome) cho δ transfer tốt hơn sang target khác họ, **và phần lợi ích đó đến từ tri thức ngữ nghĩa**, không chỉ do có thêm augmentation. Prior của Claude: E1 > E0 có thể xảy ra (augmentation scale kiểu DI), còn E2 > E1 là phần rủi ro — với surrogate, KG chỉ quyết định **vùng không gian nào** được crop; chưa có cầu nối cơ chế tới gap họ backbone (RQ1/RQ2). Novelty: cần rà Cai et al. (AAAI 2022 "Context-aware transfer attacks for object detection"; CVPR 2022 "Zero-query transfer attacks on context-aware object detectors") — dùng đồ thị đồng xuất hiện nhưng cho mục đích khác (targeted, né context-consistency check).

### Knowledge Graph (khóa, sinh 1 lần, commit trước mọi lần chạy attack)

- Nguồn: Visual Genome v1.4 `relationships.json` + `image_data.json` (để lấy `coco_id`).
- **Chống rò rỉ:** bỏ mọi ảnh VG có `coco_id` ∈ dev300 ∪ n300 ∪ n1000 (n300 ⊂ n1000, ghi tường minh để dễ kiểm tra).
- Map tên object VG → COCO-80: khớp tên chính xác (sau lowercase, bỏ khoảng trắng thừa, dùng `synsets` nếu có) + **bảng từ đồng nghĩa viết tay** (vd man/woman/boy/girl/child/people/lady/guy → person; motorbike → motorcycle; plane → airplane; tv/television → tv; phone/cellphone → cell phone; sofa → couch; table → dining table). Bảng này commit cùng ma trận, viết xong TRƯỚC khi nhìn bất kỳ kết quả attack nào; không sửa sau. Tên không map được thì bỏ.
- Đếm C_ij = số relationship instance có subject ∈ class i, object ∈ class j; đối xứng hóa C̃ = C + Cᵀ (tính cả i = j, vd person–person).
- **Chuẩn hóa (chọn 1, không thử cái khác): PPMI với min support** — S_ij = max(0, log[ C̃_ij · Σ C̃ / (C̃_i· · C̃_·j) ]), tổng biên tính trên toàn bộ C̃; sau đó **C̃_ij < 20 → S_ij = 0** (min support = 20 relationship instance, không tune). Mục đích: PPMI cho điểm rất cao với cặp cực hiếm; ngưỡng buộc KG chỉ dùng quan hệ có đủ bằng chứng. Lý do: đếm thô bị person/car chi phối → "chọn cặp KG" gần như thành "chọn cặp có person" (nhiễu do tần suất class lẫn với kích thước box), còn PPMI đo ái lực quan hệ vượt trên tần suất.
- Output: `data/kg/coco80_vg_relation_matrix.npy` (S, 80×80), `data/kg/coco80_vg_relation_metadata.json` (phiên bản VG, bảng đồng nghĩa, min support, số ảnh VG bị loại, số relation map được, số ô bị min support loại, tỉ lệ ô S > 0).

### Cặp hợp lệ, ảnh có thông tin

- Object hợp lệ = GT instance không phải crowd (`data_sample.gt_instances`, khung model). Cặp hợp lệ = 2 instance khác nhau (cho phép cùng class).
- Phân loại ảnh (tính sẵn trên dev300/n300 từ annotation, báo cáo trước khi chạy):
  - **0 cặp** (1 object): view 2 = view RRB gốc → ảnh này giống hệt E0.
  - **1 cặp:** E1 = E2 = cặp đó (có pair-augmentation nhưng không có thông tin KG).
  - **≥ 2 cặp = "ảnh có thông tin"** (KG có thể chọn khác random).
  - Nếu mọi cặp của ảnh có S = 0: E2 sample đều (= E1 về prior), ảnh tính vào nhóm "KG suy biến", báo cáo riêng.

### Chọn cặp mỗi step (E1 và E2 dùng chung luồng RNG)

Mỗi step t, RNG riêng cho việc chọn cặp, seed theo ảnh (SEED + i), giống nhau giữa E1 và E2:
1. Rút cặp tham chiếu q ~ p(i,j) ∝ S_{c_i c_j} trên các cặp hợp lệ của ảnh.
2. **E2** dùng q.
3. **E1 (geometry-matched random control)** rút đều trong tập ứng viên {p hợp lệ : |log(A_p / A_q)| ≤ log 2}, A = diện tích union box đã nở margin. Tập luôn chứa q nên không bao giờ rỗng.

→ E1 và E2 có cùng mức ngẫu nhiên mỗi step và cùng phân bố kích thước crop (xấp xỉ), chỉ khác ở prior ngữ nghĩa. Báo cáo: tỉ lệ step mà cặp E1 ≠ cặp E2, phân bố |log(A_E1/A_E2)|.

### View cặp (giống hệt nhau giữa E1 và E2)

- T_pair: union box của 2 GT box, nở mỗi phía **15%** cạnh tương ứng của union, clip vào ảnh; crop từ adv ở khung model (KHÔNG xoay); resize bilinear khả vi theo đúng luật Resize của test pipeline (keep_ratio, cạnh dài ≤ 1333, cạnh ngắn ≤ 800), không chặn hệ số phóng; sau đó cộng nhiễu Gaussian σ = 6 như các view RRB (`rrb.gaussian_blur`).
- **Thay view 2 của RRB:** mỗi step = [view1 = rotate(adv) + nhiễu (giữ nguyên RRB), view2 = T_pair(adv) + nhiễu]. 2 view forward, 1 backward → B = 50, số view forward = 2×B, bằng E0.
- **Feature sạch đúng theo view:** loss OSFD của view cặp so với k·F_clean(T_pair(x_clean)) (cùng crop, không nhiễu, k = 3); view1 vẫn so với F_clean(x_clean) như OSFD gốc. F_clean theo cặp tính no_grad, cache theo cặp (forward thêm không tính vào B; báo cáo số forward no_grad + runtime).
- Loss = L_OSFD(view1) + L_OSFD(view2), cộng như OSFD hiện tại (hệ số ½ trong spec không đổi δ vì MI chuẩn hóa gradient).
- Mọi thứ khác giữ OSFD full recipe (protocol_lock.md): MI μ = 1, ε = 5, α = 1, B = 50, noise init 0, RRB θ = 7, l_s = 10, σ = 6, seed ảnh SEED + i. Threat model GT-assisted.

### Điều kiện

- **E0** — OSFD gốc (chạy mới trên dev300; trên n300 dùng lại PNG `n300_B50_eps5`).
- **E1** — geometry-matched random control (cặp ngẫu nhiên, điều kiện theo hình học của cặp tham chiếu do KG rút) + OSFD. Trong log/paper gọi đúng tên này, KHÔNG gọi là "Random-pair" (vì E1 không ngẫu nhiên hoàn toàn).
- **E2** — KG + OSFD.

### Chỉ số

- **CrossAvg₃ = mean(drop ConvNeXt-T, Swin-T, YOLOX-S)** — ĐỊNH NGHĨA MỚI, khác CrossAvg 2 target của A′1/P2 (baseline n300: CrossAvg₃(OSFD) = 78.8, CrossAvg₂ = 79.5). R101 = cùng họ; R50 white-box, DINO-Swin-L chỉ báo cáo.
- Paired bootstrap 1000 mẫu trên ảnh, 95% CI, BOOT_SEED 2026. Báo cáo E2−E0, E2−E1, E1−E0 cho CrossAvg₃, từng target, R101 — trên toàn bộ ảnh (chính) và trên tập ảnh có thông tin (phụ, không thuộc quy tắc).
- Output `results/kg_pilot/<set>/<condition>/{metrics.json, per_image_metrics.csv, config.json, runtime.json}`: clean/adv AP, relative AP drop, AP50, suppression, retained confidence, CrossAvg₃, TransferGap, L∞, L2, runtime, số backward, số view forward (+ forward no_grad).

### Quy trình (1 đường duy nhất)

**Bước 0 — Sanity gate kỹ thuật** (20 ảnh có thông tin đầu tiên theo thứ tự `dev100.csv`; không đo transfer). Phải đạt hết, nếu không thì sửa bug hoặc dừng:
- (S1) Crop chứa trọn 2 GT box; F_clean(T_pair) đúng shape; với 1 cặp cố định và tắt nhiễu, loss tại δ = 0 của view cặp = (1−k)²·mean F_clean² (đúng như OSFD).
- (S2) Tỉ lệ step có cặp E1 ≠ cặp E2 **≥ 0.5** — thấp hơn thì KG gần như trùng random sau khi khớp hình học → phép thử vô nghĩa → **DỪNG** hướng KG.
- (S3) Trung vị |log(A_E1 / A_E2)| ≤ log 1.5 (khớp hình học thực sự hoạt động).
- (S4) Sign agreement(∇ E2, ∇ E0) tại step 0, trung bình 20 ảnh < 0.95 (view cặp thật sự đổi hướng gradient — cùng ngưỡng G1 của P2).

**Bước 1 — Sàng lọc @ dev300** (E0, E1, E2; 1 lần). QUA nếu đồng thời:
- ΔCrossAvg₃(E2 − E0) ≥ 3 (điểm ước lượng);
- ΔCrossAvg₃(E2 − E1) > 0 VÀ cận dưới CI 95% > 0;
- E2 − E0 > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S (điểm ước lượng);
- drop R101(E2) ≥ drop R101(E0) − 3.

Không qua → NO-GO hướng KG, ghi log, không thêm biến thể (không đổi chuẩn hóa S, margin, cách khớp). **Trường hợp E1 > E0 nhưng E2 ≈ E1** (CI E2−E1 chứa 0): kết luận "augmentation cặp có thể hữu ích, KG không đóng góp rõ" → NO-GO cho claim KG; pair-augmentation KHÔNG tự động thành method — muốn đi tiếp phải chốt quy tắc mới riêng.

**Bước 2 — Đóng băng cấu hình → GO @ n300** (chạy 1 lần E1 + E2; E0 = PNG baseline). GO nếu đồng thời:
- (G1) ΔCrossAvg₃(E2 − E0) ≥ 3 VÀ cận dưới CI > 0;
- (G2) ΔCrossAvg₃(E2 − E1) ≥ 1.5 VÀ cận dưới CI > 0;
- (G3) E2 − E0 > 0 ở cả 3 target khác họ VÀ ≥ 2/3 target có E2 − E0 ≥ 3;
- (G4) drop R101(E2) ≥ drop R101(E0) − 3.

Không đạt → NO-GO, ghi log. Chỉ khi GO mới cân nhắc n1000 (idea.md §14).

Sanity gate (bước 0) chỉ kiểm tính đúng kỹ thuật và tính khả thi của phép thử, KHÔNG dùng để quyết định về hiệu năng.

Ghi chú độ nhạy: CI hiệu của 2 method gần nhau rộng khoảng ±1.5–1.8 ở 100 ảnh (W1−W2 @dev100: [−1.6, 1.3]), ước tính khoảng ±0.9 ở 300 ảnh → vì vậy sàng lọc dùng đủ dev300, không dùng dev100.

---

## 2026-10-03 — Pilot KG: dựng ma trận KG + thống kê cặp trên dev300/n300 (chưa chạy attack)

Máy RTX 3090; bootstrap xong (venv có sẵn dùng được, 4 checkpoint Controlled + Generalization Panel `--gen`, COCO val2017 5000 ảnh).

**KG** (`scripts/build_kg.py` → `data/kg/coco80_vg_relation_matrix.npy` + `_metadata.json`; VG gốc ở `data/vg/`, gitignored, sha256 zip ghi trong metadata):
- **Lựa chọn thực thi (không đổi quy tắc khóa):** map tên VG → COCO-80 bằng KHỚP TÊN CHÍNH XÁC sau chuẩn hóa + bảng viết tay (`NAME_TABLE` trong script, có số nhiều; loại có chủ đích tên mơ hồ: bag, glass, glove, plant, monitor, desk, van, calf, pitcher, female/male, crowd), **không dùng synset VG** — kiểm trước khi viết bảng: synset nhiễu nặng (`dog.n.01` gồm "hot dog" 3.5k lần, `child.n.01` gồm "fries"/"french fries" ~1.5k, `accountant.n.01` = "controller"), ~17% object không có synset. Bảng viết dựa trên tần suất tên, trước khi tính S.
- Loại 577/108077 ảnh VG có coco_id ∈ dev300 ∪ n300 ∪ n1000 (1300 coco_id). Relation map được 166,201/2,303,408 (phần lớn relation VG là bộ phận cơ thể/quần áo/cảnh vật); bỏ 2,459 quan hệ object–chính nó.
- 3240 ô tam giác trên: 1294 có đếm > 0, min support (C̃ < 20) loại 871, **308 ô S > 0 (9.5%)**. Lớp không có S > 0: potted plant, toaster.
- Quan sát: PPMI cao nhất thường ở cặp CÙNG class (parking meter, traffic light, zebra, sheep, cow… 4.5–6.2) và cặp đồ vật cùng ngữ cảnh (microwave–oven 4.8, mouse–keyboard 4.0, toilet–sink 3.7); person–X chỉ ~0.5–1.0. Đo bên dưới: thiên lệch cùng class trên ảnh thật nhỏ.

**Thống kê cặp** (`scripts/kg_pair_stats.py` → `results/kg_pilot/pair_stats.json`; giải tích từ annotation, không đo transfer):

| | dev300 | n300 |
|---|---|---|
| 0 cặp (→ = E0) | 43 (14.3%) | 45 (15.0%) |
| 1 cặp (E1 = E2) | 38 (12.7%) | 50 (16.7%) |
| ≥ 2 cặp (có thông tin) | 219 (73.0%) | 205 (68.3%) |
| … KG suy biến (mọi S = 0) | 1 | 7 |
| … S bằng nhau mọi cặp (E2 = đều) | 13 | 9 |
| Kỳ vọng P(cặp E1 ≠ E2), TB [ảnh có thông tin] | 0.826 | 0.819 |
| Kỳ vọng \|log(A_E1/A_E2)\|, TB | 0.248 | 0.236 |
| TV(phân bố cặp E2, E1), TB / trung vị | 0.413 / 0.414 | 0.410 / 0.446 |
| P(cặp cùng class): E2 vs chọn đều | 0.366 vs 0.333 | 0.324 vs 0.332 |

- Kỳ vọng S2 (≥ 0.5) và S3 (≤ log 1.5 = 0.405) đạt rộng rãi; sanity gate thật vẫn đo trên 20 ảnh dev100 theo quy tắc.
- TV ≈ 0.41: KG làm phân bố chọn cặp khác rõ so với random khớp hình học (không phải phép thử rỗng). Khoảng 27% ảnh dev300 không có thông tin KG → hiệu ứng E2−E1 trên toàn bộ ảnh bị pha loãng tương ứng.
- Việc tiếp theo: code view cặp + chọn cặp (E1/E2) trong engine OSFD, chạy sanity gate (bước 0).

---

## 2026-10-03 — Pilot KG: code view cặp + sanity gate (bước 0) → ĐẠT S1–S4

**Code:**
- `attack/methods/core.py`: view có thể mang loss riêng (phần tử thứ 3 của tuple; tương thích ngược).
- `attack/methods/kg_pair.py`: `PairSampler` (mỗi step rút q ~ p ∝ S rồi e1 đều trong M(q); LUÔN rút cả hai từ numpy RNG seed theo ảnh → chuỗi (q, e1) giống hệt giữa E1/E2), `crop_resize` (= luật `mmcv.rescale_size` keep_ratio 1333×800), `kg_osfd_attack(mode='kg'|'random_matched')`. View1 tiêu thụ random/torch toàn cục đúng thứ tự như `rrb_views` (step 0 view1 trùng E0); nhiễu σ của view cặp dùng `torch.Generator` riêng (crop E1/E2 khác kích thước sẽ làm lệch RNG toàn cục nếu dùng chung). Feature sạch theo cặp cache, đếm forward no_grad.
- Unit check: ảnh 0 cặp → E2 trùng tuyệt đối E0 (E0 tất định khi lặp lại); RNG toàn cục sau 3 step giống nhau giữa E1/E2; L∞ = 5.
- **Seed:** seed ảnh = SEED + chỉ số trong **dev300** (dev300 sắp theo image_id, dev100 không nằm ở đầu) — gate chọn ảnh theo thứ tự `dev100.csv` nhưng dùng seed dev300 → chuỗi cặp kiểm ở gate = chuỗi thật của lần sàng lọc.

**Sanity gate** (`scripts/pilot_kg_gate.py` → `results/kg_pilot/dev300/sanity_gate.json`; 20 ảnh có thông tin đầu theo dev100, 50 step thật E1 và E2):
- S1 ✓: crop chứa trọn 2 box (mọi cặp), lưới feature đúng stride, loss tại δ = 0 khớp (k−1)²·mean F² (sai số tương đối 1.1e-7).
- S2 ✓: tỉ lệ step cặp E1 ≠ E2 = **0.857** (≥ 0.5).
- S3 ✓: trung vị |log(A_E1/A_E2)| = **0.248** (≤ 0.405).
- S4 ✓: sign agreement(∇E2, ∇E0) step 0 = **0.759** (< 0.95).
- → **ĐẠT, sang sàng lọc dev300.**

Chẩn đoán (không thuộc quy tắc):
- Chuỗi (q, e1) của lần chạy E1 và E2 trùng nhau ở cả 20 ảnh.
- Tỉ lệ cặp cùng class: **E2 0.302 vs E1 0.405** — trên các ảnh này KG chọn cặp KHÁC class nhiều hơn random khớp hình học (ngược với lo ngại PPMI ưu tiên cặp cùng class).
- **S4 gần như không có sức phân biệt:** sàn sign agreement E0 vs E0 khác seed = **0.516** (nhiễu σ = 6 của RRB chi phối dấu gradient step 0); E2 vs E0 cùng seed = 0.759 (view1 trùng), E2 vs E1 = 0.886. Ngưỡng 0.95 (mượn từ G1 của P2, nơi không có RRB) quá lỏng cho thiết lập có RRB — ghi nhận, không đổi quy tắc; S4 chỉ xác nhận view cặp có đổi gradient.
- 1/20 ảnh (105455, 3 cặp) có P(E1≠E2) = 0: tập khớp hình học suy biến/KG một cặp chiếm ưu thế — trường hợp tự nhiên, nằm trong tỉ lệ pha loãng.
- Runtime gate: 142 s cho 20 ảnh (mỗi ảnh 2 lần chạy 50 step + 4 lần 1 step) → ước ~3.5 s/ảnh/điều kiện ở B = 50.

---

## 2026-10-03 — Pilot KG sàng lọc @ dev300 → **KHÔNG QUA (NO-GO hướng KG)**

Run `dev300_B50_eps5` (E0 = OSFD, E1 = `OSFD-KG-E1`, E2 = `OSFD-KG-E2`; `run_baselines.py --out metrics_kg.json`, `eval_generalization.py --out generalization_kg.json`), phân tích `scripts/pilot_kg_screen.py` → `results/kg_pilot/dev300/{screen.json, E0,E1,E2/{metrics.json, per_image_metrics.csv, config.json, runtime.json}}`. Paired bootstrap 1000 mẫu, BOOT_SEED 2026. Sự cố nhỏ: `eval_generalization.py` lỗi ghi JSON ở bước cuối (locale tmux ASCII, `ensure_ascii=False`) — dets đã lưu đủ; chạy lại với `PYTHONUTF8=1`.

Relative AP drop % (điểm), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- E0 OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | **73.6**
- E1 random khớp hình học: 95.3 / 87.0 / 76.3 / 73.8 / 62.7 / 23.2 | **70.9**
- E2 KG: 95.3 / 86.6 / 74.1 / 71.2 / 61.3 / 21.9 | **68.9**

Δ [95% CI] (toàn bộ 300 ảnh):
- **E2 − E0:** CrossAvg₃ **−4.7 [−6.2, −3.1]**; ConvNeXt −3.3, Swin −3.5, YOLOX-S −7.3, R101 −6.4 [−8.4, −3.9], DINO-Swin-L −3.0.
- **E2 − E1:** CrossAvg₃ **−2.1 [−3.0, −0.8]**; ConvNeXt −2.2 [−3.5, −0.6], Swin −2.6 [−4.0, −0.9], YOLOX-S −1.3 [−3.2, 0.4], R101 −0.4.
- **E1 − E0:** CrossAvg₃ −2.6 [−4.3, −1.2]; ConvNeXt −1.1, Swin −0.8 (CI qua 0), YOLOX-S −6.0, R101 −6.0.
- Ảnh có thông tin (n = 219, chỉ báo cáo): cùng chiều, lớn hơn chút (E2−E0 −5.2, E2−E1 −2.6 [−3.6, −1.0]).

**Áp quy tắc khóa:** cả 7 điều kiện trượt (ΔE2−E0 ≥ 3 ✗, E2−E1 > 0 ✗, CI ✗, 3 target > 0 ✗, R101 ≥ E0 − 3 ✗) → **NO-GO hướng KG**. Theo quy tắc: không thêm biến thể (không đổi chuẩn hóa S, margin, cách khớp, không chạy n300).

Diễn giải (mô tả, không đổi kết luận):
- **Thay view resize của RRB bằng crop cặp làm GIẢM transfer** (E1 < E0), mạnh nhất ở R101 (−6) và YOLOX-S (−6), white-box R50 cũng giảm (98.1 → 95.3). Crop phóng to một vùng nhỏ làm loss của view 2 chỉ phủ một phần ảnh → δ ngoài vùng crop nhận ít tín hiệu hơn; view resize toàn ảnh của RRB có vẻ quan trọng hơn ta nghĩ.
- **KG còn tệ hơn random khớp hình học** (CI loại 0, hai target Mask R-CNN khác họ). Có thể do KG tập trung vào ít cặp hơn (forward no_grad / ảnh: E2 12.9 vs E1 17.4 cặp khác nhau được dùng) → ít đa dạng vùng crop hơn. Không kiểm thêm (ngoài quy tắc).
- Chẩn đoán cặp trên dev300: tỉ lệ step cặp E1 ≠ E2 = 0.705 (gồm cả ảnh 1 cặp), tỉ lệ cặp cùng class E2 0.346 vs E1 0.349; 10 ảnh có mọi cặp S = 0 (gồm cả ảnh 1 cặp). Runtime ~3.3 s/ảnh cả 3 điều kiện; số view forward 2×B như nhau.
- Mức OSFD E0 trên dev300 nhất quán với n300 (R101 93.0 vs 92.4; ConvNeXt 77.4 vs 80.4; Swin 74.6 vs 78.7; YOLOX-S 68.7 vs 77.4 — dev300 khó hơn ở YOLOX-S).

**Trạng thái RQ3:** chuỗi đã thử — A′1 (loss không gian, NO-GO @ n300), P1 (bỏ), P2 (dừng ở gate), KG (view cặp, NO-GO @ dev300, còn làm giảm transfer). Chờ user quyết hướng tiếp.

Artifact CHƯA upload HF: `artifacts/runs/dev300_B50_eps5/` (PNG OSFD/E1/E2 dev300, `dets_kg.json`, `gen_dets/`).

---

## 2026-10-03 — Chốt quy tắc pilot R1 (OSFD-adapted SGM backward + OSFD) — TRƯỚC khi code/chạy

> **Trạng thái: LOCKED (user duyệt 2026-10-03).** Gồm cả bản SGM chỉ-nhánh-residual (mục dưới). **Tên gọi: "OSFD-adapted SGM" / "stage-balanced SGM" — KHÔNG gọi là port SGM nguyên bản.** Wording cho paper (user): *"We adapt SGM to OSFD's multi-stage feature objective by applying residual-gradient decay only within residual branches, avoiding depth-dependent rescaling of stage losses."* Không claim đây là exact SGM implementation.

**Cam kết trước (user):** đây là pilot method CUỐI của RQ3. **R1 NO-GO (ở sàng lọc dev300 hoặc GO @ n300) → dừng RQ3, chuyển sang analysis paper.**

Động cơ (giả thuyết): hướng M3 trong `docs/synthesis_2026-09-23.md`; A2 thấy từ stage 2 R101 khuếch đại distortion còn ConvNeXt/Swin thì không → gradient chuẩn của R50 có thể quá đặc thù đường residual của ResNet. SGM (Wu et al., ICLR 2020, "Skip Connections Matter") giảm gradient qua nhánh residual, ưu tiên skip. Khác A3 (chọn stage) và A′1/P2 (đổi loss) và KG (đổi view): giữ forward, loss, RRB, MI; chỉ đổi đường backward. Novelty: SGM/LinBP/PAS đã có ở classification — nếu GO, đóng góp phải nằm ở bằng chứng can thiệp cho RQ2/detector, không phải "gắn SGM vào OSFD". Nếu E_best > E0 rõ, đây là thí nghiệm can thiệp gắn với quan sát khuếch đại residual của A2 (có giá trị cho analysis paper dù method không mới).

### Ngữ nghĩa SGM (port từ code chính thức `csdongxian/skip-connections-matter/utils_sgm.py`)

Code gốc với ResNet-50: hook backward trên MODULE `relu` của mọi Bottleneck trừ block đầu mỗi stage (`'0.relu'` — block có downsample), nhân gradient √γ. Module này dùng 3 lần mỗi block (sau norm1, sau norm2, sau phép cộng residual) → residual × γ^1.5, skip × √γ → **tỉ lệ residual/skip = γ** (phần cốt lõi). Thêm: relu stem cũng bị nhân; hook chuẩn hóa `grad / std` ở mỗi block. Với classification (loss vào DUY NHẤT ở đầu ra) mấy phần thêm chỉ là hằng số vô hướng → vô tác dụng sau `sign()`.

**Vì sao không port nguyên văn — loss nhiều stage:** OSFD đặt loss ở 4 stage, gradient vào backward ở 4 độ sâu. Nhân √γ sau phép cộng (cả skip) làm số hạng stage 4 bị nhân ~(√γ)^12 ≈ 6·10⁻⁵ (γ = 0.2; R50 có 2+3+5+2 = 12 block không phải block đầu) trước khi tới input, còn stage 1 chỉ ~γ → gradient gần như CHỈ từ stage 1 (A3: tấn công riêng stage 1 rất yếu, R101 15.9%). Port nguyên văn sẽ đo "OSFD-stage-1", không đo SGM. Chuẩn hóa std theo block cũng đổi trọng số tương đối giữa các số hạng stage.
→ **Khóa (OSFD-adapted SGM):** chỉ giữ phần cốt lõi — trong mọi Bottleneck trừ block đầu mỗi stage, nhân gradient √γ ở 2 lần dùng ReLU TRONG nhánh residual (sau norm1, sau norm2); KHÔNG nhân ở ReLU sau phép cộng, KHÔNG nhân relu stem, KHÔNG chuẩn hóa std. Forward giữ nguyên tuyệt đối.
- **Unit test hướng SGM:** với loss chỉ đặt ở đầu ra stage 4 (1 điểm vào), gradient theo input của bản này phải song song với bản port nguyên văn (3 lần √γ + stem + chuẩn hóa std): cosine ≥ 0.999 trên 5 ảnh, γ ∈ {0.2, 0.5}. Diễn giải: xác nhận bản adapted giữ CÙNG hiệu ứng hướng của SGM trong trường hợp loss một điểm vào — KHÔNG phải "tương đương nói chung"; với OSFD nhiều stage hai bản cố ý không tương đương.

### Điều kiện (dev300; seed ảnh SEED + i như run_baselines)

| | backward |
|---|---|
| E0 | OSFD full recipe chuẩn — dùng lại PNG OSFD của run `dev300_B50_eps5` (cùng seed, không attack lại) |
| E1a | SGM γ = 0.2 (giá trị code gốc minh họa cho ResNet) |
| E1b | SGM γ = 0.5 (nhẹ hơn; cũng là default argparse của code gốc) |
| E2 | mỗi step chọn đều γ ∈ {0.2, 0.5} (KHÔNG gồm γ = 1); numpy RNG riêng seed SEED + i (không đụng RNG của RRB) |

Mọi thứ khác giữ OSFD full recipe (MI μ = 1, RRB θ = 7, l_s = 10, ρ = 0.8, s_max = 1.1, σ = 6, ε = 5, α = 1, B = 50, k = 3, noise init 0). Compute không đổi: 1 backward + 2 view forward / step. Không thử γ khác sau khi thấy kết quả.

### Quy trình (1 đường duy nhất)

**Bước 0 — Sanity kỹ thuật** (5 ảnh đầu dev100; không đo transfer, không có ngưỡng hiệu năng):
- Forward (feature 4 stage + loss OSFD) giữ nguyên tuyệt đối khi bật/tắt SGM.
- γ = 1 tái tạo gradient chuẩn (trùng tuyệt đối hoặc sai số tương đối < 1e-6).
- Unit test tương đương với code gốc (mục trên), cosine ≥ 0.999.
- γ = 0.2 / 0.5 đổi gradient có ý nghĩa: sign agreement(∇SGM, ∇chuẩn) tại δ = 0, KHÔNG RRB/nhiễu, trung bình 5 ảnh < 0.95.
- Báo cáo thêm (giữ, user nhấn mạnh): tỉ trọng đóng góp gradient từng số hạng stage (chuẩn vs SGM) — nếu bản adapted vẫn để 1 stage chiếm gần toàn bộ gradient input thì diễn giải sẽ lẫn với A3 → phải nêu rõ.
Hỏng → sửa bug; không chạy dev300 khi chưa đạt.

**Chẩn đoán cơ chế (KHÔNG phải tiêu chí):** trên 20 ảnh đầu dev100, sau 10 step (E0, E1a, E1b), đo D_k (như A2) trên R101 / ConvNeXt-T / Swin-T và tỉ số D_4/D_1; báo cáo cạnh kết quả transfer.

**Bước 1 — Sàng lọc @ dev300** (E1a, E1b, E2; 1 lần; đánh giá R50, R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L như pilot KG). CrossAvg₃ = mean(ConvNeXt-T, Swin-T, YOLOX-S). E_best = điều kiện có CrossAvg₃ cao nhất (chênh < 0.5 → ưu tiên E1 hơn E2, và γ = 0.2 hơn 0.5). E_best QUA nếu đồng thời:
- ΔCrossAvg₃(E_best − E0) ≥ 3 (điểm ước lượng);
- E_best − E0 > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S (điểm);
- drop R101(E_best) ≥ drop R101(E0) − 3.
Báo cáo (không thuộc quy tắc): E2 − E1a, E2 − E1b, E1a − E1b với CI.
Không qua → **NO-GO R1 → dừng RQ3.**

**Bước 2 — Đóng băng E_best → GO @ n300** (1 lần; E0 = PNG `n300_B50_eps5` từ HF). GO nếu đồng thời:
- (G1) ΔCrossAvg₃(E_best − E0) ≥ 3 VÀ cận dưới CI 95% > 0;
- (G3) E_best − E0 > 0 ở cả 3 target khác họ VÀ ≥ 2/3 target ≥ 3;
- (G4) drop R101(E_best) ≥ drop R101(E0) − 3.
Không đạt → NO-GO → dừng RQ3. GO → cân nhắc n1000 (idea.md §14); diễn giải cơ chế chỉ khi chẩn đoán D_k nhất quán (không thì method có thể hữu ích nhưng không viết câu chuyện nhân quả).

---

## 2026-10-03 — Pilot R1 (OSFD-adapted SGM): sanity + sàng lọc dev300 → **NO-GO** → theo cam kết: **dừng RQ3**

**Code** (`29c763e`, `68ca341`): `attack/methods/sgm.py` — `StageBalancedSGM` (forward hook trên norm1/norm2 của Bottleneck không phải block đầu mỗi stage, chèn identity có backward × √γ trước ReLU inplace; γ đổi được mỗi step), `LiteralSGM` (port nguyên văn, chỉ cho unit test), `sgm_osfd_attack`. Thêm `OSFD-SGM-0.2/0.5/mix` vào `run_baselines.py`.

**Sanity kỹ thuật** (`scripts/pilot_r1_gate.py` → `results/r1_pilot/dev300/sanity_gate.json`, 5 ảnh đầu dev100, δ = 0, không RRB): đạt hết — forward (feature + loss) giữ nguyên tuyệt đối; γ = 1 tái tạo gradient chuẩn (sai số tương đối 1.5e-7); hướng SGM với loss chỉ ở stage 4: cos(adapted, nguyên văn) = 0.9999999 (γ 0.2 và 0.5); sign agreement với gradient chuẩn 0.612 (γ 0.2) / 0.729 (γ 0.5). Tỉ trọng ‖∇‖ theo số hạng stage 1→4: chuẩn .003/.159/.361/.478; γ 0.2 .009/.302/.400/.290; γ 0.5 .007/.280/.401/.313 — không stage nào bị triệt, nhưng SGM dời trọng tâm từ stage 4 sang stage 2–3.

**Chẩn đoán D_k** (`scripts/mech_r1_diag.py` → `mech_diag.json`; 20 ảnh dev100, 10 step; chạy TRƯỚC khi có số transfer): SGM làm distortion stage sâu GIẢM ở mọi model, kể cả target khác họ — D_4: R101 1.01 → 0.71 (γ 0.2) / 0.90 (γ 0.5); ConvNeXt 0.58 → 0.39 / 0.52; Swin 0.50 → 0.36 / 0.47; stage 1 gần như không đổi. D_4/D_1: ConvNeXt 1.30 → 0.86 / 1.12; Swin 1.53 → 1.03 / 1.38.

**Sàng lọc dev300** (`scripts/pilot_r1_screen.py` → `results/r1_pilot/dev300/{screen.json, E0,E1a,E1b,E2/...}`; E0 = PNG OSFD của run KG cùng seed; run `metrics_sgm.json` / `generalization_sgm.json`). Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- E0 OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | **73.6**
- E1a SGM γ 0.2: 79.4 / 58.2 / 37.8 / 42.4 / 39.2 / 11.0 | **39.8**
- E1b SGM γ 0.5: 97.1 / 87.9 / 68.3 / 67.8 / 63.1 / 20.9 | **66.4**
- E2 mix {0.2, 0.5}: 93.5 / 79.0 / 56.6 / 59.1 / 54.0 / 16.5 | **56.6**

Δ CrossAvg₃ [95% CI]: E1a − E0 **−33.8 [−36.8, −30.8]**; E1b − E0 **−7.2 [−8.6, −5.5]**; E2 − E0 −17.0 [−19.1, −14.8]; E2 − E1b −9.8; E1a − E1b −26.6. Mọi target, mọi điều kiện SGM đều GIẢM, CI loại 0.

**Áp quy tắc khóa:** E_best = E1b (γ 0.5); ΔCrossAvg₃ −7.2 (cần ≥ 3) ✗, ConvNeXt −9.1 ✗, Swin −6.8 ✗, YOLOX-S −5.6 ✗, R101 −5.1 (cần ≥ −3) ✗ → **NO-GO R1**. Theo cam kết đã khóa: **dừng RQ3, chuyển trọng tâm sang analysis paper.** Không thử γ/biến thể khác (LinBP, PAS…).

Diễn giải (mô tả, không đổi kết luận):
- **Hiệu ứng liều–đáp ứng đơn điệu:** γ càng nhỏ (càng giảm gradient nhánh residual) transfer càng giảm, ở mọi target kể cả cùng họ và cả white-box R50 (98.1 → 97.1 → 79.4). Ngược với SGM ở classification (Wu et al. báo tăng transfer). Với OSFD trên detector, gradient qua nhánh residual của R50 là cần thiết — không phải "overfit đặc thù ResNet" cần triệt.
- Nhất quán với chẩn đoán D_k (chạy trước): SGM làm δ phá feature sâu kém đi ở MỌI model, không riêng R101; và với dịch chuyển tỉ trọng gradient về stage 2–3 (A3: tấn công stage nông yếu hơn). Không tách được phần nào do "giảm residual" và phần nào do "dời trọng tâm stage" — ghi nhận.
- Đây là thêm 1 kết quả can thiệp âm cho RQ2: giảm đường gradient residual KHÔNG thu hẹp gap họ backbone (gap R101 − CrossAvg₃: E0 19.4, E1b 21.5, E1a 18.4 — gap gần như không đổi trong khi mọi drop đều giảm).
- Compute như nhau (B = 50, 2 view/step; runtime 3.3–3.5 s/ảnh); L∞ = 5; E2 dùng γ 0.2 ở 49.9% step.

**Tổng kết RQ3** (theo cam kết, đóng): A′1 (trọng số không gian trong loss) NO-GO @ n300; P1 bỏ (≈ OSFD); P2 (gộp lõm theo channel) dừng ở gate; KG (view cặp object) NO-GO @ dev300, làm giảm transfer; R1 (SGM backward) NO-GO @ dev300, giảm mạnh theo liều. Không can thiệp nào thu hẹp phần gap còn lại của OSFD.

Artifact CHƯA upload HF: `artifacts/runs/dev300_B50_eps5/` (PNG OSFD, KG-E1/E2, SGM-0.2/0.5/mix; `dets_kg.json`, `dets_sgm.json`, `gen_dets/`).

---

## 2026-10-03 — Mở lại RQ3 theo plan mới: ablation OSFD → cơ chế → method (`docs/plan_sota.md`, LOCKED)

**Quyết định (user):** mở lại RQ3 dù entry R1 đã cam kết đóng. Lý do: 5 pilot trước (A′1, P1, P2, KG, R1) đều đoán cơ chế rồi can thiệp, đều thất bại. Hướng mới chỉ phân tích SOTA (OSFD): bỏ/thay đúng 1 thành phần mỗi ô (7 ô: task loss, bỏ RRB, RRB bỏ nhiễu / xoay / resize, bỏ MI, k = 5), đo tụt bao nhiêu → "nguyên liệu" → suy cơ chế (forward trên PNG) → method khuếch đại nguyên liệu → GO @ n300 → n1000. Ablation có giá trị cho analysis paper kể cả khi không ra method.

Chạy 2 bậc: dev100 trước (seed theo chỉ số dev300), ô mơ hồ mới nối đủ dev300. Câu hỏi phụ DINO-Swin-L: chỉ mô tả (eval mọi ô + D_k trên backbone Swin-L), không thêm model. Chi tiết cổng và tiêu chí: `docs/plan_sota.md`.

---

## 2026-10-03 — Ablation OSFD bước 1: code + unit test (trước khi chạy)

- `attack/methods/rrb.py`: tách rút tham số (`_sample_rotation`, `_sample_resize`) khỏi áp biến đổi; `rrb_views` thêm `use_rotate/use_resize/use_noise` (tắt resize: view2 = lần xoay MỚI của adv) và `with_gt` (co-transform GT box + mask cho ô task loss).
- `attack/methods/baselines.py`: `osfd_attack(loss="task", rrb_rotate/resize/noise)`.
- `scripts/run_baselines.py`: 7 ô `ABL-*`; `--subset dev100` chạy trong run dev300 với seed theo chỉ số dev300. `scripts/eval_generalization.py`: `--subset`, resume theo (điều kiện, ảnh) qua khóa `_done` (điều kiện cũ coi như đủ cả split) — bậc 2 chỉ predict ảnh còn thiếu.
- Unit test: (1) `rrb_views` cờ mặc định trùng TUYỆT ĐỐI bản cũ (lấy từ git, 3 ảnh × 3 seed); ảnh ở chế độ `with_gt` trùng ảnh mặc định. (2) Co-transform: IoU(bbox của mask, box) — không biến đổi 0.958, chỉ resize 0.955 (khớp), chỉ xoay 0.834 (AABB của box xoay lỏng hơn mask, như AugTrans trước đây); kiểm chiều xoay: tỉ lệ pixel mask sau xoay nằm trong box đã xoay trung vị 1.000, p5 1.000, min 0.917 (n = 628). (3) Cả 7 ô chạy được, L∞ đúng.

---

## 2026-10-03 — Ablation OSFD bậc 1 (dev100) → RRB là nguyên liệu chính, loss feature KHÔNG phải

Run `dev300_B50_eps5` subset dev100 (`metrics_abl_dev100.json`, `generalization_abl_dev100.json`), phân loại `scripts/ablation_screen.py` → `results/ablation/dev100/screen.json`. n = 100, paired bootstrap 1000 mẫu.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5 | 0.83
- task loss (+ MI + RRB co-transform GT): 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8 | 0.83
- bỏ RRB: 99.6 / 84.8 / 52.8 / 41.1 / 33.2 / 13.3 | **42.4** | 0.55
- RRB bỏ nhiễu: 96.4 / 94.2 / 80.5 / 75.8 / **61.0** / 22.2 | 72.4 | 0.83
- RRB bỏ xoay: 97.0 / 92.4 / 73.8 / 67.0 / 65.9 / 19.2 | **68.9** | 0.76
- RRB bỏ resize: 96.7 / 92.3 / 79.0 / 75.7 / 67.8 / 24.3 | 74.2 | 0.84
- bỏ MI: 96.8 / 90.4 / 77.2 / 71.5 / 68.1 / 22.4 | 72.3 | 0.82
- k = 5: 96.6 / 90.9 / 78.0 / 74.4 / 69.6 / 23.9 | 74.0 | 0.84

Δ = OSFD − ô trên CrossAvg₃ [95% CI] → phân loại (quy tắc khóa):
- bỏ RRB **+32.1 [26.3, 37.8] → nguyên liệu**
- RRB bỏ xoay **+5.5 [3.3, 8.7] → nguyên liệu**
- RRB bỏ resize +0.3 [−1.4, 1.8] → không phải
- k = 5 +0.4 [−1.1, 2.0] → không phải
- task loss +0.7 [−2.8, 5.6] → mơ hồ
- RRB bỏ nhiễu +2.0 [−0.2, 4.4] → mơ hồ
- bỏ MI +2.2 [1.1, 4.1] → mơ hồ
→ Bậc 2 (đủ dev300) cho 3 ô mơ hồ: đang chạy (`results/ablation/run_dev300.log`).

Quan sát (mô tả, chưa kết luận cuối — chờ bậc 2):
- **Ưu thế của OSFD đến từ RRB, không từ loss feature:** task loss + RRB ≈ OSFD (73.8 vs 74.5). Hệ quả: OSFD thắng M-DI² (+7 CrossAvg₂ @ n300) chủ yếu vì RRB > DI, không vì loss. Bỏ RRB (1 view, không biến đổi) làm η tụt 0.83 → 0.55.
- Trong RRB: **xoay** là phần đơn lẻ quan trọng nhất; resize không cần (2 view xoay độc lập ≈ OSFD). Bỏ nhiễu làm YOLOX-S tụt mạnh (−9.2) nhưng R101 tăng (+2.4) — nhiễu có thể giúp riêng target khác pipeline (YOLOX resize 640).
- Lưu ý compute: ô bỏ RRB chỉ 1 view forward/step (OSFD 2). Nhưng 2 view giống hệt nhau cho cùng gradient với 1 view, nên phần chênh đến từ ĐA DẠNG biến đổi, không từ số forward thuần.
- **DINO-Swin-L (câu hỏi phụ):** task loss + RRB = 33.8 vs OSFD 25.3, Δ −8.5 [−11.7, −1.6] (CI loại 0) — với ca khó, loss task tốt hơn loss feature; bỏ xoay làm DINO-Swin-L tụt 6.1. Mô tả, không thuộc cổng.

---

## 2026-10-03 — Ablation OSFD bậc 2 (dev300, 3 ô mơ hồ) → kết luận Cổng 1

Run `dev300_B50_eps5` đủ 300 ảnh (`metrics_abl_dev300.json`, `generalization_abl_dev300.json`; bậc 2 chỉ attack/predict 200 ảnh mới), `scripts/ablation_screen.py --tag dev300` → `results/ablation/dev300/screen.json`.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | 73.6 | 0.82
- task loss + MI + RRB: 99.3 / 95.6 / **85.0** / **78.8** / 69.4 / **34.9** | **77.7** | 0.86
- RRB bỏ nhiễu: 98.7 / 95.6 / 81.5 / 76.5 / **59.7** / 21.5 | 72.5 | 0.83
- bỏ MI: 98.6 / 91.8 / 74.9 / 70.7 / 64.1 / 21.1 | 69.9 | 0.79

Δ = OSFD − ô trên CrossAvg₃ [95% CI]: task loss **−4.1 [−5.4, +0.3]**; bỏ nhiễu +1.0 [−0.4, 2.3]; bỏ MI **+3.7 [2.5, 4.7]**.

**Cổng 1 (quyết 1 lần ở 300 ảnh, nhị phân: nguyên liệu ⇔ cận dưới ≥ 2):**
- **Nguyên liệu: RRB (bậc 1, +32.1), xoay trong RRB (bậc 1, +5.5), MI (bậc 2, +3.7).**
- Không phải: resize trong RRB, k = 5, nhiễu σ (bậc 2, cận dưới −0.4), **loss feature** (thay bằng task loss không làm yếu — ngược lại còn cao hơn 4.1 điểm, CI vừa chạm 0).

Quan sát (mô tả):
- **Task loss + MI + RRB ≥ OSFD**: CrossAvg₃ +4.1 (CI [−0.3, 5.4] theo chiều ô − OSFD), ConvNeXt +7.6, Swin +4.2, DINO-Swin-L **+10.1 [5.8, 12.1]**, R101 +2.6. Feature loss của OSFD không những không phải nguồn ưu thế mà có thể kém task loss khi đã có RRB.
- Nhiễu σ: không đóng góp CrossAvg₃ nhưng đổi hướng theo target — bỏ nhiễu YOLOX-S −9.0, R101 +2.6, ConvNeXt +4.1, DINO-Swin-L −3.4. Có thể nhiễu giúp target khác pipeline (YOLOX 640) và ca khó, hại ConvNeXt/R101.
- Bức tranh: ưu thế OSFD ≈ đa dạng biến đổi hình học (xoay) + momentum; loại loss không quan trọng (task loss thậm chí tốt hơn).

Tiếp theo theo plan: bước 2 (cơ chế, chỉ forward). Cần user quyết trước bước 3: ô task loss không phải "khuếch đại nguyên liệu" nên theo quy tắc khóa KHÔNG tự động là ứng viên.

---

## 2026-10-03 — Bước 2 plan SOTA: cơ chế từ ablation OSFD (chỉ forward, dev100)

`scripts/mech_ablation.py` → `results/ablation/mech/mech.json` (+ `mech_run.log`). 100 ảnh dev100; 8 điều kiện chính (OSFD + 7 ô) + 5 mô tả thêm (KG-E1/E2, SGM ×3). D_k trên backbone R50 / R101 / ConvNeXt-T / Swin-T / DINO-Swin-L (cùng Resize 1333×800 + normalize); phổ δ 3 dải; năng lượng δ trong GT box.

**1. Nguyên liệu đổi gì ở δ** (OSFD − ô, CI 95% bootstrap ảnh):
- **RRB:** năng lượng δ dải thấp **+0.124** (0.26 → 0.39), dải cao −0.143; D_4 trên CHÍNH R50 **−0.37** (1.86 → 1.49) nhưng D_4 trên target khác họ **tăng**: ConvNeXt +0.17, Swin-T +0.17, DINO-Swin-L +0.19; D_4/D_1 target +0.43 / +0.56.
- **Xoay:** cùng chiều, nhỏ hơn — dải thấp +0.016, D_4 R50 −0.03, D_4 target +0.035 / +0.035 / +0.059.
- **MI:** KHÁC cơ chế — dải thấp −0.026, D_4 target gần như không đổi (ConvNeXt −0.015, Swin −0.003, Swin-L +0.017). Mô tả hiện có chưa giải thích được đóng góp +3.7 của MI.

**2. Tỉ số chuyển distortion sâu D_4(target) / D_4(R50):** OSFD ConvNeXt 0.55 / Swin 0.47 / Swin-L 0.61; bỏ RRB 0.35 / 0.28 / 0.39; bỏ xoay 0.52 / 0.44 / 0.56. → RRB/xoay làm δ BỚT ĐẶC THÙ surrogate: ít phá feature sâu của R50 hơn nhưng phá feature sâu của target khác họ nhiều hơn.

**3. Theo ảnh (trong OSFD):** Spearman(D_4 trên target, retained confidence) −0.54 đến −0.69 ở ConvNeXt-T, Swin-T, YOLOX-S (D_4 R50 chỉ −0.03 đến −0.19; phổ thấp −0.15 đến −0.32) → distortion sâu TRÊN TARGET quyết định suppression; distortion trên surrogate thì không.

**4. Giữa điều kiện** (13 điều kiện, Spearman với CrossAvg₃): D_1 (mọi model) −0.80 [−0.87, −0.69]; D_4 DINO-Swin-L +0.79; phổ thấp +0.68 [0.53, 0.80]. Với 8 điều kiện chính tương quan yếu hơn (ρ D_4 Swin-L +0.86 nhưng CI rộng [0.31, 0.98]).

**Ngoại lệ — task loss:** D_k thấp hơn OSFD ở MỌI model (D_4 ConvNeXt 0.57 vs 0.82, Swin-L 0.73 vs 0.91) nhưng CrossAvg₃ ngang và DINO-Swin-L cao hơn → task loss phá feature ÍT hơn mà vẫn lừa được bằng/hơn; độ lớn distortion feature không giải thích được khác biệt GIỮA loại loss.

**Câu cơ chế (cho bước 3):** *Đa dạng biến đổi (RRB, phần lớn qua xoay) đẩy δ về tần thấp và làm nó bớt đặc thù surrogate — giảm distortion sâu trên R50, tăng distortion sâu trên target khác họ; distortion sâu trên target là thứ quyết định suppression theo từng ảnh. MI và loại loss đóng góp theo cơ chế khác, chưa đo được bằng D_k/phổ.*

---

## 2026-10-03 — Vì sao DINO-Swin-L bền: distortion qua backbone → neck → encoder (DINO-Swin-L vs DINO-R50)

`scripts/mech_swinl.py` → `results/ablation/mech/swinl.json` (+ `swinl_run.log`); DINO-R50 eval dev100 → `results/runs/dev300_B50_eps5/generalization_dino_r50_dev100.json`. 100 ảnh dev100; OSFD / task loss / bỏ RRB. Chỉ số: rel = ‖a − c‖/‖c‖ và **cosd = 1 − cos(a, c)** (không phụ thuộc thang đo kiến trúc — rel so giữa kiến trúc dễ gây hiểu nhầm).

OSFD, cosd theo tầng (cùng detector DINO, chỉ khác backbone):

| | backbone tầng cuối | neck (lớn nhất) | **encoder (memory)** | retained conf. | drop AP |
|---|---|---|---|---|---|
| DINO-R50 | **0.873** | 0.840 | **0.324** | 0.06 | 96.0 |
| DINO-Swin-L | **0.424** | 0.356 | **0.066** | 0.76 | 25.3 |
| Mask R-CNN Swin-T (tham chiếu) | 0.244 | **0.515** (FPN khuếch đại) | — | 0.32 | 74.3 |

- **Đính chính bước 2:** "D_4 Swin-L 0.91 > Swin-T 0.70" là theo rel; theo cosd, feature sâu của Swin-L vẫn còn khá thẳng hàng với bản sạch (cos ≈ 0.58), trong khi R50 gần như trực giao (cos ≈ 0.13).
- **Encoder DINO dập distortion:** tỉ lệ cosd encoder / backbone cuối: Swin-L 0.16, R50 0.37. Stage nông của Swin-L gần như không bị chạm (cosd 0.07 / 0.11).
- **Neck đối lập:** FPN của Mask R-CNN Swin-T KHUẾCH ĐẠI distortion (0.24 → 0.49–0.52); ChannelMapper + encoder của DINO-Swin-L thì giảm.
- Theo ảnh (DINO-Swin-L): distortion ở encoder dự đoán retained mạnh nhất (ρ −0.44; backbone −0.30 đến −0.36).
- **Task loss trên DINO-Swin-L:** retained như OSFD (0.76) nhưng drop AP cao hơn (33.8 vs 25.3 @ dev100) → phần lợi của task loss ở ca khó KHÔNG đến từ làm mất GT mà từ false positive / sai định vị.

**Diễn giải (giả thuyết, mô tả):** Swin-L bền do 2 yếu tố CỘNG LẠI: (1) attack từ R50 chỉ làm lệch vừa phải feature sâu của Swin-L và gần như không chạm stage nông; (2) encoder deformable của DINO (đa scale, dùng cả stage nông còn sạch) "sửa" được distortion vừa phải. Với DINO-R50, distortion gần như toàn phần nên vượt qua được encoder. → Kết luận cũ "detector không giải thích failure" (dựa trên DINO-R50 97%) cần thu hẹp: detector không gây failure KHI distortion bão hòa (cùng họ), nhưng có thể góp phần khi distortion chỉ vừa phải (khác họ). Không kiểm được trực tiếp nếu không có DINO-Swin-T (không thêm model mới — chỉ ghi nhận).

---

## 2026-10-03 — Literature + probe hướng "distortion bất biến" → KHÔNG ủng hộ

**Rà literature (web, 2026-10-03):** input-transformation cho transfer đã đông ở classification (SIA ICCV'23, BSR, SID ICCV'25, Block Stretch&Shrink 11/2025, Local Invariance 2025, Perspective-Invariant Attack 8/2026) → "tăng xoay / thêm biến đổi" không có novelty; ở detection gần như chỉ có RRB (OSFD) và AugTrans. Tấn công encoder DETR ("Corrupting Attention", arXiv 2608.06674, có vẻ white-box trên DETR) khớp phát hiện encoder là chỗ "khóa" của DINO nhưng không áp dụng được với surrogate Mask R-CNN R50. Chưa thấy method detection 1-surrogate công khai vượt OSFD (rà chưa đầy đủ). Hướng được chọn để probe (user đồng ý): "distortion bất biến" tường minh.

**Probe** (`scripts/probe_invariance.py` → `results/ablation/mech/invariance_probe.json`; 100 ảnh dev100, 13 điều kiện; 8 góc xoay ±7° áp cho cả sạch/adv, feature R50 xoay ngược về cùng khung, vùng hợp lệ chung): E = mean_t ‖Δ_t‖², S = ‖mean_t Δ_t‖², r = S/E.
- r CAO ở MỌI điều kiện (stage 4: 0.77–0.82), kể cả bỏ RRB (0.769 vs OSFD 0.815) — chênh 0.046 trong khi CrossAvg₃ chênh 32 điểm.
- Giữa điều kiện: Spearman(r stage 4, CrossAvg₃) = 0.74 (13 điểm), nhưng biên độ nhỏ; r stage 1 tương quan âm (−0.59).
- Theo ảnh (OSFD): phần chung S dự đoán retained KHÔNG tốt hơn tổng E (vd stage 2 ConvNeXt −0.35 vs −0.35; YOLOX −0.40 vs −0.36); tỉ lệ r còn yếu hơn.
→ Distortion trên surrogate vốn đã ~80% "chung giữa các góc xoay nhỏ" dù không có RRB; phần chung không phải thứ phân biệt δ transfer tốt với δ transfer kém. **Không có cơ sở cơ chế cho hướng "distortion bất biến" tường minh** (trong phạm vi probe: xoay ±7°, đo trên surrogate). Không khóa ứng viên theo hướng này.

---

## 2026-10-03 — Probe loại lỗi (mất GT / false positive / định vị) → hallucination KHÔNG qua được DINO; lợi của task loss chủ yếu là phá ĐỊNH VỊ

`scripts/probe_error_types.py` → `results/ablation/mech/error_types.json` (+ `error_types_run.log`); dev300, OSFD vs task loss + MI + RRB, từ dets có sẵn (không GPU). Score ≥ 0.3; FP tách nền (IoU < 0.1) / định vị (cùng lớp, 0.1 ≤ IoU < 0.5) / nhầm lớp; AP_noFP = AP khi bỏ mọi FP.

- **Hallucination đã là kiểu phá chính của OSFD trên target kiểu Mask R-CNN/YOLOX** — FP nền/ảnh: R50 2.9 → 41.0, R101 → 28.5, ConvNeXt → 23.4, Swin-T → 18.2, YOLOX-S 0.9 → 9.5 — **nhưng DINO-Swin-L gần như miễn nhiễm**: 1.1 → 1.8 (task loss 2.7). → hướng "hallucination chuyển giao được" KHÔNG nhắm được ca khó; bỏ.
- DINO-Swin-L, OSFD: recall@0.3 0.76 → 0.57, FP gần như không tăng — thiệt hại đến từ mất GT.
- **Task loss phá định vị mạnh hơn ở MỌI target khác họ:** AP/AP50 — ConvNeXt 0.62 (OSFD) vs **0.49**; Swin-T 0.56 vs **0.50**; YOLOX-S 0.63 vs **0.55**; DINO-Swin-L 0.74 vs **0.72**; FP định vị/ảnh cũng tăng (ConvNeXt 2.0 → 3.2, Swin-T 1.9 → 3.1, YOLOX 0.8 → 1.7, Swin-L 1.0 → 1.4). Recall@0.3 của task loss KHÔNG thấp hơn OSFD (ConvNeXt 0.32 vs 0.26; Swin-L 0.56 vs 0.57) → task loss làm mất GT ÍT hơn nhưng làm lệch box NHIỀU hơn.
- DINO-Swin-L: AP drop task loss 34.9 vs OSFD 24.9; AP50 drop 29.5 vs 20.7 — vừa định vị vừa thêm FP (nền + định vị).

**Diễn giải (giả thuyết):** "đồng tiền" chuyển giao sang target khác họ / ca khó không phải hallucination mà là **phá định vị (bbox regression)** — có thể vì tín hiệu định vị dựa trên cấu trúc biên/hình học object chung giữa các kiến trúc. Cần kiểm: thành phần nào của task loss (rpn_cls, rpn_bbox, cls, bbox, mask) gây ra lợi ích.

---

## 2026-10-03 — Ablation thành phần task loss (trước khi chạy)

User đồng ý (mở rộng phân tích bước 2, chưa phải ứng viên method): nền = task loss + MI + RRB (co-transform GT); 3 ô chỉ giữ 1 nhóm số hạng: `ABL-task-cls` (loss_rpn_cls + loss_cls), `ABL-task-box` (loss_rpn_bbox + loss_bbox), `ABL-task-mask` (loss_mask). dev100 (seed theo chỉ số dev300), đánh giá 6 model + AP/AP50. `attack/preprocess.make_gt_loss(terms)`; unit test: 5 key trùng tuyệt đối `compute_gt_loss`.

---

## 2026-10-03 — Ablation thành phần task loss (dev100) → lợi ở DINO-Swin-L đến từ số hạng PHÂN LOẠI, không từ định vị

Run `dev300_B50_eps5` subset dev100 (`metrics_task_dev100.json`, `generalization_task_dev100.json`), `scripts/ablation_screen.py --tag task_dev100` → `results/ablation/task_dev100/{screen.json, error_types.json}`.

Relative AP drop % (dev100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5
- task loss đủ 5 số hạng: 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8
- chỉ cls (rpn_cls + cls): 98.7 / 90.8 / 78.7 / 75.7 / 59.1 / **32.5** | 71.2
- chỉ box (rpn_bbox + bbox): 95.4 / 83.6 / 65.8 / 61.5 / 55.8 / 25.1 | 61.0
- chỉ mask: 89.7 / 77.2 / 63.9 / 57.0 / 49.5 / 26.0 | 56.8

Loại lỗi (dev100): chỉ-box làm AP/AP50 thấp nhất (ConvNeXt 0.44, Swin-T 0.48, YOLOX 0.54) và tăng FP định vị, nhưng giữ recall cao (0.45–0.49) và gần như không tạo FP nền → **phá định vị có thật nhưng đóng góp ít vào AP drop tổng**. Chỉ-cls giữ phần lớn hiệu quả của task loss (CrossAvg₃ 71.2 vs 73.8), tạo FP nền trên target kiểu Mask R-CNN, và trên DINO-Swin-L cho AP 43.2 (≈ đủ 42.4; OSFD 47.8) với AP50 thấp nhất (57.1) mà recall như OSFD (0.56 vs 0.57) → hạ chất lượng xếp hạng score / thêm FP chứ không xóa GT.

**Đính chính giả thuyết entry trước:** "phá định vị là kênh chuyển giao" KHÔNG đứng — lợi của task loss ở ca khó đến từ **số hạng phân loại** (rpn_cls + cls). Phá định vị (box) chỉ đóng góp nhỏ.

**Bổ trợ theo target (mô tả):** loss feature (OSFD) tốt nhất ở YOLOX-S (70.2 vs cls 59.1); loss phân loại tốt nhất ở DINO-Swin-L (32.5–33.8 vs 25.3); ConvNeXt/Swin-T ngang nhau → hai loại objective có thể BỔ TRỢ nhau.

---

## 2026-10-03 — Probe bổ trợ "feature + cls" (trước khi chạy)

User đồng ý probe rẻ: `PROBE-osfd+cls` = MI + RRB (co-transform GT) + L_OSFD + λ·(loss_rpn_cls + loss_cls), 1 forward/view (feature backbone bắt bằng hook trong `model.loss`; kiểm: feature-only qua hook trùng OSFD gốc, sai số 1.5e-7). **λ = 4 khóa trước khi chạy**: median tỉ lệ mean|∇feat| / mean|∇cls| tại δ = 0 trên 5 ảnh đầu dev100 = 3.95 (3.7 / 4.57 / 3.95 / 7.7 / 2.15) → hai thành phần đóng góp gradient ngang nhau; không tune λ theo transfer. dev100, so OSFD / task loss / chỉ-cls. Câu hỏi mô tả: có giữ được YOLOX-S (feature) và tăng DINO-Swin-L (cls) cùng lúc không.

---

## 2026-10-03 — Probe "feature + cls" (dev100) → có dấu hiệu bổ trợ, chưa có ý nghĩa thống kê ở 100 ảnh

`results/ablation/probe_dev100/screen.json` (run `metrics_probe_dev100.json`, `generalization_probe_dev100.json`). Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / **70.2** / 25.3 | 74.5
- task loss: 98.6 / 92.5 / 80.1 / 73.7 / 67.6 / **33.8** | 73.8
- chỉ cls: 98.7 / 90.8 / 78.7 / 75.7 / 59.1 / 32.5 | 71.2
- **OSFD + 4·cls: 99.3 / 94.5 / 81.1 / 80.2 / 69.9 / 30.5 | 77.0**

So OSFD: CrossAvg₃ **+2.6 [−1.7, +5.3]**, Swin-T +5.9, ConvNeXt +2.3, YOLOX-S −0.3 (giữ được), DINO-Swin-L +5.2 [−0.8, 8.4], R101 +2.7. → Bổ trợ đúng chiều dự đoán (giữ YOLOX của loss feature, tăng DINO-Swin-L/Swin-T nhờ cls) nhưng ở n = 100 CI còn chứa 0; DINO-Swin-L thấp hơn task loss / chỉ cls (30.5 vs 33.8 / 32.5).
Lưu ý: trên dev300, task loss + MI + RRB đã hơn OSFD +4.1 (dev100 chỉ −0.7) → dev100 nhiễu; ứng viên kết hợp phải vượt CẢ OSFD lẫn task loss + RRB.

---

## 2026-10-03 — Bước 3 plan SOTA: khai báo ứng viên (TRƯỚC khi chạy sàng lọc dev300)

User đồng ý. Đúng 2 ứng viên, cấu hình đóng băng, không thêm ứng viên cứu vãn:
- **C1** = task loss (5 loss Mask R-CNN) + MI μ = 1 + RRB (co-transform GT box + mask) — `ABL-task-loss` (đã có đủ dev300).
- **C2** = MI + RRB (co-transform GT) + L_OSFD + 4·(loss_rpn_cls + loss_cls), 1 forward/view — `PROBE-osfd+cls` (λ = 4 khóa trước theo tỉ lệ gradient; nối 200 ảnh còn lại của dev300, cùng seed).
Ghi chú sửa plan (user duyệt): plan yêu cầu method "khuếch đại nguyên liệu"; C1/C2 thay/bổ sung LOẠI LOSS, dựa trên phát hiện bước 2 mở rộng (feature loss mạnh ở target kiểu CNN/YOLOX, loss phân loại mạnh ở target có encoder tự sửa — DINO-Swin-L), giữ nguyên nguyên liệu RRB + MI.

**Sàng lọc @ dev300 (tiêu chí khóa plan bước 3)**, mỗi ứng viên so OSFD, paired bootstrap 1000 mẫu: QUA nếu ΔCrossAvg₃ ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S; drop R101 ≥ R101(OSFD) − 3. Cả 2 qua → chọn ΔCrossAvg₃ lớn hơn (chênh < 0.5 → C1, đơn giản hơn). Báo cáo kèm C2 − C1 (CI) và DINO-Swin-L. Không qua cả 2 → dừng method.

---

## 2026-10-03 — Bước 3 sàng lọc @ dev300 → **C2 (OSFD + 4·cls) QUA**, C1 không qua (CI)

Run `dev300_B50_eps5` (`metrics_step3_dev300.json`, `generalization_step3_dev300.json`); `scripts/ablation_screen.py --tag step3_dev300` + `scripts/step3_screen.py` → `results/ablation/step3_dev300/{screen.json, step3.json}`. n = 300, paired bootstrap 1000 mẫu.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 98.1 / 93.0 / 77.4 / 74.6 / 68.7 / 24.9 | 73.6 | 0.82
- C1 task loss + RRB: 99.3 / 95.6 / 85.0 / 78.8 / 69.4 / 34.9 | 77.7 | 0.86
- **C2 OSFD + 4·cls: 99.5 / 96.6 / 84.7 / 79.5 / 70.6 / 31.8 | 78.3 | 0.85**

Ứng viên − OSFD [95% CI]:
- C1: CrossAvg₃ +4.1 [−0.3, 5.4] ✗ (cận dưới ≤ 0); ConvNeXt +7.6, Swin +4.1, YOLOX +0.8, R101 +2.6, DINO-Swin-L +10.1 [5.8, 12.1] → **KHÔNG QUA**.
- **C2: CrossAvg₃ +4.7 [1.7, 6.0] ✓; ConvNeXt +7.2 [3.6, 9.1], Swin-T +4.9 [1.5, 6.5], YOLOX-S +1.9 [−2.2, 4.8] (điểm > 0 ✓), R101 +3.6 ✓, DINO-Swin-L +7.0 [2.8, 8.9] (báo cáo) → QUA.**
- C2 − C1 (điểm): CrossAvg₃ +0.6, YOLOX +1.2, DINO-Swin-L −3.1.

**Theo quy tắc khóa: chọn C2, ĐÓNG BĂNG** (MI μ = 1 + RRB θ = 7, l_s = 10, ρ = 0.8, s_max = 1.1, σ = 6 với co-transform GT + L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, B = 50, ε = 5). Bước tiếp: GO @ n300 (1 lần) so với PNG OSFD của run `n300_B50_eps5` (HF).
Ghi chú: CI bootstrap percentile lệch (điểm +4.7 gần cận trên 6.0) — cùng hiện tượng đã ghi ở baseline (bootstrap mean lệch do category hiếm); không đổi kết luận.

---

## 2026-10-03 — Bước 4: GO @ n300 cho C2 (bắt đầu chạy)

Tải `n300_B50_eps5` từ HF (token đọc từ `.env`, không in ra; `.env` thêm vào `.gitignore`), sha256 9eadf774…f8d590 khớp entry 2026-09-23; đủ 300 PNG OSFD. Chạy `PROBE-osfd+cls` (C2, đóng băng) trên n300 (seed SEED + i như baseline), OSFD dùng lại PNG baseline (không attack lại). Tiêu chí GO khóa (plan_sota.md bước 4): ΔCrossAvg₃ ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S VÀ ≥ 2/3 target ≥ 3; drop R101 ≥ R101(OSFD) − 3. Chạy 1 lần.

---

## 2026-10-03 — Tạm hoãn GO @ n300 (user); C2 ghi nhận tích cực; chuyển sang nhánh Swin-L

- **User quyết:** hủy GO đang chạy; **ghi nhận C2 là kết quả tích cực** (qua sàng lọc dev300: CrossAvg₃ +4.7 [1.7, 6.0]) — chạy lại GO khi cần. GO đã attack 110/300 ảnh n300 (PNG ở `artifacts/runs/n300_B50_eps5/adv/PROBE-osfd+cls/`, stats trong `results/runs/n300_B50_eps5/attack_stats.jsonl`); **CHƯA eval, chưa nhìn số nào** → GO vẫn là "chạy 1 lần" hợp lệ; chạy lại đúng lệnh trong CLAUDE.md sẽ nối tiếp.
- Việc mở cho C2 (để sau): đối chứng cân thời gian — C2 có cùng B và số view forward nhưng wall-clock ~2.3× OSFD (7.4 vs 3.3 s/ảnh, do `model.loss` chạy RPN/ROI/head); budget sweep n30 cho OSFD B50 → B100 +2.6 cross-avg → so OSFD cùng wall-clock (B ≈ 115) trên dev300, khóa trước khi chạy.
- **Nhánh mới — tấn công cả DINO-Swin-L:** chạy chẩn đoán miễn phí (chỉ forward): điểm phân loại các query top-k ở encoder DINO (cơ chế chọn query) thay đổi ra sao giữa sạch / OSFD / C1 / C2 / chỉ-cls / bỏ RRB — kiểm kênh "xếp hạng confidence" có phải đường lọt qua encoder không.

---

## 2026-10-03 — Chẩn đoán chọn query encoder DINO → Swin-L giữ 95% GT trong top-900; ý tưởng "phá xếp hạng confidence" KHÔNG được ủng hộ

`scripts/diag_query_select.py` → `results/ablation/mech/query_select.json` (+ log); 100 ảnh dev100; tái tạo `DINO.pre_decoder` (điểm encoder s = max_c sigmoid, top-900). (Lần chạy đầu bị kill ở 20/100 khi phiên khởi động lại; chạy lại trong tmux.)

| DINO | điều kiện | top100 mean | top100 còn giữ | điểm proposal GT | GT còn trong top-900 |
|---|---|---|---|---|---|
| Swin-L | sạch | 0.130 | 1.00 | 0.521 | 0.988 |
| Swin-L | OSFD / C1 / C2 / chỉ-cls | 0.141–0.145 | 0.32–0.41 | 0.409–0.428 | **0.946–0.956** |
| Swin-L | bỏ RRB | 0.127 | 0.49 | 0.482 | 0.976 |
| R50 | sạch | 0.143 | 1.00 | 0.404 | 0.971 |
| R50 | OSFD / C1 / C2 / chỉ-cls | 0.21–0.26 | 0.04–0.18 | 0.09–0.15 | **0.52–0.66** |

- DINO-R50 thua vì GT bị đẩy KHỎI top-900 (chỉ còn ~57%) → decoder không có query để khôi phục.
- DINO-Swin-L: thứ tự query bị xáo mạnh nhưng ~95% GT vẫn vào decoder (điểm proposal GT chỉ giảm ~20%) → decoder khôi phục.
- C1 / C2 / chỉ-cls KHÔNG khác OSFD ở tầng encoder → lợi +7 đến +10 của loss phân loại trên Swin-L nằm ở decoder / điểm cuối, không ở chọn query. Ý tưởng I1 (phá xếp hạng confidence encoder) bỏ.
- Điểm nghẽn Swin-L: phải hạ điểm proposal vùng GT đủ mạnh để GT rơi khỏi top-900.

---

## 2026-10-03 — Nhánh Swin-L, probe S1: C2 + lọc thông thấp gradient (trước khi chạy)

User chọn S1 sau rà literature (ViT dựa vào thông tin tần thấp / pha tần thấp — WACV 2024 "Exploring Adversarial Robustness of ViTs in the Spectral Perspective"; COGO ICLR 2024 nhắm tần trung-thấp; TESSER làm trơn phổ) + dữ liệu của ta (RRB đẩy δ về tần thấp; phổ thấp tương quan +0.68 với CrossAvg₃ giữa điều kiện). Phần "nhắm pha" để sau nếu S1 có tín hiệu.
- `core.run_iterative_attack(grad_transform=)`; `baselines.gaussian_lowpass(g, f0)`: H(f) = exp(−f²/(2 f0²)), áp lên gradient mỗi step TRƯỚC chuẩn hóa MI.
- 2 mức liều khóa trước: `S1-lp6` (f0 = 1/6 cycles/pixel) và `S1-lp12` (f0 = 1/12), nền = C2 nguyên vẹn. Kiểm: nhiễu trắng sau lọc có 63% / 98% năng lượng ở dải thấp (trước 9%).
- Unit test: C2 tự nó KHÔNG tất định (2 lần cùng seed lệch 4.95% pixel sau 3 step — ROIAlign backward GPU; OSFD tất định); `grad_lowpass_f0=None` lệch C2 5.07% — cùng mức → tương đương.
- dev100, so OSFD / C2; báo cáo 6 model, ưu tiên DINO-Swin-L và việc giữ ConvNeXt/Swin-T/YOLOX.

---

## 2026-10-03 — Probe S1 (C2 + lọc thông thấp gradient) → ÂM, giảm đơn điệu theo liều, kể cả DINO-Swin-L

`results/ablation/s1_dev100/screen.json` (run `metrics_s1_dev100.json`, `generalization_s1_dev100.json`). Relative AP drop % (dev100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.8 / 74.3 / 70.2 / 25.3 | 74.5
- C2: 99.3 / 94.5 / 81.1 / 80.2 / 69.9 / **30.5** | 77.0
- S1 f0 = 1/6: 98.3 / 90.5 / 74.6 / 70.0 / 64.5 / **22.7** | 69.7
- S1 f0 = 1/12: 95.9 / 78.2 / 55.6 / 56.5 / 56.7 / **15.5** | 56.2

→ Ép δ về tần thấp bằng lọc gradient làm attack YẾU đi đơn điệu theo liều ở MỌI target, kể cả DINO-Swin-L (30.5 → 22.7 → 15.5). Tương quan "phổ thấp ↔ transfer" giữa điều kiện (bước 2) KHÔNG phải nhân quả theo kiểu can thiệp này: tần thấp của RRB là hệ quả đi kèm, không phải nguồn lợi; bỏ tần trung/cao làm mất thông tin hữu ích. Nhánh S1 (kể cả biến thể "nhắm pha" chưa làm) KHÔNG đi tiếp.

---

## 2026-10-04 — Phân tích vì sao OSFD/C2 không thắng DINO-Swin-L + chốt quy tắc hướng "Repair-Robust Attack" (TRƯỚC khi code/chạy)

> **Trạng thái: LOCKED (user duyệt 2026-10-04).** Không đổi định nghĩa/ngưỡng sau khi thấy kết quả. Thứ tự ưu tiên: R-a chính, R-b dự phòng (khai báo ngay bây giờ, KHÔNG phải ứng viên cứu vãn).

### Phân tích (từ số đã commit: `results/ablation/mech/{swinl,query_select,error_types}.json`, dev100)

- **Điểm nghẽn ở sau backbone, không ở backbone:** cosd cuối backbone Swin-T (Mask R-CNN) **0.244 < Swin-L 0.424**, nhưng drop Swin-T 74% vs Swin-L 25%. FPN của Swin-T khuếch đại (→ 0.49–0.52); encoder DINO dập (Swin-L 0.424 → **0.066**, ×6.4; DINO-R50 0.873 → 0.324, ×2.7).
- **Stage nông Swin-L gần như sạch:** cosd stride 4 / 8 = 0.067 / 0.108 (OSFD; task loss, bỏ RRB tương tự) → encoder deformable đa scale có "mỏ neo sạch" để sửa distortion vừa phải ở stage sâu. Theo ảnh, distortion encoder dự đoán retained mạnh nhất (ρ −0.44).
- **Chọn query không bị phá:** GT còn trong top-900 0.95–0.96 ở mọi method (DINO-R50: 0.52–0.66); điểm proposal GT 0.52 → 0.41–0.43.
- **Budget hallucination lãng phí với DINO:** FP nền/ảnh trên Swin-L 1.1 → 1.8 (Mask R-CNN target 18–41). `loss_rpn_cls + loss_cls` của C2 (ascent) đẩy CẢ mẫu âm thành foreground.
- **Lợi của loss phân loại nằm sau encoder:** task loss có cosd encoder 0.033 (< OSFD 0.066) nhưng drop cao hơn (33.8 vs 25.3).
- Đã loại: trọng số không gian, gộp lõm channel, crop cặp KG, SGM, ép tần thấp (S1), distortion bất biến tường minh.
- Caveat giữ nguyên: sức bền Swin-L phần lớn gắn với scale + pretrain IN-22k/384 → "thắng Swin-L" = thắng ca generalization khó, không phải bằng chứng nhân quả họ backbone.

**Giả thuyết (chưa là cơ chế):** encoder DINO hoạt động như một bộ lọc nhiễu ngầm dùng các mức nông còn sạch. Mô phỏng phép "sửa" đó NGAY TRÊN surrogate R50 và tối ưu δ để tín hiệu phân loại sống sót qua phép sửa (EOT trên một "defense" là thiên kiến kiến trúc của target). Vẫn đúng threat model (1 surrogate, 0 query, không dùng weight target). Khác OSFD (đa dạng đầu vào) và C2 (trộn loss): đa dạng trong không gian FEATURE theo mô hình phép sửa. Rủi ro novelty: Clean Feature Mixup (CVPR'23), Ghost Networks — phải rà literature trước GO @ n300.

### Probe phụ V — loss phân loại "chỉ đẩy GT về nền" (vanishing-only)

- **Mục đích:** tách hiệu ứng "bỏ budget hallucination" khỏi hiệu ứng phép sửa; quyết định thành phần phân loại cho R-a. Không phải đóng góp novelty.
- **Định nghĩa:** L_V = −[ mean_{a∈P_rpn} BCE(o_a, 0) + mean_{r∈P_roi} CE(z_r, background) ]; P = mọi anchor/RoI dương do assigner train_cfg của surrogate gán (RPN IoU ≥ 0.7, RCNN IoU ≥ 0.5, kể cả GT-as-proposal), KHÔNG lấy mẫu âm, KHÔNG có số hạng đẩy mẫu âm. Ascent trên L_V = kéo objectness/lớp của vùng GT về nền.
- **Phương thức:** C2 thay `(loss_rpn_cls + loss_cls)` bằng L_V; mọi thứ khác giữ C2 (MI μ = 1, RRB đủ + co-transform GT, L_OSFD k = 3, 1 forward/view, B = 50, ε = 5, α = 1, noise init 0, seed ảnh SEED + chỉ số dev300).
- **λ_V theo ĐÚNG luật đã dùng cho C2** (khóa trước): median tỉ lệ mean|∇ₓL_OSFD| / mean|∇ₓL_V| tại δ = 0 trên 5 ảnh đầu dev100, làm tròn số nguyên gần nhất (≥ 1); ghi số vào log trước khi chạy attack.
- **Unit test:** P khớp tập dương của assigner chuẩn; L_V tại δ = 0 hữu hạn; ảnh không có mẫu dương → L_V = 0 (chỉ còn L_OSFD).
- **Chạy dev100**, so C2 (cùng seed). **Quy tắc chọn (điểm ước lượng):** dùng L_V làm thành phần phân loại cho R-a nếu Δ DINO-Swin-L(V − C2) ≥ +2 VÀ ΔCrossAvg₃(V − C2) ≥ −2; ngược lại R-a dùng `loss_rpn_cls + loss_cls` của C2. Báo cáo CI + loại lỗi (FP nền, recall@0.3) trên 6 model.

### R-a — thay mức nông bằng bản sạch trước FPN (ứng viên chính)

- **Phép sửa R_a:** sau backbone R50, thay C2, C3 (stride 4, 8 — ứng với stage Swin-L cosd 0.067 / 0.108) bằng feature SẠCH (detach) rồi mới qua FPN → RPN/ROI. C4, C5 giữ adv.
- **Objective:** L = L_OSFD(backbone adv, đủ 4 stage, k = 3, KHÔNG qua phép sửa) + λ·L_cls(head(FPN(R_a(C_adv)))), L_cls = thành phần chọn theo probe V; λ thừa kế (tại δ = 0 phép sửa không đổi gì → luật λ cho cùng giá trị).
- **Ngẫu nhiên hóa (liều):** mỗi step 1 lần Bernoulli(p) (numpy RNG riêng, seed SEED + i, không đụng RNG của RRB); trúng → thay CẢ C2 và C3 ở mọi view của step đó. **Đúng 2 liều: p = 0.5, p = 1.0** (p = 0 = nền đã có).
- **Feature sạch đúng hình học view:** C2/C3 sạch tính trên ảnh SẠCH đã áp CÙNG phép xoay/resize của view RRB (không cộng nhiễu σ), no_grad; forward thêm không tính vào B — báo cáo số forward no_grad + runtime.
- **Sanity:** p = 0 trùng nền (C2 hoặc V) trong sai số không tất định đã biết (~5% pixel sau 3 step); tại δ = 0 loss R_a = loss nền; L∞ = 5.

### Cổng độ trung thực (forward-only, TRƯỚC mọi attack R-a)

PNG OSFD + C2 trên dev100 (sinh lại nếu thiếu, cùng seed) + dets DINO-Swin-L.
- Q_rep(ảnh) = retained confidence (định nghĩa `attack/mechanism.py`: GT detect được ở ảnh sạch, score ≥ 0.3, IoU ≥ 0.5) của **R50 khi C2, C3 thay bằng sạch** (p = 1, không RRB); Q_plain = cùng chỉ số với R50 thường.
- **QUA nếu ở CẢ OSFD và C2:** Spearman(Q_rep, retained DINO-Swin-L) ≥ +0.30 VÀ ρ(Q_rep) − ρ(Q_plain) ≥ +0.10 với cận dưới CI 95% paired bootstrap (1000 mẫu, BOOT_SEED 2026) > 0.
- Mô tả kèm: mean Q_rep vs Q_plain, cosd FPN có/không sửa, tương quan với Swin-T / ConvNeXt-T / YOLOX-S.
- **Trượt → R-b** (dự phòng, cùng cổng, q = 0.3). R-b cũng trượt → dừng hướng Repair-Robust.

### R-b — trộn sạch theo ô không gian (DỰ PHÒNG, chỉ khi R-a trượt cổng độ trung thực)

Chia khung model thành ô 64×64 px (căn theo stride: C2 16×16, C3 8×8, C4 4×4, C5 2×2 cell); mỗi ô thay bằng sạch với xác suất q, độc lập, CÙNG mặt nạ (theo tọa độ ảnh) cho cả 4 stage, rồi qua FPN → head. Objective/λ/RNG như R-a. Liều q ∈ {0.3, 0.5}. Nếu R-a qua cổng nhưng trượt các bước sau → KHÔNG chạy R-b.

### Liều–đáp ứng @ dev100 (R-a, hoặc R-b nếu thay)

So nền (p = 0) cùng seed. Đi tiếp nếu: ≥ 1 liều có Δ DINO-Swin-L ≥ +3 (điểm) VÀ liều đó có ΔCrossAvg₃ ≥ −2 VÀ ΔR101 ≥ −3. Chọn liều có DINO-Swin-L cao hơn (chênh < 1 → p = 0.5). Không qua → dừng hướng.

### Sàng lọc @ dev300 (1 lần, cấu hình đóng băng)

QUA nếu đồng thời:
- Δ DINO-Swin-L(ứng viên − C2) ≥ 3 VÀ cận dưới CI 95% > 0;
- ΔCrossAvg₃(ứng viên − C2) ≥ −1.5 (không thua C2, điểm);
- ΔCrossAvg₃(ứng viên − OSFD) ≥ 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S so OSFD;
- drop R101 ≥ R101(OSFD) − 3.
Không qua → dừng, ghi log, không biến thể mới. Qua → rà literature (CFM, Ghost Networks, feature-level augmentation cho detection) rồi GO @ n300 (cần hoàn tất C2 @ n300 trước để có mốc C2; tiêu chí GO khóa riêng trước khi chạy).

---

## 2026-10-04 — Repair-Robust: code + unit test + hiệu chỉnh λ; đổi cách làm tròn λ_V (user duyệt, TRƯỚC khi chạy V)

- **Môi trường:** máy mới RTX 3090, bootstrap xong; `download_checkpoints.sh --gen` đủ 7 checkpoint Generalization Panel; COCO val2017 5000 ảnh. Không có `artifacts/` (PNG dev300 cũ chưa từng lên HF → mất) → sinh lại PNG OSFD + C2 trên dev100 (cùng seed theo chỉ số dev300).
- **Code:** `attack/methods/repair.py` (`vanish_loss` = L_V; `make_view_loss` hook backbone: L_OSFD trên C_adv, nhánh head qua R-a nếu có; `repair_robust_attack`); `rrb_views(companion=)` áp cùng phép xoay/resize lên ảnh sạch, không nhiễu, không tiêu thụ RNG.
- **Unit test** (`scripts/rr_unit_test.py`) ĐẠT: rrb_views mặc định trùng tuyệt đối HEAD (3 ảnh × 3 seed, cả with_gt); companion không đổi RNG và khớp hình học; `make_view_loss(c2)` trùng `make_feature_plus_gt_loss` (rtol 1e-6); L_V hữu hạn, ≤ 0, có anchor/RoI dương (vd 22/56, 14/29, 6/12), ảnh không GT → 0; R-a tại δ = 0, σ = 0 trùng không sửa (cả c2 và vanish); attack p = 0/0.5/1 chạy, đếm forward no_grad = 2 × số step sửa.
- **Hiệu chỉnh λ** (`scripts/rr_calibrate_lambda.py` → `results/repair_robust/lambda_calibration.json`): áp luật cho C2 tái tạo được — 3.88 / 3.70 / 3.55 / 7.89 / 2.19, median 3.70 → λ = 4 (log cũ 3.95 → 4; lệch nhỏ do không tất định + resize khả vi). L_V: 0.28 / 0.19 / 6.79 / 0.14 / 0.31, median **0.28**.
- **Đổi luật làm tròn (user duyệt, chưa có số transfer nào):** luật khóa "làm tròn số nguyên, ≥ 1" cho λ_V = 1 → gradient L_V ≈ 3.6× L_OSFD, trái ý định cân bằng 1:1 của luật C2 (không lường trước tỉ lệ < 1). Sửa: **làm tròn 1 chữ số có nghĩa → λ_V = 0.3** (C2 vẫn = 4, không đổi). Ảnh 3 (11615, 6.79): R50 vốn không phát hiện GT duy nhất (objectness 0.027) → L_V gần 0; median không bị ảnh hưởng.
- λ của R-a thừa kế từ thành phần phân loại được chọn (C2: 4, V: 0.3).

---

## 2026-10-04 — Repair-Robust: probe V KHÔNG chọn; cổng độ trung thực R-a TRƯỢT → chuyển R-b (theo quy tắc khóa)

Sự cố vận hành: pipeline chờ bằng `tmux has-session -t rr` khớp TIỀN TỐ với chính session `rr2` → treo ~45 phút; sửa bằng khớp chính xác (`-t =name`). Không ảnh hưởng kết quả.

**PNG dev100 sinh lại** (seed theo chỉ số dev300): OSFD tái tạo R50 / R101 trùng tuyệt đối số đã commit (96.4 / 91.8); ConvNeXt-T 78.4 (cũ 78.8), Swin-T 74.8 (74.3), YOLOX-S 69.8 (70.2), DINO-Swin-L 24.0 (25.3) — lệch ≤ 1.3, mức không tất định của suy luận target. Runtime: OSFD ~3.3 s/ảnh, C2 ~8.9 s/ảnh, V tương tự C2.

**Probe V** (`results/repair_robust/v_dev100/screen.json`; run `metrics_rr_dev100.json`, `generalization_rr_dev100.json`). Relative AP drop % dev100, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- C2: 99.3 / 94.0 / 81.2 / 79.4 / 69.1 / 31.4 | 76.6
- V (λ_V = 0.3): 97.2 / 94.3 / 80.6 / 79.4 / 70.9 / 30.6 | 77.0
- OSFD: 96.4 / 91.8 / 78.4 / 74.8 / 69.8 / 24.0 | 74.3

V − C2: CrossAvg₃ +0.4 [−1.9, 3.3], DINO-Swin-L −0.8 [−5.1, 3.6], YOLOX-S +1.8. **Quy tắc khóa (Δ Swin-L ≥ +2) → KHÔNG chọn; R-a/R-b dùng loss phân loại của C2 (λ = 4).** Mô tả: V làm đúng cơ chế dự định — FP nền/ảnh Swin-T 24.4 → 16.8, YOLOX 11.4 → 7.6, DINO-Swin-L 3.0 → 1.7; recall@0.3 thấp hơn (Swin-T 0.35 → 0.26; DINO-Swin-L 0.57 → 0.52) — nhưng DINO-Swin-L AP drop không tăng → budget hallucination KHÔNG phải yếu tố giới hạn trên Swin-L.

**Cổng độ trung thực R-a** (`scripts/rr_fidelity_gate.py` → `results/repair_robust/fidelity_gate_Ra.json`) — **TRƯỢT ở cả OSFD và C2, không sát ngưỡng:**
- OSFD: ρ(Q_rep, retained Swin-L) +0.15 [−0.04, 0.33], ρ(Q_plain) +0.15, chênh +0.00 [−0.02, 0.03].
- C2: ρ_rep +0.12, ρ_plain +0.08, chênh +0.04 [−0.01, 0.13].
- Mô tả: thay C2/C3 bằng sạch gần như KHÔNG đổi R50 — retained 0.029 → 0.031 (OSFD), 0.047 → 0.055 (C2); cosd FPN 0.772 → 0.756, 0.647 → 0.631. Q_plain tương quan tốt với target Mask R-CNN (R101 +0.60, ConvNeXt +0.47, Swin-T +0.37) và YOLOX (+0.37) nhưng yếu với DINO-Swin-L (+0.08 đến +0.15).
- Diễn giải: R50 Mask R-CNN không có cơ chế "sửa" dựa vào mức nông — thiệt hại nằm ở C4/C5 và đi thẳng qua FPN top-down; phép thay C2/C3 không mô phỏng được encoder DINO.

**Theo quy tắc khóa:** R-a trượt cổng → chạy **R-b** (ô 64×64 thay sạch, cùng mặt nạ cho 4 stage, q = 0.3) qua CÙNG cổng. Mặt nạ cổng: 1 lần rút mỗi ảnh, seed = SEED + chỉ số dev300 (numpy). R-b trượt → dừng hướng Repair-Robust.

---

## 2026-10-04 — Dừng nhánh Repair-Robust (user); pilot xu hướng "đa dạng hình học kiểu PIA" @ dev30 — quy tắc khóa TRƯỚC khi code/chạy

**Dừng Repair-Robust (user):** cổng R-b bị dừng giữa chừng theo yêu cầu user — KHÔNG có kết quả, không kết luận về R-b. Trọng tâm chuyển sang phân tích OSFD: chấp nhận (user) "RRB đóng góp phần lớn; loss feature không phải nguồn ưu thế, có target kém hơn task loss (DINO-Swin-L −10, ConvNeXt −7.6 @ dev300) nhưng bổ trợ loss phân loại".

**Bỏ BSR trước khi code (user):** bản nháp đầu của entry này khai báo pilot BSR (GEO-B1/B2, chưa code, chưa chạy, chưa commit) — thay bằng PIA. Lý do: ablation cho thấy nguyên liệu là XOAY (biến đổi toàn cục); phép chiếu phối cảnh ⊃ affine ⊃ similarity ⊃ xoay → PIA là mở rộng trực tiếp nguyên liệu đã có tác dụng ("khuếch đại nguyên liệu", plan_sota bước 3), còn shuffle khối của BSR phá cấu trúc object (khó co-transform GT cho detection).

**Câu hỏi:** thay RRB bằng biến đổi phối cảnh đa bậc tự do kiểu PIA (Liang, Cao, Xiao — "Perspective-Invariant Attack…", arXiv 2608.15115, IEEE TIFS 2026) có tăng transfer trên detection không; phần lợi (nếu có) đến từ số hạng phối cảnh hay chỉ từ affine mạnh; liều δ_max của paper (cho ảnh phân loại 299×299) có hợp với detection không. Port PIA sang detection = BASELINE, không phải đóng góp; phát hiện riêng cho detection chỉ có thể là đường liều / vai trò số hạng phối cảnh.

**dev30 (user chọn — CHỈ xem xu hướng, mô tả):** = 30 ảnh đầu của dev100 theo thứ tự dev300 (thứ tự eval), `data/image_lists/dev30.csv`, ⊂ dev100 ⊂ dev300. **Không ra quyết định GO/sàng lọc ở dev30** (CI hiệu 2 method ở n = 30 ước ±5 điểm); ứng viên muốn đi tiếp phải khóa tiêu chí riêng rồi chạy dev100/dev300.

**Nền chung (mọi ô):** loss feature OSFD (backbone R50, 4 stage, k = 3), MI μ = 1, ε = 5, α = 1, noise init 0, 50 step, seed ảnh SEED + chỉ số dev300; mỗi step 2 view, 1 backward (trừ ô N20), mỗi view cộng nhiễu Gaussian σ = 6 như RRB (giữ để chỉ HÌNH HỌC khác nhau). **Feature sạch co-transform:** mỗi view so với k·F_clean(T(x_clean)) — cùng tham số hình học, không nhiễu, no_grad (forward thêm không tính B, báo cáo). Lý do: phối cảnh δ_max = 0.6 dời góc ảnh tới 30% cạnh → so với F_clean chưa biến đổi là vô nghĩa.

| Ô | Biến đổi mỗi view | Câu hỏi |
|---|---|---|
| OSFD | RRB gốc, F_clean KHÔNG co-transform (PNG có sẵn) | mốc |
| GEO-C0 | RRB gốc (chain xoay → resize, nhiễu), F_clean co-transform | đối chứng: riêng việc co-transform F_clean đổi OSFD bao nhiêu |
| PIA | PIA đúng tham số paper: δ_max = 0.6, π = (π₄, π₃, π₂, π₁) = (0.4, 0.3, 0.2, 0.1) | PIA có thắng RRB cùng compute? |
| PIA-d0.3 | như PIA, δ_max = 0.3 | liều: ảnh detection nhiều object nhỏ / sát biên — 0.6 có quá mạnh? |
| PIA-aff | như PIA nhưng π = (0, 1, 0, 0) (chỉ mode 3 = affine 6-DOF), δ_max = 0.6 | lợi đến từ số hạng phối cảnh (h31, h32) hay chỉ affine? |
| PIA-N20 (mô tả) | PIA tham số paper, N = 20 view/step như paper, gradient cộng dồn 4 cụm × 5 view → 4 backward/step (B thực = 200, view forward = 1000) | trần khi bỏ ràng buộc compute; báo kèm runtime |

**Biến đổi PIA (port theo paper, không có code chính thức trong paper):** khung = ảnh ở khung model (img_shape, trước pad). 4 góc ảnh ra (TL, TR, BR, BL) ánh xạ về 4 điểm nguồn trong ảnh vào, điểm nguồn = góc + (int(ξ_x·δ_max·w/2), int(ξ_y·δ_max·h/2)), ξ ~ U(−1, 1), được phép ra ngoài ảnh. Mode rút theo π mỗi view: mode 4 — 4 góc độc lập (8-DOF); mode 3 — 3 góc độc lập (góc thiếu chọn đều ngẫu nhiên), góc còn lại hoàn thiện hình bình hành (6-DOF); mode 2 — TL, BR độc lập, TR/BL dựng hình chữ nhật trục (4-DOF); mode 1 — 1 vector dịch chung (2-DOF). Giải homography 8 ẩn (float64), warp bằng `grid_sample` bilinear khả vi, vùng ngoài ảnh nguồn lấp 0 (= "crop" về cỡ gốc). RNG: `random` của Python cho mode/góc, torch cho nhiễu σ (cùng nguồn như RRB). Tham số (δ_max, π, N) lấy từ paper, KHÔNG tune; δ_max = 0.3 và π affine là 2 ô liều/đối chứng khai báo trước. Lệch đã biết: paper không có nhiễu σ (ta giữ σ = 6 ở mọi ô GEO để chỉ hình học khác OSFD); paper N = 20 (ô chính dùng N = 2 để cùng compute với OSFD).

**Báo cáo (mô tả):** relative AP drop 6 model, CrossAvg₃, Δ so OSFD và so GEO-C0 (paired bootstrap, chỉ để thấy độ rộng CI), runtime, số view forward, số forward no_grad.

---

## 2026-10-04 — Pilot PIA: code + unit test (TRƯỚC khi chạy)

- `attack/methods/pia.py`: `sample_source_corners` (multi-DOF, π, δ_max), `homography` (8 ẩn, float64), `warp_grid`/`warp` (grid_sample bilinear, ngoài = 0), `pia_views` (companion sạch cùng phép warp, không nhiễu), `geo_osfd_attack` (geo = rrb → GEO-C0 | pia; feature sạch co-transform mỗi view, no_grad). `core.run_iterative_attack(backward_per_step=)` cộng gradient nhiều backward/step (chỉ PIA-N20); mặc định 1.
- `scripts/run_baselines.py`: ô `GEO-C0`, `PIA`, `PIA-d0.3`, `PIA-aff`, `PIA-N20`; stats ghi `forward_no_grad`, `mode_frac`, `backward` thực (N20 = 4 × B).
- **Unit test** (`scripts/pia_unit_test.py`) ĐẠT: core mặc định trùng tuyệt đối HEAD (OSFD 3 step, 2 ảnh); H·dst = src và ràng buộc DOF đúng từng mode (1: tịnh tiến; 2: không xoay/không phối cảnh; 3: h31 = h32 = 0; 4: có phối cảnh); warp đồng nhất lệch ≤ 6.6e-3, tịnh tiến đúng chiều, ngoài nguồn = 0; tần suất mode (20000 mẫu) 0.398 / 0.305 / 0.193 / 0.103 ≈ π; gradient 4 view × 1 backward = 2 × 2 view (lệch tương đối 8.8e-8); σ = 0 → view == companion (PIA, GEO-C0); 5 ô chạy được, L∞ đúng, forward no_grad = step × view. Ghi chú: lần chạy test ĐẦU TIÊN sau khi khởi động có 1 lệch "core == HEAD"; kiểm riêng HEAD/HEAD, mới/mới, HEAD/mới đều trùng tuyệt đối, 2 lần chạy lại toàn bộ đều đạt → không tất định lúc khởi động GPU, không do sửa code.

---

## 2026-10-04 — Pilot PIA @ dev30 (mô tả, KHÔNG quyết định) → PIA ≈ RRB trên CrossAvg₃; tín hiệu duy nhất ở DINO-Swin-L

User bỏ ô PIA-N20 giữa chừng (để có số sớm; 15/30 ảnh GEO-C0 đã có từ lần chạy đầu, giữ nguyên, cùng seed). Run `dev300_B50_eps5` subset dev30 (`metrics_pia_dev30.json`, `generalization_pia_dev30.json`), `scripts/ablation_screen.py --subset dev30 --tag pia_dev30` → `results/ablation/pia_dev30/screen.json` (cột "verdict" của script là quy tắc ablation bước 1 — KHÔNG áp dụng cho pilot này). Log `results/pia_dev30/run.log`. n = 30, paired bootstrap 1000.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- GEO-C0 (RRB + co-transform): 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA (δ 0.6, π paper): 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / **28.8** | 79.5 | 0.89
- PIA-d0.3: 97.4 / 91.0 / 82.7 / 79.8 / 77.0 / 24.7 | 79.8 | 0.89
- PIA-aff (chỉ 6-DOF): 95.8 / 88.5 / 73.1 / 78.8 / 76.5 / 27.9 | 76.1 | 0.86

Runtime ~5.5–6.0 s/ảnh mọi ô GEO/PIA (OSFD ~3.3; thêm 2 forward no_grad/step cho feature sạch co-transform). Tỉ lệ mode 4 thực tế 0.39 (≈ π₄).

Mô tả (n = 30, CI hiệu ước ±5):
- **Thay RRB bằng PIA cùng compute: CrossAvg₃ ngang** (PIA − GEO-C0 +0.5; − OSFD +0.7 [−1.8, 5.1]). Không có lợi trung bình.
- **DINO-Swin-L:** PIA − GEO-C0 **+7.2 [−1.0, 13.1]**, nhưng GEO-C0 − OSFD −5.2 [−9.8, 0.4] (co-transform có vẻ HẠI Swin-L) → PIA − OSFD chỉ +2.0 [−4.4, 8.3]. Chưa tách được "PIA giúp" với "co-transform hại".
- R101 giảm theo độ mạnh hình học (OSFD 93.8 → PIA 89.9 → PIA-aff 88.5) trong khi cross ngang → η tăng nhẹ (0.86 → 0.89): δ bớt đặc thù surrogate, cùng chiều cơ chế của RRB.
- δ 0.3 vs 0.6: CrossAvg₃ ngang; DINO-Swin-L thấp hơn ở 0.3 (24.7 vs 28.8).
- Chỉ affine: CrossAvg₃ thấp hơn PIA 3.4, chủ yếu ConvNeXt (73.1 vs 79.7) → số hạng phối cảnh có thể có ích cho ConvNeXt; DINO-Swin-L ngang.
- Đọc chung: giống kết luận về loss — có đa dạng hình học TOÀN CỤC là đủ, LOẠI biến đổi (xoay+resize vs phối cảnh) gần như không đổi trung bình. Muốn đi tiếp với PIA phải khóa tiêu chí (trọng tâm DINO-Swin-L, đối chứng cả OSFD lẫn GEO-C0) rồi chạy dev100.

---

## 2026-10-04 — Pilot PIA bổ sung @ dev30: C2-PIA + PIA-noCT (user) — quy tắc khóa TRƯỚC khi code/chạy

Vẫn dev30, CHỈ mô tả, không quyết định. Nền như entry PIA (MI μ = 1, ε = 5, α = 1, 50 step, 2 view/step, 1 backward, σ = 6, PIA δ_max = 0.6, π paper, seed SEED + chỉ số dev300).

| Ô | Định nghĩa | Câu hỏi |
|---|---|---|
| C2-PIA | C2 (L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, feature bắt bằng hook) với RRB thay bằng PIA; feature sạch co-transform như ô PIA; GT co-transform qua homography | lợi DINO-Swin-L của PIA và của loss cls có cộng được không? so C2 (PNG có sẵn) và PIA |
| PIA-noCT | ô PIA nhưng feature sạch KHÔNG co-transform (so với F_clean của ảnh sạch chưa biến đổi, như OSFD gốc) | DINO-Swin-L +7.2 (PIA − GEO-C0) là nhờ PIA hay do co-transform hạ GEO-C0? |

**Co-transform GT qua homography (C2-PIA):** H ánh xạ pixel ra → nguồn, nên box (khung vào) đi qua H⁻¹: 4 góc mỗi box → AABB → clip vào khung; mask warp bằng CÙNG lưới grid_sample với ảnh (threshold 0.5 như `with_gt`). Instance có box sau clip rộng hoặc cao < 1 px (object bị đẩy ra khỏi khung — δ_max = 0.6 dời góc tới 30%) hoặc mẫu số homography ≤ 0 ở góc box → BỎ instance đó (box, nhãn, mask) cho view đó; ảnh mất hết GT → chỉ còn loss feature + loss cls với toàn mẫu âm (như model.loss xử lý). Báo cáo tỉ lệ instance bị bỏ.

**Unit test trước khi chạy:** (1) ô GEO-C0/PIA/PIA-d0.3/PIA-aff trùng tuyệt đối code HEAD (cờ mặc định); (2) tịnh tiến nguyên: box dời đúng (dx, dy); (3) IoU(AABB của mask đã warp, box đã warp) — so với mức của xoay RRB (0.834, entry ablation); (4) PIA-noCT: 0 forward no_grad, loss tại view = loss OSFD; (5) C2-PIA tại cờ cls_weight = 0 trùng ô PIA về loss; không NaN trên 2 ảnh × 3 step; L∞ đúng.

**Báo cáo:** 6 model, CrossAvg₃, so OSFD / C2 / PIA / GEO-C0 (paired bootstrap, chỉ để thấy CI), runtime, tỉ lệ instance bị bỏ.

**Code + unit test (TRƯỚC khi chạy):** `pia.warp_boxes` (H⁻¹, AABB, clip, bỏ < 1 px / mẫu số ≤ 0), `pia.gt_with_keep`, `pia_views(data_sample=)`; `geo_osfd_attack(co_transform=, cls_weight=)`; ô `C2-PIA`, `PIA-noCT` trong `run_baselines.py` (stats thêm `gt_kept_frac`). `scripts/pia_c2_unit_test.py` ĐẠT: 4 ô cũ trùng tuyệt đối HEAD (3 step); tịnh tiến → box dời (−dx, −dy); đẩy ra ngoài → bỏ hết; IoU(AABB mask warp, box warp) view PIA mean **0.823** (trung vị 0.839, p5 0.580; không biến đổi 0.951; xoay RRB 0.834) → co-transform GT dùng được; 159/180 instance giữ (12% bị bỏ); PIA-noCT 0 forward no_grad, loss == loss OSFD; make_view_loss(c2, 0) == loss feature; GT rỗng → loss hữu hạn; C2-PIA 3 step × 2 ảnh hữu hạn, L∞ đúng.

**Kết quả @ dev30 (mô tả, KHÔNG quyết định)** — `metrics_pia2_dev30.json`, `generalization_pia2_dev30.json`, `results/ablation/pia2_dev30/screen.json`, log `results/pia_dev30/run2.log`. C2 = PNG `PROBE-osfd+cls` sinh lại trên dev100 (cùng seed).

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- C2: 99.9 / 94.7 / 84.6 / 82.6 / 67.8 / 26.1 | 78.3 | 0.88
- GEO-C0: 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | 0.89
- **C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / 34.7 | 83.6 | 0.91**
- PIA-noCT: 53.6 / 47.6 / 32.7 / 40.0 / 37.0 / 5.3 | 36.6 | 0.76

Runtime: C2-PIA 11.5 s/ảnh, PIA-noCT 3.7. C2-PIA giữ trung bình 90.7% instance GT mỗi view.

- **C2-PIA − OSFD: CrossAvg₃ +4.8 [1.5, 10.6], DINO-Swin-L +7.9 [1.6, 13.8]**, ConvNeXt +3.3, Swin-T +8.6, YOLOX +2.5, R101 +1.0. So C2: YOLOX +10.4 [2.8, 19.1], DINO-Swin-L +8.7 [0.3, 13.6]. So PIA: DINO-Swin-L +5.9 [1.6, 9.9], CrossAvg₃ +4.1. → Loss cls và PIA CỘNG được: C2-PIA cao nhất ở mọi target khác họ, và lấy lại YOLOX mà C2 (RRB) mất trên dev30.
- **PIA-noCT sụp** (CrossAvg₃ 36.6, kể cả white-box R50 53.6): với biến đổi mạnh, so feature adv đã warp với feature sạch CHƯA warp làm loss gần như vô nghĩa → co-transform là BẮT BUỘC cho PIA; không phải co-transform "hại" (câu hỏi gây nhiễu của entry trước được trả lời: +7.2 DINO-Swin-L của PIA − GEO-C0 không phải do so với một đối chứng bị hạ giả tạo; co-transform cần cho PIA, còn với RRB biên độ nhỏ thì không cần).
- Tương tác loss × biến đổi: với RRB, thêm cls gần như không đổi CrossAvg₃ trên dev30 (C2 − OSFD −0.5; dev300 +4.7); với PIA, thêm cls +4.1 CrossAvg₃ và +5.9 DINO-Swin-L.
- Lưu ý: C2-PIA khác C2 ở 2 chỗ (RRB → PIA VÀ co-transform feature sạch). n = 30, CI rộng. Muốn đi tiếp: khóa tiêu chí sàng lọc (so OSFD và C2) rồi chạy dev100 → dev300.

---

## 2026-10-04 — Pilot MORGD @ dev30 (user: xem xu hướng trước) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh:** đề xuất MORGD (Multi-Object Relative Geometry Diversification — biến dạng mượt toàn ảnh, điều kiện theo từng object, tạo đa dạng vị trí/tỉ lệ TƯƠNG ĐỐI giữa các object) đến từ trao đổi với một trợ lý khác, không có paper/code. Prior yếu (đã nêu với user): pilot KG (quan hệ object) và A′1 (trọng số theo box) đều NO-GO; loại biến đổi toàn cục ít ảnh hưởng (RRB ≈ PIA); paper PIA: warp phi cứng (Elastic 23.0, TPS 37.5) thua phối cảnh (49.3) ở classification. Câu hỏi ở dev30 (CHỈ mô tả): MORGD có tiến tới / vượt C2-PIA không, và phần "theo object" có hơn warp mượt generic cùng biên độ không.

**Khung:** như C2-PIA (MI μ = 1, ε = 5, α = 1, 50 step, 2 view/step, 1 backward, σ = 6, L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), 1 forward/view, feature sạch co-transform, GT co-transform), CHỈ thay phép biến đổi PIA bằng:

| Ô | Biến đổi | So với |
|---|---|---|
| C2-MORGD | trường dịch chuyển mượt, neo tại từng GT object | C2-PIA, C2, OSFD |
| C2-SMOOTH | cùng cơ chế, neo tại vị trí NGẪU NHIÊN đều trên ảnh, kích thước = kích thước GT hoán vị ngẫu nhiên (cùng số neo, cùng phân phối biên độ) | C2-MORGD: phần "theo object" |

**Định nghĩa biến đổi (khóa, tự chọn vì nguồn không cho số — KHÔNG tune):** mỗi neo i (box tâm c_i, cỡ w_i × h_i ở khung model): Δ_i = (λ·w_i·r_x, λ·h_i·r_y), r ~ U(−1, 1), **λ = 0.25**; s_i = exp(U(−ln 1.25, ln 1.25)). Ánh xạ ngược của neo i (pixel ra q → nguồn): u_i(q) = (q − c_i)(1/s_i − 1) − Δ_i/s_i (đồng dạng: tâm c_i dời tới c_i + Δ_i, co giãn s_i). Trọng số K_i(q) = exp(−½[((q_x − c'_x)/w_i)² + ((q_y − c'_y)/h_i)²]), c' = c_i + Δ_i. Trường u = Σ K_i u_i / (Σ K_i + **ε = 0.05**) (xa mọi neo → u → 0, nền giữ nguyên); ảnh ra(q) = x(q + u(q)) bằng grid_sample bilinear, ngoài = 0. Đa dạng tương đối r′_ij − r_ij đến từ việc rút (Δ_i, s_i) ĐỘC LẬP mỗi object (không có sampler cặp tường minh — mức tối thiểu của ý tưởng). RNG: `random` của Python. Ảnh không GT → không neo → đồng nhất.

**GT co-transform:** box = AABB của ảnh warp (cùng lưới) của mặt nạ chữ nhật box (≥ 0.5); mask warp cùng lưới; bỏ instance rỗng / < 1 px.

**Unit test trước khi chạy:** (1) ô cũ (GEO/PIA/C2-PIA) trùng HEAD; (2) 1 neo, ε = 0 → trùng phép đồng dạng tương ứng qua homography (warp PIA); (3) xa neo → ảnh gần như giữ nguyên; (4) box 1 neo, ε = 0 khớp c' ± s·(w, h)/2 (sai số rời rạc hóa); (5) SMOOTH dùng đúng multiset kích thước GT; (6) chạy được, hữu hạn, L∞ đúng. Báo cáo thêm biên độ dịch chuyển trung bình |u| (px) mỗi view của MORGD / SMOOTH / PIA để thấy chênh liều.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/morgd.py` (`sample_anchors`, `flow_grid`, `warp_boxes_by_grid`, `morgd_views`); `geo_osfd_attack(geo="morgd" | "smooth")`; ô `C2-MORGD`, `C2-SMOOTH` (stats thêm `mean_disp_px`). `scripts/morgd_unit_test.py` ĐẠT: GEO-C0, PIA trùng tuyệt đối HEAD; C2-PIA lệch HEAD/mới 0.032 ≈ HEAD/HEAD 0.036 (không tất định đã biết); 1 neo ε = 0 trùng phép đồng dạng (lưới lệch 1.8e-7); xa neo lệch ≤ 0.006; box 1 neo khớp đúng c' ± s·(w, h)/2; SMOOTH đúng multiset kích thước GT; C2-MORGD/C2-SMOOTH chạy được, L∞ đúng. **Biên độ |u| trung bình mỗi view (3 ảnh × 10 view): MORGD 43.0 px, SMOOTH 50.1, PIA 176.7** → MORGD/SMOOTH yếu hơn PIA ~4× về biên độ hình học (nền giữ nguyên theo thiết kế) — phải nhớ khi đọc kết quả.

**Kết quả @ dev30 (mô tả, KHÔNG quyết định)** — `metrics_morgd_dev30.json`, `generalization_morgd_dev30.json`, `results/ablation/morgd_dev30/screen.json`, log `results/pia_dev30/run3.log`.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- C2: 99.9 / 94.7 / 84.6 / 82.6 / 67.8 / 26.1 | 78.3 | 0.88
- C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / **34.7** | **83.6** | 0.91
- C2-MORGD: 99.9 / 96.0 / 87.7 / 83.5 / 72.8 / 27.3 | 81.3 | 0.89
- C2-SMOOTH: 99.5 / 95.0 / 86.0 / 84.4 / 75.3 / 28.1 | 81.9 | 0.90

Runtime ~11 s/ảnh cả hai; |u| trung bình 40.9 (MORGD) / 40.5 px (SMOOTH); giữ 99.5% / 99.2% instance GT.

- **Phần "theo object" KHÔNG có tác dụng:** C2-MORGD − C2-SMOOTH: CrossAvg₃ −0.6, YOLOX −2.5 [−6.5, 0.5], DINO-Swin-L −0.8 [−5.2, 3.5]. Neo theo GT object không hơn neo ngẫu nhiên cùng biên độ.
- **Cả hai thua C2-PIA**, rõ nhất ở DINO-Swin-L: MORGD − C2-PIA −7.5 [−10.2, −0.9], SMOOTH − C2-PIA −6.7 [−9.8, −0.0]; YOLOX −5.4 [−10.7, −2.0] (MORGD). CrossAvg₃ thấp hơn C2-PIA 1.7–2.3.
- Vẫn hơn C2 (RRB) và OSFD về CrossAvg₃ (+2.5 đến +3.6 so OSFD, CI chứa 0).
- Gây nhiễu chưa tách: biên độ hình học của MORGD/SMOOTH ~¼ PIA (41 vs 177 px) → thua C2-PIA có thể do liều, KHÔNG kết luận được "phi cứng kém hơn phối cảnh". Nhưng câu hỏi chính của MORGD (object-conditioned > generic cùng liều) trả lời ÂM ở dev30, khớp prior từ KG / A′1.

---

## 2026-10-04 — MORGD / SMOOTH thuần (không cls) @ dev30 — làm baseline so RRB (user), khóa TRƯỚC khi chạy

Ô `MORGD` / `SMOOTH` = đúng khung ô `PIA` (loss feature OSFD k = 3, MI μ = 1, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, KHÔNG cls, KHÔNG GT), phép biến đổi như entry MORGD (λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05). So OSFD, GEO-C0 (RRB + co-transform — đối chứng công bằng nhất), PIA. Chỉ mô tả. Không code mới (chỉ đăng ký ô trong `run_baselines.py`; `geo_osfd_attack(geo="morgd" | "smooth", cls_weight=0)` đã qua unit test entry MORGD).

**Kết quả @ dev30 (mô tả)** — `metrics_morgd0_dev30.json`, `generalization_morgd0_dev30.json`, `results/ablation/morgd0_dev30/screen.json`, log `results/pia_dev30/run4.log`. Runtime 5.2 s/ảnh cả hai.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8 | 0.86
- GEO-C0 (RRB): 97.8 / 91.8 / 78.2 / 80.8 / 78.2 / 21.5 | 79.0 | 0.87
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | 0.89
- MORGD: 96.9 / 91.5 / 83.3 / 80.2 / 77.9 / 27.3 | 80.5 | 0.89
- SMOOTH: 97.4 / 91.7 / 83.8 / 79.1 / 76.4 / 25.7 | 79.7 | 0.89

- **Bản thuần: mọi phép biến đổi toàn cục/mượt đều ngang nhau** — CrossAvg₃ 79.0 (RRB) / 79.5 (PIA) / 79.7 (SMOOTH) / 80.5 (MORGD), chênh ≤ 1.5, CI ±4–5. MORGD − GEO-C0: DINO-Swin-L +5.8 [−3.1, 11.3]; MORGD − PIA: DINO-Swin-L −1.5.
- MORGD − SMOOTH (thuần): CrossAvg₃ +0.8, YOLOX +1.6, DINO-Swin-L +1.6 — ngược dấu với khung C2 (−0.6) → phần "theo object" ≈ 0, trong nhiễu.
- **Tương tác với cls (thuần → C2):** PIA +4.1 (79.5 → 83.6), SMOOTH +2.2 (79.7 → 81.9), MORGD +0.8 (80.5 → 81.3); DINO-Swin-L: PIA +5.9, SMOOTH +2.4, MORGD 0.0. → lợi khi thêm cls lớn nhất ở PIA (biên độ hình học lớn nhất, ~4× MORGD/SMOOTH); gợi ý (chưa kiểm) loss cls cần biến đổi MẠNH mới phát huy.

---

## 2026-10-04 — Pilot MORGD-RaPA @ dev30 (user: chỉ áp RaPA lên MORGD) — quy tắc khóa TRƯỚC khi code/chạy

**Nguồn đã đọc:** paper RaPA (Su et al., CVPR 2026, arXiv 2504.18594, TeX) + code chính thức `github.com/molarsu/RaPA` (`core/attacker/DropConnect.py`, `config/__base__/attacker/RaPA.py`). Paper: DropConnect lên weight + bias của MỌI lớp Linear và Normalization (BN/LN), M ~ Bernoulli(1 − p), KHÔNG nhân bù 1/(1 − p), mask rút lại mỗi lần inference; ResNet-50 p = 0.05; setting chính targeted, 1000 iteration, S = 5. **Phụ lục: ở untargeted RaPA cho "similar ASRs" (không lợi rõ)** — tác giả giải thích do bão hòa (CFM 98.2%); detection khác họ của ta chưa bão hòa (CrossAvg₃ ~80, DINO-Swin-L 25–35).

**Phát hiện ở code chính thức:** pre-hook gán `weight.data = original·mask`, forward hook trả `weight.data = original` ngay sau forward → khi backward autograd đọc trọng số ĐÃ KHÔI PHỤC (đổi `.data` không tăng version counter) → gradient theo input là "lai" (forward masked, đạo hàm tại chính lớp đó dùng trọng số gốc). Paper không nhắc.

**Ô `MORGD-RaPA`** = đúng ô `MORGD` thuần (loss feature OSFD k = 3, MI μ = 1, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05) + RaPA:
- Phạm vi: 53 lớp BatchNorm2d của backbone R50 (weight + bias affine, 53 120 tham số). Backbone KHÔNG có Linear; loss feature chỉ đi qua backbone → tương đương "mọi Linear + Norm" trên đường loss.
- p = 0.05 (giá trị paper cho ResNet-50), S = 1, mask Bernoulli(0.95) từng phần tử, không nhân bù, **rút mới mỗi view**; forward ảnh sạch co-transform và forward ảnh adv của CÙNG view dùng CÙNG mask (nếu khác mask, chênh feature do tham số bị tính như distortion adv).
- **Lệch có chủ đích so với code chính thức:** tham số masked dùng nhất quán cả forward lẫn backward (thay tạm tham số bằng tensor masked qua `torch.nn.utils.stateless._reparametrize_module`, không sửa `.data`) — gradient đúng toán học.
- RNG mask: `torch.Generator` riêng trên GPU, seed = `torch.initial_seed()` của ảnh (= SEED + chỉ số dev300) → không đụng RNG của biến đổi/nhiễu: p = 0 phải trùng tuyệt đối ô MORGD.
- Compute: không thêm forward/backward (forward sạch đã có do co-transform).

**Unit test trước khi chạy:** p = 0 trùng tuyệt đối MORGD; tỉ lệ phần tử bị mask ≈ 0.05; tham số model sau attack không đổi; cùng mask → tại adv = sạch, σ = 0 feature view == feature companion và khác feature không mask; gradient chảy qua tham số masked (kiểm bằng so sánh với forward thủ công có mask); chạy được, L∞ đúng.

**Báo cáo (mô tả):** so MORGD, SMOOTH, GEO-C0, PIA, OSFD; 6 model, CrossAvg₃.

**Code + unit test (TRƯỚC khi chạy):** `pia.rapa_sampler` (53 BN backbone, Bernoulli(0.95), `_reparametrize_module`, RNG riêng), `geo_osfd_attack(rapa_p=)`; ô `MORGD-RaPA`. `scripts/rapa_unit_test.py` ĐẠT: MORGD trùng tuyệt đối HEAD; rapa_p = 0 trùng tuyệt đối MORGD; tỉ lệ mask 0.0502; tham số model không đổi sau attack; cùng context → cùng mask, khác feature không mask; gradient theo input dưới context == khi gán tham số masked thật (lệch tương đối 7.1e-8, tức masked nhất quán cả forward lẫn backward); L∞ đúng.

**Kết quả @ dev30 (mô tả)** — `metrics_rapa_dev30.json`, `generalization_rapa_dev30.json`, `results/ablation/rapa_dev30/screen.json`, log `results/pia_dev30/run5.log`. Runtime 6.1 s/ảnh (MORGD 5.2). Tỉ lệ mask thực tế ≈ 0.050.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | η:
- MORGD: 96.9 / 91.5 / 83.3 / 80.2 / 77.9 / 27.3 | 80.5 | 0.89
- **MORGD-RaPA: 94.0 / 78.5 / 52.4 / 57.9 / 56.3 / 18.7 | 55.5 | 0.70**
(OSFD 78.8, GEO-C0 79.0, PIA 79.5, SMOOTH 79.7 — như entry trước.)

- **RaPA làm attack yếu đi mạnh ở mọi target:** MORGD-RaPA − MORGD: CrossAvg₃ −25.0, YOLOX −21.6 [−29.6, −12.8], DINO-Swin-L −8.6 [−16.9, −1.1], R101 −13.0; cả R50 white-box −2.9.
- **Chẩn đoán (forward-only, 5 ảnh dev30, ảnh sạch):** cos(feature backbone sạch, feature cùng ảnh dưới mask) theo stage 1→4: p = 0.05: 0.855 / 0.749 / 0.689 / 0.702; cos giữa 2 mask khác nhau: 0.827 / 0.723 / 0.676 / 0.700. p = 0.01: 0.962 / 0.934 / 0.915 / 0.925 (2 mask: 0.945 / 0.905 / 0.880 / 0.891). → Với p = 0.05, mỗi view là một biến thể surrogate lệch ~30% feature sâu; với loss feature (đo trực tiếp trên feature backbone) và chỉ 50 step × S = 1, hướng gradient giữa các biến thể gần như không cộng hưởng → mất "utility". p = 0.05 được paper chọn cho ResNet-50 CLASSIFICATION (loss trên logit, 1000 iteration, S = 5), không chuyển được sang setting này.
- Chưa tách: (a) liều p (0.01 ít phá feature hơn nhiều); (b) gradient "lai" của code chính thức (backward dùng trọng số gốc → hướng gradient gần surrogate gốc hơn) có thể chính là thứ giữ utility ở paper — bản của ta dùng gradient nhất quán. Không chạy thêm biến thể nếu user không yêu cầu.

---

## 2026-10-04 — Pilot GRGD-v0 @ dev30 (user, spec từ trợ lý khác) — quy tắc khóa TRƯỚC khi code/chạy

**Ý tưởng:** T_GRGD = T_relative ∘ T_PIA trên khung C2-PIA (không đổi gì khác: ε = 5, α = 1, B = 50, μ = 1, 2 view/step, 1 backward, σ = 6 chỉ cộng vào adv, L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), feature sạch + GT co-transform qua CẢ HAI phép). T_relative thay đổi quan hệ vị trí/tỉ lệ GIỮA các object (sau PIA), đã trừ phần giải thích được bằng một homography. Prior (đã nêu với user): MORGD neo theo object ≈ neo ngẫu nhiên (dev30, cả 2 khung) — câu hỏi lần này là sample QUAN HỆ cặp trực tiếp + trừ thành phần phối cảnh có khác không.

**T_relative (trên GT còn lại sau PIA, N object; N < 2 → đồng nhất, bỏ bước warp thứ hai):**
1. Đồ thị đầy đủ mọi cặp i < j. Mỗi cạnh: ξˣ, ξʸ, ξˢ ~ U(−1, 1); d_x = λ_r·(w_i + w_j)/2·ξˣ, d_y = λ_r·(h_i + h_j)/2·ξʸ, d_s = ln(1.25)·ξˢ; **λ_r = 0.25** (từ MORGD, không tune).
2. Ma trận liên thuộc A (E × N, A_ei = 1, A_ej = −1); giải riêng x, y, log-scale bằng KKT: [[AᵀA + 1e-6·I, ω], [ωᵀ, 0]]·[z; ν] = [Aᵀd; 0], ω_i = w_i h_i / Σ w h (trung bình có trọng số diện tích = 0) → t_i = (z_x, z_y), s_i = exp(z_s).
3. Trường thô u_raw = Σ K_i u_i / (Σ K_i + 0.05), u_i(q) = (q − c_i)(1/s_i − 1) − t_i/s_i, K_i Gaussian tâm c'_i = c_i + t_i, độ rộng (w_i, h_i) — đúng công thức MORGD.
4. **Trừ thành phần phối cảnh:** điểm điều khiển = 5 điểm/object (4 góc + tâm của box cỡ w_i × h_i đặt tại c'_i), p_m = q_m + u_raw(q_m); fit H_r: q → p bằng DLT bình phương tối thiểu (chuẩn hóa Hartley, float64, SVD); u_H(q) = H_r(q) − q; M(q) = Σ K_i / (Σ K_i + 0.05); **u_rel = u_raw − M·u_H**.
5. Ảnh ra = grid_sample(ảnh sau PIA, q + u_rel), bilinear, ngoài = 0. GT: box sau PIA → AABB của mặt nạ chữ nhật warp (≥ 0.5); mask warp; bỏ instance rỗng / < 1 px (như MORGD).

**Ô (dev30, chỉ mô tả):**
| Ô | Định nghĩa |
|---|---|
| C2-PIA | PNG có sẵn (mốc, 83.6) |
| C2-PIA-RESIDUAL | như C2-GRGD nhưng tâm neo c_i thay bằng tâm ngẫu nhiên đều trên khung; GIỮ NGUYÊN kích thước, ξ_ij, d_ij, nghiệm (t, s), kernel, bước trừ phối cảnh |
| C2-GRGD | như trên, tâm neo = tâm GT thật (sau PIA) |
Hiệu chính: Δ_extra = RESIDUAL − C2-PIA; **Δ_object = GRGD − RESIDUAL** (chỉ khi > 0 mới là bằng chứng cho đóng góp mới).

**RNG:** H của PIA và nhiễu σ dùng đúng RNG như C2-PIA (`random`, torch); ξ_ij từ `random.Random` riêng (seed = torch.initial_seed() + 1); tâm ngẫu nhiên của RESIDUAL từ `random.Random` riêng thứ hai (seed + 2) → mỗi view: H, ξ, σ trùng nhau giữa 2 ô, chỉ khác tâm neo.

**Log mỗi ảnh:** n_objects, n_edges, |u| trung bình phần PIA / u_raw / u_H·M / u_rel, R_proj = 1 − ‖u_rel‖₂ / ‖u_raw‖₂, D_pair_pos (mục tiêu, theo t) và D_pair_pos_eff (hiệu dụng, dời tâm ≈ −u_rel(c'_i)) chuẩn hóa bởi (√A_i + √A_j)/2, D_pair_scale = mean |ln((s_i/s_j))| (mục tiêu), gt_kept_frac.

**Unit test (sửa từ spec, user được báo):** C2 không tất định → "λ_r = 0 trùng C2-PIA" kiểm ở MỨC VIEW (ảnh, companion, GT trùng tuyệt đối pia_views), không ở mức attack. Danh sách: (1) λ_r = 0 → view trùng tuyệt đối PIA; (2) N < 2 → trùng; (3) ξ = 0 → |u_rel| ≈ 0; (4) 2 box: (t₁ − t₂) = d₁₂ đúng dấu và độ lớn; (5) Σ ω t = 0; (6) u_raw sinh từ MỘT homography → sau khi trừ, RMS ≈ 0; (7) 2 object chuyển động khác nhau → báo RMS dư; (8) xa object |u_rel| ≈ 0; (9) σ = 0, adv = sạch → view == companion; (10) IoU(mask, box) view GRGD; (11) ô cũ (PIA, MORGD) trùng HEAD; (12) C2-GRGD / C2-PIA-RESIDUAL chạy được, hữu hạn, L∞ đúng.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/grgd.py` (`complete_edges`, `sample_pairwise`, `solve_graph_motion` — KKT, `fit_homography_dlt` — Hartley + SVD, `RelativeField` — u_raw / M·u_H / u_rel, `relative_motion`, `grgd_views`); `geo_osfd_attack(geo="grgd" | "grgd_random")`; ô `C2-GRGD`, `C2-PIA-RESIDUAL` trong `run_baselines.py` (stats `grgd`: trung bình mỗi ảnh các chỉ số log ở trên + `frac_views_relative`). `morgd.flow_grid` tách thành `flow_field` / `pixel_coords` / `to_grid` (không đổi số). `scripts/grgd_unit_test.py` ĐẠT: λ_r = 0 và N = 0 / 1 → view (ảnh, companion, GT) trùng tuyệt đối `pia_views`; ξ = 0 → max |u_rel| 6.7e-13 px; 2 box: t₁ − t₂ = d₁₂ đúng (22.0, −14.375, log-scale 0.134); Σ ω t = Σ ω log s = 0 (≤ 1.3e-15); 1 homography → fit lại đúng (RMS dư 1.4e-13); 2 object khác chuyển động → RMS dư 3.72 px; xa object |u_rel| 1.3e-13; σ = 0, adv = sạch → view == companion; IoU(AABB mask warp, box warp) mean 0.754 (trung vị 0.753, p5 0.456; PIA 0.823 — thấp hơn do 2 lần nội suy mask); mẫu số H_r > 0 trên toàn khung; GEO-C0 / PIA / MORGD trùng tuyệt đối HEAD, `morgd_views` HEAD == mới, C2-PIA lệch HEAD/mới 0.031 ≈ HEAD/HEAD 0.034; **chẩn đoán RNG: C2-GRGD và C2-PIA-RESIDUAL có mode PIA + nhiễu σ trùng tuyệt đối C2-PIA mỗi view (4 step × 2 view)** — sampler mmdet dùng `torch.randperm` trên CPU, nhiễu σ dùng RNG CUDA nên không lệch theo GT; 2 ô chạy được, L∞ đúng, forward no_grad = 2 × step.

**Chẩn đoán liều (forward-only, dev30, 6 view/ảnh, 156 view có N ≥ 2; 26/30 ảnh có N ≥ 2, trung vị N = 10) — T_relative theo spec gần như KHÔNG đổi hình học tương đối:**
- D_pair_pos mục tiêu 0.126 → hiệu dụng (sau trừ phối cảnh) **0.025** (tỉ lệ 0.22); theo N: 2–4 object 0.180 → 0.023 (0.13), 5–10: 0.135 → 0.028, ≥ 11: 0.094 → 0.025.
- |u| trung bình raw / M·u_H / rel: 8.4 / 9.8 / 6.0 px (N 2–4), 3.7 / 3.2 / 2.8 px (N ≥ 11); PIA ~179 px, MORGD ~41 px. R_proj mean 0.21, có view âm (bước trừ thêm biến dạng lớn hơn phần nó bỏ).
- Ca 2 box (unit test 4): mục tiêu dời tương đối (22.0, −14.4) px; không trừ phối cảnh → (20.5, −13.5); có trừ → **(1.5, −0.5)**.
- Hai nguyên nhân: (1) nghiệm bình phương tối thiểu trên đồ thị đầy đủ là t_i = (1/N) Σ_j d_ij → biên độ co theo N; (2) homography 8-DOF giải thích gần hết chuyển động của vài object (affine đã đưa 3 tâm tới vị trí bất kỳ) → bước trừ phối cảnh xóa phần lớn chính quan hệ cặp mà nhánh này định tạo.
- Hệ quả: chạy đúng spec thì C2-GRGD và C2-PIA-RESIDUAL gần như là C2-PIA cộng thêm warp ~3–6 px → kết quả ngang nhau sẽ không phân biệt được "hình học tương đối không có ích" với "liều quá nhỏ". **Chưa chạy attack — chờ user quyết** (chạy đúng spec / sửa spec trước khi chạy / dừng nhánh).

---

## 2026-10-04 — GRGD-v0 sửa (B): bỏ phép trừ phối cảnh + cổng độ trung thực forward-only (user, khóa TRƯỚC khi chạy cổng; CHƯA có attack nào của GRGD)

**Quyết định (user):** *GRGD-v0 original projection is rejected before attack execution.* Chẩn đoán forward cho thấy fit rồi trừ trường phối cảnh tốt nhất xóa chính chuyển động tương đối giữa object, vì chuyển động phối cảnh và hình học tương đối giữa object KHÔNG phải hai không gian rời nhau (homography toàn cục cũng đổi khoảng cách, tỉ lệ, vị trí tương đối). GRGD được định nghĩa lại TRƯỚC khi thấy bất kỳ kết quả attack nào: thành phần toàn cục = H_PIA do chính ta rút; nhánh relative từ đồ thị cặp chỉ bị ràng buộc gauge Σ ω_i t_i = 0 và Σ ω_i a_i = 0 (a = log s). **T_GRGD(x) = W_rel(H_PIA(x)), u_rel = u_raw** (bước 4 của entry GRGD-v0 bỏ; fit H_r giữ lại CHỈ để chẩn đoán). Không tune hyperparameter attack nào; λ_r = 0.25 giữ nguyên. Mọi thứ khác của entry GRGD-v0 (đồ thị đầy đủ, KKT, kernel, RNG, 3 ô, Δ_object = GRGD − RESIDUAL) giữ nguyên.

**Code:** `RelativeField(remove_projective=False)` mặc định (True = spec v0 gốc, chỉ để đối chiếu); stats thêm `d_pair_pos_req` (theo d_ij đã rút, chuẩn hóa (√A_i + √A_j)/2), `rho_pos` = D_pair_pos_eff / D_pair_pos_req, `cos_pair_eff_req`, `d_pair_scale_req`, `rms_rel_px`, `d_pair_pos_box` / `d_pair_scale_box` (đo trên GT box trước/sau warp relative, cặp còn giữ cả hai). Unit test chạy lại ĐẠT toàn bộ + 2 mục mới: mặc định u_rel == u_raw; 2 box sửa B dời hiệu dụng / mục tiêu = 0.93 / 0.94 (spec v0: (1.5, −0.5) vs (22.0, −14.4)).

**Cổng độ trung thực (forward-only, `scripts/grgd_fidelity_gate.py` → `results/grgd/fidelity_gate.json`):** dev30, 10 view/ảnh, seed như `run_baselines` (SEED + chỉ số dev300; ξ, tâm ngẫu nhiên từ RNG riêng seed + 1 / + 2), GRGD và RESIDUAL cùng H, ξ, nhiễu; chỉ view N ≥ 2.
- **G1:** median ρ_pos (GRGD) ≥ 0.75. "Requested" = d_ij đã rút (KHÔNG phải t sau khi giải) — để cổng bắt được cả việc nghiệm bình phương tối thiểu co biên độ.
- **G2:** mean D_pair_pos_eff > 0 VÀ mean D_pair_scale_box > 0 (GRGD).
- **G3:** mean RMS |u_rel| GRGD / RESIDUAL ∈ [0.9, 1.1] (liều hình học khớp).
- **PASS cả 3 → chạy 2 ô attack** C2-GRGD, C2-PIA-RESIDUAL @ dev30 (mô tả, như entry GRGD-v0). **FAIL → KHÔNG chạy attack**, ghi log; nếu trượt G1 do co bình phương tối thiểu thì ứng viên sửa tiếp (user nêu) là đồ thị thưa liên thông (cây khung, |E| = N − 1, ràng buộc nhất quán theo cấu trúc) — cần entry khóa mới, KHÔNG tăng λ_r để bù.
- **Dự đoán ghi trước (giải tích):** với đồ thị đầy đủ và d_ij độc lập, t_i − t_j = (1/N)[2d_ij + Σ_{k≠i,j}(d_ik + d_kj)] → ‖t_i − t_j‖ / ‖d_ij‖ ≈ √(2/N) (≈ 0.45 ở N = 10); unit test (ảnh 4 object) đã cho ρ_pos mean 0.45 → **G1 nhiều khả năng TRƯỢT**.

**Kết quả cổng (`results/grgd/fidelity_gate.json`; 257 view N ≥ 2, N trung bình 12.5) — TRƯỢT (G1, G3) → KHÔNG chạy attack GRGD (theo quy tắc khóa):**
- **G1 TRƯỢT:** median ρ_pos = **0.19** (ngưỡng 0.75). Theo N: 2–4 object 0.25 (√(2/N) = 0.80), 5–10: 0.17 (0.53), ≥ 11: 0.18 (0.33). cos(dời cặp hiệu dụng, d_ij) mean 0.30.
- G2 đạt (tầm thường): D_pair_pos_eff 0.049, D_pair_scale_box 0.049.
- **G3 TRƯỢT:** RMS |u_rel| GRGD / RESIDUAL = **0.82** (7.5 vs 9.2 px) — neo ngẫu nhiên rải đều nên kernel ít chồng nhau hơn, triệt tiêu ít hơn.
- **Tách 2 tầng mất mát (mean):** d_ij yêu cầu 0.212 → nghiệm t 0.132 (×0.62, co bình phương tối thiểu — nguyên nhân 1) → hiệu dụng sau ráp trường 0.049 (×0.37). **Tầng thứ hai lớn hơn và KHÔNG được dự đoán:** kernel Gaussian rộng bằng cỡ box của các object gần / chồng nhau (COCO) bị trung bình hóa trong u = Σ K_i u_i / (Σ K_i + ε), nên tại c'_i trường không bằng t_i. Ca 2 box tách xa trong unit test (0.93) không đại diện. → Đồ thị cây khung chỉ sửa tầng ×0.62; trần lý thuyết vẫn ≈ 0.37 < 0.75 nếu giữ cách ráp trường của MORGD.
- **Quan trọng cho tính hợp lệ của phép thử E2 − E1:** thay đổi hình học tương đối mà GT box (thứ detector thấy) THỰC SỰ chịu gần như như nhau giữa hai ô — D_pair_pos_box GRGD **0.046** = RESIDUAL **0.046**; D_pair_scale_box 0.049 vs 0.039. Trường neo ngẫu nhiên cũng làm lệch quan hệ giữa các GT object ngang trường neo theo object → ở thiết kế này hai ô gần như không khác nhau về đại lượng mà giả thuyết nói tới.
- Không có attack nào của GRGD được chạy (v0 lẫn sửa B). Hướng tiếp (user quyết): dừng nhánh, hoặc khóa entry mới sửa CẢ hai tầng (cây khung + cách ráp trường giữ được t_i tại object, vd phân hoạch theo mask/box thay kernel Gaussian rộng) và đặt lại cổng (gồm điều kiện D_pair_pos_box GRGD > RESIDUAL).

---

## 2026-10-04 — Pivot (user): DINO-Swin-L làm failure case — PIA phá được Swin-L qua đâu? (forward-only @ dev30, quy tắc khóa TRƯỚC khi chạy)

**Bối cảnh:** dừng nhánh MORGD/GRGD (user). Giả thuyết mới (user): "Swin-L bền ≈ distortion hình học đa scale không tới được phân cấp nông→sâu của Swin-L + encoder dập". PIA là phép hình học duy nhất có tín hiệu rõ trên Swin-L (C2-PIA 34.7 vs OSFD 26.8 vs C2 26.1; PIA − GEO-C0 +7.2 @ dev30). Nếu lợi của PIA đi qua stage nông và/hoặc làm encoder bớt dập → cơ sở thiết kế "Multi-Scale Projective Diversity" (CHƯA code). Nếu không → bỏ tiền đề đa scale.

**Đo (chỉ forward, PNG có sẵn `artifacts/runs/dev300_B50_eps5/adv/`, dev30, `scripts/mech_swinl_geo.py` → `results/ablation/mech/swinl_geo.json`):** DINO-Swin-L, ảnh sạch vs adv qua cùng test pipeline (như `mech_swinl.py`): backbone stage 1–4 (stride 4/8/16/32), ChannelMapper 5 mức, memory encoder (toàn bộ + tách 5 mức theo `spatial_shapes`). Chỉ số cosd = 1 − cos (vector phẳng) và rel = ‖a − c‖/‖c‖, mỗi ảnh. A_enc(ảnh) = cosd_encoder(toàn bộ) / cosd_backbone stage 4. Kèm A_l = cosd_encoder mức l / cosd_neck mức l.

**Ô:** cặp CHÍNH (cùng loss, khác hình học): **P1 = C2-PIA vs C2** (`PROBE-osfd+cls`), **P2 = PIA vs GEO-C0** (cùng co-transform feature sạch — cặp sạch nhất). Mô tả: OSFD, PIA-aff, PIA-d0.3, MORGD, SMOOTH, C2-MORGD, C2-SMOOTH (tham chiếu âm).

**Quy tắc (paired bootstrap 1000 mẫu trên ảnh, BOOT_SEED 2026; Δ = mean(A) − mean(B); rel = mean(A) / mean(B)):**
- **SHALLOW-SPECIFIC** đúng trong một cặp nếu ∃ stage s ∈ {1, 2}: cận dưới CI Δcosd_s > 0 VÀ rel_s ≥ 1.25 VÀ rel_s ≥ rel_4 (tăng ở nông không kém tăng ở sâu — phân biệt "tới được scale nông" với "mọi thứ mạnh hơn đều").
- **ENCODER** đúng trong một cặp nếu cận dưới CI ΔA_enc > 0.
- **Kết luận:** SHALLOW-SPECIFIC ở CẢ P1 và P2 → ủng hộ tiền đề đa scale → được khóa entry thiết kế Multi-Scale Projective Diversity. Không SHALLOW nhưng ENCODER ở cả hai → một phần: hình học tạo distortion encoder khó sửa nhưng không qua stage nông; tiền đề đa scale KHÔNG được ủng hộ, user quyết. Không cái nào ở cả hai cặp → bác giả thuyết, KHÔNG xây hình học đa scale.
- Mô tả (không quyết định): cosd encoder theo 5 mức, A_l; Spearman giữa các ô (11 ô) của mean cosd từng tầng với AP drop DINO-Swin-L đã có; câu hỏi mode PIA (Q3) chỉ trả lời gián tiếp qua PIA-aff (chỉ 6-DOF) và PIA-d0.3 — tách thật theo mode cần attack mới, chưa làm.
- Caveat: n = 30 (CI rộng); PNG C2 là bản sinh lại trên dev100 (cùng seed).

**Kết quả (`results/ablation/mech/swinl_geo.json`, n = 30) — theo quy tắc khóa: BÁC giả thuyết (không SHALLOW-SPECIFIC ở cặp nào; ENCODER chỉ ở P1) → KHÔNG xây hình học đa scale.**

cosd trung bình, DINO-Swin-L (drop % | bb1 / bb2 / bb3 / bb4 | neck5 | enc | A_enc):
- OSFD 26.8 | 0.070 / 0.115 / 0.281 / 0.426 | 0.246 | 0.067 | 0.153
- GEO-C0 21.5 | 0.071 / 0.121 / 0.271 / 0.392 | 0.206 | 0.056 | 0.142
- PIA 28.8 | 0.071 / 0.121 / 0.285 / 0.420 | 0.232 | 0.063 | 0.147
- C2 26.1 | 0.070 / 0.106 / 0.227 / 0.337 | 0.171 | 0.042 | 0.123
- C2-PIA 34.7 | 0.070 / 0.109 / 0.246 / 0.369 | 0.195 | 0.053 | 0.140
- (PIA-aff, PIA-d0.3, MORGD, SMOOTH, C2-MORGD, C2-SMOOTH trong file.)

- **Stage nông KHÔNG đổi theo bất kỳ phương pháp nào:** bb1 = 0.070–0.071 ở cả 11 ô (P1 Δ +0.000, P2 Δ −0.000); bb2 0.106–0.122, rel P1 ×1.03, P2 ×1.00. Mức distortion stage 1 có vẻ là "sàn" của nhiễu ε = 5 ở độ phân giải cao, không phụ thuộc loss hay hình học → tiền đề "đưa distortion hình học tới scale nông" không có chỗ bám trong dữ liệu.
- **PIA tăng ở SÂU và ở encoder:** bb4 ×1.09 (P1) / ×1.07 (P2), neck5 ×1.14 / ×1.13, encoder ×1.27 / ×1.13 (mọi mức encoder trừ enc5 ở P1 có CI > 0). A_enc: P1 +0.016 [+0.008, +0.026] (ENCODER đúng), P2 +0.005 [−0.001, +0.012] (không).
- **Độ lớn distortion KHÔNG giải thích AP drop của Swin-L ở bất kỳ tầng nào:** Spearman giữa 11 ô của mean cosd từng tầng với drop Swin-L nằm trong [−0.45, +0.18] (bb2 −0.45, enc −0.15, A_enc −0.34). Họ C2 có distortion THẤP hơn họ loss-feature ở MỌI tầng (C2-PIA enc 0.053 < OSFD 0.067, PIA 0.063) nhưng C2-PIA có drop cao nhất (34.7). Khớp phát hiện cũ ("task loss có cosd encoder thấp hơn OSFD nhưng drop cao hơn"): lợi trên Swin-L đến từ HƯỚNG (tín hiệu phân loại sau encoder), không từ lượng distortion feature.
- Q3 (mode PIA, gián tiếp): PIA-aff ≈ PIA ở mọi tầng (bb4 0.410 vs 0.420, enc 0.056 vs 0.063); PIA-d0.3 ở giữa. Không có dấu hiệu số hạng phối cảnh tạo loại distortion riêng.
- Đọc chung: trong cặp cùng loss, PIA làm feature sâu + encoder lệch thêm ~10–25% tương đối, nhưng trục "lượng distortion" không theo được AP drop giữa các ô → muốn hiểu vì sao C2-PIA phá Swin-L tốt hơn cần đo ở đầu ra (điểm phân loại query decoder, xếp hạng, loại lỗi) chứ không ở độ lớn feature. Hướng tiếp: user quyết.

---

## 2026-10-04 — Phân tích đầu ra decoder DINO-Swin-L: C2-PIA phá Swin-L qua phân loại hay định vị? (forward-only @ dev30, quy tắc khóa TRƯỚC khi chạy)

**Câu hỏi (user):** lợi của C2-PIA trên DINO-Swin-L (34.7 vs C2 26.1) có nằm ở bằng chứng PHÂN LOẠI của query khớp GT (margin, xếp hạng) không, hay ở định vị / gán lại query. Chưa thiết kế method; chưa phân tích theo tầng decoder (chỉ làm nếu đầu ra cuối có hiệu ứng).

**Cặp:** P1 = C2-PIA vs C2 (`PROBE-osfd+cls`) — chính, cùng loss khác hình học; P2 = PIA vs GEO-C0. Mô tả: OSFD, PIA-aff. PNG có sẵn, dev30, test pipeline như `mech_swinl.py`.

**Định nghĩa "query cố định" cho DINO (điều chỉnh bắt buộc so với đề xuất, user được báo):** DINO two-stage — slot k của decoder = embedding nội dung học sẵn thứ k + box tham chiếu của token encoder có điểm xếp hạng thứ k (`DINO.pre_decoder`: top-900 theo max_c logit encoder). Slot k trên adv vì vậy KHÔNG cùng object với slot k trên ảnh sạch. → **Primary ("forced"):** trên adv, decoder dùng ĐÚNG tập + thứ tự token encoder đã chọn trên ảnh sạch (topk_indices sạch; box tham chiếu và memory lấy từ adv tại các token đó) → slot k cùng token, cùng embedding nội dung, chỉ khác nội dung ảnh. Loại bỏ hiệu ứng chọn/gán lại query (đo riêng ở secondary). **Secondary ("natural"):** forward bình thường trên adv.

**Chọn query trên ảnh sạch (giống nhau cho mọi điều kiện):** với mỗi GT i (lớp c_i), q*_i = argmax_q IoU(box cuối của q, GT), giữ GT nếu IoU ≥ 0.5. **Dự phòng khai báo trước:** nếu trung vị sigmoid(z_{q*,c}) trên ảnh sạch < 0.3 (argmax IoU rơi vào query trùng lặp điểm thấp) → q*_i = argmax z_{q,c_i} trong các query IoU ≥ 0.5.

**Chỉ số mỗi GT (lớp cuối decoder, logit thô — sigmoid focal):** s = z_{q*,c}; margin m = z_{q*,c} − max_{c'≠c} z_{q*,c'}; rank percentile r = (R − 1)/(Q − 1), R = hạng của z_{q*,c} trong 900 query theo lớp c (1 = cao nhất); I = IoU(box q*, GT). Δ = adv − sạch.

**Quy tắc (paired bootstrap theo ẢNH, 1000 mẫu, BOOT_SEED 2026; trung bình theo GT có trọng số ảnh):**
- **CLS** đúng ở cặp (A, B) nếu cận trên CI[Δm(A) − Δm(B)] < 0 VÀ cận dưới CI[Δr(A) − Δr(B)] > 0.
- **LOC** đúng ở cặp nếu cận trên CI[ΔI(A) − ΔI(B)] < 0 (chỉ xác định KIỂU cơ chế, không phải điều kiện của CLS).
- Diễn giải: P1 CLS + P2 CLS → hình học PIA tự tạo bất ổn phân loại query; **P1 CLS, P2 không → tương tác loss × hình học**; P1 không CLS nhưng LOC → lợi ở định vị / gán query; P1 và P2 đều không CLS, không LOC → giả thuyết mức query bị bác, chuyển sang xếp hạng toàn cục / FP.

**Secondary (mô tả):** (a) natural: tỉ lệ token sạch của q* bị loại khỏi top-900 adv; trên phần còn lại: Δm, Δr, ΔI của slot chứa token đó. (b) gán lại: s_best = max_q z_{q,c_i} (natural adv), IoU của query đạt max; Δs_best. (c) Phân rã lỗi (natural, 100 detection như `predict` của DINO, ngưỡng 0.3), trên GT được phát hiện ở ảnh sạch (đúng lớp, IoU ≥ 0.5, score ≥ 0.3): preserved (vẫn vậy trên adv) → nếu không: **conf/rank failure** (có query adv IoU ≥ 0.5 với GT, bất kể lớp; tách riêng phần nhầm lớp) → **loc failure** (có query adv sigmoid z_c ≥ 0.3 với 0.1 ≤ IoU < 0.5) → **lost**. (d) FP/ảnh (score ≥ 0.3): nền / định vị / nhầm lớp như `probe_error_types.py`, phân bố score FP.

**Sanity trước khi đọc kết quả:** forced trên ảnh sạch == natural trên ảnh sạch (logit trùng); 100 detection natural khớp `model.predict` (box, score); tỉ lệ GT có q* IoU ≥ 0.5; trung vị score sạch của q*; chọn q* độc lập điều kiện.

**Code + sanity (TRƯỚC khi chạy đủ dev30):** `scripts/diag_decoder_output.py`. Sanity trên 3 ảnh đầu: forced trên ảnh sạch == natural (logit lệch tối đa 0.0); detection tự dựng khớp tuyệt đối `model.predict` (score, box lệch 0.0) — sửa ghi chú: DINO `test_cfg.max_per_img` = **300** (không phải 100), script lấy từ model; phân rã lỗi / FP dùng ngưỡng 0.3 nên gần như không đổi. 9/9 GT có q* IoU ≥ 0.5. **Trung vị sigmoid score sạch của q* (argmax IoU) = 0.012 < 0.3 → kích hoạt dự phòng khai báo trước: chạy chính với `--select maxlogit`** (q* = argmax z_{q,c} trong các query IoU ≥ 0.5). Số của lần thử 3 ảnh KHÔNG dùng.

**Kết quả @ dev30 (`results/ablation/mech/decoder_output.json`, `--select maxlogit`) — theo quy tắc khóa: P1 không CLS, không LOC; P2 cũng không → GIẢ THUYẾT MỨC QUERY BỊ BÁC.**

Sanity đủ 30 ảnh: forced sạch == natural (0.0), khớp `predict` (0.0); 352/362 GT (97%) có q*; trung vị score sạch q* 0.51 (75.6% ≥ 0.3). q* sạch: logit 0.17, margin 4.14, rank pct 0.011, IoU 0.82.

Forced, Δ (adv − sạch) trung bình theo GT [CI 95%] — d_s / d_margin / d_rank_pct / d_iou:
- OSFD −1.21 / −1.16 / +0.067 / −0.110
- GEO-C0 −1.06 / −1.02 / +0.051 / −0.102
- PIA −1.20 / −1.15 / +0.060 / −0.109
- PIA-aff −1.16 / −1.09 / +0.058 / −0.099
- C2 −1.26 / −1.18 / +0.060 / −0.111
- C2-PIA −1.30 / −1.19 / +0.063 / −0.118

- **P1 (C2-PIA − C2):** d_margin −0.006 [−0.18, +0.18], d_rank +0.003 [−0.013, +0.022], d_iou −0.007 [−0.023, +0.009] → không khác. **P2 (PIA − GEO-C0):** d_margin −0.14 [−0.25, −0.03] (CI < 0) nhưng d_rank +0.009 [−0.006, +0.024] → CLS sai theo quy tắc (cần cả hai); d_iou −0.007.
- **Ở mức query, tác động lên DINO-Swin-L gần như BẤT BIẾN theo phương pháp:** cả 6 ô giảm margin ~1.0–1.2 logit, rank tụt 5–7 percentile, IoU −0.10 đến −0.12 — chênh giữa các ô nhỏ hơn nhiều so với CI.
- Natural (mô tả): token q* bị loại khỏi top-900 19–24% (C2-PIA 23.6% cao nhất, C2 21.3%); trên phần còn lại margin tụt 1.21–1.41. Gán lại: max_q z_c (adv) chỉ giảm 0.29–0.42 logit (forced giảm ~1.2) → decoder phần lớn "nhận lại" bằng query khác (IoU của query max chỉ 0.19–0.21 — bị lẫn bởi instance khác cùng lớp, chỉ để tham khảo).
- Phân rã lỗi (266 GT phát hiện được ở ảnh sạch): preserved C2-PIA **0.726** vs C2 0.771 / OSFD 0.763 / PIA 0.756 / GEO-C0 0.774; phần mất thêm của C2-PIA rơi vào conf/rank (0.241 vs 0.211). Loc / lost ≤ 3% ở mọi ô. FP/ảnh: họ C2 tăng FP nền (2.9 vs sạch 1.6), C2-PIA không hơn C2 (5.27 vs 5.70 tổng).
- Đọc chung: chênh AP +8.6 (C2-PIA − C2 trên Swin-L @ dev30, CI [0.3, 13.6]) KHÔNG thấy ở bằng chứng phân loại / xếp hạng / định vị trung bình theo GT trên query cố định; chỉ thấy ~4.5 điểm preserved@0.3 (12/266 GT) — cỡ nhiễu ở n = 30. Hai khả năng chưa tách: (a) lợi AP tập trung ở ít ảnh / ít GT (đuôi), (b) chênh AP dev30 phần lớn là nhiễu (cận dưới CI 0.3). Hướng tiếp: user quyết.

---

## 2026-10-04 — Xác nhận C2-PIA trên DINO-Swin-L @ dev100 (user: "khóa rồi chạy") — quy tắc khóa TRƯỚC khi chạy

**Lý do:** 3 vòng phân tích (đa scale, distortion, đầu ra decoder) không thấy dấu vết cơ chế cho chênh Swin-L +8.6 (C2-PIA − C2 @ dev30, CI [0.3, 13.6]) → phải xác nhận chính hiện tượng trước khi giải thích tiếp.

**Chạy:** `run_baselines.py --split dev300 --subset dev100 --methods OSFD PROBE-osfd+cls GEO-C0 PIA C2-PIA --out metrics_pia_dev100.json` (resume: OSFD, C2 có đủ 100 PNG; GEO-C0 / PIA / C2-PIA attack thêm 70 ảnh, cùng seed SEED + chỉ số dev300, code KHÔNG đổi so với lần chạy dev30) → `eval_generalization.py --subset dev100 --models yolox_s dino_swin_l` → `scripts/pair_screen.py --tag confirm_dev100 --pairs C2-PIA:PROBE-osfd+cls PIA:GEO-C0 C2-PIA:OSFD --exclude dev30`. Paired bootstrap 1000, BOOT_SEED 2026.

**Quy tắc (trên dev100, n = 100):**
- **P1 XÁC NHẬN** nếu Δ DINO-Swin-L (C2-PIA − C2) ≥ +3 VÀ cận dưới CI 95% > 0. Ngược lại **KHÔNG XÁC NHẬN**.
- P2 (PIA − GEO-C0) cùng quy tắc, báo riêng.
- XÁC NHẬN → hiện tượng có thật ở dev100: (i) phân tích cơ chế (decoder / theo ảnh) chạy lại trên dev100 trước khi kết luận; (ii) C2-PIA đủ điều kiện làm ứng viên sàng lọc dev300 theo plan_sota bước 3 — cần entry khóa riêng, user quyết. KHÔNG XÁC NHẬN → dừng giải thích lợi Swin-L của C2-PIA; chênh dev30 coi là nhiễu / không ổn định.
- Mô tả (không quyết định): CrossAvg₃, R101, từng target; Δ C2-PIA − OSFD; cùng các chỉ số trên 70 ảnh mới (dev100 \ dev30) — dev100 chứa dev30 nên không độc lập hoàn toàn.

**Kết quả @ dev100 (`results/ablation/confirm_dev100/screen.json`; `metrics_pia_dev100.json`, `generalization_pia_dev100.json`; log `results/confirm_dev100/run.log`) — theo quy tắc khóa: P1 XÁC NHẬN, P2 XÁC NHẬN.**

Relative AP drop % dev100 (n = 100), R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- OSFD: 96.4 / 91.8 / 78.4 / 74.8 / 69.8 / 24.0 | 74.3
- GEO-C0: 96.3 / 87.7 / 73.4 / 72.6 / 70.5 / 22.4 | 72.2
- PIA: 94.5 / 88.2 / 76.4 / 74.2 / 70.4 / 28.5 | 73.7
- C2: 99.3 / 94.0 / 81.2 / 79.4 / 69.1 / 31.4 | 76.6
- **C2-PIA: 96.7 / 93.3 / 86.2 / 84.3 / 77.1 / 37.2 | 82.6**

- **P1 (C2-PIA − C2): DINO-Swin-L +5.8 [1.7, 9.1]** (p ≤ 0: 0.004) → XÁC NHẬN; CrossAvg₃ +6.0 [4.1, 8.8]; YOLOX +7.1 [4.5, 12.8], ConvNeXt +6.0, Swin-T +5.9 (cả hai CI > 0); R101 +0.4 [−2.1, 3.8].
- **P2 (PIA − GEO-C0): DINO-Swin-L +6.0 [2.3, 9.8]** → XÁC NHẬN; CrossAvg₃ +1.5 [0.3, 4.0].
- C2-PIA − OSFD (mô tả): DINO-Swin-L +13.2 [6.2, 15.5], CrossAvg₃ +8.2 [5.4, 11.3], Δ > 0 ở cả 3 target khác họ, R101 +1.5 (93.3 vs 91.8).
- 70 ảnh mới (dev100 \ dev30, mô tả): P1 Swin-L +3.6 [0.05, 7.3], CrossAvg₃ +6.3 [4.2, 9.8]; P2 Swin-L +5.7 [1.8, 8.7]; C2-PIA − OSFD Swin-L +14.2, CrossAvg₃ +6.4 → hiệu ứng tái lập trên ảnh không dùng ở dev30.
- Đọc chung: lợi của PIA trên DINO-Swin-L CÓ THẬT ở cả hai khung loss (≈ +6 điểm), và với loss cls còn kéo theo +6 CrossAvg₃ trên mọi target khác họ. Như vậy 3 phân tích cơ chế ở dev30 (đa scale, distortion, đầu ra decoder) đã thất bại trong việc giải thích một hiệu ứng có thật, không phải đang giải thích nhiễu. Theo quy tắc khóa: (i) phân tích cơ chế chạy lại trên dev100 trước khi kết luận; (ii) C2-PIA đủ điều kiện làm ứng viên sàng lọc dev300 (plan_sota bước 3) — cần entry khóa riêng, user quyết.

---

## 2026-10-04 — Chạy lại 2 phân tích cơ chế trên dev100 (user: không chạy dev300, phân tích tiếp trên dev100) — khóa TRƯỚC khi chạy

Theo quy tắc của entry xác nhận (i). Giữ NGUYÊN định nghĩa, cặp (P1 = C2-PIA vs C2, P2 = PIA vs GEO-C0), quy tắc và bootstrap (1000, BOOT_SEED 2026) của 2 entry gốc; chỉ đổi n = 30 → 100. Ô có PNG dev100: OSFD, GEO-C0, PIA, C2, C2-PIA (PIA-aff, PIA-d0.3, MORGD/SMOOTH chỉ có dev30 → bỏ).
1. **Distortion Swin-L** (entry "Pivot … failure case"): `mech_swinl_geo.py --subset dev100 --conds OSFD GEO-C0 PIA PROBE-osfd+cls C2-PIA --gen pia_dev100 --out swinl_geo_dev100.json`. SHALLOW-SPECIFIC / ENCODER như cũ. Spearman giữa ô chỉ còn 5 ô → chỉ mô tả.
2. **Đầu ra decoder** (entry "Phân tích đầu ra decoder"): `diag_decoder_output.py --subset dev100 --conds OSFD GEO-C0 PIA PROBE-osfd+cls C2-PIA --select maxlogit --out decoder_output_dev100.json`. Chọn q* bằng maxlogit (dự phòng đã kích hoạt ở dev30 — giữ cùng định nghĩa để so được giữa 2 cỡ mẫu); sanity in ra cả trung vị score q*. CLS / LOC như cũ.
Nếu kết luận dev100 khác dev30: dev100 là kết quả dùng (n lớn hơn, hiệu ứng AP đã xác nhận trên chính tập này); dev30 giữ trong log như bản thử.

**Kết quả @ dev100** (`results/ablation/mech/{swinl_geo_dev100,decoder_output_dev100}.json`, log `results/confirm_dev100/mech.log`).

**1. Distortion Swin-L — theo quy tắc: ENCODER ở cả P1 và P2, không SHALLOW → "một phần": PIA tạo distortion encoder khó sửa hơn nhưng KHÔNG qua stage nông; tiền đề đa scale vẫn không được ủng hộ.** (dev30: ENCODER chỉ P1 — dev30 thiếu lực.)
- bb1 0.067–0.068 ở cả 5 ô (lại bất biến); bb2 P1 ×1.04, P2 ×1.00.
- P1: bb4 ×1.13, neck5 ×1.21, encoder ×1.29 (enc3 ×1.37, enc4 ×1.44), A_enc +0.016 [+0.010, +0.022]. P2: bb4 ×1.08, encoder ×1.14 (enc4 ×1.27, enc5 ×1.36), A_enc +0.007 [+0.003, +0.012]. Tăng lớn nhất ở các mức encoder THÔ (stride 32–64).
- Giữa 5 ô, độ lớn distortion vẫn KHÔNG theo drop (ρ −0.2 đến −0.8): họ C2 distortion thấp hơn nhưng drop cao hơn. Trong cùng loss, PIA tăng cả distortion sâu/encoder lẫn drop.

**2. Đầu ra decoder — theo quy tắc: P1 không CLS (margin có, rank không) nhưng LOC → nhãn khóa trước "lợi ở định vị / gán query"; P2 giống hệt.**
- Sanity: forced sạch == natural (0.0), khớp `predict`; 955/982 GT có q* (97.3%); trung vị score sạch q* 0.56.
- P1 (C2-PIA − C2): d_margin **−0.19 [−0.30, −0.08]**, d_s −0.20 [−0.33, −0.07], d_rank_pct +0.007 [−0.003, +0.017] (không), d_iou **−0.020 [−0.030, −0.010]**. P2 (PIA − GEO-C0): d_margin −0.20 [−0.30, −0.10], d_rank +0.004 (không), d_iou −0.017 [−0.029, −0.006]. (dev30: P1 margin −0.006 — dev30 thiếu lực.)
- **Phân rã lỗi (736 GT phát hiện được ở ảnh sạch) KHÔNG khớp nhãn "định vị":** preserved C2-PIA **0.693** vs C2 0.746 (−5.3 điểm ≈ 39 GT), PIA 0.736 vs GEO-C0 0.769 (−3.3); phần mất thêm gần như toàn bộ là **conf/rank** (C2-PIA 0.265 vs C2 0.221; PIA 0.228 vs GEO-C0 0.198) — GT vẫn có query IoU ≥ 0.5 nhưng score < 0.3. Loc failure ≤ 0.7% và lost ≤ 2% ở mọi ô.
- Natural: token q* bị loại khỏi top-900 C2-PIA 23.4% vs C2 20.6%, PIA 21.3% vs GEO-C0 18.5% (+2.8 điểm cả hai cặp). Gán lại: max_q z_c giảm 0.36–0.53 (forced 1.2–1.5).
- FP/ảnh: C2-PIA 5.58 vs C2 5.20 (FP nền 3.59 vs 3.02); PIA 2.80 vs GEO-C0 2.98.
- **Đọc chung (không đổi nhãn khóa):** tiêu chí CLS trượt chỉ vì rank percentile không đổi; margin và score GT giảm có ý nghĩa, ΔIoU tuy CI < 0 nhưng rất nhỏ (−0.02, IoU sạch 0.84). Chỉ số gắn trực tiếp với AP (preserved@0.3) cho thấy lợi của PIA trên Swin-L là **hạ confidence lớp GT dưới ngưỡng trong khi vẫn định vị đúng**, cộng thêm phá chọn query nhẹ (+2.8 điểm deselection). Rank percentile trong 900 query có thể kém nhạy (q* sạch ở pct 0.009, adv ~0.07 ở mọi ô). Cơ chế phía feature: PIA làm lệch mạnh hơn ở feature sâu và đặc biệt mức encoder thô, encoder dập ít hơn (A_enc ↑) — hai phân tích nhất quán ở mức "PIA đẩy thêm một lượng nhỏ, đều" chứ không mở kênh mới. Hướng tiếp: user quyết.

---

## 2026-10-04 — Cải tiến MORGD (user): chẩn đoán vì sao MORGD âm → MORGD-v2 (phân vùng theo box) xếp chồng trên PIA @ dev30 — quy tắc khóa TRƯỚC khi chạy

**Giả thuyết:** MORGD chưa từng thực hiện được chuyển động riêng theo object. Cách ráp trường u = Σ K_i u_i / (Σ K_i + ε) với kernel Gaussian rộng bằng cỡ box trung bình hóa chuyển động của các object gần / chồng nhau (đã đo ở GRGD: tầng ráp trường làm mất ×0.37) → trường "neo theo object" ≈ warp mượt chung chung → object ≈ random. Prior thấp (3 lần âm: KG, A′1, MORGD); user chọn hướng này vì novelty.

**Bước 0 — chẩn đoán MORGD hiện tại (forward-only, `scripts/morgd_fidelity.py --version v1`):** dev30, 10 view/ảnh, seed như run_baselines (SEED + chỉ số dev300), `sample_anchors` + `flow_grid` gốc (λ = 0.25, s ∈ [0.8, 1.25], ε = 0.05). Mỗi object (MORGD): dời tâm thực tế = tâm AABB(mặt nạ chữ nhật warp) − tâm gốc; ratio_pos = ‖thực tế‖ / ‖Δ_i‖ (bỏ object |Δ_i| < 1 px), cos(thực tế, Δ_i). Cả MORGD và SMOOTH: D_pair_pos_box, D_pair_scale_box (như GRGD), RMS |u|.
- **Giả thuyết ĐƯỢC ỦNG HỘ** nếu median ratio_pos < 0.75 HOẶC D_pair_pos_box(MORGD) / D_pair_pos_box(SMOOTH) < 1.5 → sang bước 1. Ngược lại (MORGD đã thực hiện đúng mà vẫn ≈ random) → kết quả âm của MORGD là thật → **dừng nhánh**.

**Bước 1 — MORGD-v2 + cổng độ trung thực (forward-only):**
- Mỗi object i (box ở khung hiện tại; bản xếp chồng: box SAU PIA): t_i = (λ w_i r_x, λ h_i r_y), r ~ U(−1, 1), a_i = U(−ln 1.25, ln 1.25), λ = 0.25 (giữ của MORGD); gauge: trừ trung bình có trọng số diện tích ω (Σ ω t = 0, Σ ω a = 0); s_i = exp(a_i); c'_i = c_i + t_i.
- Ánh xạ ngược đồng dạng chính xác: u_i(q) = c_i + (q − c'_i)/s_i − q.
- Vùng của object i (khung ra) = AABB(box gốc ∪ box đã dời cỡ s_i·(w_i, h_i) tại c'_i) — phủ cả chỗ object rời đi để không để lại "bóng" object chưa gán nhãn. Trọng số mềm α_i(q) = σ((d_x)/τ_i)·σ((d_y)/τ_i), d = khoảng cách có dấu vào trong biên vùng, τ_i = 0.05·√(w_i h_i).
- Chồng lấn: ghép "over" theo diện tích giảm dần (object nhỏ ưu tiên, thường ở trước): u ← α_i u_i + (1 − α_i) u. Ngoài mọi vùng u = 0.
- SMOOTH-v2 (đối chứng): cùng cơ chế, CÙNG (t_i, s_i, kích thước theo chỉ số), tâm c_i đều ngẫu nhiên trên khung.
- RNG: (r, a) từ `random.Random(seed + 1)`, tâm ngẫu nhiên từ `random.Random(seed + 2)`; H của PIA và nhiễu σ như C2-PIA → mỗi view H, (t, s), σ trùng giữa các ô, chỉ khác tâm.
- GT co-transform: AABB mặt nạ chữ nhật warp, mask warp, bỏ instance rỗng / < 1 px (như MORGD).
- **Cổng (bản xếp chồng trên PIA, cả 3):** median ratio_pos (MORGD-v2) ≥ 0.75; D_pair_pos_box(MORGD-v2) / D_pair_pos_box(SMOOTH-v2) ≥ 1.5; RMS |u| MORGD-v2 / SMOOTH-v2 ∈ [0.9, 1.1]. Trượt → dừng nhánh, KHÔNG attack.
- Unit test trước cổng: λ = 0 hoặc không có GT → view trùng tuyệt đối pia_views; 1 object không chồng lấn → box dời đúng c' ± s·(w, h)/2 (≤ 2 px); xa mọi vùng u = 0; σ = 0, adv = sạch → view == companion; ô cũ trùng HEAD; chạy được, L∞ đúng.

**Bước 2 — attack @ dev30 (mô tả, khung C2: L_OSFD k = 3 + 4·(loss_rpn_cls + loss_cls), MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6, feature sạch + GT co-transform):** C2-PIA (PNG có sẵn, mốc) | C2-PIA-SMOOTH2 | **C2-PIA-MORGD2** | C2-MORGD2 (MORGD-v2 không PIA). Hiệu chính **Δ_object = C2-PIA-MORGD2 − C2-PIA-SMOOTH2**; kèm Δ_extra = C2-PIA-SMOOTH2 − C2-PIA.

**Bước 3 — quyết định (dev30, điểm ước lượng):** đi tiếp (khóa entry xác nhận dev100: Δ_object DINO-Swin-L hoặc CrossAvg₃ ≥ +3 VÀ cận dưới CI > 0) nếu Δ_object ≥ +2 trên DINO-Swin-L HOẶC CrossAvg₃ VÀ Δ_object > 0 ở ≥ 3/5 target khác surrogate (R101, ConvNeXt-T, Swin-T, YOLOX-S, DINO-Swin-L). **Δ_object ≤ 0 trên cả DINO-Swin-L và CrossAvg₃ → dừng hẳn nhánh MORGD.** Không thêm biến thể cứu vãn.

**Bước 0 — kết quả (`results/morgd2/fidelity_v1.json`, 3569 object-view): GIẢ THUYẾT ĐƯỢC ỦNG HỘ → sang bước 1.** median ratio_pos 0.91 (tứ phân vị 0.48 / 0.91 / 2.12) nhưng **cos(dời thực tế, Δ_i) trung bình chỉ 0.29** — object có di chuyển cỡ mục tiêu nhưng SAI HƯỚNG (bị object lân cận chi phối). D_pair_pos_box MORGD **0.215 < SMOOTH 0.278** (tỉ lệ 0.78 < 1.5): neo theo object còn đổi quan hệ tương đối giữa GT ÍT hơn neo ngẫu nhiên. D_pair_scale_box 0.126 vs 0.121; RMS |u| 45.5 vs 44.3 px. → MORGD v1 chưa từng thực hiện được chuyển động riêng theo object; kết quả âm của nó không trả lời được câu hỏi "theo object có ích không".

**Bước 1 — code + unit test (TRƯỚC cổng):** `attack/methods/morgd2.py` (`sample_motion`, `field`, `morgd2_views`); `geo_osfd_attack(geo="pia_morgd2" | "pia_smooth2" | "morgd2")`; ô `C2-PIA-MORGD2`, `C2-PIA-SMOOTH2`, `C2-MORGD2` (stats `morgd2`); `scripts/morgd_fidelity.py --version v2`. **Sửa spec phát hiện ở unit test (trước mọi cổng / attack):** vùng = AABB(gốc ∪ đã dời) **nới thêm 3τ_i mỗi phía** — không nới thì cạnh box nằm đúng chỗ α = 0.5 nên chỉ dời nửa (lệch 6.1 px); sau sửa lệch 0.64 px. N = 1 → gauge cho t = 0, s = 1 → bỏ warp (đồng nhất). `scripts/morgd2_unit_test.py` ĐẠT: λ = 0 và không GT → trùng tuyệt đối pia_views; 2 object tách xa → box dời đúng c' ± s·(w, h)/2 (≤ 0.64 px); xa mọi vùng |u| 2e-5 px; σ = 0, adv = sạch → view == companion (3 ô); GEO-C0 / PIA / MORGD trùng tuyệt đối HEAD, C2-PIA lệch HEAD/mới 0.036 = HEAD/HEAD; C2-PIA-MORGD2 / -SMOOTH2 có mode PIA + nhiễu σ trùng tuyệt đối C2-PIA mỗi view, C2-MORGD2 tất định; 3 ô chạy được, L∞ đúng, forward no_grad = 2 × step.

**Bước 1 — cổng v2 (`results/morgd2/fidelity_v2.json`, bản xếp chồng trên PIA, 3190 object-view): TRƯỢT → theo quy tắc khóa DỪNG nhánh MORGD, KHÔNG attack.**
- median ratio_pos **0.94** ≥ 0.75 ĐẠT (tứ phân vị 0.61 / 0.94 / 1.10; v1: 0.48 / 0.91 / 2.12); **cos trung bình 0.62** (v1 0.29).
- **D_pair_pos_box MORGD-v2 / SMOOTH-v2 = 1.22 < 1.5 TRƯỢT** (0.289 vs 0.236; v1 0.78). D_pair_scale_box 0.307 vs 0.217 (×1.41, không thuộc cổng).
- RMS |u| 45.1 vs 44.8 px (×1.01) ĐẠT.
- Đọc: v2 sửa được việc thực hiện chuyển động theo object (hướng đúng hơn gấp đôi, biên độ sát mục tiêu), và object đã vượt random (1.22 so với 0.78 ở v1). Nhưng đối chứng neo ngẫu nhiên vẫn đổi quan hệ giữa các GT tới ~82% mức của bản neo theo object: các vùng cỡ GT đặt ngẫu nhiên vẫn trùm và dời cứng một phần object thật. Về cấu trúc, mọi warp cục bộ cỡ object đều đổi hình học tương đối của GT, nên tương phản object − random chỉ còn ~20% đại lượng mà giả thuyết nói tới → so 2 ô attack ở dev30 (CI ±5) gần như chắc chắn không phân biệt được. Ngưỡng 1.5 đã khóa trước; không nới sau khi thấy số.

**User BỎ QUA cổng (quyết định SAU khi thấy số cổng v2) — chạy bước 2 để xem xu hướng.** Kết quả bước 2 chỉ MÔ TẢ, không dùng để biện minh cho nhánh; quy tắc bước 3 vẫn áp dụng như đã khóa (Δ_object ≤ 0 trên cả DINO-Swin-L và CrossAvg₃ → dừng hẳn), nhưng một kết quả dương cũng phải được đọc kèm việc cổng đã trượt (tương phản object − random chỉ ~20% về hình học). Chạy: `run_baselines.py --split dev300 --subset dev30 --methods OSFD PROBE-osfd+cls C2-PIA C2-MORGD C2-PIA-SMOOTH2 C2-PIA-MORGD2 C2-MORGD2 --out metrics_morgd2_dev30.json` → `eval_generalization.py --subset dev30 --models yolox_s dino_swin_l` → `pair_screen.py --subset dev30 --tag morgd2_dev30 --pairs C2-PIA-MORGD2:C2-PIA-SMOOTH2 C2-PIA-SMOOTH2:C2-PIA C2-PIA-MORGD2:C2-PIA C2-MORGD2:C2-MORGD C2-MORGD2:C2-PIA`. Log `results/morgd2/run.log`.

**Bước 2 — kết quả @ dev30 (mô tả; cổng đã trượt, user bỏ qua) — `results/ablation/morgd2_dev30/screen.json`, `metrics_morgd2_dev30.json`, `generalization_morgd2_dev30.json`, log `results/morgd2/run.log`.**

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ (runtime s/ảnh):
- C2-PIA: 99.5 / 94.8 / 86.8 / 85.9 / 78.2 / **34.7** | **83.6** (11.3)
- C2-PIA-SMOOTH2: 98.3 / 92.6 / 78.6 / 83.3 / 76.9 / 22.5 | 79.6 (12.6)
- C2-PIA-MORGD2: 96.8 / 89.3 / 78.3 / 76.9 / 71.7 / 23.3 | 75.7 (12.4)
- C2-MORGD (v1): 99.9 / 96.0 / 87.7 / 83.5 / 72.8 / 27.3 | 81.3
- C2-MORGD2: 99.7 / 93.6 / 80.2 / 81.9 / 70.8 / 21.5 | 77.6 (11.9)

- **Δ_object (C2-PIA-MORGD2 − C2-PIA-SMOOTH2): DINO-Swin-L +0.7 [−3.3, +3.4], CrossAvg₃ −4.0 [−7.4, −0.9]**; > 0 chỉ ở 1/5 target (Swin-L) → theo bước 3: KHÔNG đi tiếp (điều kiện "dừng hẳn" Δ ≤ 0 trên CẢ HAI không kích hoạt vì Swin-L +0.7, nhưng điều kiện đi tiếp trượt rõ) → **dừng nhánh MORGD**.
- **Thêm warp cục bộ lên PIA làm attack YẾU đi mạnh, bất kể neo:** SMOOTH2 − C2-PIA: Swin-L −12.2 [−15.9, −4.8], CrossAvg₃ −4.0; MORGD2 − C2-PIA: Swin-L −11.4, CrossAvg₃ −7.9 [−12.6, −5.0]; cả R50 white-box (99.5 → 96.8–98.3) và R101 (−2.2 / −5.6) giảm → tối ưu trên surrogate kém đi, không chỉ transfer.
- **MORGD-v2 đứng riêng kém v1:** C2-MORGD2 − C2-MORGD: Swin-L −5.8 [−9.1, −2.4], CrossAvg₃ −3.7 — trường cứng từng mảnh (seam) kém trường mượt.
- Đọc (giả thuyết, chưa kiểm): **có điểm tối ưu của đa dạng hình học ở B cố định.** PIA δ_max 0.6 có vẻ đã gần mức đó; thêm biến đổi (dù toàn cục RaPA hay cục bộ SMOOTH2/MORGD2) làm gradient giữa các view mất tương quan → với 50 step × 2 view thì δ không tích lũy được (cùng chiều với MORGD-RaPA −25). Hệ quả kiểm được: tăng số view mỗi backward (giảm phương sai gradient) sẽ cho phép biến đổi mạnh hơn mà không mất utility. Hướng tiếp: user quyết.

---

## 2026-10-04 — Giai đoạn A: PIA hoạt động nhờ YẾU TỐ hình học nào? Ablation leave-one-out các thành phần homography (khung loss feature thuần, KHÔNG C2) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh (user):** bỏ khung C2, giữ loss feature OSFD; mục tiêu là một dạng đa dạng hình học vượt SOTA và có novelty so với PIA (classification). Trong khung thuần @ dev100, PIA chưa vượt OSFD trên CrossAvg₃ (73.7 vs 74.3), chỉ hơn trên DINO-Swin-L (28.5 vs 24.0). Cách làm như plan_sota với OSFD: phân tích SOTA → tìm nguyên liệu → khuếch đại (Giai đoạn B, khóa sau). Giả thuyết nghiêng về (chưa kiểm): **co giãn** — đưa object qua ranh giới mức FPN (resize ≤ 1.1× của RRB vô dụng; PIA dời góc tới 30%).

**Tách homography (Hartley–Zisserman, H = H_S·H_A·H_P):** H của PIA (pixel ra → nguồn) đổi sang tọa độ TÂM ẢNH: H' = C⁻¹ H C, C = tịnh tiến gốc về tâm khung (để xoay / co giãn quanh tâm, không lẫn tịnh tiến), chuẩn hóa H'[2,2] = 1. Viết H' = [[A, t], [vᵀ, 1]]: H_P = [[I, 0], [vᵀ, 1]] (phối cảnh v, 2-DOF); M = A − t vᵀ = s·R(θ)·K, QR với đường chéo dương: R xoay (θ), s = √det (co giãn đẳng hướng), K = [[k₁, k₂], [0, 1/k₁]] (lệch tỉ lệ k₁ + trượt k₂); t tịnh tiến (2-DOF). Tổng 8-DOF.

**Ô (mỗi ô = PIA với ĐÚNG 1 thành phần đặt về đồng nhất, các thành phần còn lại giữ nguyên giá trị đã rút; ghép lại H'' = H_S·H_A·H_P → H = C H'' C⁻¹):** PIA-noT (t = 0), PIA-noR (θ = 0), PIA-noS (s = 1), PIA-noK (K = I), PIA-noP (v = 0). Khung = ô `PIA` (loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6, feature sạch co-transform, δ_max = 0.6, π paper). RNG như PIA (rút góc giống hệt, chỉ sửa H sau khi rút) → mỗi view của ô X = view PIA tương ứng bỏ 1 thành phần.

**Báo cáo kèm (gây nhiễu liều):** mỗi ô ghi |u| trung bình mỗi view (px) và phân bố từng thành phần trong PIA (|t|, |θ|, |ln s|, |ln k₁|, |k₂|, ‖v‖·cỡ ảnh). Bỏ một thành phần vừa đổi LOẠI vừa giảm BIÊN ĐỘ → phải đọc Δ kèm Δ|u|.

**Chạy & quy tắc:** dev30 trước (mô tả, xem xu hướng), rồi mở rộng ĐỦ dev100 cho cả 5 ô (không chọn lọc theo dev30; cùng seed, resume). **Quyết định chỉ ở dev100** (n = 100, paired bootstrap 1000, BOOT_SEED 2026), Δ = drop(PIA) − drop(ô), tính RIÊNG cho DINO-Swin-L và CrossAvg₃ (như ablation OSFD bước 1): cận dưới CI ≥ 2 → **nguyên liệu**; cận trên CI < 2 → **không phải**; còn lại → **mơ hồ**. PIA trên dev100 = PNG có sẵn (`metrics_pia_dev100.json`).

**Unit test trước khi chạy:** (1) ghép lại các thành phần == H (sai số tương đối < 1e-9, 10 000 mẫu); (2) mỗi ô: tách lại H đã bỏ → thành phần bị bỏ = đồng nhất, các thành phần khác không đổi; (3) mode 3/2/1 (v = 0 sẵn) → lưới PIA-noP ≈ PIA (≤ 1e-4); mode 1 → PIA-noT ≈ đồng nhất; (4) không truyền drop → PIA trùng tuyệt đối HEAD; (5) 5 ô chạy được, L∞ đúng, RNG (mode, nhiễu σ) trùng PIA.

**Code + unit test (TRƯỚC khi chạy):** `pia.decompose_h` / `compose_h` / `drop_factor` (tọa độ tâm, QR đường chéo dương), `pia_views(drop=, factor_log=)`, `geo_osfd_attack(pia_drop=)`; ô `PIA-noT/R/S/K/P` (stats `pia_factors`). `scripts/pia_factor_unit_test.py` ĐẠT: ghép lại == H (sai số tương đối ≤ 2.4e-14, 10 000 mẫu); bỏ 1 thành phần → thành phần đó = đồng nhất, khác không đổi (≤ 2.8e-13); mode 3/2/1: PIA-noP ≈ PIA (6e-8), mode 1: PIA-noT ≈ đồng nhất; PIA không drop trùng tuyệt đối HEAD; 5 ô chạy được, L∞ đúng, mode + nhiễu σ trùng PIA.
**Phân bố thành phần PIA (10 000 mẫu, khung 800 × 1199, trung vị):** |t| 157.5 px, |θ| 3.1°, |ln s| 0.104, |ln k₁| 0.091, |k₂| 0.140, v = 0 ở ~60% view (mode 3/2/1). **|u| trung bình:** PIA 194 px; noT 91, noR 192, noS 190, noK 181, noP 198 → biên độ PIA chủ yếu là TỊNH TIẾN (kèm vùng lấp 0 khi dịch); co giãn của PIA chỉ ~±10% (giả thuyết "vượt mức FPN" kém khả năng hơn dự kiến — ghi trước khi có kết quả).

**Đổi thứ tự (user, trước khi có số dev30):** chỉ chạy dev30 để xem xu hướng, TẠM HOÃN phần mở rộng dev100 (pipeline dừng sau khi xong dev30). Quy tắc "nguyên liệu / không phải / mơ hồ" vẫn chỉ áp dụng ở dev100 khi (nếu) chạy; số dev30 chỉ mô tả.

**Kết quả @ dev30 (mô tả; dev100 hoãn) — `results/ablation/piafac_dev30/screen.json`, `metrics_piafac_dev30.json`, `generalization_piafac_dev30.json`, log `results/pia_factor/run.log`.** (Pipeline dừng chủ động khi bắt đầu dev100; watcher chỉ dừng script cha, tiến trình con `run_baselines` dev100 chạy sót ~4 phút → đã attack thêm 11 ảnh dev100 \ dev30 cho PIA-noT (cùng seed, giữ lại để resume), rồi bị dừng tay. Không có eval dev100.)

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃ | |u| px:
- OSFD: 97.8 / 93.8 / 83.5 / 77.3 / 75.6 / 26.8 | 78.8
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / 28.8 | 79.5 | ~194
- PIA-noT: 97.0 / 91.1 / 82.0 / 81.8 / 77.7 / 29.8 | 80.5 | 90.5
- PIA-noR: 97.0 / 90.2 / 73.8 / 79.7 / 77.4 / 31.1 | 77.0 | 186.8
- PIA-noS: 97.1 / 89.8 / 79.0 / 80.3 / 78.9 / 27.5 | 79.4 | 181.8
- PIA-noK: 97.2 / 91.3 / 85.2 / 82.3 / 76.1 / 27.8 | 81.2 | 177.3
- PIA-noP: 96.8 / 90.4 / 81.8 / 75.9 / 76.0 / 25.9 | 77.9 | 190.8

Δ = PIA − ô (dương = thành phần có ích), DINO-Swin-L | CrossAvg₃ [CI 95%]:
- **T (tịnh tiến): −1.0 [−4.5, +3.0] | −1.0 [−2.7, +0.6]** — bỏ T giảm |u| hơn một nửa (194 → 91 px) mà KHÔNG mất gì (hơi tốt hơn) → tịnh tiến (phần lớn biên độ của PIA) không phải nguyên liệu; giả thuyết "đa dạng vị trí/pha lưới qua tịnh tiến lớn" không được ủng hộ.
- R (xoay): −2.3 [−5.4, +0.6] | +2.5 [−0.8, +4.2] — giúp ConvNeXt (+5.9), hơi hại Swin-L.
- S (co giãn): +1.2 | +0.1 — ≈ 0 → giả thuyết "co giãn vượt mức FPN" không được ủng hộ (co giãn PIA chỉ ~±10%).
- K (lệch tỉ lệ + trượt): +1.0 | −1.7 (ConvNeXt −5.5 [−10.6, −1.4]: bỏ K TỐT hơn cho ConvNeXt).
- **P (phối cảnh): +2.8 [−1.1, +6.0] | +1.6 [−0.2, +3.5]** — thành phần duy nhất dương ở cả Swin-L và CrossAvg₃ (YOLOX +2.4 [+0.3, +7.5], Swin-T +4.5); bỏ P gần như không giảm |u| (191 px) → lợi của P là do LOẠI biến đổi, không do biên độ.
- Đọc chung (n = 30, phần lớn CI chứa 0): nguyên liệu khả dĩ nhất của PIA là **số hạng phối cảnh**; tịnh tiến chiếm phần lớn biên độ nhưng "lãng phí" (bỏ đi không mất, khớp quan sát trần đa dạng–hiệu dụng). (Lưu ý: PIA-aff @ dev30 — đổi phân phối mode sang toàn 6-DOF — có Swin-L 27.9 ≈ PIA; phép bỏ P ở đây giữ nguyên các thành phần khác nên sạch hơn.) Muốn kết luận phải chạy dev100 (quy tắc khóa).

---

## 2026-10-04 — Giai đoạn B (thử nhanh @ dev30, user): "đa dạng mô hình camera" — xoay camera ảo thuần phối cảnh + méo ống kính (ngoài họ homography) — quy tắc khóa TRƯỚC khi code/chạy

**Bối cảnh:** Giai đoạn A @ dev30 gợi ý phối cảnh là nguyên liệu của PIA, tịnh tiến (≥ nửa biên độ) vô ích. User chọn thử thẳng method mới ở dev30 (bỏ xác nhận dev100 của Giai đoạn A). Rủi ro novelty đã nêu: xoay camera ảo vẫn trong họ homography của PIA (chỉ là phân phối rút khác) → thêm **méo ống kính xuyên tâm** — biến đổi camera toàn cục, mượt, có nghĩa vật lý, KHÔNG biểu diễn được bằng homography.

**Khung:** như ô `PIA` (loss feature OSFD k = 3, MI μ = 1, ε = 5, α = 1, B = 50, 2 view/step, 1 backward, σ = 6 cộng vào adv, feature sạch co-transform cùng lưới, KHÔNG cls). Seed ảnh SEED + chỉ số dev300; tham số biến đổi rút từ `random` của Python mỗi view; vùng ngoài ảnh nguồn lấp 0.

**Biến đổi (ánh xạ ngược pixel ra q → nguồn, khung model h × w, tâm c = ((w−1)/2, (h−1)/2)):**
- **CAM-R(θ):** K = [[f, 0, c_x], [0, f, c_y], [0, 0, 1]], f = max(h, w); yaw, pitch ~ U(−θ, θ) độc lập, roll = 0; H = K·R_y(yaw)·R_x(pitch)·K⁻¹; nguồn = H·q (chia đồng nhất). Hai liều θ = 15°, 30° (khai báo trước, không tune thêm).
- **LENS(κ):** d = q − c, ρ = ‖d‖ / R, R = nửa đường chéo; nguồn = c + d·(1 + k·ρ²), k ~ U(−κ, κ), κ = 0.25 (k > 0 / < 0: pincushion / barrel theo chiều ánh xạ ngược).
- **CAM-R+LENS:** nguồn = H·(c + d·(1 + kρ²)) — méo ống kính ở khung ra, rồi xoay camera; θ = 15°, κ = 0.25.
- GT co-transform (khi cần cho khung cls về sau): AABB mặt nạ chữ nhật warp như MORGD; không dùng trong khung thuần.

**Ô @ dev30 (mô tả):** CAM-R15, CAM-R30, LENS, CAM-R15+LENS; mốc: PIA, PIA-noT, OSFD, GEO-C0 (PNG có sẵn). Báo |u| trung bình mỗi view.
**Đọc xu hướng (khai báo trước, không phải cổng):** ứng viên "triển vọng" nếu ≥ PIA + 2 trên DINO-Swin-L VÀ ≥ PIA trên CrossAvg₃ (điểm ước lượng, dev30) → khóa tiêu chí xác nhận dev100 (so PIA, ≥ +3 Swin-L, cận dưới CI > 0). LENS hoặc CAM-R15+LENS triển vọng mới hỗ trợ được claim "ngoài homography".
**Unit test trước khi chạy:** θ = 0, κ = 0 → lưới ≈ đồng nhất; yaw thuần → tâm ảnh ánh xạ tới c_x + f·tan(yaw) (đúng chiều); LENS k > 0 → góc ảnh ánh xạ ra ngoài đúng k·R; σ = 0, adv = sạch → view == companion; ô cũ trùng HEAD; chạy được, L∞ đúng.

**Code + unit test (TRƯỚC khi chạy):** `attack/methods/camgeo.py` (`rotation_homography`, `camera_grid`, `cam_views`); `geo_osfd_attack(geo="cam", cam_theta=, cam_kappa=)`; ô CAM-R15, CAM-R30, LENS, CAM-R15+LENS. **Sửa spec phát hiện ở unit test (trước khi chạy):** xoay camera bậc nhất = dịch f·tan θ (yaw 15° ≈ 320 px; CAM-R15 |u| 254 px — chính thành phần tịnh tiến Giai đoạn A cho là vô ích) → **căn lại tâm**: H ← T·H, T đưa ảnh của tâm về tâm; chỉ giữ méo phối cảnh. `scripts/camgeo_unit_test.py` ĐẠT: θ = κ = 0 → đồng nhất (8e-6 px); yaw 10° căn tâm → tâm giữ nguyên, mép trái / phải dời 31.4 / 78.3 px (keystone bất đối xứng); LENS k = 0.2 → góc ra ngoài đúng k·R; σ = 0, adv = sạch → view == companion; GEO-C0 / PIA / PIA-noT trùng tuyệt đối HEAD; 4 ô chạy được, L∞ đúng. **|u| trung bình (sau sửa):** CAM-R15 22.3, CAM-R30 84.9, LENS 21.4, CAM-R15+LENS 31.2 px (PIA ~194, PIA-noT ~91).

---

## 2026-10-04 — Thí nghiệm cơ chế: "đa dạng có ích = biến đổi NGOÀI nhóm đối xứng của surrogate"? (gradient surrogate, dev30) — quy tắc khóa TRƯỚC khi chạy

**Mục tiêu (user):** hiểu biến đổi thế nào cho lợi tối đa, rồi diễn đạt bằng phép biến đổi cơ bản. **Giả thuyết:** biến đổi mà surrogate (gần) đồng biến — tịnh tiến (conv), co giãn đẳng hướng (FPN + huấn luyện đa scale) — cho gradient view chỉ là gradient gốc dời/co lại → trung bình không thêm hướng mới (khớp: tịnh tiến PIA vô ích, resize RRB vô dụng, co giãn PIA ≈ 0). Biến đổi phá đối xứng — xoay, trượt/lệch tỉ lệ, phối cảnh, méo ống kính — cho gradient khác thật → triệt tiêu phần nhiễu đặc thù cách surrogate xử lý hướng/hình dạng cục bộ (khớp: xoay là nguyên liệu RRB, phối cảnh là thành phần dương của PIA). Điểm chưa khớp đã biết: bỏ xoay khỏi PIA làm Swin-L tốt hơn (−2.3 @ dev30).

**Đo (`scripts/mech_equivariance.py` → `results/mechanism/equivariance.json`):** surrogate R50, loss feature OSFD (backbone 4 stage, k = 3), đích feature co-transform (F(T(x_sạch))), KHÔNG nhiễu σ. Gradient theo ảnh khung model g_T = ∇ₓ L(T(x)) (backprop qua grid_sample → tự về khung gốc); g_I = gradient không biến đổi. **Đa dạng hiệu dụng d_T = 1 − cos(g_T, g_I)** trên các pixel mà view phủ (mặt nạ nguồn hợp lệ), kèm tỉ lệ trùng dấu. Hai điểm δ: δ = 0 và δ = PNG OSFD (adv cuối) của cùng ảnh.
- **Họ biến đổi cơ bản (tham số quanh tâm ảnh):** tịnh tiến (hướng ngẫu nhiên), co giãn đẳng hướng (s hoặc 1/s), xoay, trượt (k₂ ngang hoặc dọc), phối cảnh (xoay camera căn tâm, yaw hoặc pitch), méo ống kính (k).
- **Cùng biên độ:** mỗi họ hiệu chỉnh tham số để |u| trung bình = 10 / 30 / 60 px (chia đôi trên khung từng ảnh); 4 lần rút (dấu / hướng) mỗi (ảnh, họ, mức). dev30, n = 30.
- **Quy tắc:** NGUYÊN LÝ ĐƯỢC ỦNG HỘ nếu ở CẢ 3 mức biên độ và CẢ 2 điểm δ: min(d_xoay, d_phối cảnh) − max(d_tịnh tiến, d_co giãn) > 0 với cận dưới CI 95% paired bootstrap theo ảnh (1000, BOOT_SEED 2026) > 0. Ngược lại: không được ủng hộ. Trượt và méo ống kính: mô tả.
- Mô tả kèm: so d_T với các ô đã chạy (tương quan giữa họ biến đổi và lợi transfer ở Giai đoạn A/B) — không quyết định.

**Kết quả @ dev30 (mô tả) — `results/ablation/cam_dev30/screen.json`, `metrics_cam_dev30.json`, `generalization_cam_dev30.json`, log `results/camgeo/run.log`.** |u| thực: CAM-R15 22.6, CAM-R30 78.5, LENS 20.1, CAM-R15+LENS 31.2 px.

Relative AP drop %, R50 / R101 / ConvNeXt-T / Swin-T / YOLOX-S / DINO-Swin-L | CrossAvg₃:
- PIA: 96.8 / 89.9 / 79.7 / 80.4 / 78.4 / **28.8** | 79.5
- PIA-noT: 97.0 / 91.1 / 82.0 / 81.8 / 77.7 / **29.8** | 80.5
- CAM-R15: 98.2 / 92.1 / 81.8 / 81.0 / 74.5 / 25.1 | 79.1
- **CAM-R30: 96.8 / 92.3 / 84.0 / 82.7 / 81.5 / 21.3 | 82.7**
- LENS: 97.8 / 91.1 / 81.4 / 79.5 / 69.5 / 24.8 | 76.8
- CAM-R15+LENS: 98.7 / 91.5 / 81.9 / 81.2 / 77.4 / 25.3 | 80.2

- **Không ô nào "triển vọng"** theo tiêu chí khai báo (≥ PIA + 2 trên Swin-L VÀ ≥ PIA trên CrossAvg₃).
- **CAM-R30 (phối cảnh thuần, 78 px): CrossAvg₃ +3.2 [+0.3, +5.7] so PIA** (cao nhất trong mọi ô hình học khung thuần đến nay; ConvNeXt +4.3, YOLOX +3.1, Swin-T +2.3, R101 +2.4) **nhưng DINO-Swin-L −7.4 [−10.8, −0.4]**. Đánh đổi rõ: phối cảnh thuần mạnh giúp target CNN / Mask R-CNN / YOLOX, hại DINO-Swin-L.
- LENS: yếu (CrossAvg₃ −2.7, YOLOX −8.9); thêm LENS lên CAM-R15 gần như không đổi (+0.2 Swin-L, +1.1 CrossAvg₃) → claim "ngoài homography" KHÔNG được ủng hộ ở dev30.
- Đọc: lợi Swin-L của PIA KHÔNG đến từ phối cảnh thuần đứng riêng; nó trải trên nhiều thành phần (Giai đoạn A: bỏ P −2.8, S −1.2, K −1.0, bỏ R +2.3) — có thể cần TỔ HỢP. Hai nhóm target phản ứng khác nhau với cùng một loại biến đổi → một tiêu chí "đa dạng" duy nhất có thể không đủ; thí nghiệm cơ chế (đang chạy) sẽ cho biết đa dạng gradient trên surrogate có theo được nhóm nào.

**Kết quả thí nghiệm cơ chế (`results/mechanism/equivariance.json`, n = 30, 4 lần rút) — NGUYÊN LÝ KHÔNG ĐƯỢC ỦNG HỘ (theo quy tắc khóa; sai cả 3 mức × 2 điểm δ, và sai CHIỀU).**

d_T = 1 − cos (δ = 0), |u| = 10 / 30 / 60 px: tịnh tiến 0.628 / 0.634 / 0.634; co giãn 0.626 / 0.658 / 0.714; xoay 0.620 / 0.658 / 0.718; trượt 0.500 / 0.581 / 0.699; phối cảnh 0.574 / 0.651 / 0.684; méo ống kính 0.571 / 0.661 / 0.751. (δ = OSFD gần như y hệt.) min(xoay, phối cảnh) − max(tịnh tiến, co giãn): −0.054 [−0.065, −0.045] / −0.007 [−0.012, −0.002] / −0.030 [−0.037, −0.023] — ÂM, CI < 0.

- **Tịnh tiến chỉ 10 px đã làm gradient mất tương quan ~63% (cos ≈ 0.37), ngang mọi họ khác.** Gradient theo pixel của surrogate cực nhạy với BẤT KỲ phép lấy mẫu lại nào (lệch pha stride + nội suy bilinear) → đa dạng gradient mức pixel BÃO HÒA ngay ở biên độ nhỏ, không phân biệt được loại biến đổi, và không theo được transfer (tịnh tiến: d cao nhưng vô ích về transfer).
- Hệ quả: "đa dạng hiệu dụng" đo bằng cos gradient pixel KHÔNG phải trục giải thích; giả thuyết "đa dạng có ích = phá đối xứng" (theo cách đo này) bị bác. Không thêm biến thể đo hậu kiểm vào quy tắc này. Hướng tiếp theo đã khai báo trước: đo ĐỒNG HƯỚNG giữa gradient surrogate (trung bình qua biến đổi) và gradient target — dùng target CHỈ để phân tích, không sinh attack. User quyết.
