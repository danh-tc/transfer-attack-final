"""Attack core: engine chung cho IFGSM / MI-FGSM / DI-FGSM (sign-gradient iterative attack).

Port từ ref-repo/OSFD-main/attack/base/{IFGSM,MI,DI}.py, gộp lại thành 1 hàm
thay vì hệ thống registry + pipeline 4-giai-đoạn của bản gốc — bản gốc cần
kiểu đó để tổ hợp tự do nhiều method + nhiều buffer/cache cho nhiều thí
nghiệm khác nhau trong paper OSFD; ở đây chỉ cần 3 biến thể cố định
(IFGSM/MI/DI) nên không cần abstraction đó.

Quy ước:
- Không gian pixel [0,255] (KHÔNG normalize) — normalize xảy ra bên trong
  model.data_preprocessor, xem attack/preprocess.py.
- epsilon, alpha tính bằng đơn vị pixel tuyệt đối (vd epsilon=5 ứng với
  epsilon=5/255 trong idea.md/protocol_lock.md, vì pixel ở đây scale
  0-255 chứ không phải 0-1).
- Cả 3 biến thể đều là GRADIENT ASCENT trên loss_fn (loss càng cao càng tốt
  cho attacker, đúng untargeted threat model idea.md §3) — kể cả khi ghép
  với loss của OSFD (xem attack/losses/osfd.py để hiểu vì sao ascent vẫn
  đúng hướng cho loss đó).
- `views_fn` (thay cho "diversity_fn" đơn giản trước đây): trả về 1 LIST cặp
  (ảnh, data_sample) từ 1 ảnh adv — length 1 cho DI-FGSM, length 2 cho RRB
  (batch-doubling, xem attack/methods/rrb.py), length N_EOT cho AugTrans
  (xem attack/methods/augtrans.py). MỖI VIEW MANG data_sample RIÊNG vì biến
  đổi hình học (rotate/resize) làm object DỊCH CHUYỂN trong ảnh — nếu loss
  phụ thuộc GT (vd AugTrans/DI-FGSM dùng compute_gt_loss) mà vẫn dùng
  data_sample gốc (box CHƯA biến đổi) trong khi ảnh ĐÃ biến đổi, loss tính
  sai vị trí hoàn toàn, gradient gần như vô nghĩa — đây từng là bug thật
  (xem docs/progress_log.md, attack/methods/box_transforms.py). Với
  view không đổi box (RRB — loss OSFD không phụ thuộc GT), cứ trả lại
  data_sample gốc nguyên vẹn. Loss của mỗi view được CỘNG lại trước 1 lần
  .backward() duy nhất — khớp đúng định nghĩa B trong protocol_lock.md
  ("concat 2 view rồi backward 1 lần tính là B=1"): cộng 2 loss riêng rồi
  backward 1 lần tương đương toán học với batch 2 ảnh rồi backward 1 lần
  (autograd cộng gradient qua cả 2 đường như nhau), đơn giản hơn nhiều so
  với phải ghép batch thật qua model.data_preprocessor. Nhận thêm
  (step_idx, total_steps) vì AugTrans cần biết tiến độ curriculum
  (Algorithm 1, θ_max(k) phụ thuộc k/K_max) — DI/RRB không dùng, bỏ qua.
"""
from typing import Callable, List, Optional

import torch
from mmdet.structures import DetDataSample

from attack.preprocess import resize_to_model

LossFn = Callable[[torch.nn.Module, torch.Tensor, DetDataSample], torch.Tensor]
# Mỗi view: (ảnh, data_sample) hoặc (ảnh, data_sample, loss_fn riêng của view).
ViewsFn = Callable[[torch.Tensor, DetDataSample, int, int], List[tuple]]


def wp_update(p, p0, gp, rho: float, eta=None):
    """θ' mới của 1 tensor (họ WP). eta = None (WP2): θ0 − rho·‖θ0‖·ĝ (đặt lại từ θ0 mỗi bước, chỉ nhớ 1 bước).
    eta (WPA, progress_log 2026-10-23): Δ ← Δ − eta·‖θ0‖·ĝ rồi chiếu về quả cầu ‖Δ‖ ≤ rho·‖θ0‖ (Δ = θ' − θ0) -> θ' tích lũy
    hướng của nhiều bước, bán kính tổng không vượt WP2."""
    if eta is None:
        return p0 - rho * p0.norm() * gp / (gp.norm() + 1e-12)
    d = (p - p0) - eta * p0.norm() * gp / (gp.norm() + 1e-12)
    return p0 + d * torch.clamp(rho * p0.norm() / (d.norm() + 1e-12), max=1.0)


def run_iterative_attack(
    model: torch.nn.Module,
    orig_pixels: torch.Tensor,
    data_sample: DetDataSample,
    loss_fn: LossFn,
    steps: int,
    epsilon: float,
    alpha: float = 1.0,
    momentum: float = 0.0,
    views_fn: Optional[ViewsFn] = None,
    on_step: Optional[Callable[[int, torch.Tensor], None]] = None,
    grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    backward_per_step: int = 1,
    combine: str = "sum",
    season_sigma: Optional[float] = None,
    dual: Optional[str] = None,
    weight_perturb=None,
    init_noise: Optional[torch.Tensor] = None,
    spike: Optional[float] = None,
) -> torch.Tensor:
    """Trả về noise tensor cuối cùng (cùng shape orig_pixels), đã clamp [-epsilon, epsilon].

    orig_pixels: ảnh GỐC [C,ori_h,ori_w] (AttackDataset `orig_inputs`). δ được tối ưu ở
    CHÍNH không gian này, mỗi step ảnh adv được resize KHẢ VI về khung model
    (data_sample.img_shape) trước khi vào surrogate — ảnh adversarial thật (cỡ gốc,
    uint8, attack.preprocess.to_adv_image) giữ được toàn bộ δ. Bản trước tối ưu δ ở
    khung đã resize (upsample ~1.87x) nên khi đưa về ảnh thật mất ~64% biên độ
    (xem docs/progress_log.md 2026-09-23).

    momentum=0.0  -> IFGSM thuần (không tích lũy động lượng).
    momentum>0.0  -> MI-FGSM (khớp ref: chuẩn hoá gradient bằng mean(abs(.)) trước khi
                     tích lũy động lượng — attack/base/MI.py).
    views_fn      -> DI-FGSM nếu truyền `lambda img, ds: [input_diversity(img)]`, hoặc
                     RRB nếu truyền `attack.methods.rrb.rrb_views` (2 view/step).

    steps PHẢI bằng B (gradient-evaluation budget, docs/protocol_lock.md) — mỗi step
    đúng 1 lần .backward() qua surrogate (qua torch.autograd.grad), không hơn, bất kể
    views_fn sinh ra bao nhiêu view.

    backward_per_step > 1 (chỉ ô mô tả PIA-N20, progress_log 2026-10-04): mỗi step gọi views_fn
    + backward lặp lại bấy nhiêu lần và CỘNG gradient (cụm view không vừa 1 graph) — B thực
    = steps × backward_per_step. Mặc định 1 trùng tuyệt đối bản cũ.

    combine (pilot VC, progress_log 2026-10-07) — cách gộp gradient của backward_per_step lần backward:
    "sum" (mặc định, như cũ); "consensus" = trung bình, chỉ giữ pixel mà MỌI gradient cùng dấu (0 ở chỗ còn lại).

    season_sigma (progress_log 2026-10-17, theo Season arXiv 2608.04441, bỏ mask saliency): thay cập nhật MI bằng
    2 dòng — g_L = Gauss_σ(g) (bán kính 2σ, reflect), g_H = g − g_L; mỗi dòng momentum riêng (chuẩn hoá mean|·|) và δ
    riêng (+α·sign); δ_H chiếu trực giao với δ_L (tích vô hướng toàn ảnh); δ = δ_L + δ_H, mỗi pixel co cả hai theo
    cùng hệ số min(1, ε / |δ_L + δ_H|) để δ ∈ [−ε, ε]. Cần momentum > 0. None = trùng tuyệt đối bản cũ.

    dual (C2-OCT-1V-DM / -DMA, progress_log 2026-10-20): loss của view trả về CẶP (L_feat, L_cls), phần tử có thể None.
    Mỗi số hạng có gradient + momentum riêng (chuẩn hoá mean|·| như MI), hướng cập nhật = m_feat + m_cls -> mỗi kênh có
    phiếu ngang nhau ở từng bước, bất kể độ lớn gradient (thay vì sign(g_feat + λ·g_cls)). "exact": cả 2 số hạng mỗi bước
    (2 lần autograd.grad trên cùng graph); "alt": views_fn chỉ trả 1 số hạng mỗi bước (xen kẽ), 1 grad / bước.

    weight_perturb = (params, rho) (C2-OCT-1V-WP, progress_log 2026-10-21; họ AWT / SAM phía model): trong CÙNG lần
    autograd.grad lấy thêm ∇θ của loss attack; trước bước sau đặt θ' = θ − rho·‖θ_t‖·∇θ_t / ‖∇θ_t‖ (từng tensor) -> surrogate
    "chống lại" δ hiện tại, gradient ảnh bước sau tính trên θ' (trễ 1 bước, không forward thêm). Caller khôi phục θ gốc.
    mode ("awt", β, lr) (E31 AWT-OD): θ tích lũy kiểu SAM 2 bước, +1 forward/backward θ mỗi bước (rho không dùng).

    init_noise (E20 basin test, progress_log 2026-10-24): δ khởi tạo (cùng shape orig_pixels, kẹp [−ε, ε]) thay cho 0;
    momentum vẫn khởi tạo 0. None = trùng tuyệt đối bản cũ.
    """
    assert combine in ("sum", "consensus")
    assert dual in (None, "exact", "alt")
    assert dual is None or (momentum > 0.0 and season_sigma is None and backward_per_step == 1 and grad_transform is None)
    m_dual = [torch.zeros_like(orig_pixels), torch.zeros_like(orig_pixels)]
    model_size = tuple(data_sample.img_shape)
    noise = (torch.zeros_like(orig_pixels) if init_noise is None
             else torch.clamp(init_noise.detach().to(orig_pixels), min=-epsilon, max=epsilon))
    v_mem = torch.zeros_like(orig_pixels)  # E27: điện thế màng LIF (spike)
    accumulated_grad = torch.zeros_like(orig_pixels)
    if season_sigma is not None:
        assert momentum > 0.0
        r = int(round(2 * season_sigma))
        x = torch.arange(-r, r + 1, dtype=orig_pixels.dtype, device=orig_pixels.device)
        k1 = torch.exp(-x ** 2 / (2 * season_sigma ** 2))
        k1 = k1 / k1.sum()
        c = orig_pixels.shape[0]
        kx, ky = k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1), k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

        def lowpass(t):
            t = torch.nn.functional.pad(t[None], (r, r, 0, 0), mode="reflect")
            t = torch.nn.functional.conv2d(t, kx, groups=c)
            t = torch.nn.functional.pad(t, (0, 0, r, r), mode="reflect")
            return torch.nn.functional.conv2d(t, ky, groups=c)[0]

        m_lo, m_hi = torch.zeros_like(orig_pixels), torch.zeros_like(orig_pixels)
        d_lo, d_hi = torch.zeros_like(orig_pixels), torch.zeros_like(orig_pixels)

    for step_idx in range(steps):
        noise = noise.detach().requires_grad_(True)
        grad = None
        gs = []
        for _ in range(backward_per_step):
            adv_orig = torch.clamp(orig_pixels + noise, min=0.0, max=255.0)
            adv_pixels = resize_to_model(adv_orig, model_size)
            views = (views_fn(adv_pixels, data_sample, step_idx, steps)
                    if views_fn is not None else [(adv_pixels, data_sample)])

            if dual is not None:
                parts = [None, None]
                for view in views:
                    for j, t in enumerate(view[2](model, view[0], view[1])):
                        if t is not None:
                            parts[j] = t if parts[j] is None else parts[j] + t
                live = [j for j in (0, 1) if parts[j] is not None]
                gj = torch.autograd.grad([parts[j] for j in live][0], noise, retain_graph=len(live) > 1)[0]
                gl = [gj] + ([torch.autograd.grad(parts[live[1]], noise)[0]] if len(live) > 1 else [])
                for j, g in zip(live, gl):
                    m_dual[j] = momentum * m_dual[j] + g / (g.abs().mean() + 1e-12)
                with torch.no_grad():
                    noise = torch.clamp(noise + alpha * torch.sign(m_dual[0] + m_dual[1]), min=-epsilon, max=epsilon)
                break
            loss = None
            def_loss = None  # E23: thành phần loss surrogate phòng thủ (weight_perturb[5]); None = toàn bộ loss
            for view in views:
                # Phần tử thứ 3 (tùy chọn) = loss riêng của view — pilot KG: view cặp so với
                # feature sạch của CHÍNH vùng crop (attack/methods/kg_pair.py).
                view_img, view_data_sample = view[0], view[1]
                view_loss_fn = view[2] if len(view) > 2 else loss_fn
                term = view_loss_fn(model, view_img, view_data_sample)
                if isinstance(term, tuple):  # (L_feat, L_cls) từ make_view_loss(split=True)
                    dt = term[weight_perturb[5]]
                    def_loss = dt if def_loss is None else def_loss + dt
                    term = term[0] + term[1]
                loss = term if loss is None else loss + term
            if weight_perturb is not None:
                params, rho, base = weight_perturb[:3]
                eta = weight_perturb[3] if len(weight_perturb) > 3 else None
                # mode (E22, progress_log 2026-10-24): "def" (mặc định, WP2) | "asc" (đổi dấu: θ0 + ρ‖θ0‖ĝ) |
                # ("rand", gen): hướng Gauss ngẫu nhiên chuẩn hóa từng tensor, cùng bán kính, không cần ∇θ.
                mode = weight_perturb[4] if len(weight_perturb) > 4 else "def"
                if isinstance(mode, tuple) and mode[0] == "ens":
                    # E25 (progress_log 2026-10-24): K bản θ' tỏa quanh hướng phòng thủ ĝ_θ (từ θ0 bước này);
                    # ∇ₓL qua từng bản (feature sạch co-transform cũng trên θ'_k), trung bình -> cập nhật δ.
                    _, K, spread, gen = mode
                    gall = torch.autograd.grad(loss, [noise] + params)
                    with torch.no_grad():
                        ghat = [gp / (gp.norm() + 1e-12) for gp in gall[1:]]
                    gacc = torch.zeros_like(gall[0])
                    for _k in range(K):
                        with torch.no_grad():
                            for p, p0, gh in zip(params, base, ghat):
                                xi = torch.randn(p0.shape, generator=gen, device=p0.device, dtype=p0.dtype)
                                d = gh + spread * xi / (xi.norm() + 1e-12)
                                p.copy_(p0 - rho * p0.norm() * d / (d.norm() + 1e-12))
                        adv_k = resize_to_model(torch.clamp(orig_pixels + noise, 0.0, 255.0), model_size)
                        vk = views_fn(adv_k, data_sample, step_idx, steps)
                        lk = None
                        for view in vk:
                            t = (view[2] if len(view) > 2 else loss_fn)(model, view[0], view[1])
                            t = t[0] + t[1] if isinstance(t, tuple) else t
                            lk = t if lk is None else lk + t
                        gacc = gacc + torch.autograd.grad(lk, noise)[0]
                    with torch.no_grad():
                        for p, p0 in zip(params, base):
                            p.copy_(p0)
                    g = gacc / K
                elif isinstance(mode, tuple) and mode[0] == "awt":
                    # E31 AWT-OD (progress_log 2026-10-24, arXiv 2408.09469): θ_t TÍCH LŨY (không đặt lại từ θ0). ∇ₓ, ∇θ cùng
                    # 1 lần grad tại θ_t; θ̂ = θ_t + β‖θ0‖ĝθ (ascend, per-tensor); +1 forward/backward CÙNG view (ảnh detach,
                    # feature sạch giữ từ θ_t, KHÔNG rút view mới) tại θ̂; θ_{t+1} = θ_t − lr‖θ0‖·ĝθ(θ̂). Caller khôi phục θ0.
                    _, beta, lr = mode
                    gall = torch.autograd.grad(loss, [noise] + params)
                    g = gall[0]
                    with torch.no_grad():
                        cur = [p.detach().clone() for p in params]
                        for p, p0, gp in zip(params, base, gall[1:]):
                            p.add_(beta * p0.norm() * gp / (gp.norm() + 1e-12))
                    lh = None
                    for view in views:
                        t = (view[2] if len(view) > 2 else loss_fn)(model, view[0].detach(), view[1])
                        t = t[0] + t[1] if isinstance(t, tuple) else t
                        lh = t if lh is None else lh + t
                    gh = torch.autograd.grad(lh, params)
                    with torch.no_grad():
                        for p, c, p0, gp in zip(params, cur, base, gh):
                            p.copy_(c - lr * p0.norm() * gp / (gp.norm() + 1e-12))
                elif isinstance(mode, tuple) and mode[0] == "wspike":
                    # E28 (progress_log 2026-10-24): LIF trên TRỌNG SỐ — M tích lũy ∇θ chuẩn hóa per-tensor; element
                    # |M| ≥ θ_fire phát xung (nhiễu phòng thủ θ'= θ0 − ρ‖θ0‖·d/‖d‖, d = sign(M)·fire), reset M = 0.
                    _, theta_fire, mem = mode
                    gall = torch.autograd.grad(loss, [noise] + params)
                    g = gall[0]
                    if not mem:
                        mem.extend(torch.zeros_like(p) for p in params)
                    with torch.no_grad():
                        for i, (p, p0, gp) in enumerate(zip(params, base, gall[1:])):
                            mem[i].add_(gp / (gp.abs().mean() + 1e-12))
                            fire = mem[i].abs() >= theta_fire
                            d = torch.sign(mem[i]) * fire
                            dn = d.norm()
                            if float(dn) > 0:
                                p.copy_(p0 - rho * p0.norm() * d / (dn + 1e-12))
                            else:
                                p.copy_(p0)
                            mem[i].mul_(~fire)
                elif isinstance(mode, tuple):
                    g = torch.autograd.grad(loss, noise)[0]
                    with torch.no_grad():
                        for p, p0 in zip(params, base):
                            r = torch.randn(p0.shape, generator=mode[1], device=p0.device, dtype=p0.dtype)
                            p.copy_(p0 + rho * p0.norm() * r / (r.norm() + 1e-12))
                else:
                    if def_loss is not None:  # E23: attacker theo toàn bộ loss, defender chống 1 thành phần
                        g = torch.autograd.grad(loss, noise, retain_graph=True)[0]
                        gall = (g,) + torch.autograd.grad(def_loss, params)
                    else:
                        gall = torch.autograd.grad(loss, [noise] + params)
                    g = gall[0]
                    with torch.no_grad():
                        for p, p0, gp in zip(params, base, gall[1:]):
                            p.copy_(wp_update(p, p0, -gp if mode == "asc" else gp, rho, eta))
            else:
                g = torch.autograd.grad(loss, noise)[0]
            grad = g if grad is None else grad + g
            if combine == "consensus":
                gs.append(g)
        if dual is not None:
            if on_step is not None:
                on_step(step_idx + 1, noise.detach())
            continue
        if combine == "consensus":
            s = torch.stack([torch.sign(x) for x in gs])
            agree = (s == s[0]).all(0) & (s[0] != 0)
            grad = grad / len(gs) * agree
        if grad_transform is not None:  # vd lọc thông thấp gradient (nhánh S1, progress_log 2026-10-03)
            grad = grad_transform(grad)

        if season_sigma is not None:
            with torch.no_grad():
                g_lo = lowpass(grad.float())
                g_hi = grad.float() - g_lo
                m_lo = momentum * m_lo + g_lo / (g_lo.abs().mean() + 1e-12)
                m_hi = momentum * m_hi + g_hi / (g_hi.abs().mean() + 1e-12)
                d_lo = d_lo + alpha * torch.sign(m_lo)
                d_hi = d_hi + alpha * torch.sign(m_hi)
                d_hi = d_hi - (d_hi * d_lo).sum() / ((d_lo * d_lo).sum() + 1e-12) * d_lo
                s = torch.clamp(epsilon / ((d_lo + d_hi).abs() + 1e-12), max=1.0)
                d_lo, d_hi = d_lo * s, d_hi * s
                noise = torch.clamp(d_lo + d_hi, min=-epsilon, max=epsilon)
            if on_step is not None:
                on_step(step_idx + 1, noise.detach())
            continue

        if spike is not None:
            # E27 (progress_log 2026-10-24): LIF trên δ — V tích lũy gradient chuẩn hóa; pixel |V| ≥ θ_fire phát xung
            # (δ += α·sign(V)), reset V = 0; pixel dưới ngưỡng giữ V, δ không đổi.
            with torch.no_grad():
                g = grad / (grad.abs().mean() + 1e-12)
                v_mem = momentum * v_mem + g
                fire = v_mem.abs() >= spike
                noise = torch.clamp(noise + alpha * torch.sign(v_mem) * fire, min=-epsilon, max=epsilon)
                v_mem = v_mem * (~fire)
            if on_step is not None:
                on_step(step_idx + 1, noise.detach())
            continue

        if momentum > 0.0:
            grad = grad / (grad.abs().mean() + 1e-12)
            accumulated_grad = momentum * accumulated_grad + grad
            update_direction = accumulated_grad
        else:
            update_direction = grad

        with torch.no_grad():
            noise = noise + alpha * torch.sign(update_direction)
            noise = torch.clamp(noise, min=-epsilon, max=epsilon)
        if on_step is not None:
            # (số backward đã dùng, noise hiện tại) — noise sau k step đúng bằng kết quả
            # của lần chạy steps=k (không gì trong vòng lặp phụ thuộc `steps`, trừ
            # views_fn của AugTrans), nên 1 lần chạy B lớn cho cả đường cong theo B.
            on_step(step_idx + 1, noise.detach())

    return noise.detach()
