#!/usr/bin/env python3
"""Áp quy tắc SÀNG LỌC pilot A′1 @ dev100 (khóa trong docs/progress_log.md 2026-09-24).

Δ = relative AP drop(candidate) − drop(OSFD), điểm ước lượng, cùng ảnh/seed.
Qua sàng lọc nếu: ΔCrossAvg(ConvNeXt-T, Swin-T) >= 2 VÀ Δ_ConvNeXt-T >= 0 VÀ Δ_Swin-T >= 0
VÀ Δ_YOLOX-S >= 0. Cả 2 qua -> chọn ΔCrossAvg lớn hơn (chênh < 0.5 -> W1).
R50, R101, DINO-Swin-L chỉ báo cáo.

Chạy: python scripts/pilot_a1_screen.py [--run dev100_B50_eps5]
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = ["OSFD-W1", "OSFD-W2"]
REF = "OSFD"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="dev100_B50_eps5")
    args = p.parse_args()
    run_dir = os.path.join(REPO_ROOT, "results/runs", args.run)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        ctrl = json.load(f)["relative_ap_drop"]
    with open(os.path.join(run_dir, "generalization_metrics.json")) as f:
        gen = json.load(f)["relative_ap_drop"]
    drop = lambda m, k: (ctrl[m][k] if k in ctrl[m] else gen[m][k])["point"]

    models = ["r50", "r101", "convnext_t", "swin_t", "yolox_s", "dino_swin_l"]
    print(f"relative AP drop (điểm), {args.run}")
    print(f"{'':<10}" + "".join(f"{k:>13}" for k in models))
    for m in [REF] + CANDIDATES:
        print(f"{m:<10}" + "".join(f"{drop(m, k):13.1f}" for k in models))

    out = {"run": args.run, "rule": "progress_log 2026-09-24", "candidates": {}}
    for c in CANDIDATES:
        d = {k: drop(c, k) - drop(REF, k) for k in models}
        d_cross = (d["convnext_t"] + d["swin_t"]) / 2
        checks = {"cross_avg>=2": d_cross >= 2, "convnext_t>=0": d["convnext_t"] >= 0,
                  "swin_t>=0": d["swin_t"] >= 0, "yolox_s>=0": d["yolox_s"] >= 0}
        out["candidates"][c] = {"delta": d, "delta_cross_avg": d_cross, "checks": checks,
                                "pass": all(checks.values())}
        print(f"\n{c} − OSFD: ΔCrossAvg {d_cross:+.1f} | " +
              " ".join(f"{k} {v:+.1f}" for k, v in d.items()))
        print("  " + "  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()) +
              f"  -> {'QUA' if all(checks.values()) else 'KHÔNG QUA'}")

    passed = [c for c in CANDIDATES if out["candidates"][c]["pass"]]
    if not passed:
        choice = None
    elif len(passed) == 1:
        choice = passed[0]
    else:
        a, b = (out["candidates"][c]["delta_cross_avg"] for c in CANDIDATES)
        choice = "OSFD-W1" if abs(a - b) < 0.5 else max(passed, key=lambda c: out["candidates"][c]["delta_cross_avg"])
    out["selected_for_n300"] = choice
    print(f"\n=> {'Sang n300 với ' + choice if choice else 'DỪNG A′1 (không cấu hình nào qua sàng lọc)'}")
    with open(os.path.join(run_dir, "pilot_a1_screen.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
