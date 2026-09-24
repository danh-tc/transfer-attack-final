#!/usr/bin/env python3
"""Áp quy tắc GO @ n300 của pilot A′1 (khóa trong docs/progress_log.md 2026-09-24).

So OSFD-W1 với OSFD của run `n300_B50_eps5` (cùng PNG OSFD, cùng seed ảnh, paired
bootstrap 1000 mẫu BOOT_SEED 2026). GO nếu đồng thời:
- ΔCrossAvg(ConvNeXt-T, Swin-T) >= 3 VÀ cận dưới CI 95% > 0;
- Δ_ConvNeXt-T > 0 VÀ Δ_Swin-T > 0 (điểm);
- Δ_YOLOX-S >= 5 VÀ cận dưới CI 95% > 0.
DINO-Swin-L, R101: chỉ báo cáo.

Chạy: python scripts/pilot_a1_go.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO_ROOT, "results/runs/n300_B50_eps5")
CAND, REF = "OSFD-W1", "OSFD"


def main():
    with open(os.path.join(RUN_DIR, "metrics_pilot_a1.json")) as f:
        ctrl = json.load(f)
    with open(os.path.join(RUN_DIR, "generalization_pilot_a1.json")) as f:
        gen = json.load(f)
    drop = lambda m, k: (ctrl["relative_ap_drop"][m].get(k) or gen["relative_ap_drop"][m][k])["point"]

    # run_baselines ghi hiệu a − b với a < b theo thứ tự chữ: "OSFD - OSFD-W1" -> đổi dấu.
    cross = ctrl["method_diff_cross_avg"][f"{REF} - {CAND}"]
    d_cross = {"point": -cross["point"], "ci95": [-cross["ci95"][1], -cross["ci95"][0]]}
    d_yolox = gen["method_diff"][f"{CAND} - {REF}"]["yolox_s"]
    d_dino = gen["method_diff"][f"{CAND} - {REF}"]["dino_swin_l"]
    d = {k: drop(CAND, k) - drop(REF, k) for k in ["r50", "r101", "convnext_t", "swin_t"]}

    checks = {
        "cross_avg>=3": d_cross["point"] >= 3,
        "cross_avg_ci_lo>0": d_cross["ci95"][0] > 0,
        "convnext_t>0": d["convnext_t"] > 0,
        "swin_t>0": d["swin_t"] > 0,
        "yolox_s>=5": d_yolox["point"] >= 5,
        "yolox_s_ci_lo>0": d_yolox["ci95"][0] > 0,
    }
    go = all(checks.values())
    fmt = lambda s: f"{s['point']:+.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    print(f"{CAND} − {REF} @ n300 (relative AP drop, điểm %)")
    print("  " + "  ".join(f"{k} {v:+.1f}" for k, v in d.items()))
    print(f"  ΔCrossAvg {fmt(d_cross)}   YOLOX-S {fmt(d_yolox)}   DINO-Swin-L (báo cáo) {fmt(d_dino)}")
    print("  " + "  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()))
    print(f"=> {'GO' if go else 'NO-GO'} A′1")
    out = {"rule": "progress_log 2026-09-24", "delta_point": d, "delta_cross_avg": d_cross,
           "delta_yolox_s": d_yolox, "delta_dino_swin_l": d_dino, "checks": checks, "go": go}
    with open(os.path.join(RUN_DIR, "pilot_a1_go.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
