#!/usr/bin/env python3
"""Áp tiêu chí GO @ n300 (docs/plan_sota.md bước 4, LOCKED) lên results/ablation/<tag>/screen.json
(ablation_screen.py, Δ = OSFD − ô -> đổi dấu thành ứng viên − OSFD). GO nếu đồng thời:
ΔCrossAvg₃ >= 3 VÀ cận dưới CI > 0; Δ > 0 ở cả ConvNeXt-T, Swin-T, YOLOX-S VÀ >= 2/3 target >= 3;
drop R101 >= R101(OSFD) − 3.

Chạy: python scripts/go_check.py --tag go_c2_n300 --cand PROBE-osfd+cls
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS3 = ["convnext_t", "swin_t", "yolox_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--cand", required=True)
    a = p.parse_args()
    d0 = json.load(open(os.path.join(REPO_ROOT, "results/ablation", a.tag, "screen.json")))["delta_vs_osfd"][a.cand]
    d = {k: {"point": -v["point"], "ci95": [-v["ci95"][1], -v["ci95"][0]]} for k, v in d0.items()}
    checks = {
        "G1_cross_avg3>=3": d["cross_avg3"]["point"] >= 3,
        "G1_cross_avg3_ci_lo>0": d["cross_avg3"]["ci95"][0] > 0,
        "G3_all_cross>0": all(d[k]["point"] > 0 for k in CROSS3),
        "G3_>=2of3_cross>=3": sum(d[k]["point"] >= 3 for k in CROSS3) >= 2,
        "G4_r101>=-3": d["r101"]["point"] >= -3,
    }
    go = all(checks.values())
    f = lambda s: f"{s['point']:+.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
    print(f"{a.cand} − OSFD: " + "  ".join(f"{k} {f(d[k])}" for k in ["cross_avg3"] + CROSS3 + ["r101", "dino_swin_l"]))
    print("  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()) + f"\n=> {'GO' if go else 'NO-GO'}")
    json.dump({"rule": "docs/plan_sota.md bước 4 (LOCKED)", "cand": a.cand, "delta_cand_minus_osfd": d,
               "checks": checks, "go": go},
              open(os.path.join(REPO_ROOT, "results/ablation", a.tag, "go.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
