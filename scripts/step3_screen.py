#!/usr/bin/env python3
"""Áp tiêu chí sàng lọc bước 3 (docs/plan_sota.md, LOCKED) lên results/ablation/<tag>/screen.json
(do ablation_screen.py sinh, Δ = OSFD − ô) -> đổi dấu thành ứng viên − OSFD.

Chạy: python scripts/step3_screen.py --tag step3_dev300 --cands ABL-task-loss PROBE-osfd+cls
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flip(s):
    return {"point": -s["point"], "ci95": [-s["ci95"][1], -s["ci95"][0]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--cands", nargs="+", required=True)
    args = p.parse_args()
    path = os.path.join(REPO_ROOT, "results/ablation", args.tag, "screen.json")
    scr = json.load(open(path))
    out = {}
    for c in args.cands:
        d = {k: flip(v) for k, v in scr["delta_vs_osfd"][c].items()}
        checks = {"cross_avg3>=3": d["cross_avg3"]["point"] >= 3,
                  "cross_avg3_ci_lo>0": d["cross_avg3"]["ci95"][0] > 0,
                  "convnext_t>0": d["convnext_t"]["point"] > 0, "swin_t>0": d["swin_t"]["point"] > 0,
                  "yolox_s>0": d["yolox_s"]["point"] > 0, "r101>=-3": d["r101"]["point"] >= -3}
        out[c] = {"delta_cand_minus_osfd": d, "checks": checks, "pass": all(checks.values())}
        f = lambda s: f"{s['point']:+.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]"
        print(f"{c}: Cross3 {f(d['cross_avg3'])}  ConvNeXt {f(d['convnext_t'])}  Swin {f(d['swin_t'])}  "
              f"YOLOX {f(d['yolox_s'])}  R101 {f(d['r101'])}  DINO-SwinL {f(d['dino_swin_l'])}")
        print("   " + "  ".join(f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()) + f"  -> {'QUA' if out[c]['pass'] else 'KHÔNG QUA'}")
    json.dump(out, open(os.path.join(REPO_ROOT, "results/ablation", args.tag, "step3.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
