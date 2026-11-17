"""S9 (progress_log 2026-11-17, KHOA): ban do dau co QUY LUAT CHUNG an khong — dong thuan 5 WB + suy tu noi dung anh.
CPU, trong mask (hop mask GT khong-crowd), pixel-kenh, delta 5 WB seed + RD-ALL (transfer) cache S1/S6, anh sach.
[1] dong thuan: |mean_5(sign)| ; ty le >=4/5 cung dau (ngau nhien 5-chieu: P(>=4)=0.375).
[2] quy luat noi dung (sign suy tu anh sach): P_mean = sign(trung binh lan can 5x5 - gia tri) [day VE trung binh];
    P_lap = sign(Laplacian); P_gmean = sign(trung binh kenh toan anh - gia tri). Khop = dau dong thuan == P.
[3] muot khong gian: ty le cap pixel-kenh ke nhau (ngang) cung dau (ngau nhien 0.5).
Bao cho WB (dong thuan) va RD-ALL (transfer co theo quy luat chua).
  python scripts/s9_signlaw.py
"""
import json
import os
import sys

import mmcv
import numpy as np
from pycocotools.coco import COCO
from scipy.ndimage import uniform_filter, laplace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from scripts.s3_region import ADV, ANN, IMGDIR

WBS = ["WB-0", "WB-r1", "WB-r2", "WB-r3", "WB-r4"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN)
    # gom pixel-kenh trong mask qua moi anh
    cons_abs = []          # |mean sign| tung pixel-kenh (do manh dong thuan)
    ge4 = []               # 1 neu >=4/5 cung dau
    cons_sign = []         # dau dong thuan (lam tron, bo 0)
    Pm, Pl, Pg = [], [], []  # predictor signs
    rd_sign = []           # dau RD-ALL
    smooth = {c: [0, 0] for c in WBS + ["RD-ALL"]}  # [cung dau, tong] cap ngang trong mask
    for i in ids:
        img = mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.float32)  # HWC BGR
        h, w = img.shape[:2]; m = np.zeros((h, w), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            m |= coco.annToMask(a).astype(bool)[:h, :w]
        m3 = np.broadcast_to(m[:, :, None], img.shape)
        # du doan tu noi dung (per channel)
        locmean = np.stack([uniform_filter(img[:, :, k], 5) for k in range(3)], 2)
        lap = np.stack([laplace(img[:, :, k]) for k in range(3)], 2)
        gmean = img.mean((0, 1), keepdims=True)
        Pm.append(np.sign(locmean - img)[m3]); Pl.append(np.sign(lap)[m3])
        Pg.append(np.sign(np.broadcast_to(gmean, img.shape) - img)[m3])
        # dau 5 WB
        cl = mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32)
        S = []
        for c in WBS + ["RD-ALL"]:
            d = mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - cl
            sd = np.sign(d)
            if c in WBS:
                S.append(sd)
            else:
                rd_sign.append(sd[m3])
            # muot: cap ngang trong mask (ca 2 pixel trong mask)
            a = sd[:, :-1, :]; b = sd[:, 1:, :]; mm = m[:, :-1][:, :, None] & m[:, 1:][:, :, None]
            mm = np.broadcast_to(mm, a.shape)
            nz = (a != 0) & (b != 0) & mm
            smooth[c][0] += int((a[nz] == b[nz]).sum()); smooth[c][1] += int(nz.sum())
        St = np.stack(S)  # [5,H,W,3]
        ms = St.mean(0)   # mean sign
        cons_abs.append(np.abs(ms)[m3])
        ge4.append((np.abs(St.sum(0)) >= 4)[m3].astype(float))
        cons_sign.append(np.sign(ms)[m3])
    cons_abs = np.concatenate(cons_abs); ge4 = np.concatenate(ge4); cons_sign = np.concatenate(cons_sign)
    Pm = np.concatenate(Pm); Pl = np.concatenate(Pl); Pg = np.concatenate(Pg); rd_sign = np.concatenate(rd_sign)

    res = {"n_img": len(ids), "n_pixelchan_mask": int(len(cons_abs))}
    res["consensus"] = {"mean_abs_meansign": float(cons_abs.mean()), "frac_ge4of5": float(ge4.mean()),
                        "random_ge4of5": 0.375}
    print(f"[1] dong thuan 5 WB trong mask ({len(cons_abs)} pixel-kenh):")
    print(f"    |mean sign| TB {cons_abs.mean():.3f} (0=ngau nhien, 1=nhat tri)")
    print(f"    >=4/5 cung dau {ge4.mean():.3f} (ngau nhien 0.375)")

    # khop quy luat: tren pixel-kenh co dong thuan manh (>=4/5)
    strong = ge4 > 0.5
    res["content_law"] = {}
    print(f"\n[2] khop dau DONG THUAN vs quy luat noi dung (tren {int(strong.sum())} pixel-kenh >=4/5):")
    for nm, P in (("push_toward_localmean", Pm), ("laplacian", Pl), ("push_toward_globalmean", Pg)):
        nz = strong & (cons_sign != 0) & (P != 0)
        ag = float((cons_sign[nz] == P[nz]).mean())
        res["content_law"][nm] = ag
        print(f"    {nm:24} {ag:.3f}")
    # RD-ALL co theo quy luat (hoac theo dong thuan WB) chua
    print(f"\n    RD-ALL (transfer) vs cung quy luat, tren pixel-kenh >=4/5:")
    res["rd_vs_law"] = {}
    for nm, P in (("push_toward_localmean", Pm), ("laplacian", Pl), ("consensusWB", cons_sign)):
        nz = strong & (rd_sign != 0) & (P != 0)
        ag = float((rd_sign[nz] == P[nz]).mean())
        res["rd_vs_law"][nm] = ag
        print(f"    {nm:24} {ag:.3f}")

    print(f"\n[3] muot khong gian (cap pixel-kenh ke ngang cung dau, ngau nhien 0.5):")
    res["spatial_smooth"] = {}
    for c in WBS + ["RD-ALL"]:
        v = smooth[c][0] / max(smooth[c][1], 1)
        res["spatial_smooth"][c] = v
        print(f"    {c:8} {v:.3f}")

    law = res["content_law"]
    best = max(law, key=law.get)
    v1 = (f"CO quy luat noi dung: {best} khop {law[best]:.2f}" if law[best] >= 0.65 else
          f"quy luat noi dung YEU (max {best} {law[best]:.2f})" if law[best] >= 0.55 else
          "KHONG quy luat noi dung (dau ~doc lap voi noi dung)")
    sm = np.mean([res["spatial_smooth"][c] for c in WBS])
    v3 = "dau CO cau truc khong gian (muot)" if sm >= 0.6 else "dau gan nhieu trang khong gian"
    res["verdict"] = {"consensus_above_random": bool(res["consensus"]["frac_ge4of5"] > 0.45),
                      "content_law": v1, "spatial": v3}
    print(f"\n=> dong thuan {'>' if res['consensus']['frac_ge4of5']>0.45 else '~'} ngau nhien")
    print(f"=> {v1}\n=> {v3} (WB muot TB {sm:.2f})")
    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s9_signlaw_dev30.json"), "w"),
              indent=1, ensure_ascii=False, default=float)
    print("\n[s9] -> results/mechanism/s9_signlaw_dev30.json")


if __name__ == "__main__":
    main()
