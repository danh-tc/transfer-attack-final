"""S7 (progress_log 2026-11-17, KHOA): dac tinh noi dung anh cua nhieu HIEU QUA trong mask — 3 truc, dev30.
TRONG MASK (hop mask GT khong-crowd), pixel-kenh (A/clip/intensity) hoac pixel (flatness/channel voi g gop kenh).
g_t = |grad cls DINO-Swin-L| cache S3/S6 (clean cho mat do; t in {0.5,1} cho A). delta BGR tu PNG.
Truc A (gia tri pixel sach 0-255): bin VDARK<16 / DARK 16-64 / MID 64-192 / BRIGHT 192-240 / VBRIGHT>240.
  mat do |g_t| (phan ‖g‖1 / phan so pixel-kenh), A, ty le CLIP (|delta thuc| < eps-0.5 du da bao hoa dau).
Truc B (do phang cuc bo): std luminance cua sach trong cua so 5x5; tercile theo pixel-mask moi anh -> FLAT/MIDF/TEXT.
  mat do |g_t| (gop 3 kenh), A, + hanh vi keep_TEXT / keep_FLAT.
Truc C (kenh mau B/G/R): phan ‖g‖1 moi kenh, A moi kenh, + hanh vi keep_B/G/R (chi nhieu 1 kenh).
Outcome hanh vi: kill object clean-TP DINO-Swin-L (7 delta: WB-0/r1/r2, RD-ALL/DET/FEAT, OSFD-ALL).
  python scripts/s7_pixel.py [--behav 1]
"""
import argparse
import json
import os
import sys

import mmcv
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy.ndimage import uniform_filter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attack.data import load_image_ids
from attack.evaluation import to_coco_dets
from scripts.e64_cnx_drop import load
from scripts.e77_dfc import fate, group
from scripts.s3_region import ADV, ANN, GDIR, IMGDIR
from scripts.s5_dev30 import match_score

CELLS = ["WB-0", "WB-r1", "WB-r2", "RD-ALL", "RD-DET", "RD-FEAT", "OSFD-ALL"]
TS = ("0.5", "1.0")
EPS = 5.0
IBINS = [("VDARK", 0, 16), ("DARK", 16, 64), ("MID", 64, 192), ("BRIGHT", 192, 240), ("VBRIGHT", 240, 256)]
CHAN = ["B", "G", "R"]
SEED = 2026


def lumin(img):  # BGR -> luminance
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def local_std(img):
    l = lumin(img).astype(np.float32)
    m = uniform_filter(l, 5); m2 = uniform_filter(l * l, 5)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def boot_ratio(num_by_img, den_by_img, rng, n=1000):
    """CI cho ty so sum(num)/sum(den), bootstrap theo anh."""
    keys = list(num_by_img); num = np.array([num_by_img[k] for k in keys]); den = np.array([den_by_img[k] for k in keys])
    pt = num.sum() / den.sum() if den.sum() else float("nan")
    bs = []
    for _ in range(n):
        s = rng.integers(0, len(keys), len(keys))
        d = den[s].sum(); bs.append(num[s].sum() / d if d else np.nan)
    return float(pt), [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--behav", type=int, default=1); args = ap.parse_args()
    ids = load_image_ids(os.path.join(REPO_ROOT, "data/image_lists/dev100.csv"))[:30]
    coco = COCO(ANN); cat = sorted(coco.getCatIds())
    clean = {i: mmcv.imread(os.path.join(IMGDIR, coco.loadImgs(i)[0]["file_name"])).astype(np.int32) for i in ids}
    MK, FL = {}, {}
    for i in ids:
        h, w = clean[i].shape[:2]; m = np.zeros((h, w), bool)
        for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False)):
            m |= coco.annToMask(a).astype(bool)[:h, :w]
        MK[i] = m
        s = local_std(clean[i])
        sm = s[m]
        if sm.size:
            q1, q2 = np.percentile(sm, [33.3, 66.7])
            FL[i] = np.where(s <= q1, 0, np.where(s <= q2, 1, 2))  # 0 FLAT,1 MIDF,2 TEXT
        else:
            FL[i] = np.zeros((h, w), int)
    rng = np.random.default_rng(SEED)
    res = {"n_img": len(ids)}

    # ====== DIAGNOSTIC (cache g) ======
    gclip = {c: {} for c in CELLS}
    # tich luy theo anh cho CI
    accA = {t: {c: {k[0]: {"num": {}, "den": {}} for k in IBINS} for c in CELLS} for t in ("I",)}
    gden_I = {k[0]: {"mass": {}, "cnt": {}} for k in IBINS}
    gden_F = {f: {"mass": {}, "cnt": {}} for f in ("FLAT", "MIDF", "TEXT")}
    accA_F = {c: {f: {"num": {}, "den": {}} for f in ("FLAT", "MIDF", "TEXT")} for c in CELLS}
    gmass_C = {ch: {} for ch in CHAN}; accA_C = {c: {ch: {"num": {}, "den": {}} for ch in CHAN} for c in CELLS}
    clip_acc = {c: {k[0]: {"clip": {}, "cnt": {}} for k in IBINS} for c in CELLS}

    for i in ids:
        m = MK[i]; m3 = np.broadcast_to(m[:, :, None], clean[i].shape)
        gcl = np.abs(np.load(os.path.join(GDIR, f"{i}_clean.npy")).astype(np.float32)).transpose(1, 2, 0)  # HWC
        cvals = clean[i]  # HWC BGR 0-255
        ib = np.full(cvals.shape, -1, int)
        for bi, (nm, lo, hi) in enumerate(IBINS):
            ib[(cvals >= lo) & (cvals < hi)] = bi
        fl = FL[i]
        # mat do |g| theo intensity bin (pixel-kenh trong mask)
        for bi, (nm, lo, hi) in enumerate(IBINS):
            sel = m3 & (ib == bi)
            gden_I[nm]["mass"][i] = float(gcl[sel].sum()); gden_I[nm]["cnt"][i] = int(sel.sum())
        # mat do theo flatness (gop kenh: dung g pixel = sum over channel)
        gpix = gcl.sum(2)
        for fi, f in enumerate(("FLAT", "MIDF", "TEXT")):
            sel = m & (fl == fi)
            gden_F[f]["mass"][i] = float(gpix[sel].sum()); gden_F[f]["cnt"][i] = int(sel.sum())
        # mat do theo kenh
        for ci, ch in enumerate(CHAN):
            gmass_C[ch][i] = float(gcl[:, :, ci][m].sum())
        # A theo truc (dung sign delta × g tai t, TB)
        for c in CELLS:
            d = mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]
            sd = np.sign(d)
            realized = np.abs(d)
            gt = np.mean([np.load(os.path.join(GDIR, f"{i}_{c}_t{t}.npy")).astype(np.float32).transpose(1, 2, 0)
                          for t in TS], 0)
            sg = sd * gt; ag = np.abs(gt)
            for bi, (nm, lo, hi) in enumerate(IBINS):
                sel = m3 & (ib == bi)
                accA["I"][c][nm]["num"][i] = float(sg[sel].sum()); accA["I"][c][nm]["den"][i] = float(ag[sel].sum())
                # clip: pixel bao hoa dau nhung |realized| < eps-0.5
                clip_acc[c][nm]["clip"][i] = int(((realized < EPS - 0.5) & sel).sum())
                clip_acc[c][nm]["cnt"][i] = int(sel.sum())
            for fi, f in enumerate(("FLAT", "MIDF", "TEXT")):
                sel = np.broadcast_to((m & (fl == fi))[:, :, None], d.shape)
                accA_F[c][f]["num"][i] = float(sg[sel].sum()); accA_F[c][f]["den"][i] = float(ag[sel].sum())
            for ci, ch in enumerate(CHAN):
                sel = m
                accA_C[c][ch]["num"][i] = float(sg[:, :, ci][sel].sum()); accA_C[c][ch]["den"][i] = float(ag[:, :, ci][sel].sum())

    # tong hop
    tot_mass_I = sum(sum(gden_I[k[0]]["mass"].values()) for k in IBINS)
    tot_cnt_I = sum(sum(gden_I[k[0]]["cnt"].values()) for k in IBINS)
    res["A_intensity"] = {"density": {}, "A_RD-ALL": {}, "clip_RD-ALL": {}, "A_WB-0": {}}
    print("\n[A] theo GIA TRI pixel sach (trong mask, pixel-kenh):")
    print(f"{'bin':8}{'%pixel':>8}{'mat do|g|':>10}{'A RD-ALL':>10}{'A WB-0':>9}{'clip% RD':>10}")
    for k in IBINS:
        nm = k[0]
        mass = sum(gden_I[nm]["mass"].values()); cnt = sum(gden_I[nm]["cnt"].values())
        dens = (mass / tot_mass_I) / (cnt / tot_cnt_I) if cnt else float("nan")
        aR, _ = boot_ratio(accA["I"]["RD-ALL"][nm]["num"], accA["I"]["RD-ALL"][nm]["den"], rng)
        aW, _ = boot_ratio(accA["I"]["WB-0"][nm]["num"], accA["I"]["WB-0"][nm]["den"], rng)
        clipn = sum(clip_acc["RD-ALL"][nm]["clip"].values()); clipc = sum(clip_acc["RD-ALL"][nm]["cnt"].values())
        clip = 100 * clipn / clipc if clipc else float("nan")
        res["A_intensity"]["density"][nm] = dens; res["A_intensity"]["A_RD-ALL"][nm] = aR
        res["A_intensity"]["A_WB-0"][nm] = aW; res["A_intensity"]["clip_RD-ALL"][nm] = clip
        print(f"{nm:8}{100*cnt/tot_cnt_I:8.1f}{dens:10.2f}{aR:10.4f}{aW:9.4f}{clip:10.0f}")

    tot_mF = sum(sum(gden_F[f]["mass"].values()) for f in gden_F); tot_cF = sum(sum(gden_F[f]["cnt"].values()) for f in gden_F)
    res["B_flatness"] = {"density": {}, "A_RD-ALL": {}, "A_WB-0": {}}
    print("\n[B] theo DO PHANG cuc bo (trong mask, pixel gop kenh):")
    print(f"{'bin':8}{'%pixel':>8}{'mat do|g|':>10}{'A RD-ALL':>10}{'A WB-0':>9}")
    for f in ("FLAT", "MIDF", "TEXT"):
        mass = sum(gden_F[f]["mass"].values()); cnt = sum(gden_F[f]["cnt"].values())
        dens, dci = boot_ratio({i: gden_F[f]["mass"][i] / max(tot_mass_I, 1) for i in ids},
                               {i: gden_F[f]["cnt"][i] / max(tot_cnt_I, 1) for i in ids}, rng)
        dens = (mass / tot_mF) / (cnt / tot_cF) if cnt else float("nan")
        aR, aRci = boot_ratio(accA_F["RD-ALL"][f]["num"], accA_F["RD-ALL"][f]["den"], rng)
        aW, _ = boot_ratio(accA_F["WB-0"][f]["num"], accA_F["WB-0"][f]["den"], rng)
        res["B_flatness"]["density"][f] = dens; res["B_flatness"]["A_RD-ALL"][f] = {"pt": aR, "ci95": aRci}
        res["B_flatness"]["A_WB-0"][f] = aW
        print(f"{f:8}{100*cnt/tot_cF:8.1f}{dens:10.2f}{aR:10.4f}{aW:9.4f}")

    tot_mC = sum(sum(gmass_C[ch].values()) for ch in CHAN)
    res["C_channel"] = {"mass": {}, "A_RD-ALL": {}, "A_WB-0": {}}
    print("\n[C] theo KENH MAU (trong mask):")
    print(f"{'kenh':8}{'%‖g‖1':>8}{'A RD-ALL':>10}{'A WB-0':>9}")
    for ch in CHAN:
        mass = sum(gmass_C[ch].values()) / tot_mC
        aR, _ = boot_ratio(accA_C["RD-ALL"][ch]["num"], accA_C["RD-ALL"][ch]["den"], rng)
        aW, _ = boot_ratio(accA_C["WB-0"][ch]["num"], accA_C["WB-0"][ch]["den"], rng)
        res["C_channel"]["mass"][ch] = mass; res["C_channel"]["A_RD-ALL"][ch] = aR; res["C_channel"]["A_WB-0"][ch] = aW
        print(f"{ch:8}{100*mass:8.1f}{aR:10.4f}{aW:9.4f}")

    # ====== BEHAVIORAL ======
    if args.behav:
        from mmdet.apis import inference_detector
        model = load("dino_swin_l")
        BVARS = ["full", "keep_TEXT", "keep_FLAT", "keep_MIDint", "keep_EXTint", "keep_B", "keep_G", "keep_R"]
        res["behav_kill"] = {}
        with torch.no_grad():
            tp = {}
            for i in ids:
                d0 = to_coco_dets(inference_detector(model, clean[i].astype(np.uint8)), i, cat)
                g = group(d0).get(i, [])
                tp[i] = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=False))
                         if fate(a["bbox"], a["category_id"], g)[0] == "survived"]
            ntp = sum(len(v) for v in tp.values()); res["n_tp"] = ntp
            for c in CELLS:
                res["behav_kill"][c] = {}
                for v in BVARS:
                    k = 0
                    for i in ids:
                        d = mmcv.imread(os.path.join(ADV, c, f"{i}.png")).astype(np.int32) - clean[i]
                        if v == "keep_TEXT":
                            d = d * (FL[i] == 2)[:, :, None]
                        elif v == "keep_FLAT":
                            d = d * (FL[i] == 0)[:, :, None]
                        elif v == "keep_MIDint":
                            d = d * ((clean[i] >= 16) & (clean[i] < 240))
                        elif v == "keep_EXTint":
                            d = d * ((clean[i] < 16) | (clean[i] >= 240))
                        elif v in ("keep_B", "keep_G", "keep_R"):
                            mm = np.zeros(3); mm[CHAN.index(v[-1])] = 1; d = d * mm[None, None, :].astype(np.int32)
                        img = np.clip(clean[i] + d, 0, 255).astype(np.uint8)
                        dts = to_coco_dets(inference_detector(model, img), i, cat); gg = group(dts).get(i, [])
                        k += sum(fate(x["bbox"], x["category_id"], gg)[0] != "survived" for x in tp[i])
                    res["behav_kill"][c][v] = 100 * k / max(ntp, 1)
        del model; torch.cuda.empty_cache()
        print(f"\n[hanh vi] DINO-Swin-L kill % ({ntp} object):")
        print(f"{'':10}" + "".join(f"{v[:9]:>11}" for v in BVARS))
        for c in CELLS:
            print(f"{c:10}" + "".join(f"{res['behav_kill'][c][v]:11.0f}" for v in BVARS))

    json.dump(res, open(os.path.join(REPO_ROOT, "results/mechanism/s7_pixel_dev30.json"), "w"),
              indent=1, ensure_ascii=False, default=float)
    print(f"\n[s7] -> results/mechanism/s7_pixel_dev30.json")


if __name__ == "__main__":
    main()
