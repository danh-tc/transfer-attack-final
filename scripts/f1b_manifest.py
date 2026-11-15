"""Ghi MANIFEST.json (git commit, GPU, seed, sha256 checkpoint + PNG) vào artifacts/runs/f1b_r1a_eps5_3090/ trước khi upload HF."""
import hashlib, json, os, subprocess, torch
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); run = os.path.join(root, "artifacts/runs/f1b_r1a_eps5_3090")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ck = {f: sha(os.path.join(root, "checkpoints", f)) for f in sorted(os.listdir(os.path.join(root, "checkpoints")))}
pngs = {}
for d, _, fs in os.walk(os.path.join(run, "adv")):
    for f in fs:
        if f.endswith(".png"):
            p = os.path.join(d, f); pngs[os.path.relpath(p, run)] = sha(p)
m = {"git_commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"]).decode().strip(),
     "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "eps": 5, "images": "data/image_lists/dev100.csv",
     "script": "scripts/f1b_r1a.py", "protocol": "docs/progress_log.md entries 2026-11-15 (Amendment 3, F1b/R1a, OSFD control)",
     "seeds": {"RD-CNX-B100 (snap36/50/75/final)": "seed_all(1000*(n+1))", "RD-CNX-B50s2": "seed_all(1000*(n+1)+7)",
               "RD-R50-B36": "seed_all(1000*(n+1))", "ENS-WP-B36": "e88.ens_wp_attack: 1000*(n+1)+block",
               "OSFD-PUB-B50": "seed_all(1000*(n+1)), theta7 rho0.8"},
     "note": "RD-CNX không tất định bit trên 3090 ở một số ảnh (results/mechanism/f1b_snapcheck.json)",
     "checkpoints_sha256": ck, "png_sha256": pngs}
json.dump(m, open(os.path.join(run, "MANIFEST.json"), "w"), indent=1, ensure_ascii=False)
print("manifest", len(pngs), "png", len(ck), "ckpt")
