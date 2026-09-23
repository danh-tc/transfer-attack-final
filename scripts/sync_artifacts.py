#!/usr/bin/env python3
"""Đồng bộ artifact nặng (ảnh adv PNG, dets.json, gen_dets/) của 1 run với HF dataset repo
PRIVATE — vì máy GPU thuê không giữ gì sau khi trả (docs/environment_setup.md, mục
"Persistent storage"). Mỗi run = 1 file tar `runs/<run>.tar` trên repo, kèm sha256.

Token đọc từ biến môi trường HF_TOKEN (quyền write để upload) — KHÔNG ghi token vào repo.

  HF_TOKEN=... python scripts/sync_artifacts.py upload   n300_B50_eps5
  HF_TOKEN=... python scripts/sync_artifacts.py download n300_B50_eps5
"""
import argparse
import hashlib
import os
import sys
import tarfile
import tempfile

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_REPO = "congdanh99/transfer-attack"  # dataset, private (chuyển private 2026-09-23)
ART_RUNS = os.path.join(REPO_ROOT, "artifacts/runs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["upload", "download"])
    p.add_argument("run")
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Thiếu HF_TOKEN trong biến môi trường.")
    api = HfApi(token=token)
    assert api.repo_info(HF_REPO, repo_type="dataset").private, f"{HF_REPO} không còn private — dừng."
    remote = f"runs/{args.run}.tar"

    if args.action == "upload":
        src = os.path.join(ART_RUNS, args.run)
        assert os.path.isdir(src), src
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = os.path.join(tmp, f"{args.run}.tar")
            with tarfile.open(tar_path, "w") as tar:
                tar.add(src, arcname=args.run)
            digest, size = sha256(tar_path), os.path.getsize(tar_path)
            api.upload_file(path_or_fileobj=tar_path, path_in_repo=remote, repo_id=HF_REPO,
                            repo_type="dataset", commit_message=f"{args.run} sha256={digest}")
        print(f"uploaded {remote} ({size} bytes) sha256={digest}")
    else:
        path = hf_hub_download(HF_REPO, remote, repo_type="dataset", token=token)
        os.makedirs(ART_RUNS, exist_ok=True)
        with tarfile.open(path) as tar:
            tar.extractall(ART_RUNS)
        print(f"downloaded + extracted -> {os.path.join(ART_RUNS, args.run)} sha256={sha256(path)}")


if __name__ == "__main__":
    main()
