"""Download the three corpora and verify SHA-256, row counts and fraud counts.

Usage: python scripts/fetch_data.py            (downloads into data/raw/)
Exits non-zero if any file does not match the manifest.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
MAN = json.load(open(os.path.join(ROOT, "data", "manifest.json")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def download(url, dest):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    os.makedirs(RAW, exist_ok=True)
    ok = True
    for name, m in MAN.items():
        dest = os.path.join(RAW, m["file"])
        print(f"[{name}]")
        if not os.path.exists(dest):
            if m["url"].endswith(".git"):
                with tempfile.TemporaryDirectory() as tmp:
                    subprocess.check_call(["git", "clone", "-q", "--depth", "1", m["url"], tmp])
                    shutil.copy(os.path.join(tmp, m["path_in_repo"]), dest)
            else:
                download(m["url"], dest)
        h = sha256(dest)
        good = h == m["sha256"]
        print(f"  sha256 {h[:16]}… {'OK' if good else 'MISMATCH'}")
        ok &= good
        if name == "tabformer":
            csv = os.path.join(RAW, m["extract"])
            if not os.path.exists(csv):
                with tarfile.open(dest) as t:
                    t.extract(m["extract"], RAW, filter="data")
            rows = fr = 0
            for ch in pd.read_csv(csv, usecols=["Is Fraud?"], chunksize=2_000_000):
                rows += len(ch); fr += int((ch["Is Fraud?"] == "Yes").sum())
        elif name == "ulb":
            d = pd.read_csv(dest, usecols=["Class"]); rows, fr = len(d), int(d.Class.sum())
        else:
            d = pd.read_csv(dest, usecols=["fraud"]); rows, fr = len(d), int(d.fraud.sum())
        good = rows == m["expected_rows"] and fr == m["expected_frauds"]
        print(f"  rows={rows:,} frauds={fr:,} {'OK' if good else 'MISMATCH'}")
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
