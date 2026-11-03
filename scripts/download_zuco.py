"""Download ZuCo 1.0 Task 1 (sentiment reading) from OSF.

Fetches the *Preprocessed* folder (continuous cleaned EEG blocks + synchronised
eye-tracking per block, ~5 GB) plus the sentiment label files.

Resumable: files that already exist with the expected size are skipped.
Uses `curl` (the python.org build of Python on macOS often lacks SSL certs).

    python scripts/download_zuco.py --out data/zuco
"""
import argparse
import json
import os
import subprocess
import sys
import time

TASK1_FOLDER = "https://api.osf.io/v2/nodes/q3zws/files/osfstorage/5ace011fc7a6c0000cd023b8/"
MATERIALS_FOLDER = "https://api.osf.io/v2/nodes/q3zws/files/osfstorage/5af93ed67bf0b90012e5d340/"


def api(url, retries=5):
    for i in range(retries):
        r = subprocess.run(["curl", "-sL", "--max-time", "120", url], capture_output=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"OSF API failed: {url}")


def walk(url, rel=""):
    """Yield (relative_path, size, download_url) for every file under an OSF folder."""
    while url:
        d = api(url)
        for x in d["data"]:
            a = x["attributes"]
            path = os.path.join(rel, a["name"])
            if a["kind"] == "folder":
                yield from walk(x["relationships"]["files"]["links"]["related"]["href"], path)
            else:
                yield path, a["size"], x["links"]["download"]
        url = d["links"].get("next")


def fetch(path, size, url, out):
    dst = os.path.join(out, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for i in range(5):
        r = subprocess.run(["curl", "-sL", "--fail", "-C", "-", "-o", tmp, url])
        if r.returncode == 0 and os.path.getsize(tmp) == size:
            os.replace(tmp, dst)
            return "ok"
        time.sleep(3 * (i + 1))
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/zuco")
    args = ap.parse_args()

    # locate the Preprocessed subfolder of task 1
    pre = next(x for x in api(TASK1_FOLDER)["data"] if x["attributes"]["name"] == "Preprocessed")
    pre_url = pre["relationships"]["files"]["links"]["related"]["href"]

    jobs = [(os.path.join("meta", p), s, u) for p, s, u in walk(MATERIALS_FOLDER)]
    jobs += [(os.path.join("preprocessed", p), s, u) for p, s, u in walk(pre_url)]
    total = sum(s for _, s, _ in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB", flush=True)

    done = 0
    for i, (p, s, u) in enumerate(jobs, 1):
        status = fetch(p, s, u, args.out)
        done += s
        print(f"[{i}/{len(jobs)}] {status:4s} {p} ({done / total:.0%})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
