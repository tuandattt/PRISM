"""Fetch the MMLU CSVs into datasets/MMLU/data/{dev,val,test}/.

Two sources, tried in order:

  1. the Hendrycks tarball at Berkeley -- the original, and the one the layout
     comes from;
  2. the HuggingFace mirror ``cais/mmlu``, rebuilt into that same layout.

The fallback exists because the Berkeley host goes down for long stretches and
was the only source this script knew, which made "download MMLU" simply
impossible for the duration. Everything downstream globs ``data/<split>/*.csv``
and reads them HEADERLESS as ``question, A, B, C, D, correct_answer``
(MMLUDataset._load_data), so the mirror's {question, choices, answer} rows are
written back into exactly that shape rather than changing the loader -- which
would invalidate the existing cold-start graphs and every result built on them.

The earlier version also wrote r.content to data.tar with no status check, so a
404 page or a login redirect was saved as "data.tar" and surfaced later as
``tarfile.ReadError``, saying nothing about the cause; the `if not exists` guard
then made the failure permanent. Both are checked here before anything is kept.
"""
import glob
import os
import sys
import tarfile

import requests

URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
HF_REPO = "cais/mmlu"
# Hendrycks' split names vs the HuggingFace mirror's.
HF_SPLITS = {"dev": "dev", "val": "validation", "test": "test"}

HERE = os.path.dirname(os.path.abspath(__file__))
TAR_PATH = os.path.join(HERE, "data.tar")
DATA_PATH = os.path.join(HERE, "data")


def _looks_like_tar(path: str) -> bool:
    """A tar (or gz/bz2/xz of one) that tarfile will actually accept."""
    try:
        with tarfile.open(path) as t:
            return t.next() is not None
    except Exception:
        return False


def _describe(path: str) -> str:
    """First bytes of whatever we got, for a diagnosis that names the cause."""
    with open(path, "rb") as f:
        head = f.read(400)
    size = os.path.getsize(path)
    if head[:9].lower().startswith(b"<!doctype") or head[:5].lower().startswith(b"<html"):
        kind = "an HTML page (the server returned a web page, not the archive)"
    elif head[:2] == b"\x1f\x8b":
        kind = "gzip data that tarfile still rejected (truncated download?)"
    else:
        kind = "an unrecognised format"
    return f"{size} bytes, {kind}"


def _verify_layout() -> str:
    """Refuse to report success on a directory the loader cannot read."""
    for split in ("dev", "val", "test"):
        n = len(glob.glob(os.path.join(DATA_PATH, split, "*.csv")))
        print(f"  {split:<5} {n} csv files")
        if n == 0:
            raise SystemExit(
                f"No CSVs under {os.path.join(DATA_PATH, split)}.\n"
                f"The data is present but not in the layout MMLUDataset expects.")
    return DATA_PATH


def from_huggingface() -> str:
    """Rebuild the Hendrycks CSV layout from ``cais/mmlu``."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit("huggingface_hub is required:  pip install huggingface_hub")
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required:  pip install pandas")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("A parquet engine is required:  pip install pyarrow")

    print(f"Rebuilding MMLU from {HF_REPO}")
    files = list_repo_files(HF_REPO, repo_type="dataset")
    letters = "ABCD"
    total = 0

    for split, hf_split in HF_SPLITS.items():
        out_dir = os.path.join(DATA_PATH, split)
        os.makedirs(out_dir, exist_ok=True)
        # Per-subject files, mirroring <subject>_<split>.csv on the original host.
        # 'all' is skipped: it is the concatenation of the others.
        subjects = sorted({f.split("/")[0] for f in files
                           if "/" in f and f.split("/")[0] != "all"
                           and f.split("/")[1].startswith(hf_split + "-")})
        if not subjects:
            raise SystemExit(f"No {hf_split} shards found in {HF_REPO}.")

        rows_this_split = 0
        for subject in subjects:
            shard = next(f for f in files if f.startswith(subject + "/")
                         and f.split("/")[1].startswith(hf_split + "-"))
            df = pd.read_parquet(hf_hub_download(HF_REPO, shard, repo_type="dataset"))
            rows = []
            for r in df.to_dict("records"):
                choices = list(r["choices"])
                idx = int(r["answer"])
                # The loader's schema is fixed at four options A-D; anything else
                # would silently shift the columns.
                if len(choices) != 4 or not 0 <= idx < 4:
                    continue
                rows.append([r["question"], *choices, letters[idx]])
            if not rows:
                continue
            # header=False, index=False: _load_data reads these with header=None.
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, f"{subject}_{split}.csv"),
                header=False, index=False, encoding="utf-8")
            rows_this_split += len(rows)
        print(f"  {split:<5} {len(subjects)} subjects, {rows_this_split} questions")
        total += rows_this_split

    print(f"Wrote {total} questions to {DATA_PATH}")
    return _verify_layout()


def download(force: bool = False) -> str:
    if os.path.isdir(DATA_PATH) and not force:
        print(f"MMLU data already present at {DATA_PATH}")
        return DATA_PATH

    # A previously saved bad file must not be reused, or the failure is permanent.
    if os.path.exists(TAR_PATH) and not _looks_like_tar(TAR_PATH):
        print(f"Discarding unusable {TAR_PATH} ({_describe(TAR_PATH)})")
        os.remove(TAR_PATH)

    if not os.path.exists(TAR_PATH):
        print(f"Downloading {URL}")
        try:
            r = requests.get(URL, allow_redirects=True, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException as e:
            print(f"  unreachable ({type(e).__name__}); using the mirror instead.")
            return from_huggingface()

        # Check BEFORE writing: a 404 body is still a body.
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} (final URL {r.url}); "
                  f"using the mirror instead.")
            return from_huggingface()

        with open(TAR_PATH, "wb") as f:
            f.write(r.content)
        print(f"Saved {len(r.content)} bytes to {TAR_PATH}")

        if not _looks_like_tar(TAR_PATH):
            detail = _describe(TAR_PATH)
            os.remove(TAR_PATH)      # so a later run retries cleanly
            print(f"  not a tar archive ({detail}); using the mirror instead.")
            return from_huggingface()

    if not os.path.isdir(DATA_PATH):
        with tarfile.open(TAR_PATH) as tar:
            tar.extractall(HERE)
        print(f"Extracted to {DATA_PATH}")

    return _verify_layout()


if __name__ == "__main__":
    download(force="--force" in sys.argv)
