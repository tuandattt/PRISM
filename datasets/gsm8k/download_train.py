"""Fetch GSM8K's TRAIN split, so cold start need not consume the test set.

``datasets/gsm8k/gsm8k.jsonl`` holds 1,319 records -- that is GSM8K's *test*
split, the set every benchmark number in this repo is measured on. Building a
cold-start corpus from it therefore costs evaluation questions: the pipeline
reserves the first ``num_iterations x batch_size`` records, leaving 1,159 to
score. GSM8K also ships a 7,473-example train split that nothing here evaluates,
and a corpus built from that leaves all 1,319 test questions available.

Output shape matches ``gsm8k.jsonl`` exactly -- the RAW two-field form,
``question`` and ``answer``, the latter still carrying its ``#### N`` line.
``gsm_data_process`` is what splits that into task/step/answer downstream, so
writing the processed form here would raise ``KeyError: 'question'``.

    python -m datasets.gsm8k.download_train
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "gsm8k_train.jsonl")
REPO_ID = "openai/gsm8k"


def download(force: bool = False) -> str:
    if os.path.exists(OUT_PATH) and not force:
        with open(OUT_PATH, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"GSM8K train already present: {OUT_PATH} ({n} records)")
        return OUT_PATH

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required:  pip install huggingface_hub\n"
            "(the HF `datasets` library cannot be used here -- this repo's own\n"
            " datasets/ directory shadows it on sys.path)")
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required:  pip install pandas")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            raise SystemExit("A parquet engine is required:  pip install pyarrow")

    # main/train-*.parquet; the socratic config repeats the same questions with a
    # different solution style, so it is excluded to avoid duplicate queries.
    files = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
             if f.endswith(".parquet") and "train" in f and "socratic" not in f]
    if not files:
        raise SystemExit(f"No train parquet found in {REPO_ID}.")
    print(f"Found {len(files)} file(s): {files}")

    frames = []
    for fn in sorted(files):
        local = hf_hub_download(REPO_ID, fn, repo_type="dataset")
        frames.append(pd.read_parquet(local))
        print(f"  read {fn}: {len(frames[-1])} rows")
    df = pd.concat(frames, ignore_index=True)

    kept, dropped = [], 0
    for _, row in df.iterrows():
        q = str(row.get("question") or "").strip()
        sol = str(row.get("answer") or "").strip()
        # The '#### N' line is what gsm_data_process splits on; a row without it
        # would yield an unparseable gold answer and silently score as wrong.
        if not q or not re.search(r"####\s*-?[\d,]+(?:\.\d+)?", sol):
            dropped += 1
            continue
        kept.append({"question": q, "answer": sol})

    if not kept:
        raise SystemExit("No usable rows; check the repo's column names.")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} records to {OUT_PATH}"
          + (f" ({dropped} dropped: no question or no '#### N' line)" if dropped else ""))
    return OUT_PATH


if __name__ == "__main__":
    download()
