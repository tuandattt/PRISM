"""Fetch horseee/MixChain-Z-PRM12K and convert it to this repo's record shape.

Why not `from datasets import load_dataset`: this repo has its own top-level
``datasets/`` directory, which shadows the HuggingFace library on sys.path --
``import datasets`` here resolves to the local namespace package. So the parquet
files are pulled through huggingface_hub and read with pandas instead.

Field mapping (see the dataset card):

    question  -> task      the problem the student sees
    answer    -> step      the worked solution, i.e. the PRIVILEGED text
    target    -> answer    the final value, used to check correctness

``solution_0..4`` and ``correct_1..4`` are carried through untouched: MixChain's
whole point is chains of *varying length*, so "does a longer rationale make a
better teacher?" is a natural ablation and the data for it should not be thrown
away here.

Rows whose question, solution or target is empty are dropped -- a row with no
privileged text would be fed a zero vector, which after training is an arbitrary
constant modulation rather than "no information".
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "mixchain.jsonl")
REPO_ID = "horseee/MixChain-Z-PRM12K"


def download(force: bool = False) -> str:
    if os.path.exists(OUT_PATH) and not force:
        with open(OUT_PATH, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"MixChain already present: {OUT_PATH} ({n} records)")
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
    # pandas defers the parquet engine import until read_parquet(), so without
    # this the failure lands AFTER the 62MB download rather than before it.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            raise SystemExit(
                "A parquet engine is required:  pip install pyarrow\n"
                "(pandas needs pyarrow or fastparquet to read .parquet files)")

    files = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
             if f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"No parquet files found in {REPO_ID}. "
                         f"Check the repo layout on huggingface.co.")
    print(f"Found {len(files)} parquet file(s): {files}")

    frames = []
    for fn in sorted(files):
        local = hf_hub_download(REPO_ID, fn, repo_type="dataset")
        frames.append(pd.read_parquet(local))
        print(f"  read {fn}: {len(frames[-1])} rows")
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in ("question", "answer", "target") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Expected columns {missing} are absent. Present: {list(df.columns)}\n"
            f"The dataset schema has changed; update the mapping in this file.")

    extra = [c for c in df.columns if c.startswith(("solution_", "correct_"))]

    kept, dropped = 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for row in df.to_dict("records"):
            question = str(row.get("question") or "").strip()
            solution = str(row.get("answer") or "").strip()
            target = str(row.get("target") or "").strip()
            if not question or not solution or not target:
                dropped += 1
                continue
            rec = {"task": question, "step": solution, "answer": target}
            for c in extra:
                v = row.get(c)
                rec[c] = bool(v) if c.startswith("correct_") else v
            out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            kept += 1

    print(f"Wrote {kept} records to {OUT_PATH} ({dropped} dropped for missing fields)")
    if kept == 0:
        raise SystemExit("No usable records -- refusing to leave an empty file.")
    return OUT_PATH


if __name__ == "__main__":
    download(force="--force" in sys.argv)
