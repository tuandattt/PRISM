"""Fetch MBPP and write it in this repo's HumanEval record shape.

MBPP (Mostly Basic Python Problems) ships 974 short tasks, each with a natural
language statement, a reference solution, and three assert statements. That last
part is why it needs no new evaluator here: HumanEval's evaluator reads
``record["prompt"]`` and ``record["test"]`` and executes the latter against the
generated code, so MBPP written in the same shape runs through it unchanged.

    Question     text          -> prompt
    Tests        test_list     -> test        wrapped as check(candidate)
    Reference    code          -> canonical_solution   (privileged text only)
    Entry point  parsed from   -> entry_point  the first def in the reference

THE PROMPT NEEDS THE ASSERTS
----------------------------
MBPP's ``text`` alone does not pin down the function NAME or SIGNATURE -- "write
a python function to remove first and last occurrence of a given character"
could be any of a dozen signatures, and the tests call one specific one. The
standard MBPP protocol therefore shows the model its assert statements as part
of the prompt, and that is what is done here. Without them the task is
unanswerable and a near-zero score would measure the omission, not the model.

The reference solution is NOT in the prompt. It goes in ``canonical_solution``,
which only the privileged teacher reads.

SPLITS
------
MBPP's canonical layout is train 374 / test 500 / validation 90 / prompt 10.
``--split test`` (the default) is the 500-problem evaluation set everyone
reports. Because it is evaluation-only here, the script also writes a task-split
file marking every record as test, so evaluate_humaneval has one to read.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "google-research-datasets/mbpp"
CONFIG = "full"

sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..')))


def out_path(split: str) -> str:
    return os.path.join(HERE, f"mbpp_{split}.jsonl" if split != "test" else "mbpp.jsonl")


def entry_point_of(code: str) -> str:
    """The function the tests will call: the LAST top-level def in the reference.

    Last, not first: MBPP references sometimes define a helper before the
    function under test, and the asserts always call the final one.
    """
    names = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", code or "", re.MULTILINE)
    return names[-1] if names else ""


def build_prompt(text: str, tests) -> str:
    """Statement plus the asserts, which is the standard MBPP protocol."""
    body = str(text or "").strip()
    shown = "\n".join(str(t).strip() for t in (tests or []) if str(t).strip())
    return f"{body}\n\nYour code should satisfy these tests:\n{shown}\n"


def build_test(tests, setup: str, entry: str) -> str:
    """MBPP asserts as a runnable test body.

    PyExecutor.execute concatenates the candidate code with this string and runs
    the result, so the assertions must execute at MODULE level. HumanEval's
    records look like they only define ``check(candidate)``, but they end with

        def test_check():
            check(strlen)
        test_check()

    -- the call is what makes the harness fire. A first version here emitted the
    ``def check(...)`` wrapper and nothing else: concatenation would then merely
    DEFINE a function, no assertion would run, and every submission -- including
    an empty one -- would be graded correct.

    MBPP asserts already call the function by name and the candidate defines
    that name, so they are emitted directly. No wrapper, nothing to forget to
    call.
    """
    body = [str(t).strip() for t in (tests or []) if str(t).strip()]
    if not body:
        return ""
    lines = [s for s in (setup or "").splitlines() if s.strip()]
    return chr(10).join(lines + body) + chr(10)


def download(split: str = "test", force: bool = False) -> str:
    OUT = out_path(split)
    if os.path.exists(OUT) and not force:
        with open(OUT, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        # An EMPTY file is a crashed previous run, not a completed one. Treating
        # it as present makes the failure permanent: every retry short-circuits
        # on the wreckage of the first one and reports success with 0 records.
        if n:
            print(f"MBPP {split} already present: {OUT} ({n} records)")
            return OUT
        print(f"{OUT} exists but is empty -- a previous run failed; re-downloading.")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required:  pip install huggingface_hub\n"
            "(the HF `datasets` library cannot be used -- this repo's own\n"
            " datasets/ directory shadows it on sys.path)")
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required:  pip install pandas")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("A parquet engine is required:  pip install pyarrow")

    files = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
             if f.endswith(".parquet") and f"/{split}-" in f and CONFIG in f]
    if not files:
        files = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
                 if f.endswith(".parquet") and split in f]
    if not files:
        raise SystemExit(
            f"No parquet for split {split!r} in {REPO_ID}. "
            f"Available: {[f for f in list_repo_files(REPO_ID, repo_type='dataset') if f.endswith('.parquet')]}")

    frames = []
    for fn in sorted(files):
        local = hf_hub_download(REPO_ID, fn, repo_type="dataset")
        frames.append(pd.read_parquet(local))
        print(f"  read {fn}: {len(frames[-1])} rows")
    df = pd.concat(frames, ignore_index=True)

    need = ("text", "code", "test_list")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Expected columns {missing} absent. Present: {list(df.columns)}\n"
            f"The dataset schema has changed; update the mapping in this file.")

    kept, dropped, no_entry = 0, 0, 0
    with open(OUT, "w", encoding="utf-8") as out:
        for row in df.to_dict("records"):
            text = str(row.get("text") or "").strip()
            code = str(row.get("code") or "").strip()
            # pandas hands back a numpy array for a parquet list column, and
            # `arr or []` raises "truth value of an array is ambiguous" rather
            # than falling back. Test for None explicitly.
            raw_tests = row.get("test_list")
            tests = [] if raw_tests is None else list(raw_tests)
            entry = entry_point_of(code)
            test_src = build_test(tests, str(row.get("test_setup_code") or ""), entry)
            if not text or not tests or not test_src:
                dropped += 1
                continue
            if not entry:
                no_entry += 1
            out.write(json.dumps({
                "task_id": f"MBPP/{row.get('task_id')}",
                "name": f"MBPP_{row.get('task_id')}",
                "prompt": build_prompt(text, tests),
                "test": test_src,
                "entry_point": entry,
                # Privileged only -- never shown to the student.
                "canonical_solution": code,
                "language": "py",
            }, ensure_ascii=False, default=str) + "\n")
            kept += 1

    print(f"Wrote {kept} records to {OUT} ({dropped} dropped)")
    if no_entry:
        print(f"  {no_entry} had no parseable 'def' in the reference; their asserts "
              f"still run, but candidate is not bound to a name")
    if kept == 0:
        raise SystemExit("No usable records -- refusing to leave an empty file.")

    # Evaluation-only, so every record is a test record. evaluate_humaneval
    # requires a split file; without this it raises FileNotFoundError and there
    # is nothing sensible for the user to point it at.
    from experiment.paths import task_split_file, ensure
    sp = task_split_file("mbpp")
    ensure(os.path.dirname(sp))
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"base_tasks_indices": [], "finetune_tasks_indices": [],
                   "test_indices": list(range(kept))}, f)
    print(f"  task split (all {kept} records are test) -> {sp}")
    return OUT


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test",
                    choices=["test", "train", "validation", "prompt"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    download(split=a.split, force=a.force)
