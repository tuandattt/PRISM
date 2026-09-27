"""How many graphs will actually train, and which ones get dropped.

Every stage writes .pt graph files, but the loaders do not read all of them.
``load_graph_dataset(args, pretrain=True)`` -- which is what run_opsd uses --
keeps only files whose NAME contains "True" or "solved":

    mixchain_17_FullConnected_3_True.pt     kept    (cold start, D_simple, D_replay)
    eff_q17_g0.pt                           DROPPED (D_pruned)

D_pruned is written by generate_pruned_data with a name carrying no verdict, so
pointing --data_dir at a FinetuneData directory silently trains on
D_simple + D_replay only. That subset is the DENSER one, which matters: the
student's operating point tracks the anchor's density closely.

So the census reports kept and dropped separately rather than one total.

    python -m experiment.graph_census data/FinetuneData_gsm8k
"""
import os
import sys
from collections import Counter

import torch


def _mode_of(filename: str) -> str:
    """Blueprint name from the filename, which is the only place it survives."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith('eff_q'):
        return 'D_pruned'
    parts = stem.split('_')
    # <dataset>_<idx>_<mode>_<n>_<is_correct>
    return parts[-3] if len(parts) >= 5 else 'unknown'


def _kept_by_loader(filename: str) -> bool:
    base = os.path.basename(filename)
    return 'True' in base or 'solved' in base.lower()


def census(path: str, label: str = '', stat_sample: int = 2000) -> dict:
    """Count graphs under ``path`` and print what will and will not train."""
    head = f"[census] {label or path}"
    if not os.path.isdir(path):
        print(f"{head}: directory does not exist")
        return {"total": 0, "kept": 0, "dropped": 0}

    files = [f for f in sorted(os.listdir(path)) if f.endswith('.pt')
             and f != 'precomputed_role_embeddings.pkl']
    kept = [f for f in files if _kept_by_loader(f)]
    dropped = [f for f in files if not _kept_by_loader(f)]

    print(f"{head}")
    print(f"    {len(files)} .pt files -> {len(kept)} train, {len(dropped)} dropped "
          f"by the pretrain filter")

    by_mode = Counter(_mode_of(f) for f in kept)
    for mode, n in by_mode.most_common():
        print(f"      kept   {mode:<16} {n}")
    for mode, n in Counter(_mode_of(f) for f in dropped).most_common():
        print(f"      DROPPED {mode:<15} {n}   (name carries no True/solved verdict)")

    # Queries and density need the files themselves; sample when there are many.
    sample = kept[:stat_sample]
    questions, nodes, edges, read = set(), 0, 0, 0
    for f in sample:
        try:
            g = torch.load(os.path.join(path, f), map_location='cpu', weights_only=False)
        except Exception:
            continue
        q = (getattr(g, 'question', '') or getattr(g, 'task', '') or '').strip()
        if q:
            questions.add(q)
        n = int(getattr(g, 'num_nodes', 0) or 0)
        ei = getattr(g, 'edge_index', None)
        nodes += n
        edges += int(ei.shape[1]) if ei is not None and hasattr(ei, 'shape') else 0
        read += 1

    if read:
        suffix = f" (over {read} sampled)" if len(kept) > len(sample) else ""
        print(f"      {len(questions)} distinct queries, mean |V| {nodes/read:.2f}, "
              f"mean |E| {edges/read:.2f}{suffix}")
        if len(questions) < 20:
            print(f"      WARNING: only {len(questions)} distinct queries. The privilege "
                  f"gain's effective sample size is QUERIES, not graphs.")
    return {"total": len(files), "kept": len(kept), "dropped": len(dropped),
            "queries": len(questions), "by_mode": dict(by_mode)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m experiment.graph_census <dir> [<dir> ...]")
    for d in sys.argv[1:]:
        census(d)
