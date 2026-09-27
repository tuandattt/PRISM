"""Supervised fine-tuning baseline: the post-training method OPSD is compared to.

OPSD's claim is that on-policy distillation from a privileged teacher beats
simply training longer on the same supervised data. That claim is only testable
against an SFT arm matched on everything except the objective, so this script is
run_opsd with Stage 1 and Stage 2 replaced by continued MLE:

    cold-start checkpoint  ->  more epochs of  -log P_theta(G | Q)  ->  best_model.pth

WHAT IS HELD FIXED
------------------
Same checkpoint, same graph corpus, same query-level train/val split (same
--seed, so literally the same queries land in val), same optimiser and learning
rate, same epoch budget, same output format. The ONLY difference from the OPSD
arm is that the loss is the anchor term alone -- no teacher, no rollouts, no KL.

That matters for the ablation table. A reader comparing OPSD against a cold
start alone cannot tell whether the gain came from the distillation or merely
from 30 more epochs of gradient descent; comparing against this arm can.

WHY NOT JUST --anchor_weight 1.0
--------------------------------
Setting anchor_weight to 1.0 in run_opsd does zero out the KL, and is a valid
ablation of the LOSS. But it still trains the teacher, still samples on-policy
rollouts every step, and still steps the optimiser once per rollout batch rather
than once per graph batch -- so its epoch count means something different and
its wall-clock is dominated by work the SFT baseline would never do. A standalone
arm is what a reviewer expects to see labelled "SFT", and it is honest about the
compute each method actually needs.

EARLY STOPPING
--------------
On validation NLL, which is standard SFT practice and is also the charitable
choice: without it, a long SFT run overfits a few-thousand-graph corpus and the
comparison flatters OPSD for the wrong reason. --patience 0 disables it if the
fixed-epoch comparison is wanted instead.

    python -m experiment.opsd.run_sft --dataset mixchain \\
        --student_dir output/mixchain_20260829-200412 \\
        --data_dir data/FinetuneData_mixchain \\
        --sft_epochs 30 --output_dir output/mixchain_sft_seed42 --seed 42
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import experiment.process_dataset as gdata                       # noqa: E402
from experiment.args import Args                                 # noqa: E402
from experiment.graph_census import census                       # noqa: E402
from experiment.model import ARGDesigner                         # noqa: E402
from experiment.opsd.run_opsd import serialisable_args, setup    # noqa: E402
# Reused, NOT reimplemented. These decide how a loader batch becomes model
# input; a private copy here would drift from the OPSD arm and the two would
# stop being comparable, which is the one thing this script exists to guarantee.
from experiment.opsd.train_opsd import (_stack_task_embeddings,  # noqa: E402
                                        _unwrap)
from experiment.paths import OUTPUT_ROOT, cold_start_dir, ensure  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=str, default='gsm8k')
    p.add_argument('--student_dir', type=str, required=True,
                   help="Cold-start run directory -- the SAME one the OPSD arm started "
                        "from, or the comparison is not controlled.")
    p.add_argument('--student_ckpt', type=str, default='best_model.pth')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--extra_data_dir', type=str, default=None, nargs='+')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42,
                   help="Must match the OPSD arm: the train/val split is drawn from "
                        "this, so a different seed compares different splits.")
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--sft_epochs', type=int, default=30,
                   help="Match --opsd_epochs so the two arms see the same budget.")
    p.add_argument('--sft_lr', type=float, default=None,
                   help="Defaults to --opsd_lr from the checkpoint's args, so the "
                        "arms differ in objective rather than in step size.")
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=5,
                   help="Early-stopping patience on validation NLL. 0 disables it and "
                        "trains the full --sft_epochs.")
    p.add_argument('--dry_run', action='store_true')
    cli, _ = p.parse_known_args()
    return cli, Args().update_args()


def evaluate_nll(model, loader, device) -> float:
    """Mean per-graph negative log-likelihood over a loader."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            graphs = [g for g in _unwrap(batch) if g is not None]
            if not graphs:
                continue
            task = _stack_task_embeddings(graphs, device)
            log_ll, _ = model(graphs, task)
            if torch.isnan(log_ll).any():
                continue
            total += float(-log_ll.sum())
            n += len(graphs)
    model.train()
    return total / max(n, 1)


def main():
    cli, _tuned = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup(cli.seed, device)

    ckpt_path = os.path.join(cli.student_dir, cli.student_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No cold-start checkpoint at {ckpt_path}. SFT, like OPSD, is a "
            f"post-training method -- it continues from a trained designer.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = checkpoint['data_statistics']

    args = Args()
    args.update_args_from_dict(checkpoint['args'])
    args.seed = cli.seed
    args.batch_size = cli.batch_size
    args.device = device
    args.dataset = cli.dataset
    args.dataset_name = cli.dataset
    args.data_dir = cli.data_dir or cold_start_dir(cli.dataset)
    args.feed_edge_features = bool(int(args.feed_edge_features))
    lr = cli.sft_lr if cli.sft_lr is not None else float(getattr(args, 'opsd_lr', 1e-4))

    out_dir = cli.output_dir or os.path.join(
        OUTPUT_ROOT, f"{cli.dataset}_sft_seed{cli.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    ensure(out_dir)
    args.experiment_path = out_dir
    print(f"[sft] outputs -> {out_dir}")
    print(f"[sft] continuing from {ckpt_path} at lr={lr}")

    # ---- corpus, identical to the OPSD arm -------------------------------
    census(args.data_dir, label="SFT training graphs")
    graph_dataset, _ = gdata.load_graph_dataset(args)
    graphs = graph_dataset.graph_list
    if not graphs:
        raise RuntimeError(f"No cold-start graphs under {args.data_dir}")

    for extra in (cli.extra_data_dir or []):
        primary = args.data_dir
        try:
            args.data_dir = extra
            extra_ds, _ = gdata.load_graph_dataset(args)
        finally:
            args.data_dir = primary
        graphs = graphs + extra_ds.graph_list
        print(f"[sft] merged {len(extra_ds.graph_list)} graphs from {extra}")

    questions = {g.graph.get('task') for g in graphs if g.graph.get('task')}
    print(f"[sft] {len(graphs)} graphs over {len(questions)} distinct queries")

    # Split by QUERY with the same seed as run_opsd, so both arms validate on
    # the same held-out queries. A graph-level split would put siblings of one
    # query on both sides and measure recall instead of generalisation.
    by_query = {}
    for g in graphs:
        by_query.setdefault(g.graph.get('task') or id(g), []).append(g)
    keys = sorted(by_query, key=str)
    random.Random(cli.seed).shuffle(keys)
    n_val_q = max(1, int(round(len(keys) * cli.val_frac)))
    val_keys, train_keys = set(keys[:n_val_q]), keys[n_val_q:]
    train_graphs = [g for k in train_keys for g in by_query[k]]
    val_graphs = [g for k in val_keys for g in by_query[k]] or by_query[keys[0]]
    print(f"[sft] split by query: {len(train_keys)} train / {len(val_keys)} val "
          f"({len(train_graphs)} / {len(val_graphs)} graphs)")

    make_loader = lambda gs, sh: torch.utils.data.DataLoader(
        gdata.GraphListDataset(gs, args), batch_size=cli.batch_size,
        shuffle=sh, collate_fn=lambda x: x)
    loader_train = make_loader(train_graphs, True)
    loader_val = make_loader(val_graphs, False)

    if cli.dry_run:
        print(f"\n[sft] --dry_run: {len(train_graphs)} train / {len(val_graphs)} val "
              f"graphs, lr={lr}, {cli.sft_epochs} epochs. Stopping before training.")
        return

    # ---- continued MLE ----------------------------------------------------
    model = ARGDesigner(args, stats).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = evaluate_nll(model, loader_val, device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    print(f"[sft] epoch 0 (cold start)  val_nll={best_val:.4f}")

    history = [{"epoch": 0, "val_nll": best_val}]
    bad = 0
    for epoch in range(1, cli.sft_epochs + 1):
        t0 = time.time()
        total, nb = 0.0, 0
        for batch in loader_train:
            gs = [g for g in _unwrap(batch) if g is not None]
            if not gs:
                continue
            task = _stack_task_embeddings(gs, device)
            log_ll, _ = model(gs, task)
            loss = -torch.mean(log_ll)
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            if getattr(args, 'clip', True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
            nb += 1

        val = evaluate_nll(model, loader_val, device)
        train_nll = total / max(nb, 1)
        mark = ""
        if val < best_val:
            best_val, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
            mark = "  *best"
        else:
            bad += 1
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val})
        print(f"[sft] epoch {epoch}/{cli.sft_epochs} train_nll={train_nll:.4f} "
              f"val_nll={val:.4f}{mark} ({time.time()-t0:.1f}s)")

        if cli.patience and bad >= cli.patience:
            print(f"[sft] early stop: no val improvement for {cli.patience} epochs")
            break

    if best_epoch == 0:
        print("[sft] WARNING: validation NLL never improved on the cold start. The")
        print("      saved checkpoint is the cold start UNCHANGED, so this arm and")
        print("      the cold-start baseline are the same model -- report it as such")
        print("      rather than as a distinct row.")

    # Same format and filename as run_opsd, so evaluate_math needs --ef 0 and
    # nothing else changes.
    torch.save({'model_state_dict': best_state, 'data_statistics': stats,
                'args': serialisable_args(args)},
               os.path.join(out_dir, 'best_model.pth'))
    with open(os.path.join(out_dir, 'sft_history.json'), 'w', encoding='utf-8') as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_val_nll": best_val, "lr": lr}, f, indent=2)
    print(f"[sft] best epoch {best_epoch}, val_nll {best_val:.4f} -> "
          f"{os.path.join(out_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
