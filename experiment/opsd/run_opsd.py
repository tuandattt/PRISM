"""Run OPSD on top of a cold-start checkpoint.

    python -m experiment.opsd.run_opsd --dataset gsm8k \
        --student_dir output/gsm8k_<timestamp> --teacher_epochs 40 --opsd_epochs 30

Stage 0 (cold start) is NOT run here and is not optional: it produces the only
verified data in the system, and it is what both the teacher and the student are
initialised from. A student initialised from scratch samples pseudo-graphs that
are noise, and there would be no data to fit the teacher on. OPSD replaces
Stage 2 (the D_pruned generate/ablate/re-execute loop), not Stage 1.
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import pickle

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment import process_dataset as gdata
from experiment.args import Args
from experiment.model import ARGDesigner
from experiment.graph_census import census
from experiment.opsd.data import (AnswerEmbeddings, build_query_pool,
                                  privileged_context_text)
from experiment.opsd.graph_pi import GraphPrivilegedEmbeddings
from experiment.opsd.graph_stats_pi import GraphStatsEmbeddings
from experiment.opsd.privileged import build_teacher
from experiment.opsd.train_opsd import train_opsd, train_teacher
from experiment.paths import cold_start_dir, dataset_file, ensure, task_split_file, OUTPUT_ROOT
from mas_framework.tools.reader.readers import JSONLReader


# Flags OPSD owns: taken from the command line, never from the checkpoint
# (the checkpoint predates them, and its stale copies would silently win).
OPSD_KEYS = (
    'alpha', 'legacy_edge_polarity',
    'teacher_epochs', 'teacher_lr', 'privileged_only', 'privileged_dropout',
    'privileged_dropout_p', 'privileged_rank',
    'opsd_epochs', 'opsd_lr', 'opsd_batch_size', 'rollouts_per_query',
    'rollout_temperature', 'kl_direction', 'teacher_temperature', 'jsd_beta',
    'anchor_weight', 'feed_edge_features', 'pseudo_filter', 'teacher_keep_frac',
    'opsd_query_limit', 'clip',
)


def serialisable_args(args) -> dict:
    """Checkpoint-safe view of an ``Args``/Namespace.

    ``Args`` keeps its own ``ArgumentParser``, and a parser holds closures from
    ``ArgumentParser.__init__`` that pickle cannot reach -- so dumping
    ``args.__dict__`` wholesale kills the save. Keep plain data only, which also
    makes the checkpoints portable.
    """
    keep = {}
    for k, v in args.__dict__.items():
        if k.startswith('_') or k == 'parser':
            continue
        if isinstance(v, torch.device):
            keep[k] = str(v)
            continue
        try:
            pickle.dumps(v)
        except Exception:
            continue
        keep[k] = v
    return keep


def parse_args():
    """OPSD-specific flags here; everything else comes from ``Args``.

    Both parsers read the same argv with ``parse_known_args``, so a flag defined
    on either is accepted.
    """
    p = argparse.ArgumentParser(description="On-policy self-distillation for ARG-Designer")
    p.add_argument('--dataset', type=str, default='gsm8k',
                   choices=['gsm8k', 'aqua', 'humaneval', 'mmlu', 'mixchain'])
    p.add_argument('--student_dir', type=str, required=True,
                   help='Cold-start run directory containing best_model.pth')
    p.add_argument('--student_ckpt', type=str, default='best_model.pth')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=16, help='Batch size for the NLL passes')
    p.add_argument('--skip_teacher', action='store_true',
                   help='Reuse an existing teacher.pth in --output_dir')
    p.add_argument('--teacher_dataset', type=str, default=None, nargs='+',
                   choices=['gsm8k', 'aqua', 'humaneval', 'mmlu', 'mixchain'],
                   help="Dataset(s) the cold-start graphs came from, when they differ "
                        "from --dataset. Accepts several, one per graph set, so a "
                        "teacher can be fitted on gsm8k AND aqua at once. "
                        "Longer form: dataset the --data_dir cold-start graphs came from, when it "
                        "differs from --dataset (the transfer setup: teacher finetuned "
                        "on gsm8k graphs, rollouts over mixchain queries). Stage 1 looks "
                        "up privileged vectors by the GRAPH's question, so without this "
                        "every Stage-1 lookup misses, the FiLM channel receives zeros, "
                        "and the privilege gain is noise around zero by construction. "
                        "Defaults to --dataset.")
    p.add_argument('--privileged_source', type=str, default='rationale',
                   choices=['context', 'rationale', 'sibling_graph', 'gold_graph',
                            'graph_stats'],
                   help="What the teacher gets to see that the student never does. "
                        "'context' is the OPSD paper's own design: ONE model acts as both, "
                        "the teacher merely conditioned on question+solution instead of the "
                        "question alone -- no extra parameters and NO teacher finetune. "
                        "'rationale' is the per-dataset worked solution / tests / option "
                        "text. 'sibling_graph' is a UNIFIED channel needing no rationale "
                        "field: the teacher predicting graph A is shown a different graph "
                        "B that also solved the same query. 'graph_stats' shows the "
                        "SUFFICIENT STATISTICS of the whole set that solved this "
                        "query -- role frequencies, which role opens, size and "
                        "density moments -- computed leave-one-out so the "
                        "predicted graph is excluded from its own input. It is "
                        "strictly more informative than one sibling and still "
                        "names no target. 'gold_graph' shows A itself -- "
                        "the target is then in the input, copying is optimal, and the "
                        "distillation degenerates into the supervised MLE cold start "
                        "already ran; it is an ablation, not a method.")
    p.add_argument('--privileged_control', type=str, default='none',
                   choices=['none', 'shuffle', 'constant'],
                   help="Ablate the privileged INFORMATION while keeping the channel's "
                        "capacity. 'shuffle' permutes rationales onto the wrong queries; "
                        "'constant' feeds every query the mean vector. Any gain that "
                        "survives these is the FiLM bias adapting the frozen trunk to the "
                        "graph distribution, not privileged information. Always run "
                        "'shuffle' alongside a headline number.")
    p.add_argument('--data_dir', type=str, default=None,
                   help="Graph set the teacher is fitted on. Default: the cold-start "
                        "dir (D_exp). Point at data/FinetuneData_<dataset> to use D_eff, "
                        "whose labels are far more query-dependent -- D_exp uses only "
                        "resource-rich blueprints, so most queries accept all of them "
                        "and there is little for any conditioning signal to learn.")
    p.add_argument('--extra_data_dir', type=str, default=None, nargs='+',
                   help="Additional graph sets to fit the teacher on, merged with "
                        "--data_dir. Use with --teacher_dataset to fit one teacher on "
                        "several corpora at once (e.g. gsm8k + aqua). The graphs are "
                        "pooled, so this raises the privilege gain's sample size -- but "
                        "it also changes the ANCHOR's density target, and the student's "
                        "operating point tracks that closely. Do not read a density shift "
                        "after merging as an effect of the teacher.")
    p.add_argument('--dry_run', action='store_true',
                   help="Load the graphs, privileged table and rollout pool, print "
                        "what they contain, then stop -- before Stage 1. Every "
                        "data-plumbing failure this pipeline has (a split whose "
                        "indices address the wrong records, graphs carrying no "
                        "query, a stale embedding cache) is visible here in "
                        "seconds instead of after an hour of teacher training.")
    p.add_argument('--val_frac', type=float, default=0.1,
                   help='Validation fraction. The privilege go/no-go is read off this '
                        'split, and its error bar scales as 1/sqrt(n), so a probe run '
                        '(--opsd_epochs 0) is better served by 0.4-0.5 than by 0.1.')
    cli, _ = p.parse_known_args()
    return cli, Args().update_args()


def setup(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(dataset):
    """Dataset records in the {task, step, answer} shape the pipeline uses."""
    path = dataset_file(dataset)
    if dataset == 'gsm8k':
        from datasets.gsm8k_dataset import gsm_data_process
        return gsm_data_process(JSONLReader.parse_file(path))
    if dataset == 'aqua':
        from datasets.aqua_dataset import aqua_data_process
        return aqua_data_process(JSONLReader.parse_file(path))
    if dataset == 'mixchain':
        # Already written in {task, step, answer} shape by the download script.
        return JSONLReader.parse_file(path)
    if dataset == 'humaneval':
        # HumanEval keys its question as 'prompt'; everything downstream
        # (build_query_pool, AnswerEmbeddings) looks for 'task', so normalise
        # here rather than special-casing the key in three more places.
        rows = JSONLReader.parse_file(path)
        for r in rows:
            r.setdefault('task', r.get('prompt', ''))
        return rows
    if dataset == 'mmlu':
        # MMLU is a DataFrame-backed dataset, not JSONL. Flatten it to the same
        # {task, answer, options} shape so privileged_text() -- which needs the
        # correct option's TEXT, not its letter -- can index it.
        from datasets.mmlu_dataset import MMLUDataset
        rows = []
        # 'dev', NOT 'val'. cold_start_mmlu and finetune_mmlu build their graphs
        # from dev; evaluate_mmlu scores on val. Reading val here was wrong twice
        # over: the task-split indices are into dev, so records[i] returned a
        # DIFFERENT question than the graph at that index (no rollout query ever
        # matched a graph, and the pool came back empty) -- and had they matched,
        # OPSD would have rolled out on the evaluation set with the teacher
        # conditioned on its answers, which is test-set leakage through the
        # teacher.
        ds = MMLUDataset('dev')
        for i in range(len(ds)):
            rec = ds[i]
            rows.append({
                'task': MMLUDataset.record_to_input(rec)['task'],
                'answer': MMLUDataset.record_to_target_answer(rec),
                'options': [rec['A'], rec['B'], rec['C'], rec['D']],
            })
        return rows
    raise NotImplementedError(f"record loading for {dataset}")


def training_records(dataset, records):
    """Records OPSD is allowed to roll out on: the cold-start + finetune split.

    LEAKAGE: the rollout pool used to be every record in the dataset file, test
    split included. No task label is read at rollout time, but the *teacher* is
    conditioned on the gold rationale, so distilling over a test query shapes the
    student's weights with that query's answer. That is test-set leakage through
    the teacher, and it would invalidate the evaluation.

    Falls back to every record only when no split file exists, with a warning --
    a run whose numbers cannot be trusted should say so.
    """
    split_path = task_split_file(dataset)
    if not os.path.exists(split_path):
        print(f"[opsd] no task split at {split_path}; rolling out on ALL {len(records)} "
              f"records of {dataset}.")
        print(f"[opsd]   Fine IF {dataset} is a training-only corpus and you evaluate "
              f"on a DIFFERENT dataset (the MixChain -> gsm8k/aqua transfer setup).")
        print(f"[opsd]   NOT fine if you evaluate on {dataset} itself: the teacher "
              f"conditions on gold solutions, so its test split would leak.")
        return records

    with open(split_path, encoding='utf-8') as f:
        split = json.load(f)

    # cold_start_mmlu writes 'base_tasks'/'finetune_tasks'; every other cold
    # start writes the '_indices' spelling. Reading only one silently yields an
    # EMPTY allowed set, which looks like "this dataset has no training queries"
    # rather than "the key is spelled differently".
    def _idx(*names):
        for n in names:
            if n in split:
                return set(split[n])
        return set()

    base = _idx('base_tasks_indices', 'base_tasks')
    finetune = _idx('finetune_tasks_indices', 'finetune_tasks')
    allowed = sorted(base | finetune)
    if not allowed:
        raise RuntimeError(
            f"{split_path} defines no training queries. Keys present: "
            f"{sorted(split)}. Expected base_tasks_indices / finetune_tasks_indices "
            f"(or the older base_tasks / finetune_tasks).")

    test_key = 'test_indices' in split
    held_out = len(split.get('test_indices', []))
    kept = [records[i] for i in allowed if i < len(records)]
    print(f"[opsd] rollout pool restricted to {len(kept)} train-split queries "
          f"({held_out} test queries withheld)")
    if not test_key:
        # MMLU's split names only the training halves; everything after them is
        # test by omission. Excluding base+finetune still keeps the rollout pool
        # clear of it, but say so rather than reporting "0 withheld".
        print(f"[opsd]   no explicit test_indices in this split -- the {len(records)} "
              f"- {len(allowed)} records outside base+finetune are test by omission, "
              f"and are excluded from the pool.")
    return kept


def main():
    cli, tuned = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup(cli.seed, device)

    ckpt_path = os.path.join(cli.student_dir, cli.student_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No cold-start checkpoint at {ckpt_path}. Run stage 1 first "
            f"(finetune_{cli.dataset}.py --pretrain); OPSD is a post-training method.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = checkpoint['data_statistics']

    # Architecture and role mapping must come from the checkpoint -- the student
    # has to be rebuilt exactly as it was trained -- while the OPSD knobs come
    # from this invocation.
    args = Args()
    args.update_args_from_dict(checkpoint['args'])
    for key in OPSD_KEYS:
        if hasattr(tuned, key):
            setattr(args, key, getattr(tuned, key))
    # Defined on the CLI parser, not in Args(), so the OPSD_KEYS loop above --
    # which copies from the Args parse -- cannot see it. Set it explicitly, or
    # _teacher_forward falls back to the FiLM path and a plain ARGDesigner raises
    # "This model has no privileged channel".
    args.privileged_source = cli.privileged_source
    args.privileged_control = cli.privileged_control
    args.seed = cli.seed
    args.batch_size = cli.batch_size
    args.device = device
    args.dataset = cli.dataset
    args.dataset_name = cli.dataset
    args.data_dir = cli.data_dir or cold_start_dir(cli.dataset)
    args.feed_edge_features = bool(int(args.feed_edge_features))
    if cli.data_dir:
        print(f"[opsd] teacher trained on {args.data_dir} (not the default cold-start set)")

    out_dir = cli.output_dir or os.path.join(
        OUTPUT_ROOT, f"{cli.dataset}_opsd_seed{cli.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    ensure(out_dir)
    args.experiment_path = out_dir
    print(f"[opsd] outputs -> {out_dir}")

    # ---------------- cold-start graphs (D_exp) --------------------------- #
    # Before loading: say what is on disk and what the loader will refuse. The
    # pretrain filter keeps only filenames carrying a True/solved verdict, and
    # D_pruned is not written with one -- so a FinetuneData directory trains on
    # its DENSER half unless this is noticed.
    census(args.data_dir, label="Stage 1 teacher / anchor graphs")
    for _extra in (cli.extra_data_dir or []):
        census(_extra, label="merged graph set")
    graph_dataset, _ = gdata.load_graph_dataset(args)
    graphs = graph_dataset.graph_list
    if not graphs:
        raise RuntimeError(f"No cold-start graphs under {args.data_dir}")

    # Extra graph sets go through the SAME adapter, which is safe only because
    # every math dataset here shares one role vocabulary -- the adapter silently
    # remaps an unknown role to id 0, i.e. to the wrong agent, so the merge is
    # checked rather than assumed.
    for extra in (cli.extra_data_dir or []):
        if not os.path.isdir(extra):
            raise RuntimeError(f"--extra_data_dir {extra} is not a directory")
        primary_dir = args.data_dir
        try:
            args.data_dir = extra
            extra_dataset, _ = gdata.load_graph_dataset(args)
        finally:
            args.data_dir = primary_dir
        extra_graphs = extra_dataset.graph_list
        if not extra_graphs:
            raise RuntimeError(f"No cold-start graphs under {extra}")
        known = set(graph_dataset.role_to_id)
        unknown = {g.nodes[i]['role'] for g in extra_graphs for i in g.nodes()} - known
        if unknown:
            raise RuntimeError(
                f"{extra} uses role(s) {sorted(unknown)} that {cli.dataset}'s vocabulary "
                f"({sorted(known)}) does not define. Merging would map them onto the "
                f"wrong agent.")
        print(f"[opsd] merged {len(extra_graphs)} graphs from {extra}")
        graphs = graphs + extra_graphs
    questions = {g.graph.get('task') for g in graphs if g.graph.get('task')}
    print(f"[opsd] {len(graphs)} cold-start graphs over {len(questions)} distinct queries")

    # Shuffle before splitting: graphs arrive in glob order, so a positional
    # split hands validation a contiguous slice of record ids (and often of
    # topology blueprints) instead of a sample. The privileged go/no-go is read
    # off this split, so an unrepresentative one is not survivable.
    # Split by QUERY, not by graph. D_eff holds ~4 graphs per query, so a
    # graph-level split puts siblings of the same query on both sides: the
    # teacher trains on that query's rationale and is then "validated" on
    # another graph from the same query. The privileged channel can memorise
    # rationale -> topology for seen queries, so that measures recall, not
    # generalisation, and inflates the gain in exactly the direction we are
    # trying to test.
    by_query = {}
    for g in graphs:
        by_query.setdefault(g.graph.get('task') or id(g), []).append(g)
    keys = sorted(by_query, key=str)
    random.Random(cli.seed).shuffle(keys)

    n_val_q = max(1, int(round(len(keys) * cli.val_frac)))
    val_keys, train_keys = set(keys[:n_val_q]), keys[n_val_q:]
    train_graphs = [g for k in train_keys for g in by_query[k]]
    val_graphs = [g for k in val_keys for g in by_query[k]] or by_query[keys[0]]
    print(f"[opsd] split by query: {len(train_keys)} train / {len(val_keys)} val queries "
          f"({len(train_graphs)} / {len(val_graphs)} graphs); no query appears in both")
    if len(val_keys) < 20:
        print(f"[opsd] WARNING: only {len(val_keys)} validation queries. The privilege "
              f"gain's effective sample size is queries, not graphs -- expect a wide "
              f"error bar.")
    make_loader = lambda gs, shuffle: torch.utils.data.DataLoader(
        gdata.GraphListDataset(gs, args), batch_size=cli.batch_size,
        shuffle=shuffle, collate_fn=lambda x: x)
    loader_train, loader_val = make_loader(train_graphs, True), make_loader(val_graphs, False)

    # ---------------- privileged embeddings ------------------------------- #
    # Two DIFFERENT corpora need privileged vectors, and in the transfer setup
    # they are not the same dataset:
    #   Stage 1 looks them up by the COLD-START GRAPH's question (--data_dir,
    #           i.e. --teacher_dataset)
    #   Stage 2 looks them up by the ROLLOUT query          (--dataset)
    # A table built from only one of the two silently hands the other zero
    # vectors -- which is FiLM's identity, so Stage 1 degrades to "no privilege"
    # and reports a gain of noise, while Stage 2 degrades to an EMPTY pool and
    # trains on nothing. Both failures print healthy-looking logs. Encode the
    # union.
    records = load_records(cli.dataset)
    teacher_datasets = [d for d in (cli.teacher_dataset or []) if d != cli.dataset]

    def build_answers(kind: str, text_fn=None) -> AnswerEmbeddings:
        """One table covering both corpora.

        Each corpus is encoded with its OWN dataset name -- privileged_text reads
        a different field per dataset (gsm8k 'step', aqua 'rationale', mmlu the
        option text), so encoding gsm8k records under 'mixchain' would silently
        pick the wrong field for anything but the two that happen to agree.

        The cache lives in --data_dir, which is SHARED across datasets in the
        transfer setup, so the dataset name has to be in the filename: keyed by
        kind alone, a gsm8k table is silently reused for a mixchain run and every
        rollout question misses.
        """
        def one(recs, ds):
            path = os.path.join(args.data_dir, f"{kind}_embeddings_{ds}.pt")
            return AnswerEmbeddings(recs, ds, cache_path=path, text_fn=text_fn)

        table = one(records, cli.dataset)
        for ds in teacher_datasets:
            extra = one(load_records(ds), ds)
            if extra.dim and table.dim and extra.dim != table.dim:
                raise RuntimeError(
                    f"{cli.dataset} and {ds} privileged embeddings have different "
                    f"widths ({table.dim} vs {extra.dim}).")
            table.table.update(extra.table)
            table.dim = table.dim or extra.dim
            table.missing += extra.missing
        if teacher_datasets:
            print(f"[opsd] privileged table spans {cli.dataset} (rollouts) + "
                  f"{' + '.join(teacher_datasets)} (cold-start graphs): "
                  f"{len(table)} queries")
        return table

    if cli.privileged_source == 'context':
        # One model, two contexts. Cached separately from the rationale table:
        # the same key maps to a different vector here.
        answers = build_answers('context', text_fn=privileged_context_text)
        if len(answers) == 0:
            raise RuntimeError(
                f"No solution text found for {cli.dataset}, so the teacher's context "
                f"would equal the student's and the KL would be identically zero.")
    elif cli.privileged_source == 'rationale':
        answers = build_answers('privileged')
        if len(answers) == 0:
            raise RuntimeError(
                f"No privileged text found for {cli.dataset}. Without a rationale the "
                f"teacher is identical to the student and there is nothing to distil.")
    else:
        # Built from ALL graphs, not just the train split: _validate_teacher runs
        # the same lookup over the val graphs. The split is by query, so a val
        # query's siblings still come from within val.
        if cli.privileged_source == 'graph_stats':
            mode = 'stats'
            answers = GraphStatsEmbeddings(graphs, seed=cli.seed)
        else:
            mode = 'sibling' if cli.privileged_source == 'sibling_graph' else 'gold'
            answers = GraphPrivilegedEmbeddings(graphs, mode=mode, seed=cli.seed)
        if len(answers) == 0:
            raise RuntimeError(
                f"No graph carried a query string under {args.data_dir}, so the "
                f"topology channel has nothing to condition on.")
        if mode == 'sibling' and answers.no_sibling == len(answers):
            raise RuntimeError(
                "Every query has only one distinct topology, so no sibling exists "
                "for any graph and the teacher would equal the student. Use a graph "
                "set with several successful topologies per query (D_eff), or "
                "--privileged_source rationale.")

    # A graph whose question is absent from the table gets a ZERO vector, and
    # zero is exactly FiLM's identity -- so a total miss is indistinguishable
    # from "trained with no privilege at all", except that it still prints a
    # privilege gain (of noise). Say so before spending Stage 1 on it.
    if cli.privileged_source == 'rationale':
        covered = sum(1 for g in graphs if answers.get(g.graph.get('task')) is not None)
        pct = 100.0 * covered / max(len(graphs), 1)
        print(f"[opsd] Stage-1 privileged coverage: {covered}/{len(graphs)} "
              f"cold-start graphs ({pct:.1f}%)")
        if covered == 0:
            raise RuntimeError(
                f"None of the {len(graphs)} cold-start graphs in {args.data_dir} has a "
                f"privileged vector, so the teacher's FiLM channel would receive zeros "
                f"and Stage 1 could not learn anything from the rationale. Those graphs "
                f"are not from {cli.dataset}; pass --teacher_dataset <their dataset>.")
        if pct < 50:
            print(f"[opsd] WARNING: {100-pct:.1f}% of Stage-1 graphs fall back to a zero "
                  f"vector, i.e. train with no privilege. The gain is diluted accordingly.")

    # ---------------- rollout pool (built early, on purpose) --------------- #
    # Stage 1 costs ~40 epochs of teacher training. Every data-plumbing failure
    # this pipeline has produced -- a split whose indices address a different
    # split's records, graphs carrying no query text, a stale embedding cache --
    # shows up as an empty pool, and building it first turns an hour-late crash
    # into a two-second one.
    pool = build_query_pool(training_records(cli.dataset, records), cli.dataset,
                            answers, limit=args.opsd_query_limit)
    print(f"[opsd] rollout pool: {len(pool)} queries "
          f"(vs {len(graphs)} verified cold-start graphs)")
    if not pool:
        graph_qs = {g.graph.get('task') for g in graphs if g.graph.get('task')}
        sample = next(iter(graph_qs), None)
        nl = chr(10)
        raise RuntimeError(nl.join([
            f"Rollout pool is empty: no {cli.dataset} record matched a privileged "
            f"vector.",
            f"  records loaded        : {len(records)}",
            f"  graphs with a query   : {len(graph_qs)}",
            f"  privileged table size : {len(answers)}",
            f"  a graph's query text  : {str(sample)[:90]!r}",
            f"  a record's query text : {str(records[0].get('task'))[:90]!r}"
                if records else "  (no records)",
            "If those two texts are from different populations, load_records and the "
            "cold start are reading DIFFERENT dataset splits -- that is the usual "
            "cause, and it also means the task-split indices address the wrong rows.",
            f"If they look alike, delete {args.data_dir}/*_embeddings*.pt and re-run: "
            f"a cached table built from another dataset keys on questions that no "
            f"longer exist.",
        ]))

    if cli.dry_run:
        print()
        print("[opsd] --dry_run: data plumbing OK, stopping before Stage 1.")
        print(f"       {len(graphs)} graphs, {len(answers)} privileged vectors, "
              f"{len(pool)} rollout queries.")
        return

    # ---------------- stage 1: privileged teacher -------------------------- #
    # Constructed before Stage 1 because --privileged_source context makes the
    # teacher this very object.
    student = ARGDesigner(args, stats).to(device)
    student.load_state_dict(checkpoint['model_state_dict'])

    teacher_path = os.path.join(out_dir, 'teacher.pth')

    if cli.privileged_source == 'context':
        # No extra parameters and no Stage 1: the teacher differs from the student
        # only in what it is conditioned on (question + reference solution).
        #
        # It is a FROZEN SNAPSHOT of the pretrained model, not the live student.
        # Sharing the live weights admits a degenerate solution: the KL is also
        # minimised by making the model IGNORE its conditioning entirely, at
        # which point teacher and student agree, the loss goes to zero, and the
        # designer has got worse. A stationary target cannot be gamed that way --
        # the student has to move toward how the pretrained model behaves when it
        # can see the solution.
        teacher = copy.deepcopy(student)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_report = {"skipped": "privileged_source=context: frozen copy of the "
                                     "pretrained student, conditioned on question+solution"}
        print("=" * 30, "Stage 1 skipped (context mode)", "=" * 30)
        print("[opsd] teacher = frozen copy of the pretrained student,")
        print("       conditioned on question+solution. Student trains; teacher does not.")
        print("[opsd] No FiLM channel and no teacher finetune, so there is no")
        print("       privilege-gain go/no-go here. The check that replaces it is")
        print("       whether the KL is non-zero: teacher_kl in the OPSD history.")
        print("       A KL that is ~0 means the longer context did not change the")
        print("       conditioning vector enough to matter, and nothing will transfer.")
    elif cli.skip_teacher and os.path.exists(teacher_path):
        teacher = build_teacher(args, stats, checkpoint['model_state_dict'], device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device,
                                           weights_only=False)['model_state_dict'])
        print("[opsd] reusing existing teacher")
        teacher_report = {"reused": True}
    else:
        teacher = build_teacher(args, stats, checkpoint['model_state_dict'], device)
        print("=" * 30, "Stage 1: privileged teacher", "=" * 30)
        teacher_report = train_teacher(args, teacher, loader_train,
                                       answers.as_lookup(cli.privileged_control, cli.seed),
                                       loader_val)
        torch.save({'model_state_dict': teacher.state_dict(), 'data_statistics': stats,
                    'args': serialisable_args(args)}, teacher_path)

    last = teacher_report.get('history', [{}])[-1] if teacher_report.get('history') else {}
    gain, se, n = last.get('privilege_gain'), last.get('privilege_gain_stderr'), last.get('privilege_n')
    if gain is not None:
        sigma = abs(gain) / se if se and se > 0 else float('nan')
        print(f"[opsd] GO/NO-GO -- privilege gain {gain:+.4f} +/- {se:.4f} nats "
              f"over {n} validation graphs ({sigma:.2f} sigma)")
        if not (sigma == sigma) or sigma < 2:
            print("       Below 2 sigma: this measurement cannot tell you whether the "
                  "rationale helps.")
            print("       Enlarge the cold-start set before reading a sign into it.")
        elif gain < 0:
            print("       Significantly NEGATIVE: the channel is hurting. Reduce "
                  "--privileged_rank,")
            print("       --teacher_epochs or --teacher_lr before distilling.")

    # ---------------- stage 2: on-policy self-distillation ----------------- #
    print("=" * 30, "Stage 2: OPSD", "=" * 30)

    # pool was built before Stage 1; see the note there.
    report = train_opsd(args, student, teacher, pool, anchor_loader=loader_train,
                        log_path=os.path.join(out_dir, 'opsd_history.json'))

    torch.save({'model_state_dict': student.state_dict(),
                'data_statistics': stats, 'args': serialisable_args(args)},
               os.path.join(out_dir, 'best_model.pth'))

    summary = {'dataset': cli.dataset, 'seed': cli.seed,
               'student_init': ckpt_path,
               'cold_start_graphs': len(graphs),
               'rollout_queries': len(pool),
               'teacher': teacher_report, 'opsd': report}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[opsd] done -> {out_dir}")


if __name__ == '__main__':
    main()
