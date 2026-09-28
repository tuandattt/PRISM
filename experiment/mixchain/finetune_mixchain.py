import os
import json
import torch
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import asyncio
import copy
import sys
import shutil
import time
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.args import Args
from experiment.model import ARGDesigner
from experiment.paths import (cold_start_dir, finetune_dir, task_split_file, dataset_file,
                             run_dir, ensure, OUTPUT_ROOT)
from experiment.train_ARGDesigner import train
from experiment.utils import save_graph_with_features, get_kwargs, load_model, generate_graph, convert_to_pyg_graph
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata
from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from datasets.mixchain_dataset import (mixchain_data_process, mixchain_get_predict,
                                       mixchain_is_correct)
from experiment.graph_census import census
from experiment.mixchain.mixchain_prompt_set import ROLE_DESCRIPTION

FINETUNE_DATA_DIR = finetune_dir("mixchain")
COLD_START_DIR = cold_start_dir("mixchain")
TASK_SPLIT_FILE = task_split_file("mixchain")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline for MixChain: pretrain, generate data, finetune")
    parser.add_argument('--pretrain', action='store_true', help="Run pretraining if set")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model from")

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help="Learning rate for pretraining")

    parser.add_argument('--pruning_ratio', type=float, default=0.25, help="Pruning ratio for data generation")
    parser.add_argument('--replay_ratio', type=float, default=0.3, help="Replay ratio from cold-start data")

    parser.add_argument('--finetune_epochs', type=int, default=200, help="Number of finetuning epochs")
    parser.add_argument('--finetune_lr', type=float, default=5e-5, help="Learning rate for finetuning")

    parser.add_argument('--llm_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument('--domain', type=str, default="mixchain", help="Task domain")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='List of agent names')
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and data gen")
    parser.add_argument('--node_timeout', type=int, default=120,
                        help="Seconds one agent gets before it is abandoned "
                             "(TestGraph.arun max_time, default 600 upstream). The "
                             "budgets MULTIPLY: node_tries x (LLM_MAX_RETRIES x "
                             "LLM_TIMEOUT) per node, x nodes per query. At the "
                             "upstream defaults a single stuck Layered-4 query can "
                             "hold its batch for ~45 minutes.")
    parser.add_argument('--query_timeout', type=int, default=300,
                        help="Wall-clock cap on ONE query, end to end. --node_timeout "
                             "bounds a single agent, but a query runs its agents "
                             "SEQUENTIALLY, so the real bound was node_timeout x depth "
                             "-- 20 minutes for Chain-4 at 240s. Since the batch cannot "
                             "close until its slowest query returns, that one query "
                             "stalls 63 others. This caps the whole query instead.")
    parser.add_argument('--node_tries', type=int, default=1,
                        help="Attempts per agent. tenacity already retries inside "
                             "achat, so retrying here multiplies the same wait -- and "
                             "a timeout is usually deterministic, so the retries buy "
                             "nothing but wall clock.")
    parser.add_argument('--drop_degraded', action='store_true',
                        help="Discard a run in which some agent never produced "
                             "output (timeout / connection error), instead of "
                             "labelling it. OFF by default, which is the original "
                             "behaviour: such a run is kept if its final answer is "
                             "correct. The count is reported either way, because "
                             "the graphs themselves record nothing about it -- a "
                             "kept graph claims N agents contributed when only "
                             "N-1 ran, and dense topologies survive a lost node "
                             "far more often, so keeping them skews the set toward "
                             "density.")
    parser.add_argument('--no_resume', dest='resume', action='store_false',
                        help="Re-run every query in stages 2a/2b even if the ledger in "
                             "the finetune data directory says it was already attempted. "
                             "Resume is ON by default: these stages are hours of LLM "
                             "calls and the ledger is the only thing that makes an "
                             "interrupted run cheap to restart.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_task', 'no_history', 'random_first_node'],
                        help="Table-4 ablation of the task/history conditioning. Also tags "
                             "the output directory. Default: full model.")
    parser.add_argument('--cold_start_dir', type=str, default=None,
                        help="Graph set to pretrain on, replacing the default "
                             "rejection-sampled cold-start directory. Point it at an "
                             "imported AgentDropout set to train the designer on chosen "
                             "topologies instead of sampled ones; D_replay is drawn from "
                             "here too, so the whole pipeline moves together.")
    parser.add_argument('--model_output_dir', type=str, default=os.path.join(OUTPUT_ROOT, 'mixchain'),
                        help="Root directory for model outputs")
    return parser.parse_args()


def check_timeout_budgets(args):
    """The node timeout must exceed the LLM call's own retry budget.

    Two independent timers guard the same request:

        inner   tenacity in achat -- LLM_MAX_RETRIES attempts, each capped at
                LLM_TIMEOUT seconds, plus exponential backoff between them
        outer   asyncio.wait_for(max_time) around the whole node

    If outer < inner, the node is killed the moment a single call needs one
    retry -- the retry can never complete, so the inner budget is dead code and
    every transient blip costs a graph. The failure looks like a server problem
    ("TimeoutError") when it is purely a misconfiguration.
    """
    from mas_framework.llm.gpt_chat import MAX_RETRIES, REQUEST_TIMEOUT
    inner = REQUEST_TIMEOUT * MAX_RETRIES
    print(f"[mixchain] timeouts: node {args.node_timeout}s vs LLM budget "
          f"{inner:.0f}s ({MAX_RETRIES} x {REQUEST_TIMEOUT:.0f}s)")
    depth_hint = 5  # Chain-4 / Layered-4: agents run sequentially, plus FinalRefer
    print(f"[mixchain] query cap {args.query_timeout}s; worst case without it would be "
          f"~{args.node_timeout * depth_hint}s (node_timeout x depth)")
    if args.query_timeout < args.node_timeout:
        print(f"[mixchain] WARNING: --query_timeout {args.query_timeout}s is below "
              f"--node_timeout {args.node_timeout}s, so no node can ever finish.")
    if args.node_timeout < inner:
        print(f"[mixchain] WARNING: --node_timeout {args.node_timeout}s is BELOW the "
              f"LLM retry budget of {inner:.0f}s.")
        print(f"           Any call needing a retry will be killed before it can "
              f"finish, and that query is lost.")
        print(f"           Either raise --node_timeout to >= {int(inner * 1.5)}, or "
              f"lower the budget:")
        print(f"           export LLM_TIMEOUT={max(30, int(args.node_timeout / (MAX_RETRIES * 1.5)))} "
              f"LLM_MAX_RETRIES={MAX_RETRIES}")


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_unique_simple_configs():
    configs = set()
    for agent_num in range(2, 5):
        if agent_num == 2:
            configs.add(('Chain', 2))
        elif agent_num == 3:
            configs.add(('Chain', 3))
            configs.add(('Star', 3))
        else:
            configs.add(('Chain', agent_num))
            configs.add(('Star', agent_num))
            configs.add(('Layered', agent_num))
    return list(configs)


def apply_cold_start_override(args):
    """Repoint COLD_START_DIR before any stage reads it.

    Four call sites use the module-level constant -- pretraining's data_dir, the
    D_replay copy, and finetuning's data_dir -- so rebinding it once is what keeps
    them consistent. Overriding only one would train on AgentDropout graphs and
    replay rejection-sampled ones.
    """
    global COLD_START_DIR
    if not args.cold_start_dir:
        return
    if not os.path.isdir(args.cold_start_dir):
        raise SystemExit(f"--cold_start_dir {args.cold_start_dir} is not a directory")
    n = len([f for f in os.listdir(args.cold_start_dir)
             if f.endswith('.pt') and ('True' in f or 'solved' in f.lower())])
    if n == 0:
        raise SystemExit(
            f"No usable graphs in {args.cold_start_dir}. The loader keeps only files "
            f"whose name contains 'True' or 'solved'; import_agentdropout.py writes "
            f"that naming automatically.")
    COLD_START_DIR = args.cold_start_dir
    print(f"[mixchain] cold-start graphs overridden -> {COLD_START_DIR} ({n} graphs)")


def run_pretraining(args):
    print("\n" + "=" * 20 + " Stage 1: Pretraining MixChain Model " + "=" * 20)
    pre = Args().update_args()
    pre.dataset = 'mixchain'
    pre.data_dir = COLD_START_DIR
    pre.experiment_path = args.model_output_dir
    pre.pretrain = args.pretrain
    pre.epochs = args.pretrain_epochs
    pre.lr = args.pretrain_lr
    pre.batch_size = args.batch_size
    pre.seed = args.seed
    pre.model_name = 'best_model.pth'

    ds, _ = gdata.load_graph_dataset(pre)

    role_to_id, id_to_role = ds.role_to_id, ds.id_to_role
    num_node_types = len(role_to_id) + 2
    pre.role_mapping = role_to_id
    pre.id_to_role = id_to_role
    pre.START_TOKEN_ID = len(role_to_id)
    pre.END_TOKEN_ID = len(role_to_id) + 1

    stats = gdata.get_data_statistics(ds.graph_list)
    stats['num_node_labels'] = num_node_types
    stats['num_edge_labels'] = 1

    correct = [g for g in ds if g.graph.get('is_correct')]
    incorrect = [g for g in ds if not g.graph.get('is_correct')]
    random.shuffle(correct)
    random.shuffle(incorrect)

    ratio = 0.9
    train_graphs = correct[:int(len(correct) * ratio)] + incorrect[:int(len(incorrect) * ratio)]
    val_graphs = correct[int(len(correct) * ratio):] + incorrect[int(len(incorrect) * ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    dt = gdata.GraphListDataset(train_graphs, pre)
    dv = gdata.GraphListDataset(val_graphs, pre)

    lt = torch.utils.data.DataLoader(dt, batch_size=pre.batch_size, shuffle=True,
                                     drop_last=True, num_workers=pre.num_workers, collate_fn=lambda x: x)
    lv = torch.utils.data.DataLoader(dv, batch_size=pre.batch_size, shuffle=False,
                                     drop_last=False, num_workers=pre.num_workers, collate_fn=lambda x: x)

    with open(os.path.join(pre.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pre.__dict__, f, indent=2)

    pre.ablation = getattr(args, 'ablation', None)
    model = ARGDesigner(pre, stats).to(pre.device)
    model.ablation_variant = pre.ablation
    train(pre, model, lt, lv)

    print("=" * 20 + " Stage 1 Complete " + "=" * 20)
    return pre.experiment_path, os.path.join(pre.experiment_path, 'configuration.txt')


def apply_efficiency_strategy(graphs, strategy='prune', **kwargs):
    out = []
    for g in graphs:
        if strategy == 'prune':
            r = kwargs.get('pruning_ratio', 0.2)
            n = int(g.number_of_edges() * r)
            if n > 0:
                es = list(g.edges())
                random.shuffle(es)
                g.remove_edges_from(es[:n])
                if not nx.is_weakly_connected(g):
                    comps = list(nx.weakly_connected_components(g))
                    main = max(comps, key=len)
                    for c in comps:
                        if c != main:
                            g.add_edge(random.choice(list(c)), random.choice(list(main)))
        out.append(g)
    return out


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
LEDGER_NAME = 'attempts.jsonl'


def _ledger_path(od):
    return os.path.join(od, LEDGER_NAME)


def load_ledger(od):
    """Which (stage, mode, num, record) triples have already been executed.

    Existence of an output .pt is NOT a usable resume key: only SOLVED queries
    produce a file, so resuming off the directory would re-run every query the
    LLM failed -- which is most of them, and the expensive half. The ledger
    records every attempt, solved or not.
    """
    done = set()
    path = _ledger_path(od)
    if not os.path.exists(path):
        return done
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    print(f"[resume] {len(done)} queries already attempted (ledger: {path})")
    return done


def ledger_key(stage, mode, num, idx):
    return f"{stage}|{mode}|{num}|{idx}"


def record_attempts(od, keys):
    """Append after a batch RESOLVES, never before.

    Written post-hoc so a crash mid-batch re-runs that batch rather than
    skipping it -- redoing one batch is cheap, losing queries silently is not.
    """
    if not keys:
        return
    os.makedirs(od, exist_ok=True)
    with open(_ledger_path(od), 'a', encoding='utf-8') as f:
        for k in keys:
            f.write(k + "\n")


async def generate_pruned_data(args, ds, mp, od):
    print("\n" + "=" * 20 + " Stage 2a: MixChain Pruned Data " + "=" * 20)
    model = load_model(mp)
    model.eval()
    sm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    roles = ROLE_DESCRIPTION

    done = load_ledger(od) if args.resume else set()
    recs = [(idx, rec) for idx, rec in enumerate(ds)
            if ledger_key('pruned', 'prune', 0, idx) not in done]
    count = 0
    total = len(recs)
    if total < len(ds):
        print(f"[resume] skipping {len(ds) - total} already-attempted queries")

    with tqdm(total=total, desc="Pruning") as p:
        for i in range(0, total, args.batch_size):
            batch = recs[i:i+args.batch_size]
            tasks = []
            for idx, rec in batch:
                t = rec["task"]
                emb = torch.tensor(sm.encode(t), device=model.args.device).float()
                gens = generate_graph(model, emb, roles, idx)
                effs = apply_efficiency_strategy(gens, 'prune', pruning_ratio=args.pruning_ratio)
                if effs:
                    e = effs[0]
                    pd = convert_to_pyg_graph(e, t)
                    tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                                   decision_method=args.decision_method, pyg_data=pd)
                    metadata = {"pyg_data": pd, "record": rec, "task_text": t,
                                "record_idx": idx, "executed": tg}
                    tasks.append((asyncio.wait_for(
                        tg.arun({"task": t}, num_rounds=1,
                                max_tries=args.node_tries,
                                max_time=args.node_timeout),
                        timeout=args.query_timeout), metadata))
            if not tasks:
                continue
            res = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
            for out, metadata in zip(res, tasks):
                p.update(1)
                if isinstance(out, Exception):
                    continue
                ans = out[0] if isinstance(out, (list, tuple)) and out else out
                pred = mixchain_get_predict(ans)
                true = metadata[1]['record']["answer"]
                if (getattr(metadata[1].get('executed'), 'node_failures', [])
                        and getattr(args, 'drop_degraded', False)):
                    continue          # see --drop_degraded
                if mixchain_is_correct(pred, true):
                    count += 1
                    name = f"eff_q{metadata[1]['record_idx']}_g0.pt"
                    torch.save(metadata[1]['pyg_data'], os.path.join(od, name))
            record_attempts(od, [ledger_key('pruned', 'prune', 0, m[1]['record_idx'])
                                 for m in tasks])
    print(f"Stage 2a done, {count} graphs saved")


async def generate_simple_data(args, ds, od):
    print("\n" + "=" * 20 + " Stage 2b: MixChain Simple Data " + "=" * 20)
    configs = get_unique_simple_configs()
    for mode, num in configs:
        print(f"Config {mode}-{num}")
        kw = get_kwargs(mode, num)
        roles = random.choices(list(ROLE_DESCRIPTION.keys()), k=num)
        kw['node_kwargs'] = [{'role': r} for r in roles]
        g = Graph(domain=args.domain, llm_name=args.llm_name,
                  agent_names=[args.agent_names[0]]*num,
                  decision_method=args.decision_method, **kw)
        await evaluate_and_save_mixchain_simple(g, ds, args, mode, num, od)
    print("Stage 2b done")


def generate_replay_data(args, od):
    print("\n" + "=" * 20 + " Stage 2c: MixChain Replay Data " + "=" * 20)
    files = [f for f in os.listdir(COLD_START_DIR) if f.endswith(".pt") and ("True" in f or "solved" in f.lower())]
    N = min(int(len(files)*args.replay_ratio), len(files))
    pick = random.sample(files, N)
    for fn in tqdm(pick, desc="Copying"):
        shutil.copy(os.path.join(COLD_START_DIR, fn), os.path.join(od, fn))
    print("Stage 2c done")


async def evaluate_and_save_mixchain_simple(graph, ds, args, mode, num, od):
    total_solved = 0
    degraded = 0          # solved-or-not, but an agent never ran: not labellable
    done = load_ledger(od) if getattr(args, 'resume', True) else set()
    pending = [(i, r) for i, r in enumerate(ds)
               if ledger_key('simple', mode, num, i) not in done]
    if not pending:
        print(f"[resume] {mode}-{num} already complete, skipping")
        return
    if len(pending) < len(ds):
        print(f"[resume] {mode}-{num}: {len(ds) - len(pending)} of {len(ds)} done, "
              f"{len(pending)} to go")
    # Cap the batch at the concurrency limit. asyncio.gather is a barrier, and
    # the semaphore in gpt_chat caps in-flight requests anyway -- so a batch
    # larger than that runs no faster, it just makes the barrier's tail longer
    # and the progress bar coarser. A batch of 400 queries at 5 sequential calls
    # each is ~2000 LLM calls before tqdm ticks once, which is indistinguishable
    # from a hang.
    from mas_framework.llm.gpt_chat import MAX_CONCURRENCY
    bs = max(1, min(args.batch_size, MAX_CONCURRENCY))
    if bs != args.batch_size:
        print(f"[mixchain] batch {args.batch_size} -> {bs} (LLM_MAX_CONCURRENCY)")
    num_batches = math.ceil(len(pending) / bs)
    print(f"[mixchain] {mode}-{num}: {len(pending)} queries in {num_batches} "
          f"batches of {bs}")

    # Progress is per QUERY, not per batch: with a coarse bar a long batch looks
    # frozen, which is what sent the last two debugging sessions after phantom
    # server failures.
    pbar = tqdm(total=len(pending), desc=f"Processing {mode}-{num}", unit="q")
    for i_batch in range(num_batches):
        record_batch_with_indices = pending[i_batch * bs: (i_batch + 1) * bs]
        if not record_batch_with_indices:
            continue

        tasks = []
        for record_idx, record in record_batch_with_indices:
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=flow_graph)
            metadata = {"record": record, "flow_graph": flow_graph,
                        "question": input_dict['task'], "record_idx": record_idx,
                        "executed": tg}
            tasks.append((asyncio.wait_for(
                tg.arun(input_dict, args.num_rounds,
                        max_tries=args.node_tries,
                        max_time=args.node_timeout),
                timeout=args.query_timeout), metadata))

        # as_completed, not gather: gather returns only when the SLOWEST query in
        # the batch finishes, so the bar sat still for minutes and then jumped by
        # a whole batch -- which reads as a hang. Results are handled as they
        # arrive; the batch is still a barrier for the ledger write below, which
        # is what makes a crash re-run the batch instead of skipping it.
        async def _settled(i, coro):
            try:
                return i, await coro
            except Exception as exc:          # noqa: BLE001 - recorded, not raised
                return i, exc

        futures = [asyncio.ensure_future(_settled(i, t[0])) for i, t in enumerate(tasks)]
        for fut in asyncio.as_completed(futures):
            i, result = await fut
            pbar.update(1)
            if isinstance(result, Exception):
                continue

            metadata = tasks[i][1]
            raw_answer = result[0] if isinstance(result, (list, tuple)) and result else result
            predict_answer = mixchain_get_predict(raw_answer)
            true_answer = metadata['record']["answer"]
            is_solved = False
            try:
                is_solved = mixchain_is_correct(predict_answer, true_answer)
            except (ValueError, TypeError):
                pass

            # A graph that solved the query DESPITE losing an agent is not
            # evidence that this topology works -- only N-1 of its N agents ran,
            # so saving it teaches the designer to credit a seat that
            # contributed nothing. Denser graphs survive a lost node more often,
            # so keeping these would bias the training set toward density.
            failed = getattr(metadata.get('executed'), 'node_failures', []) or []
            if failed:
                degraded += 1
                if getattr(args, 'drop_degraded', False):
                    continue

            if is_solved:
                total_solved += 1
                name = "_".join(map(str, ['mixchain', metadata['record_idx'], mode, num, 'True']))
                filepath = os.path.join(od, f'{name}.pt')
                save_graph_with_features(metadata['flow_graph'], filepath, {
                    "mode": mode, "agent_nums": num, "is_correct": is_solved, "question": metadata['question']
                })

        record_attempts(od, [ledger_key('simple', mode, num, t[1]['record_idx'])
                             for t in tasks])
        pbar.set_postfix({"solved": total_solved})

    pbar.close()
    verb = "discarded" if getattr(args, 'drop_degraded', False) else "KEPT"
    print(f"Config {mode}-{num} done, {total_solved}/{len(pending)} solved"
          + (f", {degraded} {verb} (an agent never ran)" if degraded else ""))


def run_finetuning(model_path, config_path, finetune_data_dir, args):
    print("\n" + "=" * 20 + " Stage 3: MixChain Finetuning " + "=" * 20)
    finetune_args = Args()
    finetune_args = finetune_args.update_args()
    finetune_args.ablation = args.ablation
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    for key, value in config_data.items():
        if key not in ['dataset']:
            setattr(finetune_args, key, value)
    finetune_args.dataset = 'mixchain'
    finetune_args.data_dir_ef = finetune_data_dir
    finetune_args.data_dir = COLD_START_DIR
    finetune_args.experiment_path = args.model_output_dir
    finetune_args.lr = args.finetune_lr
    finetune_args.epochs = args.finetune_epochs
    finetune_args.pretrain = False
    finetune_args.model_name = 'ef_best_model.pth'
    
    efficient_dataset, _ = gdata.load_graph_dataset(finetune_args, pretrain=False)
    role_to_id = efficient_dataset.role_to_id
    num_node_types = len(role_to_id) + 2
    if not efficient_dataset.graph_list:
        print("Warning: No finetune data found, skipping finetuning.")
        return

    data_statistics = gdata.get_data_statistics(efficient_dataset.graph_list)
    data_statistics['num_node_labels'] = num_node_types
    data_statistics['num_edge_labels'] = 1
    dataset_finetune = gdata.GraphListDataset(efficient_dataset.graph_list, finetune_args)
    dataloader_finetune = torch.utils.data.DataLoader(dataset_finetune, batch_size=finetune_args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    model = ARGDesigner(finetune_args, data_statistics).to(finetune_args.device)
    model.ablation_variant = getattr(finetune_args, 'ablation', None)
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained weights, starting finetuning...")
    train(finetune_args, model, dataloader_finetune)
    print("=" * 20 + " Stage 3: MixChain Finetuning Complete " + "=" * 20)


def find_latest_model_dir(bd='output', pf='mixchain_finetuned_model'):
    if not os.path.exists(bd): return None
    cds = [d for d in os.listdir(bd) if os.path.isdir(os.path.join(bd, d)) and d.startswith(pf)]
    return os.path.join(bd, sorted(cds)[-1]) if cds else None


async def main():
    args = parse_cli_args()
    setup_environment(args)
    check_timeout_budgets(args)
    apply_cold_start_override(args)
    loop = asyncio.get_running_loop()

    if args.pretrain:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if args.ablation:
            args.model_output_dir = f"{args.model_output_dir}_{args.ablation}_{timestamp}"
        else:
            args.model_output_dir = f"{args.model_output_dir}_{timestamp}"

        os.makedirs(args.model_output_dir, exist_ok=True)
        print(f"Starting new MixChain run, outputs to: {args.model_output_dir}")
        pretrained_model_path, config_file_path = await loop.run_in_executor(None, run_pretraining, args)
    else:
        model_dir_to_load = args.load_from_dir or find_latest_model_dir()
        if not model_dir_to_load or not os.path.exists(model_dir_to_load):
            # Say which path failed and what is actually there. The fallback
            # searches for 'mixchain_finetuned_model*', but --pretrain writes
            # '<--model_output_dir>_<timestamp>', so the two only agree when
            # --model_output_dir happens to carry that prefix -- and the bare
            # message gave no way to notice that.
            root = 'output'
            found = sorted(d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d)) and 'mixchain' in d
                           ) if os.path.isdir(root) else []
            tried = args.load_from_dir or "mixchain_finetuned_model* (auto-discovery)"
            raise FileNotFoundError("\n".join([
                f"No model directory found at {tried!r}.",
                f"  MixChain directories under {root}/: {found or 'none'}",
                "  Pass one of them to --load_from_dir, or re-run with --pretrain.",
            ]))
        print(f"Loading model from existing directory: {model_dir_to_load}")
        args.model_output_dir = model_dir_to_load
        pretrained_model_path = os.path.join(model_dir_to_load)
        config_file_path = os.path.join(model_dir_to_load, "configuration.txt")
        if not os.path.exists(pretrained_model_path) or not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Missing best_model.pth or configuration.txt in directory {model_dir_to_load}.")

    finetune_data_output_dir = FINETUNE_DATA_DIR
    os.makedirs(finetune_data_output_dir, exist_ok=True)

    if not os.path.exists(TASK_SPLIT_FILE):
        raise FileNotFoundError(f"Task split file '{TASK_SPLIT_FILE}' not found. Please run cold_start_mixchain.py first.")
    with open(TASK_SPLIT_FILE, 'r') as f:
        task_split = json.load(f)
    
    full_dataset = mixchain_data_process(JSONLReader.parse_file(dataset_file("mixchain")))
    finetune_dataset_subset = [full_dataset[i] for i in task_split['finetune_tasks_indices']]
    print(f"\nLoaded {len(finetune_dataset_subset)} tasks (from training set) for MixChain finetune data generation.")

    await generate_pruned_data(args, finetune_dataset_subset, pretrained_model_path, finetune_data_output_dir)
    census(finetune_data_output_dir, label="after D_pruned")
    await generate_simple_data(args, finetune_dataset_subset, finetune_data_output_dir)
    census(finetune_data_output_dir, label="after D_simple")
    generate_replay_data(args, finetune_data_output_dir)
    census(finetune_data_output_dir, label="after D_replay -- final finetune set")

    await loop.run_in_executor(None, run_finetuning, pretrained_model_path, config_file_path, finetune_data_output_dir, args)
    print(f"\nMixChain pipeline complete! Final efficient model at: {os.path.join(args.model_output_dir)}")


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
