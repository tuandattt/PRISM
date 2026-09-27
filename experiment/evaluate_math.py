"""Evaluate one topology designer on any math benchmark.

    python experiment/evaluate_math.py --model_path output/mixchain_opsd_seed42 \
        --target gsm8k --ef 0 --seed 42 --llm_name Qwen/Qwen2.5-7B-Instruct

The per-dataset evaluate_*.py scripts assume the designer was trained on their own
dataset -- they hard-wire that dataset's records, task split and answer parser. A
designer trained on MixChain and tested on GSM8K/AQuA needs the two decoupled, so
this takes ``--model_path`` and ``--target`` independently.

This works only because every math benchmark here shares one role vocabulary
(Math Solver, Mathematical Analyst, Programming Expert, Inspector) -- MixChain's
prompt set re-exports GSM8K's for exactly this reason. The sampler emits role
NAMES from the checkpoint's own id_to_role, and the target dataset supplies the
matching prompts. A target with a different vocabulary (mmlu, humaneval) is
rejected rather than silently mapped onto the wrong agents.

Test indices come from each target's existing task_split_<ds>.json, so numbers
are directly comparable with the per-dataset runs already in res_logs/.
"""
import argparse
import asyncio
import datetime
import json
import math
import os
import sys
from typing import Any, Iterator, List

import torch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mas_framework.graph.graph import TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from mas_framework.utils.globals import Cost, CompletionTokens, PromptTokens
from experiment.paths import dataset_file, task_split_file
from experiment.utils import (convert_to_pyg_graph, encode_text, generate_graph,
                              load_model)
from experiment.model import repair_report, reset_repair_stats, REPAIR_STATS

# domain     -> execution domain, answer parser, whether a task split exists
TARGETS = {
    'gsm8k':      dict(domain='gsm8k', split=True),
    'aqua':       dict(domain='aqua',  split=True),
    # MultiArith and SVAMP have numeric answers and no pipeline of their own, so
    # they borrow gsm8k's agent prompts and parser wholesale.
    'multiarith': dict(domain='gsm8k', split=False),
    'svamp':      dict(domain='gsm8k', split=False),
}


def load_target(target: str):
    """Target records plus the answer parser to grade them with."""
    path = dataset_file(target)
    if target == 'gsm8k':
        from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
        return gsm_data_process(JSONLReader.parse_file(path)), gsm_get_predict
    if target == 'aqua':
        from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
        return aqua_data_process(JSONLReader.parse_file(path)), aqua_get_predict
    from datasets.gsm8k_dataset import (gsm_get_predict, multiarith_data_process,
                                        svamp_data_process)
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    fn = multiarith_data_process if target == 'multiarith' else svamp_data_process
    return fn(raw), gsm_get_predict


def role_descriptions(target: str):
    if target == 'aqua':
        from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
    else:
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
    return ROLE_DESCRIPTION


def graded(pred, gold, target: str) -> bool:
    if target == 'aqua':
        return str(pred).strip().upper() == str(gold).strip().upper()
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset evaluation for math designers")
    p.add_argument('--model_path', type=str, required=True,
                   help="Designer directory; may have been trained on ANY math corpus")
    p.add_argument('--target', type=str, required=True, choices=sorted(TARGETS),
                   help="Benchmark to evaluate on")
    p.add_argument('--ef', type=int, default=0, choices=[0, 1],
                   help="1: ef_best_model.pth (Stage-2 finetuned). 0: best_model.pth")
    p.add_argument('--seed', type=int, default=42, help="Graph sampling is stochastic")
    p.add_argument('--llm_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument('--decision_method', type=str, default="FinalRefer")
    p.add_argument('--eval_batch_size', type=int, default=48)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--full_test', action='store_true',
                   help="Ignore the task split and evaluate every record. Valid only "
                        "for a designer that never saw this dataset -- a designer "
                        "cold-started on it would be scored on its own training "
                        "questions. Writes to a '_full' detail file so split-based "
                        "runs are not overwritten.")
    p.add_argument('--output_file', type=str, default=None)
    p.add_argument('--summary_log_file', type=str,
                   default='./res_logs/evaluation_summary.jsonl')
    return p.parse_args()


def batches(data: List[Any], n: int) -> Iterator[List[Any]]:
    for i in range(0, len(data), n):
        yield data[i:i + n]


async def main():
    args = parse_args()
    cfg = TARGETS[args.target]
    ef = bool(args.ef)

    tag = os.path.basename(os.path.normpath(args.model_path)) or 'model'
    if not args.output_file:
        args.output_file = os.path.join(
            'res_logs', 'details',
            f"xfer_{args.target}_{tag}_{'ef' if ef else 'base'}_seed{args.seed}"
            f"{'_full' if args.full_test else ''}.jsonl")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    print(f"Per-question results -> {args.output_file}")

    import random
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    reset_repair_stats()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    model = load_model(args.model_path, ef=ef)
    model.eval()
    print(f"Designer trained on : {getattr(model.args, 'dataset', '?')}")
    print(f"Evaluating on       : {args.target}")

    roles = role_descriptions(args.target)
    # A mismatch here would silently pair generated role names with absent
    # prompts, producing empty constraints rather than an error.
    unknown = [r for r in model.id_to_role.values() if r not in roles]
    if unknown:
        raise SystemExit(
            f"Role vocabulary mismatch: the designer can emit {unknown}, which "
            f"{args.target} does not define. Known: {sorted(roles)}. Cross-dataset "
            f"evaluation requires a shared vocabulary.")

    records, get_predict = load_target(args.target)
    if cfg['split'] and args.full_test:
        # Task ids stay the record's index in both modes, so a full-test detail
        # file pairs with a split-based one on task_id; the split set is simply a
        # subset of the ids here.
        print(f"--full_test: ignoring {task_split_file(args.target)}; the "
              f"cold-start questions of this dataset are INCLUDED. Only use this "
              f"for a designer trained on a different corpus.")
    if cfg['split'] and not args.full_test:
        split_path = task_split_file(args.target)
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"{split_path} not found; run cold_start_{args.target}.py first so "
                f"the test set matches the per-dataset runs.")
        with open(split_path, encoding='utf-8') as f:
            idx = json.load(f)['test_indices']
        dataset = [records[i] for i in idx]
        ids = [f"task_{i}" for i in idx]
    else:
        dataset = records
        ids = [f"task_{i}" for i in range(len(records))]
    if args.limit:
        dataset, ids = dataset[:args.limit], ids[:args.limit]
    # Level travels with the record, not the graph, so it is looked up by id when
    # each result is written.
    meta_level = {i: str(r.get('level', '')) for i, r in zip(ids, dataset)}
    print(f"Loaded {len(dataset)} {args.target} test samples.")

    solved, results = 0, []
    total = len(dataset)
    pbar = tqdm(enumerate(batches(list(zip(ids, dataset)), args.eval_batch_size)),
                total=math.ceil(total / args.eval_batch_size), desc=f"eval:{args.target}")

    for i_batch, batch in pbar:
        coros, meta = [], []
        for offset, (task_id, record) in enumerate(batch):
            text = record["task"]
            try:
                emb = torch.tensor(encode_text(text), device=model.args.device).float()
                graphs = generate_graph(model, emb, roles,
                                        i_batch * args.eval_batch_size + offset)
                if not graphs:
                    raise RuntimeError("graph generation produced nothing")
                g = graphs[0]
                tg = TestGraph(domain=cfg['domain'], llm_name=args.llm_name,
                               decision_method=args.decision_method,
                               pyg_data=convert_to_pyg_graph(g, text))
                coros.append(tg.arun({"task": text}, num_rounds=1))
                meta.append((task_id, text, record["answer"], g))
            except Exception as e:
                results.append({"task_id": task_id, "question": text,
                                "true_answer": record.get("answer"),
                                "is_solved": False, "error": str(e)})

        if not coros:
            continue
        outs = await asyncio.gather(*coros, return_exceptions=True)

        for out, (task_id, text, gold, g) in zip(outs, meta):
            if isinstance(out, Exception):
                results.append({"task_id": task_id, "question": text,
                                "true_answer": gold, "is_solved": False,
                                "error": str(out)})
                continue
            raw = out[0] if isinstance(out, (list, tuple)) and out else out
            pred = get_predict(raw)
            ok = graded(pred, gold, args.target)
            solved += ok
            results.append({
                "task_id": task_id, "question": text, "true_answer": gold,
                "level": meta_level.get(task_id, ""),
                "predicted_answer": pred, "raw_response": raw, "is_solved": bool(ok),
                "num_nodes": g.number_of_nodes(), "num_edges": g.number_of_edges(),
                "roles": [g.nodes[v].get('role') for v in g.nodes()],
            })

        pbar.set_postfix({"Acc": f"{100*solved/max(len(results),1):.2f}%"})
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')


    pass_at_1 = 100.0 * solved / total if total else 0.0
    print("\n" + "=" * 50)
    print(f"{args.target}: {solved}/{total}  Pass@1 {pass_at_1:.2f}%")
    print(f"Prompt tokens {int(PromptTokens.instance().value)}, "
          f"completion {int(CompletionTokens.instance().value)}")
    print(repair_report())
    print(f"Details -> {args.output_file}")

    rec = {"timestamp": datetime.datetime.now().isoformat(),
           "dataset": args.target, "trained_on": getattr(model.args, 'dataset', '?'),
           "model_path": args.model_path + ('ef_best' if ef else 'best'),
           "llm_name": args.llm_name, "seed": args.seed,
           "total_tasks": total, "solved_tasks": solved, "pass_at_1": pass_at_1,
           "cost": Cost.instance().value,
           "prompt_tokens": PromptTokens.instance().value,
           "completion_tokens": CompletionTokens.instance().value,
           "detail_file": args.output_file,
           "repairs": dict(REPAIR_STATS)}
    try:
        os.makedirs(os.path.dirname(args.summary_log_file), exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception as e:
        print(f"Failed to write summary log: {e}")
    print("=" * 50)


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
