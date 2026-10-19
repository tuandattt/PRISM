import os
import json
import math
import asyncio
import time
import torch
import numpy as np
import argparse
import random
from tqdm import tqdm
from typing import List, Any, Dict, Iterator
import sys
import csv
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from mas_framework.graph.graph import TestGraph
from experiment.utils import Accuracy, load_model, generate_graph, convert_to_pyg_graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiment.paths import cold_start_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
    parser.add_argument('--agent_nums', type=int, default=6,
                        help="number of agents")
    parser.add_argument('--data_dir', type=str, default=cold_start_dir("mmlu"),
                        help="data directory")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="number of inference rounds per query")
    parser.add_argument('--llm_name', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="dataset name")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="limit number of questions to evaluate")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="decision method for final node")
    parser.add_argument('--model_path', type=str, default='output/xxx',
                        help="path to pretrained model")
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help="evaluation batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="model for task embeddings")
    parser.add_argument('--summary_log_file', type=str, default='./res_logs/evaluation_summary.jsonl',
                        help="log file to record evaluation summaries")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Per-question results as JSONL. Default: derived from the "
                             "model dir, checkpoint and seed, under res_logs/details/. "
                             "The upstream script appended every run of every arm to one "
                             "ARGDesigner_results.csv with no seed, model or graph size, "
                             "so runs could not be told apart or paired.")
    parser.add_argument('--ef', type=int, default=1, choices=[0, 1],
                        help="1: load ef_best_model.pth (the baseline's Stage-2 "
                             "efficiency finetune). 0: load best_model.pth -- what a "
                             "cold-start-only run and an OPSD run both produce.")

    return parser.parse_args()


async def evaluate(
        model,
        dataset,
        sentence_model,
        role_constraints_dict,
        args
) -> float:
    """Evaluate model on MMLU dataset"""
    print(f"Evaluating model on {dataset.__class__.__name__}")

    accuracy = Accuracy()
    limit_questions = args.limit_questions
    results_list = []

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=args.eval_batch_size)), total=num_batches):
        print(f"{'-' * 80}")

        start_ts = time.time()
        answer_tasks = []
        questions = []
        graphs = []

        for i, record in enumerate(record_batch):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict['task']
            questions.append(task_text)

            question_id = i_batch * args.eval_batch_size + i + 1

            task_embedding = torch.tensor(
                sentence_model.encode(task_text),
                device=model.args.device
            ).float()

            generated_graph = generate_graph(
                model,
                task_embedding,
                role_constraints_dict,
                question_id=question_id
            )

            graphs.append(generated_graph[0])
            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=convert_to_pyg_graph(generated_graph[0], task_text)
            )
            answer_tasks.append(asyncio.create_task(tg.arun(input_dict, args.num_rounds)))

        raw_results = await asyncio.gather(*answer_tasks)
        is_corrects = []

        for raw_answer, record in zip(raw_results, record_batch):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)

            print(f"Accuracy: {accuracy.print()} | "
                  f"Cost: ${Cost.instance().value:.4f} | "
                  f"Tokens: P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})")

            is_corrects.append(is_correct)

        for i in range(len(record_batch)):
            idx = i_batch * args.eval_batch_size + i
            g = graphs[i]
            # task_id must be stable across arms -- it is the join key that makes
            # the comparison paired rather than a difference of two marginals.
            results_list.append({
                "task_id": f"task_{idx}",
                "question": questions[i],
                "is_solved": bool(is_corrects[i]),
                "num_nodes": g.number_of_nodes(),
                "num_edges": g.number_of_edges(),
            })

        with open(args.output_file, 'w', encoding='utf-8') as f:
            for res in results_list:
                f.write(json.dumps(res) + '\n')

        print(f"Batch time: {time.time() - start_ts:.3f}s")

    accuracy.print()
    print("Evaluation complete!")
    return accuracy.get(), results_list


def write_to_csv(
        data: List[Dict],
        filename: str = "ARGDesigner_results.csv",
        fieldnames: List[str] = ["dataset", "id", "question", "mode", "size", "is_correct"]
):
    """Write result data to CSV file"""
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)


async def main(ef=None):
    args = parse_args()
    if ef is None:
        ef = bool(args.ef)

    if not args.output_file:
        tag = os.path.basename(os.path.normpath(args.model_path)) or 'model'
        args.output_file = os.path.join(
            'res_logs', 'details',
            f"mmlu_{tag}_{'ef' if ef else 'base'}_seed{args.seed}.jsonl")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    print(f"Per-question results -> {args.output_file}")

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("Loading MMLU dataset...")
    download()
    # Scored on the held-out test split. dev is the training split (cold start,
    # finetune, OPSD); val is unused.
    dataset_test = MMLUDataset('test')

    print(f"Loading Sentence Transformer model: {args.embedding_model}")
    sentence_model = SentenceTransformer(args.embedding_model)

    from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
    role_constraints_dict = {role: desc for role, desc in ROLE_DESCRIPTION.items()}

    print(f"Loading pretrained model: {args.model_path}")
    args.model_name = 'ef_best' if ef else 'best'
    print('model_name', args.model_name)
    model = load_model(args.model_path, ef=ef)

    score, results_list = await evaluate(
        model=model,
        dataset=dataset_test,
        sentence_model=sentence_model,
        role_constraints_dict=role_constraints_dict,
        args=args
    )

    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    total_tasks = min(len(dataset_test), args.limit_questions) if args.limit_questions is not None else len(dataset_test)

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Model path: {args.model_path}")
    print(f"Total tasks: {total_tasks}\nFinal accuracy (Pass@1): {score:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)
    print(f"Per-question results saved to: {args.output_file}")

    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "mmlu",
        "model_path": args.model_path + args.model_name,
        "llm_name": args.llm_name,
        "seed": args.seed,
        "total_tasks": total_tasks,
        "solved_tasks": sum(1 for r in results_list if r["is_solved"]),
        "pass_at_1": score,
        "cost": final_cost,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "detail_file": args.output_file
    }

    try:
        os.makedirs(os.path.dirname(args.summary_log_file), exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
        print(f"Summary log appended to: {args.summary_log_file}")
    except Exception as e:
        print(f"Failed to write summary log file: {e}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
