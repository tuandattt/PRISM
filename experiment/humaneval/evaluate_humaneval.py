import os
import json
import torch
import argparse
import asyncio
from tqdm import tqdm
import sys
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from mas_framework.tools.reader.readers import JSONLReader
from mas_framework.tools.coding.python_executor import PyExecutor
from mas_framework.graph.graph import TestGraph
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph
from experiment.humaneval.finetune_humaneval import setup_environment
from experiment.paths import dataset_file, task_split_file
from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model performance on HumanEval")
    parser.add_argument('--model_path', type=str, default='./output/xxxx',
                        help="Trained model path (.pth)")
    parser.add_argument('--dataset_path', type=str, default=dataset_file("humaneval"),
                        help="HumanEval dataset path")
    parser.add_argument('--task_split_path', type=str, default=task_split_file("humaneval"),
                        help="Task split file path")
    parser.add_argument('--llm_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM name")
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode",
                        help="Decision method for the final node")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Per-question results. Default: a name derived from the "
                             "model dir, checkpoint and seed, under res_logs/details/. "
                             "A single fixed name opened 'w' means every arm and every "
                             "seed overwrite each other, and paired comparison needs both.")
    parser.add_argument('--summary_log_file', type=str, default='./res_logs/evaluation_summary.jsonl',
                        help="Log file to record all evaluation run summaries")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument('--full_test', action='store_true',
                        help="Ignore the task split and evaluate every problem. Valid "
                             "only for a designer that never saw this dataset; the "
                             "corpus-construction problems are otherwise included. "
                             "Writes to a '_full' detail file.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (graph sampling is stochastic)")
    parser.add_argument('--ef', type=int, default=1, choices=[0, 1],
                        help="1: load ef_best_model.pth (the baseline's Stage-2 "
                             "efficiency finetune). 0: load best_model.pth -- what a "
                             "cold-start-only run and an OPSD run both produce.")
    return parser.parse_args()


async def main(ef=None):
    args = parse_args()
    if ef is None:
        ef = bool(args.ef)

    if not args.output_file:
        tag = os.path.basename(os.path.normpath(args.model_path)) or 'model'
        args.output_file = os.path.join(
            'res_logs', 'details',
            f"humaneval_{tag}_{'ef' if ef else 'base'}_seed{args.seed}"
            f"{'_full' if args.full_test else ''}.jsonl")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    print(f"Per-question results -> {args.output_file}")

    setup_environment(args)
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    print("Loading model and tools...")
    if ef:
        args.model_name = 'ef_best'
    else:
        args.model_name = 'best'
    model = load_model(args.model_path, ef=ef)
    model.eval()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    executor = PyExecutor()
    role_constraints_dict = ROLE_DESCRIPTION
    full_dataset = JSONLReader.parse_file(args.dataset_path)
    if args.full_test:
        # task_id comes from the record itself, so a full-test detail file still
        # pairs with a split-based one; the split set is a subset of these ids.
        print("--full_test: ignoring the task split; corpus-construction problems "
              "are INCLUDED. Only use this for a designer trained on another corpus.")
        dataset = full_dataset
    else:
        if not os.path.exists(args.task_split_path):
            raise FileNotFoundError(
                f"Task split file '{args.task_split_path}' not found. Please run cold_start_humaneval.py first.")
        with open(args.task_split_path, 'r') as f:
            task_split = json.load(f)
        test_indices = task_split.get('test_indices')
        if not test_indices:
            raise ValueError("No 'test_indices' found in task split file.")
        dataset = [full_dataset[i] for i in test_indices]
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Loaded {len(dataset)} HumanEval test set samples for evaluation.")
    total_tasks = len(dataset)
    solved_tasks = 0
    results_list = []
    from typing import Iterator, List, Any
    import math

    def eval_loader(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
        records = []
        for record in data:
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    num_batches = math.ceil(total_tasks / args.eval_batch_size)
    pbar = tqdm(enumerate(eval_loader(dataset, args.eval_batch_size)), total=num_batches, desc="Evaluating model")
    for i_batch, record_batch in pbar:
        answer_tasks = []
        metadata_for_tasks = []
        for record in record_batch:
            task_text = record["prompt"]
            task_id = record.get("task_id", f"task_{i_batch * args.eval_batch_size + len(metadata_for_tasks)}")
            task_embedding = torch.tensor(sentence_model.encode(task_text), device=model.args.device).float()
            generated_graphs = generate_graph(model, task_embedding, role_constraints_dict, task_id)
            if not generated_graphs:
                print(f"Warning: Failed to generate graph for task {task_id}.")
                results_list.append(
                    {"task_id": task_id, "prompt": task_text, "generated_code": None, "is_solved": False,
                     "error": "Graph generation failed"})
                continue
            generated_graph = generated_graphs[0]
            pyg_data = convert_to_pyg_graph(generated_graph, task_text)
            test_graph = TestGraph(domain="humaneval", llm_name=args.llm_name, decision_method=args.decision_method,
                                   pyg_data=pyg_data)
            coro = test_graph.arun({"task": task_text}, num_rounds=1)
            answer_tasks.append(coro)
            metadata_for_tasks.append({"record": record, "task_id": task_id, "generated_graph": generated_graph})
        if not answer_tasks:
            continue
        all_results = await asyncio.gather(*answer_tasks, return_exceptions=True)
        for i, result in enumerate(all_results):
            metadata = metadata_for_tasks[i]
            record = metadata["record"]
            task_id = metadata["task_id"]
            generated_graph = metadata["generated_graph"]
            if isinstance(result, Exception):
                print(f"Error processing task {task_id}: {result}")
                results_list.append(
                    {"task_id": task_id, "prompt": record["prompt"], "generated_code": None, "is_solved": False,
                     "error": str(result)})
                continue
            raw_answer = result
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]
            answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")
            is_solved, _, _ = executor.execute(answer_code, [record["test"]], timeout=100)
            if is_solved:
                solved_tasks += 1
            result_item = {
                "task_id": task_id,
                "prompt": record["prompt"],
                "generated_code": answer_code,
                "is_solved": is_solved,
                "num_nodes": generated_graph.number_of_nodes(),
                "num_edges": generated_graph.number_of_edges(),
            }
            results_list.append(result_item)
        current_processed = (i_batch * args.eval_batch_size) + len(record_batch)
        acc = solved_tasks / current_processed * 100
        pbar.set_postfix({
            "Accuracy": f"{acc:.2f}% ({solved_tasks}/{current_processed})",
            "Token": f"${PromptTokens.instance().value:.4f}",
        })
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for res in results_list:
                f.write(json.dumps(res) + '\n')
    pass_at_1 = (solved_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Model path: {args.model_path}")
    print(f"Total tasks: {total_tasks}\nSuccessfully solved: {solved_tasks}\nPass@1: {pass_at_1:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)
    print(f"Detailed evaluation results saved to: {args.output_file}")
    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "humaneval",
        "model_path": args.model_path + args.model_name,
        "llm_name": args.llm_name,
        "seed": args.seed,
        "total_tasks": total_tasks,
        "solved_tasks": solved_tasks,
        "pass_at_1": pass_at_1,
        "cost": final_cost,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "detail_file": args.output_file
    }
    try:
        os.makedirs(os.path.dirname(args.summary_log_file), exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
        print(f"Evaluation summary appended to: {args.summary_log_file}")
    except Exception as e:
        print(f"Failed to write summary log file: {e}")
    print("=" * 50)


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
