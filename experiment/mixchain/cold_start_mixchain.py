import os
import json
import math
import asyncio
import copy
import sys
import argparse
import random
import csv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from experiment.graph_census import census
from experiment.utils import get_kwargs, save_graph_with_features
from experiment.mixchain.mixchain_prompt_set import ROLE_DESCRIPTION
from datasets.mixchain_dataset import (mixchain_data_process, mixchain_get_predict,
                                       mixchain_is_correct)

from experiment.paths import cold_start_dir, task_split_file, dataset_file, ensure

OUTPUT_DIR = cold_start_dir("mixchain")
TASK_SPLIT_FILE = task_split_file("mixchain")
BASE_RATE = 0.4


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate cold-start data for MixChain")
    parser.add_argument('--dataset_json', type=str, default=dataset_file("mixchain"),
                        help="Path to MixChain JSONL dataset")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
                        help="Name of the LLM model to use")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='List of agent names')
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="Decision method for the final node")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Number of inference rounds per query")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument('--configs', type=str, default=None,
                        help="Override the blueprint list, e.g. "
                             "'Chain:2,Star:3,FullConnected:3'. Use it to test "
                             'whether topology affects outcomes on this corpus at all.')
    parser.add_argument('--drop_degraded', action='store_true',
                        help="Refuse to label a run in which some agent never "
                             "produced output. OFF by default (original behaviour); "
                             "the count is reported regardless.")
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of iterations to define training set size")
    parser.add_argument('--domain', type=str, default="mixchain",
                        help="Task domain name")
    return parser.parse_args()


def get_unique_complex_configs_mixchain(spec=None):
    """Topology configurations to try during cold start.

    Default: the resource-rich set the paper uses (FullConnected 3-4, Mesh 4).

    ``spec`` overrides it with a comma-separated "Mode:N" list, e.g.
    "Chain:2,Star:3,FullConnected:3". That matters for diagnosing a corpus: the
    default configs are all DENSE and near-identical in information flow, so
    "they all solved the same queries" is weak evidence that topology does not
    matter. Adding a sparse blueprint tests it directly -- if Chain-2 solves a
    DIFFERENT subset, the corpus is discriminative and cold start is worth
    scaling; if it solves exactly the same one, topology is irrelevant on this
    data and no amount of it will teach a conditional designer anything.
    """
    if spec:
        configs = []
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            mode, _, num = part.partition(':')
            if not num.isdigit():
                raise ValueError(
                    f"bad --configs entry {part!r}; expected Mode:N, "
                    f"e.g. Chain:2,Star:3,FullConnected:4")
            configs.append((mode.strip(), int(num)))
        return configs

    configs = set()
    for agent_num in range(2, 5):
        if agent_num == 2:
            continue
        elif agent_num == 3:
            configs.add(('FullConnected', 3))
        else:
            configs.add(('FullConnected', agent_num))
            configs.add(('Mesh', agent_num))
    return list(configs)


def write_to_csv_mixchain(data):
    filename = "records_mixchain.csv"
    fieldnames = ["dataset", "id", "question", "answer", "mode", "size", "is_correct"]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


degraded_runs = 0          # agents that never ran, across the whole pass


async def evaluate_and_save_mixchain(
        graph: Graph,
        dataset,
        args,
        current_mode: str,
        current_agent_num: int
):
    """
    Evaluation and save logic for MixChain dataset.
    """
    num_batches = math.ceil(len(dataset) / args.batch_size)
    global degraded_runs
    total_solved = 0

    for i_batch in tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}"):
        batch_records = dataset[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not batch_records:
            continue

        tasks = []
        for record_idx, record in enumerate(batch_records):
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=flow_graph)

            metadata = {
                "record": record,
                "flow_graph": flow_graph,
                "question": record["task"],
                "record_idx": i_batch * args.batch_size + record_idx,
                # The EXECUTED graph, not the template -- node_failures is
                # recorded on the object that actually ran.
                "executed": tg,
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        coroutines_to_run = [task for task, _ in tasks]
        results = await asyncio.gather(*coroutines_to_run, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task execution error: {result}")
                continue

            metadata = tasks[i][1]
            record = metadata['record']

            raw_answer = result
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]

            predict_answer = mixchain_get_predict(raw_answer)
            true_answer = record["answer"]
            is_solved = False
            try:
                is_solved = mixchain_is_correct(predict_answer, true_answer)
            except (ValueError, TypeError):
                print(f"Could not compare answers: predicted='{predict_answer}', true='{true_answer}'")

            # An agent that never ran cannot be credited. A blueprint that lands
            # on the right answer after losing a node is not evidence that THIS
            # topology works -- only N-1 of its N agents ran. Denser blueprints
            # survive a lost node far more often, so labelling these would bias
            # cold start toward density, which is the confound the whole
            # density analysis exists to control.
            if getattr(metadata.get('executed'), 'node_failures', []):
                degraded_runs += 1
                if getattr(args, 'drop_degraded', False):
                    is_solved = False

            if is_solved:
                total_solved += 1
                record_id = metadata['record_idx']
                name = "_".join(map(str, ['mixchain', record_id, current_mode, current_agent_num, is_solved]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')
                save_graph_with_features(metadata['flow_graph'], filepath, {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_solved,
                    "question": metadata['question']
                })

            batch_data = [{
                "dataset": "mixchain",
                "id": metadata['record_idx'],
                "question": metadata['question'],
                "answer": true_answer,
                "mode": current_mode,
                "size": current_agent_num,
                "is_correct": is_solved
            }]
            write_to_csv_mixchain(batch_data)

    print(f"Configuration {current_mode}-{current_agent_num} done. Solved {total_solved}/{len(dataset)} tasks.")


async def main():
    args = parse_args()
    raw_dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = mixchain_data_process(raw_dataset)

    train_set_size = args.num_iterations * args.batch_size
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_set_size]
    test_indices = all_indices[train_set_size:]

    print(f"Dataset split: {len(train_indices)} train, {len(test_indices)} test")

    finetune_candidates = train_indices.copy()
    random.shuffle(finetune_candidates)

    base_task_count = int(BASE_RATE * len(finetune_candidates))
    base_task_indices = finetune_candidates[:base_task_count]
    finetune_task_indices = finetune_candidates[base_task_count:]

    ensure(os.path.dirname(TASK_SPLIT_FILE), OUTPUT_DIR)
    with open(TASK_SPLIT_FILE, 'w') as f:
        json.dump({
            "base_tasks_indices": base_task_indices,
            "finetune_tasks_indices": finetune_task_indices,
            "test_indices": test_indices
        }, f)
    print(f"Saved task split: {TASK_SPLIT_FILE} ({len(base_task_indices)} for cold start)")

    cold_start_dataset = [dataset[i] for i in base_task_indices]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = get_unique_complex_configs_mixchain(args.configs)
    print(f"Generating MixChain cold-start data for {len(configs)} configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Configuration: Mode={mode}, Agent Nums={agent_num} ===")
        kwargs = get_kwargs(mode, agent_num)

        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': role} for role in random_roles]
        print(f"Assigned roles: {random_roles}")

        graph = Graph(domain=args.domain,
                      llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * agent_num,
                      decision_method=args.decision_method,
                      **kwargs)

        await evaluate_and_save_mixchain(
            graph=graph,
            dataset=cold_start_dataset,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )

    print("All MixChain cold-start data generation complete.")
    if degraded_runs:
        kept = "kept" if not getattr(args, 'drop_degraded', False) else "discarded"
        print(f"[coldstart] {degraded_runs} runs had an agent that never produced "
              f"output ({kept}); pass --drop_degraded to exclude them.")
    census(OUTPUT_DIR, label="Stage 0 cold start (D_exp)")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
