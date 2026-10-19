"""Central path resolution.

The upstream scripts hard-coded absolute paths such as
``/root/ARG-Designer/experiment/ColdStartData_humaneval``, which makes the
repo unrunnable anywhere else.  Everything now resolves relative to the repo
root, with an environment-variable escape hatch:

    ARG_DATA_ROOT   root for generated data (default: <repo>/data)
    ARG_OUTPUT_ROOT root for model checkpoints (default: <repo>/output)
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_ROOT = os.environ.get('ARG_DATA_ROOT') or os.path.join(REPO_ROOT, 'data')
OUTPUT_ROOT = os.environ.get('ARG_OUTPUT_ROOT') or os.path.join(REPO_ROOT, 'output')

DATASET_FILES = {
    'humaneval': os.path.join(REPO_ROOT, 'datasets', 'humaneval', 'humaneval-py.jsonl'),
    'gsm8k': os.path.join(REPO_ROOT, 'datasets', 'gsm8k', 'gsm8k.jsonl'),
    'aqua': os.path.join(REPO_ROOT, 'datasets', 'AQuA', 'AQuA.jsonl'),
    'mmlu': os.path.join(REPO_ROOT, 'datasets', 'MMLU', 'data'),
    'multiarith': os.path.join(REPO_ROOT, 'datasets', 'MultiArith', 'MultiArith.json'),
    'svamp': os.path.join(REPO_ROOT, 'datasets', 'SVAMP', 'SVAMP.json'),
    # Training-only corpus: 12k MATH/PRM800K problems, each with a worked
    # solution. Used to train a designer that is then evaluated on the other
    # math benchmarks, so there is no train/test overlap to control for.
    'mixchain': os.path.join(REPO_ROOT, 'datasets', 'MixChain', 'mixchain.jsonl'),
    # Code, training-only (the code designer is evaluated on HumanEval). Written
    # in HumanEval's record shape by datasets/MBPP/download.py, so the HumanEval
    # cold start / finetune scripts read it via --dataset_json.
    'mbpp': os.path.join(REPO_ROOT, 'datasets', 'MBPP', 'mbpp.jsonl'),
}


def cold_start_dir(dataset: str) -> str:
    return os.path.join(DATA_ROOT, f'ColdStartData_{dataset}')


def finetune_dir(dataset: str) -> str:
    return os.path.join(DATA_ROOT, f'FinetuneData_{dataset}')


def task_split_file(dataset: str) -> str:
    return os.path.join(DATA_ROOT, f'task_split_{dataset}.json')


def dataset_file(dataset: str) -> str:
    return DATASET_FILES[dataset]


def run_dir(dataset: str, arch: str, seed: int, tag: str = '') -> str:
    name = f'{dataset}_{arch}_seed{seed}' + (f'_{tag}' if tag else '')
    return os.path.join(OUTPUT_ROOT, name)


def ensure(*dirs) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
