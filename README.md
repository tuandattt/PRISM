# PRISM: PRIvileged Sibling-guided Multi-Agent Topology Design

Official implementation of **PRISM: PRIvileged Sibling-guided Multi-Agent
Topology Design**.

Privileged-teacher, on-policy self-distillation (OPSD) for multi-agent topology
design, built on [ARG-Designer](https://arxiv.org/abs/2507.18224). A teacher that
sees the query **and its gold rationale** gives dense per-step supervision to a
query-only student. The supervision is applied to pseudo-graphs that the student
samples itself. This replaces ARG-Designer's Stage 2 (`D_pruned`) and its
generate, ablate and re-execute LLM loop with a distillation step that makes
**no LLM calls**.

```
Stage 0  cold start on D_exp            LLM calls: the only LLM cost
Stage 1  privileged teacher finetune    no LLM
Stage 2  OPSD: rollout -> teacher KL    no LLM   (replaces D_pruned)
Stage 3  evaluation                     LLM calls; the teacher is discarded
```

The LLM backend can be any OpenAI-compatible endpoint (a local vLLM server or a
hosted API). You choose it in `.env`.

---

## 1. Repository layout

```
PRISM/
├── experiment/
│   ├── model.py              # ARG-Designer autoregressive graph generator
│   ├── paths.py              # all data/output paths (overridable via env vars)
│   ├── evaluate_math.py      # cross-benchmark evaluator (gsm8k, aqua, multiarith, svamp)
│   ├── opsd/                 # PRISM / OPSD: teacher, rollouts, KL losses, SFT baseline
│   │   ├── run_opsd.py       # entry point: Stage 1 + Stage 2
│   │   └── run_sft.py        # matched supervised-finetuning baseline
│   ├── gsm8k/ aqua/ mmlu/ humaneval/ mixchain/
│   │                         # per-dataset cold_start_*, finetune_*, evaluate_*
│   └── ...
├── mas_framework/            # agents, graph execution, LLM client, tools
├── requirements.txt
└── template.env
```

### Datasets

`datasets/` holds the dataset loaders (`datasets/*_dataset.py`) and the
benchmark files, at the paths set in `experiment/paths.py`. These files are
already included:

| dataset | file |
|---|---|
| GSM8K | `datasets/gsm8k/gsm8k.jsonl` |
| AQuA | `datasets/AQuA/AQuA.jsonl` |
| HumanEval | `datasets/humaneval/humaneval-py.jsonl` |
| MMLU | `datasets/MMLU/data/` (re-download with `python datasets/MMLU/download.py`) |
| MultiArith, SVAMP | `datasets/MultiArith/`, `datasets/SVAMP/` (evaluation only) |
| MBPP | `datasets/MBPP/mbpp.jsonl` (evaluation only) |

MixChain-Z-PRM12K, a math corpus used only for training, is not included. Fetch
it with `python datasets/MixChain/download.py`, run from the repo root.

> Do **not** `pip install datasets` (HuggingFace) into this environment. The
> local `datasets/` folder is a namespace package, and the HF library would
> shadow it. This is also why `sentence-transformers` is pinned to `2.7.0`.

The helper scripts referenced below (`check_vllm.py`, `smoke_opsd.py`,
`e2e_opsd_synthetic.py`) are in `scripts/` of the OPSD_ARG_Designer repo.
Copy that folder as well if you want them:
`Copy-Item -Recurse G:\Code\OPSD_ARG_Designer\scripts G:\Code\PRISM\scripts`.

---

## 2. Installation

Use two environments. vLLM pins its own torch build, so keep it out of the
framework environment.

```bash
# --- framework env (Python 3.10–3.12) ---
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# torch built for your CUDA driver, e.g. CUDA 12.4:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

```bash
# --- vLLM env (Linux or WSL2 only, on the GPU host) ---
python -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install vllm
```

If your home directory has a small quota, redirect the HuggingFace cache with a
**shell export**. Do not put it in `.env`, because `huggingface_hub` reads it at
import time:

```bash
export HF_HOME=/path/with/space/.cache/huggingface
export VLLM_CACHE_ROOT=/path/with/space/.cache/vllm
```

---

## 3. Configure the LLM backend

```bash
cp template.env .env        # PowerShell: Copy-Item template.env .env
```

**Local vLLM** (default):

```ini
BASE_URL = "http://localhost:8000/v1"
API_KEY  = "EMPTY"
LLM_MAX_CONCURRENCY = 32    # parallel requests sent to the server
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 5
```

**Hosted OpenAI API**: leave `BASE_URL = ""`, set `API_KEY` / `OPENAI_API_KEY`,
and pass `--llm_name gpt-4o-mini` to every command.

Optional path overrides (the defaults are `<repo>/data` and `<repo>/output`):

```ini
ARG_DATA_ROOT = ""
ARG_OUTPUT_ROOT = ""
```

### Serve the model with vLLM

In the vLLM env, run:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

This serves the model at `http://localhost:8000/v1`, which matches `BASE_URL`
in `.env`. The model id is also the name you pass as `--llm_name`.

Qwen2.5-7B in bf16 needs about 16 GB of VRAM. On a smaller card, serve
`Qwen/Qwen2.5-7B-Instruct-AWQ` instead. To check the pipeline cheaply, serve
`Qwen/Qwen2.5-1.5B-Instruct`; accuracy from that model is not comparable. In
either case, pass the same model id as `--llm_name`.

Check the backend before starting a long run:

```bash
python scripts/check_vllm.py --llm_name Qwen/Qwen2.5-7B-Instruct
```

---

## 4. Offline sanity checks (no LLM, no GPU)

```bash
python scripts/smoke_opsd.py            # unit checks on the OPSD machinery
python scripts/e2e_opsd_synthetic.py    # full Stage 1 -> 2 -> reload on synthetic data
```

---

## 5. Running the pipeline (GSM8K example)

`MODEL=Qwen/Qwen2.5-7B-Instruct` is used below. Substitute your own model id.

### Stage 0: cold start (all the LLM cost is here)

```bash
cd experiment/gsm8k
python cold_start_gsm8k.py --llm_name Qwen/Qwen2.5-7B-Instruct \
    --batch_size 4 --num_iterations 10
```

This writes `data/ColdStartData_gsm8k/` and `data/task_split_gsm8k.json`.

### Stage 1 pretrain (+ original ARG-Designer Stage 2, the baseline arm)

```bash
python finetune_gsm8k.py --pretrain --llm_name Qwen/Qwen2.5-7B-Instruct --batch_size 4
```

This writes `output/gsm8k_<timestamp>/best_model.pth`, the cold-start checkpoint
that PRISM starts from. The finetuned baseline is saved as
`ef_best_model.pth` in the same directory.

### Stages 1–2: PRISM / OPSD (no LLM calls)

Run from the **repo root**:

```bash
cd ../..
python -m experiment.opsd.run_opsd \
    --dataset gsm8k \
    --student_dir output/gsm8k_<timestamp> \
    --teacher_epochs 40 --opsd_epochs 30 \
    --rollouts_per_query 4 --kl_direction reverse --anchor_weight 0.3
```

Add `--dry_run` first. It loads the graphs, privileged table and rollout pool,
prints them and stops. Data-plumbing errors show up in seconds this way.

**Go/no-go check.** At the end of Stage 1 the script prints the teacher's NLL
with and without the rationale:

```
[teacher] epoch 40/40 loss=1.83 val=1.91 val_nopriv=2.24 |film|=3.41
[opsd] GO/NO-GO -- teacher NLL improves by 0.3300 nats when the rationale is visible.
```

If the gain is close to zero, the teacher is not using the privileged
information. Stop there, because distillation cannot help.

Stage 2 reports teacher/student agreement (`agree`). It also reports the mean
size of the pseudo-graphs (`|V|`, `|E|`). If PRISM does the job of `D_pruned`,
these sizes should fall while accuracy holds.

### Matched SFT baseline

This arm uses the same checkpoint, data, split and epoch budget. It adds no
teacher and no KL:

```bash
python -m experiment.opsd.run_sft --dataset gsm8k \
    --student_dir output/gsm8k_<timestamp> --sft_epochs 30
```

### Stage 3: evaluation

```bash
cd experiment/gsm8k
# PRISM / OPSD student (and SFT arm): they save best_model.pth -> --ef 0
python evaluate_gsm8k.py --model_path ../../output/gsm8k_opsd_seed42_<timestamp> \
    --ef 0 --llm_name Qwen/Qwen2.5-7B-Instruct --eval_batch_size 32

# ARG-Designer baseline (Stage-2 finetuned, ef_best_model.pth) -> --ef 1 (default)
python evaluate_gsm8k.py --model_path ../../output/gsm8k_<timestamp> \
    --ef 1 --llm_name Qwen/Qwen2.5-7B-Instruct --eval_batch_size 32
```

To evaluate across benchmarks (the designer can come from any math corpus), run
from the repo root:

```bash
python experiment/evaluate_math.py --model_path output/<run_dir> \
    --target svamp --llm_name Qwen/Qwen2.5-7B-Instruct
# --target: gsm8k | aqua | multiarith | svamp
```

---

## 6. Other datasets

The pattern is the same for each dataset: `cold_start_<ds>.py`, then
`finetune_<ds>.py --pretrain`, then `run_opsd --dataset <ds>`, then evaluate.

| dataset | directory | OPSD `--dataset` |
|---|---|---|
| AQuA | `experiment/aqua/` | `aqua` |
| MMLU | `experiment/mmlu/` | `mmlu` |
| HumanEval | `experiment/humaneval/` | `humaneval` |
| MixChain (train-only math corpus) | `experiment/mixchain/` | `mixchain` |

To fit one teacher on several corpora, pass
`--teacher_dataset gsm8k aqua --extra_data_dir data/ColdStartData_aqua`.

---

## 7. Useful OPSD ablations

```bash
# privileged dropout only, no distillation
python -m experiment.opsd.run_opsd ... --privileged_dropout 0.5 --opsd_epochs 0
# control: rationales shuffled onto the wrong queries (run next to every headline number)
python -m experiment.opsd.run_opsd ... --privileged_control shuffle
# privileged source: context | rationale | sibling_graph | graph_stats | gold_graph
python -m experiment.opsd.run_opsd ... --privileged_source graph_stats
# KL direction
python -m experiment.opsd.run_opsd ... --kl_direction forward
python -m experiment.opsd.run_opsd ... --kl_direction jsd --jsd_beta 0.5
# teacher-as-verifier filtering of rollouts
python -m experiment.opsd.run_opsd ... --pseudo_filter teacher --teacher_keep_frac 0.5
# no NLL anchor
python -m experiment.opsd.run_opsd ... --anchor_weight 0.0
```

`python -m experiment.opsd.run_opsd --help` lists every flag.

---

## 8. Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError: datasets.gsm8k_dataset` | run from the repo root; the `datasets/` folder must sit next to `experiment/` |
| `ImportError` from HF `datasets` / sentence-transformers | `pip install sentence-transformers==2.7.0` and `pip uninstall datasets` |
| `No module named class_registry` / `pkg_resources` | `pip install phx-class-registry` (not `class-registry`) |
| connection refused / model not found | check `BASE_URL` in `.env`; `--llm_name` must be exactly the model id given to `vllm serve` |
| requests time out under load | lower `LLM_MAX_CONCURRENCY` in `.env` |
| `task_split_<ds>.json` not found | run Stage 0 (cold start) first |

---

## Citation

If you use PRISM, please cite:

```bibtex
@article{dat2026prism,
 title={PRISM: PRIvileged Sibling-guided Multi-Agent Topology Design},
 author={Dat, Phuong Tuan and Li, Qi and Wang, Xinchao},
 journal={preprint},
 year={2026}
}
```

---

## Acknowledgments

Built on [ARG-Designer](https://arxiv.org/abs/2507.18224) (Li et al., AAAI 2026),
which in turn builds on [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) and
[GDesigner](https://github.com/yanweiyue/GDesigner).
