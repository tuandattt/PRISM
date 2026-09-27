"""Privileged-side data plumbing.

The student's conditioning already exists: every cold-start graph carries a
``task_embedding`` of its query. What OPSD adds is the *rationale* embedding for
the same query, kept strictly on the teacher's side and never available at
inference.

Per-dataset privileged text (all training-time only):

    gsm8k   ``item["step"]``  -- the worked solution, already parsed by
                                 ``gsm_data_process``
    aqua    ``rationale``     -- the explanation field in AQuA.jsonl
    mmlu    the correct option's *text*, not the letter -- a letter carries
            essentially no signal through a sentence encoder, which is why MMLU
            is expected to show the smallest gain

A query with no rationale falls back to its final answer; if that is empty too
the row is dropped from the teacher's view rather than fed a zero vector, so a
missing rationale never masquerades as "the answer is uninformative".
"""
import os
from typing import Callable, Dict, List, Optional, Sequence

import torch

from experiment.utils import encode_text


def privileged_text(record: Dict, dataset: str) -> str:
    """The teacher-only text for one dataset record."""
    if dataset in ("gsm8k", "multiarith", "svamp", "mixchain"):
        step = (record.get("step") or "").strip()
        answer = str(record.get("answer", "")).strip()
        if step and answer:
            return f"{step}\nThe answer is {answer}."
        return step or (f"The answer is {answer}." if answer else "")

    if dataset == "aqua":
        rationale = (record.get("rationale") or record.get("step") or "").strip()
        answer = str(record.get("answer", "")).strip()
        if rationale and answer:
            return f"{rationale}\nThe answer is {answer}."
        return rationale or (f"The answer is {answer}." if answer else "")

    if dataset == "humaneval":
        # What a correct program must satisfy, which is what determines whether a
        # Test Analyst / Bug Fixer is worth a seat.
        #
        # NOTE: the MultiPL-E humaneval-py.jsonl shipped here has NO
        # 'canonical_solution' field (its keys are name/language/prompt/doctests/
        # original/prompt_terminology/stop_tokens/entry_point/test), so in
        # practice this returns the test suite alone. That is still privileged --
        # the student never sees the tests -- but it is a specification, not a
        # worked solution, so do not describe it as one when reporting results.
        # The solution branch is kept for dataset variants that do carry it.
        sol = (record.get("canonical_solution") or record.get("step") or "").strip()
        tests = (record.get("test") or "").strip()
        return "\n".join(x for x in (sol, tests) if x)

    if dataset == "mmlu":
        # Option TEXT, never the bare letter.
        answer = str(record.get("answer", "")).strip()
        options = record.get("options") or record.get("choices") or []
        idx = "ABCDE".find(answer.upper()[:1]) if answer else -1
        if 0 <= idx < len(options):
            return f"The correct answer is: {options[idx]}"
        return f"The correct answer is: {answer}" if answer else ""

    return str(record.get("step") or record.get("answer") or "").strip()


def privileged_context_text(record: Dict, dataset: str) -> str:
    """The TEACHER's full context, in the OPSD paper's sense.

    The reference implementation (siyan-zhao, arXiv 2601.18734) uses one model as
    both student and teacher, conditioned on different CONTEXTS: the student sees
    only the problem, the teacher additionally sees the ground-truth solution,
    wrapped as

        Here is a reference solution to this problem:
        === Reference Solution Begin ===
        ...
        === Reference Solution End ===

    For an LLM that is a longer prompt. Here the conditioning is a single
    sentence-embedding, so the analogue is to encode question+solution as one
    string and push it through the SAME task_processor the student uses. No new
    parameters, no teacher finetune -- exactly the paper's setup.

    Caveat worth measuring, not assuming: the student's task_processor was fitted
    on question-only embeddings, so a question+solution embedding is off its
    training distribution. An LLM generalises over text; a small MLP over pooled
    vectors need not. The privilege gain is what tells you whether it did.
    """
    question = (record.get("task") or record.get("question") or "").strip()
    solution = privileged_text(record, dataset).strip()
    if not solution:
        return question
    return (f"{question}\n\n"
            f"Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===")


class AnswerEmbeddings:
    """Question text -> privileged embedding, with an on-disk cache.

    Encoding a few thousand rationales with MiniLM is cheap but not free, and
    the OPSD loop touches the same queries every epoch, so the table is built
    once and cached next to the cold-start data.
    """

    def __init__(self, records: Sequence[Dict], dataset: str,
                 cache_path: Optional[str] = None, verbose: bool = True,
                 text_fn: Optional[Callable] = None):
        # text_fn selects WHAT the teacher sees: privileged_text (rationale only,
        # fed through a FiLM channel) or privileged_context_text (question +
        # solution, fed through the student's own task_processor).
        self.text_fn = text_fn or privileged_text
        self.dataset = dataset
        self.dim: Optional[int] = None
        self.table: Dict[str, torch.Tensor] = {}
        self.missing = 0

        if cache_path and os.path.exists(cache_path):
            blob = torch.load(cache_path, map_location='cpu', weights_only=False)
            self.table = blob["table"]
            self.dim = blob["dim"]
            self.missing = blob.get("missing", 0)
            if verbose:
                print(f"[opsd] loaded {len(self.table)} privileged embeddings from {cache_path}")
            return

        texts, keys = [], []
        for rec in records:
            question = (rec.get("task") or rec.get("question") or "").strip()
            text = self.text_fn(rec, dataset)
            if not question:
                continue
            if not text:
                self.missing += 1
                continue
            keys.append(question)
            texts.append(text)

        if texts:
            vectors = encode_text(texts)
            self.dim = int(vectors.shape[-1])
            for k, v in zip(keys, vectors):
                self.table[k] = torch.tensor(v).float()

        if verbose:
            print(f"[opsd] encoded {len(self.table)} privileged texts "
                  f"({self.missing} records had none) for {dataset}")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"table": self.table, "dim": self.dim, "missing": self.missing},
                       cache_path)

    def __len__(self) -> int:
        return len(self.table)

    def get(self, question: str) -> Optional[torch.Tensor]:
        return self.table.get((question or "").strip())

    def for_graphs(self, graphs, device) -> torch.Tensor:
        """Privileged embeddings aligned to a batch of graphs.

        A graph whose question is not in the table gets a zero vector, which the
        FiLM layers map to the identity -- i.e. that row falls back to the
        student's own conditioning rather than being fed noise.
        """
        dim = self.dim or 384
        out = []
        for g in graphs:
            question = g.graph.get('task') if hasattr(g, 'graph') else None
            vec = self.get(question) if question else None
            out.append(vec if vec is not None else torch.zeros(dim))
        return torch.stack(out).to(device).float()

    def as_lookup(self, control: str = 'none', seed: int = 0) -> Callable:
        """Privileged lookup, optionally with the information removed.

        The FiLM fusion is ``f_Q * (1 + gamma(f_A)) + beta(f_A)``, and beta has a
        bias, so the channel can lower NLL purely by adapting the frozen trunk to
        a new graph distribution -- with no help from f_A at all. That adaptation
        is indistinguishable from a privilege gain unless it is measured.

        These controls keep the channel's capacity identical and destroy only the
        information:

        ``shuffle``   real rationale vectors, permuted onto the wrong queries.
                      Marginal distribution preserved, correspondence broken.
                      This is the primary control.
        ``constant``  every query gets the table's mean vector: pure adapter
                      capacity, zero per-query signal.
        ``none``      the real thing.

        Read the result as: gain(real) - gain(shuffle) is the part attributable
        to the rationale. If that difference is ~0, the headline gain was
        domain adaptation.
        """
        if control == 'none':
            return self.for_graphs

        if control == 'constant':
            if self.table:
                mean = torch.stack(list(self.table.values())).mean(dim=0)
            else:
                mean = torch.zeros(self.dim or 384)

            def _constant(graphs, device):
                return mean.unsqueeze(0).repeat(len(graphs), 1).to(device).float()
            return _constant

        if control == 'shuffle':
            import random as _random
            keys = sorted(self.table)
            values = [self.table[k] for k in keys]
            perm = list(range(len(values)))
            _random.Random(seed).shuffle(perm)
            # A derangement in spirit: if any key kept its own vector the control
            # would leak a little real signal.
            for i, j in enumerate(perm):
                if i == j and len(perm) > 1:
                    k = (i + 1) % len(perm)
                    perm[i], perm[k] = perm[k], perm[i]
            shuffled = {k: values[perm[i]] for i, k in enumerate(keys)}

            def _shuffled(graphs, device):
                dim = self.dim or 384
                out = []
                for g in graphs:
                    q = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
                    out.append(shuffled.get(q, torch.zeros(dim)))
                return torch.stack(out).to(device).float()
            return _shuffled

        raise ValueError(f"unknown privileged control {control!r}")


def build_query_pool(records: Sequence[Dict], dataset: str,
                     answers: AnswerEmbeddings,
                     limit: Optional[int] = None,
                     require_privileged: bool = True) -> List[Dict]:
    """Queries the OPSD loop rolls out on.

    This pool can be far larger than the cold-start graph set: a rollout costs
    no LLM calls and needs no verified graph, only a question and its rationale.
    That is precisely the coverage advantage of conditioning on the answer
    rather than on a ground-truth topology, which exists only where cold start
    happened to succeed.
    """
    selected = []
    for rec in records:
        question = (rec.get("task") or rec.get("question") or "").strip()
        if not question:
            continue
        ans = answers.get(question)
        if ans is None:
            if require_privileged:
                continue
            ans = torch.zeros(answers.dim or 384)
        selected.append((question, ans, rec.get("answer")))
        if limit and len(selected) >= limit:
            break

    if not selected:
        return []

    # One batched encode; per-query calls dominate pool construction otherwise.
    vectors = encode_text([q for q, _, _ in selected])
    return [{
        "task": question,
        "task_embedding": torch.tensor(vec).float(),
        "answer_embedding": ans,
        "answer": gold,
    } for (question, ans, gold), vec in zip(selected, vectors)]
