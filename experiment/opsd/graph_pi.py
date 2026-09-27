"""Topology-valued privileged information, as an alternative to the rationale.

The rationale channel is dataset-specific: gsm8k supplies a worked solution,
AQuA a derivation, HumanEval only a test suite, MMLU a bare option string. This
module offers a *unified* channel that needs no rationale field at all -- the
teacher is conditioned on a graph instead.

Two modes, and the difference decides whether the idea works at all:

``sibling``  the teacher predicting graph A is shown a DIFFERENT graph B that
             also solved the same query. B is informative about A but is not A,
             so copying the channel is penalised and the teacher has to learn
             which *family* of topologies a query admits.

``gold``     the teacher predicting graph A is shown A itself. The target sits
             in the input, so verbatim copying is optimal: teacher NLL collapses
             toward zero, the "privilege gain" measures memorisation rather than
             learnability, and at distillation the KL degenerates into
             cross-entropy toward the gold graph -- which is exactly the Eq. 5
             MLE objective cold start already ran. Kept only as the ablation
             that demonstrates this.

Bandwidth caveat: with <=5 nodes over 4 roles, D_eff holds ~10 distinct
blueprints, so this channel carries roughly log2(10) ~ 3.3 bits -- far less than
a worked rationale. Being unified and rationale-free is its point; it is not a
higher-capacity channel, and on gsm8k (where the rationale beats its shuffle
control by +3.80) it should be expected to underperform.
"""
import random
from typing import Callable, Dict, List, Optional

import torch

from experiment.utils import encode_text


def graph_to_text(graph) -> str:
    """Canonical text form of a topology.

    Serialising to text rather than adding a graph encoder is deliberate:

    * The FiLM output passes through a rank-16 bottleneck either way, so a
      richer encoder's extra capacity cannot reach the model.
    * Cold start yields O(100) training QUERIES. A GNN adds parameters that
      cannot be fitted at that scale -- the same reason ``privileged_rank``
      exists (the full-width channel memorised ~150 graphs).
    * Role names are natural language, and the model's own node vocabulary is
      built from sentence embeddings of those same names, so MiniLM already
      places "Math Solver" near "Mathematical Analyst".
    * Everything downstream -- ``answer_processor``, FiLM,
      ``--privileged_control shuffle``, every diagnostic -- keeps working
      unchanged, so the existing controls stay valid.

    Roles and edges are sorted, so one topology always maps to one string.
    Without that, node ordering would inject noise into the channel.
    """
    roles = sorted(str(graph.nodes[v].get('role', '?')) for v in graph.nodes())
    edges = sorted(
        f"{graph.nodes[u].get('role', '?')} -> {graph.nodes[v].get('role', '?')}"
        for u, v in graph.edges())
    return (f"Agents ({len(roles)}): " + ", ".join(roles) + ". "
            + "Edges: " + ("; ".join(edges) if edges else "none") + ".")


class GraphPrivilegedEmbeddings:
    """Drop-in replacement for AnswerEmbeddings that serves topologies.

    Exposes the same surface the rest of the pipeline expects -- ``dim``,
    ``__len__``, ``get``, ``for_graphs``, ``as_lookup`` -- so ``train_teacher``,
    ``build_query_pool`` and the shuffle/constant controls need no changes.
    """

    def __init__(self, graphs, mode: str = 'sibling', seed: int = 0,
                 verbose: bool = True):
        if mode not in ('sibling', 'gold'):
            raise ValueError(f"unknown graph privileged mode {mode!r}")
        self.mode = mode
        self.dim: Optional[int] = None
        self.missing = 0
        self._rng = random.Random(seed)

        self.by_query: Dict[str, List[str]] = {}
        self.text_of: Dict[int, str] = {}          # id(graph) -> its own text
        texts = set()
        for g in graphs:
            question = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
            if not question:
                self.missing += 1
                continue
            t = graph_to_text(g)
            self.text_of[id(g)] = t
            texts.add(t)
            bucket = self.by_query.setdefault(question, [])
            if t not in bucket:
                bucket.append(t)

        self.vec: Dict[str, torch.Tensor] = {}
        if texts:
            ordered = sorted(texts)                # deterministic encode order
            vectors = encode_text(ordered)
            self.dim = int(vectors.shape[-1])
            for t, v in zip(ordered, vectors):
                self.vec[t] = torch.tensor(v).float()

        # A query with a single distinct topology has no sibling to show.
        self.no_sibling = sum(1 for v in self.by_query.values() if len(v) < 2)

        if verbose:
            extra = ""
            if mode == 'sibling':
                extra += f", {self.no_sibling} queries have no sibling"
            if self.missing:
                extra += f", {self.missing} graphs carried no query text"
            print(f"[opsd] graph-PI ({mode}): {len(self.by_query)} queries, "
                  f"{len(self.vec)} distinct topologies{extra}")
            if mode == 'gold':
                print("[opsd] WARNING: gold_graph puts the TARGET in the teacher's input.")
                print("        Copying becomes optimal, the privilege gain measures")
                print("        memorisation rather than learnability, and the distillation")
                print("        reduces to supervised MLE on (query, graph) -- which cold")
                print("        start already did. This is an ablation, not a method.")
            if mode == 'sibling' and self.no_sibling == len(self.by_query):
                print("[opsd] WARNING: no query has two distinct topologies, so every row")
                print("        falls back to a zero vector and the teacher is the student.")

    def __len__(self) -> int:
        return len(self.by_query)

    def _zero(self) -> torch.Tensor:
        return torch.zeros(self.dim or 384)

    def get(self, question: str) -> Optional[torch.Tensor]:
        """One topology vector per query, for the rollout pool.

        Fixed per query for the whole run -- matching how the rationale behaves,
        since ``build_query_pool`` calls this once and stores the result.
        """
        opts = self.by_query.get((question or '').strip())
        if not opts:
            return None
        return self.vec.get(opts[self._rng.randrange(len(opts))])

    def for_graphs(self, graphs, device) -> torch.Tensor:
        """Privileged vectors aligned to a batch of graphs, resampled per call.

        Resampling on every call (so, every epoch) is what stops the teacher
        memorising one fixed A-shown-B pairing instead of learning the family of
        topologies a query admits.
        """
        out = []
        for g in graphs:
            question = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
            own = self.text_of.get(id(g)) or graph_to_text(g)
            if self.mode == 'gold':
                out.append(self.vec.get(own, self._zero()))
                continue
            opts = [t for t in self.by_query.get(question, []) if t != own]
            # No sibling -> zero vector, which FiLM maps to the identity, so the
            # row falls back to query-only conditioning rather than being shown a
            # topology belonging to some other question.
            out.append(self.vec[opts[self._rng.randrange(len(opts))]]
                       if opts else self._zero())
        return torch.stack(out).to(device).float()

    def as_lookup(self, control: str = 'none', seed: int = 0) -> Callable:
        """Same contract as AnswerEmbeddings.as_lookup, so the controls transfer.

        ``shuffle`` permutes which QUERY's topology set each query draws from:
        capacity and marginal distribution preserved, correspondence destroyed.
        """
        if control == 'none':
            return self.for_graphs

        if control == 'constant':
            mean = (torch.stack(list(self.vec.values())).mean(dim=0)
                    if self.vec else self._zero())

            def _constant(graphs, device):
                return mean.unsqueeze(0).repeat(len(graphs), 1).to(device).float()
            return _constant

        if control == 'shuffle':
            keys = sorted(self.by_query)
            perm = list(range(len(keys)))
            random.Random(seed).shuffle(perm)
            for i, j in enumerate(perm):           # no query keeps its own set
                if i == j and len(perm) > 1:
                    k = (i + 1) % len(perm)
                    perm[i], perm[k] = perm[k], perm[i]
            remap = {k: keys[perm[i]] for i, k in enumerate(keys)}

            def _shuffled(graphs, device):
                out = []
                for g in graphs:
                    q = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
                    opts = self.by_query.get(remap.get(q, ''), [])
                    out.append(self.vec[opts[self._rng.randrange(len(opts))]]
                               if opts else self._zero())
                return torch.stack(out).to(device).float()
            return _shuffled

        raise ValueError(f"unknown privileged control {control!r}")
