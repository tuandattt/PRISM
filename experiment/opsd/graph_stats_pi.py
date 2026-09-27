"""Family-statistics privileged information.

The other topology channels show the teacher ONE graph. ``gold`` shows the
target itself, so copying is optimal and the distillation degenerates into the
MLE cold start already ran. ``sibling`` shows a different graph that solved the
same query -- non-degenerate, but it samples one member and discards the rest of
a set that averages ~7 graphs per query.

This module conditions the teacher on the SUFFICIENT STATISTICS of that whole
set instead: how often each role appears, which role tends to open the
generation, how large and how dense the successful graphs are, and how much they
vary. It answers "what family of topologies does this query admit" without
naming any member of it.

Why that is not degenerate
--------------------------
The vector is an aggregate over several distinct graphs, so it cannot identify
which one the teacher is currently predicting. Copying is impossible, and the
only way to reduce teacher NLL is to learn the query -> family mapping -- which
is exactly what a student holding only the query can use at test time.

The aggregate is additionally computed LEAVE-ONE-OUT: the graph being predicted
is excluded from its own statistics. Without that, a query whose set holds two
graphs would put ~50% of the target's own features into the teacher's input, and
the channel would slide back toward ``gold`` as the set shrinks.

Numeric features, not text
--------------------------
``graph_to_text`` serialises a topology and lets MiniLM embed it, which is right
for a single graph -- role names are natural language. Statistics are not:
MiniLM has no useful representation of "mean 4.45 nodes, sd 0.8". So the
features are built numerically, z-scored across queries, then carried into the
model's conditioning width by a fixed random projection (Johnson-Lindenstrauss,
seeded, so it is reproducible and distance-preserving).

Bandwidth
---------
Strictly more than ``sibling``: a sibling is one sample from the family, this is
the family's moments. Whether that is ENOUGH is a property of the corpus, not of
the method -- if every query admits the same topologies, the vectors collapse to
a constant and the channel is dead. ``spread_report`` measures exactly that, and
is worth running before Stage 1.
"""
import math
import random
from typing import Callable, Dict, List, Optional

import torch


def graph_features(graph, roles: List[str]) -> Dict[str, float]:
    """Per-graph features, before aggregation.

    ``roles`` fixes the vector layout so every graph in a run produces the same
    coordinates in the same order.
    """
    idx = {r: i for i, r in enumerate(roles)}
    nodes = list(graph.nodes())
    n = len(nodes)
    m = graph.number_of_edges()

    counts = [0.0] * len(roles)
    for v in nodes:
        r = str(graph.nodes[v].get('role', '?'))
        if r in idx:
            counts[idx[r]] += 1.0

    # Which role opens the generation. ARG-Designer is autoregressive, so the
    # first node conditions everything after it; "this query starts with a
    # Knowlegable Expert" is a different claim from "it contains one".
    first = [0.0] * len(roles)
    if nodes:
        r0 = str(graph.nodes[nodes[0]].get('role', '?'))
        if r0 in idx:
            first[idx[r0]] = 1.0

    # Directed density. n < 2 has no possible edge, so 0 rather than a div0.
    density = (m / (n * (n - 1))) if n > 1 else 0.0
    return {'counts': counts, 'first': first, 'n': float(n), 'm': float(m),
            'density': float(density)}


class GraphStatsEmbeddings:
    """Drop-in replacement for AnswerEmbeddings serving family statistics.

    Same surface as GraphPrivilegedEmbeddings -- ``dim``, ``__len__``, ``get``,
    ``for_graphs``, ``as_lookup`` -- so train_teacher, build_query_pool and both
    privileged controls work unchanged.
    """

    def __init__(self, graphs, seed: int = 0, dim: int = 384,
                 verbose: bool = True):
        self.dim = int(dim)
        self.missing = 0
        self._rng = random.Random(seed)

        # One pass to fix the role vocabulary, so the feature layout is stable.
        role_set = set()
        kept = []
        for g in graphs:
            question = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
            if not question:
                self.missing += 1
                continue
            for v in g.nodes():
                role_set.add(str(g.nodes[v].get('role', '?')))
            kept.append((question, g))
        self.roles = sorted(role_set)

        # Per-graph features, and the per-query accumulators the leave-one-out
        # aggregate subtracts from.
        self.feat_of: Dict[int, Dict] = {}
        self.by_query: Dict[str, List[int]] = {}
        self._graphs_of: Dict[str, List[Dict]] = {}
        self._topo_of: Dict[str, set] = {}
        for question, g in kept:
            f = graph_features(g, self.roles)
            self.feat_of[id(g)] = f
            self.by_query.setdefault(question, []).append(id(g))
            self._graphs_of.setdefault(question, []).append(f)
            # Distinct-topology count needs a canonical key per shape.
            key = (tuple(f['counts']), f['n'], f['m'])
            self._topo_of.setdefault(question, set()).add(key)

        # Raw (un-normalised) aggregate per query, over the FULL set. Used by
        # get() for the rollout pool, and as the base the leave-one-out
        # subtraction works from.
        self._full: Dict[str, List[float]] = {
            q: self._aggregate(fs, len(self._topo_of[q]))
            for q, fs in self._graphs_of.items()}

        # z-score across queries: the coordinates have wildly different scales
        # (a role count of 0-4 next to a density of 0-1), and the teacher's
        # answer_processor opens with a LayerNorm that would otherwise be
        # dominated by whichever coordinate happens to be largest.
        self._mu, self._sd = self._moments(list(self._full.values()))

        # Fixed random projection into the model's conditioning width. Seeded,
        # so a rerun reproduces it exactly.
        k = len(self._mu)
        gen = torch.Generator().manual_seed(seed)
        self._proj = torch.randn(k, self.dim, generator=gen) / math.sqrt(k)

        self.vec: Dict[str, torch.Tensor] = {
            q: self._project(v) for q, v in self._full.items()}

        if verbose:
            extra = f", {self.missing} graphs carried no query text" if self.missing else ""
            singleton = sum(1 for v in self._graphs_of.values() if len(v) < 2)
            print(f"[opsd] graph-PI (stats): {len(self.by_query)} queries, "
                  f"{len(self.roles)} roles, {k} raw features -> {self.dim}d{extra}")
            print(f"[opsd]   {singleton} queries have a single graph, so their "
                  f"leave-one-out aggregate is empty and falls back to zero.")
            report = self.spread_report()
            print(f"[opsd]   query vectors: mean pairwise cosine "
                  f"{report['mean']:+.3f}, relative spread {report['spread']:.3f}")
            if report['spread'] < 0.05:
                print("[opsd] WARNING: query vectors are nearly identical, so the")
                print("        statistics channel carries almost no query-dependent")
                print("        signal. Every query admits the same topologies -- this")
                print("        is a property of the corpus, and the teacher will")
                print("        behave like the student. Widen the cold-start blueprint")
                print("        set (add sparse configs) before spending Stage 1.")

    # ---- feature assembly -------------------------------------------------

    def _aggregate(self, feats: List[Dict], n_topo: int) -> List[float]:
        """Mean role counts, role presence, opening role, size/density moments."""
        R = len(self.roles)
        if not feats:
            return [0.0] * (3 * R + 8)
        k = float(len(feats))

        mean_counts = [sum(f['counts'][i] for f in feats) / k for i in range(R)]
        presence = [sum(1.0 for f in feats if f['counts'][i] > 0) / k for i in range(R)]
        first = [sum(f['first'][i] for f in feats) / k for i in range(R)]

        def moments(key):
            vals = [f[key] for f in feats]
            mu = sum(vals) / k
            var = sum((v - mu) ** 2 for v in vals) / k
            return mu, math.sqrt(var)

        n_mu, n_sd = moments('n')
        m_mu, m_sd = moments('m')
        d_mu, d_sd = moments('density')
        # log1p on the counts: the difference between 1 and 3 successful graphs
        # matters far more than between 30 and 32.
        return (mean_counts + presence + first
                + [n_mu, n_sd, m_mu, m_sd, d_mu, d_sd,
                   math.log1p(k), math.log1p(float(n_topo))])

    @staticmethod
    def _moments(rows: List[List[float]]):
        if not rows:
            return [], []
        k = len(rows[0])
        mu = [sum(r[i] for r in rows) / len(rows) for i in range(k)]
        sd = []
        for i in range(k):
            var = sum((r[i] - mu[i]) ** 2 for r in rows) / len(rows)
            # A constant coordinate carries no information; 1.0 keeps it at 0
            # after centring rather than exploding it.
            sd.append(math.sqrt(var) or 1.0)
        return mu, sd

    def _project(self, raw: List[float]) -> torch.Tensor:
        z = torch.tensor([(v - m) / s for v, m, s in zip(raw, self._mu, self._sd)],
                         dtype=torch.float32)
        return z @ self._proj

    def _zero(self) -> torch.Tensor:
        return torch.zeros(self.dim)

    def _leave_one_out(self, question: str, own_id: int) -> torch.Tensor:
        """The query's statistics with the predicted graph removed."""
        feats = self._graphs_of.get(question)
        if not feats:
            return self._zero()
        own = self.feat_of.get(own_id)
        rest = [f for f in feats if f is not own]
        if not rest:
            # Single-graph query: nothing left after removal. Zero is FiLM's
            # identity, so the row degrades to query-only conditioning rather
            # than being shown the target's own statistics.
            return self._zero()
        n_topo = len({(tuple(f['counts']), f['n'], f['m']) for f in rest})
        return self._project(self._aggregate(rest, n_topo))

    # ---- the AnswerEmbeddings surface -------------------------------------

    def __len__(self) -> int:
        return len(self.by_query)

    def get(self, question: str) -> Optional[torch.Tensor]:
        """Full-set statistics for one query, for the rollout pool.

        Leave-one-out does not apply here: rollout graphs are the student's own
        samples, not members of the verified set, so nothing of the target is
        being fed back.
        """
        return self.vec.get((question or '').strip())

    def for_graphs(self, graphs, device) -> torch.Tensor:
        out = []
        for g in graphs:
            question = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
            out.append(self._leave_one_out(question, id(g)))
        return torch.stack(out).to(device).float()

    def as_lookup(self, control: str = 'none', seed: int = 0) -> Callable:
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
            for i, j in enumerate(perm):            # no query keeps its own set
                if i == j and len(perm) > 1:
                    k = (i + 1) % len(perm)
                    perm[i], perm[k] = perm[k], perm[i]
            remap = {k: keys[perm[i]] for i, k in enumerate(keys)}

            def _shuffled(graphs, device):
                out = []
                for g in graphs:
                    q = (g.graph.get('task') or '').strip() if hasattr(g, 'graph') else ''
                    out.append(self.vec.get(remap.get(q, ''), self._zero()))
                return torch.stack(out).to(device).float()
            return _shuffled

        raise ValueError(f"unknown privileged control {control!r}")

    # ---- diagnostics ------------------------------------------------------

    def spread_report(self, sample: int = 400, seed: int = 0) -> Dict[str, float]:
        """How much the query vectors actually differ. The go/no-go BEFORE Stage 1.

        ``spread`` is the statistic that matters, NOT the cosine. The features
        are z-scored across queries, so if every query admits the same family
        each coordinate sits at its own mean, every vector becomes ~0, and the
        cosine of two zero vectors is 0 -- not 1. A cosine-based test would
        therefore miss precisely the failure it exists to catch.

        ``spread`` is the median vector norm divided by sqrt(dim), i.e. the
        typical per-coordinate deviation in z-units. Near 0 means the channel is
        constant and the teacher cannot differ from the student however long it
        trains -- the failure mode already measured for 'context'. Around 1 is
        healthy: queries sit a standard deviation apart.

        The mean cosine is still reported, but read it as ORIENTATION spread
        only; centred vectors have a naturally negative mean cosine (they sum to
        zero), and that is expected rather than alarming.
        """
        vs = list(self.vec.values())
        if len(vs) < 2:
            return {'mean': 0.0, 'p95': 0.0, 'spread': 0.0, 'n': len(vs)}
        rng = random.Random(seed)
        if len(vs) > sample:
            vs = rng.sample(vs, sample)
        M = torch.stack(vs)
        norms = M.norm(dim=1)
        spread = float(norms.median()) / math.sqrt(self.dim)
        N = M / norms.clamp_min(1e-9).unsqueeze(1)
        C = N @ N.T
        iu = torch.triu_indices(len(vs), len(vs), offset=1)
        vals = C[iu[0], iu[1]]
        return {'mean': float(vals.mean()),
                'p95': float(vals.quantile(0.95)),
                'spread': spread,
                'n': len(vs)}
