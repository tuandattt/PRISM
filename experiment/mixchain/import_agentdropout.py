"""Convert AgentDropout topologies into this repo's cold-start graph format.

AgentDropout optimises node and edge masks over a multi-agent graph using
execution feedback, so its topologies are chosen rather than rejection-sampled.
This turns its output into the ``.pt`` files every downstream stage already
reads, so the graphs can be used either as

    --data_dir   training targets / OPSD anchor      (supervised)
    --privileged_source gold_graph                   (teacher conditioning)

INPUT: one JSON object per line, with a query, its roles, and its edges::

    {"question": "...", "roles": ["Math Solver", "Inspector"],
     "edges": [[0, 1]], "id": 17}

``adj`` (an NxN 0/1 matrix) is accepted in place of ``edges``. ``id`` and
``mode`` are optional. Every role must be one of the four in the shared math
vocabulary; an unknown role is an error rather than a silent remap to node 0,
because a wrong role trains the designer to build the wrong crew.

WHAT THIS PRINTS, AND WHY IT MATTERS
------------------------------------
The report ends with the number of DISTINCT topologies across queries. That is
the go/no-go for using these graphs as privileged information:

    distinct == 1     the PI is the same vector for every query. That is
                      exactly ``--privileged_control constant``, the null
                      control. A conditional designer cannot learn anything
                      query-specific from it; it collapses to one fixed
                      topology. Use the graphs as training targets instead.

    distinct >> 1     the topology varies with the query, so there is a
                      query -> topology mapping to learn.

AgentDropout's published setup learns masks per DATASET, not per query, so
check this number before building anything on top of it.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import networkx as nx
import torch

from experiment.mixchain.mixchain_prompt_set import ROLE_DESCRIPTION
from experiment.utils import convert_to_pyg_graph, save_graph_with_features


def edges_of(rec, n):
    """Edge list from either 'edges' pairs or an 'adj' matrix."""
    if rec.get('edges') is not None:
        return [(int(u), int(v)) for u, v in rec['edges']]
    adj = rec.get('adj')
    if adj is None:
        raise ValueError("record has neither 'edges' nor 'adj'")
    return [(i, j) for i in range(n) for j in range(n)
            if i != j and adj[i][j]]


def to_dag(g):
    """Drop back-edges until acyclic.

    The generator factorises P(G) over a BFS ordering, so a cyclic target has no
    valid decoding order and the loader would silently repair it anyway. Doing
    it here means the repair is reported instead of hidden.
    """
    removed = 0
    while not nx.is_directed_acyclic_graph(g):
        cycle = next(nx.simple_cycles(g))
        g.remove_edge(cycle[-1], cycle[0])
        removed += 1
    return removed


def signature(g):
    """Canonical (roles, edges) identity, for counting distinct topologies."""
    roles = tuple(g.nodes[i]['role'] for i in sorted(g.nodes()))
    return (roles, tuple(sorted(g.edges())))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='AgentDropout output, JSONL')
    p.add_argument('--output_dir', default='data/AgentDropoutData_mixchain')
    p.add_argument('--dataset', default='mixchain')
    p.add_argument('--mode', default='AgentDropout',
                   help="Recorded as the graph's blueprint name")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    known = set(ROLE_DESCRIPTION)

    written, skipped, repaired = 0, 0, 0
    sigs, sizes = Counter(), Counter()
    with open(a.input, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            question = (rec.get('question') or rec.get('task') or '').strip()
            roles = rec.get('roles') or []
            if not question or not roles:
                skipped += 1
                continue
            bad = [r for r in roles if r not in known]
            if bad:
                raise SystemExit(
                    f"line {lineno}: unknown role(s) {sorted(set(bad))}. The math "
                    f"vocabulary is {sorted(known)}. Map AgentDropout's agent names "
                    f"onto these before importing -- a silent remap would train the "
                    f"designer to build the wrong crew.")

            g = nx.DiGraph()
            for i, role in enumerate(roles):
                g.add_node(i, role=role)
            g.add_edges_from(edges_of(rec, len(roles)))
            repaired += 1 if to_dag(g) else 0

            sigs[signature(g)] += 1
            sizes[(len(roles), g.number_of_edges())] += 1

            pyg = convert_to_pyg_graph(g, question)
            rid = rec.get('id', lineno)
            name = "_".join(map(str, [a.dataset, rid, a.mode, len(roles), True]))
            save_graph_with_features(pyg, os.path.join(a.output_dir, f"{name}.pt"), {
                "mode": a.mode,
                "agent_nums": len(roles),
                "is_correct": True,   # filename must contain True to be loaded
                "question": question,
            })
            written += 1

    print(f"wrote {written} graphs to {a.output_dir}  "
          f"({skipped} skipped, {repaired} had a cycle removed)")
    if not written:
        raise SystemExit("Nothing imported; check the input format.")

    mean_v = sum(v * c for (v, _), c in sizes.items()) / written
    mean_e = sum(e * c for (_, e), c in sizes.items()) / written
    print(f"mean |V| {mean_v:.2f}, mean |E| {mean_e:.2f}")
    print(f"distinct topologies: {len(sigs)} over {written} queries")
    top = sigs.most_common(1)[0]
    print(f"most common covers {100.0*top[1]/written:.1f}% of queries: "
          f"{top[0][0]} with {len(top[0][1])} edges")
    if len(sigs) == 1:
        print("\nGO/NO-GO: every query got the SAME topology. As privileged "
              "information this is")
        print("a constant vector -- identical to --privileged_control constant, "
              "the null control.")
        print("A conditional designer cannot learn a query -> topology mapping "
              "from it. Use these")
        print("graphs as --data_dir training targets instead.")
    elif len(sigs) < 0.05 * written:
        print(f"\nWARNING: only {len(sigs)} distinct topologies for {written} "
              f"queries. The channel")
        print("carries little query-specific signal; expect most of any gain to "
              "be unconditional.")


if __name__ == '__main__':
    main()
