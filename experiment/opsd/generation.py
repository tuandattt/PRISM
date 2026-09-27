"""One unrolling routine, used for both student sampling and teacher scoring.

Why this module exists
----------------------
The KL in OPSD compares the teacher's and the student's distribution *at the
same decision point*. Two things in the published code make that hard to get
right, and both are avoided here by construction:

1. ``ARGDesigner.forward`` pushes every graph through ``BFS_Graph_to_Adj_Matrix``,
   which re-permutes nodes into a canonical BFS order. A pseudo-graph sampled
   on-policy is not in BFS order, so re-canonicalising it would have the teacher
   scoring a *different* sequence than the student produced.

2. ``ARGDesigner.sample`` and ``ARGDesigner.forward`` do not agree on the input
   at each step (see NOTE below). Scoring a rollout with ``forward`` semantics
   after sampling it with ``sample`` semantics compares two different policies.

``unroll`` therefore drives the model step by step over a *given generation
order*, and can either sample the next action or be forced onto a recorded one.
Student rollout and teacher scoring are the same function with a different
``forced`` argument.

NOTE -- which semantics this follows
------------------------------------
Where ``forward`` (training) and ``sample`` (inference) disagree, this follows
``forward``, because that is the objective the weights were actually fitted to:

* the START row's output predicts node 0 (``sample`` feeds an extra all-zero row
  first and predicts node 0 from that);
* the node input carries ``f_edge`` -- the previous node's connectivity row --
  (``sample`` leaves those dimensions zero at every step).

Two quirks of ``forward`` are reproduced deliberately rather than "fixed", so
that teacher, student and the published baseline all describe the same process:

* the history embedding at node step ``i`` aggregates roles ``0..i-2``, not
  ``0..i-1`` as Eq. 7 states -- an off-by-one in the reference implementation;
* the edge decision for node ``i`` is conditioned on the node-GRU output that
  *predicted* node ``i``, so it never sees which role was actually sampled.

Edge distribution
-----------------
``output_edge`` is a softmax over 4 channels laid out by ``graph_to_matrix`` as
``[edge, no-edge, START, END]``. At a real slot the decision is binary, so the
policy used here is the renormalised two-way distribution over
``[edge, no-edge]``; the terminal END-edge step carries no decision and is not
scored.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import networkx as nx
import torch
import torch.nn.functional as F

EDGE_CH = 0      # channel written by _process_edges_binary for a real edge
NO_EDGE_CH = 1   # complement channel (valid slot, no edge)


@dataclass
class Trajectory:
    """One pseudo-graph, recorded as the action sequence that produced it.

    Attributes
    ----------
    roles : role ids of the generated agents, in generation order.
    ended : whether the policy emitted END (False means it hit ``max_nodes``).
    edges : ``edges[i][j]`` is 1 if an edge was drawn from node ``i-j-1`` to
            node ``i``. ``len(edges[i]) == min(i, max_prev_node)``.
    """
    roles: List[int] = field(default_factory=list)
    ended: bool = False
    edges: List[List[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.roles)

    @property
    def num_edges(self) -> int:
        return sum(sum(row) for row in self.edges)


@dataclass
class Unrolled:
    """Per-step distributions and actions for a batch of trajectories.

    Shapes, with ``S`` the longest node-step count in the batch and ``K``
    ``max_prev_node``:

    node_logits  [B, S, R+1]   over roles + END
    node_action  [B, S]        index taken (R == END)
    node_mask    [B, S]        step is a real decision for this row
    edge_logits  [B, S, K, 2]  over [edge, no-edge]
    edge_action  [B, S, K]     0 == edge, 1 == no edge
    edge_mask    [B, S, K]     slot is a real decision for this row
    """
    trajectories: List[Trajectory]
    node_logits: torch.Tensor
    node_action: torch.Tensor
    node_mask: torch.Tensor
    edge_logits: torch.Tensor
    edge_action: torch.Tensor
    edge_mask: torch.Tensor


def _candidate_matrix(model) -> torch.Tensor:
    """Role embeddings in role-id order, with END appended. [R+1, d]"""
    num_roles = len(model.role_to_id)
    rows = list(range(num_roles)) + [model.END_TOKEN]
    idx = torch.as_tensor(rows, device=model.full_embedding_matrix.device)
    return model.full_embedding_matrix.index_select(0, idx)


def _edge_feature_row(edge_bits: Sequence[int], node_index: int, K: int, LE: int,
                      device) -> torch.Tensor:
    """Rebuild the flattened ``f_edge`` row that ``graph_to_matrix`` produces.

    Slot ``j`` of node ``i`` is valid when ``j < i``; channel 0 marks an edge,
    channel 1 marks a valid slot without one.
    """
    row = torch.zeros(K, LE, device=device)
    for j in range(min(node_index, K)):
        if j < len(edge_bits) and edge_bits[j]:
            row[j, EDGE_CH] = 1.0
        else:
            row[j, NO_EDGE_CH] = 1.0
    return row.view(-1)


def unroll(model,
           task_embedding: torch.Tensor,
           answer_embedding: Optional[torch.Tensor] = None,
           forced: Optional[Sequence[Trajectory]] = None,
           forced_prefix: Optional[int] = None,
           max_nodes: Optional[int] = None,
           min_nodes: Optional[int] = None,
           temperature: float = 1.0,
           greedy: bool = False,
           feed_edge_features: bool = True) -> Unrolled:
    """Drive the generator over a batch of queries.

    ``forced=None``  -> sample (on-policy rollout, produces the pseudo-graphs).
    ``forced=[...]`` -> replay those trajectories, returning the logits the model
                        assigns at each of their decision points.

    ``forced_prefix=k`` with ``forced`` set is the EXPOSURE-BIAS probe: replay
    the first k node steps of the recorded trajectory, then let the policy run
    free. k=0 is an ordinary rollout and k>=len(traj) is a full replay, so
    sweeping k interpolates between "generates from its own prefix" and
    "generates from a ground-truth prefix" -- which is exactly the train /
    inference mismatch teacher forcing hides.

    ``task_embedding`` is the *raw* sentence embedding [B, d]; conditioning is
    applied inside via ``model.encode_condition`` so the privileged teacher can
    fold in ``answer_embedding``.
    """
    device = next(model.parameters()).device
    task_embedding = task_embedding.to(device)
    if answer_embedding is not None:
        answer_embedding = answer_embedding.to(device)

    B = task_embedding.size(0)
    d = model.embedding_dim
    K = model.num_nodes_to_consider
    LE = model.len_edge_vec
    R = len(model.role_to_id)
    feature_len = d + K * LE

    stats = model.data_statistics
    if max_nodes is None:
        max_nodes = int(stats.get("max_num_nodes", 10))
    if min_nodes is None:
        min_nodes = int(stats.get("min_num_nodes", 2))

    if forced is not None:
        assert len(forced) == B, "one forced trajectory per row"
        if forced_prefix is None:
            steps = max((len(t) + (1 if t.ended else 0)) for t in forced)
            steps = max(steps, 1)
        else:
            # Free-running after the prefix needs the full budget, not the
            # recorded trajectory's length.
            steps = max_nodes + 1
    else:
        steps = max_nodes + 1                      # nodes plus the END step

    cond = model.encode_condition(task_embedding, answer_embedding)      # [B, d]
    candidates = model.role_processor(_candidate_matrix(model))          # [R+1, d]

    h_node = torch.zeros(1, B, model.args.hidden_size_node_level_transformer, device=device)
    hist_state = torch.zeros(1, B, d, device=device)
    hist_len = 0                                   # roles already folded into hist_state

    roles: List[List[int]] = [[] for _ in range(B)]
    role_embs: List[List[torch.Tensor]] = [[] for _ in range(B)]
    edge_bits: List[List[List[int]]] = [[] for _ in range(B)]
    ended = [False] * B
    alive = [True] * B

    node_logits_steps, node_action_steps, node_mask_steps = [], [], []
    edge_logits_steps, edge_action_steps, edge_mask_steps = [], [], []

    for i in range(steps):
        # -- history must cover roles[0 : i-1] (the reference off-by-one) ------
        target_hist = max(0, i - 1)
        while hist_len < target_hist:
            rows = [b for b in range(B) if len(role_embs[b]) > hist_len]
            if not rows:
                break
            idx = torch.as_tensor(rows, device=device)
            step_in = torch.stack([role_embs[b][hist_len] for b in rows], dim=0).unsqueeze(1)
            _, new_h = model.prev_nodes_aggregator(step_in, hist_state.index_select(1, idx).contiguous())
            # index_copy (not slice assignment): an in-place write here would
            # clobber a tensor an earlier step's backward pass still needs.
            hist_state = hist_state.index_copy(1, idx, new_h)
            hist_len += 1

        # -- assemble this step's input ---------------------------------------
        step_input = torch.zeros(B, 1, feature_len, device=device)
        if i == 0:
            step_input[:, 0, :d] = cond
            step_input[:, 0, d + LE - 2] = 1.0                # START edge token
        else:
            step_input[:, 0, :d] = model._history_gate(hist_state.squeeze(0), cond)
            if feed_edge_features:
                for b in range(B):
                    if len(edge_bits[b]) >= i:
                        step_input[b, 0, d:] = _edge_feature_row(
                            edge_bits[b][i - 1], i - 1, K, LE, device)

        node_out, h_node = model.node_gru(model.node_project(step_input), h_node)

        # -- node distribution -------------------------------------------------
        intent = model.output_node(node_out).squeeze(1)                  # [B, d]
        logits = intent @ candidates.t()                                 # [B, R+1]
        node_logits_steps.append(logits)

        end_idx = R
        mask_row = torch.zeros(B, dtype=torch.bool, device=device)
        action_row = torch.full((B,), end_idx, dtype=torch.long, device=device)

        forcing = forced is not None and (forced_prefix is None or i < forced_prefix)
        if forcing:
            for b in range(B):
                traj = forced[b]
                if i < len(traj):
                    action_row[b] = traj.roles[i]
                    mask_row[b] = True
                elif i == len(traj) and traj.ended:
                    action_row[b] = end_idx
                    mask_row[b] = True
                else:
                    alive[b] = False
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            if i < min_nodes:                       # cannot terminate too early
                probs = probs.clone()
                probs[:, end_idx] = 0.0
            if i >= max_nodes:                      # must terminate
                probs = torch.zeros_like(probs)
                probs[:, end_idx] = 1.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            picked = probs.argmax(dim=-1) if greedy else torch.multinomial(probs, 1).squeeze(-1)
            for b in range(B):
                if not alive[b]:
                    continue
                action_row[b] = picked[b]
                mask_row[b] = True
        node_action_steps.append(action_row)
        node_mask_steps.append(mask_row)

        # -- commit the node ---------------------------------------------------
        for b in range(B):
            if not mask_row[b]:
                continue
            a = int(action_row[b])
            if a == end_idx:
                ended[b] = True
                alive[b] = False
            else:
                roles[b].append(a)
                role_embs[b].append(model.full_embedding_matrix[a].detach())

        # -- edge distributions for the node just added ------------------------
        slots = min(i, K)
        e_logits = torch.zeros(B, K, 2, device=device)
        e_action = torch.ones(B, K, dtype=torch.long, device=device)      # default: no edge
        e_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        node_added = torch.tensor([bool(mask_row[b]) and int(action_row[b]) != end_idx
                                   for b in range(B)], device=device)

        if slots > 0 and bool(node_added.any()):
            h_edge = model.embedding_node_to_edge(node_out).transpose(0, 1).contiguous()
            edge_in = torch.zeros(B, 1, LE, device=device)
            edge_in[:, 0, LE - 2] = 1.0                                   # START edge token

            drawn = [[0] * slots for _ in range(B)]
            for j in range(slots):
                edge_out, h_edge = model.edge_gru(model.edge_project(edge_in), h_edge)
                full = model.output_edge.mlp[0](edge_out).squeeze(1)       # pre-softmax [B, LE]
                pair = full[:, [EDGE_CH, NO_EDGE_CH]]                      # [B, 2]
                e_logits[:, j, :] = pair

                if forcing:
                    take = torch.ones(B, dtype=torch.long, device=device)
                    for b in range(B):
                        if node_added[b] and j < len(forced[b].edges[i]):
                            take[b] = 0 if forced[b].edges[i][j] else 1
                else:
                    p_edge = F.softmax(pair / max(temperature, 1e-6), dim=-1)[:, 0]
                    # take == 0 means "draw the edge"
                    if greedy:
                        take = (p_edge < 0.5).long()
                    else:
                        take = 1 - torch.bernoulli(p_edge).long()

                for b in range(B):
                    if node_added[b]:
                        drawn[b][j] = int(take[b] == 0)
                        e_action[b, j] = take[b]
                        e_mask[b, j] = True

                nxt = torch.zeros(B, 1, LE, device=device)
                chan = torch.where(take == 0,
                                   torch.full_like(take, EDGE_CH),
                                   torch.full_like(take, NO_EDGE_CH))
                nxt[torch.arange(B, device=device), 0, chan] = 1.0
                edge_in = nxt

            for b in range(B):
                if node_added[b]:
                    edge_bits[b].append(drawn[b])
        else:
            for b in range(B):
                if node_added[b]:
                    edge_bits[b].append([])

        edge_logits_steps.append(e_logits)
        edge_action_steps.append(e_action)
        edge_mask_steps.append(e_mask)

        # Once past the forced prefix the policy is free-running, so the
        # normal termination check applies again.
        if not forcing and not any(alive):
            break

    trajectories = [Trajectory(roles=roles[b], ended=ended[b], edges=edge_bits[b])
                    for b in range(B)]

    return Unrolled(
        trajectories=trajectories,
        node_logits=torch.stack(node_logits_steps, dim=1),
        node_action=torch.stack(node_action_steps, dim=1),
        node_mask=torch.stack(node_mask_steps, dim=1),
        edge_logits=torch.stack(edge_logits_steps, dim=1),
        edge_action=torch.stack(edge_action_steps, dim=1),
        edge_mask=torch.stack(edge_mask_steps, dim=1),
    )


def trajectory_to_graph(traj: Trajectory, model, repair: bool = True) -> nx.DiGraph:
    """Materialise a trajectory as the DiGraph the MAS runtime executes.

    ``repair=False`` returns the policy's raw output and is what OPSD trains on
    -- repairing a sampled graph would make the scored sequence differ from the
    one the policy actually produced. Repairs are applied only at evaluation,
    where a disconnected graph cannot be run.
    """
    G = nx.DiGraph()
    for n, role_id in enumerate(traj.roles):
        role = model.id_to_role[role_id]
        G.add_node(n, label=role_id, role=role)
        emb = model.full_embedding_matrix[role_id].detach().cpu().numpy()
        G.nodes[n]['role_embedding'] = emb

    for i, row in enumerate(traj.edges):
        for j, bit in enumerate(row):
            if bit:
                u = i - j - 1
                if 0 <= u < len(traj.roles):
                    G.add_edge(u, i, label=1)

    if not repair or G.number_of_nodes() == 0:
        return G

    if not nx.is_directed_acyclic_graph(G):
        for cycle in nx.simple_cycles(G):
            if len(cycle) >= 2 and G.has_edge(cycle[0], cycle[1]):
                G.remove_edge(cycle[0], cycle[1])

    if G.number_of_nodes() > 1:
        comps = list(nx.weakly_connected_components(G))
        if len(comps) > 1:
            main = max(comps, key=len)
            for comp in comps:
                if comp is main:
                    continue
                src, tgt = min(comp), min(main)
                u, v = (src, tgt) if src < tgt else (tgt, src)
                G.add_edge(u, v, label=1)
    return G
