"""OPSD training: privileged teacher finetune, then on-policy self-distillation.

Stage ordering (see also the module docstring in ``experiment/opsd/__init__``):

  Stage 0  cold start on D_exp                    -- unchanged, the only LLM cost
  Stage 1  privileged finetune of the teacher     -- same NLL, same data, +rationale
  Stage 2  OPSD: student rolls out pseudo-graphs, teacher scores them, KL
  Stage 3  evaluation                             -- teacher discarded

Stage 2 makes **no LLM calls at all**. That is the point: it targets the same
objective as the paper's D_pruned (compact graphs that still work) without the
generate -> ablate -> re-execute loop that D_pruned pays for.
"""
import copy
import json
import time
from typing import Dict, List, Optional, Sequence

import torch

from experiment.opsd.generation import Trajectory, unroll
from experiment.opsd.losses import (agreement_stats, distillation_loss,
                                    trajectory_logprob)
from experiment.opsd.privileged import PrivilegedARGDesigner


# --------------------------------------------------------------------------- #
# Stage 1: privileged teacher finetune
# --------------------------------------------------------------------------- #
def train_teacher(args, teacher: PrivilegedARGDesigner, dataloader_train,
                  answer_lookup, dataloader_validate=None) -> Dict:
    """Fit the teacher with the published NLL, plus the rationale channel.

    ``answer_lookup`` maps a batch of graphs to their privileged embeddings; see
    ``experiment.opsd.data.AnswerEmbeddings``.

    ``privilege_gain`` and its standard error are the go/no-go: if the teacher
    does not beat its own no-rationale score on the same validation graphs, the
    rationale is not being used and OPSD has nothing to distil.
    """
    device = args.device
    teacher.to(device)

    if getattr(args, 'privileged_only', False):
        # Train only the new channel; the shared trunk stays exactly the student.
        # Isolates "what the rationale adds" from "more training".
        for name, p in teacher.named_parameters():
            p.requires_grad = name.startswith(('answer_processor', 'film_'))
        params = list(teacher.privileged_parameters())
    else:
        params = [p for p in teacher.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=args.teacher_lr)
    history = []
    best = float('inf')
    best_state = None

    for epoch in range(args.teacher_epochs):
        teacher.train()
        total, batches = 0.0, 0
        for graphs in dataloader_train:
            graphs = _unwrap(graphs)
            if not graphs:
                continue
            task_emb = _stack_task_embeddings(graphs, device)
            ans_emb = answer_lookup(graphs, device)

            optimizer.zero_grad()
            log_ll, _ = teacher(graphs, task_emb, ans_emb)
            loss = -torch.mean(log_ll)
            if torch.isnan(loss):
                continue
            loss.backward()
            if getattr(args, 'clip', True):
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1

        train_loss = total / max(batches, 1)
        row = {"epoch": epoch + 1, "teacher_train_loss": train_loss,
               "privileged_strength": teacher.privileged_strength()}

        if dataloader_validate is not None:
            row.update(_validate_teacher(teacher, dataloader_validate, answer_lookup, device))
            score = row["teacher_val_loss"]
        else:
            score = train_loss

        if score < best:
            best = score
            best_state = copy.deepcopy(teacher.state_dict())

        history.append(row)
        gain, se = row.get('privilege_gain'), row.get('privilege_gain_stderr')
        gain_str = f"gain={gain:+.4f}+/-{se:.4f} " if gain is not None and se == se else ""
        print(f"[teacher] epoch {epoch+1}/{args.teacher_epochs} "
              f"loss={train_loss:.4f} "
              f"val={row.get('teacher_val_loss', float('nan')):.4f} "
              f"{gain_str}|film|={row['privileged_strength']:.4f}")

    if best_state is not None:
        teacher.load_state_dict(best_state)
    for p in teacher.parameters():
        p.requires_grad = True
    return {"history": history, "best": best}


@torch.no_grad()
def _validate_teacher(teacher, loader, answer_lookup, device) -> Dict[str, float]:
    """Teacher NLL with and without the privileged channel, paired per graph.

    The comparison is the SAME teacher on the SAME graph, so pairing the two
    scores cancels graph-difficulty variance -- which dominates when validation
    is only O(10) graphs. Averaging batch means, as this used to, throws that
    pairing away and reports a number with no error bar, which is how a
    one-sigma fluctuation gets read as a verdict.

    ``privilege_gain`` > 0 means the rationale helps. Judge it against
    ``privilege_gain_stderr``: below ~2 sigma the sign is not evidence either way.
    """
    teacher.eval()
    diffs: List[float] = []
    with_priv: List[float] = []
    without_priv: List[float] = []

    for graphs in loader:
        graphs = _unwrap(graphs)
        if not graphs:
            continue
        task_emb = _stack_task_embeddings(graphs, device)
        ans_emb = answer_lookup(graphs, device)
        ll_priv = teacher(graphs, task_emb, ans_emb)[0]        # [B] log-likelihood
        ll_plain = teacher(graphs, task_emb, None)[0]          # [B]
        diffs.extend((ll_priv - ll_plain).tolist())
        with_priv.extend((-ll_priv).tolist())
        without_priv.extend((-ll_plain).tolist())

    n = len(diffs)
    if n == 0:
        return {"teacher_val_loss": float('nan'),
                "teacher_val_loss_no_privilege": float('nan'),
                "privilege_gain": 0.0, "privilege_gain_stderr": float('nan'),
                "privilege_n": 0}

    mean = sum(diffs) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        stderr = (var / n) ** 0.5
    else:
        stderr = float('nan')

    return {"teacher_val_loss": sum(with_priv) / n,
            "teacher_val_loss_no_privilege": sum(without_priv) / n,
            "privilege_gain": mean,
            "privilege_gain_stderr": stderr,
            "privilege_n": n}


# --------------------------------------------------------------------------- #
# Stage 2: on-policy self-distillation
# --------------------------------------------------------------------------- #
def _teacher_forward(teacher, args, task_rep, ans_rep, trajectories):
    """Score a trajectory under the teacher, however the teacher is conditioned.

    'context' (the OPSD paper's own design): teacher and student are ONE model;
    the teacher is simply conditioned on a longer context -- question + reference
    solution -- so ans_rep is its TASK input and there is no FiLM channel and no
    teacher finetune.

    'rationale' / 'sibling_graph' / 'gold_graph': the teacher is a separate
    PrivilegedARGDesigner whose FiLM channel takes ans_rep, and which must be
    finetuned first (zero-init FiLM means an untrained teacher is byte-identical
    to the student and the KL is exactly 0).
    """
    # The shared teacher inherits the student's train() mode, so dropout would
    # make the KL target stochastic. Score in eval mode and restore.
    was_training = teacher.training
    teacher.eval()
    try:
        if getattr(args, 'privileged_source', 'rationale') == 'context':
            return unroll(teacher, ans_rep, forced=trajectories,
                          feed_edge_features=args.feed_edge_features)
        return unroll(teacher, task_rep, answer_embedding=ans_rep, forced=trajectories,
                      feed_edge_features=args.feed_edge_features)
    finally:
        if was_training:
            teacher.train()


def train_opsd(args, student, teacher, queries, anchor_loader=None,
               log_path: Optional[str] = None) -> Dict:
    """Distil the privileged teacher into the query-only student, on-policy.

    ``queries`` is a list of dicts with at least ``task_embedding`` and
    ``answer_embedding`` (see ``build_query_pool``). Pseudo-graphs are re-sampled
    at *every step* from the current weights, so the states the teacher corrects
    are always the ones this policy visits -- that is what separates this from
    off-policy pseudo-labelling, where a fixed set of graphs is generated once
    and reused while the student drifts away from them.
    """
    device = args.device
    student.to(device).train()

    # In --privileged_source context the teacher IS the student (one model, two
    # contexts), so freezing "the teacher" would freeze the thing being trained:
    # every parameter loses requires_grad and loss.backward() dies with
    # "element 0 of tensors does not require grad". Only detach a SEPARATE
    # teacher; the shared one is already safe because _teacher_forward runs under
    # no_grad, which is what actually detaches the KL target.
    shared_teacher = teacher is student
    teacher.to(device)
    if not shared_teacher:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=args.opsd_lr)
    anchor_iter = _cycle(anchor_loader) if anchor_loader is not None else None

    history = []
    for epoch in range(args.opsd_epochs):
        t0 = time.time()
        epoch_stats = {"loss": 0.0, "node_kl": 0.0, "edge_kl": 0.0,
                       "anchor": 0.0, "nodes": 0.0, "edges": 0.0, "batches": 0}
        agree_acc = {"node_argmax_agree": 0.0, "edge_argmax_agree": 0.0}

        for batch in _batches(queries, args.opsd_batch_size, shuffle=True):
            task_emb = torch.stack([q["task_embedding"] for q in batch]).to(device)
            ans_emb = torch.stack([q["answer_embedding"] for q in batch]).to(device)

            # ---- roll out pseudo-graphs on-policy --------------------------
            # No repairs: a repaired graph is not what the policy sampled, so
            # scoring it would make the gradient inconsistent with the state it
            # was collected from. Repairs happen at evaluation only.
            with torch.no_grad():
                rollouts = [
                    unroll(student, task_emb, temperature=args.rollout_temperature,
                           feed_edge_features=args.feed_edge_features)
                    for _ in range(args.rollouts_per_query)
                ]

            trajectories: List[Trajectory] = []
            expand_task, expand_ans = [], []
            for r in rollouts:
                trajectories.extend(r.trajectories)
                expand_task.append(task_emb)
                expand_ans.append(ans_emb)
            task_rep = torch.cat(expand_task, dim=0)
            ans_rep = torch.cat(expand_ans, dim=0)

            trajectories, task_rep, ans_rep = _filter_trajectories(
                trajectories, task_rep, ans_rep, args, teacher)
            if not trajectories:
                continue

            # ---- score the SAME sequence with both models -------------------
            with torch.no_grad():
                teacher_out = _teacher_forward(teacher, args, task_rep, ans_rep,
                                               trajectories)
            student_out = unroll(student, task_rep, forced=trajectories,
                                 feed_edge_features=args.feed_edge_features)

            parts = distillation_loss(student_out, teacher_out,
                                      alpha=args.alpha,
                                      direction=args.kl_direction,
                                      teacher_temperature=args.teacher_temperature,
                                      jsd_beta=args.jsd_beta)
            loss = parts["loss"]

            # ---- ground-truth anchor ---------------------------------------
            anchor_val = 0.0
            if anchor_iter is not None and args.anchor_weight > 0:
                graphs = _unwrap(next(anchor_iter))
                if graphs:
                    a_task = _stack_task_embeddings(graphs, device)
                    log_ll, _ = student(graphs, a_task)
                    anchor = -torch.mean(log_ll)
                    if not torch.isnan(anchor):
                        loss = (1 - args.anchor_weight) * loss + args.anchor_weight * anchor
                        anchor_val = float(anchor)

            optimizer.zero_grad()
            loss.backward()
            if getattr(args, 'clip', True):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            stats = agreement_stats(student_out, teacher_out)
            for k in agree_acc:
                agree_acc[k] += stats[k]
            epoch_stats["loss"] += float(parts["loss"])
            epoch_stats["node_kl"] += float(parts["node_kl"])
            epoch_stats["edge_kl"] += float(parts["edge_kl"])
            epoch_stats["anchor"] += anchor_val
            epoch_stats["nodes"] += sum(len(t) for t in trajectories) / len(trajectories)
            epoch_stats["edges"] += sum(t.num_edges for t in trajectories) / len(trajectories)
            epoch_stats["batches"] += 1

        n = max(epoch_stats["batches"], 1)
        row = {"epoch": epoch + 1,
               "loss": epoch_stats["loss"] / n,
               "node_kl": epoch_stats["node_kl"] / n,
               "edge_kl": epoch_stats["edge_kl"] / n,
               "anchor_nll": epoch_stats["anchor"] / n,
               "mean_nodes": epoch_stats["nodes"] / n,
               "mean_edges": epoch_stats["edges"] / n,
               "node_argmax_agree": agree_acc["node_argmax_agree"] / n,
               "edge_argmax_agree": agree_acc["edge_argmax_agree"] / n,
               "seconds": time.time() - t0}
        history.append(row)
        print(f"[opsd] epoch {row['epoch']}/{args.opsd_epochs} "
              f"loss={row['loss']:.4f} node_kl={row['node_kl']:.4f} "
              f"edge_kl={row['edge_kl']:.4f} anchor={row['anchor_nll']:.4f} "
              f"|V|={row['mean_nodes']:.2f} |E|={row['mean_edges']:.2f} "
              f"agree={row['node_argmax_agree']:.2f}/{row['edge_argmax_agree']:.2f} "
              f"({row['seconds']:.1f}s)")

        if log_path:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

    return {"history": history}


def _filter_trajectories(trajectories, task_rep, ans_rep, args, teacher):
    """Optional pseudo-graph selection.

    ``all``       keep every rollout (pure on-policy).
    ``nonempty``  drop degenerate rollouts with no nodes.
    ``teacher``   keep the top fraction by teacher log-likelihood -- the
                  LLM-free stand-in for the execute-and-check S(Q,G) that
                  D_pruned pays for.
    """
    mode = getattr(args, 'pseudo_filter', 'nonempty')
    keep = (list(range(len(trajectories))) if mode == 'all'
            else [i for i, t in enumerate(trajectories) if len(t) > 0])

    if mode == 'teacher' and keep:
        idx = torch.as_tensor(keep, device=task_rep.device)
        sub = [trajectories[i] for i in keep]
        with torch.no_grad():
            out = _teacher_forward(teacher, args, task_rep.index_select(0, idx),
                                   ans_rep.index_select(0, idx), sub)
            scores = trajectory_logprob(out, reduction="mean")
        k = max(1, int(len(keep) * getattr(args, 'teacher_keep_frac', 0.5)))
        order = torch.argsort(scores, descending=True)[:k].tolist()
        keep = [keep[i] for i in order]

    if len(keep) == len(trajectories):
        return trajectories, task_rep, ans_rep
    idx = torch.as_tensor(keep, device=task_rep.device)
    return ([trajectories[i] for i in keep],
            task_rep.index_select(0, idx),
            ans_rep.index_select(0, idx))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unwrap(batch):
    """DataLoaders in this repo hand back a few different shapes."""
    if isinstance(batch, (list, tuple)) and batch:
        if isinstance(batch[0], dict) and 'G' in batch[0]:
            return [g['G'] for g in batch]
        if isinstance(batch, tuple) and isinstance(batch[0], list):
            return batch[0]
    return list(batch) if isinstance(batch, (list, tuple)) else [batch]


def _stack_task_embeddings(graphs, device) -> torch.Tensor:
    out = []
    for i, g in enumerate(graphs):
        emb = getattr(g, 'task_embedding', None)
        if emb is None:
            emb = g.graph.get('task_embedding')
        if emb is None:
            raise ValueError(f"graph {i} has no task_embedding")
        out.append(emb if isinstance(emb, torch.Tensor) else torch.tensor(emb).float())
    return torch.stack([t.to(device).float() for t in out])


def _batches(items: Sequence, size: int, shuffle: bool = False):
    idx = list(range(len(items)))
    if shuffle:
        import random
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch
