"""Distillation objectives over pseudo-graph decision points.

The student is supervised at every node step and every edge slot of a graph it
generated itself. Both heads are small categoricals -- roles + END for nodes,
{edge, no-edge} for edges -- so the divergence is exact, with no vocabulary
alignment and no sampling estimate.

Direction
---------
``reverse`` is the default. The student samples its own states, so coverage of
its modes is already handled by the state distribution; what is wanted at each
state is a decisive crew, which is what mode-seeking KL(S||T) gives. ``forward``
(KL(T||S), mass-covering) and ``jsd`` are available for the ablation.

Temperature
-----------
``teacher_temperature > 1`` softens the target. The student can never see the
rationale, so an irreducible gap remains; asking it to match a confident
teacher exactly pushes it to over-confident predictions on queries where it
genuinely cannot tell. Letting the target carry more entropy is the cheap guard,
alongside the ground-truth anchor in ``train_opsd``.
"""
from typing import Dict

import torch
import torch.nn.functional as F


def _masked_mean(per_step: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1)
    return (per_step * mask).sum() / denom


def _kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """KL(p || q) over the last dim, both given as log-probabilities."""
    return (p_log.exp() * (p_log - q_log)).sum(dim=-1)


def divergence(student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               mask: torch.Tensor,
               direction: str = "reverse",
               teacher_temperature: float = 1.0,
               jsd_beta: float = 0.5) -> torch.Tensor:
    """Per-decision divergence, averaged over the masked decision points."""
    if mask.sum() == 0:
        return student_logits.sum() * 0.0

    s_log = F.log_softmax(student_logits, dim=-1)
    t_log = F.log_softmax(teacher_logits / max(teacher_temperature, 1e-6), dim=-1)

    if direction == "reverse":            # KL(student || teacher), mode-seeking
        per = _kl(s_log, t_log)
    elif direction == "forward":          # KL(teacher || student), mass-covering
        per = _kl(t_log, s_log)
    elif direction == "jsd":
        m_log = torch.log(jsd_beta * t_log.exp() + (1 - jsd_beta) * s_log.exp() + 1e-12)
        per = jsd_beta * _kl(t_log, m_log) + (1 - jsd_beta) * _kl(s_log, m_log)
    else:
        raise ValueError(f"unknown direction {direction!r}")

    return _masked_mean(per, mask.float())


def distillation_loss(student, teacher_out, alpha: float = 0.2,
                      direction: str = "reverse",
                      teacher_temperature: float = 1.0,
                      jsd_beta: float = 0.5) -> Dict[str, torch.Tensor]:
    """Node + edge divergence, weighted as in Eq. 19.

    ``student`` and ``teacher_out`` are ``Unrolled`` objects produced from the
    SAME trajectories, so their masks agree elementwise.
    """
    node = divergence(student.node_logits, teacher_out.node_logits, student.node_mask,
                      direction, teacher_temperature, jsd_beta)

    edge = divergence(student.edge_logits.reshape(-1, 2),
                      teacher_out.edge_logits.reshape(-1, 2),
                      student.edge_mask.reshape(-1),
                      direction, teacher_temperature, jsd_beta)

    total = alpha * node + (1 - alpha) * edge
    return {"loss": total, "node_kl": node.detach(), "edge_kl": edge.detach()}


def trajectory_logprob(out, reduction: str = "sum") -> torch.Tensor:
    """log P(trajectory) under the model that produced ``out``. [B]

    Used to score pseudo-graphs with the teacher (teacher-as-verifier) without
    spending a single LLM call.
    """
    node_lp = F.log_softmax(out.node_logits, dim=-1)
    node_taken = node_lp.gather(-1, out.node_action.unsqueeze(-1)).squeeze(-1)
    node_taken = node_taken * out.node_mask.float()

    edge_lp = F.log_softmax(out.edge_logits, dim=-1)
    edge_taken = edge_lp.gather(-1, out.edge_action.unsqueeze(-1)).squeeze(-1)
    edge_taken = edge_taken * out.edge_mask.float()

    total = node_taken.sum(dim=1) + edge_taken.sum(dim=(1, 2))
    if reduction == "mean":
        counts = out.node_mask.sum(dim=1) + out.edge_mask.sum(dim=(1, 2))
        return total / counts.clamp_min(1).float()
    return total


def agreement_stats(student, teacher_out) -> Dict[str, float]:
    """Cheap diagnostics: do teacher and student even disagree?

    If ``node_argmax_agree`` sits at 1.0 the privileged channel is inert and the
    distillation has nothing to transfer.
    """
    with torch.no_grad():
        nm = student.node_mask
        s_arg = student.node_logits.argmax(-1)
        t_arg = teacher_out.node_logits.argmax(-1)
        node_agree = ((s_arg == t_arg) & nm).sum().float() / nm.sum().clamp_min(1)

        em = student.edge_mask
        s_e = student.edge_logits.argmax(-1)
        t_e = teacher_out.edge_logits.argmax(-1)
        edge_agree = ((s_e == t_e) & em).sum().float() / em.sum().clamp_min(1)

    return {"node_argmax_agree": float(node_agree), "edge_argmax_agree": float(edge_agree)}
