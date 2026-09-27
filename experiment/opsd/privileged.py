"""The privileged teacher.

The teacher is the *same* ARG-Designer, conditioned on the query plus the gold
rationale. Only ``encode_condition`` differs, so every other tensor in the
generation path -- the node GRU, the metric-learning role head, the edge GRU --
is shared, and the node/edge output spaces are identical. That is what makes the
KL exact and cheap.

Why the rationale and not the final answer
------------------------------------------
The conditioning channel is one mean-pooled MiniLM vector. Appending "18" to a
GSM8K question moves that vector almost not at all, so a teacher conditioned on
the final label would be indistinguishable from the student and there would be
nothing to distil. ``gsm_data_process`` already exposes the step-by-step
solution as ``item["step"]``; that is the privileged text.

Why a separate branch and not concatenated text
-----------------------------------------------
Encoding ``question + " " + rationale`` as one string lets the query dominate
the pooled embedding. A dedicated branch gives the privileged signal its own
capacity, and FiLM makes the fusion multiplicative rather than a blend that the
task vector can wash out.

Zero-init
---------
``film_gamma`` and ``film_beta`` start at zero, so a freshly built teacher is
*exactly* the student it was initialised from and the KL starts at 0. The
privileged finetune is what opens the gap. If it does not open -- watch teacher
NLL against the student baseline -- there is no teacher and OPSD cannot help.
"""
from typing import Optional

import torch
import torch.nn as nn

from experiment.model import ARGDesigner


class PrivilegedARGDesigner(ARGDesigner):
    """ARG-Designer with an extra, training-only conditioning channel."""

    accepts_privileged = True

    def __init__(self, args, data_statistics):
        super().__init__(args, data_statistics)
        d = self.embedding_dim
        p_drop = getattr(args, 'privileged_dropout_p', 0.1)

        # Rank of the privileged bottleneck. Cold start yields O(100) graphs, so
        # a full-width channel (two d x d FiLM heads plus a d x d encoder, ~592k
        # parameters at d=384) is thousands of parameters per training example
        # and memorises the set instead of learning from the rationale. Squeezing
        # f_A through a rank-r bottleneck first cuts that by ~30x at r=16.
        # rank <= 0 restores the full-width channel for the ablation.
        rank = int(getattr(args, 'privileged_rank', 16))
        self.privileged_rank = rank

        if rank > 0:
            self.answer_processor = nn.Sequential(
                nn.LayerNorm(d),
                nn.Dropout(p_drop),
                nn.Linear(d, rank),
                nn.Tanh(),
            )
            fan = rank
        else:
            self.answer_processor = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d, d),
            )
            fan = d

        self.film_gamma = nn.Linear(fan, d)
        self.film_beta = nn.Linear(fan, d)
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight); nn.init.zeros_(self.film_beta.bias)

        # Probability of hiding the privileged channel during teacher training.
        # Keeps a query-only mode alive inside the teacher, which is what the
        # "privileged dropout" ablation (Baseline 0) evaluates.
        self.privileged_dropout = float(getattr(args, 'privileged_dropout', 0.0))

    def encode_condition(self, task_embedding, answer_embedding: Optional[torch.Tensor] = None):
        base = self.task_processor(task_embedding)
        if answer_embedding is None:
            return base

        if self.training and self.privileged_dropout > 0:
            keep = (torch.rand(answer_embedding.size(0), 1, device=answer_embedding.device)
                    >= self.privileged_dropout).float()
        else:
            keep = 1.0

        a = self.answer_processor(answer_embedding)
        gamma = self.film_gamma(a)
        beta = self.film_beta(a)
        return base * (1 + gamma * keep) + beta * keep

    def privileged_parameters(self):
        """Only the parameters the privileged channel introduces."""
        yield from self.answer_processor.parameters()
        yield from self.film_gamma.parameters()
        yield from self.film_beta.parameters()

    def privileged_strength(self) -> float:
        """How far the FiLM modulation has moved from the identity.

        A number that stays near zero after the privileged finetune means the
        teacher is ignoring the rationale -- the go/no-go signal.
        """
        with torch.no_grad():
            g = self.film_gamma.weight.norm().item() + self.film_gamma.bias.norm().item()
            b = self.film_beta.weight.norm().item() + self.film_beta.bias.norm().item()
        return g + b


def build_teacher(args, data_statistics, student_state_dict=None,
                  device=None) -> PrivilegedARGDesigner:
    """Build a teacher, optionally initialised from a student checkpoint.

    Initialising from the cold-start checkpoint (rather than from scratch) keeps
    teacher and student in the same basin, so the KL is a correction rather than
    a fight between two unrelated modes -- and it is far cheaper.
    """
    teacher = PrivilegedARGDesigner(args, data_statistics)
    if student_state_dict is not None:
        missing, unexpected = teacher.load_state_dict(student_state_dict, strict=False)
        introduced = {n for n, _ in teacher.named_parameters()
                      if n.startswith(('answer_processor', 'film_'))}
        unexpected_real = [k for k in unexpected]
        missing_real = [k for k in missing if k not in introduced]
        if unexpected_real or missing_real:
            raise ValueError(
                f"Teacher/student weight mismatch beyond the privileged channel.\n"
                f"  missing:    {missing_real}\n"
                f"  unexpected: {unexpected_real}")
    if device is not None:
        teacher = teacher.to(device)
    return teacher
