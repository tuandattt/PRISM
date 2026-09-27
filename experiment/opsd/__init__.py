"""On-Policy Self-Distillation (OPSD) for ARG-Designer.

A privileged teacher, conditioned on the query *and its gold rationale*, hands
dense per-step supervision to a student that only ever sees the query. The
supervision is applied at the states the student itself generates
(``pseudo-graphs``), not at teacher-forced ground-truth states.

Modules
-------
generation  the single unrolling routine used for BOTH student sampling and
            teacher scoring -- the reason the two stay step-aligned
privileged  the privileged-conditioning teacher
losses      node/edge KL plus the ground-truth anchor
train_opsd  the OPSD training loop
"""
from experiment.opsd.generation import Trajectory, Unrolled, unroll, trajectory_to_graph
from experiment.opsd.privileged import PrivilegedARGDesigner, build_teacher

__all__ = [
    "Trajectory", "Unrolled", "unroll", "trajectory_to_graph",
    "PrivilegedARGDesigner", "build_teacher",
]
