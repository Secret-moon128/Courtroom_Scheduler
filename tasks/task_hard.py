"""
Task 3 — Hard
=============
Objective: Manage a 30-case surge docket with 5+ mandatory statutory deadlines,
           limited judicial resources, and high party conflict rates.

Setup:
  - 30 cases (30% have mandatory deadlines — statutory requirements)
  - 5 judges (busy: 45% slots unavailable, strict specialisation)
  - 3 courtrooms (busy: 25% slots unavailable)
  - 40 time slots
  - Agent has only 80 steps to clear as many cases as possible

Success criterion:
  - Mandatory deadline cases MUST be scheduled before their deadline (hard penalty if missed)
  - Primary metric: % of cases scheduled
  - Secondary metric: zero missed statutory deadlines
  - Tertiary metric: average days_pending of scheduled cases (lower = better)

Score range: 0.0 → 1.0
  0.0  = fewer than 10 cases scheduled OR any statutory deadline missed
  0.3  = 50% cases scheduled, some deadlines missed
  0.7  = 80% cases scheduled, all deadlines met
  1.0  = 90%+ cases scheduled, all deadlines met, backlog prioritised by age

Difficulty: HARD — combinatorial constraints, deadline urgency, resource scarcity.
"""

from __future__ import annotations
from env.environment import CourtSchedulerEnv


HARD_CONFIG = dict(
    n_cases=30,
    n_judges=5,
    n_rooms=3,
    total_slots=40,
    max_steps=80,
    deadline_fraction=0.30,
    seed=9999,
)


def make_hard_task() -> CourtSchedulerEnv:
    """Return a freshly-reset hard-difficulty environment."""
    env = CourtSchedulerEnv(**HARD_CONFIG)
    env.reset()
    return env