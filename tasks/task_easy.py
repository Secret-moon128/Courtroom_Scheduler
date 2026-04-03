"""
Task 1 — Easy
=============
Objective: Schedule all 5 cases with zero constraint violations.

Setup:
  - 5 cases (no mandatory deadlines)
  - 3 judges (generous availability)
  - 2 courtrooms
  - 20 time slots

Success criterion: All 5 cases scheduled without hard constraint violations.
Score range: 0.0 (nothing scheduled) → 1.0 (all 5 scheduled, no violations)
Difficulty: EASY — agent has wide availability windows and no deadline pressure.
"""

from __future__ import annotations
from env.environment import CourtSchedulerEnv


EASY_CONFIG = dict(
    n_cases=5,
    n_judges=3,
    n_rooms=2,
    total_slots=20,
    max_steps=30,
    deadline_fraction=0.0,
    seed=42,
)


def make_easy_task() -> CourtSchedulerEnv:
    """Return a freshly-reset easy-difficulty environment."""
    env = CourtSchedulerEnv(**EASY_CONFIG)
    env.reset()
    return env