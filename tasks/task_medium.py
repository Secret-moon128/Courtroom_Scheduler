"""
Task 2 — Medium
===============
Objective: Clear a 15-case backlog with party availability conflicts,
           prioritising older/higher-priority cases first.

Setup:
  - 15 cases (10% have mandatory deadlines)
  - 4 judges (moderate availability, specialisation constraints)
  - 3 courtrooms
  - 30 time slots
  - Judges are busier (40% slot busy rate)

Success criterion:
  - Primary:   >= 80% of cases scheduled (>= 12/15)
  - Secondary: Higher-priority cases (priority 1-2) scheduled before lower ones
  - Penalty:   Any missed mandatory deadline costs score

Score range: 0.0 → 1.0
  0.0  = fewer than 5 cases scheduled
  0.5  = 12 cases scheduled (80%), arbitrary ordering
  1.0  = all 15 scheduled, high-priority cases first, no deadline violations

Difficulty: MEDIUM — tighter availability, specialisation mismatches, ordering matters.
"""

from __future__ import annotations
from env.environment import CourtSchedulerEnv


MEDIUM_CONFIG = dict(
    n_cases=15,
    n_judges=4,
    n_rooms=3,
    total_slots=30,
    max_steps=60,
    deadline_fraction=0.10,
    seed=1337,
)


def make_medium_task() -> CourtSchedulerEnv:
    """Return a freshly-reset medium-difficulty environment."""
    env = CourtSchedulerEnv(**MEDIUM_CONFIG)
    env.reset()
    return env