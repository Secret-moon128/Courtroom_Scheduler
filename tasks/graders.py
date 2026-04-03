"""
Graders for all 3 tasks.

Each grader:
  - Accepts the final environment state dict (from env.state())
  - Returns a float in [0.0, 1.0]
  - Is deterministic (same inputs -> same output, always)
  - Never returns the same score for all inputs
  - Rewards partial progress (not just binary pass/fail)
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _pct_scheduled(state: dict) -> float:
    cases = state["cases"]
    if not cases:
        return 0.0
    return sum(1 for c in cases if c["is_scheduled"]) / len(cases)


def _missed_deadlines(state: dict) -> int:
    """
    Count cases with a mandatory_deadline that were never scheduled.
    Prefers overdue_count from state (set by live env) so synthetic test states
    that explicitly pass overdue_count are respected.
    """
    overdue = state.get("overdue_count", 0)
    if overdue > 0:
        return overdue
    return sum(
        1 for c in state["cases"]
        if c.get("mandatory_deadline") is not None and not c["is_scheduled"]
    )


def _priority_ordering_score(state: dict) -> float:
    """
    Score how well the agent prioritised.
    Compares priority of scheduled cases: earlier slots should have higher-priority cases.
    Returns 0.0-1.0.
    """
    hearings = state["scheduled_hearings"]
    cases_map = {c["case_id"]: c for c in state["cases"]}

    if len(hearings) < 2:
        return 1.0   # trivially ordered

    sorted_hearings = sorted(hearings, key=lambda h: h["slot"])
    n_pairs = len(sorted_hearings) - 1
    correctly_ordered = 0

    for i in range(n_pairs):
        c_early = cases_map.get(sorted_hearings[i]["case_id"])
        c_late  = cases_map.get(sorted_hearings[i+1]["case_id"])
        if c_early and c_late:
            # Lower priority number = more urgent -> should be scheduled first
            if c_early["priority"] <= c_late["priority"]:
                correctly_ordered += 1

    return correctly_ordered / n_pairs if n_pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

class EasyGrader:
    """
    Task 1 (Easy) grader.

    Scoring:
      - 0.7 weight: fraction of cases scheduled (0.0-1.0)
      - 0.3 weight: zero constraint violations (estimated from reward ratio)

    Score: 0.0 -> 1.0
    """

    task_id = "easy"
    max_cases = 5

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        pct = _pct_scheduled(state)

        scheduled_count = sum(1 for c in state["cases"] if c["is_scheduled"])
        expected_reward = scheduled_count * 1.0
        actual_reward   = state.get("cumulative_reward", 0.0)
        reward_ratio = min(1.0, max(0.0, actual_reward / max(expected_reward, 0.01)))

        score = 0.7 * pct + 0.3 * reward_ratio
        return round(min(1.0, max(0.0, score)), 4)


class MediumGrader:
    """
    Task 2 (Medium) grader.

    Scoring:
      - 0.50 weight: % cases scheduled (partial credit below threshold)
      - 0.30 weight: priority ordering score
      - 0.20 weight: deadline compliance (-0.1 per missed deadline)

    Score: 0.0 -> 1.0
    """

    task_id = "medium"
    max_cases = 15
    min_scheduled = 8

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        cases         = state["cases"]
        n_total       = len(cases)
        n_scheduled   = sum(1 for c in cases if c["is_scheduled"])

        # Scheduling completeness — partial credit below threshold
        if n_scheduled < MediumGrader.min_scheduled:
            sched_score = (n_scheduled / MediumGrader.min_scheduled) * 0.35
        else:
            sched_score = 0.35 + 0.65 * (
                (n_scheduled - MediumGrader.min_scheduled) /
                max(1, n_total - MediumGrader.min_scheduled)
            )

        # Priority ordering
        ordering_score = _priority_ordering_score(state)

        # Deadline compliance — use overdue_count so synthetic states work
        missed = _missed_deadlines(state)
        deadline_score = max(0.0, 1.0 - 0.20 * missed)

        score = 0.50 * sched_score + 0.30 * ordering_score + 0.20 * deadline_score
        return round(min(1.0, max(0.0, score)), 4)


class HardGrader:
    """
    Task 3 (Hard) grader.

    Scoring:
      - 0.40 weight: % cases scheduled (0 if < 10/30)
      - 0.35 weight: mandatory deadline compliance (-0.12 per missed)
      - 0.15 weight: priority ordering
      - 0.10 weight: backlog age (avg days_pending of scheduled cases)

    Score: 0.0 -> 1.0
    """

    task_id = "hard"
    max_cases = 30
    min_scheduled = 10

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        cases       = state["cases"]
        n_total     = len(cases)
        n_scheduled = sum(1 for c in cases if c["is_scheduled"])

        # Hard floor
        if n_scheduled < HardGrader.min_scheduled:
            return round((n_scheduled / HardGrader.min_scheduled) * 0.15, 4)

        # Scheduling completeness
        sched_score = n_scheduled / n_total

        # Deadline compliance — use overdue_count
        missed = _missed_deadlines(state)
        deadline_score = max(0.0, 1.0 - 0.15 * missed)

        # Priority ordering
        ordering_score = _priority_ordering_score(state)

        # Backlog age: reward scheduling older cases
        scheduled_cases = [c for c in cases if c["is_scheduled"]]
        if scheduled_cases:
            avg_pending = sum(c["days_pending"] for c in scheduled_cases) / len(scheduled_cases)
            all_pending = sum(c["days_pending"] for c in cases) / max(len(cases), 1)
            age_score = min(1.0, avg_pending / max(all_pending, 1))
        else:
            age_score = 0.0

        score = (
            0.40 * sched_score +
            0.35 * deadline_score +
            0.15 * ordering_score +
            0.10 * age_score
        )
        return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = [
    {
        "task_id":    "easy",
        "name":       "Clear 5-Case Docket",
        "difficulty": "easy",
        "grader":     EasyGrader,
        "description": (
            "Schedule all 5 cases to valid (slot, judge, room) triples "
            "with no constraint violations. No mandatory deadlines."
        ),
    },
    {
        "task_id":    "medium",
        "name":       "Backlog Prioritisation",
        "difficulty": "medium",
        "grader":     MediumGrader,
        "description": (
            "Clear a 15-case docket with party conflicts and 10% mandatory deadlines. "
            "Schedule at least 12 cases, prioritising older/higher-priority cases first."
        ),
    },
    {
        "task_id":    "hard",
        "name":       "Surge Docket + Statutory Deadlines",
        "difficulty": "hard",
        "grader":     HardGrader,
        "description": (
            "Manage a 30-case surge with 30% mandatory statutory deadlines, "
            "scarce judicial resources, and high party conflict rates. "
            "All deadline cases must be scheduled on time."
        ),
    },
]