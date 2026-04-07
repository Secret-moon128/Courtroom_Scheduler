"""
Graders for all 3 tasks.
Scores are strictly in (0.0, 1.0) — never exactly 0.0 or 1.0.
"""

from __future__ import annotations
from typing import Any

SCORE_MIN = 0.01
SCORE_MAX = 0.99


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return round(min(SCORE_MAX, max(SCORE_MIN, score)), 4)


def _pct_scheduled(state: dict) -> float:
    cases = state["cases"]
    if not cases:
        return 0.5
    return sum(1 for c in cases if c["is_scheduled"]) / len(cases)


def _missed_deadlines(state: dict) -> int:
    overdue = state.get("overdue_count", 0)
    if overdue > 0:
        return overdue
    return sum(
        1 for c in state["cases"]
        if c.get("mandatory_deadline") is not None and not c["is_scheduled"]
    )


def _priority_ordering_score(state: dict) -> float:
    hearings  = state["scheduled_hearings"]
    cases_map = {c["case_id"]: c for c in state["cases"]}

    if len(hearings) < 2:
        return 0.85   # trivially ordered but not perfect

    sorted_hearings = sorted(hearings, key=lambda h: h["slot"])
    n_pairs = len(sorted_hearings) - 1
    correctly_ordered = 0

    for i in range(n_pairs):
        c_early = cases_map.get(sorted_hearings[i]["case_id"])
        c_late  = cases_map.get(sorted_hearings[i+1]["case_id"])
        if c_early and c_late:
            if c_early["priority"] <= c_late["priority"]:
                correctly_ordered += 1

    raw = correctly_ordered / n_pairs if n_pairs > 0 else 0.85
    # Scale to (0.1, 0.9) so it never contributes exact 0 or 1
    return 0.1 + raw * 0.8


class EasyGrader:
    task_id  = "easy"
    max_cases = 5

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        pct = _pct_scheduled(state)

        scheduled_count = sum(1 for c in state["cases"] if c["is_scheduled"])
        expected_reward = scheduled_count * 1.0
        actual_reward   = state.get("cumulative_reward", 0.0)
        reward_ratio    = min(0.99, max(0.01, actual_reward / max(expected_reward, 0.01)))

        raw = 0.7 * pct + 0.3 * reward_ratio
        return _clamp(raw)


class MediumGrader:
    task_id       = "medium"
    max_cases     = 15
    min_scheduled = 8

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        cases       = state["cases"]
        n_total     = len(cases)
        n_scheduled = sum(1 for c in cases if c["is_scheduled"])

        if n_scheduled < MediumGrader.min_scheduled:
            sched_score = (n_scheduled / MediumGrader.min_scheduled) * 0.35
        else:
            sched_score = 0.35 + 0.65 * (
                (n_scheduled - MediumGrader.min_scheduled) /
                max(1, n_total - MediumGrader.min_scheduled)
            )

        ordering_score = _priority_ordering_score(state)
        missed         = _missed_deadlines(state)
        deadline_score = max(0.05, 1.0 - 0.20 * missed)

        raw = 0.50 * sched_score + 0.30 * ordering_score + 0.20 * deadline_score
        return _clamp(raw)


class HardGrader:
    task_id       = "hard"
    max_cases     = 30
    min_scheduled = 10

    @staticmethod
    def score(state: dict[str, Any]) -> float:
        cases       = state["cases"]
        n_total     = len(cases)
        n_scheduled = sum(1 for c in cases if c["is_scheduled"])

        if n_scheduled < HardGrader.min_scheduled:
            return _clamp((n_scheduled / HardGrader.min_scheduled) * 0.15)

        sched_score    = n_scheduled / n_total
        missed         = _missed_deadlines(state)
        deadline_score = max(0.05, 1.0 - 0.15 * missed)
        ordering_score = _priority_ordering_score(state)

        scheduled_cases = [c for c in cases if c["is_scheduled"]]
        if scheduled_cases:
            avg_pending = sum(c["days_pending"] for c in scheduled_cases) / len(scheduled_cases)
            all_pending = sum(c["days_pending"] for c in cases) / max(len(cases), 1)
            age_score   = min(0.95, avg_pending / max(all_pending, 1))
        else:
            age_score = 0.05

        raw = (
            0.40 * sched_score +
            0.35 * deadline_score +
            0.15 * ordering_score +
            0.10 * age_score
        )
        return _clamp(raw)


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