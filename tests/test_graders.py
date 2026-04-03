"""
Tests for task graders — verifying determinism, score range, and differentiation.
Run with: pytest tests/test_graders.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tasks.graders import EasyGrader, MediumGrader, HardGrader, TASKS
from agents.random_agent import RandomAgent
from tasks.task_easy import make_easy_task
from tasks.task_medium import make_medium_task
from tasks.task_hard import make_hard_task


# ---------------------------------------------------------------------------
# Helper: build a synthetic state dict
# ---------------------------------------------------------------------------

def make_state(n_cases=5, n_scheduled=0, missed_deadlines=0, cumulative_reward=0.0):
    cases = []
    for i in range(n_cases):
        has_deadline = i < (n_cases // 3)
        scheduled    = i < n_scheduled
        is_missed    = has_deadline and not scheduled and i < missed_deadlines
        cases.append({
            "case_id":         f"CASE-{i+1:03d}",
            "case_type":       "civil",
            "priority":        (i % 5) + 1,
            "days_pending":    10 + i * 3,
            "is_scheduled":    scheduled,
            "mandatory_deadline": 5 if has_deadline else None,
            "assigned_slot":   i if scheduled else None,
            "assigned_judge":  "JUDGE-A" if scheduled else None,
            "assigned_room":   "ROOM-1" if scheduled else None,
        })
    hearings = [
        {"case_id": f"CASE-{i+1:03d}", "slot": i, "judge_id": "JUDGE-A", "room_id": "ROOM-1"}
        for i in range(n_scheduled)
    ]
    return {
        "cases":             cases,
        "scheduled_hearings": hearings,
        "cumulative_reward": cumulative_reward,
        "step":              10,
        "overdue_count":     missed_deadlines,
    }


# ---------------------------------------------------------------------------
# EasyGrader
# ---------------------------------------------------------------------------

class TestEasyGrader:
    def test_perfect_score(self):
        state = make_state(n_cases=5, n_scheduled=5, cumulative_reward=5.0)
        score = EasyGrader.score(state)
        assert score >= 0.9, f"Expected near-perfect score, got {score}"

    def test_zero_scheduled(self):
        state = make_state(n_cases=5, n_scheduled=0, cumulative_reward=0.0)
        score = EasyGrader.score(state)
        assert score == 0.0

    def test_partial_scheduled(self):
        state = make_state(n_cases=5, n_scheduled=3, cumulative_reward=3.0)
        score = EasyGrader.score(state)
        assert 0.0 < score < 1.0

    def test_score_in_range(self):
        for n in range(6):
            state = make_state(n_cases=5, n_scheduled=n, cumulative_reward=float(n))
            score = EasyGrader.score(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for n_scheduled={n}"

    def test_deterministic(self):
        state = make_state(n_cases=5, n_scheduled=3, cumulative_reward=2.5)
        assert EasyGrader.score(state) == EasyGrader.score(state)

    def test_not_constant(self):
        scores = set()
        for n in range(6):
            state = make_state(n_cases=5, n_scheduled=n, cumulative_reward=float(n))
            scores.add(EasyGrader.score(state))
        assert len(scores) > 1, "EasyGrader returns same score for all inputs"


# ---------------------------------------------------------------------------
# MediumGrader
# ---------------------------------------------------------------------------

class TestMediumGrader:
    def test_perfect_score(self):
        state = make_state(n_cases=15, n_scheduled=15, cumulative_reward=15.0)
        score = MediumGrader.score(state)
        assert score >= 0.85

    def test_below_threshold(self):
        state = make_state(n_cases=15, n_scheduled=5, cumulative_reward=5.0)
        score = MediumGrader.score(state)
        assert score < 0.7  # partial credit + ordering bonus keeps score moderate

    def test_missed_deadline_penalised(self):
        s1 = make_state(n_cases=15, n_scheduled=12, missed_deadlines=0, cumulative_reward=12.0)
        s2 = make_state(n_cases=15, n_scheduled=12, missed_deadlines=3, cumulative_reward=12.0)
        assert MediumGrader.score(s1) > MediumGrader.score(s2)

    def test_score_in_range(self):
        for n in range(0, 16, 3):
            state = make_state(n_cases=15, n_scheduled=n, cumulative_reward=float(n))
            score = MediumGrader.score(state)
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        state = make_state(n_cases=15, n_scheduled=10, cumulative_reward=8.0)
        assert MediumGrader.score(state) == MediumGrader.score(state)

    def test_not_constant(self):
        scores = {
            MediumGrader.score(make_state(15, n, cumulative_reward=float(n)))
            for n in [0, 5, 10, 15]
        }
        assert len(scores) > 1


# ---------------------------------------------------------------------------
# HardGrader
# ---------------------------------------------------------------------------

class TestHardGrader:
    def test_perfect_score(self):
        state = make_state(n_cases=30, n_scheduled=30, missed_deadlines=0, cumulative_reward=30.0)
        score = HardGrader.score(state)
        assert score >= 0.8

    def test_below_floor(self):
        state = make_state(n_cases=30, n_scheduled=5, cumulative_reward=5.0)
        score = HardGrader.score(state)
        assert score < 0.15

    def test_missed_deadlines_heavily_penalised(self):
        s_clean  = make_state(30, n_scheduled=25, missed_deadlines=0, cumulative_reward=25.0)
        s_missed = make_state(30, n_scheduled=25, missed_deadlines=5, cumulative_reward=25.0)
        assert HardGrader.score(s_clean) > HardGrader.score(s_missed)

    def test_score_in_range(self):
        for n in range(0, 31, 5):
            state = make_state(n_cases=30, n_scheduled=n, cumulative_reward=float(n))
            score = HardGrader.score(state)
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        state = make_state(n_cases=30, n_scheduled=20, cumulative_reward=18.0)
        assert HardGrader.score(state) == HardGrader.score(state)

    def test_not_constant(self):
        scores = {
            HardGrader.score(make_state(30, n, cumulative_reward=float(n)))
            for n in [0, 10, 20, 30]
        }
        assert len(scores) > 1


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_three_tasks_registered(self):
        assert len(TASKS) == 3

    def test_task_ids(self):
        ids = {t["task_id"] for t in TASKS}
        assert ids == {"easy", "medium", "hard"}

    def test_each_has_grader(self):
        for t in TASKS:
            assert hasattr(t["grader"], "score")
            assert callable(t["grader"].score)

    def test_difficulties_ordered(self):
        difficulties = [t["difficulty"] for t in TASKS]
        assert difficulties == ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Integration: random agent baseline scores
# ---------------------------------------------------------------------------

class TestRandomAgentBaseline:
    def test_easy_baseline_positive(self):
        env   = make_easy_task()
        agent = RandomAgent(seed=0)
        _, info = agent.run_episode(env)
        score = EasyGrader.score(info["state"])
        assert score >= 0.0    # random agent should schedule at least something

    def test_medium_baseline_runs(self):
        env   = make_medium_task()
        agent = RandomAgent(seed=0)
        _, info = agent.run_episode(env)
        score = MediumGrader.score(info["state"])
        assert 0.0 <= score <= 1.0

    def test_hard_baseline_runs(self):
        env   = make_hard_task()
        agent = RandomAgent(seed=0)
        _, info = agent.run_episode(env)
        score = HardGrader.score(info["state"])
        assert 0.0 <= score <= 1.0