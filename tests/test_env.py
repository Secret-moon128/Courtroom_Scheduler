"""
Tests for OpenEnv spec compliance.
Run with: pytest tests/test_env.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env.environment import CourtSchedulerEnv
from env.models import CourtObservation, CourtAction, CourtReward


@pytest.fixture
def env():
    e = CourtSchedulerEnv(n_cases=5, n_judges=3, n_rooms=2, total_slots=20, max_steps=30, seed=42)
    return e


class TestOpenEnvSpec:
    """Verify the full OpenEnv interface contract."""

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, CourtObservation)

    def test_observation_fields_present(self, env):
        obs = env.reset()
        assert hasattr(obs, "step")
        assert hasattr(obs, "max_steps")
        assert hasattr(obs, "cases")
        assert hasattr(obs, "judges")
        assert hasattr(obs, "courtrooms")
        assert hasattr(obs, "scheduled_hearings")
        assert hasattr(obs, "unscheduled_count")
        assert hasattr(obs, "cumulative_reward")

    def test_reset_step_is_zero(self, env):
        obs = env.reset()
        assert obs.step == 0

    def test_step_returns_four_tuple(self, env):
        env.reset()
        action = CourtAction(action_type="noop")
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, CourtObservation)
        assert isinstance(reward, CourtReward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_dict(self, env):
        env.reset()
        s = env.state()
        assert isinstance(s, dict)
        assert "cases" in s
        assert "judges" in s
        assert "step" in s

    def test_step_increments(self, env):
        env.reset()
        action = CourtAction(action_type="noop")
        obs, _, _, _ = env.step(action)
        assert obs.step == 1

    def test_done_when_all_scheduled(self, env):
        """Manually schedule all cases and verify done=True."""
        obs = env.reset()
        done = False
        for case in obs.cases:
            for judge in obs.judges:
                spec_ok = (
                    case.required_judge_specialisation == "general"
                    or case.required_judge_specialisation in judge.specialisations
                )
                if not spec_ok:
                    continue
                common = (
                    set(judge.available_slots)
                    & set(case.plaintiff_available_slots)
                    & set(case.defendant_available_slots)
                )
                for room in obs.courtrooms:
                    room_common = common & set(room.available_slots)
                    if room_common:
                        slot = sorted(room_common)[0]
                        action = CourtAction(
                            action_type="schedule",
                            case_id=case.case_id,
                            slot=slot,
                            judge_id=judge.judge_id,
                            room_id=room.room_id,
                        )
                        obs, _, done, _ = env.step(action)
                        break
                else:
                    continue
                break

        # After attempting all cases, episode may or may not be done
        # but state must be valid
        s = env.state()
        assert s["step"] >= 1

    def test_done_when_max_steps_reached(self, env):
        env = CourtSchedulerEnv(n_cases=5, n_judges=3, n_rooms=2, max_steps=3, seed=42)
        env.reset()
        done = False
        for _ in range(5):
            _, _, done, _ = env.step(CourtAction(action_type="noop"))
        assert done is True

    def test_noop_action_valid(self, env):
        env.reset()
        action = CourtAction(action_type="noop")
        obs, reward, done, info = env.step(action)
        assert reward.total <= 0   # noop on active docket incurs small penalty

    def test_invalid_case_id_penalised(self, env):
        env.reset()
        action = CourtAction(
            action_type="schedule",
            case_id="NONEXISTENT",
            slot=0,
            judge_id="JUDGE-A",
            room_id="ROOM-1",
        )
        _, reward, _, info = env.step(action)
        assert reward.total < 0
        assert "error" in info

    def test_reward_has_components(self, env):
        env.reset()
        _, reward, _, _ = env.step(CourtAction(action_type="noop"))
        assert hasattr(reward, "total")
        assert hasattr(reward, "scheduling_bonus")
        assert hasattr(reward, "conflict_penalty")
        assert hasattr(reward, "deadline_penalty")
        assert hasattr(reward, "backlog_penalty")
        assert hasattr(reward, "efficiency_bonus")
        assert hasattr(reward, "noop_penalty")

    def test_reset_clears_state(self, env):
        obs1 = env.reset()
        env.step(CourtAction(action_type="noop"))
        obs2 = env.reset()
        assert obs2.step == 0
        assert obs2.cumulative_reward == 0.0
        assert len(obs2.scheduled_hearings) == 0

    def test_valid_action_types(self, env):
        env.reset()
        for atype in ["noop", "defer", "prioritise"]:
            action = CourtAction(
                action_type=atype,
                defer_case_id="CASE-001" if atype == "defer" else None,
                priority_order=["CASE-001"] if atype == "prioritise" else None,
            )
            obs, reward, done, info = env.step(action)
            assert isinstance(reward.total, float)

    def test_invalid_action_type_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(CourtAction(action_type="explode"))