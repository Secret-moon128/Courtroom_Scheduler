"""
CourtSchedulerEnv — OpenEnv-compliant Courtroom Case Scheduling Negotiator.

API:
    env.reset()          -> CourtObservation
    env.step(action)     -> (CourtObservation, CourtReward, bool, dict)
    env.state()          -> dict  (full internal state snapshot)

Episode:
    Each step the agent emits one CourtAction.
    The episode ends when all cases are scheduled OR max_steps is reached.
"""

from __future__ import annotations
from typing import Any

from .models import (
    Case, Judge, Courtroom, ScheduledHearing,
    CourtObservation, CourtAction, CourtReward,
)
from .simulator import (
    generate_cases, generate_judges, generate_courtrooms, build_conflict_hints
)


class CourtSchedulerEnv:
    """
    Courtroom Case Scheduling Negotiator — OpenEnv environment.

    The agent acts as a court scheduling clerk: it must assign cases
    to (slot, judge, room) triples while satisfying hard constraints
    (party availability, judge specialisation, room availability) and
    optimising for backlog clearance, priority ordering, and statutory deadlines.
    """

    VALID_ACTION_TYPES = {"schedule", "reschedule", "prioritise", "defer", "noop"}

    def __init__(
        self,
        n_cases: int          = 10,
        n_judges: int         = 3,
        n_rooms: int          = 2,
        total_slots: int      = 20,
        max_steps: int        = 40,
        deadline_fraction: float = 0.0,
        seed: int             = 42,
    ) -> None:
        self.n_cases           = n_cases
        self.n_judges          = n_judges
        self.n_rooms           = n_rooms
        self.total_slots       = total_slots
        self.max_steps         = max_steps
        self.deadline_fraction = deadline_fraction
        self.seed              = seed

        # Populated by reset()
        self._cases: list[Case]                          = []
        self._judges: list[Judge]                        = []
        self._rooms: list[Courtroom]                     = []
        self._scheduled: list[ScheduledHearing]          = []
        self._conflict_hints: dict[str, list[str]]       = {}
        self._step: int                                  = 0
        self._cumulative_reward: float                   = 0.0
        self._slot_judge_map: dict[tuple[int,str], bool] = {}  # (slot, judge_id) -> occupied
        self._slot_room_map: dict[tuple[int,str], bool]  = {}  # (slot, room_id)  -> occupied

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> CourtObservation:
        """Reset the environment to initial state and return first observation."""
        self._cases   = generate_cases(
            self.n_cases, self.total_slots,
            deadline_fraction=self.deadline_fraction,
            seed=self.seed,
        )
        self._judges  = generate_judges(self.n_judges, self.total_slots, seed=self.seed)
        self._rooms   = generate_courtrooms(self.n_rooms, self.total_slots, seed=self.seed)
        self._conflict_hints = build_conflict_hints(
            self._cases, self._judges, self._rooms, self.total_slots
        )
        self._scheduled           = []
        self._step                = 0
        self._cumulative_reward   = 0.0
        self._slot_judge_map      = {}
        self._slot_room_map       = {}

        return self._make_observation()

    def step(self, action: CourtAction) -> tuple[CourtObservation, CourtReward, bool, dict]:
        """
        Execute one action.

        Returns:
            observation: CourtObservation
            reward:      CourtReward (with component breakdown)
            done:        bool
            info:        dict with diagnostic details
        """
        if action.action_type not in self.VALID_ACTION_TYPES:
            raise ValueError(f"Unknown action_type: {action.action_type!r}")

        reward, info = self._apply_action(action)

        # Step deadline penalties: any case whose deadline == current step and not yet scheduled
        for case in self._cases:
            if (
                not case.is_scheduled
                and case.mandatory_deadline is not None
                and case.mandatory_deadline <= self._step
            ):
                penalty = -2.0
                reward.deadline_penalty += penalty
                reward.total += penalty
                info.setdefault("deadline_violations", []).append(case.case_id)

        # Per-step backlog penalty
        unscheduled = sum(1 for c in self._cases if not c.is_scheduled)
        if unscheduled > 0:
            bp = -0.1
            reward.backlog_penalty += bp
            reward.total += bp

        self._step += 1
        self._cumulative_reward += reward.total

        done = self._is_done()
        obs  = self._make_observation()
        obs.episode_done = done

        return obs, reward, done, info

    def state(self) -> dict[str, Any]:
        """Return full internal state snapshot (for debugging / logging)."""
        return {
            "step":               self._step,
            "max_steps":          self.max_steps,
            "seed":               self.seed,
            "total_slots":        self.total_slots,
            "cumulative_reward":  self._cumulative_reward,
            "cases":              [c.model_dump() for c in self._cases],
            "judges":             [j.model_dump() for j in self._judges],
            "courtrooms":         [r.model_dump() for r in self._rooms],
            "scheduled_hearings": [s.model_dump() for s in self._scheduled],
            "unscheduled_count":  sum(1 for c in self._cases if not c.is_scheduled),
            "overdue_count":      self._count_overdue(),
        }

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: CourtAction) -> tuple[CourtReward, dict]:
        info: dict[str, Any] = {"action_type": action.action_type}

        if action.action_type == "schedule":
            return self._action_schedule(action, info)

        if action.action_type == "reschedule":
            return self._action_reschedule(action, info)

        if action.action_type == "prioritise":
            return self._action_prioritise(action, info)

        if action.action_type == "defer":
            return self._action_defer(action, info)

        # noop
        unscheduled = sum(1 for c in self._cases if not c.is_scheduled)
        noop_pen = -0.05 if unscheduled > 0 else 0.0
        return CourtReward(total=noop_pen, noop_penalty=noop_pen), info

    def _action_schedule(self, action: CourtAction, info: dict) -> tuple[CourtReward, dict]:
        reward = CourtReward(total=0.0)

        case    = self._find_case(action.case_id)
        judge   = self._find_judge(action.judge_id)
        room    = self._find_room(action.room_id)
        slot    = action.slot

        if case is None:
            reward.conflict_penalty -= 0.5
            reward.total -= 0.5
            info["error"] = f"Unknown case_id: {action.case_id}"
            return reward, info

        if case.is_scheduled:
            reward.conflict_penalty -= 0.3
            reward.total -= 0.3
            info["error"] = f"{action.case_id} already scheduled"
            return reward, info

        violations = self._check_constraints(case, judge, room, slot)
        if violations:
            penalty = -0.5 * len(violations)
            reward.conflict_penalty += penalty
            reward.total += penalty
            info["constraint_violations"] = violations
            return reward, info

        # Commit the scheduling
        case.is_scheduled   = True
        case.assigned_slot  = slot
        case.assigned_judge = judge.judge_id
        case.assigned_room  = room.room_id

        self._slot_judge_map[(slot, judge.judge_id)] = True
        self._slot_room_map[(slot, room.room_id)]    = True
        self._scheduled.append(ScheduledHearing(
            case_id=case.case_id, slot=slot,
            judge_id=judge.judge_id, room_id=room.room_id
        ))

        reward.scheduling_bonus += 1.0
        reward.total            += 1.0

        # Efficiency bonus: scheduling higher-priority cases earlier
        if case.priority <= 2:
            reward.efficiency_bonus += 0.2
            reward.total            += 0.2

        # Extra bonus for clearing a deadline case on time
        if case.mandatory_deadline is not None and slot <= case.mandatory_deadline:
            reward.efficiency_bonus += 0.5
            reward.total            += 0.5
            info["deadline_met"] = case.case_id

        info["scheduled"] = case.case_id
        return reward, info

    def _action_reschedule(self, action: CourtAction, info: dict) -> tuple[CourtReward, dict]:
        """Unschedule a case then re-schedule it at a new slot."""
        reward = CourtReward(total=0.0)

        case = self._find_case(action.case_id)
        if case is None or not case.is_scheduled:
            reward.conflict_penalty -= 0.3
            reward.total -= 0.3
            info["error"] = f"Cannot reschedule {action.case_id}"
            return reward, info

        # Free old slot
        old_slot  = case.assigned_slot
        old_judge = case.assigned_judge
        old_room  = case.assigned_room
        self._slot_judge_map.pop((old_slot, old_judge), None)
        self._slot_room_map.pop((old_slot, old_room), None)
        self._scheduled = [s for s in self._scheduled if s.case_id != case.case_id]

        case.is_scheduled   = False
        case.assigned_slot  = None
        case.assigned_judge = None
        case.assigned_room  = None

        # Small penalty for rescheduling (administrative cost)
        reward.conflict_penalty -= 0.1
        reward.total -= 0.1

        # Now schedule at new slot
        judge = self._find_judge(action.judge_id)
        room  = self._find_room(action.room_id)
        slot  = action.slot

        violations = self._check_constraints(case, judge, room, slot)
        if violations:
            penalty = -0.5 * len(violations)
            reward.conflict_penalty += penalty
            reward.total += penalty
            info["constraint_violations"] = violations
            return reward, info

        case.is_scheduled   = True
        case.assigned_slot  = slot
        case.assigned_judge = judge.judge_id
        case.assigned_room  = room.room_id

        self._slot_judge_map[(slot, judge.judge_id)] = True
        self._slot_room_map[(slot, room.room_id)]    = True
        self._scheduled.append(ScheduledHearing(
            case_id=case.case_id, slot=slot,
            judge_id=judge.judge_id, room_id=room.room_id
        ))

        reward.scheduling_bonus += 0.8   # slightly less than fresh schedule
        reward.total            += 0.8
        info["rescheduled"] = case.case_id
        return reward, info

    def _action_prioritise(self, action: CourtAction, info: dict) -> tuple[CourtReward, dict]:
        """Reorder internal case priority list. Provides a small planning bonus."""
        reward = CourtReward(total=0.0)

        if not action.priority_order:
            info["error"] = "priority_order list is empty"
            return reward, info

        order = {cid: idx for idx, cid in enumerate(action.priority_order)}
        valid = [c for c in self._cases if c.case_id in order]

        if not valid:
            info["error"] = "No known case_ids in priority_order"
            return reward, info

        # Check ordering quality: are higher-priority (lower number) cases listed first?
        correctly_ordered = sum(
            1 for i in range(len(valid) - 1)
            if order[valid[i].case_id] < order[valid[i+1].case_id]
            and valid[i].priority <= valid[i+1].priority
        )
        bonus = 0.05 * correctly_ordered
        reward.efficiency_bonus += bonus
        reward.total            += bonus
        info["prioritise_bonus"] = bonus
        return reward, info

    def _action_defer(self, action: CourtAction, info: dict) -> tuple[CourtReward, dict]:
        """
        Defer a case (mark it low priority for this step).
        Small penalty if the case has a mandatory deadline approaching.
        """
        reward = CourtReward(total=0.0)

        case = self._find_case(action.defer_case_id)
        if case is None:
            info["error"] = f"Unknown case_id: {action.defer_case_id}"
            reward.conflict_penalty -= 0.2
            reward.total -= 0.2
            return reward, info

        if case.mandatory_deadline is not None:
            remaining = case.mandatory_deadline - self._step
            if remaining <= 3:
                penalty = -0.3
                reward.conflict_penalty += penalty
                reward.total += penalty
                info["warning"] = f"Deferring {case.case_id} with only {remaining} steps to deadline"

        info["deferred"] = case.case_id
        return reward, info

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def _check_constraints(
        self,
        case: Case | None,
        judge: Judge | None,
        room: Courtroom | None,
        slot: int | None,
    ) -> list[str]:
        violations: list[str] = []

        if case is None:
            violations.append("invalid case_id")
            return violations
        if judge is None:
            violations.append("invalid judge_id")
        if room is None:
            violations.append("invalid room_id")
        if slot is None or slot < 0 or slot >= self.total_slots:
            violations.append(f"slot {slot} out of range [0, {self.total_slots-1}]")

        if violations:
            return violations

        # Judge specialisation
        spec = case.required_judge_specialisation
        if spec != "general" and spec not in judge.specialisations:
            violations.append(
                f"Judge {judge.judge_id} lacks specialisation '{spec}'"
            )

        # Judge slot availability
        if slot not in judge.available_slots:
            violations.append(f"Judge {judge.judge_id} not available at slot {slot}")

        # Room slot availability
        if slot not in room.available_slots:
            violations.append(f"Room {room.room_id} not available at slot {slot}")

        # Double-booking checks
        if self._slot_judge_map.get((slot, judge.judge_id)):
            violations.append(f"Judge {judge.judge_id} double-booked at slot {slot}")

        if self._slot_room_map.get((slot, room.room_id)):
            violations.append(f"Room {room.room_id} double-booked at slot {slot}")

        # Party availability
        if slot not in case.plaintiff_available_slots:
            violations.append(f"Plaintiff not available at slot {slot}")

        if slot not in case.defendant_available_slots:
            violations.append(f"Defendant not available at slot {slot}")

        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> CourtObservation:
        return CourtObservation(
            step=self._step,
            max_steps=self.max_steps,
            cases=self._cases,
            unscheduled_count=sum(1 for c in self._cases if not c.is_scheduled),
            overdue_count=self._count_overdue(),
            judges=self._judges,
            courtrooms=self._rooms,
            scheduled_hearings=self._scheduled,
            conflict_hints=self._conflict_hints,
            cumulative_reward=self._cumulative_reward,
        )

    def _is_done(self) -> bool:
        all_scheduled = all(c.is_scheduled for c in self._cases)
        return all_scheduled or self._step >= self.max_steps

    def _count_overdue(self) -> int:
        return sum(
            1 for c in self._cases
            if not c.is_scheduled
            and c.mandatory_deadline is not None
            and c.mandatory_deadline < self._step
        )

    def _find_case(self, case_id: str | None) -> Case | None:
        if case_id is None:
            return None
        return next((c for c in self._cases if c.case_id == case_id), None)

    def _find_judge(self, judge_id: str | None) -> Judge | None:
        if judge_id is None:
            return None
        return next((j for j in self._judges if j.judge_id == judge_id), None)

    def _find_room(self, room_id: str | None) -> Courtroom | None:
        if room_id is None:
            return None
        return next((r for r in self._rooms if r.room_id == room_id), None)