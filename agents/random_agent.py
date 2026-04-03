"""
RandomAgent — baseline agent for the Courtroom Scheduler environment.

Selects random valid (case, slot, judge, room) combinations.
Used to establish baseline scores reported in README.
"""

from __future__ import annotations
import random
from env.models import CourtObservation, CourtAction


class RandomAgent:
    """
    A purely random agent that attempts to schedule one unscheduled case per step.

    Strategy:
      1. Pick a random unscheduled case.
      2. Find the intersection of (judge available slots) ∩ (plaintiff slots) ∩ (defendant slots).
      3. Pick a random slot from that intersection.
      4. Pick a random qualifying judge and room available at that slot.
      5. Emit a 'schedule' action. Falls back to 'noop' if no valid assignment found.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: CourtObservation) -> CourtAction:
        unscheduled = [c for c in obs.cases if not c.is_scheduled]
        if not unscheduled:
            return CourtAction(action_type="noop", rationale="All cases scheduled")

        # Shuffle to avoid always picking case 0
        self.rng.shuffle(unscheduled)

        for case in unscheduled:
            # Find judges that match the required specialisation
            qualifying_judges = [
                j for j in obs.judges
                if case.required_judge_specialisation == "general"
                or case.required_judge_specialisation in j.specialisations
            ]
            if not qualifying_judges:
                continue

            for judge in self.rng.sample(qualifying_judges, len(qualifying_judges)):
                # Find slots where judge, plaintiff, and defendant all available
                candidate_slots = (
                    set(judge.available_slots)
                    & set(case.plaintiff_available_slots)
                    & set(case.defendant_available_slots)
                )
                # Exclude slots the judge is already booked
                already_booked_judge = {
                    h.slot for h in obs.scheduled_hearings if h.judge_id == judge.judge_id
                }
                candidate_slots -= already_booked_judge

                if not candidate_slots:
                    continue

                for room in self.rng.sample(obs.courtrooms, len(obs.courtrooms)):
                    room_slots = candidate_slots & set(room.available_slots)
                    already_booked_room = {
                        h.slot for h in obs.scheduled_hearings if h.room_id == room.room_id
                    }
                    room_slots -= already_booked_room

                    if room_slots:
                        slot = self.rng.choice(sorted(room_slots))
                        return CourtAction(
                            action_type="schedule",
                            case_id=case.case_id,
                            slot=slot,
                            judge_id=judge.judge_id,
                            room_id=room.room_id,
                            rationale=f"Random valid assignment for {case.case_id}",
                        )

        # No valid assignment found this step
        return CourtAction(action_type="noop", rationale="No valid assignment found this step")

    def run_episode(self, env) -> tuple[float, dict]:
        """
        Run a full episode and return (final_score, info).
        Useful for collecting baseline numbers.
        """
        obs  = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action       = self.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward.total
            steps        += 1

        return total_reward, {"steps": steps, "state": env.state()}