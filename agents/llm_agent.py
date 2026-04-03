"""
LLMAgent — drives the environment using an LLM via the OpenAI-compatible client.

Reads:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — API key / Hugging Face token

The agent formats the current observation as a structured prompt,
calls the LLM, parses the JSON action response, and returns a CourtAction.
"""

from __future__ import annotations
import json
import os
import re
from openai import OpenAI
from env.models import CourtObservation, CourtAction


SYSTEM_PROMPT = """You are an expert court scheduling clerk AI.

Your job is to schedule legal cases to hearing slots by assigning each case a:
  - slot (integer index)
  - judge_id (string)
  - room_id (string)

HARD CONSTRAINTS (violations are penalised):
1. The judge must have the required specialisation for the case type.
2. The judge must be available at the chosen slot.
3. The courtroom must be available at the chosen slot.
4. The plaintiff (plaintiff_available_slots) must be available at the slot.
5. The defendant (defendant_available_slots) must be available at the slot.
6. No judge or room can be double-booked at the same slot.

SOFT GOALS (maximise reward):
- Schedule cases with mandatory_deadline first.
- Schedule higher-priority cases (priority=1) before lower-priority ones (priority=5).
- Schedule cases with more days_pending (older cases) before newer ones.
- Avoid noop unless there are truly no valid assignments.

RESPONSE FORMAT — respond ONLY with a JSON object, no markdown, no explanation:
{
  "action_type": "schedule",
  "case_id": "CASE-001",
  "slot": 3,
  "judge_id": "JUDGE-A",
  "room_id": "ROOM-1",
  "rationale": "Brief reason"
}

Valid action_types: "schedule", "reschedule", "prioritise", "defer", "noop"
"""


def _obs_to_prompt(obs: CourtObservation) -> str:
    """Convert observation to a compact text prompt for the LLM."""
    unscheduled = [c for c in obs.cases if not c.is_scheduled]
    scheduled   = [c for c in obs.cases if c.is_scheduled]

    # Build resource summary
    judge_lines = []
    for j in obs.judges:
        judge_lines.append(
            f"  {j.judge_id}: specialisations={j.specialisations}, "
            f"available_slots={j.available_slots[:10]}{'...' if len(j.available_slots)>10 else ''}"
        )

    room_lines = []
    for r in obs.courtrooms:
        room_lines.append(
            f"  {r.room_id}: available_slots={r.available_slots[:10]}{'...' if len(r.available_slots)>10 else ''}"
        )

    # Already booked slots per judge/room
    booked_judge = {}
    booked_room  = {}
    for h in obs.scheduled_hearings:
        booked_judge.setdefault(h.judge_id, []).append(h.slot)
        booked_room.setdefault(h.room_id, []).append(h.slot)

    # Unscheduled case lines (sorted by urgency)
    sorted_cases = sorted(
        unscheduled,
        key=lambda c: (
            c.mandatory_deadline if c.mandatory_deadline is not None else 9999,
            c.priority,
            -c.days_pending,
        )
    )

    case_lines = []
    for c in sorted_cases[:10]:   # show top 10 to stay within context
        deadline_str = f"DEADLINE@step{c.mandatory_deadline}" if c.mandatory_deadline else "no_deadline"
        case_lines.append(
            f"  {c.case_id}: type={c.case_type}, priority={c.priority}, "
            f"pending={c.days_pending}d, {deadline_str}, "
            f"req_spec={c.required_judge_specialisation}, "
            f"plaintiff_slots={c.plaintiff_available_slots[:8]}, "
            f"defendant_slots={c.defendant_available_slots[:8]}"
        )

    prompt = f"""STEP {obs.step}/{obs.max_steps} | Unscheduled: {obs.unscheduled_count} | Overdue: {obs.overdue_count} | Cumulative reward: {obs.cumulative_reward:.2f}

JUDGES:
{chr(10).join(judge_lines)}

COURTROOMS:
{chr(10).join(room_lines)}

ALREADY BOOKED:
  Judges: {json.dumps(booked_judge)}
  Rooms:  {json.dumps(booked_room)}

UNSCHEDULED CASES (sorted by urgency, showing up to 10):
{chr(10).join(case_lines)}

TOTAL SLOTS AVAILABLE: 0 to {obs.max_steps - 1}

Choose ONE action. Respond with valid JSON only.
"""
    return prompt


def _parse_action(response_text: str) -> CourtAction:
    """Parse LLM response into a CourtAction, with fallback to noop."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", response_text).strip().rstrip("`").strip()

    # Find the first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return CourtAction(action_type="noop", rationale="Failed to parse LLM response")

    try:
        data = json.loads(match.group())
        return CourtAction(**data)
    except Exception as e:
        return CourtAction(action_type="noop", rationale=f"Parse error: {e}")


class LLMAgent:
    """
    LLM-powered scheduling agent using the OpenAI-compatible client.
    """

    def __init__(
        self,
        api_base_url: str | None = None,
        model_name: str | None   = None,
        api_key: str | None      = None,
        max_retries: int         = 2,
    ) -> None:
        self.model       = model_name or os.environ["MODEL_NAME"]
        self.max_retries = max_retries

        self.client = OpenAI(
            base_url=api_base_url or os.environ["API_BASE_URL"],
            api_key=api_key or os.environ.get("HF_TOKEN", "hf-no-key"),
        )

    def act(self, obs: CourtObservation) -> CourtAction:
        """Call the LLM and return a parsed CourtAction."""
        user_prompt = _obs_to_prompt(obs)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                action = _parse_action(text)
                if action.action_type != "noop" or attempt == self.max_retries:
                    return action
            except Exception as e:
                if attempt == self.max_retries:
                    return CourtAction(
                        action_type="noop",
                        rationale=f"LLM call failed after {self.max_retries+1} attempts: {e}"
                    )

        return CourtAction(action_type="noop", rationale="Max retries reached")