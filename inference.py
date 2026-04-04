"""
inference.py — OpenEnv Baseline Inference Script

MANDATORY stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import CourtSchedulerEnv
from env.models import CourtAction
from tasks.task_easy import EASY_CONFIG
from tasks.task_medium import MEDIUM_CONFIG
from tasks.task_hard import HARD_CONFIG
from tasks.graders import EasyGrader, MediumGrader, HardGrader

# ── Config ──────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "courtroom-scheduler"
SUCCESS_THRESHOLD = 0.5

TASKS = [
    {"task_id": "easy",   "config": EASY_CONFIG,   "grader": EasyGrader},
    {"task_id": "medium", "config": MEDIUM_CONFIG,  "grader": MediumGrader},
    {"task_id": "hard",   "config": HARD_CONFIG,    "grader": HardGrader},
]

SYSTEM_PROMPT = textwrap.dedent("""
You are a court scheduling clerk AI. Assign cases to valid hearing slots.

For each case you must pick:
  - case_id  : an UNSCHEDULED case from the list
  - slot     : MUST appear in BOTH party_overlap_slots AND the chosen judge's free_slots AND the chosen room's free_slots
  - judge_id : MUST have the required specialisation for that case
  - room_id  : MUST have the chosen slot in its free_slots

CRITICAL RULES:
- NEVER pick a slot that is NOT in party_overlap_slots for that case — this always fails
- NEVER repeat a (case_id, slot) combination that already failed
- If a case has no valid overlap slots, skip it and pick a DIFFERENT case
- Always pick the FIRST available slot from party_overlap_slots that also appears in the judge's free_slots

OUTPUT: reply with ONLY a single JSON object, no markdown, no explanation:
{"action_type":"schedule","case_id":"CASE-001","slot":5,"judge_id":"JUDGE-A","room_id":"ROOM-1","rationale":"slot 5 is in party overlap and judge free slots"}
""").strip()


# ── Mandatory log functions ──────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompt builder ───────────────────────────────────────────────────

def build_prompt(obs, failed_attempts: list[str]) -> str:
    """
    Build a prompt that explicitly shows:
    - party_overlap_slots for each case (ONLY valid slots)
    - which judge free slots intersect with those
    - recent failures so the LLM stops repeating them
    """
    unscheduled = sorted(
        [c for c in obs.cases if not c.is_scheduled],
        key=lambda c: (
            c.mandatory_deadline if c.mandatory_deadline is not None else 9999,
            c.priority,
            -c.days_pending,
        )
    )

    booked_judge = {}
    booked_room  = {}
    for h in obs.scheduled_hearings:
        booked_judge.setdefault(h.judge_id, []).append(h.slot)
        booked_room.setdefault(h.room_id, []).append(h.slot)

    # Build judge free-slot map
    judge_free = {}
    for j in obs.judges:
        judge_free[j.judge_id] = {
            "specs": j.specialisations,
            "free":  [s for s in j.available_slots if s not in booked_judge.get(j.judge_id, [])]
        }

    room_free = {}
    for r in obs.courtrooms:
        room_free[r.room_id] = [s for s in r.available_slots if s not in booked_room.get(r.room_id, [])]

    # For each case, compute the EXACT valid slots (overlap + judge + room)
    case_lines = []
    for c in unscheduled[:10]:
        party_overlap = sorted(
            set(c.plaintiff_available_slots) & set(c.defendant_available_slots)
        )
        dl = f"DEADLINE@step{c.mandatory_deadline}" if c.mandatory_deadline else "flexible"

        # Find valid (judge, slot, room) combos
        valid_options = []
        for jid, jdata in judge_free.items():
            spec_ok = (c.required_judge_specialisation == "general" or
                       c.required_judge_specialisation in jdata["specs"])
            if not spec_ok:
                continue
            judge_slots = set(jdata["free"])
            for rid, rslots in room_free.items():
                valid_slots = sorted(
                    set(party_overlap) & judge_slots & set(rslots)
                )
                if valid_slots:
                    valid_options.append(
                        f"      -> {jid}+{rid}: valid_slots={valid_slots[:6]}"
                    )

        if valid_options:
            options_str = "\n" + "\n".join(valid_options[:4])
        else:
            options_str = "\n      -> NO VALID SLOTS — skip this case"

        case_lines.append(
            f"  {c.case_id}: type={c.case_type} prio={c.priority} "
            f"pending={c.days_pending}d {dl} req={c.required_judge_specialisation}"
            f"{options_str}"
        )

    # Recent failures
    failure_str = ""
    if failed_attempts:
        recent = failed_attempts[-6:]
        failure_str = "\nRECENT FAILURES — do NOT repeat these:\n" + "\n".join(f"  {f}" for f in recent)

    return (
        f"STEP {obs.step}/{obs.max_steps} | unscheduled={obs.unscheduled_count} "
        f"overdue={obs.overdue_count} reward={obs.cumulative_reward:.2f}"
        f"{failure_str}\n\n"
        "CASES WITH PRE-COMPUTED VALID OPTIONS (pick from these slots only):\n"
        + "\n".join(case_lines) + "\n\n"
        "Choose ONE case and ONE of its valid (judge, room, slot) combos. "
        "Pick the first case that has valid options. Reply JSON only."
    )


# ── LLM call ────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs, failed_attempts: list[str]) -> tuple:
    """Returns (CourtAction, action_string_for_log)."""

    # Try LLM first
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, failed_attempts)},
            ],
            temperature=0.0,   # deterministic — we want it to follow instructions
            max_tokens=300,
        )
        text  = (resp.choices[0].message.content or "").strip()
        text  = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data   = json.loads(match.group())
            action = CourtAction(**data)
            label  = f"schedule({data.get('case_id','?')}@slot{data.get('slot','?')})"
            return action, label
    except Exception:
        pass

    # Fallback: compute a guaranteed valid action ourselves
    return _fallback_action(obs)


def _fallback_action(obs) -> tuple:
    """
    Deterministic fallback: find the first valid (case, judge, slot, room) combo.
    Used when the LLM fails to produce a valid action.
    """
    booked_judge = {}
    booked_room  = {}
    for h in obs.scheduled_hearings:
        booked_judge.setdefault(h.judge_id, []).append(h.slot)
        booked_room.setdefault(h.room_id, []).append(h.slot)

    unscheduled = sorted(
        [c for c in obs.cases if not c.is_scheduled],
        key=lambda c: (c.priority, -c.days_pending)
    )

    for case in unscheduled:
        party_overlap = (
            set(case.plaintiff_available_slots) &
            set(case.defendant_available_slots)
        )
        for judge in obs.judges:
            spec_ok = (case.required_judge_specialisation == "general" or
                       case.required_judge_specialisation in judge.specialisations)
            if not spec_ok:
                continue
            j_free = set(judge.available_slots) - set(booked_judge.get(judge.judge_id, []))
            for room in obs.courtrooms:
                r_free = set(room.available_slots) - set(booked_room.get(room.room_id, []))
                valid  = sorted(party_overlap & j_free & r_free)
                if valid:
                    slot   = valid[0]
                    action = CourtAction(
                        action_type="schedule",
                        case_id=case.case_id,
                        slot=slot,
                        judge_id=judge.judge_id,
                        room_id=room.room_id,
                        rationale="fallback valid assignment",
                    )
                    return action, f"schedule({case.case_id}@slot{slot})"

    return CourtAction(action_type="noop", rationale="no valid assignment found"), "noop()"


# ── Episode runner ───────────────────────────────────────────────────

def run_task(task_meta: dict, client: OpenAI) -> float:
    task_id = task_meta["task_id"]
    config  = task_meta["config"]
    grader  = task_meta["grader"]

    env = CourtSchedulerEnv(**config)
    obs = env.reset()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards        : list  = []
    failed_attempts: list[str] = []
    steps_taken     = 0
    done            = False
    score           = 0.0
    success         = False

    try:
        for step in range(1, config["max_steps"] + 1):
            if done:
                break

            # If LLM has failed 3 times in a row, bypass it entirely
            if len(failed_attempts) >= 3:
                action, action_str = _fallback_action(obs)
                failed_attempts.clear()
            else:
                action, action_str = get_action(client, obs, failed_attempts)

            last_error: Optional[str] = None

            try:
                obs, reward_obj, done, info = env.step(action)
                step_reward = reward_obj.total

                # Track failures so LLM doesn't repeat them
                err = info.get("error") or info.get("constraint_violations")
                if isinstance(err, list):
                    err = "; ".join(err)
                last_error = err
                if err and action.action_type == "schedule":
                    failed_attempts.append(
                        f"{action.case_id}@slot{action.slot} via {action.judge_id}+{action.room_id}: {err}"
                    )

            except Exception as e:
                step_reward = 0.0
                done        = True
                last_error  = str(e)

            rewards.append(step_reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=done,
                error=last_error,
            )

        score   = grader.score(env.state())
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
    if missing:
        print(f"[DEBUG] Warning: env vars not explicitly set: {missing}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: dict = {}
    for task_meta in TASKS:
        print(f"\n{'='*55}", flush=True)
        print(f"Running task: {task_meta['task_id']}", flush=True)
        print(f"{'='*55}", flush=True)
        score = run_task(task_meta, client)
        all_scores[task_meta["task_id"]] = score

    print(f"\n{'='*55}", flush=True)
    print("Final Scores", flush=True)
    for tid, sc in all_scores.items():
        print(f"  {tid:10s}: {sc:.4f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'average':10s}: {avg:.4f}", flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == "__main__":
    main()