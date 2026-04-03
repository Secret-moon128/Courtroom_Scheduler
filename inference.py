"""
inference.py — OpenEnv baseline inference script.

MANDATORY requirements satisfied:
  - Named exactly 'inference.py' in the root directory
  - Uses OpenAI client for all LLM calls
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
  - Emits structured stdout logs: [START], [STEP], [END]
  - Runs all 3 tasks and produces reproducible scores
  - Completes in < 20 minutes on 2 vCPU / 8 GB RAM

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="sk-..."
    python inference.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent
from tasks import TASKS
from env.environment import CourtSchedulerEnv
from tasks.task_easy import EASY_CONFIG
from tasks.task_medium import MEDIUM_CONFIG
from tasks.task_hard import HARD_CONFIG


TASK_CONFIGS = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}


def log(tag: str, payload: dict) -> None:
    """Emit a structured log line to stdout."""
    line = f"[{tag}] {json.dumps(payload)}"
    print(line, flush=True)


def run_task(task_meta: dict, agent: LLMAgent) -> float:
    """Run a single task and return the grader score."""
    task_id   = task_meta["task_id"]
    config    = TASK_CONFIGS[task_id]
    grader    = task_meta["grader"]

    env = CourtSchedulerEnv(**config)
    obs = env.reset()

    log("START", {
        "task_id":    task_id,
        "difficulty": task_meta["difficulty"],
        "model":      os.environ.get("MODEL_NAME", "unknown"),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "n_cases":    config["n_cases"],
        "max_steps":  config["max_steps"],
    })

    done         = False
    step_idx     = 0
    total_reward = 0.0

    while not done:
        t0     = time.time()
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        elapsed = round(time.time() - t0, 3)

        total_reward += reward.total
        step_idx     += 1

        log("STEP", {
            "task_id":     task_id,
            "step":        step_idx,
            "action_type": action.action_type,
            "action":      action.model_dump(exclude_none=True),
            "reward":      round(reward.total, 4),
            "reward_breakdown": {
                "scheduling_bonus":  reward.scheduling_bonus,
                "conflict_penalty":  reward.conflict_penalty,
                "deadline_penalty":  reward.deadline_penalty,
                "backlog_penalty":   reward.backlog_penalty,
                "efficiency_bonus":  reward.efficiency_bonus,
                "noop_penalty":      reward.noop_penalty,
            },
            "cumulative_reward":  round(obs.cumulative_reward, 4),
            "unscheduled":        obs.unscheduled_count,
            "overdue":            obs.overdue_count,
            "done":               done,
            "step_time_s":        elapsed,
            "info":               info,
        })

    final_state = env.state()
    score       = grader.score(final_state)

    log("END", {
        "task_id":         task_id,
        "final_score":     score,
        "total_reward":    round(total_reward, 4),
        "steps_taken":     step_idx,
        "cases_scheduled": sum(1 for c in final_state["cases"] if c["is_scheduled"]),
        "cases_total":     len(final_state["cases"]),
        "overdue_final":   final_state["overdue_count"],
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    })

    return score


def main() -> None:
    # Validate required env vars
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}", file=sys.stderr)
        sys.exit(1)

    agent  = LLMAgent()
    scores = {}

    for task_meta in TASKS:
        task_id = task_meta["task_id"]
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id} ({task_meta['difficulty']})", flush=True)
        print(f"{'='*60}", flush=True)

        score = run_task(task_meta, agent)
        scores[task_id] = score
        print(f"\nTask '{task_id}' final score: {score:.4f}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("INFERENCE COMPLETE — Final Scores", flush=True)
    print(f"{'='*60}", flush=True)
    for tid, sc in scores.items():
        print(f"  {tid:10s}: {sc:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':10s}: {avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()