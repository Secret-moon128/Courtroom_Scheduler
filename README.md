---
title: Courtroom Case Scheduling Negotiator
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - scheduling
  - legal
  - reinforcement-learning
  - constraint-satisfaction
---

# ⚖️ Courtroom Case Scheduling Negotiator

An **OpenEnv-compliant** reinforcement learning environment where an AI agent acts as a court scheduling clerk — assigning legal cases to `(time-slot, judge, courtroom)` triples while satisfying hard constraints and optimising for backlog clearance, priority ordering, and statutory deadlines.

> **Tags:** `openenv` · `scheduling` · `constraint-satisfaction` · `legal` · `real-world`

---

## What is this?

Court backlogs are a chronic real-world problem — in many jurisdictions cases wait months or years due to scheduling inefficiencies. A skilled clerk must balance:

- **Hard constraints** — judge specialisation, party availability, room availability, no double-booking
- **Soft objectives** — clear high-priority / older cases first, meet statutory deadlines, minimise idle resources

This environment captures exactly that challenge in a clean RL API with a live interactive dashboard.

---

## Live Dashboard

Visit the **App** tab to open the interactive dashboard where you can:

- Switch between Easy / Medium / Hard tasks
- Manually schedule cases by selecting a case, judge, room, and time slot
- Watch the reward breakdown update in real time
- See which cases are scheduled (green), pending, or overdue (red)
- Reset the episode at any time

---

## Quick Start

```bash
git clone https://huggingface.co/spaces/Secret128moom/courtroom-scheduler
cd courtroom-scheduler
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start server locally
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

---

## OpenEnv API

```python
from env.environment import CourtSchedulerEnv
from env.models import CourtAction

env = CourtSchedulerEnv(n_cases=5, n_judges=3, n_rooms=2, max_steps=30, seed=42)
obs = env.reset()

while not done:
    action = CourtAction(
        action_type="schedule",
        case_id="CASE-001",
        slot=3,
        judge_id="JUDGE-A",
        room_id="ROOM-1"
    )
    obs, reward, done, info = env.step(action)

final_score = grader.score(env.state())  # 0.0 → 1.0
```

### `reset()` → `CourtObservation`
Resets the episode. Returns the initial observation with the full case docket, judges, and courtrooms.

### `step(action)` → `(CourtObservation, CourtReward, bool, dict)`
Executes one action. Returns updated observation, reward breakdown, done flag, and diagnostic info.

### `state()` → `dict`
Returns the full internal state snapshot — useful for graders and logging.

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `schedule` \| `reschedule` \| `prioritise` \| `defer` \| `noop` |
| `case_id` | `str` | e.g. `"CASE-001"` |
| `slot` | `int` | Time-slot index `[0, total_slots)` |
| `judge_id` | `str` | e.g. `"JUDGE-A"` |
| `room_id` | `str` | e.g. `"ROOM-1"` |
| `priority_order` | `list[str]` | For `prioritise` action |
| `defer_case_id` | `str` | For `defer` action |
| `rationale` | `str` | Optional explanation (logged, not scored) |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current step (0-based) |
| `max_steps` | `int` | Episode horizon |
| `cases` | `list[Case]` | Full docket with scheduling state |
| `unscheduled_count` | `int` | Cases not yet assigned |
| `overdue_count` | `int` | Cases past mandatory deadline |
| `judges` | `list[Judge]` | Judges with specialisations and availability |
| `courtrooms` | `list[Courtroom]` | Rooms with availability slots |
| `scheduled_hearings` | `list[ScheduledHearing]` | Committed schedule so far |
| `conflict_hints` | `dict` | Slot-level conflict hints |
| `cumulative_reward` | `float` | Running reward total |

---

## Reward Function

Dense rewards emitted every step — not just at episode end.

| Component | Value | Trigger |
|---|---|---|
| `scheduling_bonus` | `+1.0` | Valid new scheduling action |
| `efficiency_bonus` | `+0.2` | Scheduling a priority 1–2 case |
| `efficiency_bonus` | `+0.5` | Scheduling before mandatory deadline |
| `conflict_penalty` | `−0.5` | Hard constraint violation attempted |
| `deadline_penalty` | `−2.0` | Case reaches deadline unscheduled |
| `backlog_penalty` | `−0.1` | Per step while unscheduled cases remain |
| `noop_penalty` | `−0.05` | Noop when schedulable cases exist |

---

## Tasks

### Task 1 — Easy: Clear 5-Case Docket
| | |
|---|---|
| Cases | 5 |
| Judges | 3 |
| Courtrooms | 2 |
| Time slots | 20 |
| Max steps | 30 |
| Deadlines | None |

**Goal:** Schedule all 5 cases with zero constraint violations.  
**Grader:** 70% scheduling completeness + 30% reward quality.

---

### Task 2 — Medium: Backlog Prioritisation
| | |
|---|---|
| Cases | 15 |
| Judges | 4 |
| Courtrooms | 3 |
| Time slots | 30 |
| Max steps | 60 |
| Deadlines | ~10% of cases |

**Goal:** Schedule ≥ 12 cases, high-priority first, no missed deadlines.  
**Grader:** 50% completeness + 30% priority ordering + 20% deadline compliance.

---

### Task 3 — Hard: Surge Docket + Statutory Deadlines
| | |
|---|---|
| Cases | 30 |
| Judges | 5 (45% busy) |
| Courtrooms | 3 (25% busy) |
| Time slots | 40 |
| Max steps | 80 |
| Deadlines | ~30% of cases (statutory) |

**Goal:** Schedule ≥ 90% of cases, meet ALL statutory deadlines, prioritise by case age.  
**Grader:** 40% completeness + 35% deadline compliance + 15% priority + 10% case age.

---

## Baseline Scores

Measured with `RandomAgent(seed=0)` — a heuristic agent that finds valid constraint-satisfying assignments without any LLM.

| Task | Random Agent | LLM Agent (llama-3.1-8b) |
|---|---|---|
| Easy | 1.00 | ~0.90 |
| Medium | 0.91 | ~0.96 |
| Hard | 0.93 | ~0.94 |
| **Average** | **0.95** | **~0.93** |

To reproduce random agent baseline:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from agents.random_agent import RandomAgent
from tasks import TASKS
from tasks.task_easy import make_easy_task
from tasks.task_medium import make_medium_task
from tasks.task_hard import make_hard_task

makers = {"easy": make_easy_task, "medium": make_medium_task, "hard": make_hard_task}
for task in TASKS:
    env   = makers[task["task_id"]]()
    agent = RandomAgent(seed=0)
    _, info = agent.run_episode(env)
    score   = task["grader"].score(info["state"])
    print(f"{task['task_id']:8s}: {score:.4f}")
EOF
```

---

## Running Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your-api-key"

python inference.py
```

Output format:
```
[START] task=easy env=courtroom-scheduler model=llama-3.1-8b-instant
[STEP] step=1 action=schedule(CASE-002@slot5) reward=1.20 done=false error=null
[STEP] step=2 action=schedule(CASE-001@slot0) reward=1.20 done=false error=null
[END] success=true steps=5 score=0.920 rewards=1.20,1.20,1.20,1.20,1.20
```

---

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive dashboard UI |
| `/health` | GET | Liveness check — returns `{"status": "ok"}` |
| `/reset` | POST | Reset environment |
| `/reset/{task_id}` | POST | Reset to specific task (easy/medium/hard) |
| `/step` | POST | Execute one action |
| `/state` | GET | Full internal state |
| `/tasks` | GET | List all tasks |
| `/docs` | GET | Swagger UI |

---

## Docker

```bash
docker build -t courtroom-scheduler .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="your-key" \
  courtroom-scheduler
```

---

## Project Structure

```
courtroom/
├── inference.py          # Baseline inference script (mandatory name)
├── server.py             # FastAPI server + frontend dashboard
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── environment.py    # step() / reset() / state()
│   ├── models.py         # Pydantic typed models
│   └── simulator.py      # Procedural case/judge/room generator
├── tasks/
│   ├── task_easy.py
│   ├── task_medium.py
│   ├── task_hard.py
│   └── graders.py        # Deterministic 0.0–1.0 graders
├── agents/
│   ├── random_agent.py   # Heuristic baseline
│   └── llm_agent.py      # OpenAI-client LLM agent
└── tests/
    ├── test_env.py        # OpenEnv spec compliance
    └── test_graders.py    # Grader tests
```

---

## License

MIT