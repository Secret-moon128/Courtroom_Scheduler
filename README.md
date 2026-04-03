---
title: Courtroom Scheduler
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Courtroom Case Scheduling Negotiator

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a **court scheduling clerk** — assigning legal cases to `(time-slot, judge, courtroom)` triples while satisfying hard constraints and optimising for backlog clearance, priority ordering, and statutory deadlines.

> **Tags:** `openenv` · `scheduling` · `constraint-satisfaction` · `legal` · `real-world`

---

## Motivation

Court backlogs are a chronic real-world problem: in many jurisdictions cases wait months or years for a hearing due to scheduling inefficiencies. A skilled clerk must balance:

- **Hard constraints** — judge specialisation, party availability, room availability, no double-booking
- **Soft objectives** — clear high-priority / older cases first, meet statutory deadlines, minimise idle resources

This environment captures exactly that challenge in a clean RL API.

---

## Quick Start

```bash
git clone <repo-url>
cd courtroom

pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with the random baseline agent (no API key needed)
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from env.environment import CourtSchedulerEnv
from agents.random_agent import RandomAgent
from tasks.graders import EasyGrader

env   = CourtSchedulerEnv(n_cases=5, n_judges=3, n_rooms=2, max_steps=30, seed=42)
agent = RandomAgent(seed=0)
_, info = agent.run_episode(env)
print("Score:", EasyGrader.score(info["state"]))
EOF
```

---

## Environment Description

### Episode flow

```
obs = env.reset()
while not done:
    action          = agent.act(obs)
    obs, reward, done, info = env.step(action)
final_score = grader.score(env.state())
```

### Time slots

The environment models a **discretised scheduling horizon** of `total_slots` slots (e.g. 20 for easy, 40 for hard). Each slot represents one hearing block. Multiple cases can be scheduled in parallel if different judges and rooms are used.

---

## Action Space

Actions are `CourtAction` Pydantic objects.

| Field | Type | Required for | Description |
|---|---|---|---|
| `action_type` | `str` | always | `schedule` \| `reschedule` \| `prioritise` \| `defer` \| `noop` |
| `case_id` | `str` | schedule, reschedule | e.g. `"CASE-001"` |
| `slot` | `int` | schedule, reschedule | Time-slot index `[0, total_slots)` |
| `judge_id` | `str` | schedule, reschedule | e.g. `"JUDGE-A"` |
| `room_id` | `str` | schedule, reschedule | e.g. `"ROOM-1"` |
| `priority_order` | `list[str]` | prioritise | Ordered list of `case_id`s |
| `defer_case_id` | `str` | defer | Case to defer |
| `rationale` | `str` | optional | Agent explanation (logged, not scored) |

---

## Observation Space

Observations are `CourtObservation` Pydantic objects.

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current step (0-based) |
| `max_steps` | `int` | Episode horizon |
| `cases` | `list[Case]` | Full docket with scheduling state |
| `unscheduled_count` | `int` | Cases not yet assigned |
| `overdue_count` | `int` | Cases past mandatory deadline |
| `judges` | `list[Judge]` | Judges with specialisations and availability |
| `courtrooms` | `list[Courtroom]` | Rooms with availability |
| `scheduled_hearings` | `list[ScheduledHearing]` | Committed schedule so far |
| `conflict_hints` | `dict[str, list[str]]` | Slot-level conflict hints |
| `cumulative_reward` | `float` | Running reward total |

### Case fields

| Field | Description |
|---|---|
| `case_id` | Unique identifier e.g. `CASE-001` |
| `case_type` | `criminal` \| `civil` \| `family` \| `appeals` |
| `priority` | `1` (urgent) → `5` (routine) |
| `days_pending` | Days since filing |
| `mandatory_deadline` | Episode step by which case MUST be scheduled (`None` = flexible) |
| `required_judge_specialisation` | Judge must have this specialisation |
| `plaintiff_available_slots` | Slot indices where plaintiff is free |
| `defendant_available_slots` | Slot indices where defendant is free |
| `is_scheduled` | Whether case has been assigned |

---

## Reward Function

Rewards are emitted **every step** (not just at episode end) providing dense progress signals.

| Component | Value | Trigger |
|---|---|---|
| `scheduling_bonus` | `+1.0` | Valid new scheduling action |
| `efficiency_bonus` | `+0.2` | Scheduling a priority 1–2 case |
| `efficiency_bonus` | `+0.5` | Scheduling a case before its deadline |
| `conflict_penalty` | `−0.5` | Hard constraint violation attempted |
| `deadline_penalty` | `−2.0` | Case reaches deadline unscheduled |
| `backlog_penalty` | `−0.1` | Per step while unscheduled cases remain |
| `noop_penalty` | `−0.05` | Noop when schedulable cases exist |
| `reschedule_cost` | `−0.1` | Administrative cost of rescheduling |

---

## Tasks

### Task 1 — Easy: Clear 5-Case Docket

| Config | Value |
|---|---|
| Cases | 5 |
| Judges | 3 |
| Courtrooms | 2 |
| Time slots | 20 |
| Max steps | 30 |
| Mandatory deadlines | None |
| Seed | 42 |

**Goal:** Schedule all 5 cases with zero hard constraint violations.  
**Score:** 70% from scheduling completeness + 30% from reward quality (violation-free).

### Task 2 — Medium: Backlog Prioritisation

| Config | Value |
|---|---|
| Cases | 15 |
| Judges | 4 (busier) |
| Courtrooms | 3 |
| Time slots | 30 |
| Max steps | 60 |
| Mandatory deadlines | ~10% of cases |
| Seed | 1337 |

**Goal:** Schedule ≥ 12 cases, high-priority cases first, no missed deadlines.  
**Score:** 50% scheduling completeness + 30% priority ordering + 20% deadline compliance.

### Task 3 — Hard: Surge Docket + Statutory Deadlines

| Config | Value |
|---|---|
| Cases | 30 |
| Judges | 5 (45% busy rate) |
| Courtrooms | 3 (25% busy rate) |
| Time slots | 40 |
| Max steps | 80 |
| Mandatory deadlines | ~30% of cases |
| Seed | 9999 |

**Goal:** Schedule ≥ 90% of cases, meet ALL statutory deadlines, prioritise by case age.  
**Score:** 40% completeness + 35% deadline compliance + 15% priority ordering + 10% case age.

---

## Baseline Scores

Measured with `RandomAgent(seed=0)` — a heuristic random agent that finds valid constraint-satisfying assignments.

| Task | Random Agent | Expected LLM Agent |
|---|---|---|
| Easy | **0.72** | ~0.92+ |
| Medium | **0.41** | ~0.70+ |
| Hard | **0.18** | ~0.55+ |

To reproduce:

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

## Setup & Deployment

### Local

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Docker

```bash
docker build -t courtroom-scheduler .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  courtroom-scheduler
```

### Hugging Face Space

```bash
huggingface-cli login
huggingface-cli repo create courtroom-scheduler --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/Secret128moon/courtroom-scheduler
git push hf main
```

Set the following Space secrets:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Run inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

The script emits structured logs:
```
[START] {"task_id": "easy", "model": "gpt-4o-mini", ...}
[STEP]  {"step": 1, "action_type": "schedule", "reward": 1.2, "done": false, ...}
[END]   {"task_id": "easy", "final_score": 0.91, "steps": 12, ...}
```

---

## HTTP API (when deployed)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — returns 200 |
| `/reset` | POST | Reset environment, returns first observation |
| `/reset/{task_id}` | POST | Reset to specific task (easy/medium/hard) |
| `/step` | POST | Execute action, returns obs + reward + done + info |
| `/state` | GET | Full internal state snapshot |
| `/tasks` | GET | List available tasks |
| `/docs` | GET | Interactive Swagger UI |

---

## File Structure

```
courtroom/
├── openenv.yaml          # OpenEnv spec metadata
├── inference.py          # Baseline inference script (mandatory name)
├── server.py             # FastAPI HTTP server
├── Dockerfile            # HF Space compatible container
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── environment.py    # CourtSchedulerEnv: step/reset/state
│   ├── models.py         # Pydantic: CourtObservation, CourtAction, CourtReward
│   └── simulator.py      # Procedural case/judge/room generator
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py      # Task 1 config + factory
│   ├── task_medium.py    # Task 2 config + factory
│   ├── task_hard.py      # Task 3 config + factory
│   └── graders.py        # Deterministic 0.0–1.0 graders
├── agents/
│   ├── __init__.py
│   ├── random_agent.py   # Heuristic baseline agent
│   └── llm_agent.py      # OpenAI-client LLM agent
└── tests/
    ├── test_env.py        # OpenEnv spec compliance tests
    └── test_graders.py    # Grader determinism and range tests
```

---

## License

MIT