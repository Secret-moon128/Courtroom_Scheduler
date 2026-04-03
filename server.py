"""
server.py — FastAPI HTTP server exposing the OpenEnv interface.

Endpoints:
    GET  /health              → 200 OK (HF Space ping)
    POST /reset               → CourtObservation (JSON)
    POST /step                → {observation, reward, done, info}
    GET  /state               → full internal state dict
    GET  /tasks               → list of available tasks
    POST /reset/{task_id}     → reset to a specific task config

The server maintains ONE environment instance per session.
For concurrent access, deploy multiple replicas.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from env.environment import CourtSchedulerEnv
from env.models import CourtAction, CourtObservation, CourtReward
from tasks import TASKS
from tasks.task_easy import EASY_CONFIG
from tasks.task_medium import MEDIUM_CONFIG
from tasks.task_hard import HARD_CONFIG

app = FastAPI(
    title="Courtroom Case Scheduling Negotiator",
    description="OpenEnv-compliant environment for AI scheduling agents.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_CONFIGS: dict[str, dict] = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}

# Global environment instance (single-session)
_env: CourtSchedulerEnv = CourtSchedulerEnv(**EASY_CONFIG)


# ── Response models ────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: dict[str, Any]
    done: bool
    info: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    current_task: str
    step: int


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """HF Space liveness check — must return 200."""
    s = _env.state()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        current_task=f"n_cases={_env.n_cases}",
        step=s["step"],
    )


@app.post("/reset")
def reset() -> dict[str, Any]:
    """Reset the environment to initial state. Returns first observation."""
    global _env
    obs = _env.reset()
    return obs.model_dump()


@app.post("/reset/{task_id}")
def reset_task(task_id: str) -> dict[str, Any]:
    """Reset to a specific task configuration (easy / medium / hard)."""
    global _env
    if task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}"
        )
    _env = CourtSchedulerEnv(**TASK_CONFIGS[task_id])
    obs  = _env.reset()
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(action: CourtAction) -> StepResponse:
    """Execute one action. Returns observation, reward, done, info."""
    try:
        obs, reward, done, info = _env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> dict[str, Any]:
    """Return full internal state snapshot."""
    return _env.state()


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """Return metadata for all registered tasks."""
    return [
        {
            "task_id":    t["task_id"],
            "name":       t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
        }
        for t in TASKS
    ]


@app.get("/")
def root():
    return {
        "name":        "Courtroom Case Scheduling Negotiator",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints":   ["/reset", "/reset/{task_id}", "/step", "/state", "/tasks"],
    }