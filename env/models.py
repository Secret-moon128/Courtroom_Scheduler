"""
Typed Pydantic v2 models for the Courtroom Case Scheduling Negotiator.
Defines the full OpenEnv contract: Observation, Action, Reward.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Case(BaseModel):
    case_id: str
    case_type: str                          # "criminal" | "civil" | "family" | "appeals"
    priority: int                           # 1 (urgent) – 5 (routine)
    days_pending: int                       # calendar days since filing
    estimated_duration_hours: float         # expected hearing length
    mandatory_deadline: int | None          # episode step by which it MUST be scheduled (None = flexible)
    required_judge_specialisation: str      # "general" | "criminal" | "family" | "appeals"
    plaintiff_available_slots: list[int]    # slot indices where plaintiff/prosecution is free
    defendant_available_slots: list[int]    # slot indices where defence is free
    is_scheduled: bool = False
    assigned_slot: int | None = None
    assigned_judge: str | None = None
    assigned_room: str | None = None


class Judge(BaseModel):
    judge_id: str
    specialisations: list[str]              # subset of case_types this judge can hear
    available_slots: list[int]              # slot indices where judge is free
    max_hours_per_day: float = 7.0


class Courtroom(BaseModel):
    room_id: str
    capacity: int                           # max simultaneous parties (always >= 2)
    available_slots: list[int]


class ScheduledHearing(BaseModel):
    case_id: str
    slot: int
    judge_id: str
    room_id: str


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

class CourtObservation(BaseModel):
    """Full observation returned by reset() and step()."""

    # Episode metadata
    step: int               = Field(..., description="Current step index (0-based)")
    max_steps: int          = Field(..., description="Episode horizon")
    episode_done: bool      = Field(False)

    # Docket
    cases: list[Case]       = Field(..., description="All cases in the current docket")
    unscheduled_count: int  = Field(..., description="Number of cases not yet assigned")
    overdue_count: int      = Field(..., description="Cases past their mandatory deadline")

    # Resources
    judges: list[Judge]
    courtrooms: list[Courtroom]

    # Committed schedule
    scheduled_hearings: list[ScheduledHearing] = Field(default_factory=list)

    # Soft conflict hints (partial observability layer)
    conflict_hints: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Known conflicts per slot key, e.g. {'slot_3': ['Judge_A busy', 'Lawyer_X unavailable']}"
    )

    # Running reward (visible to agent for progress tracking)
    cumulative_reward: float = 0.0


class CourtAction(BaseModel):
    """One scheduling action the agent emits per step."""

    action_type: str = Field(
        ...,
        description="One of: 'schedule' | 'reschedule' | 'prioritise' | 'defer' | 'noop'"
    )

    # Fields used by 'schedule' and 'reschedule'
    case_id: str | None      = None
    slot: int | None         = None          # target time-slot index
    judge_id: str | None     = None
    room_id: str | None      = None

    # Fields used by 'prioritise'
    priority_order: list[str] | None = None  # ordered list of case_ids (highest priority first)

    # Fields used by 'defer'
    defer_case_id: str | None   = None
    defer_reason: str | None    = None

    # Metadata (logged, not used in reward computation)
    rationale: str | None = None


class CourtReward(BaseModel):
    """
    Reward breakdown for one step.
    Partial progress signals are emitted throughout the episode (not just at termination).
    """

    total: float = Field(..., description="Scalar reward for this step")

    # Additive components
    scheduling_bonus: float  = 0.0    # +1.0 per valid new scheduling action
    conflict_penalty: float  = 0.0    # -0.5 per hard constraint violation attempted
    deadline_penalty: float  = 0.0    # -2.0 per case that hits mandatory deadline unscheduled
    backlog_penalty: float   = 0.0    # -0.1 per step while unscheduled cases remain
    efficiency_bonus: float  = 0.0    # +0.2 for scheduling higher-priority cases before lower ones
    noop_penalty: float      = 0.0    # -0.05 per unnecessary noop when schedulable cases exist

    info: dict[str, Any] = Field(default_factory=dict)