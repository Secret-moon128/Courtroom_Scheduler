"""
Simulator: procedurally generates cases, judges, and courtrooms
for the Courtroom Case Scheduling Negotiator environment.

All generation is seeded for reproducibility.
"""

from __future__ import annotations
import random
from .models import Case, Judge, Courtroom

CASE_TYPES        = ["criminal", "civil", "family", "appeals"]
SPECIALISATIONS   = ["general", "criminal", "family", "appeals"]

CASE_TYPE_WEIGHTS = [0.30, 0.40, 0.20, 0.10]   # realistic distribution
PRIORITY_WEIGHTS  = [0.10, 0.20, 0.35, 0.25, 0.10]  # priority 1..5

DURATION_BY_TYPE = {
    "criminal": (2.0, 6.0),
    "civil":    (1.0, 4.0),
    "family":   (1.0, 3.0),
    "appeals":  (0.5, 2.0),
}


def generate_cases(
    n: int,
    total_slots: int,
    deadline_fraction: float = 0.0,
    seed: int = 42,
    min_available_slots: int = 3,
) -> list[Case]:
    """
    Generate `n` synthetic cases.

    Args:
        n: number of cases
        total_slots: total time-slots in the episode
        deadline_fraction: fraction of cases that have a mandatory deadline
        seed: RNG seed
        min_available_slots: minimum party availability slots per case
    """
    rng = random.Random(seed)
    cases: list[Case] = []

    for i in range(n):
        ctype   = rng.choices(CASE_TYPES, weights=CASE_TYPE_WEIGHTS)[0]
        prio    = rng.choices([1, 2, 3, 4, 5], weights=PRIORITY_WEIGHTS)[0]
        pending = rng.randint(1, 180)
        lo, hi  = DURATION_BY_TYPE[ctype]
        dur     = round(rng.uniform(lo, hi) * 2) / 2   # 0.5-hour increments

        # Mandatory deadline: a random future step
        deadline: int | None = None
        if rng.random() < deadline_fraction:
            deadline = rng.randint(max(1, total_slots // 4), total_slots - 1)

        # Party availability: random subset of slots (at least min_available_slots)
        n_plaintiff = rng.randint(min_available_slots, max(min_available_slots, total_slots // 2))
        n_defendant = rng.randint(min_available_slots, max(min_available_slots, total_slots // 2))
        plaintiff_slots = sorted(rng.sample(range(total_slots), min(n_plaintiff, total_slots)))
        defendant_slots = sorted(rng.sample(range(total_slots), min(n_defendant, total_slots)))

        # Judge specialisation: appeals/family need specialist, others accept general
        if ctype in ("appeals", "family"):
            req_spec = ctype
        else:
            req_spec = rng.choices(["general", ctype], weights=[0.6, 0.4])[0]

        cases.append(Case(
            case_id=f"CASE-{i+1:03d}",
            case_type=ctype,
            priority=prio,
            days_pending=pending,
            estimated_duration_hours=dur,
            mandatory_deadline=deadline,
            required_judge_specialisation=req_spec,
            plaintiff_available_slots=plaintiff_slots,
            defendant_available_slots=defendant_slots,
        ))

    return cases


def generate_judges(
    n: int,
    total_slots: int,
    seed: int = 42,
    busy_fraction: float = 0.3,
) -> list[Judge]:
    """
    Generate `n` judges with random specialisations and availability.

    Args:
        n: number of judges
        total_slots: total time-slots in the episode
        seed: RNG seed
        busy_fraction: fraction of slots a judge is unavailable
    """
    rng = random.Random(seed + 1000)
    judges: list[Judge] = []

    spec_pools = [
        ["general", "criminal"],
        ["general", "civil"],
        ["family"],
        ["appeals", "general"],
        ["criminal", "general"],
    ]

    for i in range(n):
        specs = spec_pools[i % len(spec_pools)]
        n_busy = int(total_slots * busy_fraction)
        busy_slots = set(rng.sample(range(total_slots), min(n_busy, total_slots)))
        available = sorted(set(range(total_slots)) - busy_slots)

        judges.append(Judge(
            judge_id=f"JUDGE-{chr(65 + i)}",
            specialisations=specs,
            available_slots=available,
            max_hours_per_day=rng.choice([6.0, 7.0, 7.0, 8.0]),
        ))

    return judges


def generate_courtrooms(
    n: int,
    total_slots: int,
    seed: int = 42,
    busy_fraction: float = 0.15,
) -> list[Courtroom]:
    """
    Generate `n` courtrooms with random availability.

    Args:
        n: number of courtrooms
        total_slots: total time-slots in the episode
        seed: RNG seed
        busy_fraction: fraction of slots a room is unavailable (maintenance etc.)
    """
    rng = random.Random(seed + 2000)
    rooms: list[Courtroom] = []

    for i in range(n):
        n_busy = int(total_slots * busy_fraction)
        busy_slots = set(rng.sample(range(total_slots), min(n_busy, total_slots)))
        available = sorted(set(range(total_slots)) - busy_slots)

        rooms.append(Courtroom(
            room_id=f"ROOM-{i+1}",
            capacity=rng.choice([20, 30, 50, 100]),
            available_slots=available,
        ))

    return rooms


def build_conflict_hints(
    cases: list[Case],
    judges: list[Judge],
    rooms: list[Courtroom],
    total_slots: int,
) -> dict[str, list[str]]:
    """
    Build a slot -> [conflict reason] map to give the agent partial observability hints.
    """
    hints: dict[str, list[str]] = {}

    for slot in range(total_slots):
        reasons: list[str] = []
        for j in judges:
            if slot not in j.available_slots:
                reasons.append(f"{j.judge_id} unavailable")
        for r in rooms:
            if slot not in r.available_slots:
                reasons.append(f"{r.room_id} unavailable")
        if reasons:
            hints[f"slot_{slot}"] = reasons

    return hints