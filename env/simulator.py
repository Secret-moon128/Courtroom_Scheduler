"""
Simulator: procedurally generates cases, judges, and courtrooms.
Fixed: party overlaps guaranteed, judge specialisations include case types.
"""

from __future__ import annotations
import random
from .models import Case, Judge, Courtroom

CASE_TYPES      = ["criminal", "civil", "family", "appeals"]
CASE_WEIGHTS    = [0.30, 0.40, 0.20, 0.10]
PRIORITY_WEIGHTS= [0.10, 0.20, 0.35, 0.25, 0.10]

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
    min_available_slots: int = 4,
) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []

    for i in range(n):
        ctype   = rng.choices(CASE_TYPES, weights=CASE_WEIGHTS)[0]
        prio    = rng.choices([1, 2, 3, 4, 5], weights=PRIORITY_WEIGHTS)[0]
        pending = rng.randint(1, 180)
        lo, hi  = DURATION_BY_TYPE[ctype]
        dur     = round(rng.uniform(lo, hi) * 2) / 2

        deadline: int | None = None
        if rng.random() < deadline_fraction:
            deadline = rng.randint(max(1, total_slots // 3), total_slots - 1)

        # Judge specialisation: always use general OR the case type
        # so a qualifying judge always exists
        req_spec = rng.choices(["general", ctype], weights=[0.5, 0.5])[0]

        # Guarantee a non-empty party overlap by:
        # 1. Pick a shared core set of slots
        # 2. Add extra slots for each party
        n_overlap = max(3, total_slots // 4)
        core_slots = sorted(rng.sample(range(total_slots), min(n_overlap, total_slots)))

        extra_p = rng.randint(0, max(0, total_slots // 4))
        extra_d = rng.randint(0, max(0, total_slots // 4))

        remaining = sorted(set(range(total_slots)) - set(core_slots))
        extra_plaintiff = sorted(rng.sample(remaining, min(extra_p, len(remaining))))
        extra_defendant = sorted(rng.sample(remaining, min(extra_d, len(remaining))))

        plaintiff_slots = sorted(set(core_slots) | set(extra_plaintiff))
        defendant_slots = sorted(set(core_slots) | set(extra_defendant))

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
    rng = random.Random(seed + 1000)

    # Each judge covers multiple specialisations including "general"
    # so at least one judge can always handle any case
    spec_pools = [
        ["general", "criminal", "civil"],
        ["general", "civil", "appeals"],
        ["general", "family", "criminal"],
        ["general", "appeals", "civil"],
        ["general", "criminal", "family"],
    ]

    judges: list[Judge] = []
    for i in range(n):
        specs   = spec_pools[i % len(spec_pools)]
        n_busy  = int(total_slots * busy_fraction)
        busy    = set(rng.sample(range(total_slots), min(n_busy, total_slots)))
        avail   = sorted(set(range(total_slots)) - busy)

        judges.append(Judge(
            judge_id=f"JUDGE-{chr(65 + i)}",
            specialisations=specs,
            available_slots=avail,
            max_hours_per_day=rng.choice([6.0, 7.0, 7.0, 8.0]),
        ))

    return judges


def generate_courtrooms(
    n: int,
    total_slots: int,
    seed: int = 42,
    busy_fraction: float = 0.15,
) -> list[Courtroom]:
    rng = random.Random(seed + 2000)
    rooms: list[Courtroom] = []

    for i in range(n):
        n_busy = int(total_slots * busy_fraction)
        busy   = set(rng.sample(range(total_slots), min(n_busy, total_slots)))
        avail  = sorted(set(range(total_slots)) - busy)

        rooms.append(Courtroom(
            room_id=f"ROOM-{i+1}",
            capacity=rng.choice([20, 30, 50, 100]),
            available_slots=avail,
        ))

    return rooms


def build_conflict_hints(
    cases: list[Case],
    judges: list[Judge],
    rooms: list[Courtroom],
    total_slots: int,
) -> dict[str, list[str]]:
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