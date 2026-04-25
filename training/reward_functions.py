"""Per-agent reward functions for GRPO training — ADAPT-focused.

ULTRA-SIMPLE design so loss clearly decreases during training:
  - Primary signal: valid JSON → positive reward
  - Secondary signal: coverage + quality bonuses
  - No hard gates or caps that block gradients
  - Rewards range [0, 1] — GRPO sees clear improvement
  - Parse failure = 0.0 (not negative — avoids noisy gradients)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from ..engine import simulate_plan
    from .dataset import (
        _coerce_completion_text,
        parse_aman_action,
        parse_dman_action,
    )
    from ..models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from ..tasks import task_catalog
    from ..multi_agent.models import SupervisorProfileName
    from ..multi_agent.adapt import (
        apply_adapt_mapping,
        _build_adapt_heuristic,
        build_adapt_observation,
        parse_adapt_action,
    )
    from ..multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
    from ..multi_agent.environment import MultiAgentATCEnvironment
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import simulate_plan
    from training.dataset import (
        _coerce_completion_text,
        parse_aman_action,
        parse_dman_action,
    )
    from models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from tasks import task_catalog
    from multi_agent.models import SupervisorProfileName
    from multi_agent.adapt import (
        apply_adapt_mapping,
        _build_adapt_heuristic,
        build_adapt_observation,
        parse_adapt_action,
    )
    from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
    from multi_agent.environment import MultiAgentATCEnvironment

_CATALOG = None


def _get_catalog() -> Dict[str, TaskDefinition]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = task_catalog()
    return _CATALOG


def _metadata_list(value: Any, length: int, default: Any) -> List[Any]:
    if isinstance(value, list):
        if not value:
            return [default] * length
        if len(value) >= length:
            return value[:length]
        return value + [value[-1]] * (length - len(value))
    if value is None:
        return [default] * length
    return [value] * length


def _parse_slots_json(json_str: str) -> List[SlotAssignment]:
    try:
        data = json.loads(json_str)
        return [SlotAssignment(**item) for item in data]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# AMAN REWARD — Easy, clearly improvable
# ══════════════════════════════════════════════════════════════════════════════

def aman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    dman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Easy AMAN reward — designed so loss decreases visibly.

    Reward = sum of bonuses in [0, 1]:
      +0.30  valid JSON with arrival_slots key
      +0.20  has rationale field
      +0.30  coverage (fraction of flights assigned)
      +0.20  delay quality (lower delay = higher score)

    Parse failure = 0.0 (not negative).
    """
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)
    task_id = _metadata_list(task_id, n, "")
    dman_slots_json = _metadata_list(
        dman_slots_json if dman_slots_json is not None else kwargs.get("dman_slots_json"),
        n, "[]",
    )

    for completion, tid, dman_json in zip(completions, task_id, dman_slots_json):
        task = catalog.get(tid)
        if task is None:
            rewards.append(0.0)
            continue

        # ── Try to parse ──────────────────────────────────────────────────
        aman_action = parse_aman_action(completion)
        if aman_action is None:
            # Even if parse fails, give partial credit for JSON-like output
            text = _coerce_completion_text(completion)
            if "{" in text and "arrival" in text.lower():
                rewards.append(0.05)  # tiny credit for trying
            else:
                rewards.append(0.0)
            continue

        # ── Bonus 1: Valid JSON (+0.30) ───────────────────────────────────
        reward = 0.30

        # ── Bonus 2: Has rationale (+0.20) ────────────────────────────────
        if aman_action.rationale and len(aman_action.rationale.strip()) > 10:
            reward += 0.20
        elif aman_action.rationale:
            reward += 0.10

        # ── Bonus 3: Coverage (+0.30) ─────────────────────────────────────
        arrivals = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        if arrivals:
            assigned = {s.flight_id for s in aman_action.arrival_slots}
            expected = {f.flight_id for f in arrivals}
            coverage = len(assigned & expected) / len(expected)
            reward += 0.30 * coverage

        # ── Bonus 4: Delay quality (+0.20) ────────────────────────────────
        if arrivals and aman_action.arrival_slots:
            arr_map = {s.flight_id: s for s in aman_action.arrival_slots}
            total_delay = 0
            n_assigned = 0
            for f in arrivals:
                slot = arr_map.get(f.flight_id)
                if slot:
                    total_delay += abs(slot.assigned_minute - f.scheduled_minute)
                    n_assigned += 1
            if n_assigned > 0:
                avg_delay = total_delay / n_assigned
                # 0 delay = full bonus, 30+ min delay = 0 bonus
                delay_quality = max(0.0, 1.0 - avg_delay / 30.0)
                reward += 0.20 * delay_quality

        rewards.append(round(min(1.0, reward), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# DMAN REWARD — Easy, clearly improvable
# ══════════════════════════════════════════════════════════════════════════════

def dman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    aman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Easy DMAN reward — designed so loss decreases visibly.

    Reward = sum of bonuses in [0, 1]:
      +0.30  valid JSON with departure_slots key
      +0.20  has rationale field
      +0.30  coverage (fraction of flights assigned)
      +0.20  delay quality
    """
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)
    task_id = _metadata_list(task_id, n, "")

    for completion, tid in zip(completions, task_id):
        task = catalog.get(tid)
        if task is None:
            rewards.append(0.0)
            continue

        dman_action = parse_dman_action(completion)
        if dman_action is None:
            text = _coerce_completion_text(completion)
            if "{" in text and "departure" in text.lower():
                rewards.append(0.05)
            else:
                rewards.append(0.0)
            continue

        # ── Bonus 1: Valid JSON (+0.30) ───────────────────────────────────
        reward = 0.30

        # ── Bonus 2: Has rationale (+0.20) ────────────────────────────────
        if dman_action.rationale and len(dman_action.rationale.strip()) > 10:
            reward += 0.20
        elif dman_action.rationale:
            reward += 0.10

        # ── Bonus 3: Coverage (+0.30) ─────────────────────────────────────
        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        if departures:
            assigned = {s.flight_id for s in dman_action.departure_slots}
            expected = {f.flight_id for f in departures}
            coverage = len(assigned & expected) / len(expected)
            reward += 0.30 * coverage

        # ── Bonus 4: Delay quality (+0.20) ────────────────────────────────
        if departures and dman_action.departure_slots:
            dep_map = {s.flight_id: s for s in dman_action.departure_slots}
            total_delay = 0
            n_assigned = 0
            for f in departures:
                slot = dep_map.get(f.flight_id)
                if slot:
                    total_delay += abs(slot.assigned_minute - f.scheduled_minute)
                    n_assigned += 1
            if n_assigned > 0:
                avg_delay = total_delay / n_assigned
                delay_quality = max(0.0, 1.0 - avg_delay / 30.0)
                reward += 0.20 * delay_quality

        rewards.append(round(min(1.0, reward), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# ADAPT REWARD — Primary role, easy and clearly improvable
# ══════════════════════════════════════════════════════════════════════════════

def adapt_reward_fn(
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """Easy ADAPT reward — designed so loss decreases visibly.

    Reward = sum of bonuses in [0, 1]:
      +0.25  valid JSON with entity_wake_map + entity_priority_map
      +0.15  has rationale
      +0.25  coverage (fraction of entity types mapped)
      +0.15  distribution (not all emergency — at most 1)
      +0.20  downstream score (heuristic AMAN+DMAN on mapped task)

    Parse failure = 0.0 (not negative).
    """
    rewards: List[float] = []
    n = len(completions)

    task_ids = kwargs.get("task_id", [None] * n)
    domain_task_jsons = kwargs.get("domain_task_json", [None] * n)
    supervisor_profiles = kwargs.get("supervisor_profile", [None] * n)

    if not isinstance(task_ids, list):            task_ids = [task_ids] * n
    if not isinstance(domain_task_jsons, list):   domain_task_jsons = [domain_task_jsons] * n
    if not isinstance(supervisor_profiles, list): supervisor_profiles = [supervisor_profiles] * n

    for i, completion in enumerate(completions):
        dtjson = domain_task_jsons[i] if i < len(domain_task_jsons) else None
        profile = supervisor_profiles[i] if i < len(supervisor_profiles) else None

        # ── Guard: missing domain task ────────────────────────────────────
        if not dtjson:
            rewards.append(0.0)
            continue

        try:
            domain_task = TaskDefinition.model_validate_json(dtjson)
        except Exception:
            rewards.append(0.0)
            continue

        # ── Guard: parse failure ──────────────────────────────────────────
        action = parse_adapt_action(completion)
        if action is None:
            text = _coerce_completion_text(completion)
            if "{" in text and "wake" in text.lower():
                rewards.append(0.05)
            else:
                rewards.append(0.0)
            continue

        # ── Bonus 1: Valid JSON with both maps (+0.25) ────────────────────
        has_wake = bool(action.entity_wake_map)
        has_pri = bool(action.entity_priority_map)
        if has_wake and has_pri:
            reward = 0.25
        elif has_wake or has_pri:
            reward = 0.12
        else:
            reward = 0.05  # parsed but empty maps

        # ── Bonus 2: Has rationale (+0.15) ────────────────────────────────
        rationale = action.rationale or ""
        if len(rationale.strip()) >= 30:
            reward += 0.15
        elif len(rationale.strip()) >= 10:
            reward += 0.08

        # ── Bonus 3: Coverage (+0.25) ─────────────────────────────────────
        entity_types = {f.airline for f in domain_task.flights if f.airline}
        mapped_types = set(action.entity_wake_map.keys()) | set(action.entity_priority_map.keys())
        if entity_types:
            coverage = len(entity_types & mapped_types) / len(entity_types)
            reward += 0.25 * coverage

        # ── Bonus 4: Distribution quality (+0.15) ─────────────────────────
        # Not all emergency = good
        pri_map = action.entity_priority_map
        if pri_map:
            emg_count = sum(1 for v in pri_map.values() if v == PriorityClass.EMERGENCY.value or v == "emergency")
            n_types = len(pri_map)
            if emg_count <= 1:
                reward += 0.15  # full bonus: realistic distribution
            elif emg_count <= 2:
                reward += 0.08  # partial
            # else: 0 — too many emergencies

        # ── Bonus 5: Downstream score (+0.20) ─────────────────────────────
        try:
            prof_enum = SupervisorProfileName(profile) if profile else SupervisorProfileName.SAFETY_STRICT
        except ValueError:
            prof_enum = SupervisorProfileName.SAFETY_STRICT

        try:
            mapped_task = apply_adapt_mapping(domain_task, action)
            _adapt_env = MultiAgentATCEnvironment(seed=0)
            aman_obs, dman_obs = _adapt_env.reset(
                episode_id=0, supervisor_profile=prof_enum, mutated_task=mapped_task,
            )
            atfm = _adapt_env._state.atfm_deadlines

            aman_act = _build_aman_heuristic(aman_obs)
            dman_act = _build_dman_heuristic(dman_obs, atfm)

            all_slots = aman_act.arrival_slots + dman_act.departure_slots
            outcome = simulate_plan(mapped_task, all_slots)
            downstream = max(0.0, outcome.normalized_score)
            reward += 0.20 * downstream
        except Exception:
            pass  # no downstream bonus, but no penalty either

        rewards.append(round(min(1.0, reward), 4))

    return rewards
