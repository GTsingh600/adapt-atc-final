"""Per-agent reward functions for GRPO training — ADAPT-focused.

Simplified design for stable training with 1.5B–7B models:
  1. ADAPT gets the richest reward signal (primary training role)
  2. AMAN/DMAN get simple 5-component rewards (JSON-format + scheduling)
  3. Generator/Supervisor reward functions preserved but not called in training
  4. All experimental loss components DISABLED (preserved in loss_functions.py)
  5. Soft penalties for parse failure (-0.2) to allow gradient flow
  6. Safety gates widened for early-training signal diversity
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..engine import simulate_plan
    from .dataset import (
        _coerce_completion_text,
        parse_aman_action,
        parse_dman_action,
        parse_generator_action,
    )
    from ..models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from ..tasks import task_catalog
    from ..multi_agent.generator import ChallengeGenerator
    from ..multi_agent.models import SupervisorProfileName
    from ..multi_agent.supervisor import SupervisorAgent
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
        parse_generator_action,
    )
    from models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from tasks import task_catalog
    from multi_agent.generator import ChallengeGenerator
    from multi_agent.models import SupervisorProfileName
    from multi_agent.supervisor import SupervisorAgent
    from multi_agent.adapt import (
        apply_adapt_mapping,
        _build_adapt_heuristic,
        build_adapt_observation,
        parse_adapt_action,
    )
    from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
    from multi_agent.environment import MultiAgentATCEnvironment

_CATALOG = None
_SUPERVISOR = SupervisorAgent()
_GENERATOR = ChallengeGenerator()
_TRACE_REWARDS = os.getenv("ATC_REWARD_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
_DEFAULT_SUPERVISOR_PROFILE = SupervisorProfileName.SAFETY_STRICT


def _debug_reward_trace(role: str, components: Dict[str, float]) -> None:
    if not _TRACE_REWARDS:
        return
    print(f"\n[REWARD TRACE] role={role}")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")


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


def _safe_supervisor_profile(profile_value: Any) -> SupervisorProfileName:
    try:
        return SupervisorProfileName(profile_value)
    except Exception:
        return _DEFAULT_SUPERVISOR_PROFILE


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ── JSON format scorer ────────────────────────────────────────────────────────

def _json_format_score(completion: Any) -> float:
    """Returns 1.0 if completion contains valid JSON with expected keys, else 0.0."""
    text = _coerce_completion_text(completion)
    try:
        stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data = json.loads(stripped)
        if isinstance(data, dict) and (
            "arrival_slots" in data or "departure_slots" in data
            or "entity_wake_map" in data or "rationale" in data
        ):
            return 1.0
        return 0.5  # valid JSON but unexpected shape
    except Exception:
        if "{" in text and ("slots" in text or "rationale" in text or "map" in text):
            return 0.2
        return 0.0


def _parse_slots_json(json_str: str) -> List[SlotAssignment]:
    try:
        data = json.loads(json_str)
        return [SlotAssignment(**item) for item in data]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# AMAN REWARD — Simplified 5-component (support role)
# ══════════════════════════════════════════════════════════════════════════════

def aman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    dman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Simple 5-component AMAN reward for stable training.

    Components:
      0.35 × delay_efficiency   — minimize delay vs budget
      0.25 × emergency_score    — handle emergencies within 5 min
      0.20 × coverage           — assign all arrival flights
      0.10 × json_format        — produce valid JSON output
      0.10 × conflict_free      — no separation conflicts (binary)

    Safety gates (softened for gradient flow):
      conflict_count > 0  → max 0.50
      emergency missed    → max 0.60
      coverage < 50%      → penalty -0.15
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
            rewards.append(-0.3)
            continue

        aman_action = parse_aman_action(completion)
        if aman_action is None:
            rewards.append(-0.2)
            continue

        dman_slots = _parse_slots_json(dman_json)
        merged = aman_action.arrival_slots + dman_slots
        outcome = simulate_plan(task, merged)

        arrivals = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        arr_map = {s.flight_id: s for s in aman_action.arrival_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0

        for f in arrivals:
            slot = arr_map.get(f.flight_id)
            if slot:
                delay_total += abs(slot.assigned_minute - f.scheduled_minute)
                if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                    if abs(slot.assigned_minute - f.scheduled_minute) <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
            else:
                missing += 1

        arr_count = max(1, len(arrivals))
        budget = task.delay_budget / 2.0
        delay_eff = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / arr_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0
        conflict_free = 1.0 if outcome.metrics.conflict_count == 0 else 0.0
        json_fmt = _json_format_score(completion)

        # ── 5-component reward ────────────────────────────────────────────────
        reward = (
            0.35 * delay_eff
            + 0.25 * emg_score
            + 0.20 * coverage
            + 0.10 * json_fmt
            + 0.10 * conflict_free
        )

        # ── Softened safety gates ─────────────────────────────────────────────
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.50)
        if emg_miss > 0:
            reward = min(reward, 0.60)
        if coverage < 0.50:
            reward = max(-0.5, reward - 0.15)

        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace("AMAN", {
            "delay_eff": delay_eff, "emg_score": emg_score,
            "coverage": coverage, "json_fmt": json_fmt,
            "conflict_free": conflict_free, "final": reward,
        })
        rewards.append(reward)

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# DMAN REWARD — Simplified 6-component (support role)
# ══════════════════════════════════════════════════════════════════════════════

def dman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    aman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Simple 6-component DMAN reward for stable training.

    Components:
      0.30 × delay_efficiency
      0.20 × emergency_score
      0.20 × atfm_compliance
      0.15 × coverage
      0.10 × json_format
      0.05 × conflict_free
    """
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    aman_slots_json = _metadata_list(
        aman_slots_json if aman_slots_json is not None else kwargs.get("aman_slots_json"),
        n, "[]",
    )
    atfm_deadlines_json = _metadata_list(
        atfm_deadlines_json if atfm_deadlines_json is not None else kwargs.get("atfm_deadlines_json"),
        n, "{}",
    )

    for completion, tid, aman_json, atfm_json in zip(
        completions, task_id, aman_slots_json, atfm_deadlines_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-0.3)
            continue

        dman_action = parse_dman_action(completion)
        if dman_action is None:
            rewards.append(-0.2)
            continue

        aman_slots = _parse_slots_json(aman_json)
        atfm = json.loads(atfm_json) if atfm_json else {}
        merged = aman_slots + dman_action.departure_slots
        outcome = simulate_plan(task, merged)

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        dep_map = {s.flight_id: s for s in dman_action.departure_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0
        atfm_ok = atfm_viol = 0

        for f in departures:
            slot = dep_map.get(f.flight_id)
            if slot:
                delay_total += abs(slot.assigned_minute - f.scheduled_minute)
                if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                    if abs(slot.assigned_minute - f.scheduled_minute) <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
                deadline = atfm.get(f.flight_id)
                if deadline is not None:
                    if slot.assigned_minute <= deadline:
                        atfm_ok += 1
                    else:
                        atfm_viol += 1
            else:
                missing += 1

        dep_count = max(1, len(departures))
        budget = task.delay_budget / 2.0
        delay_eff = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage = 1.0 - missing / dep_count
        emg_score = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0
        atfm_score = atfm_ok / max(1, atfm_ok + atfm_viol) if (atfm_ok + atfm_viol) else 1.0
        conflict_free = 1.0 if outcome.metrics.conflict_count == 0 else 0.0
        json_fmt = _json_format_score(completion)

        # ── 6-component reward ────────────────────────────────────────────────
        reward = (
            0.30 * delay_eff
            + 0.20 * emg_score
            + 0.20 * atfm_score
            + 0.15 * coverage
            + 0.10 * json_fmt
            + 0.05 * conflict_free
        )

        # ── Softened safety gates ─────────────────────────────────────────────
        if outcome.metrics.conflict_count > 0:
            reward = min(reward, 0.50)
        if emg_miss > 0:
            reward = min(reward, 0.60)
        if coverage < 0.50:
            reward = max(-0.5, reward - 0.15)

        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace("DMAN", {
            "delay_eff": delay_eff, "atfm_score": atfm_score,
            "emg_score": emg_score, "coverage": coverage,
            "json_fmt": json_fmt, "conflict_free": conflict_free,
            "final": reward,
        })
        rewards.append(reward)

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR REWARD — Preserved but NOT called in training
# ══════════════════════════════════════════════════════════════════════════════

def generator_reward_fn(
    completions: List[str],
    task_id: List[str],
    controller_scores: Optional[List[float]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for GENERATOR role (not used in ADAPT-focused training)."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    controller_scores = _metadata_list(
        controller_scores if controller_scores is not None else kwargs.get("controller_scores"),
        n, 0.5,
    )

    for completion, tid, ctrl_score in zip(completions, task_id, controller_scores):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        gen_action = parse_generator_action(completion)
        if gen_action is None:
            rewards.append(-0.5)
            continue

        _, is_solvable, ctx = _GENERATOR.adapt(task, gen_action)
        reward = _GENERATOR.compute_reward(_safe_float(ctrl_score), is_solvable, ctx)
        rewards.append(round(max(-1.0, min(1.0, reward)), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR REWARD — Preserved but NOT called in training
# ══════════════════════════════════════════════════════════════════════════════

def supervisor_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    merged_plan_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """GRPO reward for SUPERVISOR role (not used in ADAPT-focused training)."""
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    supervisor_profile = _metadata_list(
        supervisor_profile if supervisor_profile is not None else kwargs.get("supervisor_profile"),
        n, _DEFAULT_SUPERVISOR_PROFILE.value,
    )
    merged_plan_json = _metadata_list(
        merged_plan_json if merged_plan_json is not None else kwargs.get("merged_plan_json"),
        n, "[]",
    )

    for completion, tid, profile_str, plan_json in zip(
        completions, task_id, supervisor_profile, merged_plan_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        try:
            plan_data = json.loads(plan_json)
            slots = [SlotAssignment(**s) for s in plan_data]
        except Exception:
            rewards.append(-0.5)
            continue

        outcome = simulate_plan(task, slots)
        profile = _safe_supervisor_profile(profile_str)
        score = _SUPERVISOR.score_plan(outcome, task, profile)
        rewards.append(round(min(1.0, score), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# ADAPT REWARD — PRIMARY training role. Fast structural evaluation.
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_distribution_penalty(action) -> float:
    """Penalise degenerate priority distributions that cause AMAN/DMAN starvation.

    Two violations, each worth up to 0.30:
      1. emergency_overuse: more than 1 entity type mapped to emergency
      2. high_tier_concentration: >60% of entity types mapped to emergency+medical
    Total capped at 0.50.
    """
    pri_map = action.entity_priority_map
    if not pri_map:
        return 0.0

    n_types = len(pri_map)
    emergency_count = sum(1 for v in pri_map.values() if v == PriorityClass.EMERGENCY.value)
    medical_count = sum(1 for v in pri_map.values() if v == PriorityClass.MEDICAL.value)
    high_tier_count = emergency_count + medical_count

    excess_emg = max(0, emergency_count - 1)
    emg_penalty = 0.30 * (excess_emg / n_types) if excess_emg > 0 else 0.0

    concentration = high_tier_count / n_types
    conc_penalty = 0.0
    if concentration > 0.60:
        conc_penalty = 0.30 * min(1.0, (concentration - 0.60) / 0.40)

    return round(min(0.50, emg_penalty + conc_penalty), 4)


def adapt_reward_fn(
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """ADAPT reward — primary training signal. Fast 5-component evaluation.

    Components (summed, clamped to [-1, 1]):
      0.15 × parse_quality      — valid JSON with correct keys
      0.20 × coverage           — fraction of entity types mapped
      0.20 × distribution       — realistic priority distribution (not all emergency)
      0.35 × downstream_score   — ONE heuristic AMAN+DMAN episode on mapped task
      0.10 × rationale_quality  — rationale cites numbers, ≥30 chars

    Parse failure → -0.2 (soft, allows gradient flow).
    Missing domain task → -0.3.
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

        # ── Guard: missing domain task ────────────────────────────────────────
        if not dtjson:
            rewards.append(-0.3)
            continue

        try:
            domain_task = TaskDefinition.model_validate_json(dtjson)
        except Exception:
            rewards.append(-0.3)
            continue

        # ── Guard: parse failure (soft penalty) ───────────────────────────────
        action = parse_adapt_action(completion)
        if action is None:
            rewards.append(-0.2)
            continue

        # ── 1. Parse quality (0.15) — valid JSON with both maps present ──────
        has_wake = bool(action.entity_wake_map)
        has_pri = bool(action.entity_priority_map)
        parse_quality = 1.0 if (has_wake and has_pri) else (0.5 if (has_wake or has_pri) else 0.0)

        # ── 2. Coverage (0.20) — fraction of entity types mapped ─────────────
        entity_types = {f.airline for f in domain_task.flights if f.airline}
        mapped_types = set(action.entity_wake_map.keys()) | set(action.entity_priority_map.keys())
        coverage = len(entity_types & mapped_types) / max(1, len(entity_types))

        # ── 3. Distribution quality (0.20) — realistic priority spread ───────
        dist_penalty = _adapt_distribution_penalty(action)
        distribution = max(0.0, 1.0 - dist_penalty * 2.0)  # penalty 0.5 → score 0.0

        # ── 4. Downstream composite (0.35) — single heuristic episode ────────
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
            downstream = outcome.normalized_score
        except Exception:
            downstream = 0.0

        # ── 5. Rationale quality (0.10) — cites numbers, explains logic ──────
        rationale = action.rationale or ""
        rationale_score = 0.0
        if len(rationale.strip()) >= 30:
            rationale_score += 0.5
        if re.search(r"\d+\.\d+", rationale):
            rationale_score += 0.3
        has_mapping_words = any(w in rationale.lower() for w in
                                ("wake", "priority", "emergency", "medical", "map", "score"))
        if has_mapping_words:
            rationale_score += 0.2
        rationale_score = min(1.0, rationale_score)

        # ── Compose final reward ─────────────────────────────────────────────
        reward = (
            0.15 * parse_quality
            + 0.20 * coverage
            + 0.20 * distribution
            + 0.35 * downstream
            + 0.10 * rationale_score
        )
        reward = round(max(-1.0, min(1.0, reward)), 4)

        _debug_reward_trace("ADAPT", {
            "parse_quality": parse_quality,
            "coverage": coverage,
            "distribution": distribution,
            "downstream": downstream,
            "rationale": rationale_score,
            "final": reward,
        })
        rewards.append(reward)

    return rewards
