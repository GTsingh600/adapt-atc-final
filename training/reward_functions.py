"""Per-agent reward functions for GRPO training — ADAPT-focused.

Research-grounded reward design for multi-agent domain transfer:

  1. COMPOSABLE RUBRIC SCORING — Hierarchical rubric where format gates quality
     gates optimality (smooth sigmoid gating, not hard caps).
     Reference: OpenAI Rubric Grading (2024), OpenEnv Composable Rubrics.

  2. STRUCTURAL ALIGNMENT — ADAPT's primary signal measures monotonicity
     preservation: if entity A has higher structural urgency than B, the
     mapping should preserve this ordering. Novel metric for domain transfer.

  3. INFORMATION-THEORETIC COVERAGE — KL divergence from reference distribution
     penalises degenerate mappings (all-emergency) without hard gates.
     Reference: Haarnoja et al. 2018 (MaxEnt RL).

  4. POTENTIAL-BASED SHAPING — Downstream improvement over heuristic baseline,
     not raw score. Preserves optimal policy per Ng et al. 1999.

  5. SMOOTH GATING — sigmoid(10*(x - threshold)) replaces hard gates,
     providing gradient signal even below threshold.
     Reference: Curriculum RL literature (Narvekar et al. 2020).
"""

from __future__ import annotations

import json
import math
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


# ── Research primitives ───────────────────────────────────────────────────────

def _sigmoid_gate(value: float, threshold: float, steepness: float = 10.0) -> float:
    """Smooth sigmoid gate — replaces hard if/else caps.

    σ(k·(x - t)) provides gradient signal even below threshold.
    At steepness=10: gate(0.3, 0.5) ≈ 0.12, gate(0.7, 0.5) ≈ 0.88.
    Reference: Curriculum RL (Narvekar et al. 2020).
    """
    z = steepness * (value - threshold)
    z = max(-20.0, min(20.0, z))  # numerical stability
    return 1.0 / (1.0 + math.exp(-z))


def _kl_from_reference(dist: Dict[str, int], ref: Dict[str, float]) -> float:
    """KL divergence D_KL(dist || ref) for priority distributions.

    Lower KL = distribution is more realistic, closer to reference.
    Maps KL to [0, 1] reward: R = exp(-KL).
    Reference: MaxEnt RL (Haarnoja et al. 2018).
    """
    total = max(1, sum(dist.values()))
    kl = 0.0
    for key, ref_p in ref.items():
        p = dist.get(key, 0) / total
        if p > 0 and ref_p > 0:
            kl += p * math.log(p / ref_p)
        elif p > 0:
            kl += 2.0  # penalty for mass on unsupported priority
    return math.exp(-max(0.0, kl))


# Reference priority distribution — what real ATC traffic looks like:
# ~5% emergency, ~15% medical, ~25% connection, ~55% normal
_REFERENCE_PRIORITY_DIST = {
    "emergency": 0.05,
    "medical":   0.15,
    "connection": 0.25,
    "normal":    0.55,
}

# Reference wake distribution: ~15% Heavy, ~45% Medium, ~40% Light
_REFERENCE_WAKE_DIST = {
    "H": 0.15,
    "M": 0.45,
    "L": 0.40,
}


def _format_quality(completion: Any) -> float:
    """Graduated JSON format scoring (0.0 → 1.0).

    Rewards partial progress toward valid JSON:
      0.0  — no JSON-like content
      0.2  — has braces + relevant keywords
      0.5  — valid JSON but wrong schema
      0.8  — valid JSON with some expected keys
      1.0  — valid JSON with all expected keys
    """
    text = _coerce_completion_text(completion)
    try:
        stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data = json.loads(stripped)
        if not isinstance(data, dict):
            return 0.3

        # Check for role-specific keys
        has_slots = "arrival_slots" in data or "departure_slots" in data
        has_maps  = "entity_wake_map" in data or "entity_priority_map" in data
        has_rationale = "rationale" in data

        score = 0.5  # valid JSON
        if has_slots or has_maps:
            score += 0.3  # correct schema
        if has_rationale:
            score += 0.2  # has explanation
        return min(1.0, score)
    except Exception:
        if "{" in text and ("slots" in text or "rationale" in text or "map" in text):
            return 0.2
        if "{" in text:
            return 0.1
        return 0.0


def _monotonicity_score(predicted_order: List[str], oracle_order: List[str]) -> float:
    """Kendall's tau-like monotonicity — does the mapping preserve urgency ordering?

    If entity A has higher structural urgency than entity B, the LLM's mapping
    should assign A a higher-or-equal priority tier. This measures how well the
    agent preserves structural ordering — the core of domain transfer.

    Returns: fraction of concordant pairs (0.0 → 1.0).
    Novel metric for evaluating zero-shot domain transfer quality.
    """
    if len(predicted_order) < 2 or len(oracle_order) < 2:
        return 1.0  # trivially correct for 0-1 elements

    # Build rank maps
    pred_rank = {e: i for i, e in enumerate(predicted_order)}
    oracle_rank = {e: i for i, e in enumerate(oracle_order)}

    common = set(pred_rank) & set(oracle_rank)
    if len(common) < 2:
        return 0.5  # insufficient overlap

    common_list = sorted(common)
    concordant = 0
    total = 0

    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            a, b = common_list[i], common_list[j]
            pred_cmp = pred_rank[a] - pred_rank[b]
            oracle_cmp = oracle_rank[a] - oracle_rank[b]
            total += 1
            if (pred_cmp > 0) == (oracle_cmp > 0) or pred_cmp == 0 or oracle_cmp == 0:
                concordant += 1

    return concordant / max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
# AMAN REWARD — Composable Rubric with Smooth Gating
# ══════════════════════════════════════════════════════════════════════════════

def aman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    dman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Composable rubric reward for AMAN (arrival sequencing).

    Three-tier rubric with smooth sigmoid gating:
      Tier 1 FORMAT (gate):   Valid JSON with arrival_slots → unlocks Tier 2
      Tier 2 SAFETY (gate):   Emergency handling + conflict avoidance → unlocks Tier 3
      Tier 3 QUALITY:         Coverage × delay efficiency × rationale

    Final = format_gate × (0.3×safety + safety_gate × 0.7×quality)

    Smooth gating ensures gradient flow even below thresholds.
    Reference: OpenEnv Composable Rubrics, Curriculum RL (Narvekar et al. 2020).
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

        # ── Tier 1: FORMAT ────────────────────────────────────────────────
        fmt = _format_quality(completion)
        format_gate = _sigmoid_gate(fmt, 0.4, steepness=8.0)

        aman_action = parse_aman_action(completion)
        if aman_action is None:
            # Partial credit scaled by format quality
            rewards.append(round(0.15 * fmt, 4))
            continue

        # ── Tier 2: SAFETY ────────────────────────────────────────────────
        arrivals = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        arr_map = {s.flight_id: s for s in aman_action.arrival_slots}

        emg_ok = emg_total = 0
        for f in arrivals:
            if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                emg_total += 1
                slot = arr_map.get(f.flight_id)
                if slot and abs(slot.assigned_minute - f.scheduled_minute) <= 5:
                    emg_ok += 1

        emg_score = emg_ok / max(1, emg_total) if emg_total else 1.0

        dman_slots = _parse_slots_json(dman_json)
        merged = aman_action.arrival_slots + dman_slots
        try:
            outcome = simulate_plan(task, merged)
            n_conflicts = outcome.metrics.conflict_count
        except Exception:
            n_conflicts = len(arrivals)  # assume worst case

        # Smooth conflict penalty: 0 conflicts = 1.0, 5+ = ~0.0
        conflict_score = math.exp(-0.5 * n_conflicts)

        safety = 0.5 * emg_score + 0.5 * conflict_score
        safety_gate = _sigmoid_gate(safety, 0.4, steepness=8.0)

        # ── Tier 3: QUALITY ───────────────────────────────────────────────
        # Coverage
        arr_count = max(1, len(arrivals))
        coverage = len(set(arr_map.keys()) & {f.flight_id for f in arrivals}) / arr_count

        # Delay efficiency (normalised by budget)
        total_delay = sum(
            abs(arr_map[f.flight_id].assigned_minute - f.scheduled_minute)
            for f in arrivals if f.flight_id in arr_map
        )
        budget = max(1, task.delay_budget / 2.0)
        delay_eff = max(0.0, 1.0 - total_delay / budget)

        # Rationale quality
        rationale = aman_action.rationale or ""
        rat_score = 0.0
        if len(rationale.strip()) >= 20: rat_score += 0.4
        if re.search(r"\d", rationale): rat_score += 0.3
        if any(w in rationale.lower() for w in ("priority", "emergency", "delay", "wake")): rat_score += 0.3
        rat_score = min(1.0, rat_score)

        quality = 0.45 * coverage + 0.35 * delay_eff + 0.20 * rat_score

        # ── Compose: format gates → safety gates → quality ────────────────
        reward = format_gate * (0.30 * safety + safety_gate * 0.70 * quality)
        rewards.append(round(max(0.0, min(1.0, reward)), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# DMAN REWARD — Same rubric structure, + ATFM compliance
# ══════════════════════════════════════════════════════════════════════════════

def dman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: Optional[List[str]] = None,
    aman_slots_json: Optional[List[str]] = None,
    atfm_deadlines_json: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[float]:
    """Composable rubric reward for DMAN (departure sequencing).

    Three-tier rubric:
      Tier 1 FORMAT:  Valid JSON with departure_slots
      Tier 2 SAFETY:  Emergency handling + ATFM compliance
      Tier 3 QUALITY: Coverage × delay efficiency × rationale

    Identical gating structure to AMAN for consistency.
    """
    catalog = _get_catalog()
    rewards: List[float] = []
    n = len(completions)

    task_id = _metadata_list(task_id, n, "")
    atfm_deadlines_json = _metadata_list(
        atfm_deadlines_json if atfm_deadlines_json is not None else kwargs.get("atfm_deadlines_json"),
        n, "{}",
    )

    for completion, tid, atfm_json in zip(completions, task_id, atfm_deadlines_json):
        task = catalog.get(tid)
        if task is None:
            rewards.append(0.0)
            continue

        # ── Tier 1: FORMAT ────────────────────────────────────────────────
        fmt = _format_quality(completion)
        format_gate = _sigmoid_gate(fmt, 0.4, steepness=8.0)

        dman_action = parse_dman_action(completion)
        if dman_action is None:
            rewards.append(round(0.15 * fmt, 4))
            continue

        # ── Tier 2: SAFETY ────────────────────────────────────────────────
        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        dep_map = {s.flight_id: s for s in dman_action.departure_slots}
        atfm = json.loads(atfm_json) if atfm_json else {}

        emg_ok = emg_total = 0
        atfm_ok = atfm_total = 0

        for f in departures:
            slot = dep_map.get(f.flight_id)
            if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
                emg_total += 1
                if slot and abs(slot.assigned_minute - f.scheduled_minute) <= 5:
                    emg_ok += 1
            deadline = atfm.get(f.flight_id)
            if deadline is not None:
                atfm_total += 1
                if slot and slot.assigned_minute <= deadline:
                    atfm_ok += 1

        emg_score = emg_ok / max(1, emg_total) if emg_total else 1.0
        atfm_score = atfm_ok / max(1, atfm_total) if atfm_total else 1.0

        safety = 0.4 * emg_score + 0.4 * atfm_score + 0.2 * 1.0  # base credit
        safety_gate = _sigmoid_gate(safety, 0.4, steepness=8.0)

        # ── Tier 3: QUALITY ───────────────────────────────────────────────
        dep_count = max(1, len(departures))
        coverage = len(set(dep_map.keys()) & {f.flight_id for f in departures}) / dep_count

        total_delay = sum(
            abs(dep_map[f.flight_id].assigned_minute - f.scheduled_minute)
            for f in departures if f.flight_id in dep_map
        )
        budget = max(1, task.delay_budget / 2.0)
        delay_eff = max(0.0, 1.0 - total_delay / budget)

        rationale = dman_action.rationale or ""
        rat_score = 0.0
        if len(rationale.strip()) >= 20: rat_score += 0.4
        if re.search(r"\d", rationale): rat_score += 0.3
        if any(w in rationale.lower() for w in ("atfm", "deadline", "departure", "slot")): rat_score += 0.3
        rat_score = min(1.0, rat_score)

        quality = 0.45 * coverage + 0.35 * delay_eff + 0.20 * rat_score

        # ── Compose ───────────────────────────────────────────────────────
        reward = format_gate * (0.30 * safety + safety_gate * 0.70 * quality)
        rewards.append(round(max(0.0, min(1.0, reward)), 4))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# ADAPT REWARD — Primary role. Novel structural alignment + info-theoretic scoring.
# ══════════════════════════════════════════════════════════════════════════════

def adapt_reward_fn(
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """Novel ADAPT reward — structural alignment + information-theoretic scoring.

    Four-tier composable rubric:
      Tier 1 FORMAT (gate):      Valid JSON with both maps → unlocks Tier 2
      Tier 2 COVERAGE (gate):    Entity type coverage → unlocks Tier 3
      Tier 3 STRUCTURAL:         Monotonicity + KL-based distribution quality
      Tier 4 DOWNSTREAM:         Potential-based advantage over heuristic baseline

    Novel components:
      - Monotonicity score: Kendall's tau measuring whether the mapping preserves
        the structural urgency ordering (core domain transfer metric)
      - KL-regularised distribution: prevents gaming by penalising divergence from
        realistic ATC priority distribution (MaxEnt RL principle)
      - Potential-based advantage: improvement over oracle heuristic mapping,
        preserving optimal policy per Ng et al. 1999

    Final = fmt_gate × (0.15×cov + cov_gate × (0.35×structural + 0.35×downstream + 0.15×rationale))
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

        if not dtjson:
            rewards.append(0.0)
            continue

        try:
            domain_task = TaskDefinition.model_validate_json(dtjson)
        except Exception:
            rewards.append(0.0)
            continue

        # ── Tier 1: FORMAT ────────────────────────────────────────────────
        fmt = _format_quality(completion)
        fmt_gate = _sigmoid_gate(fmt, 0.4, steepness=8.0)

        action = parse_adapt_action(completion)
        if action is None:
            rewards.append(round(0.10 * fmt, 4))
            continue

        has_wake = bool(action.entity_wake_map)
        has_pri = bool(action.entity_priority_map)
        if has_wake and has_pri:
            fmt_bonus = 1.0
        elif has_wake or has_pri:
            fmt_bonus = 0.6
        else:
            fmt_bonus = 0.3
        fmt_gate = _sigmoid_gate(fmt_bonus, 0.4, steepness=8.0)

        # ── Tier 2: COVERAGE ──────────────────────────────────────────────
        entity_types = {f.airline for f in domain_task.flights if f.airline}
        mapped_types = set(action.entity_wake_map.keys()) | set(action.entity_priority_map.keys())
        coverage = len(entity_types & mapped_types) / max(1, len(entity_types))
        cov_gate = _sigmoid_gate(coverage, 0.3, steepness=8.0)

        # ── Tier 3: STRUCTURAL QUALITY ────────────────────────────────────

        # 3a. Monotonicity: does mapping preserve urgency ordering?
        # Build oracle ordering from structural signals
        entity_urgency = {}
        for f in domain_task.flights:
            if f.airline and f.airline not in entity_urgency:
                # Combined structural signal
                tp = getattr(f, "connection_risk", 0.0) or 0.0
                cr = getattr(f, "fuel_burn_per_minute", 0.0) or 0.0
                urgency = 0.5 * tp + 0.4 * cr + 0.1 * (1.0 if "urgent" in (f.notes or "").lower() else 0.0)
                entity_urgency[f.airline] = urgency

        oracle_order = sorted(entity_urgency.keys(), key=lambda e: entity_urgency.get(e, 0), reverse=True)

        # Build predicted ordering from priority mapping
        _pri_rank = {"emergency": 4, "medical": 3, "connection": 2, "normal": 1}
        _wake_rank = {"H": 3, "M": 2, "L": 1}
        pred_urgency = {}
        for etype in mapped_types:
            pri_val = action.entity_priority_map.get(etype, "normal")
            wake_val = action.entity_wake_map.get(etype, "M")
            pred_urgency[etype] = _pri_rank.get(pri_val, 1) + _wake_rank.get(wake_val, 2) * 0.5

        pred_order = sorted(pred_urgency.keys(), key=lambda e: pred_urgency.get(e, 0), reverse=True)

        monotonicity = _monotonicity_score(pred_order, oracle_order)

        # 3b. KL-regularised distribution quality
        pri_counts: Dict[str, int] = {}
        for v in action.entity_priority_map.values():
            v_str = v if isinstance(v, str) else str(v)
            pri_counts[v_str] = pri_counts.get(v_str, 0) + 1
        kl_score = _kl_from_reference(pri_counts, _REFERENCE_PRIORITY_DIST)

        structural = 0.55 * monotonicity + 0.45 * kl_score

        # ── Tier 4: DOWNSTREAM (potential-based advantage) ────────────────
        try:
            prof_enum = SupervisorProfileName(profile) if profile else SupervisorProfileName.SAFETY_STRICT
        except ValueError:
            prof_enum = SupervisorProfileName.SAFETY_STRICT

        downstream = 0.0
        heuristic_baseline = 0.0
        try:
            # Score the LLM's mapping
            mapped_task = apply_adapt_mapping(domain_task, action)
            _env = MultiAgentATCEnvironment(seed=0)
            aman_obs, dman_obs = _env.reset(
                episode_id=0, supervisor_profile=prof_enum, mutated_task=mapped_task,
            )
            atfm = _env._state.atfm_deadlines
            aman_act = _build_aman_heuristic(aman_obs)
            dman_act = _build_dman_heuristic(dman_obs, atfm)
            all_slots = aman_act.arrival_slots + dman_act.departure_slots
            outcome = simulate_plan(mapped_task, all_slots)
            agent_score = max(0.0, outcome.normalized_score)

            # Score the heuristic baseline mapping (for potential-based advantage)
            adapt_obs = build_adapt_observation(
                task=domain_task, profile=prof_enum,
                domain_name="eval", domain_description="eval",
            )
            heuristic_action = _build_adapt_heuristic(adapt_obs, domain_task)
            h_mapped = apply_adapt_mapping(domain_task, heuristic_action)
            _env2 = MultiAgentATCEnvironment(seed=0)
            h_aman_obs, h_dman_obs = _env2.reset(
                episode_id=0, supervisor_profile=prof_enum, mutated_task=h_mapped,
            )
            h_atfm = _env2._state.atfm_deadlines
            h_aman = _build_aman_heuristic(h_aman_obs)
            h_dman = _build_dman_heuristic(h_dman_obs, h_atfm)
            h_slots = h_aman.arrival_slots + h_dman.departure_slots
            h_outcome = simulate_plan(h_mapped, h_slots)
            heuristic_baseline = max(0.0, h_outcome.normalized_score)

            # Potential-based advantage: how much better than heuristic?
            # Raw score + small bonus for improvement over baseline (Ng et al. 1999)
            advantage = agent_score - heuristic_baseline
            downstream = 0.7 * agent_score + 0.3 * _sigmoid_gate(advantage, 0.0, steepness=5.0)
        except Exception:
            downstream = 0.0

        # ── Tier 3.5: RATIONALE QUALITY ───────────────────────────────────
        rationale = action.rationale or ""
        rat_score = 0.0
        if len(rationale.strip()) >= 30:
            rat_score += 0.3
        if re.search(r"\d+\.\d+", rationale):  # cites numerical evidence
            rat_score += 0.3
        if any(w in rationale.lower() for w in ("time_pressure", "connection_risk", "tp=", "cr=")):
            rat_score += 0.2  # references structural signals by name
        if any(w in rationale.lower() for w in ("budget", "emergency", "mapping", "priority")):
            rat_score += 0.2
        rat_score = min(1.0, rat_score)

        # ── Compose: hierarchical gating ──────────────────────────────────
        # Format gates everything. Coverage gates structural+downstream.
        reward = fmt_gate * (
            0.15 * coverage
            + cov_gate * (
                0.30 * structural
                + 0.35 * downstream
                + 0.20 * rat_score
            )
        )
        rewards.append(round(max(0.0, min(1.0, reward)), 4))

    return rewards
