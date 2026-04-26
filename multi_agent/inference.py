"""Multi-agent ATC inference runner.

Runs AMAN + DMAN agents against the ATC environment, either with:
  - heuristic-baseline: deterministic planner (no LLM required)
  - any HF/OpenAI-compatible model: LLM-driven AMAN and DMAN agents

Output format mirrors inference.py structured logs:
  [START] task=... env=... model=...
  [STEP]  step=... role=AMAN|DMAN reward=... conflicts=... done=...
  [NEG]   round=... conflicts_resolved=... messages=...
  [END]   task=... success=... composite=... aman=... dman=... coord=...

Usage:
  python multi_agent/inference.py                           # heuristic demo
  python multi_agent/inference.py --model Qwen/Qwen2.5-7B  # LLM agents
  python multi_agent/inference.py --task bengaluru_irrops_hard --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import simulate_plan
from graders import grade_multi_agent
from models import (
    FlightRecord,
    OperationType,
    PriorityClass,
    SlotAssignment,
    TaskDefinition,
)
from planner import build_heuristic_plan, build_refined_plan, _flight_sort_key
from tasks import task_catalog, ordered_tasks
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from multi_agent.models import (
    AMANAction,
    AgentRole,
    DMANAction,
    NegotiationMessage,
    MessageType,
    MultiAgentObservation,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
)
from training.dataset import (
    AMAN_SYSTEM,
    DMAN_SYSTEM,
    parse_aman_action,
    parse_dman_action,
)


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME", "heuristic-baseline")
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
BENCHMARK    = "atc_multiagent_openenv"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3
SUCCESS_THRESHOLD = 0.60


# ── Structured logging ────────────────────────────────────────────────────────

def _p(msg: str) -> None:
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass


def log_start(task: str, model: str) -> None:
    _p(f"[START] task={task} env={BENCHMARK} model={model}")


def log_step(step: int, role: str, reward: float, conflicts: int, done: bool) -> None:
    _p(f"[STEP]  step={step} role={role} reward={reward:.3f} conflicts={conflicts} done={str(done).lower()}")


def log_neg(rnd: int, resolved: int, messages: int) -> None:
    _p(f"[NEG]   round={rnd} conflicts_resolved={resolved} messages={messages}")


def log_end(
    task: str, success: bool, composite: float,
    aman: float, dman: float, coord: float, gen_diff: int,
) -> None:
    _p(
        f"[END]   task={task} success={str(success).lower()} "
        f"composite={composite:.3f} aman={aman:.3f} dman={dman:.3f} "
        f"coord={coord:.3f} gen_difficulty={gen_diff}"
    )


# ── Heuristic AMAN planner ────────────────────────────────────────────────────

def _build_aman_heuristic(obs: MultiAgentObservation) -> AMANAction:
    """Deterministic arrival sequencer — priority-sorted, wake-turbulence aware."""
    arrivals = sorted(obs.my_flights, key=_flight_sort_key)
    runway_last: Dict[str, Tuple[int, str]] = {
        r.runway_id: (0, "M") for r in obs.all_runways
    }
    from constants import SEPARATION_BY_WAKE
    slots: List[SlotAssignment] = []
    emergency_yields: List[str] = []

    # Check for DMAN emergency broadcasts — yield slots around those minutes
    dman_emg_slots: List[Tuple[str, int]] = [
        (m.runway_id, m.requested_minute)
        for m in obs.incoming_messages
        if m.is_emergency and m.from_role == AgentRole.DMAN
    ]

    for flight in arrivals:
        best_rwy, best_min = None, None
        best_gap = -1

        for rwy_id in flight.allowed_runways:
            rwy = next((r for r in obs.all_runways if r.runway_id == rwy_id), None)
            if not rwy:
                continue
            if OperationType.ARRIVAL not in rwy.allowed_operations:
                continue

            last_min, last_wake = runway_last.get(rwy_id, (0, "M"))
            from engine import _capacity_spacing
            cap_gap = _capacity_spacing(rwy)
            wake_gap = SEPARATION_BY_WAKE.get(
                (last_wake, flight.wake_class.value), 3
            )
            min_gap = max(cap_gap, wake_gap)
            earliest = max(flight.earliest_minute, last_min + min_gap)

            if earliest > flight.latest_minute:
                continue

            # Avoid DMAN emergency slots (±3 min)
            blocked = any(
                rwy_id == er and abs(earliest - et) < 3
                for er, et in dman_emg_slots
            )
            if blocked:
                earliest = min(flight.latest_minute, earliest + 3)
                if earliest > flight.latest_minute:
                    continue
                emergency_yields.append(flight.flight_id)

            gap_to_scheduled = abs(earliest - flight.scheduled_minute)
            if best_rwy is None or gap_to_scheduled < best_gap:
                best_rwy = rwy_id
                best_min = earliest
                best_gap = gap_to_scheduled

        if best_rwy and best_min is not None:
            slots.append(SlotAssignment(
                flight_id=flight.flight_id,
                runway=best_rwy,
                assigned_minute=best_min,
                hold_minutes=max(0, best_min - flight.scheduled_minute),
            ))
            runway_last[best_rwy] = (best_min, flight.wake_class.value)

    # Build outgoing messages — claim slots and broadcast emergencies
    messages: List[NegotiationMessage] = []
    for s in slots:
        flight = next((f for f in arrivals if f.flight_id == s.flight_id), None)
        if flight and flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
            messages.append(NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType.EMERGENCY_BCAST,
                flight_id=s.flight_id,
                requested_minute=s.assigned_minute,
                runway_id=s.runway,
                priority=flight.priority,
                reason=f"Emergency arrival — runway {s.runway} must be clear at T+{s.assigned_minute}",
                is_emergency=True,
            ))
        else:
            messages.append(NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType.RUNWAY_CLAIM,
                flight_id=s.flight_id if flight else "?",
                requested_minute=s.assigned_minute,
                runway_id=s.runway,
                priority=flight.priority if flight else PriorityClass.NORMAL,
                reason="Arrival slot claim",
                is_emergency=False,
            ))

    return AMANAction(
        arrival_slots=slots,
        rationale=f"Heuristic AMAN: {len(slots)}/{len(arrivals)} arrivals sequenced, "
                  f"priority-sorted, wake-turbulence aware.",
        emergency_yields=emergency_yields,
        outgoing_messages=messages,
        commit=False,
    )


# ── Heuristic DMAN planner ────────────────────────────────────────────────────

def _build_dman_heuristic(
    obs: MultiAgentObservation,
    atfm_deadlines: Dict[str, int],
) -> DMANAction:
    """Deterministic departure sequencer — ATFM-aware, emergency priority."""
    from constants import SEPARATION_BY_WAKE
    from engine import _capacity_spacing
    from planner import PRIORITY_RANK

    def _departure_sort_key(flight: FlightRecord) -> Tuple[int, int, int, int, float, int]:
        deadline = atfm_deadlines.get(flight.flight_id)
        # Keep emergency traffic first, then pull ATFM-constrained flights ahead of
        # unconstrained traffic so hard network slot windows are not accidentally lost.
        deadline_bucket = 0 if flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL) else (
            1 if deadline is not None else 2
        )
        return (
            deadline_bucket,
            deadline if deadline is not None else flight.scheduled_minute,
            PRIORITY_RANK[flight.priority],
            flight.scheduled_minute,
            -flight.connection_risk,
            -flight.passengers,
        )

    departures = sorted(obs.my_flights, key=_departure_sort_key)
    runway_last: Dict[str, Tuple[int, str]] = {
        r.runway_id: (0, "M") for r in obs.all_runways
    }

    # Reserve slots claimed by AMAN
    aman_claims: Dict[str, List[int]] = {}
    for m in obs.incoming_messages:
        if m.from_role == AgentRole.AMAN:
            aman_claims.setdefault(m.runway_id, []).append(m.requested_minute)

    slots: List[SlotAssignment] = []
    atfm_compliance: Dict[str, int] = {}
    emergency_broadcasts: List[str] = []
    messages: List[NegotiationMessage] = []

    for flight in departures:
        is_emg = flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        deadline = atfm_deadlines.get(flight.flight_id)
        best_rwy, best_min = None, None
        best_score = float("inf")

        for rwy_id in flight.allowed_runways:
            rwy = next((r for r in obs.all_runways if r.runway_id == rwy_id), None)
            if not rwy:
                continue
            if OperationType.DEPARTURE not in rwy.allowed_operations:
                continue

            last_min, last_wake = runway_last.get(rwy_id, (0, "M"))
            cap_gap  = _capacity_spacing(rwy)
            wake_gap = SEPARATION_BY_WAKE.get((last_wake, flight.wake_class.value), 3)
            min_gap  = max(cap_gap, wake_gap)
            earliest = max(flight.earliest_minute, last_min + min_gap)

            if earliest > flight.latest_minute:
                continue

            # Avoid AMAN-claimed slots (±3 min) unless this is an emergency
            if not is_emg:
                aman_mins = aman_claims.get(rwy_id, [])
                while any(abs(earliest - am) < 3 for am in aman_mins):
                    earliest += 1
                if earliest > flight.latest_minute:
                    continue

            # ATFM deadline constraint
            if deadline is not None and earliest > deadline:
                continue

            score = abs(earliest - flight.scheduled_minute)
            if score < best_score:
                best_rwy  = rwy_id
                best_min  = earliest
                best_score = score

        if best_rwy and best_min is not None:
            slots.append(SlotAssignment(
                flight_id=flight.flight_id,
                runway=best_rwy,
                assigned_minute=best_min,
                hold_minutes=max(0, best_min - flight.scheduled_minute),
            ))
            runway_last[best_rwy] = (best_min, flight.wake_class.value)

            if deadline is not None:
                atfm_compliance[flight.flight_id] = deadline

            msg_type = MessageType.EMERGENCY_BCAST if is_emg else MessageType.RUNWAY_CLAIM
            if is_emg:
                emergency_broadcasts.append(flight.flight_id)
            messages.append(NegotiationMessage(
                from_role=AgentRole.DMAN,
                message_type=msg_type,
                flight_id=flight.flight_id,
                requested_minute=best_min,
                runway_id=best_rwy,
                priority=flight.priority,
                reason=(
                    f"Emergency departure — yield runway {best_rwy} at T+{best_min}"
                    if is_emg else "Departure slot claim"
                ),
                is_emergency=is_emg,
            ))

    return DMANAction(
        departure_slots=slots,
        rationale=f"Heuristic DMAN: {len(slots)}/{len(departures)} departures sequenced, "
                  f"ATFM-compliant, emergency-priority.",
        atfm_compliance=atfm_compliance,
        emergency_broadcasts=emergency_broadcasts,
        outgoing_messages=messages,
        commit=False,
    )


# ── LLM agent ─────────────────────────────────────────────────────────────────

def _llm_action(
    client,
    model_name: str,
    system: str,
    obs: MultiAgentObservation,
    sup_desc: str,
    role: AgentRole,
) -> Optional[AMANAction | DMANAction]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system + f"\n\nSUPERVISOR TODAY: {sup_desc}"},
                {"role": "user",   "content": obs.to_prompt_text()},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        if role == AgentRole.AMAN:
            return parse_aman_action(text)
        return parse_dman_action(text)
    except Exception as exc:
        _p(f"[WARN] LLM call failed ({role.value}): {exc} — falling back to heuristic")
        return None


# ── Episode runner ─────────────────────────────────────────────────────────────

def _save_transcript(
    transcript_dir: Path,
    episode_id: int,
    data: Dict[str, Any],
) -> None:
    """Write one episode transcript to <transcript_dir>/episode_NNNNN.json."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"episode_{episode_id:05d}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except OSError as exc:
        _p(f"[WARN] Could not save transcript for episode {episode_id}: {exc}")


_PROFILE_LIST = list(SupervisorProfileName)


def run_episode(
    task_id: str,
    client,
    env: Optional[MultiAgentATCEnvironment] = None,
    curriculum: Optional[ContextAdaptiveCurriculum] = None,
    episode_id: int = 0,
    use_curriculum: bool = False,
    model_name: Optional[str] = None,
    transcript_dir: Optional[Path] = None,
) -> Dict:
    """Run one full AMAN/DMAN episode. Returns result dict for logging.

    Minimal call:
        result = run_episode("delhi_monsoon_recovery_easy", client=None)
    """
    if env is None:
        env = MultiAgentATCEnvironment(seed=42)
    catalog  = task_catalog()
    base_task = catalog.get(task_id, list(catalog.values())[0])
    profile   = _PROFILE_LIST[episode_id % len(_PROFILE_LIST)]
    sup_desc  = SUPERVISOR_PROFILES[profile]["description"]

    mutations_applied: List[str] = []
    if use_curriculum and curriculum is not None:
        mutated_task, solvable = curriculum.mutate(base_task)
        gen_difficulty = curriculum.difficulty_level
        mutations_applied = [
            m["type"]
            for m in getattr(curriculum, "_mutation_history", [])[-3:]
        ]

    else:
        mutated_task, solvable = base_task, True
        gen_difficulty = 1

    # Enable domain randomisation when the generator is inactive — replaces the
    # structural diversity that generator mutations would otherwise provide.
    aman_obs, dman_obs = env.reset(
        episode_id=episode_id,
        supervisor_profile=profile,
        mutated_task=mutated_task,
        randomize=not use_curriculum,
    )
    selected_model = model_name or MODEL_NAME
    atfm = env._state.atfm_deadlines

    # Accumulate round data for the transcript
    rounds: List[Dict[str, Any]] = []

    # ── Round 0: BID ──────────────────────────────────────────────────────────
    aman_action = (
        _llm_action(client, selected_model, AMAN_SYSTEM, aman_obs, sup_desc, AgentRole.AMAN)
        or _build_aman_heuristic(aman_obs)
    )
    dman_action = (
        _llm_action(client, selected_model, DMAN_SYSTEM, dman_obs, sup_desc, AgentRole.DMAN)
        or _build_dman_heuristic(dman_obs, atfm)
    )

    aman_obs2, dman_obs2, partial_r, done = env.step_bid(aman_action, dman_action)
    n_conflicts_before = len(env._state.conflict_log)
    log_step(1, "BID", partial_r, n_conflicts_before, done)

    rounds.append({
        "round": "BID",
        "partial_reward": partial_r,
        "conflicts_detected": n_conflicts_before,
        "aman_slots": len(aman_action.arrival_slots),
        "dman_slots": len(dman_action.departure_slots),
        "aman_messages": len(aman_action.outgoing_messages),
        "dman_messages": len(dman_action.outgoing_messages),
        "done": done,
    })

    # ── Round 1: NEGOTIATE (if conflicts) ─────────────────────────────────────
    if not done:
        aman_action2 = (
            _llm_action(client, selected_model, AMAN_SYSTEM, aman_obs2, sup_desc, AgentRole.AMAN)
            or _build_aman_heuristic(aman_obs2)
        )
        dman_action2 = (
            _llm_action(client, selected_model, DMAN_SYSTEM, dman_obs2, sup_desc, AgentRole.DMAN)
            or _build_dman_heuristic(dman_obs2, atfm)
        )
        aman_obs3, dman_obs3, partial_r2, _ = env.step_negotiate(aman_action2, dman_action2)
        n_conflicts_after = len(env._state.conflict_log)
        resolved = max(0, n_conflicts_before - n_conflicts_after)
        total_msgs = len(aman_action2.outgoing_messages) + len(dman_action2.outgoing_messages)
        log_neg(1, resolved, total_msgs)
        log_step(2, "NEGOTIATE", partial_r2, n_conflicts_after, True)

        rounds.append({
            "round": "NEGOTIATE",
            "partial_reward": partial_r2,
            "conflicts_before": n_conflicts_before,
            "conflicts_after": n_conflicts_after,
            "conflicts_resolved": resolved,
            "total_messages": total_msgs,
            "aman_alternatives_offered": sum(
                len(getattr(m, "proposed_alternatives", [])) for m in aman_action2.outgoing_messages
            ),
            "dman_alternatives_offered": sum(
                len(getattr(m, "proposed_alternatives", [])) for m in dman_action2.outgoing_messages
            ),
        })

    # ── Finalize ──────────────────────────────────────────────────────────────
    result = env.finalize()
    if curriculum is not None:
        curriculum.update(result.composite_score)
        gen_difficulty = curriculum.difficulty_level

    episode_result = {
        "composite":      result.composite_score,
        "aman_reward":    result.aman_reward,
        "dman_reward":    result.dman_reward,
        "coord_score":    result.per_role.coordination_score,
        "conflicts":      result.per_role.cross_lane_conflicts,
        "atfm_viol":      result.per_role.atfm_violations,
        "emg_arr_ok":     result.per_role.emergency_arrivals_ok,
        "emg_dep_ok":     result.per_role.emergency_departures_ok,
        "neg_rounds":     result.negotiation_rounds,
        "gen_difficulty": gen_difficulty,
        "supervisor":     profile.value,
        "solvable":       solvable,
    }

    if transcript_dir is not None:
        _save_transcript(
            transcript_dir,
            episode_id,
            {
                "episode_id":        episode_id,
                "task_id":           task_id,
                "supervisor_profile": profile.value,
                "gen_difficulty":    gen_difficulty,
                "mutations_applied": mutations_applied,
                "solvable":          solvable,
                "rounds":            rounds,
                "final": {
                    "composite_score":    result.composite_score,
                    "aman_reward":        result.aman_reward,
                    "dman_reward":        result.dman_reward,
                    "coordination_score": result.per_role.coordination_score,
                    "cross_lane_conflicts": result.per_role.cross_lane_conflicts,
                    "negotiation_rounds": result.negotiation_rounds,
                    "atfm_violations":    result.per_role.atfm_violations,
                    "emergency_arrivals_ok":   result.per_role.emergency_arrivals_ok,
                    "emergency_departures_ok": result.per_role.emergency_departures_ok,
                },
            },
        )

    return episode_result

def run_domain_episode(
    domain_task_id: str,
    client,
    env: MultiAgentATCEnvironment,
    episode_id: int,
    model_name: Optional[str] = None,
    transcript_dir: Optional[Path] = None,
) -> Dict:
    """Run one ADAPT domain-transfer episode.

    ADAPT maps the domain task → ATC-parameterised task.
    AMAN and DMAN then solve it with no code changes.
    """
    from domains import get_all_domain_tasks, get_domain_description
    from multi_agent.adapt import (
        build_adapt_observation,
        _build_adapt_heuristic,
        apply_adapt_mapping,
        parse_adapt_action,
    )
    from training.dataset import ADAPT_SYSTEM

    all_domain_tasks = get_all_domain_tasks()
    if domain_task_id not in all_domain_tasks:
        available = ", ".join(sorted(all_domain_tasks.keys()))
        raise ValueError(
            f"Domain task {domain_task_id!r} not found. Available: {available}"
        )

    domain_task = all_domain_tasks[domain_task_id]
    profile     = _PROFILE_LIST[episode_id % len(_PROFILE_LIST)]
    sup_desc    = SUPERVISOR_PROFILES[profile]["description"]

    # ── ADAPT step ────────────────────────────────────────────────────────────
    adapt_obs = build_adapt_observation(task=domain_task, profile=profile)

    adapt_action: Optional[object] = None
    adapt_rationale = "(heuristic)"

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=model_name or MODEL_NAME,
                messages=[
                    {"role": "system", "content": ADAPT_SYSTEM},
                    {"role": "user",   "content": adapt_obs.to_prompt_text()},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = (resp.choices[0].message.content or "").strip()
            adapt_action   = parse_adapt_action(text)
            adapt_rationale = (adapt_action.rationale if adapt_action else "(LLM parse failed)")
        except Exception as exc:
            _p(f"[WARN] ADAPT LLM call failed: {exc} — falling back to structural heuristic")

    if adapt_action is None:
        adapt_action   = _build_adapt_heuristic(adapt_obs, domain_task)
        adapt_rationale = adapt_action.rationale

    _p(f"[ADAPT] domain={domain_task_id} entities={adapt_obs.entity_types}")
    _p(f"[ADAPT] wake_map={adapt_action.entity_wake_map}")
    _p(f"[ADAPT] priority_map={adapt_action.entity_priority_map}")
    _p(f"[ADAPT] rationale={adapt_rationale[:120].encode('ascii', errors='replace').decode()}")


    # ── Apply ADAPT mapping and run AMAN/DMAN ────────────────────────────────
    mapped_task = apply_adapt_mapping(domain_task, adapt_action)

    aman_obs, dman_obs = env.reset(
        episode_id=episode_id,
        supervisor_profile=profile,
        mutated_task=mapped_task,
    )
    atfm = env._state.atfm_deadlines

    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, atfm)
    if client is not None:
        aman_action = (
            _llm_action(client, model_name or MODEL_NAME, AMAN_SYSTEM, aman_obs, sup_desc, AgentRole.AMAN)
            or aman_action
        )
        dman_action = (
            _llm_action(client, model_name or MODEL_NAME, DMAN_SYSTEM, dman_obs, sup_desc, AgentRole.DMAN)
            or dman_action
        )

    aman_obs2, dman_obs2, partial_r, done = env.step_bid(aman_action, dman_action)
    log_step(1, "BID", partial_r, len(env._state.conflict_log), done)

    if not done:
        aman_action2 = _build_aman_heuristic(aman_obs2)
        dman_action2 = _build_dman_heuristic(dman_obs2, atfm)
        _, _, partial_r2, _ = env.step_negotiate(aman_action2, dman_action2)
        resolved = max(0, len(env._state.conflict_log))
        log_neg(1, resolved, 0)
        log_step(2, "NEGOTIATE", partial_r2, 0, True)

    result = env.finalize()

    episode_result = {
        "domain_task_id":  domain_task_id,
        "composite":       result.composite_score,
        "aman_reward":     result.aman_reward,
        "dman_reward":     result.dman_reward,
        "coord_score":     result.per_role.coordination_score,
        "conflicts":       result.per_role.cross_lane_conflicts,
        "supervisor":      profile.value,
        "adapt_wake_map":  adapt_action.entity_wake_map,
        "adapt_priority_map": adapt_action.entity_priority_map,
    }

    _p(
        f"[END]   domain={domain_task_id} success={str(result.composite_score >= SUCCESS_THRESHOLD).lower()} "
        f"composite={result.composite_score:.3f} aman={result.aman_reward:.3f} dman={result.dman_reward:.3f}"
    )
    return episode_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent ATC Inference Runner")
    parser.add_argument("--task",     default="bengaluru_irrops_hard",
                        help="Task ID (default: bengaluru_irrops_hard)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--model",    default=MODEL_NAME,
                        help="Model name or 'heuristic-baseline'")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable self-adapting curriculum (use base tasks)")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Run all ATC tasks in sequence")
    parser.add_argument("--domain", default=None,
                        help=(
                            "Run ADAPT domain-transfer mode on a domain task ID "
                            "(e.g. icu_mass_casualty, icu_flu_surge, icu_normal_day). "
                            "Use --domain list to show all available domain tasks."
                        ))
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--transcript_dir", default=None,
                        help="Directory to write per-episode JSON transcripts "
                             "(default: no transcripts saved)")
    args = parser.parse_args()

    model = args.model or MODEL_NAME
    use_llm = model != "heuristic-baseline" and HF_TOKEN

    client = None
    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            _p(f"[INFO] LLM client initialised: {model}")
        except ImportError:
            _p("[WARN] openai package missing — using heuristic baseline")

    env       = MultiAgentATCEnvironment(seed=args.seed)
    curriculum = ContextAdaptiveCurriculum(seed=args.seed)

    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None

    # ── ADAPT domain-transfer mode ────────────────────────────────────────────
    if args.domain is not None:
        if args.domain == "list":
            from domains import get_all_domain_tasks
            all_dt = get_all_domain_tasks()
            _p("Available domain tasks:")
            for tid in sorted(all_dt.keys()):
                t = all_dt[tid]
                _p(f"  {tid:<35} ({t.difficulty.value}, {len(t.flights)} entities, {len(t.runways)} resources)")
            return

        _p(f"[INFO] ADAPT domain-transfer mode: {args.domain}")
        all_results: List[Dict] = []
        for ep in range(args.episodes):
            result = run_domain_episode(
                domain_task_id=args.domain,
                client=client,
                env=env,
                episode_id=ep,
                model_name=model,
                transcript_dir=transcript_dir,
            )
            all_results.append(result)

        if len(all_results) > 1:
            _p("\n=== DOMAIN TRANSFER SUMMARY ===")
            _p(f"Domain:          {args.domain}")
            _p(f"Episodes:        {len(all_results)}")
            _p(f"Mean composite:  {sum(r['composite'] for r in all_results)/len(all_results):.3f}")
            _p(f"Success rate:    {sum(1 for r in all_results if r['composite'] >= SUCCESS_THRESHOLD)/len(all_results):.1%}")
        return

    # ── Standard ATC mode ─────────────────────────────────────────────────────
    task_ids = (
        [t.task_id for t in ordered_tasks()]
        if args.all_tasks
        else [args.task]
    )

    all_results: List[Dict] = []

    for task_id in task_ids:
        log_start(task_id, model)
        for ep in range(args.episodes):
            result = run_episode(
                task_id=task_id,
                client=client,
                model_name=model,
                env=env,
                curriculum=curriculum,
                episode_id=ep,
                use_curriculum=not args.no_curriculum,
                transcript_dir=transcript_dir,
            )
            success = result["composite"] >= SUCCESS_THRESHOLD
            log_end(
                task=task_id,
                success=success,
                composite=result["composite"],
                aman=result["aman_reward"],
                dman=result["dman_reward"],
                coord=result["coord_score"],
                gen_diff=result["gen_difficulty"],
            )
            all_results.append({"task_id": task_id, "episode": ep, **result})

    # Summary
    if len(all_results) > 1:
        _p("\n=== MULTI-EPISODE SUMMARY ===")
        _p(f"Episodes:          {len(all_results)}")
        _p(f"Mean composite:    {sum(r['composite'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean AMAN reward:  {sum(r['aman_reward'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean DMAN reward:  {sum(r['dman_reward'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean coord score:  {sum(r['coord_score'] for r in all_results)/len(all_results):.3f}")
        _p(f"Total conflicts:   {sum(r['conflicts'] for r in all_results)}")
        _p(f"Success rate:      {sum(1 for r in all_results if r['composite'] >= SUCCESS_THRESHOLD)/len(all_results):.1%}")
        _p(f"Final gen level:   {all_results[-1]['gen_difficulty']}/6")


if __name__ == "__main__":
    main()

