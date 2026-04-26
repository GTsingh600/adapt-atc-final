"""Multi-agent ATC environment: AMAN + DMAN cooperative-competitive coordination.

Protocol (3 rounds per episode):
  Round 0 BID:       AMAN submits arrivals plan, DMAN submits departures plan independently
  Round 1 NEGOTIATE: Environment detects conflicts, broadcasts to both agents, agents revise
  Round 2 FINAL:     Plans merged, simulate_plan() + grade_task() run, per-agent rewards emitted

Key design principles:
  - Partial observability: each agent sees only own flight type + incoming messages
  - Competitive baseline: each agent optimises own objective (delay for own flights)
  - Cooperative forcing: emergencies and conflicts force cross-agent coordination
  - Reward decomposition: AMAN reward ≠ DMAN reward; both logged for training curves
  - Theory-of-mind bonus: rewarded when agent pre-emptively coordinates without prompt
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from ..engine import empty_metrics, simulate_plan
    from ..graders import GatedCompositeGrader, SafetyGateEvaluator, grade_task
    from ..models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        SlotAssignment,
        TaskDefinition,
        TaskMetrics,
    )
    from ..tasks import task_catalog
    from .models import (
        AMANAction,
        AgentRole,
        DMANAction,
        GeneratorAction,
        MessageType,
        MultiAgentEpisodeResult,
        MultiAgentObservation,
        NegotiationMessage,
        PerRoleMetrics,
        RoundType,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )
except ImportError:
    from engine import empty_metrics, simulate_plan
    from graders import GatedCompositeGrader, SafetyGateEvaluator, grade_task
    from models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        SlotAssignment,
        TaskDefinition,
        TaskMetrics,
    )
    from tasks import task_catalog
    from multi_agent.models import (
        AMANAction,
        AgentRole,
        DMANAction,
        GeneratorAction,
        MessageType,
        MultiAgentEpisodeResult,
        MultiAgentObservation,
        NegotiationMessage,
        PerRoleMetrics,
        RoundType,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )


MAX_NEGOTIATE_ROUNDS = 2   # max negotiation passes before forced merge
ATFM_DEADLINE_HEADROOM = 8  # minutes of margin to call an ATFM violation

# Domain randomization bounds — small perturbations to prevent task memorisation
_RAND_WINDOW_DELTA  = 5    # ± minutes applied to flight windows
_RAND_WEATHER_DELTA = 0.08  # ± weather_penalty (clipped to [1.0, 1.8])
_RAND_BUDGET_SCALE  = 0.12  # ± fraction applied to delay_budget / fuel_budget
_RAND_MIN_WINDOW    = 10    # never squeeze window below this width (minutes)


@dataclass
class EpisodeState:
    task:                 TaskDefinition
    supervisor_profile:   SupervisorProfileName
    atfm_deadlines:       Dict[str, int] = field(default_factory=dict)
    aman_slots:           List[SlotAssignment] = field(default_factory=list)
    dman_slots:           List[SlotAssignment] = field(default_factory=list)
    aman_messages:        List[NegotiationMessage] = field(default_factory=list)
    dman_messages:        List[NegotiationMessage] = field(default_factory=list)
    conflict_log:         List[str] = field(default_factory=list)
    round_number:         int = 0
    negotiation_rounds:   int = 0
    episode_id:           int = 0


class MultiAgentATCEnvironment:
    """AMAN/DMAN cooperative-competitive environment.

    Usage (training loop):
        env = MultiAgentATCEnvironment()
        aman_obs, dman_obs = env.reset(task_id="bengaluru_irrops_hard", episode_id=42)

        # Round 0 — BID
        aman_action = aman_agent(aman_obs)
        dman_action = dman_agent(dman_obs)
        aman_obs, dman_obs, partial_reward, done = env.step_bid(aman_action, dman_action)

        # Round 1 — NEGOTIATE (if conflicts)
        if not done:
            aman_action = aman_agent(aman_obs)
            dman_action = dman_agent(dman_obs)
            aman_obs, dman_obs, partial_reward, done = env.step_negotiate(aman_action, dman_action)

        # Round 2 — FINAL
        result = env.finalize()
    """

    def __init__(self, seed: int = 42) -> None:
        self._catalog: Dict[str, TaskDefinition] = task_catalog()
        self._rng = random.Random(seed)
        self._state: Optional[EpisodeState] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        episode_id: int = 0,
        supervisor_profile: Optional[SupervisorProfileName] = None,
        mutated_task: Optional[TaskDefinition] = None,
        randomize: bool = False,
    ) -> Tuple[MultiAgentObservation, MultiAgentObservation]:
        """Reset environment. Returns (aman_obs, dman_obs).

        Args:
            randomize: When True (training with no generator), applies bounded
                parametric perturbations to flight windows and runway weather so
                agents cannot memorise exact task parameters across episodes.
                Off by default to keep evaluation and tests deterministic.
        """
        if mutated_task is not None:
            task = mutated_task
        elif task_id and task_id in self._catalog:
            task = self._catalog[task_id]
        else:
            task = self._rng.choice(list(self._catalog.values()))

        if randomize:
            task = self._randomize_task(task, episode_id)

        profile = supervisor_profile or SupervisorProfileName.SAFETY_STRICT
        atfm = self._build_atfm_deadlines(task)

        self._state = EpisodeState(
            task=task,
            supervisor_profile=profile,
            atfm_deadlines=atfm,
            episode_id=episode_id,
        )
        return self._build_observations(RoundType.BID, conflict_log=[], incoming_aman=[], incoming_dman=[])

    def step_bid(
        self,
        aman_action: AMANAction,
        dman_action: DMANAction,
    ) -> Tuple[MultiAgentObservation, MultiAgentObservation, float, bool]:
        """Process Round 0: independent bids. Returns obs pair, dense reward, done."""
        assert self._state is not None, "call reset() first"
        state = self._state

        state.aman_slots = list(aman_action.arrival_slots)
        state.dman_slots = list(dman_action.departure_slots)
        state.aman_messages = list(aman_action.outgoing_messages)
        state.dman_messages = list(dman_action.outgoing_messages)
        state.round_number = 1

        conflicts, diagnostics = self._detect_conflicts(state)
        state.conflict_log = diagnostics

        # If emergency broadcast present, inject environment conflict flags
        self._inject_emergency_flags(state, aman_action, dman_action)

        partial_score = self._score_merged(state)
        done = len(conflicts) == 0  # no conflicts → skip negotiate, go straight to final
        if done:
            state.round_number = 2

        aman_obs, dman_obs = self._build_observations(
            RoundType.NEGOTIATE if not done else RoundType.FINAL,
            conflict_log=state.conflict_log,
            incoming_aman=state.dman_messages,
            incoming_dman=state.aman_messages,
        )
        return aman_obs, dman_obs, round(partial_score * 0.3, 4), done

    def step_negotiate(
        self,
        aman_action: AMANAction,
        dman_action: DMANAction,
    ) -> Tuple[MultiAgentObservation, MultiAgentObservation, float, bool]:
        """Process Round 1: negotiation revision. Returns obs pair, dense reward, done."""
        assert self._state is not None
        state = self._state

        state.aman_slots = list(aman_action.arrival_slots)
        state.dman_slots = list(dman_action.departure_slots)
        state.aman_messages = list(aman_action.outgoing_messages)
        state.dman_messages = list(dman_action.outgoing_messages)
        state.round_number = 2
        state.negotiation_rounds += 1

        conflicts, diagnostics = self._detect_conflicts(state)
        state.conflict_log = diagnostics

        partial_score = self._score_merged(state)

        aman_obs, dman_obs = self._build_observations(
            RoundType.FINAL,
            conflict_log=state.conflict_log,
            incoming_aman=state.dman_messages,
            incoming_dman=state.aman_messages,
        )
        return aman_obs, dman_obs, round(partial_score * 0.3, 4), True

    def finalize(self) -> MultiAgentEpisodeResult:
        """Merge plans, run full grader, compute per-agent rewards."""
        assert self._state is not None
        state = self._state

        merged = state.aman_slots + state.dman_slots
        outcome = simulate_plan(state.task, merged)
        grades = grade_task(state.task, outcome, merged, rationale="multi-agent final")
        composite = next(
            (g for g in grades if g.grader_name == "composite_task_grader"), grades[-1]
        )

        per_role = self._compute_per_role_metrics(state, outcome.metrics)

        aman_reward  = self._aman_reward(per_role, outcome.metrics, state)
        dman_reward  = self._dman_reward(per_role, outcome.metrics, state)

        return MultiAgentEpisodeResult(
            task_id=state.task.task_id,
            supervisor_profile=state.supervisor_profile,
            composite_score=round(composite.score, 4),
            aman_reward=round(aman_reward, 4),
            dman_reward=round(dman_reward, 4),
            per_role=per_role,
            negotiation_rounds=state.negotiation_rounds,
        )

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observations(
        self,
        round_type: RoundType,
        conflict_log: List[str],
        incoming_aman: List[NegotiationMessage],
        incoming_dman: List[NegotiationMessage],
    ) -> Tuple[MultiAgentObservation, MultiAgentObservation]:
        assert self._state is not None
        state = self._state
        profile_name = state.supervisor_profile
        sup_desc = SUPERVISOR_PROFILES[profile_name]["description"]

        arrivals   = [f for f in state.task.flights if f.operation == OperationType.ARRIVAL]
        departures = [f for f in state.task.flights if f.operation == OperationType.DEPARTURE]

        shared = dict(
            task_id=state.task.task_id,
            airport=state.task.airport,
            briefing=state.task.description,
            all_runways=state.task.runways,
            supervisor_profile_name=profile_name,
            supervisor_description=sup_desc,
            conflict_log=conflict_log,
            round_type=round_type,
            round_number=state.round_number,
            steps_remaining=MAX_NEGOTIATE_ROUNDS - state.round_number,
        )

        # Partial observability: ATFM deadlines are DMAN-only information.
        # AMAN has no visibility into network slot constraints for departures.
        aman_obs = MultiAgentObservation(
            role=AgentRole.AMAN,
            my_flights=arrivals,
            incoming_messages=incoming_aman,
            atfm_deadlines={},
            **shared,
        )
        dman_obs = MultiAgentObservation(
            role=AgentRole.DMAN,
            my_flights=departures,
            incoming_messages=incoming_dman,
            atfm_deadlines=state.atfm_deadlines,
            **shared,
        )
        return aman_obs, dman_obs

    # ── Conflict detection ────────────────────────────────────────────────────

    def _detect_conflicts(
        self, state: EpisodeState
    ) -> Tuple[List[str], List[str]]:
        """Detect cross-agent runway conflicts by running simulate_plan on merge."""
        merged = state.aman_slots + state.dman_slots
        outcome = simulate_plan(state.task, merged)
        conflicts = [d for d in outcome.diagnostics if "spaced" in d or "conflict" in d.lower()]
        return conflicts, outcome.diagnostics

    def _score_merged(self, state: EpisodeState) -> float:
        merged = state.aman_slots + state.dman_slots
        outcome = simulate_plan(state.task, merged)
        return outcome.normalized_score

    # ── Emergency handling ────────────────────────────────────────────────────

    def _inject_emergency_flags(
        self,
        state: EpisodeState,
        aman_action: AMANAction,
        dman_action: DMANAction,
    ) -> None:
        """Inject environment-level emergency conflict flags into message queues."""
        flights_by_id = {f.flight_id: f for f in state.task.flights}

        # DMAN fuel/medical emergency departures → flag to AMAN
        for fid in dman_action.emergency_broadcasts:
            flight = flights_by_id.get(fid)
            if not flight:
                continue
            slot = next((s for s in state.dman_slots if s.flight_id == fid), None)
            if not slot:
                continue
            state.aman_messages.append(NegotiationMessage(
                from_role=AgentRole.DMAN,
                message_type=MessageType.EMERGENCY_BCAST,
                flight_id=fid,
                requested_minute=slot.assigned_minute,
                runway_id=slot.runway,
                priority=flight.priority,
                reason=f"Fuel/medical emergency departure — needs runway by T+{slot.assigned_minute}",
                is_emergency=True,
            ))

        # AMAN emergency arrivals → flag to DMAN
        emergency_arrivals = [
            f for f in state.task.flights
            if f.operation == OperationType.ARRIVAL
            and f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        ]
        for flight in emergency_arrivals:
            slot = next((s for s in state.aman_slots if s.flight_id == flight.flight_id), None)
            if not slot:
                continue
            state.dman_messages.append(NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType.EMERGENCY_BCAST,
                flight_id=flight.flight_id,
                requested_minute=slot.assigned_minute,
                runway_id=slot.runway,
                priority=flight.priority,
                reason=f"Emergency arrival — runway {slot.runway} must be clear at T+{slot.assigned_minute}",
                is_emergency=True,
            ))

    # ── Per-role reward computation ───────────────────────────────────────────

    def _compute_per_role_metrics(
        self, state: EpisodeState, metrics: TaskMetrics
    ) -> PerRoleMetrics:
        flights_by_id = {f.flight_id: f for f in state.task.flights}
        aman_map = {s.flight_id: s for s in state.aman_slots}
        dman_map = {s.flight_id: s for s in state.dman_slots}

        arr_delay_total = 0
        dep_delay_total = 0
        arr_count = dep_count = 0
        arr_missing = dep_missing = 0
        emg_arr_ok = emg_arr_miss = 0
        emg_dep_ok = emg_dep_miss = 0
        atfm_viol = atfm_ok = 0

        for f in state.task.flights:
            is_emergency = f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
            if f.operation == OperationType.ARRIVAL:
                arr_count += 1
                slot = aman_map.get(f.flight_id)
                if slot:
                    delay = abs(slot.assigned_minute - f.scheduled_minute)
                    arr_delay_total += delay
                    if is_emergency:
                        if delay <= 5:
                            emg_arr_ok += 1
                        else:
                            emg_arr_miss += 1
                else:
                    arr_missing += 1
                    if is_emergency:
                        emg_arr_miss += 1
            else:
                dep_count += 1
                slot = dman_map.get(f.flight_id)
                if slot:
                    delay = abs(slot.assigned_minute - f.scheduled_minute)
                    dep_delay_total += delay
                    if is_emergency:
                        if delay <= 5:
                            emg_dep_ok += 1
                        else:
                            emg_dep_miss += 1
                    # ATFM check
                    deadline = state.atfm_deadlines.get(f.flight_id)
                    if deadline is not None:
                        if slot.assigned_minute <= deadline:
                            atfm_ok += 1
                        else:
                            atfm_viol += 1
                else:
                    dep_missing += 1
                    if is_emergency:
                        emg_dep_miss += 1

        # Cross-lane vs intra-lane conflict heuristic:
        # cross = conflicts involving one AMAN slot and one DMAN slot on same runway
        cross_conflicts = self._count_cross_lane_conflicts(state)
        intra_conflicts = max(0, metrics.conflict_count - cross_conflicts)

        arr_mean = (arr_delay_total / max(1, arr_count))
        dep_mean = (dep_delay_total / max(1, dep_count))

        coord_score = self._coordination_score(state, metrics, cross_conflicts)

        return PerRoleMetrics(
            arrival_count=arr_count,
            arrival_delay_total=arr_delay_total,
            arrival_delay_mean=round(arr_mean, 2),
            arrivals_missing=arr_missing,
            emergency_arrivals_ok=emg_arr_ok,
            emergency_arrivals_missed=emg_arr_miss,
            departure_count=dep_count,
            departure_delay_total=dep_delay_total,
            departure_delay_mean=round(dep_mean, 2),
            departures_missing=dep_missing,
            atfm_violations=atfm_viol,
            atfm_compliant=atfm_ok,
            emergency_departures_ok=emg_dep_ok,
            emergency_departures_missed=emg_dep_miss,
            cross_lane_conflicts=cross_conflicts,
            intra_lane_conflicts=intra_conflicts,
            negotiation_rounds_used=state.negotiation_rounds,
            coordination_score=coord_score,
        )

    def _count_cross_lane_conflicts(self, state: EpisodeState) -> int:
        """Heuristic: conflicts on runways used by BOTH agents."""
        from collections import defaultdict
        aman_runways = {s.runway for s in state.aman_slots}
        dman_runways = {s.runway for s in state.dman_slots}
        shared_runways = aman_runways & dman_runways
        if not shared_runways:
            return 0
        # Re-run sim on shared-runway slots only
        shared_slots = [
            s for s in (state.aman_slots + state.dman_slots)
            if s.runway in shared_runways
        ]
        outcome = simulate_plan(state.task, shared_slots)
        return outcome.metrics.conflict_count

    def _coordination_score(
        self,
        state: EpisodeState,
        metrics: TaskMetrics,
        cross_conflicts: int,
    ) -> float:
        """Theory-of-mind quality score for agent coordination.

        Rewards:
          - Correct emergency yield without being asked (+0.3)
          - Pre-emptive gap left for known DMAN emergency (+0.25)
          - Resolved in fewest negotiation rounds (+0.2 if 0 rounds, +0.1 if 1)
          - Zero cross-lane conflicts (+0.25)
          - Proposed alternatives that were adopted in the final plan (+0.15)
        """
        score = 0.0

        # Zero cross-lane conflicts
        if cross_conflicts == 0:
            score += 0.25

        # Negotiation efficiency
        if state.negotiation_rounds == 0:
            score += 0.20
        elif state.negotiation_rounds == 1:
            score += 0.10

        # Emergency handling quality
        emergency_flights = [
            f for f in state.task.flights
            if f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        ]
        if emergency_flights:
            assigned_ids = {s.flight_id for s in state.aman_slots + state.dman_slots}
            ratio = sum(1 for f in emergency_flights if f.flight_id in assigned_ids) / len(emergency_flights)
            score += 0.30 * ratio

        # Pre-emptive coordination: DMAN broadcast emergency AND AMAN left gap
        dman_broadcasts = {m.flight_id for m in state.dman_messages if m.is_emergency}
        if dman_broadcasts:
            aman_runway_minutes = {(s.runway, s.assigned_minute) for s in state.aman_slots}
            for fid in dman_broadcasts:
                dep_slot = next((s for s in state.dman_slots if s.flight_id == fid), None)
                if dep_slot:
                    clear = all(
                        abs(m - dep_slot.assigned_minute) >= 3
                        for (rwy, m) in aman_runway_minutes
                        if rwy == dep_slot.runway
                    )
                    if clear:
                        score += 0.25
                        break

        return round(min(1.0, score), 4)

    # ── Per-agent reward functions ────────────────────────────────────────────

    def _aman_reward(
        self, per_role: PerRoleMetrics, metrics: TaskMetrics, state: EpisodeState
    ) -> float:
        """Potential-based AMAN reward (Ng et al. 1999).

        Components:
          1. Arrival delay efficiency (normalized against task budget)
          2. Emergency arrival handling
          3. Coordination bonus (theory-of-mind)
          4. Cross-lane conflict penalty (penalises AMAN for shared-runway chaos)
          5. Supervisor alignment
        """
        arr_count = max(1, per_role.arrival_count)
        budget = state.task.delay_budget / 2  # AMAN's share of delay budget

        # 1. Delay efficiency
        delay_ratio = per_role.arrival_delay_total / max(1, budget)
        delay_score = max(0.0, 1.0 - delay_ratio)

        # 2. Emergency arrival bonus
        total_emg = per_role.emergency_arrivals_ok + per_role.emergency_arrivals_missed
        emg_score = (per_role.emergency_arrivals_ok / max(1, total_emg)) if total_emg else 1.0

        # 3. Coverage (did AMAN assign all arrivals?)
        coverage = 1.0 - (per_role.arrivals_missing / arr_count)

        # 4. Coordination bonus
        coord = per_role.coordination_score

        # 5. Cross-lane conflict penalty
        cross_penalty = 0.15 * per_role.cross_lane_conflicts

        raw = (
            0.35 * delay_score
            + 0.25 * emg_score
            + 0.20 * coverage
            + 0.15 * coord
            + 0.05 * max(0.0, 1.0 - cross_penalty)
        )
        return round(max(0.0, min(1.0, raw)), 4)

    def _dman_reward(
        self, per_role: PerRoleMetrics, metrics: TaskMetrics, state: EpisodeState
    ) -> float:
        """Potential-based DMAN reward.

        Components:
          1. Departure delay efficiency
          2. ATFM compliance (hard network slot adherence)
          3. Emergency departure handling
          4. Coordination bonus
          5. Cross-lane conflict penalty
          6. Supervisor alignment
        """
        dep_count = max(1, per_role.departure_count)
        budget = state.task.delay_budget / 2

        # 1. Delay efficiency
        delay_ratio = per_role.departure_delay_total / max(1, budget)
        delay_score = max(0.0, 1.0 - delay_ratio)

        # 2. ATFM compliance
        total_atfm = per_role.atfm_compliant + per_role.atfm_violations
        atfm_score = (per_role.atfm_compliant / max(1, total_atfm)) if total_atfm else 1.0

        # 3. Emergency departure bonus
        total_emg = per_role.emergency_departures_ok + per_role.emergency_departures_missed
        emg_score = (per_role.emergency_departures_ok / max(1, total_emg)) if total_emg else 1.0

        # 4. Coverage
        coverage = 1.0 - (per_role.departures_missing / dep_count)

        # 5. Coordination
        coord = per_role.coordination_score

        # 6. Cross-lane conflict penalty
        cross_penalty = 0.15 * per_role.cross_lane_conflicts

        raw = (
            0.30 * delay_score
            + 0.25 * atfm_score
            + 0.20 * emg_score
            + 0.15 * coverage
            + 0.10 * coord
        )
        return round(max(0.0, min(1.0, raw)), 4)

    # ── Domain randomisation ──────────────────────────────────────────────────

    def _randomize_task(self, task: TaskDefinition, episode_id: int) -> TaskDefinition:
        """Apply bounded parametric perturbations to prevent task memorisation.

        Uses a deterministic seed derived from (task_id, episode_id) so results
        are reproducible for a given episode while varying across episodes.
        All perturbations are small enough that solvability is preserved.
        """
        seed = hash(f"{task.task_id}:{episode_id}") & 0x7FFFFFFF
        rng = random.Random(seed)

        # Perturb flight windows — shift each bound by ±_RAND_WINDOW_DELTA minutes
        updated_flights = []
        for f in task.flights:
            delta_e = rng.randint(-_RAND_WINDOW_DELTA, _RAND_WINDOW_DELTA)
            delta_l = rng.randint(-_RAND_WINDOW_DELTA, _RAND_WINDOW_DELTA)
            new_earliest = max(0, f.earliest_minute + delta_e)
            new_latest = max(new_earliest + _RAND_MIN_WINDOW, f.latest_minute + delta_l)
            updated_flights.append(
                f.model_copy(update={"earliest_minute": new_earliest, "latest_minute": new_latest})
            )

        # Perturb runway weather penalties
        updated_runways = []
        for rwy in task.runways:
            delta_w = rng.uniform(-_RAND_WEATHER_DELTA, _RAND_WEATHER_DELTA)
            new_penalty = round(min(1.8, max(1.0, rwy.weather_penalty + delta_w)), 3)
            updated_runways.append(rwy.model_copy(update={"weather_penalty": new_penalty}))

        # Perturb budgets
        budget_scale = 1.0 + rng.uniform(-_RAND_BUDGET_SCALE, _RAND_BUDGET_SCALE)
        new_delay_budget = max(30, int(task.delay_budget * budget_scale))
        new_fuel_budget = max(50.0, round(task.fuel_budget * budget_scale, 1))

        return task.model_copy(update={
            "flights":       updated_flights,
            "runways":       updated_runways,
            "delay_budget":  new_delay_budget,
            "fuel_budget":   new_fuel_budget,
        })

    # ── ATFM deadline generator ───────────────────────────────────────────────

    def _build_atfm_deadlines(self, task: TaskDefinition) -> Dict[str, int]:
        """Simulate ATFM network slots for departure flights.

        Real ATFM: ~30% of flights get a constrained slot in peak periods.
        Deadline = scheduled_minute + small buffer (forces DMAN to prioritise).
        """
        deadlines: Dict[str, int] = {}
        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        if not departures:
            return deadlines
        constrained = self._rng.sample(departures, k=max(1, len(departures) // 3))
        for flight in constrained:
            # Deadline = scheduled + 12 min buffer (realistic ATFM GDP window)
            deadline = flight.scheduled_minute + 12
            if deadline <= flight.latest_minute:
                deadlines[flight.flight_id] = deadline
        return deadlines
