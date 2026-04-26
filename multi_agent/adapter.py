"""Context-Adaptive Curriculum for multi-agent ATC training.

Replaces adversarial self-play with *diagnostic self-adaptation*:

  Self-improving  → escalate difficulty when composite score rises
  Self-adapting   → identify the agent's weakest skill dimension and
                    generate scenarios that specifically stress that gap

The ContextAdaptiveCurriculum tracks a per-component skill profile across the
last N episodes and answers one question per episode:

    "Which dimension does this agent need the most practice on right now?"

It then selects mutations that directly exercise that dimension, and scales
the reward weights dynamically so the gradient signal is loudest on the
component the agent is failing at. This is self-adaptation, not escalation.

Key concepts:
  SkillProfile        — per-component rolling mean (conflict, delay, emergency,
                        atfm, coverage, coordination, fairness)
  AdaptiveMutationSet — mutation catalogue grouped by the skill each mutation
                        trains (e.g. INJECT_EMERGENCY trains emergency handling)
  DynamicRewardScaler — returns per-component weight adjustments based on the
                        current skill gap (weakest component → highest weight)
  AdaptationContext   — returned per episode, carries chosen mutations + weights
                        so the reward function can apply dynamic scaling

Long-Horizon Planning note:
  For extended planning windows (> 60 min), the curriculum automatically adds
  cascade mutations — scenarios where an early decision (minute 5) causes a
  conflict that only manifests at minute 45. This forces agents to reason about
  downstream effects, directly targeting Theme #2: Super Long-Horizon Planning.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

try:
    from ..engine import simulate_plan
    from ..models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from .models import GeneratorAction, GeneratorMutation, MutationType
except ImportError:
    from engine import simulate_plan
    from models import (
        FlightRecord,
        OperationType,
        PriorityClass,
        RunwaySpec,
        SlotAssignment,
        TaskDefinition,
        WakeClass,
    )
    from multi_agent.models import GeneratorAction, GeneratorMutation, MutationType


# ── Skill dimensions ──────────────────────────────────────────────────────────

SKILL_DIMENSIONS = [
    "conflict_avoidance",    # reduces wake-turbulence and cross-lane conflicts
    "delay_efficiency",      # minimizes total system delay
    "emergency_handling",    # on-time dispatch of EMERGENCY/MEDICAL flights
    "atfm_compliance",       # DMAN meets network slot deadlines
    "coverage",              # fraction of flights assigned valid slots
    "coordination",          # multi-agent negotiation quality
    "fairness",              # equitable delay distribution across airlines
]

# Mutations grouped by which skill dimension they train
_SKILL_MUTATION_MAP: Dict[str, List[MutationType]] = {
    "conflict_avoidance":  [MutationType.ADD_CONFLICTING_FLIGHT,
                            MutationType.CLOSE_RUNWAY_WINDOW],
    "delay_efficiency":    [MutationType.TIGHTEN_WINDOW,
                            MutationType.INCREASE_WEATHER_PENALTY],
    "emergency_handling":  [MutationType.INJECT_EMERGENCY],
    "atfm_compliance":     [MutationType.ADD_ATFM_DEADLINE],
    "coverage":            [MutationType.TIGHTEN_WINDOW,
                            MutationType.CLOSE_RUNWAY_WINDOW],
    "coordination":        [MutationType.ADD_CONFLICTING_FLIGHT,
                            MutationType.INJECT_EMERGENCY],
    "fairness":            [MutationType.INJECT_EMERGENCY,
                            MutationType.ADD_ATFM_DEADLINE],
}

MIN_WINDOW_WIDTH = 8
MAX_MUTATIONS_PER_EPISODE = 3
SKILL_WINDOW = 10            # rolling window for per-component skill estimates
ADAPTATION_FLOOR = 0.25      # weight floor: weakest component always ≥ this multiplier
ADAPTATION_CEILING = 2.50    # weight ceiling: strongest component never above this

# Priority alias normalization (same as original generator)
_PRIORITY_ALIASES: Dict[str, str] = {
    "high": "emergency", "urgent": "emergency", "critical": "emergency",
    "med": "medical", "low": "normal", "standard": "normal",
    "routine": "normal", "conn": "connection",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SkillProfile:
    """Rolling per-component skill estimates for one agent role."""
    histories: Dict[str, Deque[float]] = field(default_factory=dict)

    def update(self, component: str, score: float) -> None:
        if component not in self.histories:
            self.histories[component] = deque(maxlen=SKILL_WINDOW)
        self.histories[component].append(max(0.0, min(1.0, score)))

    def mean(self, component: str) -> float:
        h = self.histories.get(component)
        if not h:
            return 0.5  # unknown → assume mid-level competence
        return sum(h) / len(h)

    def weakest_dimension(self) -> str:
        """Return the dimension where the agent has the lowest rolling mean."""
        return min(SKILL_DIMENSIONS, key=self.mean)

    def gap_vector(self) -> Dict[str, float]:
        """Return how far each dimension is below 1.0 (gap = 1 - mean)."""
        return {d: round(1.0 - self.mean(d), 4) for d in SKILL_DIMENSIONS}

    def summary(self) -> Dict[str, float]:
        return {d: round(self.mean(d), 4) for d in SKILL_DIMENSIONS}


@dataclass
class AdaptationContext:
    """Per-episode adaptation decision: which skill to train and weight adjustments."""
    target_skill: str
    mutations_used: List[str]
    dynamic_weights: Dict[str, float]   # component → multiplier relative to baseline
    rationale: str


# ── Dynamic reward scaler ─────────────────────────────────────────────────────

def compute_dynamic_weights(skill_profile: SkillProfile) -> Dict[str, float]:
    """Convert skill gaps into reward weight multipliers.

    Weakest dimension → multiplier close to ADAPTATION_CEILING
    Strongest dimension → multiplier close to ADAPTATION_FLOOR

    Formula (softmax-like normalization over gaps):
        raw_i = exp(gap_i * 3)          # amplify gap differences
        w_i   = raw_i / mean(raw)       # normalize so mean multiplier = 1.0
        w_i   = clamp(w_i, FLOOR, CEILING)
    """
    gaps = skill_profile.gap_vector()
    raws = {d: math.exp(v * 3.0) for d, v in gaps.items()}
    mean_raw = max(1e-8, sum(raws.values()) / len(raws))
    weights = {}
    for d, raw in raws.items():
        w = raw / mean_raw
        weights[d] = round(max(ADAPTATION_FLOOR, min(ADAPTATION_CEILING, w)), 4)
    return weights


# ── Context-Adaptive Curriculum ───────────────────────────────────────────────

class ContextAdaptiveCurriculum:
    """Diagnostic self-adaptation: generates scenarios targeting agent weaknesses.

    Usage (training loop):
        curriculum = ContextAdaptiveCurriculum(seed=42)

        # per episode:
        mutated_task, is_solvable, ctx = curriculum.adapt(base_task)
        # ... run episode ...
        curriculum.record(task_id=..., skill_scores=..., composite=...)
        curriculum_reward = curriculum.compute_reward(composite, is_solvable, ctx)
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._skill_profile = SkillProfile()
        self._composite_history: Deque[float] = deque(maxlen=20)
        self._last_heuristic_score: float = 0.5
        self._last_mutated_task: Optional[TaskDefinition] = None
        self._episode_count: int = 0
        self._adaptation_log: List[Dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def adapt(
        self,
        base_task: TaskDefinition,
        generator_action: Optional[GeneratorAction] = None,
        long_horizon: bool = False,
    ) -> Tuple[TaskDefinition, bool, AdaptationContext]:
        """Produce an adapted task targeting the agent's weakest skill.

        Returns:
            mutated_task  — task with diagnostic mutations applied
            is_solvable   — whether the mutated task has valid solutions
            ctx           — AdaptationContext (target skill, mutations, weights)
        """
        self._episode_count += 1
        task = self._deep_copy_task(base_task)

        # Identify target skill and mutations
        target_skill = self._skill_profile.weakest_dimension()
        dyn_weights = compute_dynamic_weights(self._skill_profile)

        if generator_action and generator_action.mutations:
            mutations = generator_action.mutations[:MAX_MUTATIONS_PER_EPISODE]
        else:
            mutations = self._sample_diagnostic_mutations(task, target_skill, long_horizon)

        mutation_names: List[str] = []
        for mut in mutations:
            task = self._apply_mutation(task, mut)
            mutation_names.append(mut.mutation_type.value)

        # Long-horizon: optionally inject cascade scenario
        if long_horizon and base_task.planning_horizon_minutes > 60:
            task = self._inject_cascade_scenario(task)
            mutation_names.append("cascade_trigger")

        solvable = self._check_solvability(task)
        self._last_mutated_task = task
        self._last_heuristic_score = (
            self._score_scheduled_baseline(task) if solvable else 0.0
        )

        ctx = AdaptationContext(
            target_skill=target_skill,
            mutations_used=mutation_names,
            dynamic_weights=dyn_weights,
            rationale=(
                f"Targeting '{target_skill}' "
                f"(rolling mean={self._skill_profile.mean(target_skill):.3f}). "
                f"Mutations: {mutation_names}. "
                f"Weakest→weight={dyn_weights.get(target_skill, 1.0):.2f}x"
            ),
        )
        self._adaptation_log.append({
            "episode": self._episode_count,
            "target_skill": target_skill,
            "mutations": mutation_names,
            "skill_means": self._skill_profile.summary(),
        })
        return task, solvable, ctx

    def record(
        self,
        task_id: str,
        skill_scores: Dict[str, float],
        composite: float,
    ) -> None:
        """Update skill profile after an episode.

        skill_scores should contain per-component values in [0,1] for any
        subset of SKILL_DIMENSIONS. Missing components are not updated.
        """
        self._composite_history.append(composite)
        for dim in SKILL_DIMENSIONS:
            if dim in skill_scores:
                self._skill_profile.update(dim, skill_scores[dim])

    def compute_reward(
        self,
        controller_score: float,
        is_solvable: bool,
        ctx: Optional[AdaptationContext] = None,
    ) -> float:
        """Adaptation-aware reward: regret vs heuristic, weighted by target skill gap.

        Reward is NEGATIVE when agents beat baseline (curriculum was too easy).
        Reward is POSITIVE when agents fall below baseline (curriculum is working).
        An unsolvable scenario is penalized — the curriculum must adapt, not destroy.

        Bonus: extra regret weight for the specific skill being targeted so the
        curriculum is incentivized to generate the kind of pressure that matters.
        """
        if not is_solvable:
            return -1.0

        regret = controller_score - self._last_heuristic_score  # + = agents beat baseline
        base_reward = max(-1.0, min(1.0, -regret))              # flip: high regret = good

        # Boost reward when the target skill scenario produced high regret
        # (i.e. the adaptive mutation actually stressed the target dimension)
        if ctx is not None:
            skill_gap = 1.0 - self._skill_profile.mean(ctx.target_skill)
            # Weight by how big the skill gap is — bigger gap → bigger curriculum signal
            boosted = base_reward * (1.0 + 0.5 * skill_gap)
            return round(max(-1.0, min(1.0, boosted)), 4)

        return round(base_reward, 4)

    def diagnostic_report(self) -> Dict:
        """Full skill profile report for logging and debugging."""
        return {
            "skill_profile": self._skill_profile.summary(),
            "skill_gaps": self._skill_profile.gap_vector(),
            "weakest_dimension": self._skill_profile.weakest_dimension(),
            "dynamic_weights": compute_dynamic_weights(self._skill_profile),
            "composite_ema": (
                sum(self._composite_history) / max(1, len(self._composite_history))
            ),
            "episode_count": self._episode_count,
        }

    # Keep ema_score and difficulty_level as shims for backwards compatibility
    # with code that imported from the old ChallengeGenerator interface.
    @property
    def ema_score(self) -> float:
        h = self._composite_history
        return round(sum(h) / max(1, len(h)), 3)

    @property
    def difficulty_level(self) -> int:
        """Approximate difficulty from composite EMA — shim for old interface."""
        ema = self.ema_score
        if ema > 0.80:
            return 6
        if ema > 0.65:
            return 5
        if ema > 0.50:
            return 4
        if ema > 0.35:
            return 3
        if ema > 0.20:
            return 2
        return 1

    def mutate(self, base_task: TaskDefinition) -> Tuple[TaskDefinition, bool]:
        """Convenience shim: adapt a task and return (mutated_task, solvable).

        Callers that don't need the AdaptationContext can use this instead
        of the full ``adapt()`` API.
        """
        task, solvable, _ctx = self.adapt(base_task)
        return task, solvable

    def update(self, composite_score: float) -> None:
        """Convenience shim: record a composite score for difficulty tracking.

        Updates the composite history (used by ``difficulty_level`` and ``ema_score``).
        For full skill-level recording, use ``record()`` instead.
        """
        self._composite_history.append(composite_score)

    # ── Diagnostic mutation selection ─────────────────────────────────────────

    def _sample_diagnostic_mutations(
        self,
        task: TaskDefinition,
        target_skill: str,
        long_horizon: bool,
    ) -> List[GeneratorMutation]:
        """Select mutations that specifically exercise the target skill."""
        candidate_types = _SKILL_MUTATION_MAP.get(target_skill, [MutationType.TIGHTEN_WINDOW])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for mt in candidate_types:
            if mt not in seen:
                seen.add(mt)
                unique.append(mt)

        n_muts = min(MAX_MUTATIONS_PER_EPISODE, max(1, len(unique)))
        selected = self._rng.choices(unique, k=n_muts) if unique else [MutationType.TIGHTEN_WINDOW]

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        arrivals   = [f for f in task.flights if f.operation == OperationType.ARRIVAL]

        mutations: List[GeneratorMutation] = []
        for mtype in selected:
            mut = self._build_mutation(mtype, task, arrivals, departures, long_horizon)
            if mut is not None:
                mutations.append(mut)

        return mutations

    def _build_mutation(
        self,
        mtype: MutationType,
        task: TaskDefinition,
        arrivals: List[FlightRecord],
        departures: List[FlightRecord],
        long_horizon: bool,
    ) -> Optional[GeneratorMutation]:
        if mtype == MutationType.TIGHTEN_WINDOW:
            target = self._rng.choice(task.flights)
            # For long-horizon: squeeze windows later in the horizon (harder to plan for)
            squeeze = self._rng.randint(3, 7 if long_horizon else 5)
            return GeneratorMutation(
                mutation_type=mtype,
                target_flight_id=target.flight_id,
                params={"squeeze_minutes": squeeze},
                rationale=f"[ADAPTIVE] Squeeze {target.flight_id} window by {squeeze}min each side"
                          + (" (long-horizon stress)" if long_horizon else ""),
            )

        elif mtype == MutationType.ADD_ATFM_DEADLINE and departures:
            target = self._rng.choice(departures)
            # Tighter buffer for long-horizon (downstream network pressure)
            buffer = self._rng.randint(3, 8 if long_horizon else 10)
            return GeneratorMutation(
                mutation_type=mtype,
                target_flight_id=target.flight_id,
                params={"deadline_offset": buffer},
                rationale=f"[ADAPTIVE] ATFM deadline: {target.flight_id} must depart by scheduled+{buffer}",
            )

        elif mtype == MutationType.INCREASE_WEATHER_PENALTY:
            target_rwy = self._rng.choice(task.runways)
            delta = round(self._rng.uniform(0.15, 0.40), 2)
            return GeneratorMutation(
                mutation_type=mtype,
                target_runway_id=target_rwy.runway_id,
                params={"penalty_delta": delta},
                rationale=f"[ADAPTIVE] Weather degrades {target_rwy.runway_id} by {delta}x",
            )

        elif mtype == MutationType.INJECT_EMERGENCY and arrivals:
            base = self._rng.choice(arrivals)
            center = self._rng.randint(
                base.earliest_minute,
                min(base.latest_minute, base.earliest_minute + 20),
            )
            # For long-horizon: inject emergency later in window to test cascade recovery
            if long_horizon:
                center = min(base.latest_minute, center + 15)
            return GeneratorMutation(
                mutation_type=mtype,
                params={
                    "flight_id": f"EMG{self._rng.randint(100, 999)}",
                    "priority":  "emergency",
                    "minute":    center,
                    "runway":    self._rng.choice(task.runways).runway_id,
                },
                rationale="[ADAPTIVE] Emergency diversion — targeting emergency handling skill",
            )

        elif mtype == MutationType.ADD_CONFLICTING_FLIGHT and arrivals:
            anchor = self._rng.choice(arrivals)
            return GeneratorMutation(
                mutation_type=mtype,
                params={
                    "flight_id":  f"WKT{self._rng.randint(100, 999)}",
                    "wake_class": "H",
                    "operation":  "arrival",
                    "minute":     max(0, anchor.earliest_minute - 4),
                    "runway":     self._rng.choice(anchor.allowed_runways),
                },
                rationale="[ADAPTIVE] Heavy arrival 4min before window — wake trap (conflict avoidance)",
            )

        elif mtype == MutationType.CLOSE_RUNWAY_WINDOW:
            target_rwy = self._rng.choice(task.runways)
            duration = self._rng.randint(12, 25 if long_horizon else 20)
            return GeneratorMutation(
                mutation_type=mtype,
                target_runway_id=target_rwy.runway_id,
                params={"close_duration": duration},
                rationale=f"[ADAPTIVE] Runway {target_rwy.runway_id} closed {duration}min (coverage stress)",
            )

        return None

    # ── Long-horizon cascade injection ────────────────────────────────────────

    def _inject_cascade_scenario(self, task: TaskDefinition) -> TaskDefinition:
        """Add a cascade trigger: an early Heavy arrival that constrains later slots.

        This forces agents to reason: "If I put this Heavy here at minute 10,
        what wake gaps does it create for minute 30? minute 50?"
        """
        if not task.runways or not task.flights:
            return task

        # Place a Heavy at the earliest window to cascade into the planning horizon
        horizon_mid = task.planning_horizon_minutes // 3
        runway = self._rng.choice(task.runways)
        cascade_flight = FlightRecord(
            flight_id=f"CAS{self._rng.randint(100, 999)}",
            airline="FRT",
            operation=OperationType.ARRIVAL,
            wake_class=WakeClass.HEAVY,
            scheduled_minute=horizon_mid,
            earliest_minute=max(0, horizon_mid - 3),
            latest_minute=horizon_mid + 8,
            allowed_runways=[runway.runway_id],
            passengers=1,
            fuel_burn_per_minute=9.0,
            priority=PriorityClass.NORMAL,
            notes=(
                "[CASCADE] Heavy arrival — creates 6-min wake gap that cascades "
                "through the rest of the planning horizon"
            ),
            connection_risk=0.65,
        )
        return task.model_copy(update={"flights": list(task.flights) + [cascade_flight]})

    # ── Mutation application (identical logic to original generator) ──────────

    def _apply_mutation(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        if mut.mutation_type == MutationType.TIGHTEN_WINDOW:
            return self._tighten_window(task, mut)
        elif mut.mutation_type == MutationType.ADD_ATFM_DEADLINE:
            return task  # handled at environment level via atfm_deadlines dict
        elif mut.mutation_type == MutationType.INCREASE_WEATHER_PENALTY:
            return self._increase_weather(task, mut)
        elif mut.mutation_type == MutationType.INJECT_EMERGENCY:
            return self._inject_emergency(task, mut)
        elif mut.mutation_type == MutationType.ADD_CONFLICTING_FLIGHT:
            return self._add_conflicting_flight(task, mut)
        elif mut.mutation_type == MutationType.CLOSE_RUNWAY_WINDOW:
            return self._close_runway_window(task, mut)
        return task

    def _tighten_window(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        squeeze = mut.params.get("squeeze_minutes", 3)
        updated = []
        for f in task.flights:
            if f.flight_id == mut.target_flight_id:
                new_earliest = f.earliest_minute + squeeze
                new_latest   = f.latest_minute   - squeeze
                if new_latest - new_earliest >= MIN_WINDOW_WIDTH:
                    f = f.model_copy(update={
                        "earliest_minute": new_earliest,
                        "latest_minute":   new_latest,
                    })
            updated.append(f)
        return task.model_copy(update={"flights": updated})

    def _increase_weather(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        delta = mut.params.get("penalty_delta", 0.15)
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                rwy = rwy.model_copy(update={"weather_penalty": round(min(2.0, rwy.weather_penalty + delta), 2)})
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    def _inject_emergency(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "EMG001")
        minute = int(p.get("minute", 20))
        priority_str = str(p.get("priority", "emergency")).lower().strip()
        priority_str = _PRIORITY_ALIASES.get(priority_str, priority_str)
        try:
            priority = PriorityClass(priority_str)
        except ValueError:
            priority = PriorityClass.EMERGENCY
        runway_id = p.get("runway", task.runways[0].runway_id)

        new_flight = FlightRecord(
            flight_id=fid,
            airline="GOV",
            operation=OperationType.ARRIVAL,
            wake_class=WakeClass.MEDIUM,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 2),
            latest_minute=minute + 6,
            allowed_runways=[runway_id],
            passengers=8,
            fuel_burn_per_minute=7.5,
            priority=priority,
            notes=f"[ADAPTIVE] {priority.value} diversion — diagnostic mutation",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _add_conflicting_flight(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        p = mut.params
        fid = p.get("flight_id", "WKT001")
        minute = int(p.get("minute", 10))
        try:
            wake = WakeClass(str(p.get("wake_class", "H")).upper())
        except ValueError:
            wake = WakeClass.HEAVY
        try:
            operation = OperationType(str(p.get("operation", "arrival")).lower())
        except ValueError:
            operation = OperationType.ARRIVAL
        runway_id = p.get("runway", task.runways[0].runway_id)

        new_flight = FlightRecord(
            flight_id=fid,
            airline="FRT",
            operation=operation,
            wake_class=wake,
            scheduled_minute=minute,
            earliest_minute=max(0, minute - 1),
            latest_minute=minute + 5,
            allowed_runways=[runway_id],
            passengers=1,
            fuel_burn_per_minute=6.0,
            priority=PriorityClass.NORMAL,
            notes="[ADAPTIVE] Wake-turbulence trap — conflict avoidance training",
        )
        return task.model_copy(update={"flights": list(task.flights) + [new_flight]})

    def _close_runway_window(self, task: TaskDefinition, mut: GeneratorMutation) -> TaskDefinition:
        duration = mut.params.get("close_duration", 15)
        delta = min(1.9, 0.05 * duration)
        updated = []
        for rwy in task.runways:
            if rwy.runway_id == mut.target_runway_id:
                rwy = rwy.model_copy(update={
                    "weather_penalty": min(2.0, rwy.weather_penalty + delta),
                    "notes": rwy.notes + f" [CLOSED {duration}min — adaptive curriculum]",
                })
            updated.append(rwy)
        return task.model_copy(update={"runways": updated})

    # ── Solvability check + baseline scoring ─────────────────────────────────

    def _check_solvability(self, task: TaskDefinition) -> bool:
        for f in task.flights:
            if f.latest_minute - f.earliest_minute < 2:
                return False
            if not f.allowed_runways:
                return False
        from collections import defaultdict
        runway_demand: Dict[str, int] = defaultdict(int)
        for f in task.flights:
            for rwy_id in f.allowed_runways:
                runway_demand[rwy_id] += 1
        for rwy in task.runways:
            demand = runway_demand.get(rwy.runway_id, 0)
            effective_capacity = rwy.hourly_capacity / rwy.weather_penalty
            horizon_hours = task.planning_horizon_minutes / 60.0
            max_ops = effective_capacity * horizon_hours
            if demand > max_ops * 1.5:
                return False
        return True

    def _score_scheduled_baseline(self, task: TaskDefinition) -> float:
        slots = []
        for f in task.flights:
            if not f.allowed_runways:
                continue
            minute = max(f.earliest_minute, min(f.latest_minute, f.scheduled_minute))
            slots.append(SlotAssignment(
                flight_id=f.flight_id,
                runway=f.allowed_runways[0],
                assigned_minute=minute,
                hold_minutes=0,
            ))
        try:
            outcome = simulate_plan(task, slots)
            return outcome.normalized_score
        except Exception:
            return 0.5

    def _deep_copy_task(self, task: TaskDefinition) -> TaskDefinition:
        return TaskDefinition.model_validate(task.model_dump())


# ── Convenience: extract skill scores from episode metrics ────────────────────

def extract_skill_scores(
    outcome,
    aman_action=None,
    dman_action=None,
    task=None,
    atfm_deadlines: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Derive per-skill scores from a SimulationOutcome for curriculum recording.

    This is the inverse transform: reward function components → skill labels.
    """
    m = outcome.metrics

    scores: Dict[str, float] = {
        "conflict_avoidance": max(0.0, 1.0 - m.conflict_count * 0.25),
        "delay_efficiency":   m.delay_efficiency,
        "emergency_handling": 1.0 if m.emergency_violations == 0 else
                              max(0.0, 1.0 - m.emergency_violations / max(1, getattr(m, "emergency_count", 1))),
        "coverage":           m.schedule_completeness,
        "fairness":           m.fairness,
        "coordination":       m.connection_impact_score,  # proxy
    }

    # ATFM compliance from DMAN action
    if dman_action is not None and atfm_deadlines:
        dep_map = {s.flight_id: s for s in getattr(dman_action, "departure_slots", [])}
        ok = viol = 0
        for fid, deadline in atfm_deadlines.items():
            slot = dep_map.get(fid)
            if slot:
                (ok if slot.assigned_minute <= deadline else viol).__class__  # noqa
                if slot.assigned_minute <= deadline:
                    ok += 1
                else:
                    viol += 1
        total = ok + viol
        scores["atfm_compliance"] = ok / total if total > 0 else 1.0
    else:
        scores["atfm_compliance"] = 1.0 - min(1.0, getattr(m, "priority_violations", 0) * 0.1)

    return scores
