"""Episode dataset builder for multi-agent GRPO training.

Each training sample = one agent turn in one episode.
Format required by TRL GRPOTrainer:
    {"prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
     "task_id": ..., "agent_role": ..., ...metadata...}

System prompts encode:
  - Role identity + operational rules
  - Output JSON schema (strict)
  - Supervisor preference for this episode
  - Negotiation protocol rules

Parsing utilities decode LLM JSON completions back to typed actions.
"""

from __future__ import annotations

import json
import re
import sys, os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import OperationType, SlotAssignment
from tasks import task_catalog, ordered_tasks
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import (
    AMANAction,
    ADAPTAction,
    DMANAction,
    GeneratorAction,
    GeneratorMutation,
    MutationType,
    NegotiationMessage,
    MessageType,
    AgentRole,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
)
from multi_agent.supervisor import SupervisorAgent
from multi_agent.adapt import (
    apply_adapt_mapping,
    build_adapt_observation,
    _build_adapt_heuristic,
    parse_adapt_action,
)


# ── System prompts ────────────────────────────────────────────────────────────

AMAN_SYSTEM = """You are AMAN (Arrival Manager) at a busy Indian airport.
You ONLY control ARRIVAL flights. Do NOT assign departure flights.

CORE RULES (non-negotiable):
1. EMERGENCY and MEDICAL arrivals land FIRST — delay them ≤5 min maximum.
2. Respect wake turbulence separation: H→H≥4min, H→M≥5min, H→L≥6min, M→M≥3min.
3. Every arrival must stay within its [earliest, latest] window.
4. Only assign each flight to runways listed in its allowed_runways.
5. If DMAN broadcasts an EMERGENCY departure, yield your next runway slot to them.
6. Pre-empt gaps: if you know a DMAN emergency is at T+N, leave runway clear ±3 min.

NEGOTIATION PROTOCOL:
- Round BID: submit your best independent plan.
- Round NEGOTIATE: if conflicts reported, revise plan and send yield/acknowledge messages.
- Use outgoing_messages to communicate runway claims and yields to DMAN.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "arrival_slots": [
    {"flight_id": "...", "runway": "...", "assigned_minute": N, "hold_minutes": N}
  ],
  "rationale": "explain your sequencing decisions and how you satisfy supervisor preference",
  "emergency_yields": ["flight_id_you_yielded_for"],
  "outgoing_messages": [
    {
      "from_role": "AMAN",
      "message_type": "runway_claim|yield|acknowledge|request_gap|emergency_broadcast",
      "flight_id": "...",
      "requested_minute": N,
      "runway_id": "...",
      "priority": "normal|connection|medical|emergency",
      "reason": "...",
      "is_emergency": false
    }
  ],
  "commit": false
}"""


DMAN_SYSTEM = """You are DMAN (Departure Manager) at a busy Indian airport.
You ONLY control DEPARTURE flights. Do NOT assign arrival flights.

CORE RULES (non-negotiable):
1. ATFM network slot deadlines are HARD — missing them cascades to 3+ airports.
2. MEDICAL and EMERGENCY departures jump to the front of the departure queue.
3. Every departure must stay within its [earliest, latest] window.
4. Only assign each flight to runways listed in its allowed_runways.
5. If AMAN broadcasts an EMERGENCY arrival, clear the runway immediately.
6. Broadcast your own fuel/medical emergencies to AMAN in outgoing_messages.

PRIORITY RULE (air vs ground):
If BOTH a medical ARRIVAL and a medical DEPARTURE need the same slot:
→ The ARRIVAL wins (airborne aircraft cannot divert fuel-free; ground can hold).

NEGOTIATION PROTOCOL:
- Round BID: submit your best independent plan.
- Round NEGOTIATE: revise after conflict report; send messages to AMAN.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "departure_slots": [
    {"flight_id": "...", "runway": "...", "assigned_minute": N, "hold_minutes": N}
  ],
  "rationale": "explain sequencing and ATFM compliance and supervisor preference",
  "atfm_compliance": {"flight_id": deadline_minute_you_respected},
  "emergency_broadcasts": ["flight_id_of_your_emergency_departures"],
  "outgoing_messages": [
    {
      "from_role": "DMAN",
      "message_type": "runway_claim|yield|acknowledge|request_gap|emergency_broadcast",
      "flight_id": "...",
      "requested_minute": N,
      "runway_id": "...",
      "priority": "normal|connection|medical|emergency",
      "reason": "...",
      "is_emergency": false
    }
  ],
  "commit": false
}"""


GENERATOR_SYSTEM = """You are the Scenario Generator for multi-agent ATC training.
Your goal: mutate the scenario to make AMAN and DMAN fail to coordinate.
You are rewarded when they score LOW. You are penalised if the scenario is UNSOLVABLE.

MUTATION TYPES:
- tighten_window: squeeze a flight's time window (make it harder to sequence)
- inject_emergency: add a new EMERGENCY/MEDICAL arrival to disrupt sequencing
- increase_weather_penalty: degrade runway capacity
- add_atfm_deadline: add a hard network slot constraint to a departure
- close_runway_window: make a runway unavailable during peak period
- add_conflicting_flight: inject a Heavy arrival before a Light to create wake trap

STRATEGY TIPS:
- Simultaneous medical arrival + fuel emergency departure on same runway = maximum conflict
- Injecting emergency during peak hour breaks AMAN's sequence
- ATFM deadlines during weather degradation stress DMAN

OUTPUT FORMAT (strict JSON, no markdown):
{
  "mutations": [
    {
      "mutation_type": "tighten_window|inject_emergency|increase_weather_penalty|add_atfm_deadline|close_runway_window|add_conflicting_flight",
      "target_flight_id": "flight_id or null",
      "target_runway_id": "runway_id or null",
      "params": {"key": "value"},
      "rationale": "why this breaks coordination"
    }
  ],
  "strategy": "overall explanation of how these mutations disrupt AMAN/DMAN coordination"
}"""


ADAPT_SYSTEM = """You are ADAPT (Adaptive Decision Agent for Problem Transfer).

You receive scheduling problems from UNKNOWN domains. You have NO prior knowledge
of what the domain is. Your task is to analyse the structural properties of the
entities — as shown in the Entity Type Structural Profiles — and map them to
Air Traffic Control parameters so that the existing AMAN and DMAN coordination
agents can solve the problem without any code changes.

ATC PARAMETER REFERENCE:
  wake_class:  "H" = highest resource demand / tightest separation required
               "M" = moderate demand / standard separation
               "L" = lowest demand / minimum separation needed
  priority:    "emergency"  = handle FIRST, zero delay tolerance
               "medical"    = high urgency, ≤5 min delay maximum
               "connection" = hard external deadline that must be met
               "normal"     = standard flexible scheduling

STRUCTURAL REASONING GUIDE:
Read the numerical profiles. Do NOT reason from entity type names.

  time_pressure (0.0 → 1.0):
    > 0.85  → very tight window → strong urgency signal
    0.60–0.85 → moderate urgency
    < 0.60  → flexible, low urgency

  connection_risk (0.0 → 1.0):
    > 0.80  → emergency-level cascade risk if delayed
    0.50–0.80 → medical-level risk
    0.20–0.50 → connection deadline risk
    < 0.20  → normal, deferrable

  resource use (intensity/min × units):
    High values → entity needs more separation (Heavy equivalent)
    Low values  → entity needs less separation (Light equivalent)

  urgency_in_notes: YES = direct operator urgency signal → increase tier by 1.

COMBINED SCORE FORMULA:
  combined = 0.5 × time_pressure + 0.4 × connection_risk + 0.1 × urgency_flag
  ≥ 0.70 → "H" | 0.35–0.70 → "M" | < 0.35 → "L"

PRIORITY FORMULA:
  connection_risk ≥ 0.80 OR (time_pressure ≥ 0.95 AND urgency) → "emergency"
  connection_risk ≥ 0.50 OR time_pressure ≥ 0.80               → "medical"
  connection_risk ≥ 0.20 OR time_pressure ≥ 0.60               → "connection"
  else                                                           → "normal"

CRITICAL — PRIORITY DISTRIBUTION CONSTRAINT:
AMAN and DMAN are designed for a realistic priority distribution where emergencies
are RARE. Mapping too many entity types to "emergency" causes resource starvation:
AMAN yields all capacity to emergencies, DMAN gets nothing, and the joint score collapses.

Enforce these hard budgets (N = number of distinct entity types):
  - "emergency": EXACTLY 1 entity type maximum, regardless of N.
  - "H" wake:    at most floor(N / 3) entity types, minimum 1.
  - "medical":   at most ceil(N / 3) entity types (after emergency slot is taken).
  - Everything else cascades to "connection" or "normal".

If multiple entity types score ≥ 0.80 connection_risk, assign "emergency" only to the
SINGLE highest scorer. Demote the rest to "medical". Cite this explicitly in rationale.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "entity_wake_map": {
    "ENTITY_A": "H",
    "ENTITY_B": "M",
    "ENTITY_C": "L"
  },
  "entity_priority_map": {
    "ENTITY_A": "emergency",
    "ENTITY_B": "medical",
    "ENTITY_C": "normal"
  },
  "rationale": "per entity: 'ENTITY_A: tp=0.97 cr=0.93 score=0.86 → H/emergency (budget slot 1/1)'"
}"""



SUPERVISOR_SYSTEM_TEMPLATE = """You are an ATC Supervisor evaluating a completed runway plan.
Your preference this shift: {preference}

Score the plan 0.0-1.0 based on how well it satisfies YOUR preference (not generic quality).
Be specific about what satisfies or violates your stated priority.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "score": 0.0,
  "alignment": "explain how well the plan matches your stated preference",
  "key_violations": ["list specific violations of your preference"]
}}"""


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_episode_dataset(
    n_episodes: int = 200,
    seed: int = 42,
    include_generator: bool = False,
    include_supervisor: bool = False,
    include_adapt: bool = True,
    domain_episode_ratio: float = 0.50,
    long_horizon_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build ADAPT-focused multi-agent training dataset.

    ADAPT is the primary training role (~50% of samples).
    AMAN + DMAN are support roles (~25% each) that teach JSON format.

    Training mix:
      - 50% domain-transfer episodes: 1 ADAPT + 1 AMAN + 1 DMAN = 3 samples
      - 50% regular ATC episodes: 1 AMAN + 1 DMAN = 2 samples
      - Generator and Supervisor disabled by default
    """
    import random
    rng = random.Random(seed)
    catalog = task_catalog()
    task_list = list(ordered_tasks())
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)

    # Lazy-import ICU domain to avoid circular dependencies
    domain_tasks: List = []
    domain_name: str = ""
    domain_description: str = ""
    if include_adapt:
        from domains.icu import icu_task_catalog, ICU_DOMAIN_DESCRIPTION
        icu_catalog = icu_task_catalog()
        domain_tasks = list(icu_catalog.values())
        domain_name = "Hospital ICU Surge Management"
        domain_description = ICU_DOMAIN_DESCRIPTION

    samples: List[Dict[str, Any]] = []

    for ep_id in range(n_episodes):
        # ~50% of episodes are domain-transfer (ADAPT) episodes
        is_domain_ep = (
            include_adapt
            and bool(domain_tasks)
            and rng.random() < domain_episode_ratio
        )

        if is_domain_ep:
            domain_task = rng.choice(domain_tasks)
            profile = supervisor.sample_profile(ep_id)

            # Build ADAPT observation and heuristic action
            adapt_obs = build_adapt_observation(
                task=domain_task,
                profile=profile,
                domain_name=domain_name,
                domain_description=domain_description,
            )
            adapt_action = _build_adapt_heuristic(adapt_obs, domain_task)

            # Emit ADAPT training sample
            samples.append(_make_adapt_sample(
                ep_id=ep_id,
                obs=adapt_obs,
                domain_task=domain_task,
            ))

            # Apply ADAPT mapping so AMAN/DMAN see a properly parameterised task
            mapped_task = apply_adapt_mapping(domain_task, adapt_action)

            aman_obs, dman_obs = env.reset(
                episode_id=ep_id,
                supervisor_profile=profile,
                mutated_task=mapped_task,
            )
            atfm_json = json.dumps(env._state.atfm_deadlines)
            sup_desc = SUPERVISOR_PROFILES[profile]["description"]

            samples.append(_make_aman_sample(
                ep_id=ep_id,
                obs=aman_obs,
                atfm_json=atfm_json,
                dman_slots_json="[]",
                sup_desc=sup_desc,
                profile=profile,
                round_name="bid",
            ))
            samples.append(_make_dman_sample(
                ep_id=ep_id,
                obs=dman_obs,
                atfm_json=atfm_json,
                aman_slots_json="[]",
                sup_desc=sup_desc,
                profile=profile,
                round_name="bid",
            ))
            continue

        # Regular ATC episode — use base tasks with domain randomisation
        base_task = rng.choice(task_list)
        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        aman_obs, dman_obs = env.reset(
            episode_id=ep_id,
            supervisor_profile=profile,
            mutated_task=base_task,
            randomize=True,  # small perturbations prevent memorisation
        )

        atfm_json = json.dumps(env._state.atfm_deadlines)

        # AMAN BID sample
        samples.append(_make_aman_sample(
            ep_id=ep_id,
            obs=aman_obs,
            atfm_json=atfm_json,
            dman_slots_json="[]",
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
        ))

        # DMAN BID sample
        samples.append(_make_dman_sample(
            ep_id=ep_id,
            obs=dman_obs,
            atfm_json=atfm_json,
            aman_slots_json="[]",
            sup_desc=sup_desc,
            profile=profile,
            round_name="bid",
        ))

    return samples


# ── Sample builders ───────────────────────────────────────────────────────────

def _make_aman_sample(
    ep_id: int,
    obs,
    atfm_json: str,
    dman_slots_json: str,
    sup_desc: str,
    profile: SupervisorProfileName,
    round_name: str,
) -> Dict[str, Any]:
    system = AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
    user = obs.to_prompt_text()
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "task_id":            obs.task_id,
        "agent_role":         AgentRole.AMAN.value,
        "episode_id":         ep_id,
        "round":              round_name,
        "supervisor_profile": profile.value,
        "atfm_deadlines_json": atfm_json,
        "dman_slots_json":    dman_slots_json,
    }


def _make_dman_sample(
    ep_id: int,
    obs,
    atfm_json: str,
    aman_slots_json: str,
    sup_desc: str,
    profile: SupervisorProfileName,
    round_name: str,
) -> Dict[str, Any]:
    system = DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
    user = obs.to_prompt_text()
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "task_id":            obs.task_id,
        "agent_role":         AgentRole.DMAN.value,
        "episode_id":         ep_id,
        "round":              round_name,
        "supervisor_profile": profile.value,
        "atfm_deadlines_json": atfm_json,
        "aman_slots_json":    aman_slots_json,
    }


def _make_generator_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    difficulty_level: int,
    ema_score: float,
) -> Dict[str, Any]:
    user_content = (
        f"Current agent performance (EMA): {ema_score:.2f}\n"
        f"Target difficulty level: {difficulty_level}/6\n\n"
        f"Base task: {task.task_id} ({task.difficulty.value})\n"
        f"Flights: {len(task.flights)} | Runways: {len(task.runways)}\n"
        f"Airport: {task.airport}\n\n"
        f"Design mutations that will make AMAN and DMAN fail to coordinate "
        f"at difficulty level {difficulty_level}. Remember: solvable but hard."
    )
    return {
        "prompt": [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        "task_id":            task.task_id,
        "agent_role":         AgentRole.GENERATOR.value,
        "episode_id":         ep_id,
        "round":              "generate",
        "supervisor_profile": profile.value,
        "controller_scores":  ema_score,
    }


def _make_adapt_sample(
    ep_id: int,
    obs,
    domain_task,
) -> Dict[str, Any]:
    system = ADAPT_SYSTEM
    user = obs.to_prompt_text()
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "task_id":            obs.domain_id,
        "agent_role":         AgentRole.ADAPT.value,
        "episode_id":         ep_id,
        "round":              "adapt",
        "supervisor_profile": obs.supervisor_profile_name.value,
        "domain_task_json":   domain_task.model_dump_json(),
    }


def _make_supervisor_sample(
    ep_id: int,
    task,
    profile: SupervisorProfileName,
    sup_desc: str,
) -> Dict[str, Any]:
    system = SUPERVISOR_SYSTEM_TEMPLATE.format(preference=sup_desc)
    user_content = (
        f"Task: {task.task_id}\nAirport: {task.airport}\n"
        f"Flights: {len(task.flights)} | Runways: {len(task.runways)}\n\n"
        f"A merged AMAN+DMAN plan was submitted. Evaluate it against your preference."
    )
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "task_id":            task.task_id,
        "agent_role":         AgentRole.SUPERVISOR.value,
        "episode_id":         ep_id,
        "round":              "evaluate",
        "supervisor_profile": profile.value,
        "merged_plan_json":   "[]",
    }


# ── Action parsers (completion → typed action) ────────────────────────────────

def _coerce_completion_text(completion: Any) -> str:
    """Normalise chat-style completions from TRL into plain text."""
    if completion is None:
        return ""
    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="ignore")
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "completion", "generated_text"):
            if key in completion:
                return _coerce_completion_text(completion[key])
        try:
            return json.dumps(completion)
        except Exception:
            return str(completion)
    if isinstance(completion, list):
        parts = [_coerce_completion_text(item) for item in completion]
        return "\n".join(part for part in parts if part)
    return str(completion)


def _extract_json(text: Any) -> Optional[str]:
    """Extract first JSON object from an LLM completion."""
    text = _coerce_completion_text(text)
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def parse_aman_action(completion: Any) -> Optional[AMANAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        slots = [SlotAssignment(**s) for s in data.get("arrival_slots", [])]
        msgs  = [
            NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType(m.get("message_type", "runway_claim")),
                flight_id=m.get("flight_id", ""),
                requested_minute=int(m.get("requested_minute", 0)),
                runway_id=m.get("runway_id", ""),
                priority=m.get("priority", "normal"),
                reason=m.get("reason", ""),
                is_emergency=bool(m.get("is_emergency", False)),
            )
            for m in data.get("outgoing_messages", [])
        ]
        return AMANAction(
            arrival_slots=slots,
            rationale=data.get("rationale", ""),
            emergency_yields=data.get("emergency_yields", []),
            outgoing_messages=msgs,
            commit=bool(data.get("commit", False)),
        )
    except Exception:
        return None


def parse_dman_action(completion: Any) -> Optional[DMANAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        slots = [SlotAssignment(**s) for s in data.get("departure_slots", [])]
        msgs  = [
            NegotiationMessage(
                from_role=AgentRole.DMAN,
                message_type=MessageType(m.get("message_type", "runway_claim")),
                flight_id=m.get("flight_id", ""),
                requested_minute=int(m.get("requested_minute", 0)),
                runway_id=m.get("runway_id", ""),
                priority=m.get("priority", "normal"),
                reason=m.get("reason", ""),
                is_emergency=bool(m.get("is_emergency", False)),
            )
            for m in data.get("outgoing_messages", [])
        ]
        return DMANAction(
            departure_slots=slots,
            rationale=data.get("rationale", ""),
            atfm_compliance=data.get("atfm_compliance", {}),
            emergency_broadcasts=data.get("emergency_broadcasts", []),
            outgoing_messages=msgs,
            commit=bool(data.get("commit", False)),
        )
    except Exception:
        return None


def parse_generator_action(completion: Any) -> Optional[GeneratorAction]:
    raw = _extract_json(completion)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        mutations = []
        for m in data.get("mutations", []):
            try:
                mutations.append(GeneratorMutation(
                    mutation_type=MutationType(m.get("mutation_type", "tighten_window")),
                    target_flight_id=m.get("target_flight_id"),
                    target_runway_id=m.get("target_runway_id"),
                    params=m.get("params", {}),
                    rationale=m.get("rationale", ""),
                ))
            except Exception:
                continue
        return GeneratorAction(
            mutations=mutations,
            strategy=data.get("strategy", ""),
        )
    except Exception:
        return None
