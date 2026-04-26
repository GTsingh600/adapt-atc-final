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
from tasks import task_catalog, ordered_tasks, micro_task_catalog
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from multi_agent.models import (
    AMANAction,
    ADAPTAction,
    DMANAction,
    NegotiationMessage,
    MessageType,
    AgentRole,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
)
from multi_agent.adapt import (
    apply_adapt_mapping,
    build_adapt_observation,
    _build_adapt_heuristic,
    parse_adapt_action,
)


# ── System prompts ────────────────────────────────────────────────────────────

AMAN_SYSTEM = """You are AMAN (Arrival Manager). Schedule ARRIVAL flights ONLY.

RULES:
1. EMERGENCY/MEDICAL arrivals land FIRST — delay ≤5 min.
2. Wake separation (minutes between consecutive arrivals on same runway):
   H→H=4  H→M=5  H→L=6  M→H=3  M→M=3  M→L=4  L→H=3  L→M=3  L→L=3
3. Each flight within its [earliest, latest] window.
4. Use only allowed_runways listed for each flight.
5. Assign EVERY arrival — coverage matters.

OUTPUT (strict JSON, no markdown, no extra text):
{"arrival_slots":[{"flight_id":"...","runway":"...","assigned_minute":N,"hold_minutes":N}],"rationale":"brief reason"}"""


DMAN_SYSTEM = """You are DMAN (Departure Manager). Schedule DEPARTURE flights ONLY.

RULES:
1. ATFM deadlines are HARD — missing cascades to 3+ airports.
2. EMERGENCY/MEDICAL departures go FIRST.
3. Each flight within its [earliest, latest] window.
4. Use only allowed_runways listed for each flight.
5. Assign EVERY departure — coverage matters.

OUTPUT (strict JSON, no markdown, no extra text):
{"departure_slots":[{"flight_id":"...","runway":"...","assigned_minute":N,"hold_minutes":N}],"rationale":"brief reason with ATFM notes"}"""


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



# ── Dataset builder ───────────────────────────────────────────────────────────

def build_episode_dataset(
    n_episodes: int = 200,
    seed: int = 42,
    include_adapt: bool = True,
    domain_episode_ratio: float = 0.65,
    long_horizon_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build ADAPT-first multi-agent training dataset.

    ADAPT is the PRIMARY training role (~65% of samples).
    AMAN + DMAN use micro tasks (5-6 flights) for short-context format learning.

    Training mix:
      - 65% domain-transfer episodes: 1 ADAPT + 1 AMAN + 1 DMAN = 3 samples
      - 35% regular ATC micro episodes: 1 AMAN + 1 DMAN = 2 samples
    """
    import random
    rng = random.Random(seed)
    catalog = task_catalog()
    # Use micro tasks for AMAN/DMAN format-learning — short context, 1.5B-friendly
    micro_catalog = micro_task_catalog()
    task_list = list(micro_catalog.values()) if micro_catalog else list(ordered_tasks())
    _profiles = list(SupervisorProfileName)
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
            profile = _profiles[ep_id % len(_profiles)]

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
        profile = _profiles[ep_id % len(_profiles)]
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


def parse_generator_action(completion: Any) -> Optional[Dict]:
    """Legacy stub — returns None. Generator agent has been removed."""
    return None
