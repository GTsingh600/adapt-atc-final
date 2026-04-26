"""OpenEnv-compliant Pydantic models for the ATC multi-agent environment.

Inherits from openenv.core base classes so the models satisfy the OpenEnv
server and client contracts without boilerplate.

Action  → ATCAction   (carries one full AMAN+DMAN turn)
Observation → ATCObservation (what both agents receive after each step)
State   → ATCState    (full episode state snapshot, JSON-serialisable)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    # Graceful fallback so the module can be imported without openenv installed
    from pydantic import BaseModel as Action, BaseModel as Observation, BaseModel as State


class ATCAction(Action):
    """Combined AMAN + DMAN action submitted in a single environment step.

    Both agents generate completions independently; the server side parses each
    into its typed action (AMANAction / DMANAction) and routes them to the
    multi-agent environment.
    """

    aman_completion: str = Field(default="", description="Raw LLM completion from AMAN agent")
    dman_completion: str = Field(default="", description="Raw LLM completion from DMAN agent")
    round_type: str = Field(
        default="bid",
        description="Episode round: 'bid' | 'negotiate'",
    )


class ATCObservation(Observation):
    """What both agents receive after each environment step.

    Inherits `done`, `reward`, and `metadata` from the OpenEnv base class.
    Adds ATC-specific fields that the agents need to generate their next action.
    """

    aman_prompt: str = Field(default="", description="Full prompt text for AMAN agent")
    dman_prompt: str = Field(default="", description="Full prompt text for DMAN agent")
    round_type: str = Field(default="bid")
    round_number: int = Field(default=0, ge=0)
    conflict_log: List[str] = Field(default_factory=list)
    task_id: str = Field(default="")

    # Per-role rewards are emitted on the final step (done=True).
    aman_reward: Optional[float] = Field(default=None)
    dman_reward: Optional[float] = Field(default=None)
    composite_score: Optional[float] = Field(default=None)


class ATCState(State):
    """Full episode state snapshot.

    Inherits `episode_id` and `step_count` from the OpenEnv base class.
    """

    task_id: str = Field(default="")
    supervisor_profile: str = Field(default="safety_strict")
    negotiation_rounds: int = Field(default=0, ge=0)
    aman_reward: float = Field(default=0.0)
    dman_reward: float = Field(default=0.0)
    composite_score: float = Field(default=0.0)
    curriculum_difficulty: int = Field(default=1, ge=1, le=6)
