"""OpenEnv server-side environment wrapping the multi-agent ATC core.

Implements the three-method OpenEnv Environment interface:
  reset(seed, episode_id, **kwargs) -> ATCObservation
  step(action, **kwargs)           -> ATCObservation
  state (property)                 -> ATCState

The underlying MultiAgentATCEnvironment runs a 3-round protocol:
  Round 0 BID       → step(round_type="bid")
  Round 1 NEGOTIATE → step(round_type="negotiate")   [skipped when no conflicts]
  Round 2 FINAL     → done=True, per-role rewards emitted

SUPPORTS_CONCURRENT_SESSIONS = True because TRL spawns multiple rollout workers.
Each worker creates its own server instance via the OpenEnv create_app factory,
so no shared mutable state exists between sessions.
"""

from __future__ import annotations

import sys
import os
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from openenv.core.env_server import Environment
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from tasks import ordered_tasks
from training.dataset import parse_aman_action, parse_dman_action

from ..models import ATCAction, ATCObservation, ATCState


def _make_env_base():
    if _OPENENV_AVAILABLE:
        return Environment
    # Minimal duck-type base when openenv not installed
    class _Base:
        SUPPORTS_CONCURRENT_SESSIONS = True
    return _Base


_EnvBase = _make_env_base()


class ATCEnvironment(_EnvBase):
    """Multi-agent ATC environment conforming to the OpenEnv server interface."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        if _OPENENV_AVAILABLE:
            super().__init__()
        self._env = MultiAgentATCEnvironment(seed=42)
        self._curriculum = ContextAdaptiveCurriculum(seed=42)
        self._task_list = list(ordered_tasks())
        self._episode_state = ATCState()

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> ATCObservation:
        """Start a new episode.

        kwargs accepted:
            task_id (str): specific task to load; random if omitted
            use_curriculum (bool): apply curriculum mutations (default True)
        """
        task_id = kwargs.get("task_id")
        use_curriculum = kwargs.get("use_curriculum", True)
        ep_id = int(episode_id or 0)

        # Apply adaptive curriculum mutation when requested
        mutated_task = None
        if use_curriculum:
            import random
            rng = random.Random(seed or ep_id)
            if task_id:
                from tasks import task_catalog
                base = task_catalog().get(task_id)
            else:
                base = rng.choice(self._task_list)
            if base:
                mutated_task, _ = self._curriculum.mutate(base)

        aman_obs, dman_obs = self._env.reset(
            task_id=task_id,
            episode_id=ep_id,
            mutated_task=mutated_task,
        )

        self._episode_state = ATCState(
            episode_id=str(ep_id),
            task_id=aman_obs.task_id,
            supervisor_profile=aman_obs.supervisor_profile_name.value,
            curriculum_difficulty=self._curriculum.difficulty_level,
        )

        return ATCObservation(
            aman_prompt=aman_obs.to_prompt_text(),
            dman_prompt=dman_obs.to_prompt_text(),
            round_type="bid",
            round_number=0,
            task_id=aman_obs.task_id,
            done=False,
        )

    def step(self, action: ATCAction, timeout_s=None, **kwargs) -> ATCObservation:
        """Process one agent turn.

        round_type="bid"       → BID round, may trigger auto-finalize if no conflicts
        round_type="negotiate" → NEGOTIATE + FINAL, always returns done=True
        """
        self._episode_state.step_count += 1

        aman_action = parse_aman_action(action.aman_completion)
        dman_action  = parse_dman_action(action.dman_completion)

        # Fallback to empty actions when parsing fails (penalised by reward)
        from multi_agent.models import AMANAction, DMANAction
        if aman_action is None:
            aman_action = AMANAction(arrival_slots=[], rationale="parse_error")
        if dman_action is None:
            dman_action = DMANAction(departure_slots=[], rationale="parse_error")

        if action.round_type == "bid":
            aman_obs, dman_obs, partial_reward, done = self._env.step_bid(
                aman_action, dman_action
            )
            if done:
                return self._finalize(partial_reward)

            self._episode_state.step_count += 1  # negotiate will be next
            return ATCObservation(
                aman_prompt=aman_obs.to_prompt_text(),
                dman_prompt=dman_obs.to_prompt_text(),
                round_type="negotiate",
                round_number=1,
                conflict_log=aman_obs.conflict_log,
                task_id=self._episode_state.task_id,
                reward=partial_reward,
                done=False,
            )

        # negotiate → always terminates the episode
        aman_obs, dman_obs, partial_reward, _ = self._env.step_negotiate(
            aman_action, dman_action
        )
        return self._finalize(partial_reward)

    @property
    def state(self) -> ATCState:
        return self._episode_state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finalize(self, partial_reward: float) -> ATCObservation:
        result = self._env.finalize()

        # Update curriculum
        self._curriculum.update(result.composite_score)

        self._episode_state.aman_reward = result.aman_reward
        self._episode_state.dman_reward = result.dman_reward
        self._episode_state.composite_score = result.composite_score
        self._episode_state.negotiation_rounds = result.negotiation_rounds
        self._episode_state.curriculum_difficulty = self._curriculum.difficulty_level

        composite = (result.aman_reward + result.dman_reward) / 2.0

        return ATCObservation(
            aman_prompt="",
            dman_prompt="",
            round_type="final",
            round_number=2,
            task_id=self._episode_state.task_id,
            reward=round(composite, 4),
            done=True,
            aman_reward=result.aman_reward,
            dman_reward=result.dman_reward,
            composite_score=result.composite_score,
        )
