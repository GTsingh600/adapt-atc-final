"""FastAPI application for the multi-agent ATC environment.

Exposes:
  OpenEnv standard surface (via create_app):
    POST /reset
    POST /step
    GET  /state
    GET  /health

  Extended multi-agent REST endpoints:
    POST /multi_agent/reset
    POST /multi_agent/step/bid
    POST /multi_agent/finalize
    POST /multi_agent/episode
    GET  /multi_agent/profiles
    GET  /multi_agent/status

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv is required. Install dependencies first."
    ) from exc

try:
    from ..atc_env.models import ATCAction, ATCObservation
    from ..atc_env.server.atc_environment import ATCEnvironment
    from ..multi_agent.environment import MultiAgentATCEnvironment
    from ..multi_agent.adapter import ContextAdaptiveCurriculum
    from ..multi_agent.inference import run_episode
    from ..multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )
except ImportError:
    from atc_env.models import ATCAction, ATCObservation
    from atc_env.server.atc_environment import ATCEnvironment
    from multi_agent.environment import MultiAgentATCEnvironment
    from multi_agent.adapter import ContextAdaptiveCurriculum
    from multi_agent.inference import run_episode
    from multi_agent.models import (
        AMANAction, DMANAction, MultiAgentEpisodeResult, SupervisorProfileName,
    )

# Shared multi-agent singletons (serialised via lock — one env per server process)
_ma_env        = MultiAgentATCEnvironment(seed=0)
_ma_curriculum = ContextAdaptiveCurriculum(seed=0)
_MA_LOCK = asyncio.Lock()

app = create_app(
    ATCEnvironment,
    ATCAction,
    ATCObservation,
    env_name="atc_env",
    max_concurrent_envs=8,
)


# ── Multi-agent REST endpoints ─────────────────────────────────────────────────

class MAResetRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    supervisor_profile: Optional[str] = None


class MABidRequest(BaseModel):
    aman_action: Dict[str, Any]
    dman_action: Dict[str, Any]


class MAEpisodeRequest(BaseModel):
    task_id: str
    episode_id: int = 0
    use_curriculum: bool = True
    use_heuristic: bool = True


@app.post("/multi_agent/reset", tags=["multi-agent"])
async def ma_reset(req: MAResetRequest) -> Dict[str, Any]:
    """Reset multi-agent environment, return AMAN and DMAN observations."""
    async with _MA_LOCK:
        try:
            profile = (
                SupervisorProfileName(req.supervisor_profile)
                if req.supervisor_profile
                else None
            )
            aman_obs, dman_obs = await asyncio.to_thread(
                _ma_env.reset,
                task_id=req.task_id,
                episode_id=req.episode_id,
                supervisor_profile=profile,
            )
            return {
                "aman_observation": json.loads(aman_obs.model_dump_json()),
                "dman_observation": json.loads(dman_obs.model_dump_json()),
                "atfm_deadlines": _ma_env._state.atfm_deadlines,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/step/bid", tags=["multi-agent"])
async def ma_step_bid(req: MABidRequest) -> Dict[str, Any]:
    """Submit BID-round actions from both agents."""
    async with _MA_LOCK:
        try:
            aman_action = AMANAction.model_validate(req.aman_action)
            dman_action  = DMANAction.model_validate(req.dman_action)
            aman_obs2, dman_obs2, reward, done = await asyncio.to_thread(
                _ma_env.step_bid, aman_action, dman_action
            )
            return {
                "aman_observation": json.loads(aman_obs2.model_dump_json()),
                "dman_observation": json.loads(dman_obs2.model_dump_json()),
                "partial_reward": reward,
                "done": done,
                "round_number": _ma_env._state.round_number,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/multi_agent/finalize", tags=["multi-agent"])
async def ma_finalize() -> Dict[str, Any]:
    """Finalize episode, return full scored result."""
    async with _MA_LOCK:
        try:
            result: MultiAgentEpisodeResult = await asyncio.to_thread(_ma_env.finalize)
            return json.loads(result.model_dump_json())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/multi_agent/episode", tags=["multi-agent"])
async def ma_run_episode(req: MAEpisodeRequest) -> Dict[str, Any]:
    """Run a full multi-agent episode with heuristic or LLM agents."""
    async with _MA_LOCK:
        try:
            result = await asyncio.to_thread(
                run_episode,
                task_id=req.task_id,
                client=None,
                env=_ma_env,
                curriculum=_ma_curriculum if req.use_curriculum else None,
                episode_id=req.episode_id,
                use_curriculum=req.use_curriculum,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/multi_agent/profiles", tags=["multi-agent"])
async def ma_profiles() -> List[str]:
    """List available supervisor preference profiles."""
    return [p.value for p in SupervisorProfileName]


@app.get("/multi_agent/status", tags=["multi-agent"])
async def ma_status() -> Dict[str, Any]:
    """Current multi-agent environment state summary."""
    state = _ma_env._state
    if state is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "task_id": state.task.task_id if state.task else None,
        "round_number": state.round_number,
        "aman_slots_count": len(state.aman_slots),
        "dman_slots_count": len(state.dman_slots),
        "curriculum_difficulty": _ma_curriculum.difficulty_level,
        "supervisor_profile": (
            state.supervisor_profile.value if state.supervisor_profile else None
        ),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
