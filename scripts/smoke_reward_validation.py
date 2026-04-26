"""Five-episode smoke test for multi-agent reward signal validation.

This script is intentionally lightweight and does not require torch/trl/unsloth.
It validates the critical data path before expensive GRPO runs:

1) raw planner output (JSON)
2) parser round-trip (parse_aman_action / parse_dman_action)
3) reward component traces (ATC_REWARD_TRACE=1)
4) final environment composite/coordination outputs

Usage:
  ATC_REWARD_TRACE=1 /home/keshav/ats/.venv/bin/python scripts/smoke_reward_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Enable detailed component-level traces from reward_functions.
os.environ.setdefault("ATC_REWARD_TRACE", "1")
os.environ.setdefault("REWARD_FAILURE_MODE", "strict")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import ordered_tasks
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
from multi_agent.models import SupervisorProfileName
from training.dataset import parse_aman_action, parse_dman_action
from training.reward_functions import aman_reward_fn, dman_reward_fn


def main() -> None:
    seed = 42
    env = MultiAgentATCEnvironment(seed=seed)
    curriculum = ContextAdaptiveCurriculum(seed=seed)
    task_ids = [t.task_id for t in ordered_tasks()]
    _profiles = list(SupervisorProfileName)

    print("=== 5-EPISODE SMOKE TEST: reward-signal validation ===")

    for ep in range(5):
        task_id = task_ids[ep % len(task_ids)]
        profile = _profiles[ep % len(_profiles)]

        base_task = env._catalog[task_id]
        mutated_task, _ = curriculum.mutate(base_task)

        aman_obs, dman_obs = env.reset(
            task_id=task_id,
            episode_id=ep,
            supervisor_profile=profile,
            mutated_task=mutated_task,
        )

        # Heuristic proposals act as deterministic stand-in for model completions.
        aman_action = _build_aman_heuristic(aman_obs)
        dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)

        raw_aman = json.dumps(
            {
                "arrival_slots": [s.model_dump() for s in aman_action.arrival_slots],
                "rationale": aman_action.rationale,
                "emergency_yields": aman_action.emergency_yields,
                "outgoing_messages": [m.model_dump(mode="json") for m in aman_action.outgoing_messages],
                "commit": aman_action.commit,
            }
        )
        raw_dman = json.dumps(
            {
                "departure_slots": [s.model_dump() for s in dman_action.departure_slots],
                "rationale": dman_action.rationale,
                "atfm_compliance": dman_action.atfm_compliance,
                "emergency_broadcasts": dman_action.emergency_broadcasts,
                "outgoing_messages": [m.model_dump(mode="json") for m in dman_action.outgoing_messages],
                "commit": dman_action.commit,
            }
        )

        parsed_aman = parse_aman_action(raw_aman)
        parsed_dman = parse_dman_action(raw_dman)

        if parsed_aman is None or parsed_dman is None:
            raise RuntimeError(f"Parser failure at episode {ep}")

        _, _, _, done = env.step_bid(parsed_aman, parsed_dman)
        if not done:
            env.step_negotiate(parsed_aman, parsed_dman)
        result = env.finalize()

        aman_slots_json = json.dumps([s.model_dump() for s in parsed_aman.arrival_slots])
        dman_slots_json = json.dumps([s.model_dump() for s in parsed_dman.departure_slots])
        atfm_json = json.dumps(env._state.atfm_deadlines)

        aman_r = aman_reward_fn(
            [raw_aman],
            task_id=[env._state.task.task_id],
            supervisor_profile=[profile.value],
            dman_slots_json=[dman_slots_json],
            atfm_deadlines_json=[atfm_json],
        )[0]
        dman_r = dman_reward_fn(
            [raw_dman],
            task_id=[env._state.task.task_id],
            supervisor_profile=[profile.value],
            aman_slots_json=[aman_slots_json],
            atfm_deadlines_json=[atfm_json],
        )[0]

        print(f"\n--- EPISODE {ep} ---")
        print(f"task={env._state.task.task_id} profile={profile.value}")
        print(f"raw_aman_len={len(raw_aman)} raw_dman_len={len(raw_dman)}")
        print(
            f"parsed_aman_slots={len(parsed_aman.arrival_slots)} "
            f"parsed_dman_slots={len(parsed_dman.departure_slots)}"
        )
        print(f"aman_reward={aman_r} dman_reward={dman_r}")
        print(
            f"composite={result.composite_score} "
            f"coord={result.per_role.coordination_score} "
            f"conflicts={result.per_role.cross_lane_conflicts}"
        )

    print("\nSMOKE TEST PASSED: 5 episodes executed with reward traces and parsed actions.")


if __name__ == "__main__":
    main()
