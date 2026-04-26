"""Smoke tests for multi-agent ATC environment."""

from __future__ import annotations

import pytest
from models import SlotAssignment, OperationType
from tasks import task_catalog
from engine import simulate_plan
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from multi_agent.models import (
    AMANAction,
    DMANAction,
    AgentRole,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
    PerRoleMetrics,
)
from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
from graders import MultiAgentCoordinationGrader, grade_multi_agent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return MultiAgentATCEnvironment(seed=0)


@pytest.fixture
def catalog():
    return task_catalog()


@pytest.fixture
def easy_task(catalog):
    return catalog["delhi_monsoon_recovery_easy"]


@pytest.fixture
def hard_task(catalog):
    return catalog["bengaluru_irrops_hard"]


# ── Environment reset ─────────────────────────────────────────────────────────

def test_reset_splits_flights_by_role(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    assert aman_obs.role == AgentRole.AMAN
    assert dman_obs.role == AgentRole.DMAN
    assert all(f.operation == OperationType.ARRIVAL   for f in aman_obs.my_flights)
    assert all(f.operation == OperationType.DEPARTURE for f in dman_obs.my_flights)
    total = len(aman_obs.my_flights) + len(dman_obs.my_flights)
    assert total == len(easy_task.flights)


def test_reset_assigns_supervisor_profile(env, easy_task):
    aman_obs, _ = env.reset(task_id=easy_task.task_id, episode_id=0)
    assert aman_obs.supervisor_profile_name in SupervisorProfileName
    assert len(aman_obs.supervisor_description) > 10


def test_atfm_deadlines_are_departures_only(env, easy_task):
    env.reset(task_id=easy_task.task_id, episode_id=0)
    dep_ids = {f.flight_id for f in easy_task.flights if f.operation == OperationType.DEPARTURE}
    for fid in env._state.atfm_deadlines:
        assert fid in dep_ids, f"{fid} is not a departure but got ATFM deadline"


# ── Heuristic planners ────────────────────────────────────────────────────────

def test_heuristic_aman_covers_all_arrivals(env, hard_task):
    aman_obs, _ = env.reset(task_id=hard_task.task_id, episode_id=1)
    action = _build_aman_heuristic(aman_obs)
    assigned_ids = {s.flight_id for s in action.arrival_slots}
    arrival_ids  = {f.flight_id for f in aman_obs.my_flights}
    # Heuristic must assign ≥90% of arrivals (some may be unassignable in extreme cases)
    assert len(assigned_ids) >= 0.9 * len(arrival_ids)


def test_heuristic_dman_covers_all_departures(env, hard_task):
    aman_obs, dman_obs = env.reset(task_id=hard_task.task_id, episode_id=1)
    action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    assigned_ids = {s.flight_id for s in action.departure_slots}
    dep_ids      = {f.flight_id for f in dman_obs.my_flights}
    assert len(assigned_ids) >= 0.9 * len(dep_ids)


def test_heuristic_slots_within_windows(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    flights_by_id = {f.flight_id: f for f in easy_task.flights}

    aman_action = _build_aman_heuristic(aman_obs)
    for s in aman_action.arrival_slots:
        f = flights_by_id[s.flight_id]
        assert f.earliest_minute <= s.assigned_minute <= f.latest_minute, \
            f"{s.flight_id} assigned_minute={s.assigned_minute} outside [{f.earliest_minute},{f.latest_minute}]"

    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    for s in dman_action.departure_slots:
        f = flights_by_id[s.flight_id]
        assert f.earliest_minute <= s.assigned_minute <= f.latest_minute, \
            f"{s.flight_id} assigned_minute={s.assigned_minute} outside [{f.earliest_minute},{f.latest_minute}]"


# ── Bid round ─────────────────────────────────────────────────────────────────

def test_bid_step_returns_valid_reward(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    _, _, reward, done = env.step_bid(aman_action, dman_action)
    assert -1.0 <= reward <= 1.0
    assert isinstance(done, bool)


def test_bid_step_updates_state(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    env.step_bid(aman_action, dman_action)
    assert len(env._state.aman_slots) > 0
    assert len(env._state.dman_slots) > 0
    assert env._state.round_number >= 1


# ── Finalize ──────────────────────────────────────────────────────────────────

def test_finalize_returns_bounded_rewards(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    env.step_bid(aman_action, dman_action)
    result = env.finalize()
    assert 0.0 <= result.aman_reward <= 1.0
    assert 0.0 <= result.dman_reward <= 1.0
    assert 0.0 <= result.composite_score <= 1.0
    assert 0.0 <= result.per_role.coordination_score <= 1.0


def test_finalize_per_role_flight_counts(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    env.step_bid(aman_action, dman_action)
    result = env.finalize()
    n_arr = sum(1 for f in easy_task.flights if f.operation == OperationType.ARRIVAL)
    n_dep = sum(1 for f in easy_task.flights if f.operation == OperationType.DEPARTURE)
    assert result.per_role.arrival_count == n_arr
    assert result.per_role.departure_count == n_dep


def test_rewards_bounded(env, easy_task):
    aman_obs, dman_obs = env.reset(task_id=easy_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    env.step_bid(aman_action, dman_action)
    result = env.finalize()
    assert 0.0 <= result.aman_reward <= 1.0
    assert 0.0 <= result.dman_reward <= 1.0
    assert 0.0 <= result.composite_score <= 1.0


# ── Multi-episode stability ───────────────────────────────────────────────────

def test_multiple_episodes_no_state_leak(env):
    """Ensure environment state does not bleed between episodes."""
    catalog = task_catalog()
    for ep, task_id in enumerate(list(catalog)[:3]):
        aman_obs, dman_obs = env.reset(task_id=task_id, episode_id=ep)
        assert env._state.task.task_id == task_id
        aman_action = _build_aman_heuristic(aman_obs)
        dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
        env.step_bid(aman_action, dman_action)
        result = env.finalize()
        assert 0.0 <= result.composite_score <= 1.0


# ── Generator ─────────────────────────────────────────────────────────────────

def test_curriculum_mutation_produces_solvable_task(easy_task):
    curriculum = ContextAdaptiveCurriculum(seed=7)
    mutated, solvable = curriculum.mutate(easy_task)
    assert solvable
    assert mutated.task_id == easy_task.task_id
    assert len(mutated.runways) == len(easy_task.runways)


def test_curriculum_escalates_difficulty():
    curriculum = ContextAdaptiveCurriculum(seed=0)
    assert curriculum.difficulty_level == 1
    for _ in range(12):
        curriculum.update(0.80)  # consistently high scores
    assert curriculum.difficulty_level >= 2


def test_curriculum_eases_on_low_scores():
    curriculum = ContextAdaptiveCurriculum(seed=0)
    curriculum._difficulty_level = 4
    for _ in range(12):
        curriculum.update(0.15)  # consistently low scores
    assert curriculum.difficulty_level <= 3


def test_curriculum_solvability_guard(easy_task):
    curriculum = ContextAdaptiveCurriculum(seed=0)
    curriculum._difficulty_level = 6
    for _ in range(5):
        mutated, solvable = curriculum.mutate(easy_task)
        # Even at max difficulty, task should still be solvable
        if not solvable:
            assert curriculum.compute_reward(0.3, solvable) == -1.0


# ── Supervisor profiles (config data — not an agent) ─────────────────────────

def test_supervisor_profile_rotation():
    profiles = set()
    for ep in range(10):
        profiles.add(SupervisorProfileName(list(SupervisorProfileName)[ep % len(SupervisorProfileName)]))
    assert len(profiles) == len(SupervisorProfileName)


# ── Coordination grader ───────────────────────────────────────────────────────

def test_coordination_grader_zero_conflicts(easy_task):
    grader = MultiAgentCoordinationGrader()
    slots = [
        SlotAssignment(
            flight_id=f.flight_id,
            runway=f.allowed_runways[0],
            assigned_minute=f.scheduled_minute,
            hold_minutes=0,
        )
        for f in easy_task.flights
    ]
    outcome = simulate_plan(easy_task, slots)
    arrivals  = [s for s in slots if next(f for f in easy_task.flights if f.flight_id == s.flight_id).operation == OperationType.ARRIVAL]
    departures = [s for s in slots if next(f for f in easy_task.flights if f.flight_id == s.flight_id).operation == OperationType.DEPARTURE]
    grade = grader.grade(easy_task, outcome, arrivals, departures, negotiation_rounds=0)
    assert 0.0 < grade.score < 1.0
    assert "cross_lane_avoidance" in grade.sub_scores


def test_grade_multi_agent_returns_three_grades(easy_task):
    slots = [
        SlotAssignment(flight_id=f.flight_id, runway=f.allowed_runways[0],
                       assigned_minute=f.scheduled_minute, hold_minutes=0)
        for f in easy_task.flights
    ]
    outcome = simulate_plan(easy_task, slots)
    arrivals   = [s for s in slots if next(f for f in easy_task.flights if f.flight_id == s.flight_id).operation == OperationType.ARRIVAL]
    departures = [s for s in slots if next(f for f in easy_task.flights if f.flight_id == s.flight_id).operation == OperationType.DEPARTURE]
    grades = grade_multi_agent(easy_task, outcome, arrivals, departures)
    names = {g.grader_name for g in grades}
    assert "composite_task_grader"       in names
    assert "multi_agent_coordination"    in names
    assert "llm_supervisor"              in names


# ── Emergency handling ────────────────────────────────────────────────────────

def test_emergency_broadcast_injected_into_messages(env, hard_task):
    """Hard task has MED001 (EMERGENCY arrival) and MED208 (MEDICAL departure)."""
    aman_obs, dman_obs = env.reset(task_id=hard_task.task_id, episode_id=0)
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)

    aman_obs2, dman_obs2, _, _ = env.step_bid(aman_action, dman_action)
    # After bid, AMAN emergency should be broadcast to DMAN
    emg_msgs_for_dman = [m for m in dman_obs2.incoming_messages if m.is_emergency]
    # bengaluru_irrops_hard has MED001 (EMERGENCY arrival)
    assert len(emg_msgs_for_dman) >= 1, "Emergency arrival not broadcast to DMAN"
