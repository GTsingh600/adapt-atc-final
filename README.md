---
title: ATC Optimization OpenEnv
sdk: docker
app_port: 8000
license: mit
tags:
  - openenv
  - multi-agent
  - grpo
  - air-traffic-control
  - long-horizon-planning
  - self-adapting-curriculum
  - cooperative-rl
  - domain-transfer
  - adapt
---

![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)
![Tasks](https://img.shields.io/badge/tasks-4%20ATC%20%2B%203%20ICU-green)
![Modes](https://img.shields.io/badge/modes-single%20%2B%20multi--agent%20%2B%20long--horizon-orange)
![Training](https://img.shields.io/badge/training-GRPO%20%2B%20Unsloth-purple)
![ADAPT](https://img.shields.io/badge/ADAPT-domain%20transfer-red)
![Loss](https://img.shields.io/badge/loss%20functions-7%20novel%20components-yellow)
![HF Space](https://img.shields.io/badge/HF%20Space-live-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

# Shared Runways, Split Intelligence

*Multi-agent reinforcement learning for cooperative air traffic control — with self-adapting curriculum, long-horizon planning, and seven novel loss function components*

---

> **The problem**: At any major airport, hundreds of aircraft compete for a handful of runways every hour. The arrival manager doesn't see the departure queue. The departure manager doesn't see the landing sequence. Emergencies arrive unannounced. And the definition of "a good plan" shifts every session.
>
> **Our answer**: Two LLM agents, partial observability, real physics, a self-adapting curriculum that diagnoses what they still can't do — and a training signal designed for long-horizon planning.

---

## Judge Quick View

| Item | Detail |
|---|---|
| Domain | Real ATC disruption recovery — 4 Indian airports + ICU domain transfer |
| Problem this solves | LLMs can't plan across hours — decisions at minute 5 cascade to minute 180 |
| Agents | AMAN, DMAN, ADAPT meta-agent, self-adapting curriculum |
| Theme | **#2 — Super Long-Horizon Planning & Instruction Following** |
| Coordination protocol | BID → NEGOTIATE → FINAL (genuine partial observability) |
| Novel contribution | Self-adapting curriculum (diagnoses weaknesses) + 7 research-grade loss functions |
| Training | GRPO · N=4 groups · Unsloth 4-bit QLoRA · Colab T4 · full-model option |
| Reward design | Verifiable physics — no LLM judge in the loop |
| OpenEnv | Full compliance: `ATCAction`, `ATCObservation`, `ATCState` |
| Space | https://huggingface.co/spaces/GTsingh12/ATS-openenv |

---

## The Problem We're Solving

Standard LLM benchmarks test reasoning over a single context window. But real air traffic control requires planning across *hours* — where an early sequencing decision creates a wake-turbulence constraint that only bites 45 minutes later.

**What makes this hard, specifically:**

1. **Temporal cascade**: A Heavy arrival placed at minute 5 forces a 6-minute gap that cascades through every subsequent Light departure on the same runway.

2. **Cross-agent information asymmetry**: AMAN (arrivals) cannot see DMAN's departure queue. DMAN cannot see AMAN's ATFM deadlines. They must infer each other's constraints from messages alone.

3. **Sparse, delayed reward**: The damage from a bad minute-5 decision isn't visible until minute 45. Standard terminal reward gives no gradient signal for the causal action.

4. **Shifting objectives**: Each episode activates a different supervisor preference (`safety_strict`, `throughput_max`, `fuel_economy`, `emergency_priority`, `fairness_balanced`). Agents cannot overfit to a fixed objective.

5. **Beyond context limits**: A full morning peak covers 4–6 hours. No LLM context window holds 6 hours of flight data. Agents must decompose, remember, and adapt across planning epochs.

---

## What We Built

### Theme #2: Long-Horizon Planning

We directly target the capability gap identified in Theme #2:

> *Enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes — pushing beyond shallow next-token reasoning toward structured planning and durable internal representations.*

Our implementation has three components:

#### 1. Hierarchical Plan Decomposition

Long tasks (horizon > 60 min) are split into 45-minute **planning epochs**. Each epoch:
- Contains only the flights with windows in that slice
- Inherits **carry-over constraints** from the previous epoch (boundary wake gaps)
- Is solved by a fresh AMAN+DMAN BID→NEGOTIATE→FINAL cycle
- Contributes to the aggregate reward with per-epoch and recovery-adjusted scoring

Agents are explicitly instructed to reason at three levels:
```
STRATEGIC  → which flights need which runway across the full shift?
TACTICAL   → which 15-minute window does each flight go in THIS epoch?
OPERATIONAL → what exact slot minute minimizes delay within the window?
```

#### 2. Cascade Detection & Recovery Reward

A `CascadeDetector` identifies when an epoch-t decision caused an epoch-(t+k) problem:

```
Heavy at boundary (T+42) → created 6-min wake gap → conflict in next epoch
```

When detected, the recovery gradient rewards agents that **recognized and fixed** the cascade in the subsequent epoch — directly training the specific skill of recovering from early mistakes.

#### 3. EpisodeMemory: Planning Beyond Context Window

Agents write decisions to a structured `EpisodeMemory` between epochs. The memory is injected into the next epoch's system prompt (≤2048 chars, stays in context). This enables genuine multi-session planning that persists across the context window limit.

---

## Self-Adapting Curriculum (not self-improving)

The original design used a **self-improving** curriculum: track EMA composite score → escalate difficulty when score rises. This is blind escalation — it makes things harder but doesn't target what the agent actually can't do.

We replaced it with a **self-adapting** curriculum.

### The Difference

| Self-Improving | Self-Adapting |
|---|---|
| EMA composite → difficulty level | Per-component skill profile → targeted scenarios |
| Escalates blindly | Diagnoses which dimension is weakest |
| Same type of challenge, harder | Specifically exercises the skill gap |
| Fixed reward weights | Dynamic reward weights: weakest component gets loudest gradient |

### How It Works

The `ContextAdaptiveCurriculum` maintains a rolling mean for 7 skill dimensions:

| Dimension | What it measures |
|---|---|
| `conflict_avoidance` | Wake-turbulence and cross-lane conflict rate |
| `delay_efficiency` | Total system delay vs budget |
| `emergency_handling` | On-time dispatch of EMERGENCY/MEDICAL flights |
| `atfm_compliance` | DMAN meets network slot deadlines |
| `coverage` | Fraction of flights assigned valid slots |
| `coordination` | Multi-agent negotiation quality |
| `fairness` | Equitable delay distribution across airlines |

Each episode: identify the weakest dimension → select mutations that exercise it → generate scenarios that stress exactly that gap.

```
Weakest: emergency_handling (mean=0.42)
→ INJECT_EMERGENCY mutation selected
→ Emergency arrives mid-peak, tight 8-minute window
→ Reward weight for emergency_score: 2.1x baseline

Weakest: conflict_avoidance (mean=0.51)
→ ADD_CONFLICTING_FLIGHT + CLOSE_RUNWAY_WINDOW selected
→ Heavy arrives 4 minutes before Light departure window
→ Reward weight for conflict_avoidance: 1.8x baseline
```

The `dynamic_weights` vector is computed via softmax over skill gaps:

```python
raw_i = exp(gap_i * 3.0)          # amplify differences
w_i   = raw_i / mean(raw)         # normalize so mean = 1.0
w_i   = clamp(w_i, 0.25, 2.50)   # floor and ceiling
```

This ensures the training signal is always loudest on the dimension the agent most needs to improve.

---

## Seven Novel Loss Function Components

Standard GRPO uses a single terminal reward per episode. We decompose the reward into seven composable components, each targeting a specific training objective.

### 1. Temporal Credit Assignment

**Problem**: bad minute-5 decisions cause minute-45 conflicts. GRPO assigns no gradient to the causal action.

**Solution**: adaptive discounting across planning rounds:

```
γ = 1 - 1 / √(horizon_steps + 1)

horizon =  60 min  → γ = 0.50  (focus on terminal)
horizon = 180 min  → γ = 0.63  (early decisions count)
horizon = 360 min  → γ = 0.72  (shift-level planning)

G_t = Σ γ^(k-t) · δ_k  (discounted incremental improvements)
```

This is Ng et al. 1999 potential-based shaping — provably preserves the optimal policy while providing dense intermediate signal.

### 2. Hierarchical Decomposition Reward

**Problem**: all-or-nothing gradient — agents that get strategic priorities right but wrong slot minutes receive the same zero reward as agents that get everything wrong.

**Solution**: separate loss heads for each planning layer:

```
strategic  (0.25) → did the agent put emergency flights first?
tactical   (0.35) → did the agent achieve high coverage and window feasibility?
operational(0.40) → did the agent select precise, conflict-free, ATFM-compliant slots?

R_hier = 0.25 × R_strategic + 0.35 × R_tactical + 0.40 × R_operational
```

An agent that correctly prioritizes emergencies (strategic=1.0) but mis-times slots still gets 0.25 × 1.0 = 0.25 gradient from the strategic head. Learning doesn't stall.

### 3. Recovery Gradient

**Problem**: GRPO only rewards the final state. An agent that tanks round 1 and recovers gets the same terminal reward as one that was always good. No signal for the skill of *recovering from mistakes*.

**Solution**: explicit recovery bonus:

```
R_rec = max(0, score_final - score_initial) × recency_weight
      + conflict_resolution_bonus (if agent explicitly resolved conflicts)

Anti-gaming: if score_initial < 0.10, apply penalty proportional to suspicious gap
```

### 4. Contrastive Pair Reward

**Problem**: GRPO's group-relative advantage normalizes within the generation group. Agents that are uniformly bad get zero gradient.

**Solution**: contrastive comparison against the naive baseline:

```
R_contrastive = sigmoid(k × (actual_score - counterfactual_score)) × 2 - 1

actual >> naive  → R_contrastive → +1
actual ≈ naive   → R_contrastive → 0
actual << naive  → R_contrastive → -1
```

The sigmoid avoids the hard discontinuity of sign(Δ) while preserving directionality.

### 5. Information-Theoretic Coordination

**Problem**: agents send boilerplate messages ("I yield runway X") with no actual information about their constraints. This passes a format check but provides no coordination value.

**Solution**: reward messages that contain actionable features *and* correlate with positive outcome:

```
message_features = {mentions_emergency, mentions_flight_id, mentions_runway,
                    mentions_delay_cost, proposes_alternative,
                    mentions_wake_turbulence, theory_of_mind_claim}

R_info = feature_density × min(1.0, outcome_delta × 3.0) × 0.15
```

High only when messages are feature-rich AND the outcome improved — ensuring correlation between message quality and coordination success.

### 6. Causal Credit Assignment (Shapley)

**Problem**: both AMAN and DMAN contribute to the joint outcome. Giving each agent the same joint reward doesn't tell them which specific actions caused success.

**Solution**: approximate Shapley values with N=2 counterfactuals:

```
φ_AMAN = 0.5 × [V({AMAN}) - V(∅)] + 0.5 × [V({AMAN,DMAN}) - V({DMAN})]
φ_DMAN = 0.5 × [V({DMAN}) - V(∅)] + 0.5 × [V({AMAN,DMAN}) - V({AMAN})]
```

where V({AMAN}) = AMAN's actual plan + DMAN naive baseline, and V(∅) = both naive.

This satisfies the efficiency axiom: φ_AMAN + φ_DMAN ≈ V_joint - V_empty.

### 7. Adaptive KL Regularization

**Problem**: for whole-model fine-tuning (no LoRA), a fixed KL coefficient either over-regularizes (prevents learning) or under-regularizes (causes catastrophic forgetting).

**Solution**: KL coefficient adapts to the reward improvement rate:

```
improvement_rate = (mean_recent - mean_earlier) / |mean_earlier|

β_adaptive = β_base × max(β_floor, 1.0 - α × improvement_rate)

Fast improvement  → relax KL (agent is learning, let it move)
Plateau/decline   → tighten KL (stabilize, prevent drift)
β_floor = 0.001   (never zero — prevents mode collapse)
```

---

## Results

### Before vs. After GRPO Training

| Metric | Heuristic Baseline | GRPO-Trained |
|---|---:|---:|
| Composite score | ~0.47 | ~0.71 |
| Emergency handling | 61% on-time | 94% on-time |
| Conflict rate | 18% episodes | 4% episodes |
| ATFM compliance | 74% | 91% |
| Theory-of-mind bonuses | 0.08 avg | 0.34 avg |
| Curriculum target skill | N/A | Adapts per episode |

*Metrics from `training/train_grpo.py --run_eval` on 4-task evaluation set.*

### Heuristic Baseline (Single-Agent Grader)

| Task | Difficulty | Random | Heuristic | Δ |
|---|---|---|---|---|
| Delhi Monsoon Recovery | Easy | 0.21 | **0.9446** | +0.73 |
| Mumbai Hub Bank Balance | Medium | 0.18 | **0.9900** | +0.81 |
| Bengaluru IRROPS | Hard | 0.12 | **0.8615** | +0.74 |
| Hyderabad Cargo Crunch | Hard | 0.15 | **0.8576** | +0.71 |
| **Average** | | **0.165** | **0.9134** | **+0.748** |

Random agents score below 0.22 even on the easy task — the 3-layer gated grader requires passing all separation constraints before partial efficiency credit is awarded.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenEnv HTTP Surface                        │
│   POST /reset  │  POST /step  │  GET /state  │  GET /health     │
└──────────────────────────┬──────────────────────────────────────┘
                           │  ATCAction
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MultiAgentATCEnvironment                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │     AMAN     │  │     DMAN     │  │  ADAPT (domain   │   │
│  │ (arrivals    │  │ (departures  │  │  transfer)        │   │
│  │  only view)  │  │  only view)  │  │                   │   │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────┘   │
│         │                 │                                     │
│         └────── messages ──┘                                    │
│                    │                                            │
│  ┌─────────────────▼──────────────────────────────────────┐    │
│  │  BID → NEGOTIATE → FINAL  (partial observability)       │    │
│  └─────────────────┬──────────────────────────────────────┘    │
│                    │  slots                                     │
└────────────────────┼────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   simulate_plan()   │
          │   + graders.py      │
          │                     │
          │  Safety Gate ───────┤ conflict cap
          │  Priority Rubric    │ emergency cap
          │  Efficiency Rubric  │ coverage floor
          │  Long-Horizon Grade │ cascade management
          │  Recovery Grader    │ mistake recovery
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────────────────────────────────┐
          │            Reward Functions                      │
          │                                                  │
          │  aman_reward_fn    ← 7 loss components          │
          │  dman_reward_fn    ← temporal credit + TCA      │
          │  adapt_reward_fn   ← downstream composite       │
          └──────────┬──────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   ContextAdaptive   │
          │    Curriculum       │
          │                     │
          │  Diagnose weakness  │
          │  Select mutations   │
          │  Scale reward       │
          │  weights            │
          └─────────────────────┘
```

### Long-Horizon Mode

```
Task (6-hour horizon)
       │
       ▼
HierarchicalPlanDecomposer
  splits into 4 × 45-min epochs
       │
   ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
   │Epoch 0│→ │Epoch 1│→ │Epoch 2│→ │Epoch 3│
   └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
       │carry-over│ cascade   │ recovery  │
       └──────────┴───────────┴───────────┘
                                 │
                                 ▼
                         LongHorizonResult
                         aggregate_score
                         recovery_score
                         cascade_events
```

---

## ADAPT — Domain Transfer

When the same agents that learned ATC scheduling are thrown at a completely different domain — ICU surge management — they solve it without any retraining.

ADAPT reads *structural signals*, not domain labels:

| Signal | Weight | Description |
|---|---:|---|
| `time_pressure` | 0.50 | `1 − avg_window / planning_horizon` |
| `connection_risk` | 0.40 | Cascade failure probability |
| `urgency_in_notes` | 0.10 | Domain-agnostic English urgency markers |

```
ICU Mass Casualty Scenario:
  TRAUMA   → time_pressure=0.95  connection_risk=0.93  → score=0.94 → H / emergency
  CARDIAC  → time_pressure=0.78  connection_risk=0.70  → score=0.67 → M / medical
  POST_OP  → time_pressure=0.56  connection_risk=0.45  → score=0.46 → M / connection
  ROUTINE  → time_pressure=0.05  connection_risk=0.05  → score=0.05 → L / normal
```

ADAPT never sees the word "TRAUMA". It infers urgency from the numbers. The same formula that identifies a Heavy-priority aircraft identifies a trauma patient.

### Why This Matters

Most RL environments are domain-locked. Ours isn't. The structural abstraction (time pressure + cascade risk + urgency) is general enough to describe:

- Hospital beds during surge events
- Container berths at a port under weather delays  
- Factory assembly lanes during component shortage
- Power grid maintenance windows with demand spikes

ADAPT is the bridge. It requires no retraining of AMAN or DMAN to cross any of them.

---

## 3-Round Coordination Protocol

```
Episode start
     │
     ▼
Round 0: BID
  AMAN submits arrival slots (arrivals only — no DMAN visibility)
  DMAN submits departure slots (departures only — no AMAN visibility)
  Engine detects cross-runway conflicts
  If no conflicts: skip to FINAL
     │
     ▼
Round 1: NEGOTIATE
  Both agents receive conflict log + emergency broadcasts
  Agents send NegotiationMessages (typed: runway_claim, yield, request_gap,
                                         emergency_broadcast, theory_of_mind)
  Theory-of-mind bonus: agent predicts other agent's constraint → cited in message
     │
     ▼
Round 2: FINAL
  Merged plan → simulate_plan() → 3-layer grader → per-role rewards
  done=True; ATCObservation carries aman_reward, dman_reward, composite_score
```

---

## Scoring

### Three-Layer Gated Composite

```
score = min(gate_ceiling, 0.30 × priority_score + 0.70 × efficiency_score)
```

**Layer 1 — SafetyGateEvaluator** (hard ceilings):

| Violation | Effect |
|---|---|
| 1 separation conflict | ceiling = 0.40 |
| 2+ conflicts | ceiling -= 0.05 per additional (min 0.10) |
| Any EMERGENCY flight > 5 min late | ceiling = 0.35 |
| Missing flights | ceiling = max(0.20, 0.50 - 0.04 × missing) |

**Layer 2 — PriorityRubricGrader** (weight 0.30):

```
priority_score = 0.50 × emergency_score + 0.30 × medical_score + 0.20 × connection_score
```

**Layer 3 — EfficiencyRubricGrader** (weight 0.70):

```
efficiency_score = 0.35 × delay_efficiency + 0.25 × fuel_efficiency
                 + 0.20 × fairness + 0.20 × connection_impact_score
```

**New graders (auxiliary)**:
- `LongHorizonGrader` — consistency, cascade management, epoch decomposition quality
- `RecoveryGrader` — improvement from initial to final score with anti-gaming protection

---

## Reward Design

### AMAN Reward Components

| Component | Weight | Description |
|---|---:|---|
| `delay_efficiency` | 0.22 | 1 − total_delay / delay_budget |
| `emergency_score` | 0.18 | Fraction of EMERGENCY/MEDICAL arrivals ≤5 min |
| `coverage` | 0.15 | Fraction of arrivals assigned |
| `cf_advantage` (COMA) | 0.11 | Improvement over naive do-nothing baseline |
| `theory_of_mind` | 0.09 | Pre-emptive gap left for DMAN emergency |
| `hierarchical_bonus` | 0.08 | Strategic + tactical + operational layers |
| `recovery_bonus` | 0.06 | Recovery from round-1 baseline |
| `temporal_credit` | 0.06 | Adaptive discounted return across rounds |
| `rationale_score` | 0.04 | Constraint-aware explanation quality |
| `json_format` | 0.04 | Strict schema compliance |

### DMAN Reward Components

Similar to AMAN with:
- `atfm_compliance` (0.15): network slot deadline adherence
- `hierarchical_bonus` (0.10): includes ATFM in operational layer
- `temporal_credit` (0.07): tighter due to ATFM deadline pressure

### Safety Gates (cannot be offset)

| Gate | Condition | Effect |
|---|---|---|
| Conflict-free | `conflict_count > 0` | `reward = min(reward, 0.30)` |
| Emergency hard | `emg_miss > 0` | `reward = min(reward, 0.40)` |
| Coverage floor | `coverage < 0.50` | `reward -= 0.30` (floor at -0.50) |

---

## Tasks

| Task ID | Airport | Difficulty | Flights | Runways | Key Challenge |
|---|---|---|---:|---:|---|
| `delhi_monsoon_recovery_easy` | Delhi IGI | Easy | 12 | 2 | Monsoon disruption, VVIP slot, wake-spacing edge cases |
| `mumbai_bank_balance_medium` | Mumbai CSIA | Medium | 15 | 2 | Mixed cargo/pax hub bank under disruption |
| `bengaluru_irrops_hard` | Bengaluru KIA | Hard | 18 | 2 | Emergency arrival + medical departure + dual-runway IRROPS |
| `hyderabad_cargo_crunch_hard` | Hyderabad RGIA | Hard | 20 | 1 | Single-runway wake asymmetry, cargo priority |

**Domain Transfer**: 3 ICU surge scenarios via ADAPT (no retraining):
`icu_normal_day` · `icu_flu_surge` · `icu_mass_casualty`

---

## Wake Turbulence Separation Matrix

From `constants.py` — enforced on every slot assignment:

| Leader → Follower | Heavy | Medium | Light |
|---|---:|---:|---:|
| **Heavy** | 4 min | 5 min | 6 min |
| **Medium** | 3 min | 3 min | 4 min |
| **Light** | 3 min | 3 min | 3 min |

---

## Training Stack

### Standard Mode (LoRA, Colab T4)

```python
Model:       Qwen2.5-7B-Instruct
Quantization: 4-bit QLoRA  (load_in_4bit=True)
LoRA rank:   16  (q_proj, v_proj, k_proj, o_proj)
Batch size:  2,  gradient accumulation 4  → effective batch 8
Generations: N=4 per prompt  (minimum for stable GRPO advantage variance)
Max tokens:  512 per completion
LR:          5e-5
KL coeff:    0.0  (disabled — avoids ref_per_token_logps crash on Unsloth+PEFT)
Training:    ~200 episodes ≈ 800 samples ≈ 2 hr on T4
```

### Full-Model Mode (A100/H100)

```python
Quantization: None  (full precision)
LR:           5e-6  (10x lower than LoRA)
LLRD:         0.85  (layer-wise LR decay — early layers learn slower)
Grad clip:    0.5   (critical for stability without reference model KV)
KL coeff:     adaptive  (0.001 – 0.10, scales with reward improvement rate)
```

Layer-wise learning rate decay prevents catastrophic forgetting:
```
LR_layer_0 (embedding)  = 5e-6 × 0.85^0  = 5e-6
LR_layer_8 (mid)        = 5e-6 × 0.85^8  = 2.3e-6
LR_layer_32 (final)     = 5e-6 × 0.85^32 = 0.9e-7
```

### Advantage Computation

```
A_i = (r_i - mean(group)) / (std(group) + ε)

N=4 ensures non-degenerate std. N=2 collapses std → meaningless advantage.
```

---

## Repository Layout

| Path | Purpose |
|---|---|
| `models.py` | Single-agent data contracts and domain models |
| `engine.py` | Deterministic simulation: wake separation, ATFM, delay, fuel |
| `graders.py` | Composite + coordination + **LongHorizonGrader** + **RecoveryGrader** |
| `tasks.py` | 4-scenario ATC catalog with task briefing generation |
| `planner.py` | Deterministic heuristic baseline planner |
| `constants.py` | Wake separation matrix, scoring weights, shared constants |
| `multi_agent/adapter.py` | **ContextAdaptiveCurriculum** — self-adapting, skill-targeted curriculum |
| `multi_agent/environment.py` | MultiAgentATCEnvironment (BID/NEGOTIATE/FINAL protocol) |
| `multi_agent/adapt.py` | ADAPT meta-agent — structural domain transfer |
| `multi_agent/inference.py` | Multi-agent heuristic/LLM episode runner |
| `training/loss_functions.py` | **7 novel loss components**: TCA, hierarchical, recovery, contrastive, ITC, causal, adaptive KL |
| `training/long_horizon.py` | **Long-horizon utilities**: EpisodeMemory, HierarchicalPlanDecomposer, CascadeDetector |
| `training/reward_functions.py` | GRPO reward functions integrating all loss components |
| `training/train_grpo.py` | Training entry point (LoRA + full-model + long-horizon modes) |
| `training/dataset.py` | GRPO dataset builder and output parsers |
| `training/eval.py` | Before/after evaluation |
| `domains/icu.py` | ICU surge management — ADAPT transfer demo |
| `server/app.py` | FastAPI/OpenEnv HTTP server |
| `tests/` | Automated tests: single-agent, multi-agent, ADAPT, graders |

---

## Setup

```bash
pip install uv
uv sync --extra dev          # core + tests
```

Training extras (GPU required):

```bash
uv sync --extra dev --extra training     # unsloth, trl, torch
```

---

## Running the Environment

### Start the Server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Validate

```bash
python -m openenv.cli validate .
python -m pytest -q
python scripts/run_graders.py
```

### Multi-Agent Heuristic Baseline

```bash
python multi_agent/inference.py --all_tasks --episodes 1
```

### Train — Standard (Colab T4, QLoRA)

```bash
python training/train_grpo.py --episodes 200 --output_dir ./outputs/atc-grpo
```

### Train — Long-Horizon (Theme #2)

```bash
python training/train_grpo.py --long_horizon --episodes 200 --output_dir ./outputs/long-horizon
```

### Train — Full Model (A100/H100)

```bash
python training/train_grpo.py --full_model --episodes 500 --output_dir ./outputs/full-model
```

### Colab Quick Start

Open `training/atc_multiagent_colab.ipynb` — installs Unsloth + TRL, mounts environment, runs 200 training episodes, prints before/after comparison.

### Evaluate a Trained Checkpoint

```bash
python training/eval.py --base heuristic-baseline --trained ./outputs/atc-multiagent --episodes 10
```

---

## Design Decisions

**Self-adapting over self-improving**: Blind EMA escalation wastes episodes on tasks the agent already handles. Diagnostic skill profiling targets the actual gap — faster learning, more interpretable curriculum.

**Seven loss components, not one**: A single terminal reward gives no gradient to the decisions that mattered. Each component provides a distinct learning signal: temporal (when), hierarchical (what level), recovery (after mistakes), contrastive (vs. baseline), informational (message quality), causal (who contributed), adaptive (how much to regularize).

**Verifiable rewards**: Every score is computed from physics (wake separation, ATFM deadlines, delay budgets). No LLM judge in the reward loop means no hallucination, no inconsistency, no reward hacking via prompt injection.

**Partial observability is structural**: AMAN receives `atfm_deadlines={}`. DMAN receives the real deadline map. Neither can cheat. Information crosses only through the negotiation message channel — testing genuine coordination, not informed cooperation.

**Safety gates are absolute**: A plan with conflicts is rejected, not averaged. Emergency violations cap the score below 0.40 regardless of how efficient the rest of the plan is. Safety cannot be bought off by throughput.

**GRPO over PPO**: no value network required. Critical for Colab T4 with a 7B model and five roles in the same training loop. N=4 generation groups provide stable advantage estimation without a learned baseline.

**Single model, multiple roles**: AMAN and DMAN are system-prompt-differentiated instances of the same weights. This tests whether one model can reason from asymmetric information frames — not whether two separately-tuned models can each individually perform well.

---

## Docker

```bash
docker build -t atc-openenv .
docker run --rm -p 8000:8000 atc-openenv
```

---

## HuggingFace Space Deployment

```bash
export HF_TOKEN="hf_xxx"
export HF_SPACE_ID="<owner>/<space-name>"
python scripts/deploy_hf_space.py --space-id "$HF_SPACE_ID" --repo-dir .
```

Set secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.