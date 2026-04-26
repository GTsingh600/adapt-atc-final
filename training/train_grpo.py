"""ADAPT ATC — GRPO training with Unsloth.

Agents: AMAN (arrivals) · DMAN (departures) · ADAPT (domain transfer)
All three roles share one LLM differentiated by system prompts.

Usage:
  python training/train_grpo.py                        # 150 episodes, full dataset
  python training/train_grpo.py --easy --episodes 80   # AMAN+DMAN only, fast ~1 h
  python training/train_grpo.py --episodes 300 --n_gen 4  # A100 full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import build_episode_dataset
from training.reward_functions import aman_reward_fn, dman_reward_fn, adapt_reward_fn
from multi_agent.models import AgentRole

# ── Hyperparameters ───────────────────────────────────────────────────────────
#   T4  (16 GB fp16): episodes=150, n_gen=2, batch=1, accum=4  → ~2 h
#   A100 (40 GB bf16): episodes=300, n_gen=4, batch=4, accum=2  → ~2–3 h

MODEL      = "unsloth/Qwen2.5-1.5B-Instruct"
LORA_RANK  = 8        # ~0.4% trainable params — fast convergence for 1.5B
MAX_SEQ    = 2048     # prompt 1792 + completion 256
MAX_TOKENS = 256      # completion window
N_GEN      = 2        # GRPO group size (min 2 for non-degenerate std)
# IMPORTANT: per_device_train_batch_size MUST be divisible by num_generations.
# TRL enforces this hard; batch=2, n_gen=2 satisfies the constraint on T4.
BATCH      = 2        # per-device batch (must be >= N_GEN and divisible by N_GEN)
ACCUM      = 4        # gradient accumulation → effective batch = 8
LR         = 5e-6
EPOCHS     = 3
WARMUP     = 0.10

# ── Reward dispatcher ─────────────────────────────────────────────────────────

_DISPATCH: Dict[str, Any] = {
    AgentRole.AMAN.value:  aman_reward_fn,
    AgentRole.DMAN.value:  dman_reward_fn,
    AgentRole.ADAPT.value: adapt_reward_fn,
}


def _partial(text: str, role: str) -> float:
    """Partial credit when JSON parse fails — keeps reward_std > 0 during cold start."""
    s = 0.05
    if "{" in text: s += 0.03
    if role == "AMAN"  and "arrival_slots"   in text: s += 0.06
    if role == "DMAN"  and "departure_slots" in text: s += 0.06
    if role == "ADAPT" and "entity_wake_map" in text: s += 0.06
    if '"flight_id"' in text: s += 0.04
    if "rationale"    in text: s += 0.02
    return min(s, 0.22)


def reward_fn(completions: List, **kwargs) -> List[float]:
    """Route each completion to its role-specific composable rubric.

    Returns List[float] in [0, 1] — guaranteed plain Python floats, never tensors.
    TRL GRPOTrainer calls this after generating num_generations completions per prompt.
    """
    n     = len(completions)
    roles = kwargs.get("agent_role", ["AMAN"] * n)
    if not isinstance(roles, list):
        roles = [roles] * n
    while len(roles) < n:
        roles.append(roles[-1])

    rewards: List[float] = []
    for i, (comp, role) in enumerate(zip(completions, roles)):
        # TRL may pass completion as list-of-dicts (conversational format)
        if isinstance(comp, list):
            comp = comp[-1].get("content", "") if comp else ""
        comp = str(comp)

        fn = _DISPATCH.get(str(role), aman_reward_fn)

        # Slice kwargs to a single-item list (reward fns expect List[...])
        kw = {}
        for k, v in kwargs.items():
            if k == "agent_role":
                continue
            if isinstance(v, list):
                kw[k] = [v[i] if i < len(v) else (v[-1] if v else "")]
            else:
                kw[k] = [v]

        try:
            r      = fn([comp], **kw)
            reward = float(r[0]) if r else _partial(comp, str(role))
            reward = max(0.0, min(1.0, reward))
            if reward != reward:   # NaN guard
                reward = _partial(comp, str(role))
        except Exception:
            reward = _partial(comp, str(role))

        rewards.append(float(reward))

    return rewards


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model_name: str  = MODEL,
    output_dir: str  = "./outputs/atc-grpo",
    n_episodes: int  = 150,
    lora_rank:  int  = LORA_RANK,
    easy:       bool = False,
    n_gen:      int  = N_GEN,
) -> None:
    """Run GRPO training.

    Args:
        easy: When True, trains AMAN+DMAN only (no ADAPT domain transfer).
              Converges faster — good for a first run or quick hackathon demo.
        n_gen: Generations per prompt. T4 → 2, A100 → 4.
    """
    import torch
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError:
        raise ImportError("pip install unsloth")
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    # ── Patch GRPO generation hooks before model load ─────────────────────────
    try:
        PatchFastRL("GRPO", FastLanguageModel)
        print("[ok] PatchFastRL")
    except Exception as e:
        print(f"[note] PatchFastRL: {e}")

    IS_BF16 = torch.cuda.is_bf16_supported()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*54}")
    print(f"  ADAPT ATC GRPO Training")
    print(f"  model    : {model_name}")
    print(f"  gpu      : {gpu_name}")
    print(f"  dtype    : {'bf16 (A100)' if IS_BF16 else 'fp16 (T4)'}")
    print(f"  episodes : {n_episodes}  easy={easy}")
    print(f"  n_gen    : {n_gen}")
    print(f"  output   : {output_dir}")
    print(f"{'='*54}\n")

    # ── Load model — 4-bit QLoRA via Unsloth ─────────────────────────────────
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,             # auto-detect: fp16 on T4, bf16 on A100
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── LoRA ─────────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",   # Unsloth memory-efficient variant
        random_state=42,
    )
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    to = sum(p.numel() for p in model.parameters())
    print(f"[ok] LoRA rank={lora_rank}: {tr:,}/{to:,} trainable ({100*tr/to:.2f}%)\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Building dataset ({n_episodes} episodes, easy={easy})...")
    raw = build_episode_dataset(
        n_episodes=n_episodes,
        seed=42,
        include_adapt=not easy,
        domain_episode_ratio=0.65,
        long_horizon_ratio=0.0,
    )
    dataset = Dataset.from_list(raw)

    from collections import Counter
    counts = Counter(s["agent_role"] for s in raw)
    for role, cnt in sorted(counts.items()):
        print(f"  {role:6s}: {cnt}")
    print(f"  total : {len(dataset)}\n")

    # ── GRPO config ───────────────────────────────────────────────────────────
    # TRL hard requirement: per_device_train_batch_size % num_generations == 0
    # Compute the smallest batch_size >= N_GEN that satisfies divisibility.
    import inspect as _inspect
    batch_actual = max(BATCH, n_gen)
    if batch_actual % n_gen != 0:
        batch_actual = n_gen   # fallback: exactly one group per step

    grpo_fields = _inspect.signature(GRPOConfig.__init__).parameters

    cfg_kwargs: Dict[str, Any] = {
        # Generation
        "num_generations":              n_gen,
        "max_completion_length":        MAX_TOKENS,

        # Optimisation
        "learning_rate":                LR,
        "per_device_train_batch_size":  batch_actual,
        "gradient_accumulation_steps":  ACCUM,
        "num_train_epochs":             EPOCHS,
        "warmup_ratio":                 WARMUP,
        "max_grad_norm":                1.0,

        # Precision — explicit, never auto-detect
        "bf16":  IS_BF16,          # A100 (Ampere) → bf16
        "fp16":  not IS_BF16,      # T4  (Turing)  → fp16

        # Memory
        "optim":                 "adamw_8bit",
        "gradient_checkpointing": True,

        # KL — 0.0 avoids ref_per_token_logps=None crash with PEFT
        "beta":  0.0,

        # Logging
        "output_dir":       output_dir,
        "logging_steps":    1,
        "save_steps":       50,
        "save_total_limit": 2,
        "report_to":        "none",
        "seed":             42,
    }

    # max_prompt_length: present in TRL >= 0.15
    if "max_prompt_length" in grpo_fields:
        cfg_kwargs["max_prompt_length"] = MAX_SEQ - MAX_TOKENS

    config = GRPOConfig(**cfg_kwargs)
    print(f"[ok] batch={batch_actual}  n_gen={n_gen}  accum={ACCUM}  "
          f"dtype={'bf16' if IS_BF16 else 'fp16'}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # GRPOTrainer API changed across TRL versions:
    #   TRL < 0.17:  tokenizer=..., config=...
    #   TRL >= 0.17: processing_class=..., args=...
    trainer_sig = _inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model":          model,
        "reward_funcs":   [reward_fn],
        "train_dataset":  dataset,
    }
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["args"]             = config
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        trainer_kwargs["config" if "config" in trainer_sig else "args"] = config

    trainer = GRPOTrainer(**trainer_kwargs)

    print("Training...\n")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    vals: List[float] = []
    for entry in trainer.state.log_history:
        for k in ["rewards/reward_fn", "reward", "rewards/combined_reward_fn",
                  "rewards/reward_fn_mean"]:
            if k in entry and isinstance(entry[k], (int, float)):
                vals.append(float(entry[k]))
                break

    print(f"\n{'='*54}")
    print(f"  Done in {elapsed/60:.0f} min")
    if vals:
        q = max(1, len(vals) // 4)
        f_, l_ = sum(vals[:q]) / q, sum(vals[-q:]) / q
        arrow = "↑" if l_ > f_ + 0.02 else ("↓" if l_ < f_ - 0.02 else "→")
        print(f"  Reward: {f_:.3f} → {l_:.3f}  ({l_-f_:+.3f} {arrow})")
    print(f"  Saved : {output_dir}")
    print(f"{'='*54}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ADAPT ATC GRPO Training")
    p.add_argument("--model",      default=MODEL,
                   help="HF model name or local path (default: unsloth/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--output_dir", default="./outputs/atc-grpo")
    p.add_argument("--episodes",   type=int, default=150,
                   help="Training episodes. 80=fast (~1 h T4), 150=good, 300=best")
    p.add_argument("--lora_rank",  type=int, default=LORA_RANK,
                   help="LoRA rank. 8 for T4, 16 for A100")
    p.add_argument("--n_gen",      type=int, default=N_GEN,
                   help="GRPO generations per prompt. 2 for T4, 4 for A100")
    p.add_argument("--easy",       action="store_true",
                   help="Train AMAN+DMAN only (skip ADAPT). Faster convergence, good first run.")
    args = p.parse_args()

    train(
        model_name=args.model,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        lora_rank=args.lora_rank,
        easy=args.easy,
        n_gen=args.n_gen,
    )


if __name__ == "__main__":
    main()
