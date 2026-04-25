"""ADAPT-Focused Multi-Agent ATC GRPO Training with Unsloth.

Architecture:
  - Single LLM plays 3 roles via system prompts: ADAPT (primary), AMAN, DMAN
  - GRPO: group-relative advantage  A_i = (r_i - mean(group)) / (std(group) + eps)
  - ADAPT: domain transfer agent — maps unknown domains to ATC parameters
  - AMAN/DMAN: support roles for JSON format learning and scheduling vocabulary
  - Simplified 5-component rewards per role for stable convergence
  - Per-role reward curves saved to reward_curves.json for demo

Training mix:
  ~50% domain-transfer (ADAPT) episodes — primary training signal
  ~25% AMAN episodes — JSON format + arrival scheduling
  ~25% DMAN episodes — JSON format + departure scheduling

Usage:
  python training/train_grpo.py [--episodes 200] [--model Qwen/Qwen2.5-1.5B-Instruct]
  python training/train_grpo.py --model Qwen/Qwen2.5-7B-Instruct --episodes 300

Colab one-liner (LoRA):
  !python training/train_grpo.py --episodes 100 --output_dir /content/atc-multiagent
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_training_deps():
    if sys.version_info >= (3, 14):
        print("[ERROR] Python 3.14 not supported. Use 3.11 or 3.12.")
        sys.exit(1)
    try:
        import torch
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install torch")
        sys.exit(1)
    try:
        # Unsloth should be imported before TRL/Transformers/PEFT.
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as e:
        print(f"[ERROR] unsloth import failed: {e}")
        print("Install: pip install unsloth unsloth-zoo")
        sys.exit(1)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install trl transformers")
        sys.exit(1)
    return torch, FastLanguageModel, GRPOConfig, GRPOTrainer


from training.dataset import (
    build_episode_dataset,
    parse_aman_action,
    parse_dman_action,
)
from training.reward_functions import (
    aman_reward_fn,
    dman_reward_fn,
    adapt_reward_fn,
)
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.models import AgentRole, SupervisorProfileName
from multi_agent.supervisor import SupervisorAgent
from tasks import task_catalog, ordered_tasks


# ── Hyperparameters — ADAPT-focused, 1.5B/7B-stable ──────────────────────────
#
# Tuned for stable convergence on T4 Colab with Qwen2.5-1.5B to 7B.
# ADAPT is the primary training role. AMAN/DMAN are support roles.
# Lower LR + higher grad_accum = more stable updates.

DEFAULT_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT = "./outputs/atc-adapt"

# QLoRA config — all 7 projection types, moderate rank
LORA_RANK    = 32
LORA_ALPHA   = 64   # 2x rank keeps adapter scale stable
LORA_TARGETS = [
    "q_proj", "v_proj", "k_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",       # FFN / MLP
]

MAX_SEQ_LEN    = 1536   # save VRAM for batch size headroom
MAX_NEW_TOKENS = 384    # ADAPT outputs ~150 tokens
TEMPERATURE    = 0.8    # slightly more exploration
N_GENERATIONS  = 2      # T4 practical max
BATCH_SIZE     = 2
GRAD_ACCUM     = 8      # effective batch = 16 for stable gradients
LR             = 5e-6   # low LR prevents reward collapse — key fix
KL_COEFF       = 0.0    # keep 0.0 — non-zero KL crashes Unsloth+PEFT
WARMUP_RATIO   = 0.10   # longer warmup for early gradient stability
SAVE_STEPS     = 50
SAVE_TOTAL_LIMIT = 3


# ── Role-dispatch table (ADAPT-focused: only 3 roles) ────────────────────────

REWARD_FN_DISPATCH = {
    AgentRole.AMAN.value:  aman_reward_fn,
    AgentRole.DMAN.value:  dman_reward_fn,
    AgentRole.ADAPT.value: adapt_reward_fn,   # primary training role
}


def _reward_failure_mode() -> str:
    mode = os.getenv("REWARD_FAILURE_MODE", "strict").strip().lower()
    return mode if mode in {"strict", "penalize"} else "strict"


def _config_supports(param: str, config_cls) -> bool:
    try:
        return param in inspect.signature(config_cls.__init__).parameters
    except Exception:
        return False


def _trainer_supports(param: str, trainer_cls) -> bool:
    try:
        return param in inspect.signature(trainer_cls.__init__).parameters
    except Exception:
        return False


def _maybe_patch_trainer_sampler(trainer) -> None:
    """Handle TRL/Transformers sampler signature drift across versions."""
    try:
        sampler = getattr(type(trainer), "_get_train_sampler", None)
        if sampler is None:
            return
        # Old TRL versions expose _get_train_sampler(self) while newer
        # Transformers call sampler_fn(dataset). Patch only that old form.
        if len(inspect.signature(sampler).parameters) == 1:
            from types import MethodType

            original = trainer._get_train_sampler

            def _compat_get_train_sampler(self, train_dataset=None):
                return original()

            trainer._get_train_sampler = MethodType(_compat_get_train_sampler, trainer)
            print("[WARN] Applied sampler compatibility shim for this TRL/Transformers pair.")
    except Exception as exc:
        print(f"[WARN] Could not apply sampler compatibility shim: {exc}")


def _maybe_patch_unsloth_grad_accum(trainer) -> None:
    """Provide missing attribute expected by some Unsloth GRPO trainer builds."""
    if hasattr(trainer, "current_gradient_accumulation_steps"):
        return
    steps = 1
    try:
        args = getattr(trainer, "args", None)
        if args is not None:
            steps = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    except Exception:
        steps = 1
    trainer.current_gradient_accumulation_steps = steps
    print(
        "[WARN] Applied Unsloth GRPO compatibility shim: "
        f"current_gradient_accumulation_steps={steps}"
    )


def _maybe_patch_unsloth_loss_type(trainer) -> None:
    """Ensure loss_type exists for Unsloth/TRL compatibility.

    Some compiled trainer paths branch on hasattr(self.args, "loss_type").
    If missing, they may take an older unpack path that mismatches newer
    return signatures and crashes with "too many values to unpack".
    """
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return
        if getattr(args, "loss_type", None) is None:
            setattr(args, "loss_type", "grpo")
            print("[WARN] Applied Unsloth GRPO compatibility shim: loss_type='grpo'")
    except Exception as exc:
        print(f"[WARN] Could not apply loss_type compatibility shim: {exc}")


def _maybe_patch_unsloth_runtime_attrs(trainer) -> None:
    """Backfill runtime attrs expected by some Unsloth compiled trainer builds."""
    try:
        args = getattr(trainer, "args", None)

        if not hasattr(trainer, "importance_sampling_level"):
            level = "token"
            if args is not None:
                level = getattr(args, "importance_sampling_level", level) or level
            setattr(trainer, "importance_sampling_level", level)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"importance_sampling_level={level!r}"
            )

        if not hasattr(trainer, "epsilon_low"):
            epsilon_low = 0.2
            if args is not None:
                epsilon_low = float(getattr(args, "epsilon", epsilon_low) or epsilon_low)
            setattr(trainer, "epsilon_low", epsilon_low)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_low={epsilon_low}"
            )

        if not hasattr(trainer, "epsilon_high"):
            epsilon_high = getattr(trainer, "epsilon_low", 0.2)
            if args is not None:
                epsilon_high = float(
                    getattr(args, "epsilon_high", epsilon_high) or epsilon_high
                )
            setattr(trainer, "epsilon_high", epsilon_high)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_high={epsilon_high}"
            )

        if not hasattr(trainer, "vllm_importance_sampling_cap"):
            cap = 2.0
            if args is not None:
                cap = float(
                    getattr(args, "vllm_importance_sampling_cap", cap) or cap
                )
            setattr(trainer, "vllm_importance_sampling_cap", cap)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"vllm_importance_sampling_cap={cap}"
            )

        if not hasattr(trainer, "loss_type"):
            loss_type = "grpo"
            if args is not None:
                loss_type = getattr(args, "loss_type", loss_type) or loss_type
            setattr(trainer, "loss_type", loss_type)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"loss_type={loss_type!r}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply runtime attr compatibility shim: {exc}")


def _maybe_patch_unsloth_args_attrs(trainer) -> None:
    """Backfill args fields expected by some Unsloth compiled trainer paths."""
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return

        if not hasattr(args, "delta"):
            setattr(args, "delta", None)
            print("[WARN] Applied Unsloth GRPO compatibility shim: args.delta=None")

        if not hasattr(args, "temperature"):
            setattr(args, "temperature", TEMPERATURE)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.temperature={TEMPERATURE}"
            )

        if not hasattr(args, "max_completion_length"):
            setattr(args, "max_completion_length", MAX_NEW_TOKENS)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.max_completion_length={MAX_NEW_TOKENS}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply args attr compatibility shim: {exc}")


def _maybe_patch_nanmin_symbols() -> None:
    """Provide nanmin/nanmax symbols expected by some generated trainer code."""
    try:
        import builtins
        import torch as _torch

        def _compat_nanmin(x, *args, **kwargs):
            if hasattr(_torch, "nanmin"):
                return _torch.nanmin(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("inf"))
            if args or kwargs:
                return _torch.amin(x2, *args, **kwargs)
            return _torch.min(x2)

        def _compat_nanmax(x, *args, **kwargs):
            if hasattr(_torch, "nanmax"):
                return _torch.nanmax(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("-inf"))
            if args or kwargs:
                return _torch.amax(x2, *args, **kwargs)
            return _torch.max(x2)

        if not hasattr(builtins, "nanmin"):
            builtins.nanmin = _compat_nanmin
            print("[WARN] Applied compatibility shim: builtins.nanmin")
        if not hasattr(builtins, "nanmax"):
            builtins.nanmax = _compat_nanmax
            print("[WARN] Applied compatibility shim: builtins.nanmax")
    except Exception as exc:
        print(f"[WARN] Could not apply nanmin/nanmax compatibility shim: {exc}")


def _resolve_num_generations(batch_size: int, requested: int) -> int:
    requested = max(1, requested)
    for candidate in range(min(requested, batch_size), 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


def _effective_kl_coeff() -> float:
    raw = os.getenv("ATC_KL_COEFF", str(KL_COEFF)).strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid ATC_KL_COEFF={raw!r}; using default {KL_COEFF}.")
        return KL_COEFF

    if value < 0.0:
        print(f"[WARN] Negative KL coeff {value} is invalid; clamping to 0.0.")
        value = 0.0

    if value > 0.0:
        print(
            "[WARN] Non-zero KL enabled. On trl==0.16.0 + unsloth==2026.4.7 this may "
            "trigger ref_per_token_logps=None errors with PEFT."
        )
    return value


def _select_sample_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[index] if index < len(value) else value[-1]
    return value


# ── Unified reward dispatcher ─────────────────────────────────────────────────

def combined_reward_fn(completions: List[str], **kwargs) -> List[float]:
    """Route each completion to its role-specific reward function.

    TRL GRPOTrainer calls this with a batch of completions.
    kwargs contains per-sample metadata from the dataset.
    """
    roles = kwargs.get("agent_role", [AgentRole.AMAN.value] * len(completions))
    if not isinstance(roles, list):
        roles = [roles] * len(completions)
    elif len(roles) < len(completions):
        roles = roles + [roles[-1] if roles else AgentRole.AMAN.value] * (
            len(completions) - len(roles)
        )

    rewards: List[float] = []
    failure_mode = _reward_failure_mode()

    for i, (completion, role) in enumerate(zip(completions, roles)):
        fn = REWARD_FN_DISPATCH.get(role, aman_reward_fn)
        sample_kwargs = {k: [_select_sample_value(v, i)] for k, v in kwargs.items()}
        try:
            r = fn([completion], **sample_kwargs)
            if not r:
                raise RuntimeError(f"empty reward list for role={role}")
            rewards.append(r[0])
        except Exception as exc:
            msg = f"reward_fn({role}) failed at index={i}: {exc}"
            if failure_mode == "strict":
                raise RuntimeError(msg) from exc
            print(f"[WARN] {msg}")
            rewards.append(-1.0)

    return rewards


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    model_name:   str  = DEFAULT_MODEL,
    output_dir:   str  = DEFAULT_OUTPUT,
    n_episodes:   int  = 200,
    lora_rank:    int  = LORA_RANK,
    seed:         int  = 42,
    push_to_hub:  bool = False,
    hub_model_id: Optional[str] = None,
    run_eval:     bool = True,
) -> None:
    torch, FastLanguageModel, GRPOConfig, GRPOTrainer = _require_training_deps()

    num_generations = _resolve_num_generations(BATCH_SIZE, N_GENERATIONS)
    if num_generations != N_GENERATIONS:
        print(
            f"[WARN] Adjusted num_generations {N_GENERATIONS} -> {num_generations} "
            f"to satisfy GRPO batch-size divisibility constraint."
        )

    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n{'='*60}")
    print(f"  ATC Multi-Agent GRPO Training")
    print(f"  Model:         {model_name}")
    print(f"  Mode:          QLoRA rank={lora_rank} (all 7 projection types)")
    print(f"  Focus:         ADAPT domain transfer (~50% of samples)")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Generations:   {num_generations} per prompt")
    print(f"  LR:            {LR}")
    print(f"  Output:        {output_dir}")
    print(f"  Device:        {device_str}")
    print(f"{'='*60}\n")

    # ── 1. Capture pre-training baseline metrics ──────────────────────────────
    if run_eval:
        print("[0/5] Capturing pre-training baseline metrics...")
        baseline = _quick_heuristic_eval(n_episodes=min(10, n_episodes))
        _save_json(baseline, Path(output_dir) / "baseline_metrics.json")
        print(f"    Baseline composite: {baseline['mean_composite']:.3f}")

    # ── 2. Load model — 4-bit QLoRA, all 7 projection types ─────────────────
    # Targeting q/k/v/o + gate/up/down covers ~85% of transformer params.
    # This is the maximum expressiveness achievable inside Unsloth's stable
    # compiled GRPO path — raw full-model (no PEFT) crashes torch._dynamo.
    print(f"[1/5] Loading model with Unsloth 4-bit QLoRA (rank={lora_rank}, all projections)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,   # 2x rank keeps adapter scale stable
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100.0 * trainable / max(1, total)
    print(f"    rank={lora_rank}, targets=7 projections, trainable={trainable:,} ({pct:.1f}%)")

    # ── 2b. Base model eval (before any gradient steps) ──────────────────────
    base_model_metrics: Optional[Dict[str, Any]] = None
    if run_eval:
        print("\n[1.5/5] Measuring base model score (untrained LoRA)...")
        model.eval()
        base_model_metrics = _run_model_episodes(
            model, tokenizer, n_episodes=3, tag="BASE MODEL (no fine-tune)"
        )
        model.train()
        _save_json(base_model_metrics, Path(output_dir) / "base_model_metrics.json")
        print(f"    Base model composite: {base_model_metrics['mean_composite']:.3f}"
              f"  (AMAN {base_model_metrics['mean_aman_reward']:.3f}"
              f" / DMAN {base_model_metrics['mean_dman_reward']:.3f})")

    # ── 3. Build ADAPT-focused training dataset ────────────────────────────────
    print(f"\n[2/5] Building {n_episodes}-episode ADAPT-focused dataset...")
    print(f"      ADAPT domain ratio=0.50  |  Generator=OFF  |  Supervisor=OFF")
    t0 = time.time()
    dataset_raw = build_episode_dataset(
        n_episodes=n_episodes,
        seed=seed,
        include_generator=False,
        include_supervisor=False,
        include_adapt=True,
        domain_episode_ratio=0.50,      # 50% domain-transfer ADAPT episodes
        long_horizon_ratio=0.0,         # disabled for stable convergence
    )
    print(f"    Dataset: {len(dataset_raw)} samples ({time.time()-t0:.1f}s)")

    role_counts: Dict[str, int] = {}
    for s in dataset_raw:
        r = s.get("agent_role", "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count} samples")

    try:
        from datasets import Dataset
        dataset = Dataset.from_list(dataset_raw)
    except ImportError:
        print("[ERROR] pip install datasets")
        sys.exit(1)

    # ── 4. GRPO config ────────────────────────────────────────────────────────
    kl_coeff = _effective_kl_coeff()
    print(
        f"\n[3/5] Configuring GRPO (group_size={num_generations}, lr={LR}, kl={kl_coeff})..."
    )
    grpo_kwargs: Dict[str, Any] = {
        "num_generations":              num_generations,
        "temperature":                  TEMPERATURE,
        "learning_rate":                LR,
        "per_device_train_batch_size":  BATCH_SIZE,
        "gradient_accumulation_steps":  GRAD_ACCUM,
        "num_train_epochs":             1,
        "warmup_ratio":                 WARMUP_RATIO,
        "logging_steps":                10,
        "save_steps":                   SAVE_STEPS,
        "save_total_limit":             SAVE_TOTAL_LIMIT,
        "output_dir":                   output_dir,
        "run_name":                     f"atc-multiagent-grpo-{int(time.time())}",
        "bf16":                         torch.cuda.is_bf16_supported(),
        "fp16":                         not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing":       True,
        "optim":                        "paged_adamw_8bit",
    }

    if _wandb_available():
        grpo_kwargs["report_to"] = "wandb"
    else:
        grpo_kwargs["report_to"] = "none"

    # Compatibility shims for different TRL versions
    if _config_supports("max_completion_length", GRPOConfig):
        grpo_kwargs["max_completion_length"] = MAX_NEW_TOKENS
    elif _config_supports("max_new_tokens", GRPOConfig):
        grpo_kwargs["max_new_tokens"] = MAX_NEW_TOKENS

    if _config_supports("beta", GRPOConfig):
        grpo_kwargs["beta"] = kl_coeff
    elif _config_supports("kl_coeff", GRPOConfig):
        grpo_kwargs["kl_coeff"] = kl_coeff

    if _config_supports("use_vllm", GRPOConfig):
        grpo_kwargs["use_vllm"] = False

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── 5. Per-role reward logger ─────────────────────────────────────────────
    print("\n[4/5] Setting up per-role reward logging...")

    # Separate lists so we can show per-role curves in the demo
    reward_log: Dict[str, List[float]] = {
        "ADAPT": [], "AMAN": [], "DMAN": [], "composite": []
    }

    class RewardLogger:
        __name__ = "combined_reward_fn"

        def __call__(self, *args, **kwargs):
            # TRL <0.17: reward_func(completions=..., **kwargs)
            # TRL >=0.17: reward_func(prompts, completions, **kwargs)
            if "completions" in kwargs:
                completions = kwargs.pop("completions")
            elif len(args) >= 2:
                completions = args[1]
            elif args:
                completions = args[0]
            else:
                completions = []
            kwargs.pop("prompts", None)
            kwargs.pop("prompt_ids", None)

            # Some TRL versions pass conversational turns. Extract the assistant text.
            if completions and isinstance(completions[0], list):
                flattened = []
                for c in completions:
                    if c and isinstance(c[-1], dict) and "content" in c[-1]:
                        flattened.append(c[-1]["content"])
                    else:
                        flattened.append(str(c))
                completions = flattened

            rewards = combined_reward_fn(completions, **kwargs)

            roles = kwargs.get("agent_role", [])
            if not isinstance(roles, list):
                roles = [roles] * len(rewards)
            elif len(roles) < len(rewards):
                roles = roles + [
                    roles[-1] if roles else AgentRole.AMAN.value
                ] * (len(rewards) - len(roles))

            for r, role in zip(rewards, roles):
                if role in reward_log:
                    reward_log[role].append(r)
                reward_log["composite"].append(r)

            # ── Parseable log line every 20 reward calls ──────────────────
            # Notebook parses [ATC_REWARD step=N] lines for live display.
            call_count = len(reward_log["composite"])
            if call_count % 20 == 0 and call_count > 0:
                tail = 10
                role_parts = []
                for _role in ("ADAPT", "AMAN", "DMAN"):
                    rs = reward_log.get(_role, [])
                    if rs:
                        mean_r = sum(rs[-tail:]) / min(tail, len(rs))
                        role_parts.append(f"{_role}={mean_r:.4f}")
                comp = reward_log["composite"]
                comp_mean = sum(comp[-tail:]) / min(tail, len(comp)) if comp else 0.0
                role_parts.append(f"composite={comp_mean:.4f}")
                print(f"[ATC_REWARD step={call_count}] {' '.join(role_parts)}", flush=True)

            # Reward-hacking detection: warn when composite rises but per-role variance
            # collapses (all roles getting same score = likely gaming)
            if call_count % 50 == 0 and call_count > 50:
                _check_reward_hacking(reward_log)

            return rewards

    reward_logger = RewardLogger()

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    trainer_kwargs: Dict[str, Any] = {
        "model":            model,
        "processing_class": tokenizer,
        "reward_funcs":     [reward_logger],
        "train_dataset":    dataset,
    }
    if _trainer_supports("args", GRPOTrainer):
        trainer_kwargs["args"] = grpo_config
    else:
        trainer_kwargs["config"] = grpo_config

    trainer = GRPOTrainer(**trainer_kwargs)
    
    # ── CRITICAL: Apply ALL compatibility patches BEFORE any training ──
    _maybe_patch_trainer_sampler(trainer)
    _maybe_patch_unsloth_grad_accum(trainer)
    _maybe_patch_unsloth_loss_type(trainer)
    _maybe_patch_unsloth_runtime_attrs(trainer)
    _maybe_patch_unsloth_args_attrs(trainer)
    _maybe_patch_nanmin_symbols()
    
    # ── ADD THIS: Extra safety check for any remaining missing attrs ──
    # Some Unsloth compiled paths access these directly on the trainer
    _safety_attrs = {
        "importance_sampling_level": "token",
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "vllm_importance_sampling_cap": 2.0,
        "current_gradient_accumulation_steps": getattr(trainer, "current_gradient_accumulation_steps", 1),
    }
    for attr, default in _safety_attrs.items():
        if not hasattr(trainer, attr):
            setattr(trainer, attr, default)
            print(f"[SAFETY] Added missing trainer.{attr} = {default!r}")
    
    # ── ADD THIS: Delete stale compiled cache so Unsloth rebuilds with patched attrs ──
    import shutil
    compiled_cache_dirs = [
        Path.cwd() / "unsloth_compiled_cache",
        Path.home() / ".cache" / "unsloth" / "compiled_cache",
    ]
    for cache_dir in compiled_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[INFO] Deleted stale compiled cache: {cache_dir}")
    
    # Now train with fresh cache
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    curves_path = Path(output_dir) / "reward_curves.json"
    _save_json(reward_log, curves_path)
    print(f"Reward curves -> {curves_path}")

    _print_final_stats(reward_log)

    # ── Post-training eval ────────────────────────────────────────────────────
    if run_eval:
        print("\n[Post] Measuring trained model score...")
        FastLanguageModel.for_inference(model)  # fuse LoRA weights for faster generation
        trained_model_metrics = _run_model_episodes(
            model, tokenizer, n_episodes=3, tag="TRAINED MODEL"
        )
        _save_json(trained_model_metrics, Path(output_dir) / "trained_model_metrics.json")

        if base_model_metrics is not None:
            _print_improvement(base_model_metrics, trained_model_metrics)
        else:
            # Fallback: compare heuristic baseline vs trained model
            _print_improvement(
                {**baseline, "tag": "HEURISTIC BASELINE"},
                {**trained_model_metrics, "tag": "TRAINED MODEL"},
            )

    if push_to_hub and hub_model_id:
        print(f"\nPushing to Hub: {hub_model_id}")
        trainer.push_to_hub(hub_model_id)

    return trainer


# ── Quick heuristic eval (no LLM needed — uses planner baseline) ──────────────

def _quick_heuristic_eval(n_episodes: int = 6) -> Dict[str, Any]:
    """Run heuristic-only multi-agent episodes (client=None → deterministic planner).

    Uses run_episode so metrics are AMAN/DMAN rewards from the real multi-agent
    environment — not single-agent grades. Same format as _run_model_episodes so
    _print_improvement can compare them directly.
    """
    from multi_agent.inference import run_episode as _run_ep

    env = MultiAgentATCEnvironment(seed=99)
    sup = SupervisorAgent()

    # Fixed task list — no generator mutations for a stable repeatable baseline
    eval_tasks = ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id      = task_id,
                client       = None,   # heuristic mode — no LLM
                env          = env,
                generator    = None,
                supervisor   = sup,
                episode_id   = ep,
                use_generator= False,
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0)))
        except Exception as exc:
            print(f"  [WARN] Heuristic eval ep {ep} failed: {exc}")

    def _mean(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              "HEURISTIC BASELINE",
        "n_episodes":       n_episodes,
        "mean_composite":   _mean(composites),
        "mean_aman_reward": _mean(aman_rews),
        "mean_dman_reward": _mean(dman_rews),
        "mean_conflicts":   _mean(conflict_list),
        "mean_emg_handled": _mean(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


def _print_improvement(
    before: Dict[str, Any], after: Dict[str, Any]
) -> None:
    tag_b = before.get("tag", "BEFORE")
    tag_a = after.get("tag", "AFTER")
    rows = [
        ("mean_composite",   "Composite score"),
        ("mean_aman_reward", "AMAN reward"),
        ("mean_dman_reward", "DMAN reward"),
        ("mean_conflicts",   "Avg conflicts"),
        ("mean_emg_handled", "Emg handled"),
    ]
    width = 56
    print(f"\n{'='*width}")
    print(f"  BEFORE vs AFTER TRAINING")
    print(f"  {tag_b!r:24s}  →  {tag_a!r}")
    print(f"{'='*width}")
    for key, label in rows:
        bv = before.get(key, 0.0)
        av = after.get(key, 0.0)
        delta = av - bv
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        sign = "+" if delta >= 0 else ""
        print(f"  {label:20s}: {bv:6.3f}  →  {av:6.3f}  ({sign}{delta:.3f} {arrow})")
    print(f"{'='*width}")


# ── Local model client for in-process inference eval ──────────────────────────

class _LocalModelClient:
    """Duck-type OpenAI client wrapping a locally loaded Unsloth/PEFT model."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def _create(self, *, model=None, messages, temperature=0.3, max_tokens=512, **kw):
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        use_cuda = torch.cuda.is_available() and str(self._model.device).startswith("cuda")
        cast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=cast_dtype):
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=min(int(max_tokens), 512),
                        temperature=max(float(temperature), 0.01),
                        do_sample=float(temperature) > 0.01,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            else:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=min(int(max_tokens), 512),
                    temperature=max(float(temperature), 0.01),
                    do_sample=float(temperature) > 0.01,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        class _Msg:
            content = text
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]

        return _Resp()

    @property
    def chat(self):
        _self = self
        class _Comp:
            def create(self, **kw):
                return _self._create(**kw)
        class _Chat:
            completions = _Comp()
        return _Chat()


def _run_model_episodes(
    model,
    tokenizer,
    n_episodes: int = 3,
    tag: str = "MODEL",
    use_generator: bool = False,
) -> Dict[str, Any]:
    """Run multi-agent episodes using an in-process model client.

    use_generator=False keeps tasks fixed so base and trained models
    see identical scenarios — essential for a fair comparison.
    """
    from multi_agent.inference import run_episode as _run_ep

    client = _LocalModelClient(model, tokenizer)
    env = MultiAgentATCEnvironment(seed=77)
    gen = ChallengeGenerator(seed=77)
    sup = SupervisorAgent()

    # Two representative tasks: one easy, one hard
    eval_tasks = ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id=task_id,
                client=client,
                env=env,
                generator=gen if use_generator else None,
                supervisor=sup,
                episode_id=ep,
                use_generator=use_generator,
                model_name="local",
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(
                int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0))
            )
        except Exception as exc:
            print(f"  [WARN] model eval episode {ep} failed: {exc}")

    def _m(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              tag,
        "n_episodes":       n_episodes,
        "mean_composite":   _m(composites),
        "mean_aman_reward": _m(aman_rews),
        "mean_dman_reward": _m(dman_rews),
        "mean_conflicts":   _m(conflict_list),
        "mean_emg_handled": _m(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _wandb_available() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _check_reward_hacking(reward_log: Dict[str, List[float]]) -> None:
    """Warn when mean composite rises but role rewards collapse (gaming signal)."""
    comp = reward_log["composite"]
    if len(comp) < 100:
        return
    recent_50  = comp[-50:]
    earlier_50 = comp[-100:-50]
    mean_recent  = sum(recent_50)  / 50
    mean_earlier = sum(earlier_50) / 50
    if mean_recent > mean_earlier + 0.1:
        # Check if any role's recent std collapsed (< 0.05 = suspiciously uniform)
        for role in ("AMAN", "DMAN"):
            rs = reward_log.get(role, [])
            if len(rs) >= 20:
                recent = rs[-20:]
                mean_r = sum(recent) / len(recent)
                std_r = (sum((x - mean_r) ** 2 for x in recent) / len(recent)) ** 0.5
                if std_r < 0.05:
                    print(
                        f"[WARN] Possible reward hacking: {role} std={std_r:.4f} "
                        f"while composite reward is rising. Sample outputs and inspect."
                    )


def _print_final_stats(reward_log: Dict[str, List[float]]) -> None:
    print("\n=== TRAINING REWARD SUMMARY ===")
    for role, rewards in reward_log.items():
        if not rewards:
            continue
        n = len(rewards)
        first_q = rewards[:max(1, n // 4)]
        last_q  = rewards[max(0, 3 * n // 4):]
        mean_all   = sum(rewards) / n
        mean_first = sum(first_q) / len(first_q)
        mean_last  = sum(last_q)  / len(last_q)
        trend = "↑" if mean_last > mean_first + 0.05 else (
            "↓" if mean_last < mean_first - 0.05 else "→"
        )
        print(
            f"  {role:12s}: mean={mean_all:.3f} | "
            f"first_q={mean_first:.3f} -> last_q={mean_last:.3f} {trend}"
        )


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name_or_path: str, n_episodes: int = 20, seed: int = 99) -> Dict[str, Any]:
    """Run trained model on evaluation episodes."""
    torch, FastLanguageModel, _, _ = _require_training_deps()

    print(f"\nEvaluating {model_name_or_path} on {n_episodes} episodes...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name_or_path, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    import random
    from training.dataset import AMAN_SYSTEM, DMAN_SYSTEM, SUPERVISOR_PROFILES

    env        = MultiAgentATCEnvironment(seed=seed)
    generator  = ChallengeGenerator(seed=seed)
    supervisor = SupervisorAgent()
    task_list  = list(ordered_tasks())
    rng        = random.Random(seed)

    results: Dict[str, List] = {
        "aman_rewards": [], "dman_rewards": [], "composite_scores": [],
        "conflict_counts": [], "coordination_scores": [],
        "generator_difficulty": [],
    }

    for ep in range(n_episodes):
        base_task = rng.choice(task_list)
        profile   = supervisor.sample_profile(ep)
        mutated, is_solvable = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(episode_id=ep, mutated_task=mutated)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        def _chat(system, user):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        def _gen(prompt):
            import torch as _torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with _torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        aman_comp = _gen(_chat(AMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", aman_obs.to_prompt_text()))
        dman_comp = _gen(_chat(DMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", dman_obs.to_prompt_text()))

        aman_action = parse_aman_action(aman_comp)
        dman_action = parse_dman_action(dman_comp)
        if not aman_action or not dman_action:
            continue

        aman_obs, dman_obs, _, done = env.step_bid(aman_action, dman_action)
        if not done:
            env.step_negotiate(aman_action, dman_action)

        result = env.finalize()
        generator.update(result.composite_score)

        results["aman_rewards"].append(result.aman_reward)
        results["dman_rewards"].append(result.dman_reward)
        results["composite_scores"].append(result.composite_score)
        results["conflict_counts"].append(result.per_role.cross_lane_conflicts)
        results["coordination_scores"].append(result.per_role.coordination_score)
        results["generator_difficulty"].append(generator.difficulty_level)

        print(
            f"  ep{ep:3d} | composite={result.composite_score:.3f} | "
            f"AMAN={result.aman_reward:.3f} | DMAN={result.dman_reward:.3f} | "
            f"coord={result.per_role.coordination_score:.3f} | "
            f"gen_lvl={generator.difficulty_level}"
        )

    def _mean(lst):
        return round(sum(lst) / max(1, len(lst)), 3)

    summary = {
        "mean_composite":    _mean(results["composite_scores"]),
        "mean_aman_reward":  _mean(results["aman_rewards"]),
        "mean_dman_reward":  _mean(results["dman_rewards"]),
        "mean_coordination": _mean(results["coordination_scores"]),
        "mean_conflicts":    _mean(results["conflict_counts"]),
        "final_gen_difficulty": results["generator_difficulty"][-1] if results["generator_difficulty"] else 1,
    }
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {**results, "summary": summary}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ADAPT-Focused ATC Multi-Agent GRPO Training")
    parser.add_argument("--model",          default=DEFAULT_MODEL,
                        help="Base model (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--output_dir",     default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes",       type=int, default=200)
    parser.add_argument("--lora_rank",      type=int, default=LORA_RANK)
    parser.add_argument("--n_generations",  type=int, default=None,
                        help="GRPO group size (default: 2 for T4 Colab).")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--no_eval",      action="store_true", help="Skip before/after eval")
    parser.add_argument("--eval_only",    action="store_true")
    parser.add_argument("--push_to_hub",  action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    args = parser.parse_args()

    # Allow CLI override of group size (useful for Colab memory tuning)
    if args.n_generations is not None:
        global N_GENERATIONS, BATCH_SIZE
        N_GENERATIONS = args.n_generations
        # Adjust batch size to stay divisible
        if BATCH_SIZE % N_GENERATIONS != 0:
            BATCH_SIZE = N_GENERATIONS

    if args.eval_only:
        evaluate(args.model, n_episodes=20, seed=args.seed)
    else:
        train(
            model_name=args.model,
            output_dir=args.output_dir,
            n_episodes=args.episodes,
            lora_rank=args.lora_rank,
            seed=args.seed,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            run_eval=not args.no_eval,
        )


if __name__ == "__main__":
    main()
