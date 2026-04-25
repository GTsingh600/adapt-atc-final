"""SFT Warmup — Teach JSON Format Before GRPO.

Generates supervised fine-tuning data from heuristic baselines so the model
learns correct JSON output format BEFORE GRPO optimises decision quality.

Pipeline:
  1. Run heuristic AMAN/DMAN/ADAPT on all tasks → collect (prompt, completion) pairs
  2. SFT on these pairs with Unsloth QLoRA → model learns to output valid JSON
  3. GRPO then starts from a model that can already produce parseable actions

Usage:
  python training/train_sft.py --episodes 50 --model Qwen/Qwen2.5-1.5B-Instruct
  python training/train_sft.py --episodes 100 --output_dir ./outputs/sft-warmup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import (
    AMAN_SYSTEM,
    DMAN_SYSTEM,
    ADAPT_SYSTEM,
    SUPERVISOR_PROFILES,
    build_episode_dataset,
)
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.models import AgentRole, SupervisorProfileName
from multi_agent.supervisor import SupervisorAgent
from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
from multi_agent.adapt import (
    build_adapt_observation,
    _build_adapt_heuristic,
    apply_adapt_mapping,
)
from tasks import task_catalog, ordered_tasks

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT = "./outputs/sft-warmup"
LORA_RANK      = 32
LORA_ALPHA     = 64
LORA_TARGETS   = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LEN    = 1536
LR             = 2e-5   # SFT can use higher LR than GRPO
BATCH_SIZE     = 2
GRAD_ACCUM     = 4      # effective batch = 8
WARMUP_RATIO   = 0.05
NUM_EPOCHS     = 3
SAVE_STEPS     = 50


# ── SFT Dataset Generation ───────────────────────────────────────────────────

def _action_to_json(action) -> str:
    """Convert a heuristic action to the JSON string the model should learn."""
    try:
        return action.model_dump_json(indent=2)
    except Exception:
        return json.dumps(action.__dict__, indent=2, default=str)


def build_sft_dataset(n_episodes: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate SFT training pairs: (system+user prompt, heuristic completion).

    Each sample has the format TRL SFTTrainer expects:
      {"messages": [
          {"role": "system", "content": ...},
          {"role": "user",   "content": ...},
          {"role": "assistant", "content": <heuristic JSON output>}
      ]}
    """
    import random
    rng = random.Random(seed)
    task_list = list(ordered_tasks())
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)

    # Also load domain tasks for ADAPT
    from domains.icu import icu_task_catalog, ICU_DOMAIN_DESCRIPTION
    icu_catalog = icu_task_catalog()
    domain_tasks = list(icu_catalog.values())
    domain_name = "Hospital ICU Surge Management"
    domain_description = ICU_DOMAIN_DESCRIPTION

    samples: List[Dict[str, Any]] = []

    for ep_id in range(n_episodes):
        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        # ── Decide: domain episode (ADAPT) or ATC episode ─────────────────
        is_domain = rng.random() < 0.50 and domain_tasks

        if is_domain:
            # ADAPT sample
            domain_task = rng.choice(domain_tasks)
            adapt_obs = build_adapt_observation(
                task=domain_task,
                profile=profile,
                domain_name=domain_name,
                domain_description=domain_description,
            )
            adapt_action = _build_adapt_heuristic(adapt_obs, domain_task)
            adapt_json = _action_to_json(adapt_action)

            samples.append({
                "messages": [
                    {"role": "system", "content": ADAPT_SYSTEM},
                    {"role": "user",   "content": adapt_obs.to_prompt_text()},
                    {"role": "assistant", "content": adapt_json},
                ]
            })

            # Also generate AMAN/DMAN from the mapped task
            mapped_task = apply_adapt_mapping(domain_task, adapt_action)
            aman_obs, dman_obs = env.reset(
                episode_id=ep_id,
                supervisor_profile=profile,
                mutated_task=mapped_task,
            )
        else:
            # Regular ATC episode
            base_task = rng.choice(task_list)
            aman_obs, dman_obs = env.reset(
                task_id=base_task.task_id,
                episode_id=ep_id,
                supervisor_profile=profile,
                randomize=True,
            )

        # ── AMAN sample ──────────────────────────────────────────────────
        aman_action = _build_aman_heuristic(aman_obs)
        aman_json = _action_to_json(aman_action)
        aman_system = AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"

        samples.append({
            "messages": [
                {"role": "system", "content": aman_system},
                {"role": "user",   "content": aman_obs.to_prompt_text()},
                {"role": "assistant", "content": aman_json},
            ]
        })

        # ── DMAN sample ──────────────────────────────────────────────────
        atfm = env._state.atfm_deadlines if env._state else {}
        dman_action = _build_dman_heuristic(dman_obs, atfm)
        dman_json = _action_to_json(dman_action)
        dman_system = DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"

        samples.append({
            "messages": [
                {"role": "system", "content": dman_system},
                {"role": "user",   "content": dman_obs.to_prompt_text()},
                {"role": "assistant", "content": dman_json},
            ]
        })

    rng.shuffle(samples)

    # Count roles
    role_counts: Dict[str, int] = {}
    for s in samples:
        sys_msg = s["messages"][0]["content"]
        if "ADAPT" in sys_msg[:50]:
            role = "ADAPT"
        elif "AMAN" in sys_msg[:50]:
            role = "AMAN"
        elif "DMAN" in sys_msg[:50]:
            role = "DMAN"
        else:
            role = "OTHER"
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"  SFT dataset: {len(samples)} samples")
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count}")

    return samples


# ── SFT Training ─────────────────────────────────────────────────────────────

def train_sft(
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT,
    n_episodes: int = 50,
    lora_rank: int = LORA_RANK,
    seed: int = 42,
    num_epochs: int = NUM_EPOCHS,
) -> str:
    """Run SFT warmup training. Returns path to saved checkpoint."""

    print(f"\n{'='*60}")
    print(f"  SFT Warmup — JSON Format Learning")
    print(f"  Model:    {model_name}")
    print(f"  Rank:     {lora_rank}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Epochs:   {num_epochs}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}\n")

    # ── 1. Build SFT dataset ─────────────────────────────────────────────
    print("[1/4] Building SFT dataset from heuristic baselines...")
    t0 = time.time()
    sft_data = build_sft_dataset(n_episodes=n_episodes, seed=seed)
    print(f"  Built in {time.time()-t0:.1f}s\n")

    # ── 2. Load model ────────────────────────────────────────────────────
    print(f"[2/4] Loading model with Unsloth QLoRA (rank={lora_rank})...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # ── 3. Prepare dataset for SFTTrainer ────────────────────────────────
    print("[3/4] Preparing dataset...")
    from datasets import Dataset
    dataset = Dataset.from_list(sft_data)

    # ── 4. Train ─────────────────────────────────────────────────────────
    print(f"[4/4] Starting SFT training ({num_epochs} epochs)...\n")
    from trl import SFTConfig, SFTTrainer

    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        seed=seed,
        bf16=False,    # T4 doesn't support bf16
        fp16=True,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,  # pack short samples together for efficiency
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()

    # Save final checkpoint
    final_dir = str(Path(output_dir) / "sft-final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n  SFT checkpoint saved to: {final_dir}")
    print(f"  Use this as --model for GRPO: python training/train_grpo.py --model {final_dir}")

    return final_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT Warmup — JSON Format Learning")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="Base model (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes",   type=int, default=50,
                        help="Number of episodes for SFT data (each → 2-3 samples)")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--lora_rank",  type=int, default=LORA_RANK)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        lora_rank=args.lora_rank,
        seed=args.seed,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
