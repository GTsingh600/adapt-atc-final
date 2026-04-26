# -*- coding: utf-8 -*-
"""ADAPT-Focused ATC Multi-Agent GRPO Training — Colab Notebook

ADAPT (Adaptive Decision Agent for Problem Transfer) is the primary training target.
It learns to map unknown scheduling domains to ATC parameters using structural reasoning.

Runtime → Change runtime type → T4 GPU
Run cells top-to-bottom.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Mount Drive + Clone Repo
# ══════════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount("/content/drive")

import subprocess, sys, os

BRANCH     = "main"
REPO_URL   = "https://github.com/GTsingh600/ats.git"
REPO_DIR   = "/content/ATC"
OUTPUT_DIR = "/content/drive/MyDrive/atc-adapt"

subprocess.run(["rm", "-rf", REPO_DIR], check=True)
subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, REPO_DIR],
    check=True,
)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(f"Repo ready: {REPO_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 2 — Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "trl"], check=False)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "unsloth[colab-new]",
        "trl==0.15.2",
        "transformers==4.51.3",
        "accelerate>=0.32.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.20.0",
        "numpy>=1.26.0",
        "matplotlib>=3.9.0",
        "openenv-core[core]>=0.2.3",
        "openai>=1.30.0",
        "fastapi>=0.111.0",
        "pydantic>=2.7.0",
        "uvicorn>=0.29.0",
    ],
    check=True,
)

os.environ["WANDB_MODE"]             = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Install complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Smoke-test Imports
# (unsloth MUST be imported before torch/trl/transformers)
# ══════════════════════════════════════════════════════════════════════════════

import unsloth                          # ← first, enables Unsloth kernel patches
import torch
import trl
import transformers
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

print(f"Python      : {sys.version.split()[0]}")
print(f"Torch       : {torch.__version__}")
print(f"TRL         : {trl.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {props.total_memory / 1e9:.1f} GB")

# Repo imports — verify ADAPT-focused dataset builds correctly
from training.dataset import build_episode_dataset
data = build_episode_dataset(n_episodes=4, seed=42)
roles = sorted({x['agent_role'] for x in data})
print(f"\nDataset smoke: {len(data)} samples | roles: {roles}")
print(f"  Expected: ADAPT + AMAN + DMAN (3 roles)")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Train ADAPT-Focused Model
#
# T4 settings: N_GENERATIONS=2, BATCH_SIZE=2, GRAD_ACCUM=8
#   → effective batch = 16, group size = 2
#
# Model choices:
#   Qwen/Qwen2.5-1.5B-Instruct  — faster, fits T4 easily, good for iteration
#   Qwen/Qwen2.5-7B-Instruct    — higher quality, needs more VRAM
#
# run_eval=True → runs base-model eval BEFORE and trained-model eval AFTER
# ══════════════════════════════════════════════════════════════════════════════

import training.train_grpo as _grpo

os.makedirs(OUTPUT_DIR, exist_ok=True)

_grpo.train(
    model_name  = "Qwen/Qwen2.5-1.5B-Instruct",   # use 7B for higher quality
    output_dir  = OUTPUT_DIR,
    n_episodes  = 100,       # ~1 hr on T4 for 1.5B; use 200+ for full training
    lora_rank   = 32,
    seed        = 42,
    run_eval    = True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5 — Plot Reward Curves
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
from IPython.display import display, Image

PLOTS_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

subprocess.run(
    [
        sys.executable, "training/plot_rewards.py",
        "--input",   f"{OUTPUT_DIR}/reward_curves.json",
        "--save",    PLOTS_DIR,
        "--no_show",
    ],
    check=False,
    cwd=REPO_DIR,
)

for png in sorted(Path(PLOTS_DIR).glob("*.png")):
    print(png.name)
    display(Image(str(png)))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6 — Standalone Eval  (optional — already included in Cell 4)
# ══════════════════════════════════════════════════════════════════════════════

import json

EVAL_OUT = f"{OUTPUT_DIR}/eval_results.json"

subprocess.run(
    [
        sys.executable, "training/eval.py",
        "--base",     "heuristic-baseline",
        "--trained",  OUTPUT_DIR,
        "--episodes", "5",
        "--output",   EVAL_OUT,
    ],
    check=False,
    cwd=REPO_DIR,
)

if Path(EVAL_OUT).exists():
    results = json.loads(Path(EVAL_OUT).read_text())
    print("\n=== EVAL RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
# Cell 7 — Heuristic Sanity Check  (no model needed — verifies environment)
# ══════════════════════════════════════════════════════════════════════════════

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.inference import run_episode

env = MultiAgentATCEnvironment(seed=0)

result = run_episode(
    task_id      = "bengaluru_irrops_hard",
    client       = None,          # heuristic mode — no LLM
    env          = env,
    curriculum   = None,
    episode_id   = 0,
    use_curriculum= False,
)
print(f"\nHeuristic sanity: composite={result['composite']:.3f} "
      f"aman={result['aman_reward']:.3f} dman={result['dman_reward']:.3f} "
      f"conflicts={result['conflicts']}")
