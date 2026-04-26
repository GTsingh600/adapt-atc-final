"""Before/after evaluation: base model vs trained checkpoint.

Mirrors kube-sre-gym eval.py pattern:
  - Run N episodes with base model (heuristic or untrained LLM)
  - Run same N episodes with trained checkpoint
  - Compare AMAN reward, DMAN reward, composite, coordination score
  - Print structured comparison table

Usage:
  python training/eval.py --base heuristic-baseline --trained ./outputs/atc-multiagent
  python training/eval.py --base Qwen/Qwen2.5-7B-Instruct --trained ./outputs/atc-multiagent --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapter import ContextAdaptiveCurriculum
from multi_agent.inference import run_episode
from multi_agent.models import SUPERVISOR_PROFILES
from tasks import ordered_tasks


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_client(model_name_or_path: str) -> Optional[object]:
    """Load OpenAI-compatible client for a model or checkpoint."""
    if model_name_or_path == "heuristic-baseline":
        return None

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    hf_token = os.getenv("HF_TOKEN", "")

    # Local checkpoint: load with Unsloth and spin up inference
    if Path(model_name_or_path).exists():
        try:
            import json as _json
            from unsloth import FastLanguageModel

            ckpt = Path(model_name_or_path)
            adapter_cfg_path = ckpt / "adapter_config.json"
            full_cfg_path    = ckpt / "config.json"

            if adapter_cfg_path.exists() and not full_cfg_path.exists():
                # TRL saves LoRA adapters without the base model config —
                # load base model first, then apply the adapter on top.
                adapter_cfg = _json.loads(adapter_cfg_path.read_text())
                base_name   = adapter_cfg.get(
                    "base_model_name_or_path",
                    os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                )
                print(f"  LoRA adapter detected — loading base: {base_name}")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    base_name,
                    max_seq_length=4096,
                    load_in_4bit=True,
                )
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(ckpt))
            else:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    str(ckpt),
                    max_seq_length=4096,
                    load_in_4bit=True,
                )

            FastLanguageModel.for_inference(model)
            print(f"  Loaded local checkpoint: {model_name_or_path}")
            return _LocalModelClient(model, tokenizer)
        except ImportError:
            print(f"  [WARN] unsloth not installed — using API for {model_name_or_path}")

    try:
        from openai import OpenAI
        return OpenAI(base_url=api_base, api_key=hf_token)
    except ImportError:
        print("  [WARN] openai not installed — using heuristic fallback")
        return None


class _LocalModelClient:
    """Thin OpenAI-compatible wrapper for a locally loaded Unsloth model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    class _Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _Resp:
        def __init__(self, text):
            self.choices = [_LocalModelClient._Choice(text)]

    def chat_completions_create(self, *, model, messages, temperature, max_tokens, **kw):
        import torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._Resp(text)

    @property
    def chat(self):
        return type("Chat", (), {"completions": type("C", (), {"create": self.chat_completions_create})()})()


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    n_episodes: int,
    task_ids: List[str],
    seed: int,
    use_curriculum: bool,
    label: str,
) -> Dict:
    """Run N episodes and return aggregated metrics."""
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {label} ({model_name})")
    print(f"  Episodes:   {n_episodes} × {len(task_ids)} tasks")
    print(f"{'─'*50}")

    client    = _load_client(model_name)
    env       = MultiAgentATCEnvironment(seed=seed)
    curriculum = ContextAdaptiveCurriculum(seed=seed)

    records: List[Dict] = []
    t0 = time.time()

    for task_id in task_ids:
        for ep in range(n_episodes):
            try:
                r = run_episode(
                    task_id=task_id,
                    client=client,
                    env=env,
                    curriculum=curriculum if use_curriculum else None,
                    episode_id=ep,
                    use_curriculum=use_curriculum,
                )
                r["task_id"] = task_id
                records.append(r)
                print(
                    f"  {task_id[:30]:30s} ep{ep:2d} | "
                    f"composite={r['composite']:.3f} aman={r['aman_reward']:.3f} "
                    f"dman={r['dman_reward']:.3f} coord={r['coord_score']:.3f}"
                )
            except Exception as exc:
                print(f"  [ERROR] {task_id} ep{ep}: {exc}")

    elapsed = time.time() - t0

    def _m(key): return sum(r[key] for r in records) / max(1, len(records))

    summary = {
        "label":           label,
        "model":           model_name,
        "n_episodes":      len(records),
        "elapsed_s":       round(elapsed, 1),
        "mean_composite":  round(_m("composite"),   3),
        "mean_aman":       round(_m("aman_reward"), 3),
        "mean_dman":       round(_m("dman_reward"), 3),
        "mean_coord":      round(_m("coord_score"), 3),
        "mean_conflicts":  round(_m("conflicts"),   2),
        "success_rate":    round(sum(1 for r in records if r["composite"] >= 0.60) / max(1, len(records)), 3),
        "emg_arr_handled": int(sum(r["emg_arr_ok"] for r in records)),
        "emg_dep_handled": int(sum(r["emg_dep_ok"] for r in records)),
        "atfm_violations": int(sum(r["atfm_viol"] for r in records)),
        "final_gen_level": records[-1]["gen_difficulty"] if records else 1,
        "records":         records,
    }
    return summary


# ── Comparison printer ────────────────────────────────────────────────────────

def print_comparison(base: Dict, trained: Dict) -> None:
    def _delta(key: str) -> str:
        d = trained[key] - base[key]
        arrow = "↑" if d > 0.01 else ("↓" if d < -0.01 else "→")
        return f"{d:+.3f} {arrow}"

    print(f"\n{'='*65}")
    print(f"  BEFORE vs AFTER TRAINING — Multi-Agent ATC")
    print(f"{'='*65}")
    print(f"  {'Metric':<22}  {'Base':>10}  {'Trained':>10}  {'Delta':>12}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*12}")

    rows = [
        ("Composite score",  "mean_composite"),
        ("AMAN reward",      "mean_aman"),
        ("DMAN reward",      "mean_dman"),
        ("Coordination",     "mean_coord"),
        ("Success rate",     "success_rate"),
        ("Mean conflicts",   "mean_conflicts"),
    ]
    for label, key in rows:
        print(f"  {label:<22}  {base[key]:>10.3f}  {trained[key]:>10.3f}  {_delta(key):>12}")

    print(f"\n  Emergency arrivals handled:   {base['emg_arr_handled']:>4}  →  {trained['emg_arr_handled']}")
    print(f"  Emergency departures handled: {base['emg_dep_handled']:>4}  →  {trained['emg_dep_handled']}")
    print(f"  ATFM violations:              {base['atfm_violations']:>4}  →  {trained['atfm_violations']}")
    print(f"  Generator difficulty:         {base['final_gen_level']:>4}  →  {trained['final_gen_level']}")
    print(f"{'='*65}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent ATC Evaluation")
    parser.add_argument("--base",    default="heuristic-baseline")
    parser.add_argument("--trained", default="./outputs/atc-multiagent")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tasks",   nargs="+",
                        default=["delhi_monsoon_recovery_easy",
                                 "bengaluru_irrops_hard"])
    parser.add_argument("--seed",    type=int, default=99)
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--output",  default=None, help="Save results JSON")
    args = parser.parse_args()

    base_results    = evaluate_model(
        args.base,    args.episodes, args.tasks, args.seed, not args.no_curriculum, "BASE"
    )
    trained_results = evaluate_model(
        args.trained, args.episodes, args.tasks, args.seed, not args.no_curriculum, "TRAINED"
    )

    print_comparison(base_results, trained_results)

    if args.output:
        out = {
            "base":    {k: v for k, v in base_results.items()    if k != "records"},
            "trained": {k: v for k, v in trained_results.items() if k != "records"},
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
