"""Comprehensive generation sweep script for TRM model evaluation.

This script:
1. Finds all training runs and their latest checkpoints
2. Tests diverse prompts (in-distribution and out-of-distribution)
3. Sweeps across multiple generation hyperparameters (temperature, top-k)
4. Tests with different ACT settings (halt_max_steps = 1 and 4)
5. Saves all results organized by checkpoint directory
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch

from tokenizer_configs import get_tokenizer_for_dataset
from utils.functions import load_model_class


# Diverse prompts mixing in-distribution and out-of-distribution scenarios
DIVERSE_PROMPTS: List[str] = [
    # Classic story openings (in-distribution)
    "Once upon a time, there was a little girl named Lily",
    "One day, a little boy named Tim found a big box",

    # Mid-sentence prompts (test continuation)
    "She was very happy when",
    "The dog ran very fast to",

    # Action/adventure focus
    "One sunny day, Ben and his friends went to explore",
    "The brave little mouse decided to",

    # Emotional/character-driven
    "Emma felt sad because her toy was broken. She",
    "The kind teacher smiled and said",

    # Problem-solving scenarios
    "There was a problem. The ball was stuck in the tree, so",
    "When the rain started, everyone needed to find",
]


@dataclass
class GenerationConfig:
    """Configuration for a generation sweep."""
    temperature: float
    top_k: int
    halt_max_steps: int  # ACT iterations


# Define sweep configurations
GENERATION_CONFIGS = [
    # Greedy decoding with different ACT settings
    GenerationConfig(temperature=0.0, top_k=0, halt_max_steps=1),
    GenerationConfig(temperature=0.0, top_k=0, halt_max_steps=4),

    # Low temperature sampling
    GenerationConfig(temperature=0.3, top_k=10, halt_max_steps=1),
    GenerationConfig(temperature=0.3, top_k=10, halt_max_steps=4),

    # Medium temperature sampling
    GenerationConfig(temperature=0.5, top_k=20, halt_max_steps=1),
    GenerationConfig(temperature=0.5, top_k=20, halt_max_steps=4),

    # High temperature sampling
    GenerationConfig(temperature=0.8, top_k=40, halt_max_steps=1),
    GenerationConfig(temperature=0.8, top_k=40, halt_max_steps=4),
]


def _select_device(name: Optional[str]) -> torch.device:
    if name is not None and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in a run directory by file modification time."""
    checkpoint_files = list(run_dir.glob("step_*.pt"))
    if not checkpoint_files:
        return None

    # Sort by modification time (most recent first)
    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return latest


def find_all_runs(checkpoints_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Find all training runs and their latest checkpoints.

    Returns:
        List of (model_type, run_dir, checkpoint_path) tuples
    """
    runs = []

    # Iterate through model type directories
    for model_dir in checkpoints_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_type = model_dir.name

        # Find all run directories (e.g., "careful-buffalo")
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Find latest checkpoint
            latest_ckpt = find_latest_checkpoint(run_dir)
            if latest_ckpt:
                runs.append((model_type, run_dir, latest_ckpt))

    return runs


def _load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"{path} does not look like a TRM training checkpoint")
    return checkpoint


def _load_model(checkpoint_path: Path, device: torch.device, force_halt_max_steps: Optional[int] = None):
    """Load model from checkpoint, optionally overriding halt_max_steps."""
    checkpoint = _load_checkpoint(checkpoint_path)
    config_dict = checkpoint.get("config", {})

    if not config_dict:
        raise ValueError("Checkpoint missing config metadata")

    # Extract architecture config
    arch_config = config_dict.get("arch", {}).copy()
    model_name = arch_config.get("name")
    loss_name = arch_config.get("loss", {}).get("name")

    if not model_name:
        raise ValueError("Checkpoint missing model architecture name")

    # Override halt_max_steps if specified
    if force_halt_max_steps is not None:
        arch_config["halt_max_steps"] = force_halt_max_steps

    # Build model config
    tokenizer_name = config_dict.get("tokenizer_name") or "gpt-neo-10k"
    tokenizer = get_tokenizer_for_dataset("tinystories", tokenizer_name)

    model_cfg = dict(
        **arch_config,
        batch_size=len(DIVERSE_PROMPTS),  # Batch size for parallel generation
        vocab_size=tokenizer.vocab_size,
        seq_len=config_dict.get("seq_len", 512),
        num_puzzle_identifiers=1,
        causal=True,
    )

    # Load model
    model_cls = load_model_class(model_name)
    loss_head_cls = load_model_class(loss_name) if loss_name else None
    loss_type = arch_config.get("loss", {}).get("loss_type") if loss_name else None

    model = model_cls(model_cfg)
    if loss_head_cls:
        model = loss_head_cls(model, loss_type=loss_type)

    # Load weights with backward compatibility for old checkpoints
    state_dict = checkpoint["model_state_dict"]

    # Fix: Old flat transformer checkpoints have L_level.layers.X, new code expects flat_layers.X
    if arch_config.get("flat_transformer", False):
        new_state_dict = {}
        for key, value in state_dict.items():
            # Rename L_level.layers.X -> flat_layers.X
            if "L_level.layers." in key:
                new_key = key.replace("L_level.layers.", "flat_layers.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, config_dict


def generate_stories_batched(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> List[str]:
    """Generate multiple stories in parallel using batched inference."""
    batch_size = len(prompts)

    # Get model and config
    act_model = model.model if hasattr(model, 'model') else model
    max_seq_len = model.config.seq_len if hasattr(model, 'config') else model.model.config.seq_len
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Encode all prompts
    all_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    if any(not token_ids for token_ids in all_token_ids):
        raise ValueError("Tokenizer produced empty sequence for a prompt")

    # Track prompt lengths and current positions for each sequence
    prompt_lens = [len(token_ids) for token_ids in all_token_ids]
    current_positions = list(prompt_lens)

    # Pad all to max_seq_len and create batch tensor
    padded_sequences = []
    for token_ids in all_token_ids:
        padded = token_ids + [pad_token_id] * (max_seq_len - len(token_ids))
        padded_sequences.append(padded)

    input_ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)  # [batch_size, max_seq_len]

    # Prepare batch
    batch = {
        "inputs": input_ids,
        "targets": input_ids,
        "labels": input_ids,
        "puzzle_identifiers": torch.zeros((batch_size,), dtype=torch.long, device=device),
    }

    # Initialize carry
    carry = act_model.initial_carry(batch)

    # Track which sequences are done
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if all sequences are done
            if finished.all():
                break

            # Forward pass with ACT iterations
            act_model.eval()
            for act_step in range(act_model.config.halt_max_steps):
                carry, outputs = act_model(carry=carry, batch=batch)
                if carry.halted.all():
                    break

            # Process each sequence in the batch
            for b_idx in range(batch_size):
                if finished[b_idx] or current_positions[b_idx] >= max_seq_len:
                    continue

                # Get logits for current position
                logits = outputs["logits"][b_idx, current_positions[b_idx] - 1, :]  # [vocab_size]

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Sample
                if temperature > 0:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(logits).item()

                # Check for EOS
                if eos_id is not None and next_token == eos_id:
                    finished[b_idx] = True
                    continue

                # Update input at current position
                input_ids[b_idx, current_positions[b_idx]] = next_token
                current_positions[b_idx] += 1

            # Update batch
            batch["inputs"] = input_ids
            batch["targets"] = input_ids
            batch["labels"] = input_ids

    # Decode completions
    completions = []
    for b_idx in range(batch_size):
        # Extract generated tokens (from prompt_len to current_position)
        completion_ids = input_ids[b_idx, prompt_lens[b_idx]:current_positions[b_idx]].cpu().tolist()
        completion_text = tokenizer.decode(completion_ids)
        completions.append(completion_text.rstrip())

    return completions


def run_generation_sweep(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    max_new_tokens: int = 512,
    seed: int = 42,
):
    """Run generation sweep for a single checkpoint."""

    print(f"\n{'='*80}")
    print(f"Processing: {checkpoint_path}")
    print(f"{'='*80}")

    results = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "generations": []
    }

    for config_idx, gen_config in enumerate(GENERATION_CONFIGS):
        print(f"\n[Config {config_idx + 1}/{len(GENERATION_CONFIGS)}] "
              f"temp={gen_config.temperature}, top_k={gen_config.top_k}, "
              f"ACT={gen_config.halt_max_steps}")

        # Load model with specific halt_max_steps
        model, tokenizer, config_dict = _load_model(
            checkpoint_path,
            device,
            force_halt_max_steps=gen_config.halt_max_steps
        )

        # Set seed for reproducibility (different per config)
        if gen_config.temperature > 0:
            torch.manual_seed(seed + config_idx)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + config_idx)

        config_results = {
            "temperature": gen_config.temperature,
            "top_k": gen_config.top_k,
            "halt_max_steps": gen_config.halt_max_steps,
            "stories": []
        }

        # Generate all prompts in a single batch for speed
        print(f"  Generating {len(DIVERSE_PROMPTS)} prompts in batch...")
        try:
            completions = generate_stories_batched(
                model,
                tokenizer,
                DIVERSE_PROMPTS,
                max_new_tokens,
                gen_config.temperature,
                gen_config.top_k,
                device
            )

            # Store results
            for prompt_idx, (prompt, completion) in enumerate(zip(DIVERSE_PROMPTS, completions)):
                config_results["stories"].append({
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "completion": completion,
                })
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to individual generation on error
            for prompt_idx, prompt in enumerate(DIVERSE_PROMPTS):
                config_results["stories"].append({
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "error": str(e),
                })

        results["generations"].append(config_results)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save results to run directory
    output_path = run_dir / "generation_sweep_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive generation sweep across all training runs."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("runs/checkpoints"),
        help="Root directory containing all checkpoint subdirectories",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (cpu, cuda, mps, auto)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per story"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help="Only process runs matching this model type (e.g., 'trm_tinystories_1m_flat')",
    )
    parser.add_argument(
        "--filter-run",
        type=str,
        default=None,
        help="Only process runs matching this run name (e.g., 'careful-buffalo')",
    )

    args = parser.parse_args()

    # Find all runs
    all_runs = find_all_runs(args.checkpoints_dir)

    if not all_runs:
        print(f"No training runs found in {args.checkpoints_dir}")
        return

    # Apply filters
    if args.filter_model:
        all_runs = [(mt, rd, cp) for mt, rd, cp in all_runs if args.filter_model in mt]
    if args.filter_run:
        all_runs = [(mt, rd, cp) for mt, rd, cp in all_runs if args.filter_run in rd.name]

    print(f"Found {len(all_runs)} training runs to process:")
    for model_type, run_dir, checkpoint_path in all_runs:
        print(f"  - {model_type}/{run_dir.name}: {checkpoint_path.name}")

    device = _select_device(args.device)
    print(f"\nUsing device: {device}")

    # Process each run
    all_results = []
    for idx, (model_type, run_dir, checkpoint_path) in enumerate(all_runs, 1):
        print(f"\n{'='*80}")
        print(f"Run {idx}/{len(all_runs)}: {model_type}/{run_dir.name}")
        print(f"{'='*80}")

        try:
            results = run_generation_sweep(
                checkpoint_path,
                run_dir,
                device,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR processing {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✓ Completed! Processed {len(all_results)}/{len(all_runs)} runs successfully")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
