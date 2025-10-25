"""Bulk story generation helper for TRM model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from tokenizer_configs import get_tokenizer_for_dataset
from utils.functions import load_model_class


DEFAULT_PROMPTS: List[str] = [
    "Once upon a time, in a peaceful town,",
    "Once upon a time, there was a little boy named Tim",
    "Once upon a time, in a big forest,",
    "Once upon a time, there was a clever little dog named Max",
    "Once upon a time, there was a little girl named Lily",
    "One day, a little fish named Fin was swimming near the shore",
    "One day, a little girl named Lily found a needle in her room",
    "Once upon a time, in a small house, there lived a little girl named Amy",
    "Once upon a time, there was a tiny mushroom in a big forest",
    "Once upon a time, there was a queen who wanted to relax",
]


def _select_device(name: Optional[str]) -> torch.device:
    if name is not None and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"{path} does not look like a TRM training checkpoint")
    return checkpoint


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = _load_checkpoint(checkpoint_path)
    config_dict = checkpoint.get("config", {})

    if not config_dict:
        raise ValueError("Checkpoint missing config metadata")

    # Extract architecture config
    arch_config = config_dict.get("arch", {})
    model_name = arch_config.get("name")
    loss_name = arch_config.get("loss", {}).get("name")

    if not model_name:
        raise ValueError("Checkpoint missing model architecture name")

    # Build model config
    tokenizer_name = config_dict.get("tokenizer_name") or "gpt-neo-10k"
    tokenizer = get_tokenizer_for_dataset("tinystories", tokenizer_name)

    model_cfg = dict(
        **arch_config,
        batch_size=1,  # Generate one at a time
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


def generate_story(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    """Generate a single story from a prompt."""
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Tokenizer produced empty sequence for prompt")

    # Get max sequence length from model config
    max_seq_len = model.config.seq_len if hasattr(model, 'config') else model.model.config.seq_len

    # Pad token IDs to max_seq_len to avoid size mismatches
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
    prompt_len = len(token_ids)

    # Start with padded sequence
    padded_ids = token_ids + [pad_token_id] * (max_seq_len - prompt_len)
    input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)

    # Prepare batch
    batch = {
        "inputs": input_ids,
        "targets": input_ids,  # Not used for generation
        "labels": input_ids,  # For loss head
        "puzzle_identifiers": torch.zeros((1,), dtype=torch.long, device=device),
    }

    # Get the model (use outer model with ACT wrapper, not inner)
    act_model = model.model if hasattr(model, 'model') else model

    # Initialize carry once
    carry = act_model.initial_carry(batch)

    # Generate tokens
    generated_ids = list(token_ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    current_pos = prompt_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if current_pos >= max_seq_len:
                break

            # Forward pass - use ACT wrapper to do multiple refinement iterations
            # Set model to eval mode to use halt_max_steps iterations
            act_model.eval()

            # Do multiple ACT iterations until halting
            for act_step in range(act_model.config.halt_max_steps):
                carry, outputs = act_model(carry=carry, batch=batch)
                # Check if halted (all sequences in batch have halted)
                if carry.halted.all():
                    break

            # Get logits for current position
            logits = outputs["logits"][0, current_pos - 1, :]  # [vocab_size]

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
                break

            generated_ids.append(next_token)

            # Update the input at current position
            input_ids[0, current_pos] = next_token
            batch["inputs"] = input_ids
            batch["targets"] = input_ids
            batch["labels"] = input_ids
            current_pos += 1

    # Decode only the new tokens
    completion_ids = generated_ids[prompt_len:]
    completion_text = tokenizer.decode(completion_ids)
    return completion_text.rstrip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate TinyStories completions from a TRM checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: stories.jsonl next to checkpoint)",
    )
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, mps, auto)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling cutoff")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Random seeds (one per prompt)")

    args = parser.parse_args()

    prompts = list(DEFAULT_PROMPTS)
    seeds = args.seeds if args.seeds is not None else list(range(len(prompts)))
    if len(seeds) != len(prompts):
        raise ValueError(f"Number of seeds ({len(seeds)}) must match prompts ({len(prompts)})")

    device = _select_device(args.device)
    model, tokenizer, config_dict = _load_model(args.checkpoint, device)

    # Set seed for reproducibility
    if args.temperature > 0:
        base_seed = sum((s * (i + 1) * 1013904223) % (2 ** 32) for i, s in enumerate(seeds))
        torch.manual_seed(base_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed)

    output_path = args.output or args.checkpoint.parent / "stories.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(prompts)} stories...")
    completions = []
    for idx, prompt in enumerate(prompts):
        print(f"  [{idx + 1}/{len(prompts)}] Generating story...")
        completion = generate_story(
            model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, device
        )
        completions.append(completion)

    stories = [
        {
            "id": idx,
            "prompt": prompt,
            "seed": int(seed),
            "completion": text,
        }
        for idx, (prompt, seed, text) in enumerate(zip(prompts, seeds, completions))
    ]

    payload = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "tokenizer": config_dict.get("tokenizer_name", "gpt-neo-10k"),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
        "seeds": list(map(int, seeds)),
        "stories": stories,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(stories)} stories to {output_path}")


if __name__ == "__main__":
    main()
