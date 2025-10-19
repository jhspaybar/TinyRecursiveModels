from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import os
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import hydra
import wandb
import coolname
import pydantic
from omegaconf import DictConfig, OmegaConf

from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from utils.functions import get_model_source_path, load_model_class
from text_datasets import TextDatasetConfig, create_dataset_loader
from models.moeut_layers import SigmaMoE, SwitchHeadCore


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Architecture
    arch: ArchConfig

    # Dataset
    dataset_name: str = "tinystories"
    data_path: str = "data/hf/TinyStories"
    tokenizer_name: Optional[str] = None
    seq_len: int = 512
    streaming: bool = True
    max_sequences_per_epoch: Optional[int] = None

    # Training schedule
    global_batch_size: int = 64
    epochs: Optional[int] = None
    max_sequences: Optional[int] = None
    lr: float = 3e-4
    lr_min_ratio: float = 0.1
    lr_warmup_percent: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: Optional[float] = 1.0

    # WandB
    project_name: Optional[str] = "trm-tinystories"
    run_name: Optional[str] = None
    wandb_mode: str = "offline"  # online, offline, disabled
    wandb_base_url: Optional[str] = None

    # Misc
    seed: int = 0
    checkpoint_path: Optional[str] = None
    checkpoint_interval_sequences: Optional[int] = 100000  # Save checkpoint every N sequences
    load_checkpoint: Optional[str] = None
    eval_interval_sequences: Optional[int] = None

    @pydantic.model_validator(mode="after")
    def _validate(self):
        if self.epochs is not None and self.max_sequences is not None:
            raise ValueError("Specify only one of `epochs` or `max_sequences`.")
        if self.epochs is None and self.max_sequences is None:
            raise ValueError("Provide `epochs` or `max_sequences` to control training length.")
        return self


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    sequences_consumed: int
    total_sequences: int


def compute_moe_metrics(model: nn.Module) -> Dict[str, float]:
    """Compute MoE diagnostic metrics similar to moeut reference implementation."""
    ff_load, att_load = [], []
    ff_entropy, att_entropy = [], []
    ff_topk_max, att_topk_max = [], []
    ff_sel_norms, att_sel_v_norms, att_sel_o_norms = [], [], []
    ff_bias_abs, att_v_bias_abs, att_o_bias_abs = [], [], []
    ff_route_scales, att_route_scales = [], []

    def process_moe_stats(indices: torch.Tensor, scores: torch.Tensor, logits: torch.Tensor, n_experts: int):
        """Process MoE layer statistics to compute load balance, entropy, and top-k max."""
        try:
            # Load balance: std deviation of expert usage
            idx = indices.detach().to(torch.long)
            counts = torch.bincount(idx.reshape(-1), minlength=n_experts).float()
            total = counts.sum()
            if total > 0:
                load_std = float((counts / total).std(unbiased=False).cpu())
            else:
                load_std = 0.0

            # Top-k max scores
            max_scores = float(scores.detach().max(dim=-1).values.mean().cpu())

            # Entropy of gating distribution
            logits_flat = logits.detach().float()
            probs = torch.sigmoid(logits_flat)
            # Compute entropy: -sum(p * log(p) + (1-p) * log(1-p))
            eps = 1e-10
            probs_safe = torch.clamp(probs, eps, 1 - eps)
            entropy = -(probs_safe * torch.log(probs_safe) + (1 - probs_safe) * torch.log(1 - probs_safe))
            entropy_val = float(entropy.mean().cpu())

            return load_std, max_scores, entropy_val
        except Exception as e:
            # Silently skip this module if there's an error
            return None, None, None

    # Collect stats from all MoE modules
    for module in model.modules():
        if isinstance(module, SigmaMoE):
            if hasattr(module, 'last_topk_indices') and hasattr(module, 'last_topk_scores') and hasattr(module, 'last_selection_logits'):
                load_std, max_scores, entropy_val = process_moe_stats(
                    module.last_topk_indices,
                    module.last_topk_scores,
                    module.last_selection_logits,
                    module.n_experts
                )
                if load_std is not None:
                    ff_load.append(load_std)
                    ff_topk_max.append(max_scores)
                    ff_entropy.append(entropy_val)

                    # Selector weight norm and bias magnitude (if available)
                    with torch.no_grad():
                        if hasattr(module, "expert_sel"):
                            ff_sel_norms.append(float(module.expert_sel.norm().cpu()))
                        if hasattr(module, "expert_bias"):
                            ff_bias_abs.append(float(module.expert_bias.abs().mean().cpu()))
                        if hasattr(module, "route_scale"):
                            route_scale = module.route_scale.detach().cpu().float().mean().item()
                            ff_route_scales.append(route_scale)

        elif isinstance(module, SwitchHeadCore):
            # For attention, we have both V and O experts
            if hasattr(module, 'last_v_topk') and hasattr(module, 'last_v_scores') and hasattr(module, 'last_v_logits'):
                # Combine V and O statistics for attention metrics
                v_load_std, v_max_scores, v_entropy = process_moe_stats(
                    module.last_v_topk,
                    module.last_v_scores,
                    module.last_v_logits,
                    module.n_experts
                )
                o_load_std, o_max_scores, o_entropy = process_moe_stats(
                    module.last_o_topk,
                    module.last_o_scores,
                    module.last_o_logits,
                    module.n_experts
                )
                if v_load_std is not None and o_load_std is not None:
                    # Average V and O stats
                    att_load.append((v_load_std + o_load_std) / 2)
                    att_topk_max.append((v_max_scores + o_max_scores) / 2)
                    att_entropy.append((v_entropy + o_entropy) / 2)

                with torch.no_grad():
                    if hasattr(module, "sel_v"):
                        att_sel_v_norms.append(float(module.sel_v.norm().cpu()))
                    if hasattr(module, "sel_o"):
                        att_sel_o_norms.append(float(module.sel_o.norm().cpu()))
                    if hasattr(module, "expert_v_bias"):
                        att_v_bias_abs.append(float(module.expert_v_bias.abs().mean().cpu()))
                    if hasattr(module, "expert_o_bias"):
                        att_o_bias_abs.append(float(module.expert_o_bias.abs().mean().cpu()))
                    if hasattr(module, "route_scale"):
                        route_scale = module.route_scale.detach().cpu().float().mean().item()
                        att_route_scales.append(route_scale)

    # Compute averages
    metrics: Dict[str, float] = {}
    if ff_load:
        metrics["moe/ff_load_std"] = float(sum(ff_load) / len(ff_load))
    if att_load:
        metrics["moe/att_load_std"] = float(sum(att_load) / len(att_load))
    if ff_entropy:
        metrics["moe/ff_gate_entropy"] = float(sum(ff_entropy) / len(ff_entropy))
    if att_entropy:
        metrics["moe/att_gate_entropy"] = float(sum(att_entropy) / len(att_entropy))
    if ff_topk_max:
        metrics["moe/ff_topk_max"] = float(sum(ff_topk_max) / len(ff_topk_max))
    if att_topk_max:
        metrics["moe/att_topk_max"] = float(sum(att_topk_max) / len(att_topk_max))

    if ff_sel_norms:
        metrics["moe/ff_sel_norm"] = float(sum(ff_sel_norms) / len(ff_sel_norms))
    if att_sel_v_norms:
        metrics["moe/att_sel_v_norm"] = float(sum(att_sel_v_norms) / len(att_sel_v_norms))
    if att_sel_o_norms:
        metrics["moe/att_sel_o_norm"] = float(sum(att_sel_o_norms) / len(att_sel_o_norms))

    if ff_bias_abs:
        metrics["moe/ff_bias_abs_mean"] = float(sum(ff_bias_abs) / len(ff_bias_abs))
    if att_v_bias_abs:
        metrics["moe/att_v_bias_abs_mean"] = float(sum(att_v_bias_abs) / len(att_v_bias_abs))
    if att_o_bias_abs:
        metrics["moe/att_o_bias_abs_mean"] = float(sum(att_o_bias_abs) / len(att_o_bias_abs))

    if ff_route_scales:
        metrics["moe/ff_route_scale"] = float(sum(ff_route_scales) / len(ff_route_scales))
    if att_route_scales:
        metrics["moe/att_route_scale"] = float(sum(att_route_scales) / len(att_route_scales))

    return metrics


def _device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _world_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


def create_dataloader(config: PretrainConfig, split: str, world_size: int) -> Tuple[DataLoader, Any]:
    per_replica_batch = config.global_batch_size // world_size
    if per_replica_batch == 0:
        raise ValueError(
            f"Global batch size {config.global_batch_size} must be >= world size {world_size}."
        )
    max_sequences_override: Optional[int] = None
    if config.max_sequences_per_epoch is not None:
        max_sequences_override = config.max_sequences_per_epoch
    elif config.max_sequences is not None:
        # Distribute sequences evenly per replica
        max_sequences_override = max(1, config.max_sequences // world_size)
    dataset_config = TextDatasetConfig(
        dataset_name=config.dataset_name,
        data_path=config.data_path,
        seq_len=config.seq_len,
        batch_size=per_replica_batch,
        seed=config.seed,
        tokenizer_name=config.tokenizer_name,
        max_sequences_per_epoch=config.max_sequences_per_epoch,
        streaming=config.streaming,
        max_sequences_override=max_sequences_override,
    )
    return create_dataset_loader(dataset_config, split=split)


def create_model(config: PretrainConfig, metadata, device: torch.device, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=getattr(metadata, "num_dataset_identifiers", 1),
        causal=True,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model: nn.Module = model_cls(model_cfg)
    if config.arch.loss.name:
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore[arg-type]

    if config.load_checkpoint:
        state_dict = torch.load(config.load_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    optimizers: List[torch.optim.Optimizer] = [
        torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    ]
    optimizer_lrs = [config.lr]

    if hasattr(model, "model") and hasattr(model.model, "puzzle_emb"):
        # ensure sparse embedding optimizer if puzzle embeddings exist
        puzzle_emb = getattr(model.model, "puzzle_emb", None)
        if puzzle_emb is not None:
            optimizers.insert(
                0,
                CastedSparseEmbeddingSignSGD_Distributed(
                    puzzle_emb.buffers(), lr=config.lr, weight_decay=0.0, world_size=world_size
                ),
            )
            optimizer_lrs.insert(0, config.lr)

    return model.to(device), optimizers, optimizer_lrs


def cosine_schedule(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def init_train_state(config: PretrainConfig, metadata, device: torch.device, world_size: int) -> TrainState:
    total_sequences = (
        int(config.max_sequences) if config.max_sequences is not None else int(metadata.mean_dataset_examples * config.epochs)
    )
    total_steps = max(1, math.ceil(total_sequences / config.global_batch_size))

    model, optimizers, optimizer_lrs = create_model(config, metadata, device, world_size)

    return TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=total_steps,
        sequences_consumed=0,
        total_sequences=total_sequences,
    )


def _update_learning_rates(config: PretrainConfig, state: TrainState) -> float:
    warmup_steps = int(config.lr_warmup_percent * state.total_steps)
    effective_lrs: List[float] = []
    for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
        lr_value = cosine_schedule(state.step, base_lr, warmup_steps, state.total_steps, config.lr_min_ratio)
        for group in optim.param_groups:
            group["lr"] = lr_value
        effective_lrs.append(lr_value)
    return effective_lrs[-1] if effective_lrs else config.lr


def _log_metrics(rank: int, metrics: Dict[str, float], extra: Dict[str, float], run: Optional[wandb.sdk.wandb_run.Run]) -> None:
    if rank != 0:
        return
    payload = {**metrics, **extra}
    if run is not None:
        run.log(payload)
    msg = ", ".join(f"{k}={v:.4f}" for k, v in payload.items())
    print(f"[train] {msg}")


def run_evaluation(config: PretrainConfig, state: TrainState, device: torch.device, rank: int, world_size: int) -> Optional[Dict[str, float]]:
    """Run evaluation on validation split."""
    if rank != 0:
        return None

    state.model.eval()

    # Create eval dataloader
    eval_loader, eval_metadata = create_dataloader(config, "validation", world_size)

    eval_losses = []
    eval_accuracies = []
    eval_steps_list = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch["inputs"].shape[0]

            # Initialize carry for eval
            carry = state.model.initial_carry(batch)

            # Run ACT loop until all sequences halt (max halt_max_steps iterations)
            # During eval, the model runs at max steps, but we still need to loop
            max_steps = config.arch.__pydantic_extra__.get("halt_max_steps", 4)

            cumulative_loss = 0.0
            cumulative_accuracy = 0.0
            cumulative_count = 0.0

            for step in range(max_steps):
                carry, loss, metrics, _, _ = state.model(carry=carry, batch=batch, return_keys=[])

                # Accumulate metrics - loss is already summed over batch, so normalize it
                if loss is not None:
                    cumulative_loss += (loss.item() / config.global_batch_size)

                if "accuracy" in metrics:
                    acc = metrics["accuracy"]
                    cumulative_accuracy += (acc.item() if torch.is_tensor(acc) else float(acc))

                if "count" in metrics:
                    cumulative_count += (metrics["count"].item() if torch.is_tensor(metrics["count"]) else float(metrics["count"]))
                else:
                    cumulative_count += batch_size

                # Check if all sequences have halted
                if carry.halted.all():
                    break

            # Average loss over ACT steps (already normalized per-token)
            eval_losses.append(cumulative_loss / (step + 1))

            if cumulative_count > 0:
                eval_accuracies.append(cumulative_accuracy / cumulative_count)

            # Limit to 10 batches for faster evaluation
            if len(eval_losses) >= 10:
                break

    # Compute averages
    avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
    avg_accuracy = sum(eval_accuracies) / len(eval_accuracies) if eval_accuracies else 0.0

    eval_metrics = {
        "eval/loss": avg_loss,
        "eval/perplexity": math.exp(min(avg_loss, 10.0)) if avg_loss > 0 else 1.0,
    }

    if eval_accuracies:
        eval_metrics["eval/accuracy"] = avg_accuracy

    print(f"[eval] loss={avg_loss:.4f}, perplexity={eval_metrics['eval/perplexity']:.4f}, accuracy={avg_accuracy:.4f}")

    state.model.train()
    return eval_metrics


def save_checkpoint(config: PretrainConfig, state: TrainState, rank: int, config_dict: Dict[str, Any]) -> None:
    """Save a training checkpoint with model weights and metadata."""
    if rank != 0 or not config.checkpoint_path:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(config.checkpoint_path, f"step_{state.step}.pt")

    checkpoint = {
        "model_state_dict": state.model.state_dict(),
        "config": config_dict,
        "step": state.step,
        "sequences_consumed": state.sequences_consumed,
    }

    torch.save(checkpoint, checkpoint_file)
    print(f"[checkpoint] Saved checkpoint to {checkpoint_file}")


def _setup_wandb(config: PretrainConfig, rank: int, config_dict: Dict[str, Any], model: nn.Module) -> Optional[wandb.sdk.wandb_run.Run]:
    mode = config.wandb_mode.lower()
    if mode == "disabled" or rank != 0:
        os.environ["WANDB_MODE"] = "disabled"
        return None
    os.environ["WANDB_MODE"] = mode
    if config.wandb_base_url:
        os.environ["WANDB_BASE_URL"] = config.wandb_base_url
        # Auto-load API key from .netrc for localhost URLs
        if "localhost" in config.wandb_base_url and not os.environ.get("WANDB_API_KEY"):
            try:
                from netrc import netrc
                from urllib.parse import urlparse

                parsed = urlparse(config.wandb_base_url)
                host = parsed.netloc or parsed.hostname
                auth = netrc().authenticators(host)
                if auth and auth[2]:
                    os.environ["WANDB_API_KEY"] = auth[2]
                    print(f"[wandb] Using API key from .netrc for {host}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"[wandb] Unable to auto-load local wandb API key: {exc}")
    project = config.project_name or "trm-tinystories"
    run_name = config.run_name or "-".join(coolname.generate(2))

    source_path = get_model_source_path(config.arch.name)
    wandb_run = wandb.init(
        project=project,
        name=run_name,
        config={**config_dict, "model_source": source_path},
    )

    # Log model size statistics
    if wandb_run is not None:
        total_params = sum(p.numel() for p in model.parameters())

        # Count embedding and core parameters
        embedding_params = 0
        core_params = 0

        for name, param in model.named_parameters():
            name_lower = name.lower()
            if 'embed' in name_lower or 'lm_head' in name_lower:
                embedding_params += param.numel()
            else:
                core_params += param.numel()

        model_size_stats = {
            "model_size/total": total_params,
            "model_size/embeddings": embedding_params,
            "model_size/core": core_params,
        }

        wandb_run.log(model_size_stats, step=0)
        wandb_run.summary.update(model_size_stats)

        print(f"Model parameters - Total: {total_params:,}, Embeddings: {embedding_params:,}, Core: {core_params:,}")

    return wandb_run


def train(config: PretrainConfig, device: torch.device, rank: int, world_size: int):
    train_loader, metadata = create_dataloader(config, "train", world_size)
    state = init_train_state(config, metadata, device, world_size)

    wandb_run = _setup_wandb(config, rank, config.model_dump(), state.model)

    # Update checkpoint path to include run name
    if wandb_run is not None and config.checkpoint_path:
        run_name = wandb_run.name
        config.checkpoint_path = os.path.join(config.checkpoint_path, run_name)
        if rank == 0:
            print(f"[checkpoint] Checkpoint path: {config.checkpoint_path}")

    if rank == 0:
        print(f"Using device: {device}")

    train_iterator = iter(train_loader)
    for step_idx in range(state.total_steps):
        batch = next(train_iterator)

        state.step = step_idx + 1

        batch_size = batch["inputs"].shape[0]
        state.sequences_consumed += batch_size * world_size

        batch = {k: v.to(device) for k, v in batch.items()}

        if state.carry is None:
            state.carry = state.model.initial_carry(batch)  # type: ignore[call-arg]

        lr_this_step = _update_learning_rates(config, state)

        state.carry, loss, metrics, _, _ = state.model(carry=state.carry, batch=batch, return_keys=[])

        (loss / config.global_batch_size).backward()

        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.grad_clip_norm)

        # Update expert biases for loss-free balancing (DeepSeek approach)
        # This is done after backward but before optimizer step, using no_grad
        with torch.no_grad():
            for module in state.model.modules():
                if isinstance(module, (SigmaMoE, SwitchHeadCore)):
                    if hasattr(module, 'update_expert_bias'):
                        module.update_expert_bias()

        for optim in state.optimizers:
            optim.step()
            optim.zero_grad()

        # Re-project selectors after optimiser updates so logits stay in range
        with torch.no_grad():
            for module in state.model.modules():
                if isinstance(module, (SigmaMoE, SwitchHeadCore)):
                    if getattr(module, "bias_balancing", False) and hasattr(module, "renorm_selectors"):
                        module.renorm_selectors()

        extra_logs = {"train/lr": lr_this_step}

        # Compute MoE diagnostic metrics if using MoEUT
        with torch.no_grad():
            moe_metrics = compute_moe_metrics(state.model)
            extra_logs.update(moe_metrics)

        if metrics:
            with torch.no_grad():
                metric_values = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}
            count = max(metric_values.get("count", float(batch_size)), 1.0)
            reduced = {}
            for key, value in metric_values.items():
                if key == "count":
                    continue
                denom = config.global_batch_size if key.endswith("loss") else count
                reduced[f"train/{key}"] = value / denom
            reduced.update(extra_logs)
            _log_metrics(
                rank,
                reduced,
                {
                    "train/step": float(state.step),
                    "train/sequences": float(min(state.sequences_consumed, state.total_sequences)),
                },
                wandb_run,
            )

        # Run evaluation if we've crossed an eval_interval_sequences boundary
        if config.eval_interval_sequences is not None:
            # Check if we've crossed an evaluation checkpoint
            prev_eval_checkpoint = (state.sequences_consumed - batch_size * world_size) // config.eval_interval_sequences
            curr_eval_checkpoint = state.sequences_consumed // config.eval_interval_sequences

            if curr_eval_checkpoint > prev_eval_checkpoint:
                eval_metrics = run_evaluation(config, state, device, rank, world_size)
                if eval_metrics is not None and wandb_run is not None:
                    eval_metrics["train/step"] = float(state.step)
                    eval_metrics["train/sequences"] = float(state.sequences_consumed)
                    wandb_run.log(eval_metrics, step=state.step)

        # Save checkpoint if we've crossed a checkpoint_interval_sequences boundary
        if config.checkpoint_interval_sequences is not None:
            prev_checkpoint = (state.sequences_consumed - batch_size * world_size) // config.checkpoint_interval_sequences
            curr_checkpoint = state.sequences_consumed // config.checkpoint_interval_sequences

            if curr_checkpoint > prev_checkpoint:
                save_checkpoint(config, state, rank, config.model_dump())

    # Save final checkpoint
    save_checkpoint(config, state, rank, config.model_dump())

    if rank == 0:
        print("Training loop complete.")

    if wandb_run is not None:
        artifact_dir = os.path.join(wandb_run.dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        if config.checkpoint_path and os.path.isdir(config.checkpoint_path):
            for name in os.listdir(config.checkpoint_path):
                src = os.path.join(config.checkpoint_path, name)
                dst = os.path.join(artifact_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        wandb_run.finish()


@hydra.main(version_base=None, config_path="config", config_name="cfg_tinystories_1m")
def main(cfg: DictConfig) -> None:
    raw_config = PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    device = _device()

    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    _seed_everything(raw_config.seed + rank)

    try:
        train(raw_config, device, rank, world_size)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    main()
