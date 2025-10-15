"""Dataset loader helpers for TinyStories language modeling."""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

from tinystories_dataset import TinyStoriesDataset, TinyStoriesDatasetConfig, TextDatasetMetadata


@dataclass
class TextDatasetConfig:
    dataset_name: str = "tinystories"
    data_path: str = "data/hf/TinyStories"
    seq_len: int = 128
    batch_size: int = 64
    seed: int = 42
    tokenizer_name: Optional[str] = None
    max_sequences_per_epoch: Optional[int] = None
    streaming: bool = True
    max_sequences_override: Optional[int] = None


def create_dataset_loader(config: TextDatasetConfig, split: str = "train") -> Tuple[DataLoader, TextDatasetMetadata]:
    """Build the TinyStories dataloader and metadata bundle."""
    if config.dataset_name != "tinystories":
        raise ValueError(f"Unsupported dataset '{config.dataset_name}'. Only 'tinystories' is available.")

    max_stories = config.max_sequences_per_epoch
    # When training with max_sequences, cap per-replica stories to avoid hanging streaming iterators.
    if max_stories is None and hasattr(config, "max_sequences_override") and config.max_sequences_override is not None:
        max_stories = config.max_sequences_override

    dataset = TinyStoriesDataset(
        TinyStoriesDatasetConfig(
            seed=config.seed,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            streaming=config.streaming,
            max_stories=max_stories,
            tokenizer_name=config.tokenizer_name,
            cache_dir=config.data_path,
        ),
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return dataloader, dataset.metadata
