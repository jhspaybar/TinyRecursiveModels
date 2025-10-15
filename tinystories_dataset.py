"""TinyStories dataset loader compatible with TinyRecursiveModels training loops."""

from dataclasses import dataclass, field
from typing import Any, Optional, Iterable, Dict
import itertools
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

from tokenizer_configs import get_tokenizer_for_dataset


@dataclass
class TinyStoriesDatasetConfig:
    seed: int = 42
    seq_len: int = 128
    batch_size: int = 64
    streaming: bool = True
    max_stories: Optional[int] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = "data/hf/TinyStories"


@dataclass
class TextDatasetMetadata:
    pad_id: int
    vocab_size: int
    seq_len: int
    num_dataset_identifiers: int = 1
    total_groups: int = 1
    mean_dataset_examples: int = 0
    sets: list = field(default_factory=lambda: ["train"])
    ignore_label_id: Any = None
    blank_identifier_id: int = 0


class TinyStoriesDataset(IterableDataset):
    """Stream or iterate over TinyStories yielding batched token sequences."""

    def __init__(self, config: TinyStoriesDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.tokenizer = get_tokenizer_for_dataset("tinystories", config.tokenizer_name)

        self._prepare_dataset()

        estimated = getattr(self, "estimated_stories", 2_100_000 if split == "train" else 21_000)
        if config.max_stories is not None:
            estimated = min(estimated, int(config.max_stories))

        self.metadata = TextDatasetMetadata(
            pad_id=self.tokenizer.pad_token_id,
            vocab_size=self.tokenizer.vocab_size,
            seq_len=config.seq_len,
            mean_dataset_examples=estimated,
        )
        self._epoch = 0

    def _prepare_dataset(self):
        cache_dir = None
        if self.config.cache_dir:
            cache_dir = os.path.abspath(self.config.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading TinyStories split='{self.split}' (streaming={self.config.streaming})")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=self.split,
            streaming=self.config.streaming,
            cache_dir=cache_dir,
        )
        self.dataset = dataset

        if self.config.streaming:
            self.estimated_stories = 2_100_000 if self.split == "train" else 21_000
            # When a finite number of stories is requested, materialise them eagerly to avoid
            # lingering streaming workers during short smoke tests.
            if self.config.max_stories is not None:
                print(f"Materialising {self.config.max_stories} TinyStories samples for local iteration")
                materialised = list(itertools.islice(self.dataset, int(self.config.max_stories)))
                self.dataset = materialised
                self.config.streaming = False
                self.estimated_stories = len(materialised)

        if not self.config.streaming:
            if isinstance(self.dataset, list):
                self.estimated_stories = len(self.dataset)
            else:
                self.estimated_stories = len(self.dataset)
            if self.config.max_stories is not None:
                limit = min(self.config.max_stories, self.estimated_stories)
                rng = np.random.Generator(np.random.Philox(seed=self.config.seed))
                self._fixed_indices = rng.choice(self.estimated_stories, size=limit, replace=False)
            else:
                self._fixed_indices = None
        else:
            self._fixed_indices = None

    def _tokenize_story(self, text: str) -> np.ndarray:
        text = text.strip() + self.tokenizer.eos_token
        target_len = self.config.seq_len + 1  # need one extra token for next-token labels
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=target_len,
        )
        tokens = np.asarray(tokens, dtype=np.int32)

        if tokens.shape[0] < target_len:
            pad_id = self.tokenizer.pad_token_id
            tokens = np.pad(tokens, (0, target_len - tokens.shape[0]), constant_values=pad_id)
        else:
            tokens = tokens[:target_len]

        return tokens

    def _iterate_streaming(self) -> Iterable[Dict[str, Any]]:
        dataset_iter = self.dataset.shuffle(seed=self.config.seed + self._epoch, buffer_size=10_000)
        if self.config.max_stories is not None:
            dataset_iter = itertools.islice(dataset_iter, self.config.max_stories)
        yield from dataset_iter

    def _iterate_offline(self) -> Iterable[Dict[str, Any]]:
        if self._fixed_indices is not None:
            base_indices = self._fixed_indices
        else:
            base_indices = np.arange(self.estimated_stories)
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._epoch))
        indices = rng.permutation(base_indices)
        for idx in indices:
            yield self.dataset[int(idx)]

    def __iter__(self):
        self._epoch += 1

        if self.config.streaming:
            dataset_iter = self._iterate_streaming()
        else:
            dataset_iter = self._iterate_offline()

        batch_tokens = []
        dataset_ids = []

        for item in dataset_iter:
            text = item.get("text", "")
            if not text or not text.strip():
                continue

            tokens = self._tokenize_story(text)
            batch_tokens.append(tokens)
            dataset_ids.append(0)

            if len(batch_tokens) == self.config.batch_size:
                batch_np = np.stack(batch_tokens)
                inputs = batch_np[:, :-1]
                labels = batch_np[:, 1:].copy()
                pad_id = self.tokenizer.pad_token_id
                labels[labels == pad_id] = -100

                batch = {
                    "inputs": torch.from_numpy(inputs).long(),
                    "labels": torch.from_numpy(labels).long(),
                    "dataset_ids": torch.zeros(self.config.batch_size, dtype=torch.long),
                    "puzzle_identifiers": torch.zeros(self.config.batch_size, dtype=torch.long),
                }
                yield batch
                batch_tokens.clear()
                dataset_ids.clear()

        # Yield remainder
        if batch_tokens:
            size = len(batch_tokens)
            batch_np = np.stack(batch_tokens)
            inputs = batch_np[:, :-1]
            labels = batch_np[:, 1:].copy()
            pad_id = self.tokenizer.pad_token_id
            labels[labels == pad_id] = -100

            batch = {
                "inputs": torch.from_numpy(inputs).long(),
                "labels": torch.from_numpy(labels).long(),
                "dataset_ids": torch.zeros(size, dtype=torch.long),
                "puzzle_identifiers": torch.zeros(size, dtype=torch.long),
            }
            yield batch
