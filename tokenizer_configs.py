"""Tokenizer configuration helpers for TinyRecursiveModels text training."""

from typing import Optional, Any

from transformers import AutoTokenizer, T5Tokenizer


class TokenizerConfig:
    """Factory for constructing tokenizers with optional vocabulary limits."""

    @staticmethod
    def get_tokenizer(tokenizer_name: str, vocab_limit: Optional[int] = None):
        """
        Return a tokenizer configured for TinyStories experiments.

        Supported names:
            - ``t5-small``: canonical T5 tokenizer (32k vocab).
            - ``gpt-neo-125M``: GPT-Neo tokenizer (50,257 vocab).
            - ``gpt-neo-10k``: GPT-Neo tokenizer truncated to the most common 10k tokens
              (matches TinyStories paper setup).
        """
        if tokenizer_name == "t5-small":
            tokenizer = T5Tokenizer.from_pretrained("t5-small")

        elif tokenizer_name == "gpt-neo-125M":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        elif tokenizer_name == "gpt-neo-10k":
            base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            if base_tokenizer.pad_token is None:
                base_tokenizer.pad_token = base_tokenizer.eos_token
            tokenizer = LimitedVocabTokenizer(base_tokenizer, vocab_limit=10_000)

        else:
            raise ValueError(f"Unknown tokenizer '{tokenizer_name}'")

        if vocab_limit and tokenizer_name != "gpt-neo-10k":
            tokenizer = LimitedVocabTokenizer(tokenizer, vocab_limit=vocab_limit)

        return tokenizer


class LimitedVocabTokenizer:
    """Wrap a tokenizer and remap token ids to a fixed-size vocabulary."""

    def __init__(self, base_tokenizer: Any, vocab_limit: int = 10_000):
        self.base_tokenizer = base_tokenizer
        self.vocab_limit = int(vocab_limit)

        self.special_tokens = {
            getattr(base_tokenizer, "pad_token_id", None),
            getattr(base_tokenizer, "eos_token_id", None),
            getattr(base_tokenizer, "bos_token_id", None),
            getattr(base_tokenizer, "unk_token_id", None),
        }
        self.special_tokens = {tok for tok in self.special_tokens if tok is not None}

        self._build_token_maps()

    def _build_token_maps(self):
        num_special = len(self.special_tokens)
        num_regular = max(0, self.vocab_limit - num_special)

        self.forward_map = {}
        new_id = 0

        for token_id in sorted(self.special_tokens):
            self.forward_map[token_id] = new_id
            new_id += 1

        tokens_added = 0
        for original_id in range(len(self.base_tokenizer)):
            if original_id in self.special_tokens:
                continue
            if tokens_added >= num_regular:
                break
            self.forward_map[original_id] = new_id
            new_id += 1
            tokens_added += 1

        self.reverse_map = {v: k for k, v in self.forward_map.items()}
        self._vocab_size = len(self.forward_map)

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs):
        full_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        limited = []
        unk_id = getattr(self.base_tokenizer, "unk_token_id", None)
        mapped_unk = self.forward_map.get(unk_id, None)
        for token_id in full_ids:
            new_id = self.forward_map.get(token_id, mapped_unk)
            if new_id is not None:
                limited.append(new_id)
        return limited

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        original_ids = [self.reverse_map[token_id] for token_id in token_ids if token_id in self.reverse_map]
        return self.base_tokenizer.decode(original_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def pad_token_id(self):
        token_id = getattr(self.base_tokenizer, "pad_token_id", None)
        if token_id is None:
            return 0
        return self.forward_map.get(token_id, 0)

    @property
    def eos_token_id(self):
        token_id = getattr(self.base_tokenizer, "eos_token_id", None)
        if token_id is None:
            return 1
        return self.forward_map.get(token_id, 1)

    @property
    def pad_token(self):
        return getattr(self.base_tokenizer, "pad_token", "<pad>")

    @property
    def eos_token(self):
        return getattr(self.base_tokenizer, "eos_token", "</s>")


def get_tokenizer_for_dataset(dataset_name: str, tokenizer_override: Optional[str] = None) -> Any:
    """Return the configured tokenizer for the requested dataset."""
    if tokenizer_override:
        tokenizer_name = tokenizer_override
    else:
        defaults = {
            "tinystories": "gpt-neo-10k",
        }
        tokenizer_name = defaults.get(dataset_name, "gpt-neo-10k")

    return TokenizerConfig.get_tokenizer(tokenizer_name)
