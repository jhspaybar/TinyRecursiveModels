# Language Modeling with Hierarchical Reasoning Models

This repository contains experiments in applying Hierarchical Reasoning Model (HRM) style architectures to language modeling tasks, with a focus on efficient training on Apple Silicon.

For a detailed writeup of this work, see: [Language Modeling with Hierarchical Reasoning Models](https://williamthurston.com/ml/language-models/transformers/2025/10/25/language-modeling-with-hierarchical-reasoning-models.html)

## Overview

We explore several architectural variants for language modeling on the TinyStories dataset:

- **Flat Transformer**: Standard transformer baseline for comparison
- **TRM (Tiny Recursive Model)**: Simplified recursive reasoning with L-level and H-level states
- **MoEUT (Mixture of Experts with Universal Transformers)**: Combining MoE with recursive processing
- **Adaptive Computation Time (ACT)**: Dynamic depth via learned halting

All models are trained from scratch on TinyStories (~2.1M sequences) and evaluated on their ability to generate coherent children's stories.

## Requirements

- Python 3.10+
- Apple Silicon with Metal Performance Shaders (MPS)
- [uv](https://github.com/astral-sh/uv) for dependency management

Install dependencies:

```bash
uv sync
```

## Training

Training uses Hydra configs located in `config/`. The main entrypoint is `pretrain_text.py`.

### Quick Test

```bash
uv run pretrain_text.py --config-name cfg_tinystories_1m \
  max_sequences=64 \
  global_batch_size=32 \
  streaming=true \
  wandb_mode=disabled
```

### Full Training Runs

**Flat Transformer (~1M params)**:
```bash
uv run pretrain_text.py --config-name cfg_tinystories_1m_flat
```

**TRM with MoEUT (~1M params)**:
```bash
uv run pretrain_text.py --config-name cfg_tinystories_1m_moeut
```

**TRM with MoEUT and ACT (~1M params)**:
```bash
uv run pretrain_text.py --config-name cfg_tinystories_1m_moeut_act
```

**Larger MoEUT variant (~10M params)**:
```bash
uv run pretrain_text.py --config-name cfg_tinystories_10m_moeut
```

## Generation

Generate stories from trained checkpoints:

```bash
uv run python generate_stories.py \
  --checkpoint-path runs/checkpoints/trm_tinystories_1m/your-run-name/step_12345.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 512
```

Run comprehensive generation sweeps across all checkpoints:

```bash
uv run python sweep_generation.py
```

This tests each checkpoint with:
- 10 diverse prompts (in/out of distribution)
- 8 generation configs (varying temperature, top-k, ACT settings)
- Results saved as `generation_sweep_results.json` in each checkpoint directory

## Dataset

- **Default**: Streams from Hugging Face (`roneneldan/TinyStories`)
- **Offline**: Download with `datasets-cli download roneneldan/TinyStories` to `data/hf/TinyStories`

## Metrics & Monitoring

Training metrics are logged to Weights & Biases:
- Local server: `http://localhost:8080` (default)
- Set `wandb_mode=disabled` to skip logging
- Set `wandb_mode=offline` for local-only logging

## Architecture Highlights

### MoEUT Features
- **Sigmoid-based routing**: Sparse expert selection in FFN and attention layers
- **Bias balancing**: DeepSeek-style loss-free load balancing via routing bias adjustment
- **Post-normalization**: RMS norm after attention/FFN for numerical stability

### Key Fix: Post-Norm for MoEUT
In `models/recursive_reasoning/trm.py:152`, we always apply post-normalization for MoEUT to prevent activation explosions:

```python
# Always use post-norm for MoEUT to prevent activation explosions
self._use_moeut_post_norm = self.config.use_moeut
```

This fix was critical for stable generation when `halt_max_steps == 1`.

## Testing

```bash
uv run --group dev pytest
```

## Reference

This code is based on:
- [Tiny Recursive Model (TRM)](https://arxiv.org/abs/2510.04871) by Alexia Jolicoeur-Martineau
- [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734) by Wang et al.
- Original HRM codebase: https://github.com/sapientinc/HRM

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}

@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```
