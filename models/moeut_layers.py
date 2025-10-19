"""
MoEUT sparse expert layers - exact copy from reference implementation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


def log_mean(x: torch.Tensor, dim: int = 0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return - (l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return - entropy_l(sel).mean()


class SigmaMoE(torch.nn.Module):
    """Sparse mixture of experts for MLP layer."""

    def __init__(self, dmodel: int, n_experts: int, expert_size: int, k: int,
                 activation=F.relu,
                 v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 balance_coef: float = 0.0,
                 bias_balancing: bool = False,
                 bias_update_rate: float = 0.001,
                 bias_momentum: float = 0.9,
                 bias_clip: Optional[float] = 0.2,
                 normalize_weights: bool = True):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation = activation
        self.expert_dropout = expert_dropout
        self.balance_coef = balance_coef
        self.bias_balancing = bias_balancing
        self.bias_update_rate = bias_update_rate
        self.bias_momentum = bias_momentum
        self.bias_clip = bias_clip
        self.normalize_weights = normalize_weights

        self.sel_hist = []
        self.last_balance_loss: Optional[torch.Tensor] = None

        # Loss-free balancing state (DeepSeek approach)
        if self.bias_balancing:
            self.register_buffer('expert_bias', torch.zeros(self.n_experts))
            self.register_buffer('expert_token_count', torch.zeros(self.n_experts))
            self.register_buffer('expert_running_load', torch.zeros(self.n_experts))

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))
        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))
        self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))
        self.register_buffer('selector_target_std', torch.tensor(1.0))
        self.route_scale = torch.nn.Parameter(torch.ones(1))

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.values, 0, std_scale / math.sqrt(self.n_experts * self.expert_size))

        target_std = self.expert_sel.std().clamp_min(1e-12)
        self.selector_target_std.fill_(target_std.item())
        self.renorm_keep_std(self.expert_sel, dim=1, target_std=self.selector_target_std)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0, target_std: Optional[torch.Tensor] = None):
        with torch.no_grad():
            norm = weight.norm(dim=dim, keepdim=True).clamp_min(1e-12)
            weight.div_(norm)
            if target_std is None:
                target = weight.std().detach()
            else:
                target = target_std.to(weight.device)
            current_std = weight.std().clamp_min(1e-12)
            weight.mul_(target / current_std)

    def _bias_scale(self) -> float:
        if self.n_experts <= 0:
            return 0.0
        base_std = float(self.selector_target_std)
        return base_std * math.sqrt(self.k_dim * max(self.n_experts, 1))

    def get_reg_loss(self) -> torch.Tensor:
        if not self.sel_hist:
            return 0

        # Average over time and layers.
        loss = entropy_reg(torch.stack(self.sel_hist, dim=-2).flatten(-3, -2), -2)
        self.sel_hist = []
        return loss

    def forward(self, input: torch.Tensor, sel_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        sel_logits = F.linear(sel_input if sel_input is not None else input, self.expert_sel, None)

        if self.training:
            self.sel_hist.append(sel_logits)

        # Compute UNBIASED probabilities for final weighting
        sel_probs_unbiased = torch.sigmoid(sel_logits)

        # For top-K selection, use BIASED logits if bias balancing is enabled
        if self.bias_balancing:
            sel_logits_biased = sel_logits + self.expert_bias
            sel_probs_for_topk = torch.sigmoid(sel_logits_biased)
        else:
            sel_probs_for_topk = sel_probs_unbiased

        # Apply expert dropout if needed
        sel_for_topk = sel_probs_for_topk
        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel_probs_for_topk) < self.expert_dropout
            sel_for_topk = sel_probs_for_topk.masked_fill(mask, float("-inf"))

        # Select top-K using BIASED probs (if enabled)
        _, sel_index = sel_for_topk.topk(self.n_heads, dim=-1, sorted=False)

        # But use UNBIASED probs for final weighting!
        sel_val = torch.gather(sel_probs_unbiased, -1, sel_index)
        if self.normalize_weights:
            sel_val = sel_val / sel_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        sel_val = sel_val * self.route_scale

        # Track expert usage for bias balancing
        if self.bias_balancing and self.training:
            self._update_expert_counts(sel_index)

        if self.balance_coef > 0.0:
            gate_probs = sel_probs_unbiased / sel_probs_unbiased.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            prob_mean = gate_probs.mean(dim=(0, 1))
            dispatch = F.one_hot(sel_index, num_classes=self.n_experts).float()
            total_assignments = float(sel_index.numel())
            load = dispatch.sum(dim=(0, 1, 2)) / max(total_assignments, 1.0)
            balance_loss = (prob_mean * load).sum() * self.n_experts
            self.last_balance_loss = self.balance_coef * balance_loss
        else:
            self.last_balance_loss = None

        self.last_selection_logits = sel_logits.detach()
        self.last_topk_indices = sel_index.detach()
        self.last_topk_scores = sel_val.detach()

        scores_all = torch.einsum('btd,edh->bteh', input, self.keys)
        scores_all = self.activation(scores_all)
        values_all = torch.einsum('bteh,ehv->btev', scores_all, self.values)
        selected = torch.gather(
            values_all,
            2,
            sel_index.unsqueeze(-1).expand(-1, -1, -1, self.v_dim)
        )
        out = (selected * sel_val.unsqueeze(-1)).sum(dim=2)
        return out

    @torch.no_grad()
    def renorm_selectors(self):
        """Re-project selector weights to keep logits in a stable range."""
        self.renorm_keep_std(self.expert_sel, dim=1, target_std=self.selector_target_std)

    def _update_expert_counts(self, sel_index: torch.Tensor):
        """Track token assignments to experts for bias balancing."""
        # sel_index shape: [batch, seq, k]
        dispatch = F.one_hot(sel_index, num_classes=self.n_experts).float()
        # Sum across batch, sequence, and k dimensions to get per-expert token counts
        self.expert_token_count += dispatch.sum(dim=(0, 1, 2))

    @torch.no_grad()
    def update_expert_bias(self):
        """Update expert biases based on load imbalance (DeepSeek loss-free balancing)."""
        if not self.bias_balancing:
            return
        if self.n_experts <= 1:
            self.expert_token_count.zero_()
            return

        # Calculate average load per expert
        total_tokens = self.expert_token_count.sum()
        if total_tokens == 0:
            return

        current_load = self.expert_token_count / total_tokens

        if self.bias_momentum > 0:
            self.expert_running_load.mul_(self.bias_momentum).add_(current_load, alpha=1 - self.bias_momentum)
            effective_load = self.expert_running_load
        else:
            effective_load = current_load

        target = 1.0 / self.n_experts
        load_error = effective_load - target
        # Positive error means over-used expert: reduce bias; negative -> boost bias
        scale = self._bias_scale()
        effective_rate = self.bias_update_rate * scale
        self.expert_bias.add_(-effective_rate * load_error / max(target, 1e-9))

        if self.bias_clip is not None:
            clip = self.bias_clip * scale
            self.expert_bias.clamp_(-clip, clip)

        # Reset token counts for next update
        self.expert_token_count.zero_()


class SwitchHeadCore(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 balance_coef: float = 0.0, bias_balancing: bool = False, bias_update_rate: float = 0.001,
                 bias_momentum: float = 0.9, bias_clip: Optional[float] = 0.2,
                 normalize_weights: bool = True):

        super().__init__()

        self.input_size = state_size
        self.output_size = state_size
        self.pe_size = self.input_size
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.n_experts = n_experts
        self.bias_balancing = bias_balancing
        update_scale = math.sqrt(max(self.n_experts, 1))
        self.bias_update_rate = bias_update_rate * update_scale

        self.sel_hist = []

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)
        self.balance_coef = balance_coef
        self.last_balance_loss: Optional[torch.Tensor] = None
        self.bias_momentum = bias_momentum
        self.bias_clip = bias_clip
        self.normalize_weights = normalize_weights

        # Loss-free balancing state (DeepSeek approach)
        # Separate biases for V and O expert selection
        if self.bias_balancing and self.n_experts > 1:
            self.register_buffer('expert_v_bias', torch.zeros(self.n_heads * self.n_experts))
            self.register_buffer('expert_o_bias', torch.zeros(self.n_heads * self.n_experts))
            self.register_buffer('expert_v_token_count', torch.zeros(self.n_heads * self.n_experts))
            self.register_buffer('expert_o_token_count', torch.zeros(self.n_heads * self.n_experts))
            self.register_buffer('expert_v_running_load', torch.zeros(self.n_heads * self.n_experts))
            self.register_buffer('expert_o_running_load', torch.zeros(self.n_heads * self.n_experts))

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size, self.projection_size))
            self.o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.projection_size, self.output_size))
            self.sel_v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
            self.register_buffer('sel_v_target_std', torch.tensor(1.0))
        else:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.projection_size, self.input_size))
            self.o = torch.nn.Parameter(torch.empty(self.output_size, self.n_heads * self.projection_size))
            self.register_buffer('sel_v_target_std', torch.tensor(1.0), persistent=False)

        self.sel_o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        self.register_buffer('sel_o_target_std', torch.tensor(1.0))
        self.route_scale = torch.nn.Parameter(torch.ones(1))

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    @torch.no_grad
    def reset_parameters(self, std_scale: float):
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, std_scale / math.sqrt(self.input_size))
            target_std_v = self.sel_v.std().clamp_min(1e-12)
            self.sel_v_target_std.fill_(target_std_v.item())
            self.renorm_rows(self.sel_v, self.sel_v_target_std)

        torch.nn.init.normal_(self.sel_o, 0, std_scale / math.sqrt(self.input_size))
        target_std_o = self.sel_o.std().clamp_min(1e-12)
        self.sel_o_target_std.fill_(target_std_o.item())
        self.renorm_rows(self.sel_o, self.sel_o_target_std)

        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.o, 0, std_scale / math.sqrt(self.n_heads * self.projection_size))

    def renorm_rows(self, x: torch.Tensor, target_std: torch.Tensor):
        with torch.no_grad():
            norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            x.div_(norm)
            target = target_std.to(x.device)
            current_std = x.std().clamp_min(1e-12)
            x.mul_(target / current_std)

    def _bias_scale_v(self) -> float:
        if self.n_experts <= 1:
            return 0.0
        base_std = float(self.sel_v_target_std)
        return base_std * math.sqrt(self.input_size * max(self.n_experts, 1))

    def _bias_scale_o(self) -> float:
        if self.n_experts <= 1:
            return 0.0
        base_std = float(self.sel_o_target_std)
        return base_std * math.sqrt(self.input_size * max(self.n_experts, 1))

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_sel(self, t: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sel_logits = F.linear(t, w).float()
        sel_raw = sel_logits.view(*sel_logits.shape[:-1], self.n_heads, -1)

        # Compute UNBIASED probabilities for final weighting
        sel_probs_unbiased = torch.sigmoid(sel_raw)

        # For top-K selection, use BIASED logits if bias balancing is enabled
        if bias is not None:
            bias_reshaped = bias.view(self.n_heads, -1)
            sel_raw_biased = sel_raw + bias_reshaped
            sel_probs_for_topk = torch.sigmoid(sel_raw_biased)
        else:
            sel_probs_for_topk = sel_probs_unbiased

        # Apply expert dropout if needed
        sel_for_topk = sel_probs_for_topk
        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel_probs_for_topk) < self.expert_dropout
            sel_for_topk = sel_probs_for_topk.masked_fill(mask, float('-inf'))

        # Select top-K using BIASED probs (if enabled)
        _, sel_index = sel_for_topk.topk(self.moe_k, dim=-1, sorted=False)

        # But use UNBIASED probs for final weighting!
        sel_val = torch.gather(sel_probs_unbiased, -1, sel_index)
        if self.normalize_weights:
            sel_val = sel_val / sel_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        sel_val = sel_val * self.route_scale
        return sel_index, sel_val, sel_raw, sel_probs_unbiased

    def get_reg_loss(self) -> torch.Tensor:
        loss = 0
        if self.sel_hist:
            for i in range(len(self.sel_hist[0])):
                loss = loss + entropy_reg(torch.stack([l[i] for l in self.sel_hist], dim=-3).flatten(-4,-3), -3)
        self.sel_hist = []
        return loss

    def _compute_balance_loss(self, probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gate_probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        prob_mean = gate_probs.mean(dim=(0, 1, 2))
        dispatch = F.one_hot(indices, num_classes=self.n_experts).float()
        total = float(indices.numel())
        load = dispatch.sum(dim=(0, 1, 2, 3)) / max(total, 1.0)
        return (prob_mean * load).sum() * self.n_experts

    def _update_expert_counts_v(self, sel_index: torch.Tensor):
        """Track token assignments to V experts for bias balancing."""
        # sel_index shape: [batch, seq, heads, k]
        dispatch = F.one_hot(sel_index, num_classes=self.n_experts).float()
        # Sum and flatten to per-expert-per-head counts
        counts = dispatch.sum(dim=(0, 1, 3))  # [heads, n_experts]
        self.expert_v_token_count += counts.flatten()

    def _update_expert_counts_o(self, sel_index: torch.Tensor):
        """Track token assignments to O experts for bias balancing."""
        # sel_index shape: [batch, seq, heads, k]
        dispatch = F.one_hot(sel_index, num_classes=self.n_experts).float()
        # Sum and flatten to per-expert-per-head counts
        counts = dispatch.sum(dim=(0, 1, 3))  # [heads, n_experts]
        self.expert_o_token_count += counts.flatten()

    @torch.no_grad()
    def update_expert_bias(self):
        """Update expert biases based on load imbalance (DeepSeek loss-free balancing)."""
        if not self.bias_balancing or self.n_experts <= 1:
            return

        # Update V expert biases
        total_tokens_v = self.expert_v_token_count.sum()
        if total_tokens_v > 0:
            current_v = self.expert_v_token_count / total_tokens_v
            if self.bias_momentum > 0:
                self.expert_v_running_load.mul_(self.bias_momentum).add_(current_v, alpha=1 - self.bias_momentum)
                effective_v = self.expert_v_running_load
            else:
                effective_v = current_v
            target_v = 1.0 / (self.n_heads * self.n_experts)
            load_error_v = effective_v - target_v
            scale_v = self._bias_scale_v()
            if scale_v > 0.0:
                effective_rate_v = self.bias_update_rate * scale_v
                self.expert_v_bias.add_(-effective_rate_v * load_error_v / max(target_v, 1e-9))
                if self.bias_clip is not None:
                    clip_v = self.bias_clip * scale_v
                    self.expert_v_bias.clamp_(-clip_v, clip_v)
            self.expert_v_token_count.zero_()

        # Update O expert biases
        total_tokens_o = self.expert_o_token_count.sum()
        if total_tokens_o > 0:
            current_o = self.expert_o_token_count / total_tokens_o
            if self.bias_momentum > 0:
                self.expert_o_running_load.mul_(self.bias_momentum).add_(current_o, alpha=1 - self.bias_momentum)
                effective_o = self.expert_o_running_load
            else:
                effective_o = current_o
            target_o = 1.0 / (self.n_heads * self.n_experts)
            load_error_o = effective_o - target_o
            scale_o = self._bias_scale_o()
            if scale_o > 0.0:
                effective_rate_o = self.bias_update_rate * scale_o
                self.expert_o_bias.add_(-effective_rate_o * load_error_o / max(target_o, 1e-9))
                if self.bias_clip is not None:
                    clip_o = self.bias_clip * scale_o
                    self.expert_o_bias.clamp_(-clip_o, clip_o)
            self.expert_o_token_count.zero_()


class RotaryPosEncoding(torch.nn.Module):
    # RoPE based on: https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rot(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, seq_dim: int, offset: int) -> torch.Tensor:
        sin = sin.narrow(seq_dim, offset, x.shape[seq_dim])
        cos = cos.narrow(seq_dim, offset, x.shape[seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                             seq_dim: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(k, sin, cos, seq_dim, 0)

    def get(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            self.cos_cached = emb.cos().view(*tgt_shape)
            self.sin_cached = emb.sin().view(*tgt_shape)

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)


class SwitchHeadRope(SwitchHeadCore):
    """Sparse attention with RoPE positional encoding."""

    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 projection_size: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 rotate_fraction: float = 0.5, rope_base: float = 10000, balance_coef: float = 0.0,
                 bias_balancing: bool = False, bias_update_rate: float = 0.001,
                 bias_momentum: float = 0.9, bias_clip: Optional[float] = 0.2,
                 normalize_weights: bool = True):

        super().__init__(
            state_size, n_heads, n_experts, dropout, projection_size, expert_dropout, moe_k, balance_coef,
            bias_balancing, bias_update_rate, bias_momentum, bias_clip, normalize_weights)

        self.n_rotate = int(rotate_fraction * self.projection_size)
        if self.n_rotate % 2 != 0:
            self.n_rotate -= 1
        if self.n_rotate < 2:
            self.n_rotate = 0
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.projection_size:
            r_k = k[..., :self.n_rotate]
            nr_k = k[..., self.n_rotate:]
            r_q = q[..., :self.n_rotate]
            nr_q = q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, r_k, offset)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.pe(q, k, offset)

    def forward(
        self,
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
    ) -> torch.Tensor:
        # *src: [batch_size, seq_len, hidden_dim]

        scale = self.scale.sqrt()

        q = self.q(q_src)
        k = self.k(k_src)
        q = q * scale.type_as(q)
        k = k * scale.type_as(k)

        o_gate = None
        if self.n_experts > 1:
            # Apply bias balancing if enabled
            v_bias = self.expert_v_bias if self.bias_balancing else None
            o_bias = self.expert_o_bias if self.bias_balancing else None

            v_idx, v_scores, v_logits, v_probs = self.get_sel(k_src, self.sel_v, v_bias)
            o_idx, o_scores, o_logits, o_probs = self.get_sel(q_src, self.sel_o, o_bias)

            # Track expert usage for bias balancing
            if self.bias_balancing and self.training:
                self._update_expert_counts_v(v_idx)
                self._update_expert_counts_o(o_idx)

            # Store diagnostic information for metrics
            self.last_v_topk = v_idx.detach()
            self.last_v_scores = v_scores.detach()
            self.last_v_logits = v_logits.detach()
            self.last_o_topk = o_idx.detach()
            self.last_o_scores = o_scores.detach()
            self.last_o_logits = o_logits.detach()

            if self.training:
                self.sel_hist.append((o_logits, v_logits))

            if self.balance_coef > 0.0:
                self.last_balance_loss = self.balance_coef * 0.5 * (
                    self._compute_balance_loss(v_probs, v_idx) +
                    self._compute_balance_loss(o_probs, o_idx)
                )
            else:
                self.last_balance_loss = None

            v_matrix = self.v.view(self.n_heads, self.n_experts, self.input_size, self.projection_size)
            v_all = torch.einsum('btd,hedp->bthep', v_src, v_matrix)
            v_selected = torch.gather(
                v_all,
                3,
                v_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.projection_size)
            )
            v = (v_selected * v_scores.unsqueeze(-1)).sum(dim=3).permute(0, 2, 1, 3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))
            self.last_balance_loss = None

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

        # Apply RoPE
        if self.n_rotate > 0:
            q, k = self.rotate(q, k, 0)

        q = self.dropout(q)

        # Scaled dot product attention (always causal for language modeling)
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, scale=1.0)
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            o_matrix = self.o.view(self.n_heads, self.n_experts, self.projection_size, self.output_size)
            o_all = torch.einsum('bthp,hepq->btheq', res, o_matrix)
            o_selected = torch.gather(
                o_all,
                3,
                o_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.output_size)
            )
            weighted = (o_selected * o_scores.unsqueeze(-1)).sum(dim=3)
            out = weighted.sum(dim=2)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out

    @torch.no_grad()
    def renorm_selectors(self):
        """Re-project selector weights to keep logits in a stable range."""
        if self.n_experts > 1:
            self.renorm_rows(self.sel_v, self.sel_v_target_std)
        self.renorm_rows(self.sel_o, self.sel_o_target_std)
