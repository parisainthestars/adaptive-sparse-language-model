
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_ID, OUT_ID


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


def straight_through_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    # scores: [..., G]
    topk = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, topk, 1.0)
    soft = torch.sigmoid(scores)
    return hard + soft - soft.detach()


class PromptCompressor(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compress only prompt-side digit tokens; preserve special/task tokens
        raw = self.scorer(x).squeeze(-1)
        keep_prob = torch.sigmoid(raw)

        special = (
            (input_ids == PAD_ID)
            | (input_ids == OUT_ID)
            | (input_ids < 9)  # pad,bos,eos,sep,out + task tokens
        )
        # force keep for specials; others are soft-kept.
        keep_prob = torch.where(special, torch.ones_like(keep_prob), keep_prob)
        x = x * keep_prob.unsqueeze(-1)
        return x, keep_prob


class SupportController(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_tasks: int,
        n_layers: int,
        n_heads: int,
        n_ff_blocks: int,
        m_max: int,
        base_measurements: int,
        min_measurements: int,
        gamma_entropy: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_ff_blocks = n_ff_blocks
        self.m_max = m_max
        self.base_measurements = base_measurements
        self.min_measurements = min_measurements
        self.gamma_entropy = gamma_entropy

        self.task_embed = nn.Embedding(n_tasks, d_model)
        bank = torch.randn(n_tasks, m_max, d_model) / math.sqrt(d_model)
        self.register_buffer("measurement_bank", bank)

        self.head_decoder = nn.ModuleList([nn.Linear(m_max, n_heads) for _ in range(n_layers)])
        self.ff_decoder = nn.ModuleList([nn.Linear(m_max, n_ff_blocks) for _ in range(n_layers)])

        self.uncertainty_head = nn.Linear(d_model, 20)

    def _measurement_count(self, entropy: torch.Tensor) -> torch.Tensor:
        # entropy is [B, T] in nats, max about log(vocab)
        m = torch.floor(self.base_measurements * (1.0 + self.gamma_entropy * entropy))
        m = torch.clamp(m, min=self.min_measurements, max=self.m_max)
        return m.long()

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape
        device = x.device
        task_vec = self.task_embed(task_ids)[:, None, :]
        u = x + task_vec

        cheap_logits = self.uncertainty_head(u)
        probs = F.softmax(cheap_logits, dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)
        entropy = entropy / math.log(cheap_logits.size(-1))
        m_t = self._measurement_count(entropy)

        bank = self.measurement_bank[task_ids]  # [B, m_max, D]
        # Full sketch first, then progressively mask rows according to m_t.
        z_full = torch.einsum("bmd,btd->btm", bank, u)
        row_idx = torch.arange(self.m_max, device=device)[None, None, :]
        active_rows = (row_idx < m_t.unsqueeze(-1)).float()
        z = z_full * active_rows

        head_scores = []
        ff_scores = []
        for l in range(self.n_layers):
            head_scores.append(self.head_decoder[l](z))
            ff_scores.append(self.ff_decoder[l](z))
        head_scores = torch.stack(head_scores, dim=1)  # [B, L, T, H]
        ff_scores = torch.stack(ff_scores, dim=1)      # [B, L, T, Bf]

        return {
            "entropy": entropy,
            "measurement_budget": m_t,
            "head_scores": head_scores,
            "ff_scores": ff_scores,
        }


class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        scores = scores + causal_mask(T, x.device)[None, None, :, :]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, T, d_head]
        out = out.transpose(1, 2)  # [B, T, H, d_head]

        out = out * head_mask.unsqueeze(-1)  # [B, T, H, d_head]
        out = out.reshape(B, T, D)
        return self.out_proj(out)


class SparseFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, ff_mask: torch.Tensor) -> torch.Tensor:
        outs = []
        for block in self.blocks:
            outs.append(block(x))
        stacked = torch.stack(outs, dim=2)  # [B, T, n_blocks, D]
        masked = stacked * ff_mask.unsqueeze(-1)
        denom = ff_mask.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        return masked.sum(dim=2) / denom.squeeze(-1)


class DynamicTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int, n_ff_blocks: int):
        super().__init__()
        self.attn = SparseMultiheadAttention(d_model, n_heads)
        self.ff = SparseFFN(d_model, ff_hidden, n_ff_blocks)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, head_mask: torch.Tensor, ff_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), head_mask)
        x = x + self.ff(self.ln2(x), ff_mask)
        return x


@dataclass
class ModelConfig:
    vocab_size: int = 20
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_hidden: int = 96
    n_ff_blocks: int = 4
    max_seq_len: int = 32
    prompt_keep_threshold: float = 0.5
    active_heads: int = 2
    active_ff_blocks: int = 2
    measurement_max: int = 16
    measurement_base: int = 8
    measurement_min: int = 4
    gamma_entropy: float = 0.6
    n_tasks: int = 4


class CSLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.prompt_compressor = PromptCompressor(cfg.d_model)
        self.controller = SupportController(
            d_model=cfg.d_model,
            n_tasks=cfg.n_tasks,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_ff_blocks=cfg.n_ff_blocks,
            m_max=cfg.measurement_max,
            base_measurements=cfg.measurement_base,
            min_measurements=cfg.measurement_min,
            gamma_entropy=cfg.gamma_entropy,
        )
        self.layers = nn.ModuleList(
            [
                DynamicTransformerLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ff_hidden=cfg.ff_hidden,
                    n_ff_blocks=cfg.n_ff_blocks,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, task_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)[None, :]
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        x, keep_prob = self.prompt_compressor(x, input_ids)
        ctrl = self.controller(x, task_ids)
        head_scores = ctrl["head_scores"]
        ff_scores = ctrl["ff_scores"]

        head_masks = []
        ff_masks = []
        temporal_smoothness = 0.0

        for l, layer in enumerate(self.layers):
            head_mask = straight_through_topk(head_scores[:, l], self.cfg.active_heads)
            ff_mask = straight_through_topk(ff_scores[:, l], self.cfg.active_ff_blocks)
            head_masks.append(head_mask)
            ff_masks.append(ff_mask)
            if T > 1:
                temporal_smoothness = temporal_smoothness + (head_mask[:, 1:] - head_mask[:, :-1]).pow(2).mean()
                temporal_smoothness = temporal_smoothness + (ff_mask[:, 1:] - ff_mask[:, :-1]).pow(2).mean()
            x = layer(x, head_mask, ff_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        head_masks = torch.stack(head_masks, dim=1)  # [B, L, T, H]
        ff_masks = torch.stack(ff_masks, dim=1)

        return {
            "logits": logits,
            "prompt_keep_prob": keep_prob,
            "measurement_budget": ctrl["measurement_budget"],
            "entropy": ctrl["entropy"],
            "head_masks": head_masks,
            "ff_masks": ff_masks,
            "temporal_smoothness": temporal_smoothness if isinstance(temporal_smoothness, torch.Tensor) else torch.tensor(temporal_smoothness, device=input_ids.device),
        }
