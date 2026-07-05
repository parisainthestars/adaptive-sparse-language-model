
from __future__ import annotations

from typing import Dict, List

import torch

from .data import ITOS, PAD_ID, EOS_ID


def decode_target_tokens(token_ids: List[int]) -> List[str]:
    toks = []
    for idx in token_ids:
        if idx == EOS_ID or idx == PAD_ID or idx < 0:
            break
        toks.append(ITOS[int(idx)])
    return toks


def extract_predictions(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float | List[List[str]]]:
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0, "sequence_exact_match": 0.0, "pred_tokens": []}

    token_acc = ((preds == labels) & mask).sum().item() / mask.sum().item()

    pred_tokens = []
    gold_tokens = []
    exact = 0
    for i in range(labels.size(0)):
        gold = decode_target_tokens(labels[i][mask[i]].tolist())
        pred = decode_target_tokens(preds[i][mask[i]].tolist())
        pred_tokens.append(pred)
        gold_tokens.append(gold)
        exact += int(pred == gold)
    seq_em = exact / labels.size(0)
    return {
        "token_accuracy": token_acc,
        "sequence_exact_match": seq_em,
        "pred_tokens": pred_tokens,
        "gold_tokens": gold_tokens,
    }


def support_metrics(
    pred_head_masks: torch.Tensor,
    pred_ff_masks: torch.Tensor,
    oracle_heads: torch.Tensor,
    oracle_ff: torch.Tensor,
) -> Dict[str, float]:
    # compare sequence-averaged active support to oracle
    pred_h = (pred_head_masks.mean(dim=2) > 0.5).float()
    pred_f = (pred_ff_masks.mean(dim=2) > 0.5).float()

    tp = ((pred_h == 1) & (oracle_heads == 1)).sum() + ((pred_f == 1) & (oracle_ff == 1)).sum()
    fp = ((pred_h == 1) & (oracle_heads == 0)).sum() + ((pred_f == 1) & (oracle_ff == 0)).sum()
    fn = ((pred_h == 0) & (oracle_heads == 1)).sum() + ((pred_f == 0) & (oracle_ff == 1)).sum()

    tp = tp.item()
    fp = fp.item()
    fn = fn.item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # token support drift
    h_drift = (pred_head_masks[:, :, 1:] != pred_head_masks[:, :, :-1]).float().mean().item() if pred_head_masks.size(2) > 1 else 0.0
    f_drift = (pred_ff_masks[:, :, 1:] != pred_ff_masks[:, :, :-1]).float().mean().item() if pred_ff_masks.size(2) > 1 else 0.0

    active_support_fraction = 0.5 * (
        pred_head_masks.float().mean().item() + pred_ff_masks.float().mean().item()
    )
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "support_drift": 0.5 * (h_drift + f_drift),
        "active_support_fraction": active_support_fraction,
    }
