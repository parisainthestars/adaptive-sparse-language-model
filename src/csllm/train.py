
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch

torch.set_num_threads(1)
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BenchmarkDataset, collate_batch
from .metrics import extract_predictions, support_metrics
from .model import CSLLM, ModelConfig


def load_config(config_path: str | Path) -> Dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dataloaders(data_dir: str | Path, batch_size: int) -> Dict[str, DataLoader]:
    data_dir = Path(data_dir)
    return {
        split: DataLoader(
            BenchmarkDataset(data_dir / f"{split}.jsonl"),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_batch,
        )
        for split in ["train", "val", "test"]
    }


def compute_loss(batch: Dict, out: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    logits = out["logits"]
    labels = batch["labels"].to(logits.device)
    lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    keep_oracle = batch["prompt_keep_oracle"].to(logits.device)
    keep_prob = out["prompt_keep_prob"]
    keep_loss = F.binary_cross_entropy(keep_prob, keep_oracle)

    oracle_heads = batch["oracle_heads"].to(logits.device)
    oracle_ff = batch["oracle_ff"].to(logits.device)
    pred_heads = out["head_masks"].mean(dim=2)
    pred_ff = out["ff_masks"].mean(dim=2)
    support_loss = F.binary_cross_entropy(pred_heads, oracle_heads) + F.binary_cross_entropy(pred_ff, oracle_ff)

    budget = out["measurement_budget"].float()
    budget_penalty = budget.mean() / cfg["model"]["measurement_max"]

    total = (
        lm_loss
        + cfg["train"]["lambda_keep"] * keep_loss
        + cfg["train"]["lambda_support"] * support_loss
        + cfg["train"]["lambda_temporal"] * out["temporal_smoothness"]
        + cfg["train"]["lambda_budget"] * budget_penalty
    )
    return {
        "total": total,
        "lm_loss": lm_loss,
        "keep_loss": keep_loss,
        "support_loss": support_loss,
        "budget_penalty": budget_penalty,
    }


@torch.no_grad()
def evaluate(model: CSLLM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    agg = {
        "loss": 0.0,
        "token_accuracy": 0.0,
        "sequence_exact_match": 0.0,
        "prompt_compression_ratio": 0.0,
        "measurement_budget_mean": 0.0,
        "support_precision": 0.0,
        "support_recall": 0.0,
        "support_f1": 0.0,
        "support_drift": 0.0,
        "active_support_fraction": 0.0,
    }
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        task_ids = batch["task_id"].to(device)
        out = model(input_ids, task_ids)
        loss = F.cross_entropy(
            out["logits"].view(-1, out["logits"].size(-1)),
            batch["labels"].to(device).view(-1),
            ignore_index=-100,
        )
        pred_m = extract_predictions(out["logits"].cpu(), batch["labels"])
        supp_m = support_metrics(
            out["head_masks"].cpu(),
            out["ff_masks"].cpu(),
            batch["oracle_heads"],
            batch["oracle_ff"],
        )

        prompt_keep = out["prompt_keep_prob"].detach().cpu()
        keep_oracle = batch["prompt_keep_oracle"]
        prompt_ratio = (prompt_keep * keep_oracle).sum().item() / keep_oracle.sum().item()

        agg["loss"] += loss.item()
        agg["token_accuracy"] += pred_m["token_accuracy"]
        agg["sequence_exact_match"] += pred_m["sequence_exact_match"]
        agg["prompt_compression_ratio"] += prompt_ratio
        agg["measurement_budget_mean"] += out["measurement_budget"].float().mean().item()
        for k, v in supp_m.items():
            agg[k] += v
        n_batches += 1

    for k in agg:
        agg[k] /= max(n_batches, 1)
    return agg


def train_from_config(config_path: str | Path, output_dir: str | Path) -> Dict:
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", False) else "cpu")
    torch.manual_seed(cfg["train"]["seed"])

    model_cfg = ModelConfig(**cfg["model"])
    model = CSLLM(model_cfg).to(device)
    loaders = make_dataloaders(cfg["data"]["data_dir"], cfg["train"]["batch_size"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = {"total": 0.0, "lm_loss": 0.0, "keep_loss": 0.0, "support_loss": 0.0, "budget_penalty": 0.0}
        n_batches = 0

        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            task_ids = batch["task_id"].to(device)
            out = model(input_ids, task_ids)
            losses = compute_loss(batch, out, cfg)

            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        train_stats = {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}
        val_stats = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)

        print(f"Epoch {epoch}: train_total={train_stats['train_total']:.4f} val_loss={val_stats['loss']:.4f} val_em={val_stats['sequence_exact_match']:.4f}")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save({"model_state": best_state, "model_config": asdict(model_cfg), "config": cfg}, out_dir / "checkpoint.pt")

    # Reload best
    model.load_state_dict(best_state)
    train_eval = evaluate(model, loaders["train"], device)
    val_eval = evaluate(model, loaders["val"], device)
    test_eval = evaluate(model, loaders["test"], device)

    metrics = {
        "train": train_eval,
        "val": val_eval,
        "test": test_eval,
        "best_val_loss": best_val,
        "epochs": cfg["train"]["epochs"],
    }

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(7, 4))
    df = pd.DataFrame(history)
    plt.plot(df["epoch"], df["train_total"], label="train_total")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()

    return metrics
