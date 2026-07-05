
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

TASKS = ["copy", "reverse", "sort", "parity"]
DIGITS = [str(i) for i in range(10)]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<out>"]
TASK_TOKENS = [f"<task_{t}>" for t in TASKS]
LABEL_TOKENS = ["EVEN", "ODD"]

VOCAB = SPECIAL_TOKENS + TASK_TOKENS + DIGITS + LABEL_TOKENS
STOI = {tok: i for i, tok in enumerate(VOCAB)}
ITOS = {i: tok for tok, i in STOI.items()}

PAD_ID = STOI["<pad>"]
BOS_ID = STOI["<bos>"]
EOS_ID = STOI["<eos>"]
SEP_ID = STOI["<sep>"]
OUT_ID = STOI["<out>"]


def task_to_token(task: str) -> str:
    return f"<task_{task}>"


def oracle_support(task: str) -> Dict[str, List[List[int]]]:
    # Two layers. Four heads and four FF blocks per layer.
    support = {
        "copy": {
            "heads": [[1, 1, 0, 0], [1, 0, 1, 0]],
            "ff": [[1, 1, 0, 0], [1, 0, 1, 0]],
        },
        "reverse": {
            "heads": [[0, 1, 1, 0], [0, 1, 0, 1]],
            "ff": [[0, 1, 1, 0], [0, 1, 0, 1]],
        },
        "sort": {
            "heads": [[0, 0, 1, 1], [1, 1, 0, 0]],
            "ff": [[0, 0, 1, 1], [1, 1, 0, 0]],
        },
        "parity": {
            "heads": [[1, 0, 0, 1], [0, 0, 1, 1]],
            "ff": [[1, 0, 0, 1], [0, 0, 1, 1]],
        },
    }
    return support[task]


def generate_example(rng: random.Random, min_len: int = 3, max_len: int = 7) -> Dict:
    task = rng.choice(TASKS)
    n = rng.randint(min_len, max_len)
    digits = [rng.choice(DIGITS) for _ in range(n)]

    if task == "copy":
        target = digits[:]
    elif task == "reverse":
        target = list(reversed(digits))
    elif task == "sort":
        target = sorted(digits)
    elif task == "parity":
        total = sum(int(x) for x in digits)
        target = ["EVEN" if total % 2 == 0 else "ODD"]
    else:
        raise ValueError(task)

    prompt = ["<bos>", task_to_token(task), "<sep>"] + digits + ["<out>"]
    full = prompt + target + ["<eos>"]

    # Labels only apply after <out>, excluding the prompt.
    out_pos = prompt.index("<out>")
    labels = [-100] * len(prompt) + [STOI[tok] for tok in target] + [EOS_ID]

    keep_mask = []
    for tok in prompt:
        if tok in {"<bos>", "<sep>", "<out>"} or tok.startswith("<task_>"):
            keep_mask.append(1)
        else:
            keep_mask.append(0)
    full_keep_mask = keep_mask + [1] * (len(full) - len(prompt))

    return {
        "task": task,
        "digits": digits,
        "target": target,
        "prompt_tokens": prompt,
        "full_tokens": full,
        "input_ids": [STOI[tok] for tok in full],
        "labels": labels,
        "prompt_keep_oracle": full_keep_mask,
        "oracle_support": oracle_support(task),
    }


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class BenchmarkDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = load_jsonl(Path(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "task_id": torch.tensor(TASKS.index(row["task"]), dtype=torch.long),
            "prompt_keep_oracle": torch.tensor(row["prompt_keep_oracle"], dtype=torch.float32),
            "oracle_heads": torch.tensor(row["oracle_support"]["heads"], dtype=torch.float32),
            "oracle_ff": torch.tensor(row["oracle_support"]["ff"], dtype=torch.float32),
            "task": row["task"],
            "full_tokens": row["full_tokens"],
            "target": row["target"],
        }
        return item


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor | List]:
    max_len = max(len(x["input_ids"]) for x in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    keep_oracle = torch.zeros((batch_size, max_len), dtype=torch.float32)
    tasks = torch.zeros((batch_size,), dtype=torch.long)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    oracle_heads = []
    oracle_ff = []
    task_names = []
    full_tokens = []
    targets = []

    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        keep_oracle[i, :n] = item["prompt_keep_oracle"]
        tasks[i] = item["task_id"]
        lengths[i] = n
        oracle_heads.append(item["oracle_heads"])
        oracle_ff.append(item["oracle_ff"])
        task_names.append(item["task"])
        full_tokens.append(item["full_tokens"])
        targets.append(item["target"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "task_id": tasks,
        "lengths": lengths,
        "prompt_keep_oracle": keep_oracle,
        "oracle_heads": torch.stack(oracle_heads),
        "oracle_ff": torch.stack(oracle_ff),
        "task": task_names,
        "full_tokens": full_tokens,
        "target": targets,
    }
