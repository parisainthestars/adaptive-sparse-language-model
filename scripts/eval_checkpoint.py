import argparse
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csllm.data import collate_batch, BenchmarkDataset
from csllm.model import CSLLM, ModelConfig
from csllm.train import evaluate, load_config
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = CSLLM(ModelConfig(**cfg["model"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = BenchmarkDataset(Path(cfg["data"]["data_dir"]) / f"{args.split}.jsonl")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_batch)
    metrics = evaluate(model, loader, torch.device("cpu"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
