'''
train_benchmark.py runs the model training pipeline using a YAML config.

- Sets up project root import path for src
- Parses CLI options for --config and --output_dir
- Calls train_from_config from train.py
- Prints the returned training metrics as formatted JSON
'''

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csllm.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    metrics = train_from_config(args.config, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
