'''
generate_benchmark.py generates benchmark datasets for training, validation, and testing.

Loads project root and imports generate_example and save_jsonl from src/csllm/data
Accepts CLI options for output directory, train/val/test sizes, and random seed
Creates the output directory if needed
Generates examples for each split using a seeded RNG
Saves each split as a JSONL file in the specified output folder
'''
# Andrew Kiruluta, UC Berkeley, May 2026
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


import argparse
import random
from csllm.data import generate_example, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/benchmark")
    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    splits = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    for split, n in splits.items():
        rows = [generate_example(rng) for _ in range(n)]
        save_jsonl(out_dir / f"{split}.jsonl", rows)
        print(f"Wrote {split}: {n} examples -> {out_dir / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()