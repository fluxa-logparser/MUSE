# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for one MUSE training and inference pipeline."""
import argparse
from pathlib import Path

import torch

from .pipeline import fit, load_config, prepare, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "prepare", "score"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config, paths = load_config(args.config)
    {"fit": fit, "prepare": prepare, "score": score}[args.command](config, paths, args.work.resolve())


if __name__ == "__main__":
    main()
