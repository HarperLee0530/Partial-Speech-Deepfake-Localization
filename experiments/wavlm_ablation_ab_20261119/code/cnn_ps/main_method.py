"""Train, freeze and evaluate frozen WavLM + CNN + boundary auxiliary design.

WavLM features are cached with a frozen public encoder. The projection, CNN
and adapter are trained jointly. Use a new output directory for a new run.
Existing paired runs retain their original train_wavlm frozen plans.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import train_wavlm


MAIN_ARM = "wavlm_frozen"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train the main method from random initialization")
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--dev-cache", type=Path, required=True)
    train.add_argument("--train-ssl", type=Path, required=True)
    train.add_argument("--dev-ssl", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=8)
    freeze = commands.add_parser("freeze", help="Freeze a completed main-only run using dev selection")
    evaluate = commands.add_parser("evaluate", help="Evaluate with the frozen dev threshold")
    evaluate.add_argument("--eval-cache", type=Path, required=True)
    evaluate.add_argument("--eval-ssl", type=Path, required=True)
    for command in (train, freeze, evaluate):
        command.add_argument("--output", type=Path, required=True,
                             help="Root of a main-only run; its model is stored in wavlm_frozen/")
        command.add_argument("--seed", type=int, default=666)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--device", choices=("cpu", "cuda"),
                             default="cpu" if command is freeze else "cuda")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    forwarded = (["--arm", MAIN_ARM] if command == "train" else
                 ["--arms", MAIN_ARM, "--freeze-only" if command == "freeze" else "--evaluate"])
    for name, value in args.items():
        forwarded.extend(["--" + name.replace("_", "-"), str(value)])
    return train_wavlm.main(forwarded)


if __name__ == "__main__":
    main()
