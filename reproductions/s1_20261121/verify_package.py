"""Read-only checks for the published S1/A0 reproducibility package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise AssertionError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def main() -> None:
    metadata = HERE / "data_protocol/cache_metadata"
    verified = json.loads((metadata / "verified.json").read_text())
    check(HERE / "data_protocol/manifests/audit.json", verified["audio_audit_sha256"])
    for cache_name, expected in verified["cache_metadata"].items():
        check(metadata / cache_name / "complete.json", expected)

    results = {}
    for seed in (666, 667):
        pair = {}
        for arm in ("A0", "S1"):
            folder = HERE / "artifacts" / f"{arm}_{seed}_formal"
            complete = json.loads((folder / "complete.json").read_text())
            protocol = json.loads((folder / "protocol.json").read_text())
            check(folder / "protocol.json", complete["protocol_sha256"])
            check(folder / "best.pt", complete["checkpoint_sha256"])
            check(folder / "dev_scores20.npz", complete["dev_scores20_sha256"])
            assert complete["status"] == "complete"
            assert complete["trained_epochs"] == 20 and complete["updates"] == 15880
            assert complete["seed"] == seed and complete["arm"] == arm
            assert protocol["train_uid_sha256"] == "3b02c95393d0de6c632e7313312f53650879c369efe1923baf78566772b186f7"
            assert protocol["dev_uid_sha256"] == "10bfc5d5e8ff2cf86b67702e48876b1115e23ec8f30352d73276958ea38c53e1"
            assert protocol["losses"]["main160"] == (0.0 if arm == "S1" else 1.0)
            assert protocol["losses"]["coarse_aux"] == 0.5
            assert len(json.loads((folder / "history.json").read_text())) == 21
            assert protocol["initial_backbone_sha256"] == json.loads(
                (HERE / "artifacts" / f"A0_{seed}_formal" / "protocol.json").read_text()
            )["initial_backbone_sha256"]
            if arm == "S1":
                source = REPO / "experiments" / (
                    "cross_resolution_supervision" if seed == 666 else "cross_resolution_multiseed"
                ) / "trainer.py"
                check(source, protocol["implementation_sha256"]["trainer.py"])
                check(
                    REPO / "experiments/cross_resolution_supervision/supervision.py",
                    protocol["implementation_sha256"]["supervision.py"],
                )
            pair[arm] = complete["event_f1_iou05_percent"]
        check(
            HERE / f"artifacts/A0_{seed}_formal/epochs/epoch_00.pt",
            {
                666: "9dfba6fcde6a3bf8a17787dc86be1d649d44c27646c4da5b46074c71b3b4d35a",
                667: "d878717974d9374bd3716eb24a51c3fb8162f5cecda174cdf6bce8adece22730",
            }[seed],
        )
        results[seed] = pair

    with (HERE / "evaluation/PAIRED_COMPARISONS.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            assert abs(float(row["a0_dev_event_f1_percent"]) - results[seed]["A0"]) < 1e-10
            assert abs(float(row["s1_dev_event_f1_percent"]) - results[seed]["S1"]) < 1e-10

    print("PASS: 4 formal 20-Epoch results, checkpoints, DEV logits, protocols and S1 source hashes")
    for seed, pair in results.items():
        print(f"seed {seed}: A0 {pair['A0']:.4f}%, S1 {pair['S1']:.4f}%")


if __name__ == "__main__":
    main()
