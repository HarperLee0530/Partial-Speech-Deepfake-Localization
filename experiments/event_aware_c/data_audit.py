"""Prepare fixed TRAIN/DEV labels and audit the original reduced WAV inventory."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf

from cnn_ps.io import manifest_hash, read_json, save_json, sha256
from cnn_ps.manifests import LABEL_ARCHIVE_SHA256, relative

SPLITS = ("train", "dev", "eval")
EXPECTED = dict(train=(6345, "3b02c95393d0de6c632e7313312f53650879c369efe1923baf78566772b186f7"),
                dev=(2484, "10bfc5d5e8ff2cf86b67702e48876b1115e23ec8f30352d73276958ea38c53e1"),
                eval=(3562, "203792d5718e75e54614ad005386a3037cd9ddb83dd6dc7a93266dfc9c4252b1"))
ROLES = dict(train={"train_detector", "train_edit"}, dev={"dev"}, eval={"eval"})
CANONICAL_FIXED_SHA256 = "f71b9e0a74373e436d6d4db1fd53d73ad70e966ca2dbe3da0af18c76679bcd12"


def check_fixed(ids):
    fixed = read_json(ids)
    canonical = json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != CANONICAL_FIXED_SHA256:
        raise ValueError("Fixed UID or role assignment differs from the pinned manifest")
    seen = set()
    for split in SPLITS:
        rows = fixed[split]
        uids = [r["uid"] for r in rows]
        count, digest = EXPECTED[split]
        if (len(uids) != count or len(set(uids)) != count or manifest_hash(uids) != digest
                or seen.intersection(uids) or {r["role"] for r in rows} != ROLES[split]):
            raise ValueError(f"Fixed manifest diverges from the historical {split} protocol")
        seen.update(uids)
    return fixed


def extract_training_labels(archive, fixed, output):
    if sha256(archive) != LABEL_ARCHIVE_SHA256:
        raise ValueError("Expected original PartialSpoof v1.2 label archive SHA-256")
    output.mkdir(parents=True, exist_ok=False)
    expected = {f"database/segment_labels/{split}_seglab_{grid}.npy": (split, grid)
                for split in ("train", "dev") for grid in ("0.02", "0.16")}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in expected:
                continue
            split, grid = expected.pop(member.name)
            raw = tar.extractfile(member).read()
            # Deserialize only after matching the official pinned archive SHA-256.
            source = np.load(io.BytesIO(raw), allow_pickle=True).item()
            selected = {}
            for row in fixed[split]:
                uid = row["uid"]
                y = np.asarray(source[uid], dtype=np.float64)
                if y.ndim != 1 or not y.size or not np.isin(y, (0, 1)).all():
                    raise ValueError(f"{uid}: expected official binary {grid}s labels")
                selected[uid] = (1 - y if grid == "0.02" else y).astype(np.uint8)
            if grid == "0.02":
                np.savez_compressed(output / f"{split}.npz", **selected)
            else:
                # Match A0's raw coarse label bytes and encoding.
                (output / f"{split}_seglab_0.16.npy").write_bytes(raw)
    if expected:
        raise ValueError(f"Missing expected official labels: {list(expected)}")
    save_json(output / "provenance.json", dict(official_archive_sha256=LABEL_ARCHIVE_SHA256,
              files={p.name: sha256(p) for p in sorted(output.iterdir())}))


def audio_inventory(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    result = {r["relative_path"]: r["sha256"].lower() for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate files in original audio inventory")
    return result


def save_source_inventory(reduced, output, fixed):
    """Run on the original reduced data before transferring it."""
    with output.open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["relative_path", "sha256"])
        writer.writeheader()
        for split in ("train", "dev"):
            for row in fixed[split]:
                rel = f"{row['role']}/audio/{row['uid']}.wav"
                writer.writerow(dict(relative_path=rel, sha256=sha256(reduced / rel)))


def audit_audio(args, fixed):
    if args.output.exists():
        raise FileExistsError(args.output)
    reduced = args.reduced.resolve()
    inventory = audio_inventory(args.original_sha256_inventory)
    validated = {}
    manifests = {}
    contents = {}
    provenance = read_json(args.labels / "provenance.json")
    if provenance["official_archive_sha256"] != LABEL_ARCHIVE_SHA256:
        raise ValueError("Fine label provenance mismatch")
    for split in ("train", "dev"):
        files = list(fixed[split])
        rows = []
        coarse_path = args.labels / f"{split}_seglab_0.16.npy"
        fine_path = args.labels / f"{split}.npz"
        for path in (coarse_path, fine_path):
            if sha256(path) != provenance["files"][path.name]:
                raise ValueError(f"Changed official label extraction: {path}")
        coarse = np.load(coarse_path, allow_pickle=True).item()
        with np.load(fine_path, allow_pickle=False) as fine:
            if fine.files != [r["uid"] for r in files]:
                raise ValueError(f"{split}: fine label UID keys or order differ")
            for record in files:
                uid, role = record["uid"], record["role"]
                rel = f"{role}/audio/{uid}.wav"
                path = (reduced / rel).resolve(strict=True)
                if not path.is_relative_to(reduced):
                    raise ValueError(f"{uid}: audio resolved outside reduced root")
                digest = sha256(path)
                if inventory.get(rel) != digest:
                    raise ValueError(f"{uid}: WAV SHA-256 differs from the original reduced inventory")
                info = sf.info(path)
                count = info.frames
                if info.samplerate != 16000 or info.channels != 1 or count <= 0:
                    raise ValueError(f"{uid}: expected nonempty 16 kHz mono WAV")
                if count > min(len(fine[uid]) * 320, len(coarse[uid]) * 2560):
                    raise ValueError(f"{uid}: official 20/160 ms labels do not cover duration")
                rows.append(dict(uid=uid, role=role, split=split, audio=relative(path, args.output),
                                 labels_file=relative(coarse_path, args.output)))
                contents[rel] = dict(sha256=digest, samples=count)
        manifests[split] = dict(split=split, count=len(rows), records=rows)
        validated[split] = dict(recordings=len(rows), uid_order_sha256=EXPECTED[split][1],
                               fine_sha256=sha256(fine_path), coarse_sha256=sha256(coarse_path))
    args.output.mkdir(parents=True)
    for split, value in manifests.items():
        save_json(args.output / f"{split}.json", value)
        validated[split]["manifest_sha256"] = sha256(args.output / f"{split}.json")
    save_json(args.output / "audit.json", dict(status="verified", splits=validated,
              eval_uid_count=EXPECTED["eval"][0], eval_uid_order_sha256=EXPECTED["eval"][1], eval_audio_read=False,
              archive_sha256=LABEL_ARCHIVE_SHA256, fixed_ids_file_sha256=sha256(args.ids),
              original_inventory_sha256=sha256(args.original_sha256_inventory),
              audio_inventory_canonical_sha256=hashlib.sha256(
                  "\n".join(f"{name}\t{item['sha256']}" for name, item in contents.items()).encode()).hexdigest(),
              reduced_root=str(reduced), audio=contents))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--archive", required=True, type=Path)
    audio = sub.add_parser("audio")
    audio.add_argument("--reduced", required=True, type=Path)
    audio.add_argument("--labels", required=True, type=Path)
    audio.add_argument("--original-sha256-inventory", required=True, type=Path)
    inventory = sub.add_parser("source-inventory")
    inventory.add_argument("--reduced", required=True, type=Path)
    args = parser.parse_args()
    fixed = check_fixed(args.ids)
    if args.action == "labels":
        extract_training_labels(args.archive, fixed, args.output)
    elif args.action == "source-inventory":
        save_source_inventory(args.reduced, args.output, fixed)
    else:
        audit_audio(args, fixed)


if __name__ == "__main__":
    main()
