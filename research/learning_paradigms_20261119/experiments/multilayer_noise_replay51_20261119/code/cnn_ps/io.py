"""Small portable file helpers and explicit PS audio manifests."""
from pathlib import Path
import hashlib
import json
import re


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    """Atomically replace JSON, refusing non-finite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash(uids):
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()


def load_manifest(path, expected_split=None):
    """Read records with uid/audio/labels_file; paths are manifest-relative.

    Official PS .npy label dictionaries use 1=real, 0=fake. Both a top-level
    records list and {"split": ..., "records": [...]} are accepted. An optional
    labels20_file refers to actual fine labels, never repeated coarse labels.
    Record ordering is preserved for reproducible training and score alignment.
    """
    path = Path(path).resolve()
    payload = read_json(path)
    if isinstance(payload, dict):
        declared_split = payload.get("split")
        records = payload.get("records")
    else:
        declared_split, records = None, payload
    if expected_split not in (None, "train", "dev", "eval"):
        raise ValueError("PS split must be train, dev or eval")
    if declared_split and declared_split not in ("train", "dev", "eval"):
        raise ValueError(f"Not a PS manifest split: {declared_split}")
    if expected_split and declared_split and expected_split != declared_split:
        raise ValueError(f"Expected {expected_split}, received {declared_split}")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty records list")
    result, seen = [], set()
    allowed = {"train": "T", "dev": "D", "eval": "E"}
    for original in records:
        if not isinstance(original, dict):
            raise ValueError("Manifest entries must be objects")
        record = dict(original)
        for key in ("uid", "audio", "labels_file"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Manifest entry requires a nonempty {key}")
        uid = record["uid"]
        if uid in seen:
            raise ValueError(f"Duplicate recording ID: {uid}")
        seen.add(uid)
        split = record.get("split", declared_split or expected_split)
        if split not in ("train", "dev", "eval"):
            raise ValueError(f"{uid}: record or manifest must declare its PS split")
        if expected_split and split != expected_split:
            raise ValueError(f"{uid}: wrong official split {split}")
        if declared_split and split != declared_split:
            raise ValueError(f"{uid}: record disagrees with manifest split")
        marker = re.search(r"_(T|D|E)_", uid)
        if marker and marker.group(1) != allowed[split]:
            raise ValueError(f"{uid}: ID and split disagree")
        record["split"] = split
        for key in ("audio", "labels_file", "labels20_file", "path"):
            if record.get(key):
                target = Path(record[key]).expanduser()
                record[key] = str((target if target.is_absolute() else path.parent / target).resolve())
        result.append(record)
    return result
