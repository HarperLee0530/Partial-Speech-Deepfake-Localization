"""Frozen CNN caches for per-record dual-resolution modules.

Both backbones use the same extraction path. ``--labels20`` is the trusted,
UID-keyed NPZ of actual official 20 ms labels, already converted to 1=fake;
the manifest's official 160 ms NPY dictionaries use 1=real and FineData flips
them. Coarse labels are never repeated to manufacture fine targets.

Storage is padded per recording to 8*N160, exactly as the frozen CNN input.
Durations and actual fine-label lengths identify valid cells. Cache creation
does not train, select an epoch, calibrate a threshold, or compute metrics.
Eval is a separate explicit invocation and only allows the original channel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .augmentation import CHANNELS, NOISE_SEED
from .backbone import load_backbone
from .finetune import FineData, fine_collate
from .io import manifest_hash, read_json, save_json, sha256


SCHEMA = "cnn_dual_record_cache_v1"
ARRAY_DTYPES = {
    "native": "float32", "context": "float16", "labels20": "float32",
    "weights20": "float32", "labels160": "float32", "weights160": "float32",
}


class StaticChannelData(FineData):
    """Reuse FineData's exact waveform/frontend/label arithmetic without RAM caching."""

    def set_channel(self, channel):
        if channel not in (0, 1, 2) or (self.split != "train" and channel != 0):
            raise ValueError("Only train may use augmented channels")
        self.channel = int(channel)
        # FineData uses (assignment + epoch - 1) % 3 for training recordings.
        self.assignment.fill(channel)
        self.epoch = 1 if self.split == "train" else 0
        self.dev_items.clear()

    def __getitem__(self, index):
        item = super().__getitem__(index)
        self.dev_items.pop(index, None)
        if item["channel"] != self.channel:
            raise RuntimeError("FineData did not produce the requested fixed channel")
        return item


def duration_weights(count, step, duration):
    ticks = np.arange(count, dtype=np.float64)
    return np.maximum(0., np.minimum((ticks + 1) * step, duration) - ticks * step).astype(np.float32)


def record_metadata(dataset):
    lengths, lengths20, durations = [], [], []
    for row in dataset.records:
        uid = row["uid"]
        info = sf.info(row["audio"])
        if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"{uid}: expected nonempty 16 kHz mono audio")
        duration = info.frames / info.samplerate
        n160, n20 = len(dataset.coarse[uid]), len(dataset.fine[uid])
        if n160 * .16 < duration - 1e-9 or n20 * .02 < duration - 1e-9:
            raise ValueError(f"{uid}: labels do not cover the complete waveform")
        lengths.append(n160)
        lengths20.append(n20)
        durations.append(duration)
    storage = np.asarray(lengths, dtype=np.int64) * 8
    offsets = np.r_[0, np.cumsum(storage, dtype=np.int64)]
    return dict(uids=list(dataset.uids), lengths=lengths, lengths160=lengths,
                lengths20=lengths20, storage_lengths20=storage.tolist(),
                offsets=offsets.tolist(), durations=durations)


def allocate_arrays(folder, metadata, channels):
    n20 = int(metadata["offsets"][-1])
    n160 = sum(metadata["lengths160"])
    shapes = {"native": (channels, n20), "context": (channels, n20, 256),
              "labels20": (n20,), "weights20": (n20,),
              "labels160": (n160,), "weights160": (n160,)}
    return {name: np.lib.format.open_memmap(
        folder / f"{name}.npy", mode="w+", dtype=ARRAY_DTYPES[name], shape=shape)
        for name, shape in shapes.items()}


def write_labels(dataset, metadata, arrays):
    for i, uid in enumerate(metadata["uids"]):
        a, b = metadata["offsets"][i:i+2]
        n20, n160 = metadata["lengths20"][i], metadata["lengths160"][i]
        duration = metadata["durations"][i]
        arrays["labels20"][a:b] = 0
        arrays["labels20"][a:a+n20] = dataset.fine[uid]
        arrays["weights20"][a:b] = 0
        arrays["weights20"][a:a+n20] = duration_weights(n20, .02, duration)
        arrays["labels160"][a//8:b//8] = dataset.coarse[uid]
        arrays["weights160"][a//8:b//8] = duration_weights(n160, .16, duration)
    for name in ("labels20", "weights20", "labels160", "weights160"):
        arrays[name].flush()


@torch.inference_mode()
def build_cache(checkpoint, manifest, labels20, output, split, channels=1,
                device="cuda", batch_size=8, cpu_threads=4, seed=666,
                expected_sha256=None):
    """Extract one split and checkpoint; refuse reuse of incomplete outputs."""
    checkpoint, manifest, labels20, output = map(Path, (checkpoint, manifest, labels20, output))
    if split not in ("train", "dev", "eval"):
        raise ValueError("split must be train, dev or eval")
    if channels not in (1, 3) or (split != "train" and channels != 1):
        raise ValueError("channels must be 1 or 3; dev/eval require 1")
    if min(batch_size, cpu_threads) <= 0 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid runtime configuration")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output must be empty; no implicit cache resume or overwrite")
    before_sha = sha256(checkpoint)
    if expected_sha256 is not None and before_sha != expected_sha256:
        raise ValueError("Checkpoint differs from the requested backbone SHA256")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, _ = load_backbone(str(checkpoint), device=device)
    if 2 * model.config.hidden_size != 256:
        raise ValueError("Dual-resolution cache requires 256-dimensional BiGRU context")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Cache backbone must be frozen and in eval mode")
    dataset = StaticChannelData(manifest, labels20, model.config, split, seed)
    metadata = record_metadata(dataset)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source_files = {name: sha256(Path(__file__).with_name(name)) for name in
                    ("dual_cache.py", "backbone.py", "finetune.py", "baseline.py", "augmentation.py")}
    label_sources = {name: sha256(name) for name in sorted(dataset.label_files)}
    protocol = dict(schema=SCHEMA, split=split, recordings=len(dataset), channels=channels,
                    channel_names=list(CHANNELS[:channels]), backbone_sha256=before_sha,
                    checkpoint=str(checkpoint.resolve()), manifest=str(manifest.resolve()),
                    manifest_sha256=sha256(manifest), uid_order_sha256=manifest_hash(dataset.uids),
                    labels20_file=str(labels20.resolve()), labels20_sha256=sha256(labels20),
                    labels160_source_sha256=label_sources, source_code_sha256=source_files,
                    model_config=model.config.to_dict(), device=device, batch_size=batch_size,
                    cpu_threads=cpu_threads, seed=seed, noise_seed=NOISE_SEED,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    label_encoding="all cached labels: 1=fake; actual20 NPZ already fake1; official160 NPY real1 inverted",
                    storage="Per-record waveform zero-padded to N160*2560 samples; storage20=8*N160",
                    lengths_semantics="lengths and lengths160 are N160; lengths20 are actual fine-label counts; offsets index storage20",
                    feature_semantics="raw frozen BiGRU context before classifier; no external standardization fitted or applied",
                    labels20_padding="Zero labels and zero duration weights after actual fine-label length",
                    weights="Actual intersected audio duration in seconds per cell; no event weighting",
                    extraction="Full waveform transformed before zero padding; each channel traverses records separately in manifest order",
                    eval_access=(split == "eval"), trained=False,
                    threshold_selection=False, checkpoint_selection=False)
    save_json(output / "protocol.json", protocol)
    arrays = allocate_arrays(output, metadata, channels)
    write_labels(dataset, metadata, arrays)
    extracted = 0
    last_progress = time.monotonic()
    try:
        for channel in range(channels):
            dataset.set_channel(channel)
            cursor = 0
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=fine_collate, num_workers=0)
            for batch in loader:
                logits, context = model(batch["features"].to(device), batch["lengths"], return_features=True)
                logits = logits.float().cpu().numpy()
                context = context.float().cpu().numpy()
                for j, uid in enumerate(batch["uids"]):
                    i = cursor + j
                    a, b = metadata["offsets"][i:i+2]
                    n = b-a
                    if (uid != metadata["uids"][i] or int(batch["lengths"][j]) != n
                            or int(batch["label_lengths"][j]) != metadata["lengths160"][i]
                            or batch["lengths20"][j] != metadata["lengths20"][i]
                            or batch["durations"][j] != metadata["durations"][i]):
                        raise ValueError("Waveform/label/manifest alignment changed during extraction")
                    h, z = context[j, :n], logits[j, :n]
                    if h.shape != (n, 256) or not np.isfinite(z).all() or not np.isfinite(h).all():
                        raise ValueError(f"{uid}: invalid backbone outputs")
                    h16 = h.astype(np.float16)
                    if not np.isfinite(h16).all():
                        raise ValueError(f"{uid}: float16 context overflow")
                    np.testing.assert_array_equal(batch["fine_labels"][j, :n].numpy(), arrays["labels20"][a:b])
                    np.testing.assert_allclose(batch["fine_weights"][j, :n].numpy(), arrays["weights20"][a:b], rtol=0, atol=1e-8)
                    np.testing.assert_array_equal(batch["labels"][j, :n//8].numpy(), arrays["labels160"][a//8:b//8])
                    np.testing.assert_allclose(batch["weights"][j, :n//8].numpy(), arrays["weights160"][a//8:b//8], rtol=0, atol=1e-8)
                    arrays["native"][channel, a:b] = z
                    arrays["context"][channel, a:b] = h16
                cursor += len(batch["uids"])
                extracted += len(batch["uids"])
                if time.monotonic() - last_progress >= 20:
                    progress = dict(split=split, channel=channel, completed_recording_views=extracted,
                                    total_recording_views=channels*len(dataset), elapsed_seconds=time.monotonic()-start)
                    save_json(output / "progress.json", progress)
                    print(progress, flush=True)
                    last_progress = time.monotonic()
            if cursor != len(dataset):
                raise RuntimeError("Incomplete channel extraction")
            arrays["native"].flush()
            arrays["context"].flush()
        after_sha = sha256(checkpoint)
        if before_sha != after_sha:
            raise RuntimeError("Backbone checkpoint changed during cache extraction")
        if sha256(manifest) != protocol["manifest_sha256"] or sha256(labels20) != protocol["labels20_sha256"]:
            raise RuntimeError("Input manifest or actual20 label archive changed during extraction")
        complete = dict(protocol, **metadata, status="complete", completed_recording_views=extracted,
                        backbone_sha256_after=after_sha, checkpoint_unchanged=True,
                        elapsed_seconds=time.monotonic()-start,
                        arrays={name: dict(file=f"{name}.npy", shape=list(a.shape), dtype=str(a.dtype),
                                           sha256=sha256(output/f"{name}.npy")) for name, a in arrays.items()})
        save_json(output / "complete.json", complete)
        save_json(output / "progress.json", dict(status="complete", split=split, completed_recording_views=extracted))
        return complete
    except Exception as exc:
        save_json(output / "failure.json", dict(status="failed", completed_recording_views=extracted,
                                                error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        for a in arrays.values():
            a.flush()


class DualCache:
    """Read-only memmaps; one record cannot cross another recording's boundary.

    get_record returns NumPy views. Copy before mutating, or convert with
    torch.tensor. h/z/y20/w20 use storage20; y160/w160 use N160. Mask attention
    and losses with valid20, and crop predictions to length20 for evaluation.
    """

    def __init__(self, folder, expected_checkpoint_sha256=None, expected_split=None, verify_hashes=False):
        self.folder = Path(folder)
        self.metadata = read_json(self.folder / "complete.json")
        m = self.metadata
        if m.get("schema") != SCHEMA or m.get("status") != "complete":
            raise ValueError("Expected a completed dual-resolution CNN cache")
        if expected_checkpoint_sha256 is not None and m["backbone_sha256"] != expected_checkpoint_sha256:
            raise ValueError("Cache belongs to a different CNN backbone")
        if expected_split is not None and m["split"] != expected_split:
            raise ValueError("Cache split mismatch")
        if not m.get("checkpoint_unchanged") or m["backbone_sha256"] != m["backbone_sha256_after"]:
            raise ValueError("Cache checkpoint provenance is incomplete")
        self.uids = list(m["uids"])
        self.channels = int(m["channels"])
        self.lengths20 = np.asarray(m["lengths20"], dtype=np.int64)
        self.lengths160 = np.asarray(m["lengths160"], dtype=np.int64)
        self.storage_lengths20 = np.asarray(m["storage_lengths20"], dtype=np.int64)
        self.offsets = np.asarray(m["offsets"], dtype=np.int64)
        self.durations = np.asarray(m["durations"], dtype=np.float64)
        n = len(self.uids)
        if (n != m["recordings"] or len(set(self.uids)) != n or self.channels not in (1, 3)
                or (m["split"] != "train" and self.channels != 1)
                or any(len(v) != n for v in (self.lengths20, self.lengths160, self.storage_lengths20, self.durations))
                or len(self.offsets) != n+1 or self.offsets[0] != 0
                or not np.array_equal(np.diff(self.offsets), self.storage_lengths20)
                or not np.array_equal(self.storage_lengths20, self.lengths160*8)
                or not np.array_equal(np.asarray(m["lengths"]), self.lengths160)
                or (self.lengths20 <= 0).any() or (self.lengths20 > self.storage_lengths20).any()
                or not np.isfinite(self.durations).all() or (self.durations <= 0).any()):
            raise ValueError("Invalid recording cache alignment metadata")
        n20, n160 = int(self.offsets[-1]), int(self.lengths160.sum())
        expected = {"native": (self.channels, n20), "context": (self.channels, n20, 256),
                    "labels20": (n20,), "weights20": (n20,),
                    "labels160": (n160,), "weights160": (n160,)}
        self.arrays = {}
        for name, shape in expected.items():
            path = self.folder / f"{name}.npy"
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            info = m["arrays"][name]
            if (a.shape != shape or a.dtype != np.dtype(ARRAY_DTYPES[name])
                    or info["shape"] != list(shape) or info["dtype"] != str(a.dtype)):
                raise ValueError(f"Invalid {name} cache array")
            if verify_hashes and sha256(path) != info["sha256"]:
                raise ValueError(f"Changed {name} cache array")
            self.arrays[name] = a

    def __len__(self):
        return len(self.uids)

    def get_record(self, index, channel=0):
        if not 0 <= index < len(self) or not 0 <= channel < self.channels:
            raise IndexError("Recording index or channel is outside the cache")
        a, b = map(int, self.offsets[index:index+2])
        w20 = self.arrays["weights20"][a:b]
        return dict(uid=self.uids[index], duration=float(self.durations[index]), channel=int(channel),
                    length20=int(self.lengths20[index]), length160=int(self.lengths160[index]),
                    storage_length20=b-a, h=self.arrays["context"][channel, a:b],
                    z=self.arrays["native"][channel, a:b], y20=self.arrays["labels20"][a:b],
                    w20=w20, valid20=w20 > 0, y160=self.arrays["labels160"][a//8:b//8],
                    w160=self.arrays["weights160"][a//8:b//8])

    def __getitem__(self, index):
        return self.get_record(index, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--manifest", "--train-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--labels20", type=Path, required=True,
                        help="Actual official 20ms annotations as UID-keyed NPZ, already 1=fake")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "eval"), required=True)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    result = build_cache(**vars(args))
    print({key: result[key] for key in ("status", "split", "recordings", "channels", "backbone_sha256")}, flush=True)


if __name__ == "__main__":
    main()
