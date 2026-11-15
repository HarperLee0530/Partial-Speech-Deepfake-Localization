"""Model-independent float32 LogMel caches for matched end-to-end CNN arms.

The complete waveform transforms, label-grid padding and LogMel operations are
exactly FineData's. Only frontend configuration is read from --checkpoint;
no CNN is instantiated or evaluated. This cache contains no CNN features or
predictions and may be shared by all arms with the same frontend settings.

The actual20 label NPZ uses 1=fake; manifest labels_file references official
160ms NPY dictionaries with 1=real. Cache creation is separate per split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import CHANNELS, NOISE_SEED
from .backbone import ModelConfig
from .dual_cache import ARRAY_DTYPES, StaticChannelData, record_metadata, write_labels
from .io import manifest_hash, read_json, save_json, sha256


SCHEMA = "cnn_waveform_logmel_cache_v1"
FRONTEND_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels")


def frontend_config(config):
    return {key: getattr(config, key) for key in FRONTEND_KEYS}


@torch.inference_mode()
def build_feature_cache(checkpoint, manifest, labels20, output, split,
                        channels=3, cpu_threads=1, seed=666, model_config=None):
    """Create immutable full-recording LogMel arrays without model inference."""
    manifest, labels20, output = map(Path, (manifest, labels20, output))
    if (checkpoint is None) == (model_config is None):
        raise ValueError('Supply exactly one source: checkpoint or pure model_config JSON')
    if split not in ("train", "dev", "eval"):
        raise ValueError("split must be train, dev or eval")
    if channels not in (1, 3) or (split != "train" and channels != 1):
        raise ValueError("Train supports 1/3 channels; dev/eval require 1")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output must be empty; no implicit cache resume or overwrite")
    source = Path(model_config if model_config is not None else checkpoint)
    source_sha = sha256(source)
    if model_config is not None:
        payload = read_json(source)
        config = ModelConfig(**payload.get('model_config', payload))
    else:
        payload = torch.load(str(source), map_location="cpu", weights_only=False)
        config = ModelConfig(**payload["model_config"])
    del payload
    if config.n_mels != 80:
        raise ValueError("The matched CNN experiment requires 80-bin LogMel features")
    torch.set_num_threads(cpu_threads)
    dataset = StaticChannelData(manifest, labels20, config, split, seed)
    metadata = record_metadata(dataset)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    protocol = dict(schema=SCHEMA, split=split, recordings=len(dataset), channels=channels,
                    channel_names=list(CHANNELS[:channels]), frontend_config=frontend_config(config),
                    frontend_source_checkpoint=None if model_config is not None else str(source.resolve()),
                    frontend_source_checkpoint_sha256=None if model_config is not None else source_sha,
                    frontend_source_config=str(source.resolve()) if model_config is not None else None,
                    frontend_source_config_sha256=source_sha if model_config is not None else None,
                    checkpoint_usage="No trained checkpoint read; pure configuration JSON" if model_config is not None else "Read model_config only; no CNN instantiated or evaluated",
                    records=dataset.records, manifest=str(manifest.resolve()),
                    manifest_sha256=sha256(manifest), uid_order_sha256=manifest_hash(dataset.uids),
                    labels20_file=str(labels20.resolve()), labels20_sha256=sha256(labels20),
                    labels160_source_sha256={name: sha256(name) for name in sorted(dataset.label_files)},
                    source_code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                        ("dual_features.py", "dual_cache.py", "finetune.py", "backbone.py", "augmentation.py")},
                    cpu_threads=cpu_threads, noise_seed=NOISE_SEED,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    label_encoding="cached labels 1=fake; actual20 NPZ already fake1; official160 NPY real1 inverted",
                    storage="Per-record full waveform padded to N160*2560 samples before LogMel; storage20=8*N160",
                    lengths_semantics="lengths and lengths160 are N160; lengths20 are actual fine-label counts; offsets index storage20",
                    feature_semantics="Float32 LogMel, no learned normalization; CNN input LayerNorm remains trainable in each model",
                    weights="Actual intersected duration in seconds; no event weighting",
                    augmentation="Complete waveform transformed before padding; deterministic per UID; no cropping or label changes",
                    eval_access=(split == "eval"), trained=False, cnn_inference=False,
                    threshold_selection=False, checkpoint_selection=False)
    save_json(output / "protocol.json", protocol)
    n20, n160 = metadata["offsets"][-1], sum(metadata["lengths160"])
    specs = {"features": ((channels, n20, 80), "float32"),
             "labels20": ((n20,), "float32"), "weights20": ((n20,), "float32"),
             "labels160": ((n160,), "float32"), "weights160": ((n160,), "float32")}
    arrays = {name: np.lib.format.open_memmap(output/f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in specs.items()}
    write_labels(dataset, metadata, arrays)
    completed = 0
    last_progress = time.monotonic()
    try:
        for channel in range(channels):
            dataset.set_channel(channel)
            for i in range(len(dataset)):
                row = dataset[i]
                a, b = metadata["offsets"][i:i+2]
                if (row["uid"] != metadata["uids"][i] or row["duration"] != metadata["durations"][i]
                        or row["length20"] != metadata["lengths20"][i]
                        or row["features"].shape != (1, b-a, 80)):
                    raise ValueError("Waveform/label alignment changed during feature extraction")
                features = row["features"][0].numpy()
                if features.dtype != np.float32 or not np.isfinite(features).all():
                    raise ValueError(f"{row['uid']}: invalid LogMel features")
                for key, name, left, right in (
                    ("labels", "labels160", a//8, b//8),
                    ("weights", "weights160", a//8, b//8),
                    ("fine_labels", "labels20", a, b),
                    ("fine_weights", "weights20", a, b)):
                    np.testing.assert_array_equal(row[key].numpy(), arrays[name][left:right])
                arrays["features"][channel, a:b] = features
                completed += 1
                if time.monotonic()-last_progress >= 20:
                    progress = dict(split=split, channel=channel, completed_recording_views=completed,
                                    total_recording_views=channels*len(dataset), elapsed_seconds=time.monotonic()-start)
                    save_json(output/"progress.json", progress)
                    print(progress, flush=True)
                    last_progress = time.monotonic()
            arrays["features"].flush()
        if sha256(source) != source_sha:
            raise RuntimeError("Frontend source checkpoint changed during extraction")
        if sha256(manifest) != protocol["manifest_sha256"] or sha256(labels20) != protocol["labels20_sha256"]:
            raise RuntimeError("Manifest or actual20 labels changed during extraction")
        complete = dict(protocol, **metadata, status="complete", completed_recording_views=completed,
                        source_checkpoint_unchanged=(model_config is None),
                        source_configuration_unchanged=(model_config is not None),
                        elapsed_seconds=time.monotonic()-start,
                        arrays={name: dict(file=f"{name}.npy", shape=list(a.shape), dtype=str(a.dtype),
                                           sha256=sha256(output/f"{name}.npy")) for name, a in arrays.items()})
        save_json(output/"complete.json", complete)
        save_json(output/"progress.json", dict(status="complete", split=split, completed_recording_views=completed))
        return complete
    except Exception as exc:
        save_json(output/"failure.json", dict(status="failed", completed_recording_views=completed,
                                              error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        for a in arrays.values():
            a.flush()


class FeatureData(Dataset):
    """FineData-compatible, per-record float32 LogMel memory-map reader.

    Epoch zero always returns original audio. Training epochs rotate the same
    deterministic per-record channel assignment as FineData. New tensors are
    copied per record so training cannot mutate shared disk-backed arrays.
    """

    def __init__(self, folder, split=None, seed=666, config=None, verify_hashes=False):
        self.folder = Path(folder)
        self.metadata = read_json(self.folder/"complete.json")
        m = self.metadata
        if m.get("schema") != SCHEMA or m.get("status") != "complete":
            raise ValueError("Expected a completed waveform LogMel cache")
        if split is not None and m["split"] != split:
            raise ValueError("Feature cache split mismatch")
        if config is not None and frontend_config(config) != m["frontend_config"]:
            raise ValueError("Feature cache frontend differs from model configuration")
        self.split, self.epoch = m["split"], 0
        self.uids, self.records = list(m["uids"]), list(m["records"])
        self.channels = int(m["channels"])
        self.offsets = np.asarray(m["offsets"], dtype=np.int64)
        self.lengths20 = np.asarray(m["lengths20"], dtype=np.int64)
        self.lengths160 = np.asarray(m["lengths160"], dtype=np.int64)
        self.storage_lengths20 = np.asarray(m["storage_lengths20"], dtype=np.int64)
        self.durations = np.asarray(m["durations"], dtype=np.float64)
        n = len(self.uids)
        if (n == 0 or n != m["recordings"] or len(set(self.uids)) != n
                or [r["uid"] for r in self.records] != self.uids
                or any(r["split"] != self.split for r in self.records)
                or self.channels not in (1, 3) or (self.split != "train" and self.channels != 1)
                or any(len(v) != n for v in (self.lengths20, self.lengths160, self.storage_lengths20, self.durations))
                or len(self.offsets) != n+1 or self.offsets[0] != 0
                or not np.array_equal(np.diff(self.offsets), self.storage_lengths20)
                or not np.array_equal(self.storage_lengths20, self.lengths160*8)
                or (self.lengths20 <= 0).any() or (self.lengths20 > self.storage_lengths20).any()
                or not np.isfinite(self.durations).all() or (self.durations <= 0).any()):
            raise ValueError("Invalid per-record LogMel cache alignment")
        total20, total160 = int(self.offsets[-1]), int(self.lengths160.sum())
        shapes = {"features": (self.channels, total20, 80), "labels20": (total20,),
                  "weights20": (total20,), "labels160": (total160,), "weights160": (total160,)}
        self.arrays = {}
        for name, shape in shapes.items():
            path = self.folder/f"{name}.npy"
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            info = m["arrays"][name]
            if (a.shape != shape or a.dtype != np.float32
                    or info["shape"] != list(shape) or info["dtype"] != "float32"):
                raise ValueError(f"Invalid feature cache array {name}")
            if verify_hashes and sha256(path) != info["sha256"]:
                raise ValueError(f"Changed feature cache array {name}")
            self.arrays[name] = a
        assignment = np.arange(n) % 3
        np.random.default_rng(seed+20261113).shuffle(assignment)
        self.assignment = assignment
        self.fine, self.coarse = {}, {}
        for i, uid in enumerate(self.uids):
            a, b = map(int, self.offsets[i:i+2])
            self.fine[uid] = self.arrays["labels20"][a:a+int(self.lengths20[i])]
            self.coarse[uid] = self.arrays["labels160"][a//8:b//8]

    def __len__(self):
        return len(self.uids)

    def get_record(self, index, channel=0):
        if not 0 <= index < len(self) or not 0 <= channel < self.channels:
            raise IndexError("Feature recording index or channel outside cache")
        a, b = map(int, self.offsets[index:index+2])
        # Explicit copies avoid writable-tensor aliases into read-only shared memmaps.
        tensor = lambda array: torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        return dict(uid=self.uids[index], features=tensor(self.arrays["features"][channel, a:b]).unsqueeze(0),
                    labels=tensor(self.arrays["labels160"][a//8:b//8]), duration=float(self.durations[index]),
                    fine_labels=tensor(self.arrays["labels20"][a:b]),
                    fine_weights=tensor(self.arrays["weights20"][a:b]),
                    weights=tensor(self.arrays["weights160"][a//8:b//8]),
                    length20=int(self.lengths20[index]), channel=int(channel), frontend_seconds=0.)

    def __getitem__(self, index):
        channel = int((self.assignment[index]+self.epoch-1) % 3) if self.split == "train" and self.epoch and self.channels == 3 else 0
        return self.get_record(index, channel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--checkpoint", type=Path, help="Read frontend model_config only")
    sources.add_argument("--model-config", type=Path, help="Pure configuration JSON; no trained checkpoint required")
    parser.add_argument("--manifest", "--train-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--labels20", type=Path, required=True, help="UID-keyed actual20 NPZ, 1=fake")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "eval"), required=True)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    result = build_feature_cache(**vars(args))
    print({key: result[key] for key in ("status", "split", "recordings", "channels", "frontend_config")}, flush=True)


if __name__ == "__main__":
    main()
