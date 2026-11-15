"""Prepare portable frozen-CNN arrays from PS waveforms and true annotations.

Example (run from code/):
  python -m cnn_ps.prepare --manifest train.json --labels20 train.npz \
      --labels20-encoding fake --checkpoint baseline.pt --output cache/train \
      --device cuda --channels

Official PS .npy annotation dictionaries encode 1=real: pass
--labels20-encoding real for official 20ms labels. The experiment's exported
.npz dictionaries already encode 1=fake. Coarse labels always use official
1=real encoding. Fine supervision must come from actual 20ms annotations, not
repeated 160ms labels. Only trusted official .npy dictionaries should be used,
as that format requires NumPy pickle loading.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .augmentation import CHANNELS, NOISE_SEED, transform
from .backbone import extract_context, load_backbone
from .io import load_manifest, read_json, save_json, sha256


def configure_runtime(threads=2):
    random.seed(NOISE_SEED)
    np.random.seed(NOISE_SEED)
    torch.manual_seed(NOISE_SEED)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _binary(values, uid, grid):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{uid}: expected nonempty binary {grid} annotations")
    return values.astype(np.float32)


class FineLabels:
    """Keep NPZ labels lazy; official NPY stores one trusted dictionary."""
    def __init__(self, path, encoding):
        self.path = Path(path).resolve()
        self.encoding = encoding
        if encoding not in ("fake", "real"):
            raise ValueError("Fine-label encoding must explicitly name its positive class")
        if self.path.suffix.lower() == ".npz":
            self.values = np.load(self.path, allow_pickle=False)
        elif self.path.suffix.lower() == ".npy":
            self.values = np.load(self.path, allow_pickle=True).item()
            if not isinstance(self.values, dict):
                raise ValueError("Official NPY annotations must contain a UID dictionary")
        else:
            raise ValueError("20ms labels must be an NPZ archive or official NPY dictionary")

    def __getitem__(self, uid):
        if uid not in self.values:
            raise KeyError(f"Missing actual 20ms labels: {uid}")
        labels = _binary(self.values[uid], uid, "20ms")
        return labels if self.encoding == "fake" else 1 - labels

    def close(self):
        if hasattr(self.values, "close"):
            self.values.close()


def describe_records(records, fine, split):
    """Read headers/labels only, retaining the declared manifest order."""
    coarse = {}
    described = []
    expected_marker = {"train": "_T_", "dev": "_D_", "eval": "_E_"}[split]
    for record in records:
        uid = record["uid"]
        if uid.startswith("CON_") and expected_marker not in uid:
            raise ValueError(f"{uid}: incompatible official {split} UID")
        if record.get("split", split) != split:
            raise ValueError(f"{uid}: record split differs from manifest split")
        source = str(record["labels_file"])
        if source not in coarse:
            coarse[source] = np.load(source, allow_pickle=True).item()
            if not isinstance(coarse[source], dict):
                raise ValueError("Official 160ms labels must contain a UID dictionary")
        labels = 1 - _binary(coarse[source][uid], uid, "160ms")
        labels20 = fine[uid]
        header = sf.info(str(record["audio"]))
        if header.samplerate != 16000 or header.channels != 1 or header.frames <= 0:
            raise ValueError(f"{uid}: expected nonempty mono 16 kHz audio")
        n = len(labels)
        if header.frames > n * 2560 or (n - 1) * 2560 >= header.frames:
            raise ValueError(f"{uid}: 160ms labels do not match waveform duration")
        if len(labels20) > n * 8 or len(labels20) * 320 < header.frames:
            raise ValueError(f"{uid}: actual 20ms labels do not cover waveform")
        described.append({**record, "labels": labels, "labels20": labels20,
                          "samples": header.frames, "duration": header.frames / 16000,
                          "length": n, "length20": len(labels20)})
    if not described or len({r["uid"] for r in described}) != len(described):
        raise ValueError("Empty or duplicate recording manifest")
    return described


class WaveformViews(Dataset):
    def __init__(self, records, frontend, channels):
        self.records, self.frontend = records, frontend
        # Original extraction has its own batches. Enhanced views use the same
        # per-record (noise, EQ) task ordering as the previous experiment.
        self.tasks = [(i, 0) for i in range(len(records))]
        if channels:
            self.tasks.extend((i, channel) for i in range(len(records)) for channel in (1, 2))

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        record_index, channel = self.tasks[index]
        record = self.records[record_index]
        wave, sample_rate = sf.read(str(record["audio"]), dtype="float32")
        if sample_rate != 16000 or wave.ndim != 1 or len(wave) != record["samples"]:
            raise ValueError(f"{record['uid']}: waveform changed after metadata validation")
        changed = transform(wave, record["uid"], channel)
        waveform = F.pad(torch.from_numpy(changed), (0, record["length"] * 2560 - len(changed)))
        return {"record_index": record_index, "channel": channel,
                "features": self.frontend(waveform)}


def collate_views(items):
    lengths = torch.tensor([r["features"].shape[1] for r in items], dtype=torch.long)
    features = torch.stack([
        F.pad(r["features"], (0, 0, 0, int(lengths.max()) - r["features"].shape[1]))
        for r in items])
    return features, lengths, [(r["record_index"], r["channel"]) for r in items]


def allocate_arrays(output, records, channels):
    n = sum(r["length"] for r in records)
    specs = {"native": ("float32", (n, 8)),
             "context": ("float16", (n, 8, 256)),
             "labels": ("float32", (n,)), "weights": ("float32", (n,)),
             "fine_labels": ("float32", (n, 8)), "fine_weights": ("float32", (n, 8))}
    if channels:
        specs.update(native_channels=("float32", (3, n, 8)),
                     context_channels=("float16", (3, n, 8, 256)))
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    offsets = np.r_[0, np.cumsum([r["length"] for r in records])]
    for i, record in enumerate(records):
        a, b = int(offsets[i]), int(offsets[i + 1])
        n, nf, duration = b - a, record["length20"], record["duration"]
        arrays["labels"][a:b] = record["labels"]
        arrays["weights"][a:b] = np.maximum(
            0., np.minimum((np.arange(n) + 1) * .16, duration) - np.arange(n) * .16).astype(np.float32)
        fine_labels = np.zeros(n * 8, dtype=np.float32)
        fine_weights = np.zeros(n * 8, dtype=np.float32)
        fine_labels[:nf] = record["labels20"]
        fine_weights[:nf] = np.maximum(
            0., np.minimum((np.arange(nf) + 1) * .02, duration) - np.arange(nf) * .02)
        arrays["fine_labels"][a:b] = fine_labels.reshape(n, 8)
        arrays["fine_weights"][a:b] = fine_weights.reshape(n, 8)
    return arrays, offsets


@torch.inference_mode()
def prepare(manifest, labels20, labels20_encoding, checkpoint, output,
            device="cpu", channels=False, batch_size=8, threads=2):
    manifest, checkpoint, output = map(lambda p: Path(p).resolve(), (manifest, checkpoint, output))
    split = read_json(manifest).get("split")
    if split not in ("train", "dev", "eval"):
        raise ValueError("Manifest must declare split=train/dev/eval")
    if channels and split != "train":
        raise ValueError("Training channel views are permitted only for the train split")
    if batch_size <= 0 or threads <= 0:
        raise ValueError("Batch size and CPU threads must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory; existing cache is preserved: {output}")
    fine = FineLabels(labels20, labels20_encoding)
    try:
        records = describe_records(load_manifest(manifest, expected_split=split), fine, split)
    finally:
        fine.close()
    configure_runtime(threads)
    model, frontend = load_backbone(checkpoint, device=device)
    if model.config.hidden_size * 2 != 256:
        raise ValueError("This residual head requires the original 256-dimensional BiGRU context")
    output.mkdir(parents=True, exist_ok=True)
    arrays, offsets = allocate_arrays(output, records, channels)
    dataset = WaveformViews(records, frontend, channels)
    # Do not mix the final original batch with the first enhanced batch: the
    # historical extraction processes those views in separate passes.
    ranges = [(0, len(records))]
    if channels:
        ranges.append((len(records), len(dataset)))
    batches = [list(range(start, min(start + batch_size, stop)))
               for first, stop in ranges for start in range(first, stop, batch_size)]
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0, collate_fn=collate_views)
    started = last = time.monotonic()
    completed = 0
    for features, lengths, task_ids in loader:
        logits, context = extract_context(model, features.to(device), lengths)
        logits, context = logits.cpu().numpy(), context.cpu().numpy()
        for i, (record_index, channel) in enumerate(task_ids):
            a, b = int(offsets[record_index]), int(offsets[record_index + 1])
            n, native_n = b - a, int(lengths[i])
            if native_n != 8 * n:
                raise ValueError("Native features must preserve exactly 8 frames per 160ms block")
            native = logits[i, :native_n].astype(np.float32).reshape(n, 8)
            ctx = context[i, :native_n].astype(np.float16).reshape(n, 8, 256)
            if not np.isfinite(native).all() or not np.isfinite(ctx).all():
                raise ValueError("Frozen backbone returned non-finite output")
            if channel == 0:
                arrays["native"][a:b] = native
                arrays["context"][a:b] = ctx
            if channels:
                arrays["native_channels"][channel, a:b] = native
                arrays["context_channels"][channel, a:b] = ctx
        completed += len(task_ids)
        if time.monotonic() - last >= 20 or completed == len(dataset):
            print(f"{split}: {completed}/{len(dataset)} waveform views", flush=True)
            last = time.monotonic()
    for array in arrays.values():
        array.flush()
    meta = dict(status="complete", split=split, recordings=len(records),
                uids=[r["uid"] for r in records], lengths=[r["length"] for r in records],
                lengths20=[r["length20"] for r in records], durations=[r["duration"] for r in records],
                label_encoding="1=fake; 0=real", actual_20ms_labels=True,
                source_labels20_encoding=f"1={labels20_encoding}",
                manifest_sha256=sha256(manifest), labels20_sha256=sha256(Path(labels20)),
                checkpoint_sha256=sha256(checkpoint), extraction_device=device,
                extraction_batch_size=batch_size, cpu_threads=threads,
                source_labels160_sha256={str(p): sha256(Path(p)) for p in sorted({r["labels_file"] for r in records}, key=str)},
                channel_names=list(CHANNELS) if channels else [CHANNELS[0]],
                recipe=dict(seed=NOISE_SEED, noise="20dB whole-waveform RMS SNR, zero-mean white noise",
                            eq="x + float32(0.15) * (x - previous); previous[0]=x[0]",
                            clipping=False, time_warp=False, labels_changed=False),
                elapsed_seconds=time.monotonic() - started,
                note="Regenerated caches can differ slightly across device, batch size or library version; use the archived caches for exact historical score replay.")
    save_json(output / "complete.json", meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels20", required=True)
    parser.add_argument("--labels20-encoding", choices=("fake", "real"), required=True,
                        help="Which class is 1 in the supplied TRUE 20ms annotations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--channels", action="store_true", help="Also generate noise/EQ views (train only)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
