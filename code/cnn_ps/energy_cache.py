"""Aligned, pitch-free energy5 caches paired with immutable waveform LogMel.

Only train-clean actual cells fit normalization. Development/evaluation never
fit statistics. No learned CNN is instantiated, and no checkpoint is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .augmentation import transform, waveform_sha256, NOISE_SEED
from .backbone import LogMel, ModelConfig
from .dual_features import FeatureData, frontend_config
from .energy_features import extract, description, FEATURE_NAMES
from .finetune import fine_collate
from .io import load_manifest, read_json, save_json, sha256


SCHEMA = "cnn_energy5_cache_v1"
NORMALIZATION_SCHEMA = "cnn_energy5_train_clean_normalization_v1"
ALIGNMENT_KEYS = ("split", "recordings", "channels", "uids", "offsets", "lengths20",
                  "lengths160", "storage_lengths20", "durations", "uid_order_sha256")


@torch.inference_mode()
def build_energy_cache(frontend_cache, output, threads=1):
    frontend_cache, output = Path(frontend_cache).resolve(), Path(output).resolve()
    if threads <= 0:
        raise ValueError("threads must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Energy output must be empty; no implicit overwrite or resume")
    torch.set_num_threads(threads)
    dataset = FeatureData(frontend_cache, verify_hashes=True)
    source = dataset.metadata
    expected_frontend = frontend_config(ModelConfig())
    if source["frontend_config"] != expected_frontend or source.get("noise_seed") != NOISE_SEED:
        raise ValueError("Energy5 requires the matching 16k/20ms frontend and channel transform")
    manifest = Path(source["manifest"])
    if sha256(manifest) != source["manifest_sha256"]:
        raise ValueError("Frontend manifest changed")
    if load_manifest(manifest, expected_split=dataset.split) != dataset.records:
        raise ValueError("Frontend records differ from their original manifest")
    source_sha = sha256(frontend_cache / "complete.json")
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                  ("energy_features.py", "energy_cache.py", "augmentation.py", "backbone.py")}
    metadata = {key: source[key] for key in ALIGNMENT_KEYS}
    protocol = dict(schema=SCHEMA, **metadata, frontend_cache=str(frontend_cache),
                    frontend_complete_sha256=source_sha, manifest=str(manifest),
                    manifest_sha256=source["manifest_sha256"], feature_definition=description(),
                    source_sha256=code_hashes, noise_seed=NOISE_SEED,
                    augmentation=source["augmentation"],
                    alignment_check="Every recording/channel LogMel regenerated and exactly compared to paired frontend cache",
                    trained=False, cnn_inference=False, pitch_extraction=False,
                    threshold_selection=False, eval_access=dataset.split == "eval",
                    created_utc=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "protocol.json", protocol)
    shape = (dataset.channels, int(dataset.offsets[-1]), 5)
    array = np.lib.format.open_memmap(output / "energy.npy", mode="w+", dtype="float32", shape=shape)
    array[:] = 0
    frontend = LogMel(ModelConfig(**source["frontend_config"])).eval()
    total = np.zeros(5, dtype=np.float64)
    squares = np.zeros(5, dtype=np.float64)
    count = 0
    waveform_hashes, channel_hashes = [], []
    started, last, completed = time.monotonic(), time.monotonic(), 0
    try:
        for i, record in enumerate(dataset.records):
            wave, sr = sf.read(record["audio"], dtype="float32")
            if sr != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError(f"{record['uid']}: expected finite nonempty 16k mono")
            n = (len(wave) + 319) // 320
            a, b = map(int, dataset.offsets[i:i+2])
            if len(wave) / sr != dataset.durations[i] or n != dataset.lengths20[i] or len(wave) > (b-a)*320:
                raise ValueError(f"{record['uid']}: waveform and cached grids disagree")
            weights = dataset.arrays["weights20"][a:b]
            expected_weights = np.zeros(b-a, dtype=np.float32)
            expected_weights[:n] = np.maximum(0., np.minimum((np.arange(n)+1)*.02, len(wave)/sr)-np.arange(n)*.02)
            if not np.array_equal(weights, expected_weights) or not (weights[:n] > 0).all():
                raise ValueError(f"{record['uid']}: actual-duration mask changed")
            waveform_hashes.append(waveform_sha256(wave))
            view_hashes = []
            for channel in range(dataset.channels):
                changed = transform(wave, record["uid"], channel)
                view_hashes.append(waveform_sha256(changed))
                padded = F.pad(torch.from_numpy(changed), (0, (b-a)*320-len(wave)))
                regenerated = frontend(padded)[0].numpy()
                if not np.array_equal(regenerated, dataset.arrays["features"][channel, a:b]):
                    raise ValueError(f"{record['uid']} channel{channel}: waveform/frontend cache mismatch")
                features = extract(changed, sr)
                array[channel, a:a+n] = features
                if dataset.split == "train" and channel == 0:
                    values = features.astype(np.float64)
                    total += values.sum(0)
                    squares += np.square(values).sum(0)
                    count += n
                completed += 1
            channel_hashes.append(view_hashes)
            if time.monotonic() - last >= 20:
                progress = dict(split=dataset.split, recordings=i+1, recording_views=completed,
                                total_recordings=len(dataset), elapsed_seconds=time.monotonic()-started)
                save_json(output / "progress.json", progress)
                print(progress, flush=True)
                last = time.monotonic()
        array.flush()
        if sha256(frontend_cache / "complete.json") != source_sha or sha256(manifest) != source["manifest_sha256"]:
            raise ValueError("Input metadata changed during energy extraction")
        if any(sha256(Path(__file__).with_name(name)) != value for name, value in code_hashes.items()):
            raise ValueError("Energy/frontend source changed during extraction")
        energy_hash = sha256(output / "energy.npy")
        normalization = None
        if dataset.split == "train":
            mean = total / count
            std = np.sqrt(np.maximum(squares / count - mean * mean, 0.)).clip(1e-4)
            stats = dict(schema=NORMALIZATION_SCHEMA, mean=mean.tolist(), std=std.tolist(),
                         valid_train_frames=count, feature_names=list(FEATURE_NAMES),
                         source="train clean actual frames; equal frame weight; includes valid zero feature vectors",
                         frontend_complete_sha256=source_sha, energy_sha256=energy_hash,
                         fit_split="train", fit_channel=0, std_floor=1e-4, clip=[-10., 10.],
                         padded_frames_in_statistics=False, normalization_applied_to_cache=False)
            save_json(output / "normalization.json", stats)
            normalization = dict(file="normalization.json", sha256=sha256(output / "normalization.json"))
        complete = dict(protocol, status="complete", completed_recording_views=completed,
                        logmel_exact_matches=completed, waveform_sha256=waveform_hashes,
                        channel_waveform_sha256=channel_hashes, normalization=normalization,
                        array=dict(file="energy.npy", shape=list(shape), dtype="float32", sha256=energy_hash),
                        elapsed_seconds=time.monotonic()-started)
        save_json(output / "complete.json", complete)
        save_json(output / "progress.json", dict(status="complete", split=dataset.split, recording_views=completed))
        return complete
    except Exception as exc:
        save_json(output / "failure.json", dict(status="failed", recording_views=completed,
                   error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        array.flush()


class EnergyFeatureData(FeatureData):
    """Preserve FeatureData behaviour and append aligned raw energy5 tensors."""
    def __init__(self, frontend_cache, energy_cache, split=None, seed=666, config=None, verify_hashes=False):
        super().__init__(frontend_cache, split=split, seed=seed, config=config, verify_hashes=verify_hashes)
        self.energy_folder = Path(energy_cache)
        self.energy_metadata = read_json(self.energy_folder / "complete.json")
        m = self.energy_metadata
        if (m.get("schema") != SCHEMA or m.get("status") != "complete"
                or m.get("frontend_complete_sha256") != sha256(self.folder / "complete.json")
                or any(m.get(key) != self.metadata[key] for key in ALIGNMENT_KEYS)
                or m.get("feature_definition") != description()):
            raise ValueError("Energy cache is incomplete or differs from the acoustic recording/grid identity")
        self.energy_array = np.load(self.energy_folder / "energy.npy", mmap_mode="r", allow_pickle=False)
        shape = (self.channels, int(self.offsets[-1]), 5)
        info = m["array"]
        if (self.energy_array.shape != shape or self.energy_array.dtype != np.float32
                or info["shape"] != list(shape) or info["dtype"] != "float32"):
            raise ValueError("Invalid energy cache array")
        if verify_hashes and sha256(self.energy_folder / "energy.npy") != info["sha256"]:
            raise ValueError("Energy cache array changed")
        if verify_hashes and m["normalization"] is not None:
            if sha256(self.energy_folder / "normalization.json") != m["normalization"]["sha256"]:
                raise ValueError("Energy normalization changed")

    def get_record(self, index, channel=0):
        row = super().get_record(index, channel)
        a, b = map(int, self.offsets[index:index+2])
        row["energy_features"] = torch.from_numpy(np.array(self.energy_array[channel, a:b], dtype=np.float32, copy=True))
        return row


def energy_collate(rows):
    batch = fine_collate(rows)
    maximum = int(batch["lengths"].max())
    values = []
    for row in rows:
        energy = row["energy_features"]
        if energy.shape != (len(row["fine_weights"]), 5) or not torch.isfinite(energy).all():
            raise ValueError("Expected finite energy_features[storage20,5] aligned with actual duration mask")
        values.append(F.pad(energy, (0, 0, 0, maximum-len(energy))))
    batch["energy_features"] = torch.stack(values)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    result = build_energy_cache(**vars(parser.parse_args(argv)))
    print({key: result[key] for key in ("status", "split", "recordings", "channels", "logmel_exact_matches")}, flush=True)


if __name__ == "__main__":
    main()
