"""Prepare auditable TRAIN/DEV caches using only original reduced WAVs."""
import argparse
from pathlib import Path
import os
import shutil
import time

import numpy as np
import torch

from cnn_ps.dual_features import build_feature_cache
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.train_wavlm_multilayer import model_config, setup
from cnn_ps.wavlm_multilayer import build_cache

from scripts.monitor_training import process_start
from trainer import HISTORICAL_CHECKPOINT_SHA256, cache_datasets


def inputs(args, split, audit):
    manifest = args.manifests / f"{split}.json"
    labels = args.labels / f"{split}.npz"
    if (sha256(manifest) != audit["splits"][split]["manifest_sha256"]
            or sha256(labels) != audit["splits"][split]["fine_sha256"]):
        raise ValueError(f"{split}: audited manifests or labels were changed")
    return manifest, labels


def match_cache_to_audit(args, audit):
    for data in cache_datasets(args):
        split = data.split
        manifest, labels = inputs(args, split, audit)
        if not np.isclose(data.durations.sum(), sum(
                audit["audio"][f"{row['role']}/audio/{row['uid']}.wav"]["samples"]
                for row in read_json(manifest)["records"]) / 16000, rtol=0, atol=1e-8):
            raise ValueError(f"{split}: audited waveform duration differs from cache")
        with np.load(labels, allow_pickle=False) as y:
            for i, row in enumerate(read_json(manifest)["records"]):
                uid = row["uid"]
                expected = audit["audio"][f"{row['role']}/audio/{uid}.wav"]["samples"] / 16000
                if expected != data.durations[i] or not np.array_equal(data.fine[uid], y[uid]):
                    raise ValueError(f"{uid}: mismatched cached duration or 20 ms labels")
        digests = set(data.metadata["labels160_source_sha256"].values())
        if digests != {audit["splits"][split]["coarse_sha256"]}:
            raise ValueError(f"{split}: cache coarse-label provenance mismatch")
        if args.reuse_historical:
            hist = read_json(args.historical_protocol)
            if sha256(data.folder / "complete.json") != hist[f"{split}_cache_metadata_sha256"]:
                raise ValueError(f"{split}: historical reference-cache digest mismatch")
            ssl = data.ssl_metadata
            if split == "train":
                if ssl["features_sha256"] != hist["external_pretraining"]["features_sha256"]:
                    raise ValueError("Historical train WavLM feature digest mismatch")
            else:
                # Match the digest of historical DEV WavLM metadata, which includes the feature digest.
                if sha256(args.cache / "layers" / split / "complete.json") != args.historical_dev_layers_sha256:
                    raise ValueError("Historical DEV WavLM metadata digest mismatch")


def prepare(args, progress):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One explicitly selected CUDA GPU is required")
    if sha256(args.model / "pytorch_model.bin") != HISTORICAL_CHECKPOINT_SHA256:
        raise ValueError("Wrong pinned WavLM Base Plus checkpoint")
    audit_path = args.manifests / "audit.json"
    audit = read_json(audit_path)
    if audit["status"] != "verified" or audit["eval_audio_read"]:
        raise ValueError("A verified TRAIN/DEV reduced data audit is required")
    args.cache.mkdir(parents=True, exist_ok=True)
    ready = args.cache / "verified.json"
    if ready.exists():
        raise FileExistsError(ready)
    if not args.reuse_historical:
        expected_paths = [args.cache / kind / split for kind in ("reference", "layers") for split in ("train", "dev")]
        if any(path.exists() for path in expected_paths):
            raise FileExistsError("Cache requires fresh paths. Preserve partial caches and choose a fresh directory.")
        train_total, dev_total = 1110368, 442776
        secondary = args.secondary_storage
        train_bytes = 3 * train_total * 3 * 768 * 4
        aux_bytes = ((3 * train_total + dev_total) * 80 + dev_total * 3 * 768) * 4
        if secondary is not None:
            secondary = secondary.resolve()
            if not secondary.is_relative_to(Path("/root/research/nndl_partial_speech/cache")):
                raise ValueError("Second-volume storage must belong to NNDL")
            secondary.mkdir(parents=True, exist_ok=False)
        if secondary is None or args.cache.stat().st_dev == secondary.stat().st_dev:
            train_bytes += aux_bytes
        elif shutil.disk_usage(secondary).free < aux_bytes + 2**28:
            raise RuntimeError("Insufficient free space for reference and DEV layer caches")
        if shutil.disk_usage(args.cache).free < train_bytes + 2 * 2**30:
            raise RuntimeError(f"Cache volume needs {train_bytes / 2**30 + 2:.1f} GiB free including run headroom")
        model_config_file = args.cache / "reference_model_config.json"
        save_json(model_config_file, model_config().to_dict())
        setup(666, 2)
        for split, channels in (("train", 3), ("dev", 1)):
            progress(detail=f"Validating source WAVs for {split}")
            manifest, labels = inputs(args, split, audit)
            for row in read_json(manifest)["records"]:
                target = (manifest.parent / row["audio"]).resolve()
                reduced = Path(audit["reduced_root"]).resolve()
                key = target.relative_to(reduced).as_posix()
                if sha256(target) != audit["audio"][key]["sha256"]:
                    raise ValueError(f"{row['uid']}: audited reduced WAV was changed")
            logical_reference = args.cache / "reference" / split
            reference = (secondary or args.cache) / "reference" / split
            progress(detail=f"Building original reference cache for {split}")
            build_feature_cache(None, manifest, labels, reference, split, channels=channels,
                                model_config=model_config_file)
            if logical_reference != reference:
                logical_reference.parent.mkdir(parents=True, exist_ok=True)
                logical_reference.symlink_to(reference, target_is_directory=True)
            logical = args.cache / "layers" / split
            physical = secondary / "layers" / split if secondary is not None and split == "dev" else logical
            physical.parent.mkdir(parents=True, exist_ok=True)
            progress(detail=f"Extracting frozen WavLM 6/9/12 on CUDA for {split}")
            build_cache(reference, args.model, physical, device="cuda")
            if logical != physical:
                logical.symlink_to(physical, target_is_directory=True)
    progress(detail="Verifying labels, multiview data and cache SHA-256")
    match_cache_to_audit(args, audit)
    cache_heads = {f"{kind}/{split}": sha256(args.cache / kind / split / "complete.json")
                   for kind in ("reference", "layers") for split in ("train", "dev")}
    save_json(ready, dict(status="verified", historical=args.reuse_historical, audio_audit_sha256=sha256(audit_path),
                         gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), cache_metadata=cache_heads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--secondary-storage", type=Path, help="NNDL-owned reference and DEV storage on a second volume")
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--reuse-historical", action="store_true")
    parser.add_argument("--historical-protocol", type=Path)
    parser.add_argument("--historical-dev-layers-sha256")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.reuse_historical and not (args.historical_protocol and args.historical_dev_layers_sha256):
        parser.error("Historical reuse requires independent pinned reference and DEV feature provenance")

    started = time.time()

    def progress(phase="Preparing", error=None, detail=None):
        if args.log:
            save_json(args.log.parent / "active.json", dict(arm="A0/C1/C2/C1+C2", phase=phase, mode="feature cache",
                      log_path=str(args.log), checkpoint_dir=str(args.cache), detail=detail, error=error,
                      started_at=started, updated_at=time.time(), gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      pid=os.getpid(), pid_start=process_start(os.getpid()), seed=args.seed or 666,
                      finished_at=time.time() if phase in ("Failed", "Completed") else None))
        print(detail or error or phase, flush=True)

    progress(detail="Checking source data and CUDA")
    try:
        prepare(args, progress)
    except BaseException as error:
        progress("Failed", error=f"{type(error).__name__}: {error}")
        raise
    progress("Completed", detail="Verified TRAIN/DEV caches are ready")


if __name__ == "__main__":
    main()
