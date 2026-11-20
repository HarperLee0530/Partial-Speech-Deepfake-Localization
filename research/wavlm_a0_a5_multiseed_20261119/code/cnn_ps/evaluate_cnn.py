"""Evaluate a fixed CNN-BiGRU checkpoint with dev-only calibration.

--checkpoint loads an original or fine-tuned CNN and regenerates clean dev
scores to calibrate a 5% real-duration FPR budget at native 20 ms. --run retains
the selected checkpoint and saved native20 threshold of a completed historical
fine-tuning run. Neither mode trains or reselects a checkpoint. A supplementary
160 ms threshold maximizes dev frame F1. Both thresholds are written to disk
BEFORE eval data is opened. Fine-label NPZ files use actual20 labels, 1=fake.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from . import metrics160, metrics20
from .backbone import load_backbone
from .finetune import FineData, fine_collate, save_scores
from .io import manifest_hash, read_json, save_json, sha256
from .report_native20 import evaluate_report


def load_selected_run(run):
    run = Path(run).resolve()
    complete = read_json(run / "complete.json")
    if complete.get("status") != "complete":
        raise ValueError("Only completed fine-tuning runs may be evaluated")
    checkpoint = Path(complete["checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else run / checkpoint
    digest = sha256(checkpoint)
    if digest != complete.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint hash differs from completed run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("arm") != "cnn_channel_finetune":
        raise ValueError("Expected a full-CNN channel fine-tuning checkpoint")
    if payload.get("epoch") != complete.get("epoch"):
        raise ValueError("Selected epoch differs between checkpoint and completion marker")
    threshold = float(payload["threshold"])
    if (not np.isfinite(threshold) or threshold != float(complete["threshold"])
            or threshold != float(payload["dev_metrics"]["threshold"])):
        raise ValueError("Native20 dev threshold differs between saved artifacts")
    return run, checkpoint, payload, digest


def saved_dev_records(run, dataset):
    """Validate saved storage-grid scores against the explicit dev annotations."""
    with np.load(run / "dev_scores20.npz", allow_pickle=False) as archive:
        scores = np.asarray(archive["scores20"], dtype=np.float32).reshape(-1)
        uids = np.asarray(archive["uids"]).tolist()
        lengths = np.asarray(archive["lengths"])
        lengths20 = np.asarray(archive["lengths20"])
        durations = np.asarray(archive["durations"], dtype=np.float64)
    if uids != dataset.uids:
        raise ValueError("Saved dev score UIDs/order differ from the supplied dev manifest")
    if any(v.shape != (len(uids),) for v in (lengths, lengths20, durations)):
        raise ValueError("Saved dev score metadata dimensions disagree")
    if (not np.isfinite(scores).all() or not np.isfinite(durations).all()
            or (durations <= 0).any() or (lengths <= 0).any() or (lengths20 <= 0).any()
            or not np.equal(lengths, np.floor(lengths)).all()
            or not np.equal(lengths20, np.floor(lengths20)).all()
            or len(scores) != int(lengths.sum()) * 8):
        raise ValueError("Invalid saved dev score storage grid")
    rows160, rows20, offset = [], [], 0
    for i, uid in enumerate(uids):
        coarse, fine = dataset.coarse[uid], dataset.fine[uid]
        if len(coarse) != lengths[i] or len(fine) != lengths20[i]:
            raise ValueError(f"{uid}: saved score lengths differ from actual annotation grids")
        header = sf.info(dataset.records[i]["audio"])
        if (header.samplerate != 16000 or header.channels != 1
                or abs(header.frames / header.samplerate - durations[i]) > 1e-9):
            raise ValueError(f"{uid}: saved duration differs from the source dev waveform")
        count = len(coarse) * 8
        native = scores[offset:offset + count]
        offset += count
        rows160.append(dict(uid=uid, labels=coarse, labels20=fine,
                            scores=native.reshape(-1, 8).mean(1), duration=float(durations[i])))
        rows20.append(dict(uid=uid, labels20=fine, scores20=native[:len(fine)],
                           duration=float(durations[i])))
    return rows160, rows20


@torch.inference_mode()
def infer_records(model, dataset, device, split):
    """Infer full clean waveforms once; retain storage tails for legacy pooling."""
    model.eval()
    rows160, rows20, stored = [], [], []
    metadata = dict(uids=dataset.uids, lengths=[], lengths20=[], durations=[])
    last = time.monotonic()
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=fine_collate)
    for index, batch in enumerate(loader):
        native = model(batch["features"].to(device), batch["lengths"])[0].float().cpu().numpy()
        n = int(batch["lengths"][0])
        native = native[:n].copy()
        uid = batch["uids"][0]
        coarse, fine, duration = dataset.coarse[uid], dataset.fine[uid], float(batch["durations"][0])
        if (uid != dataset.uids[index] or n != len(coarse) * 8
                or not np.isfinite(native).all() or batch["channels"] != [0]):
            raise ValueError("Inference changed recording order, grid or unaugmented input")
        rows160.append(dict(uid=uid, labels=coarse, labels20=fine,
                            scores=native.reshape(-1, 8).mean(1), duration=duration))
        rows20.append(dict(uid=uid, labels20=fine, scores20=native[:len(fine)], duration=duration))
        stored.append(native)
        metadata["lengths"].append(len(coarse))
        metadata["lengths20"].append(len(fine))
        metadata["durations"].append(duration)
        if time.monotonic() - last >= 20 or index + 1 == len(dataset):
            print(f"{split} {index + 1}/{len(dataset)}", flush=True)
            last = time.monotonic()
    return rows160, rows20, np.concatenate(stored), metadata


@torch.inference_mode()
def evaluate(run=None, dev_manifest=None, dev_labels20=None, eval_manifest=None,
             eval_labels20=None, output=None, device="cpu", threads=4, *, checkpoint=None):
    if (run is None) == (checkpoint is None):
        raise ValueError("Supply exactly one of run or checkpoint")
    if any(value is None for value in (dev_manifest, dev_labels20, eval_manifest, eval_labels20, output)):
        raise ValueError("Explicit dev/eval manifests, actual20 label files and output are required")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty evaluation output directory")
    if threads <= 0:
        raise ValueError("CPU thread count must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select device=cpu explicitly")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from_run = run is not None
    if from_run:
        run, checkpoint, payload, digest = load_selected_run(run)
    else:
        checkpoint = Path(checkpoint).resolve()
        digest = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_backbone(checkpoint, device=device)
    dev = FineData(dev_manifest, dev_labels20, model.config, "dev", payload.get("seed", 666))
    output.mkdir(parents=True, exist_ok=True)
    mode = "completed_run_saved_threshold" if from_run else "fixed_checkpoint_fresh_dev_calibration"
    save_json(output / "protocol.json", dict(mode=mode, checkpoint_sha256=digest,
        checkpoint_reselection=False, training_performed=False, eval_read=False,
        selected_epoch=payload.get("epoch"), metric_protocol_version="ps_native20_event_v2",
        native20_calibration="Preserve completed run threshold" if from_run else "Fresh clean dev; real-duration FPR cap5%",
        checkpoint_selection="Historical supplied checkpoint; no v2 checkpoint selection performed",
        score_archive_layout="scores20 contains 8*lengths stored cells per UID; trim each UID to lengths20 for native20 metrics",
        device=device, precision="float32", tf32=False, source_sha256=sha256(__file__)))
    if from_run:
        settings = payload.get("config", {})
        if ("dev_manifest_sha256" in settings
                and manifest_hash(dev.uids) != settings["dev_manifest_sha256"]):
            raise ValueError("Dev manifest differs from checkpoint selection data")
        if ("dev_labels20_sha256" in settings
                and sha256(dev_labels20) != settings["dev_labels20_sha256"]):
            raise ValueError("Actual20 dev label file differs from checkpoint selection data")
        dev160, dev20 = saved_dev_records(run, dev)
        dev_scores_path = run / "dev_scores20.npz"
        threshold20 = float(payload["threshold"])
        checked_dev20 = metrics20.evaluate(dev20, threshold20)
        for field in ("duration_f1_percent", "event_f1_iou05_percent", "real_fpr_percent",
                      "selection_score_j_percent"):
            if not np.isclose(checked_dev20[field], payload["dev_metrics"][field], rtol=0., atol=1e-7):
                raise ValueError(f"Saved dev scores do not reproduce selected checkpoint metric: {field}")
        threshold_selection = None
    else:
        dev160, dev20, dev_scores, dev_metadata = infer_records(model, dev, device, "dev")
        dev_scores_path = output / "dev_scores20.npz"
        save_scores(dev_scores_path, dev_scores, dev_metadata)
        threshold_selection = metrics20.choose_threshold(dev20, max_real_fpr=.05)
        threshold20 = float(threshold_selection["threshold"])
    dev_report20 = evaluate_report(dev20, threshold20)
    if not from_run and dev_report20["real_duration_fpr_percent"] > 5 + 1e-10:
        raise ValueError("Fresh dev calibration exceeded the declared real-duration FPR budget")
    save_json(output / "dev_metrics20.json", dev_report20)
    y = np.concatenate([row["labels"] for row in dev160])
    s = np.concatenate([row["scores"] for row in dev160])
    if len(np.unique(y)) != 2 and from_run:
        raise ValueError("Dev calibration requires both real and fake reference frames")
    threshold160, dev_f1 = metrics160.choose_threshold(y, s) if len(np.unique(y)) == 2 else (None, None)
    name = f"cnn_channel_finetune_{int(payload.get('seed', 666))}" if from_run else checkpoint.stem
    calibration = dict(model=name, mode=mode, checkpoint_sha256=digest, selected_epoch=payload.get("epoch"),
                       threshold160=threshold160, threshold20=threshold20,
                       dev_frame_f1_percent=dev_f1, dev_recordings=len(dev),
                       dev_manifest_sha256=sha256(dev_manifest), dev_labels20_sha256=sha256(dev_labels20),
                       dev_scores20_sha256=sha256(dev_scores_path),
                       dev_real_duration_fpr_percent=dev_report20["real_duration_fpr_percent"],
                       threshold20_selection=threshold_selection,
                       checkpoint_reselection=False, eval_read=False,
                       coarse_scope="NumPy float32 mean of ALL eight stored logits; threshold maximizes dev frame F1",
                       fine_scope="Actual20 reporting at saved run threshold" if from_run else "Actual20 reporting at fresh dev real-duration FPR cap5% threshold",
                       source_sha256=sha256(__file__))
    # Do not move eval manifest/label loading above this write.
    save_json(output / "frozen_dev_calibration.json", calibration)
    test = FineData(eval_manifest, eval_labels20, model.config, "eval", payload.get("seed", 666))
    if set(dev.uids) & set(test.uids):
        raise ValueError("Dev/eval recording IDs overlap")
    started = time.monotonic()
    rows160, rows20, stored, metadata = infer_records(model, test, device, "eval")
    eval_has_both_classes = len(np.unique(np.concatenate([row["labels"] for row in rows160]))) == 2
    if not eval_has_both_classes and from_run:
        raise ValueError("EER reporting requires both real and fake eval reference frames")
    # Fixed-threshold evaluation only: never call evaluate_dev on eval records.
    result160 = (metrics160.evaluate(rows160, threshold160)
                 if threshold160 is not None and eval_has_both_classes else
                 dict(status="not_reported", recordings=len(test), threshold=threshold160,
                      reason="Supplementary160 reporting requires both classes in dev and eval; native20 metrics remain available."))
    result20 = evaluate_report(rows20, threshold20)
    save_json(output / "metrics160.json", result160)
    save_json(output / "metrics20.json", result20)
    summary160 = {name: {key: value for key, value in result160.items() if key != "per_recording"}}
    summary20 = {name: {key: value for key, value in result20.items() if key != "per_recording"}}
    save_json(output / "summary.json", summary160)
    save_json(output / "summary20.json", summary20)
    save_scores(output / "eval_scores20.npz", stored, metadata)
    if sha256(checkpoint) != digest:
        raise ValueError("Selected checkpoint changed during evaluation")
    save_json(output / "complete.json", dict(status="complete", model=name, mode=mode, selected_epoch=payload.get("epoch"),
              checkpoint_sha256=digest, eval_recordings=len(test), checkpoint_reselection=False,
              calibration_sha256=sha256(output / "frozen_dev_calibration.json"),
              dev_scores20_sha256=sha256(dev_scores_path), training_performed=False,
              threshold_frozen_before_eval=True,
              eval_manifest_sha256=sha256(eval_manifest), eval_labels20_sha256=sha256(eval_labels20),
              eval_scores20_sha256=sha256(output / "eval_scores20.npz"),
              device=device, elapsed_seconds=time.monotonic() - started))
    return dict(summary160=summary160, summary20=summary20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Completed fine-tuning directory; preserve its saved native20 threshold")
    source.add_argument("--checkpoint", help="Original or fine-tuned CNN weights; calibrate fresh native20 dev scores")
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--dev-labels20", required=True, help="Actual20 NPZ, 1=fake")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-labels20", required=True, help="Actual20 NPZ, 1=fake")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
