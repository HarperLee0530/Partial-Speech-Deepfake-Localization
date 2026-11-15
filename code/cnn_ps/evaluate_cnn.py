"""Evaluate a completed full-CNN fine-tuning run with dev-only calibration.

The saved selected checkpoint and native20 dev threshold stay fixed. A separate
160ms reporting threshold is fitted to its saved dev scores, then written to
disk BEFORE the eval manifest/labels are opened. Fine-label NPZ files must
contain actual 20ms annotations encoded as 1=fake, 0=real.
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
def evaluate(run, dev_manifest, dev_labels20, eval_manifest, eval_labels20,
             output, device="cpu", threads=4):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
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
    run, checkpoint, payload, digest = load_selected_run(run)
    model, _ = load_backbone(checkpoint, device=device)
    dev = FineData(dev_manifest, dev_labels20, model.config, "dev", payload.get("seed", 666))
    settings = payload.get("config", {})
    if ("dev_manifest_sha256" in settings
            and manifest_hash(dev.uids) != settings["dev_manifest_sha256"]):
        raise ValueError("Dev manifest differs from checkpoint selection data")
    if ("dev_labels20_sha256" in settings
            and sha256(dev_labels20) != settings["dev_labels20_sha256"]):
        raise ValueError("Actual20 dev label file differs from checkpoint selection data")
    dev160, dev20 = saved_dev_records(run, dev)
    threshold20 = float(payload["threshold"])
    checked_dev20 = metrics20.evaluate(dev20, threshold20)
    for name in ("duration_f1_percent", "event_f1_iou05_percent", "real_fpr_percent",
                 "selection_score_j_percent"):
        if not np.isclose(checked_dev20[name], payload["dev_metrics"][name], rtol=0., atol=1e-7):
            raise ValueError(f"Saved dev scores do not reproduce selected checkpoint metric: {name}")
    y = np.concatenate([row["labels"] for row in dev160])
    s = np.concatenate([row["scores"] for row in dev160])
    if len(np.unique(y)) != 2:
        raise ValueError("Dev calibration requires both real and fake reference frames")
    threshold160, dev_f1 = metrics160.choose_threshold(y, s)
    name = f"cnn_channel_finetune_{int(payload.get('seed', 666))}"
    calibration = dict(model=name, checkpoint_sha256=digest, selected_epoch=payload["epoch"],
                       threshold160=threshold160, threshold20=threshold20,
                       dev_frame_f1_percent=dev_f1, dev_recordings=len(dev),
                       dev_manifest_sha256=sha256(dev_manifest), dev_labels20_sha256=sha256(dev_labels20),
                       dev_scores20_sha256=sha256(run / "dev_scores20.npz"),
                       checkpoint_reselection=False, eval_read=False,
                       coarse_scope="NumPy float32 mean of ALL eight stored logits; threshold maximizes dev frame F1",
                       fine_scope="Actual20 duration/event metrics at the saved checkpoint's dev threshold",
                       source_sha256=sha256(__file__))
    output.mkdir(parents=True, exist_ok=True)
    # Do not move eval manifest/label loading above this write.
    save_json(output / "frozen_dev_calibration.json", calibration)
    test = FineData(eval_manifest, eval_labels20, model.config, "eval", payload.get("seed", 666))
    if set(dev.uids) & set(test.uids):
        raise ValueError("Dev/eval recording IDs overlap")
    model.eval()
    rows160, rows20, stored = [], [], []
    metadata = dict(uids=test.uids, lengths=[], lengths20=[], durations=[])
    started = last = time.monotonic()
    loader = DataLoader(test, batch_size=1, num_workers=0, collate_fn=fine_collate)
    for index, batch in enumerate(loader):
        native = model(batch["features"].to(device), batch["lengths"])[0].float().cpu().numpy()
        n = int(batch["lengths"][0])
        native = native[:n].copy()
        uid = batch["uids"][0]
        coarse, fine, duration = test.coarse[uid], test.fine[uid], float(batch["durations"][0])
        if (uid != test.uids[index] or n != len(coarse) * 8
                or not np.isfinite(native).all() or batch["channels"] != [0]):
            raise ValueError("Eval inference changed recording order, grid or unaugmented input")
        rows160.append(dict(uid=uid, labels=coarse, labels20=fine,
                            scores=native.reshape(-1, 8).mean(1), duration=duration))
        rows20.append(dict(uid=uid, labels20=fine, scores20=native[:len(fine)], duration=duration))
        stored.append(native)
        metadata["lengths"].append(len(coarse))
        metadata["lengths20"].append(len(fine))
        metadata["durations"].append(duration)
        if time.monotonic() - last >= 20 or index + 1 == len(test):
            print(f"eval {index + 1}/{len(test)}", flush=True)
            last = time.monotonic()
    if len(np.unique(np.concatenate([row["labels"] for row in rows160]))) != 2:
        raise ValueError("EER reporting requires both real and fake eval reference frames")
    # Fixed-threshold evaluation only: never call evaluate_dev on eval records.
    result160 = metrics160.evaluate(rows160, threshold160)
    result20 = metrics20.evaluate(rows20, threshold20)
    save_json(output / "metrics160.json", result160)
    save_json(output / "metrics20.json", result20)
    summary160 = {name: {key: value for key, value in result160.items() if key != "per_recording"}}
    summary20 = {name: {key: value for key, value in result20.items() if key != "per_recording"}}
    save_json(output / "summary.json", summary160)
    save_json(output / "summary20.json", summary20)
    save_scores(output / "eval_scores20.npz", np.concatenate(stored), metadata)
    if sha256(checkpoint) != digest:
        raise ValueError("Selected checkpoint changed during evaluation")
    save_json(output / "complete.json", dict(status="complete", model=name, selected_epoch=payload["epoch"],
              checkpoint_sha256=digest, eval_recordings=len(test), checkpoint_reselection=False,
              calibration_sha256=sha256(output / "frozen_dev_calibration.json"),
              eval_manifest_sha256=sha256(eval_manifest), eval_labels20_sha256=sha256(eval_labels20),
              eval_scores20_sha256=sha256(output / "eval_scores20.npz"),
              device=device, elapsed_seconds=time.monotonic() - started))
    return dict(summary160=summary160, summary20=summary20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed fine-tuning directory")
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
