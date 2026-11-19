"""Train the original CNN-BiGRU on explicit PS train/dev manifests.

The original checkpoint was trained on all 25,380 PS train recordings, selected
on all 24,844 dev recordings. Supplying a smaller manifest is a new experiment,
not a reproduction of that checkpoint. The 160 ms loss and historical dev EER
selection are retained; the main comparison evaluator has a separate protocol.
"""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .backbone import ModelConfig, LogMel, CNNBiGRU, pool_logits, masked_bce
from .io import load_manifest, manifest_hash, save_json, sha256


class PSDataset(Dataset):
    """Official coarse labels, 160 ms aligned crops, unchanged short recordings."""
    resolution = 0.16

    def __init__(self, manifest, model_config=None, crop_seconds=None, split=None):
        self.records = sorted(load_manifest(manifest, expected_split=split),
                              key=lambda record: record["audio"])
        splits = {record["split"] for record in self.records}
        if len(splits) != 1:
            raise ValueError("A baseline dataset must have one official split")
        self.split = splits.pop()
        self.frontend = LogMel(model_config or ModelConfig())
        self.utt_ids = [record["uid"] for record in self.records]
        self.manifest_sha256 = manifest_hash(self.utt_ids)
        self.crop_frames = None
        if crop_seconds is not None:
            frames = round(crop_seconds / self.resolution)
            if (self.split != "train" or frames <= 0 or
                    abs(frames * self.resolution - crop_seconds) > 1e-8):
                raise ValueError("Crop must be a positive multiple of 160 ms, on train only")
            self.crop_frames = frames
        # Official label files are pickled NumPy dictionaries: trusted PS files only.
        self.label_files = {}
        for record in self.records:
            filename = record["labels_file"]
            if filename not in self.label_files:
                self.label_files[filename] = np.load(filename, allow_pickle=True).item()
            labels = np.asarray(self.label_files[filename][record["uid"]])
            if labels.ndim != 1 or not len(labels) or not np.isin(labels, (0, 1)).all():
                raise ValueError(f"{record['uid']}: expected nonempty binary 160 ms labels")
            if not Path(record["audio"]).is_file():
                raise FileNotFoundError(record["audio"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # The old reader loaded float64 then converted to torch.float32.
        wave, sample_rate = sf.read(record["audio"])
        if sample_rate != 16000 or wave.ndim != 1 or not len(wave):
            raise ValueError(f"Expected nonempty 16 kHz mono audio: {record['audio']}")
        duration = len(wave) / sample_rate
        waveform = torch.from_numpy(wave).float()
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Nonfinite audio: {record['uid']}")
        labels = torch.as_tensor(np.asarray(self.label_files[record["labels_file"]][record["uid"]]),
                                 dtype=torch.float32)
        padding = len(labels) * 2560 - len(waveform)
        if padding < 0:
            raise ValueError(f"Coarse labels do not cover the waveform: {record['uid']}")
        waveform = F.pad(waveform, (0, padding))
        start = 0
        if self.crop_frames is not None and len(labels) > self.crop_frames:
            start = random.randint(0, len(labels) - self.crop_frames)
            labels = labels[start:start + self.crop_frames]
            waveform = waveform[start * 2560:(start + self.crop_frames) * 2560]
        tick = time.perf_counter()
        features = self.frontend(waveform)
        if features.shape[1] != len(labels) * 8:
            raise ValueError(f"Feature/label misalignment: {record['uid']}")
        return dict(uid=record["uid"], features=features, labels=1 - labels,
                    duration=duration, crop_start_seconds=start * self.resolution,
                    frontend_seconds=time.perf_counter() - tick)


def collate(batch):
    lengths = torch.tensor([row["features"].shape[1] for row in batch], dtype=torch.long)
    label_lengths = torch.tensor([len(row["labels"]) for row in batch], dtype=torch.long)
    features = torch.stack([F.pad(row["features"], (0, 0, 0, int(lengths.max()) - row["features"].shape[1]))
                            for row in batch])
    labels = torch.stack([F.pad(row["labels"], (0, int(label_lengths.max()) - len(row["labels"])))
                          for row in batch])
    return dict(features=features, labels=labels, lengths=lengths, label_lengths=label_lengths,
                uids=[row["uid"] for row in batch], durations=[row["duration"] for row in batch],
                frontend_seconds=[row.get("frontend_seconds", 0.) for row in batch])


def legacy_dev_metrics(spoof_scores, spoof_targets):
    """Preserve original training's float32 genuine-positive nearest-crossing EER.

    Equal scores are NOT combined before cumulative counts, and the selected
    value is FPR, not the mean of FPR/FNR. This deliberately differs from the
    sklearn ROC average EER used by the final 160 ms comparison table.
    """
    scores = torch.as_tensor(spoof_scores, dtype=torch.float32).detach().cpu().flatten()
    targets = torch.as_tensor(spoof_targets, dtype=torch.float32).detach().cpu().flatten()
    if (not len(scores) or scores.shape != targets.shape or not torch.isfinite(scores).all()
            or not torch.isin(targets, torch.tensor([0., 1.])).all()):
        raise ValueError("Invalid score/target vectors")
    genuine = 1 - targets
    margin = -scores
    order = torch.argsort(margin, descending=True)
    ordered = genuine[order]
    tp = torch.cumsum(ordered, dim=0)
    fp = torch.cumsum(1 - ordered, dim=0)
    single_class = bool(tp[-1] == 0 or fp[-1] == 0)
    if single_class:
        eer, threshold = 100., 0.
    else:
        tpr, fpr = tp / tp[-1], fp / fp[-1]
        crossing = torch.argmin(torch.abs(fpr - (1 - tpr)))
        eer, threshold = float(fpr[crossing] * 100), float(margin[order][crossing])
    # argmax([spoof_score, 0]) chooses spoof on exact ties.
    predicted_genuine = (scores < 0).float()
    true_positive = (predicted_genuine * genuine).sum()
    false_positive = (predicted_genuine * (1 - genuine)).sum()
    false_negative = ((1 - predicted_genuine) * genuine).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = float(100 * 2 * true_positive / denominator) if denominator else 0.
    return dict(eer_percent=eer, eer_margin_threshold=threshold,
                f1_bonafide_percent=f1,
                accuracy_percent=float(100 * (predicted_genuine == genuine).float().mean()),
                frame_count=len(scores), single_class_eer_sentinel=single_class)


@torch.inference_mode()
def validate(model, dataset, device):
    model.eval()
    scores, labels = [], []
    loss_sum = frame_count = 0
    for batch in DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=0):
        native = model(batch["features"].to(device), batch["lengths"])
        coarse, lengths = pool_logits(native, batch["lengths"], 0.16)
        if not torch.equal(lengths, batch["label_lengths"]):
            raise ValueError("Validation target alignment mismatch")
        coarse = coarse.cpu()
        count = int(lengths.sum())
        scores.append(coarse[0, :count])
        labels.append(batch["labels"][0, :count])
        loss_sum += float(masked_bce(coarse, batch["labels"], lengths)) * count
        frame_count += count
    return dict(metrics=legacy_dev_metrics(torch.cat(scores), torch.cat(labels)),
                loss=loss_sum / frame_count)


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = frame_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        native = model(batch["features"].to(device), batch["lengths"])
        coarse, lengths = pool_logits(native, batch["lengths"], 0.16)
        if not torch.equal(lengths, batch["label_lengths"]):
            raise ValueError("Training target alignment mismatch")
        loss = masked_bce(coarse, batch["labels"].to(device), lengths)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        count = int(lengths.sum())
        loss_sum += float(loss.detach()) * count
        frame_count += count
    return loss_sum / frame_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.4)
    parser.add_argument("--crop-seconds", type=float, default=4.)
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.patience, args.cpu_threads) <= 0 or
            args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        parser.error("Invalid training hyperparameters")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; pass --device cpu explicitly")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output is not empty; use a new run directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = ModelConfig(dropout=args.dropout)
    train = PSDataset(args.train_manifest, config, args.crop_seconds, split="train")
    dev = PSDataset(args.dev_manifest, config, split="dev")
    if set(train.utt_ids) & set(dev.utt_ids):
        raise ValueError("Train/dev recording IDs overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(model=config.to_dict(), train_utterances=len(train), dev_utterances=len(dev),
                    train_manifest_sha256=train.manifest_sha256,
                    dev_manifest_sha256=dev.manifest_sha256,
                    train_manifest_file_sha256=sha256(args.train_manifest),
                    dev_manifest_file_sha256=sha256(args.dev_manifest),
                    selection="Original float32 genuine-positive dev EER; lower is better",
                    source_code_sha256=sha256(__file__), eval_read=False)
    save_json(args.output / "config.json", settings)
    model = CNNBiGRU(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    best_eer, stale, history = float("inf"), 0, []
    checkpoint = args.output / "cnn_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, args.device)
        validation = validate(model, dev, args.device)
        if validation["metrics"]["single_class_eer_sentinel"]:
            raise ValueError("Validation must contain both real and fake frames")
        row = dict(epoch=epoch, train_loss=train_loss, **validation)
        history.append(row)
        save_json(args.output / "history.json", history)
        eer = validation["metrics"]["eer_percent"]
        if eer < best_eer:
            best_eer, stale, best_epoch = eer, 0, epoch
            payload = dict(model_config=config.to_dict(),
                           state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                           config=settings, seed=args.seed, epoch=epoch, trained=True,
                           validation_metrics=validation["metrics"],
                           internal_labels={"bonafide": 0, "spoof": 1},
                           train_manifest_sha256=train.manifest_sha256,
                           dev_manifest_sha256=dev.manifest_sha256)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        else:
            stale += 1
        save_json(args.output / "progress.json", dict(epoch=epoch, best_epoch=best_epoch, best_dev_eer_percent=best_eer))
        print(f"epoch={epoch} train_loss={train_loss:.6f} dev_eer={eer:.6f} best_epoch={best_epoch}", flush=True)
        if stale >= args.patience:
            break
    save_json(args.output / "complete.json", dict(status="complete", best_epoch=best_epoch,
              best_dev_eer_percent=best_eer, epochs_completed=len(history),
              checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
              stop_reason="early_stopping" if stale >= args.patience else "epoch_limit", eval_read=False))


if __name__ == "__main__":
    main()
