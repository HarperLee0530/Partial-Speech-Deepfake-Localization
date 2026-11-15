"""Fine-tune all original CNN-BiGRU parameters with three waveform channels.

This is a separate experiment from the frozen-CNN residual-head main method.
No residual head is added. Train/dev manifests and genuine fine labels are
explicit; this command never reads eval audio or eval labels.
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

from .augmentation import transform, CHANNELS
from .backbone import LogMel, load_backbone, pool_logits
from .baseline import collate
from .io import load_manifest, save_json, sha256, manifest_hash
from .metrics20 import evaluate_dev


class FineData(Dataset):
    """Full waveforms, explicit actual 20 ms labels, no duration-changing edits."""
    def __init__(self, manifest, labels20_file, config, split, seed):
        self.records = load_manifest(manifest, expected_split=split)
        self.uids = [row["uid"] for row in self.records]
        self.split, self.epoch = split, 0
        self.frontend = LogMel(config)
        self.coarse, self.fine, self.label_files = {}, {}, {}
        with np.load(labels20_file, allow_pickle=False) as archive:
            for row in self.records:
                uid = row["uid"]
                fine = np.asarray(archive[uid], dtype=np.float32).copy()
                filename = row["labels_file"]
                if filename not in self.label_files:
                    self.label_files[filename] = np.load(filename, allow_pickle=True).item()
                original = np.asarray(self.label_files[filename][uid], dtype=np.float32)
                if (original.ndim != 1 or fine.ndim != 1 or not len(original) or not len(fine)
                        or len(fine) > 8 * len(original)
                        or not np.isin(original, (0, 1)).all() or not np.isin(fine, (0, 1)).all()):
                    raise ValueError(f"{uid}: invalid original 160 ms or actual 20 ms labels")
                self.coarse[uid], self.fine[uid] = 1 - original, fine
        assignment = np.arange(len(self.records)) % 3
        np.random.default_rng(seed + 20261113).shuffle(assignment)
        self.assignment = assignment
        # Cache clean dev features only; no random transform or model output is cached.
        self.dev_items = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.split == "dev" and index in self.dev_items:
            return self.dev_items[index]
        row = self.records[index]
        uid = row["uid"]
        wave, sr = sf.read(row["audio"], dtype="float32")
        if sr != 16000 or wave.ndim != 1 or not len(wave):
            raise ValueError(f"{uid}: expected nonempty 16 kHz mono audio")
        duration = len(wave) / sr
        coarse, fine = self.coarse[uid], self.fine[uid]
        if len(fine) * .02 < duration - 1e-9:
            raise ValueError(f"{uid}: actual fine labels do not cover the waveform")
        channel = int((self.assignment[index] + self.epoch - 1) % 3) if self.split == "train" and self.epoch else 0
        changed = transform(wave, uid, channel)
        padding = len(coarse) * 2560 - len(wave)
        if padding < 0:
            raise ValueError(f"{uid}: coarse labels do not cover the waveform")
        features = self.frontend(F.pad(torch.from_numpy(changed), (0, padding)))
        n20 = len(coarse) * 8
        fine_labels = np.zeros(n20, dtype=np.float32)
        fine_labels[:len(fine)] = fine
        fine_weights = np.zeros(n20, dtype=np.float32)
        fine_weights[:len(fine)] = np.maximum(0., np.minimum((np.arange(len(fine)) + 1) * .02, duration) - np.arange(len(fine)) * .02)
        coarse_weights = np.maximum(0., np.minimum((np.arange(len(coarse)) + 1) * .16, duration) - np.arange(len(coarse)) * .16).astype(np.float32)
        item = dict(uid=uid, features=features, labels=torch.from_numpy(coarse.copy()), duration=duration,
                    fine_labels=torch.from_numpy(fine_labels), fine_weights=torch.from_numpy(fine_weights),
                    weights=torch.from_numpy(coarse_weights), length20=len(fine), channel=channel)
        if features.shape[1] != n20:
            raise ValueError("Feature grid disagrees with 160 ms storage length")
        if self.split == "dev":
            self.dev_items[index] = item
        return item


def fine_collate(items):
    batch = collate(items)
    for name in ("fine_labels", "fine_weights"):
        batch[name] = torch.stack([F.pad(row[name], (0, int(batch["lengths"].max()) - len(row[name]))) for row in items])
    batch["weights"] = torch.stack([F.pad(row["weights"], (0, int(batch["label_lengths"].max()) - len(row["weights"]))) for row in items])
    batch["lengths20"] = [row["length20"] for row in items]
    batch["channels"] = [row["channel"] for row in items]
    return batch


def objective(native, batch, device):
    coarse, lengths = pool_logits(native, batch["lengths"], .16)
    if not torch.equal(lengths, batch["label_lengths"]):
        raise ValueError("Fine-tuning coarse target alignment mismatch")
    cw, fw = batch["weights"].to(device), batch["fine_weights"].to(device)
    coarse_loss = F.binary_cross_entropy_with_logits(coarse, batch["labels"].to(device), reduction="none")
    fine_loss = F.binary_cross_entropy_with_logits(native, batch["fine_labels"].to(device), reduction="none")
    return (coarse_loss * cw).sum() / cw.sum().clamp_min(1e-8) + (fine_loss * fw).sum() / fw.sum().clamp_min(1e-8)


@torch.inference_mode()
def validate(model, dataset, device, audit_original=False):
    model.eval()
    records, stored, lengths, lengths20, durations = [], [], [], [], []
    max_difference, compared, violations = 0., 0, 0
    for index, batch in enumerate(DataLoader(dataset, batch_size=1, collate_fn=fine_collate, num_workers=0)):
        native = model(batch["features"].to(device), batch["lengths"])[0].float().cpu().numpy()
        n = int(batch["lengths"][0])
        native = native[:n].copy()
        row = dataset.records[index]
        uid, duration = row["uid"], batch["durations"][0]
        records.append(dict(uid=uid, scores20=native, labels20=dataset.fine[uid], duration=duration))
        stored.append(native)
        lengths.append(n // 8)
        lengths20.append(batch["lengths20"][0])
        durations.append(duration)
        if audit_original:
            if "path" not in row:
                raise ValueError("Epoch-zero audit requires original native-cache paths in dev manifest")
            with np.load(row["path"], allow_pickle=False) as archive:
                reference = np.asarray(archive["native_logits"], dtype=np.float32).reshape(-1)
            if reference.shape != native.shape:
                raise ValueError(f"Epoch-zero cached score length mismatch: {uid}")
            max_difference = max(max_difference, float(np.max(np.abs(reference - native))))
            violations += not np.allclose(reference, native, rtol=2e-5, atol=2e-4)
            compared += 1
        if (index + 1) % 500 == 0:
            print(f"dev {index+1}/{len(dataset)}", flush=True)
    metric = evaluate_dev(records, max_real_fpr=.05)
    metadata = dict(uids=dataset.uids, lengths=lengths, lengths20=lengths20, durations=durations)
    audit = dict(compared_recordings=compared, max_absolute_logit_difference=max_difference,
                 tolerance_rtol=2e-5, tolerance_atol=2e-4, recordings_outside_tolerance=int(violations))
    return metric, np.concatenate(stored), metadata, audit


def save_scores(path, scores, metadata):
    np.savez_compressed(path, scores20=scores, uids=np.asarray(metadata["uids"]),
                        lengths=np.asarray(metadata["lengths"], dtype=np.int64),
                        lengths20=np.asarray(metadata["lengths20"], dtype=np.int64),
                        durations=np.asarray(metadata["durations"], dtype=np.float64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--train-labels20", type=Path, required=True)
    parser.add_argument("--dev-labels20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--audit-original-cache", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.cpu_threads) <= 0 or args.learning_rate <= 0:
        parser.error("Training settings must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output is not empty; select a new experiment directory")
    source_hash = sha256(args.checkpoint)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise ValueError("Starting CNN checkpoint hash differs from the declared source")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, _ = load_backbone(args.checkpoint, device=args.device)
    model.requires_grad_(True)
    train = FineData(args.train_manifest, args.train_labels20, model.config, "train", args.seed)
    dev = FineData(args.dev_manifest, args.dev_labels20, model.config, "dev", args.seed)
    if set(train.uids) & set(dev.uids):
        raise ValueError("Train/dev IDs overlap")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(arm="cnn_channel_finetune", model_config=model.config.to_dict(),
                    backbone_source_sha256=source_hash, train_utterances=len(train), dev_utterances=len(dev),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    train_manifest_sha256=manifest_hash(train.uids), dev_manifest_sha256=manifest_hash(dev.uids),
                    train_labels20_sha256=sha256(args.train_labels20), dev_labels20_sha256=sha256(args.dev_labels20),
                    channels=list(CHANNELS), assignment_seed=args.seed + 20261113,
                    channel_assignment="(shuffled balanced per-record assignment + epoch - 1) modulo 3",
                    input="Full recording waveform, transform before label-grid padding, no crop",
                    loss="Duration-weighted coarse160 BCE + actual20 BCE, coefficients 1 and 1",
                    optimizer="AdamW, weight_decay=1e-4, gradient clip norm=5",
                    selection="Dev actual20 real-duration FPR<=5%; J=.5 durationF1+.5 eventF1, epoch0 eligible, earliest ties",
                    source_code_sha256=sha256(__file__), augmentation_sha256=sha256(Path(__file__).with_name("augmentation.py")),
                    eval_read=False)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "protocol.json", settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=fine_collate, generator=generator)
    best, best_scores, history = None, None, []
    checkpoint = args.output / "cnn_best.pt"
    start = time.monotonic()
    for epoch in range(args.epochs + 1):
        tick = time.monotonic()
        loss_sum = batches = 0
        train_seconds = 0.
        if epoch:
            train.epoch = epoch
            model.train()
            report_tick = time.monotonic()
            for batch_index, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                native = model(batch["features"].to(args.device), batch["lengths"])
                loss = objective(native, batch, args.device)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fine-tuning loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach())
                batches += 1
                if time.monotonic() - report_tick >= 30:
                    progress = dict(epoch=epoch, phase="train", batches_completed=batch_index + 1,
                                    batches_total=len(loader), loss_mean=loss_sum / batches,
                                    epoch_elapsed_seconds=time.monotonic() - tick)
                    save_json(args.output / "progress.json", progress)
                    print(progress, flush=True)
                    report_tick = time.monotonic()
            train_seconds = time.monotonic() - tick
        save_json(args.output / "progress.json", dict(epoch=epoch, phase="dev", train_seconds=train_seconds))
        metric, scores, metadata, audit = validate(model, dev, args.device,
                                                 audit_original=args.audit_original_cache and epoch == 0)
        if epoch == 0:
            save_json(args.output / "epoch_zero_dev.json", metric)
            save_json(args.output / "epoch_zero_cache_audit.json", audit)
            save_scores(args.output / "epoch_zero_dev_scores20.npz", scores, metadata)
            if args.audit_original_cache and audit["recordings_outside_tolerance"]:
                raise ValueError("Original CNN epoch-zero parity check failed; see cache audit")
        j = metric["selection_score_j_percent"]
        if best is None or j > best["selection_score_j_percent"] + 1e-10:
            best = dict(epoch=epoch, threshold=metric["threshold"], selection_score_j_percent=j)
            best_scores = scores.copy()
            payload = dict(state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                           model_config=model.config.to_dict(), arm="cnn_channel_finetune", seed=args.seed,
                           backbone_source_sha256=source_hash, dev_metrics=metric, config=settings,
                           trained=True, internal_labels={"bonafide": 0, "spoof": 1}, **best)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
            save_scores(args.output / "dev_scores20.npz", scores, metadata)
            save_json(args.output / "dev_metrics.json", metric)
        row = dict(epoch=epoch, loss_mean=loss_sum / batches if batches else None,
                   train_seconds=train_seconds, total_seconds=time.monotonic() - tick,
                   best_epoch=best["epoch"], **{key: metric[key] for key in
                       ("threshold", "selection_score_j_percent", "duration_f1_percent", "event_f1_iou05_percent")})
        history.append(row)
        save_json(args.output / "history.json", history)
        save_json(args.output / "progress.json", dict(phase="epoch_complete", **row))
        print(row, flush=True)
    selected, _ = load_backbone(checkpoint, device=args.device)
    first_batch = fine_collate([dev[0]])
    with torch.inference_mode():
        reloaded = selected(first_batch["features"].to(args.device), first_batch["lengths"])[0].cpu().numpy()
    np.testing.assert_array_equal(reloaded, best_scores[:len(reloaded)])
    save_json(args.output / "complete.json", dict(status="complete", **best, seed=args.seed,
              checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
              backbone_source_sha256=source_hash, trained_epochs=args.epochs,
              elapsed_seconds=time.monotonic() - start, reload_first_record_exact=True, eval_read=False))


if __name__ == "__main__":
    main()
