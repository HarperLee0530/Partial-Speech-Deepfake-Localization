"""Plan C training, using the audited A0 components and DEV-only protocol."""
import argparse
import csv
from itertools import islice
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cnn_ps.finetune import fine_collate, save_scores
from cnn_ps.io import manifest_hash, read_json, save_json, sha256
from cnn_ps.metrics20 import evaluate_dev
from cnn_ps.train_dual import forward_batch, infer, training_loss
from cnn_ps.train_wavlm_multilayer import (
    ARMS as ORIGINAL_ARMS, TrainingView, _cpu_state, is_better, make_adapter,
    model_config, setup, state_digest,
)
from cnn_ps.wavlm_frontend import REVISION
from cnn_ps.wavlm_multilayer import LayerFusionCNN, MultiLayerData

from event_losses import event_losses, prepare_regions
from scripts.monitor_training import process_start, query_nvidia

BASE = "fusion6_9_12"
ARMS = {"A0": (0., 0.), "C1": (.1, 0.), "C2": (0., .1), "C1+C2": (.1, .1)}
HISTORICAL_CHECKPOINT_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
UID_DIGESTS = {"train": "3b02c95393d0de6c632e7313312f53650879c369efe1923baf78566772b186f7",
               "dev": "10bfc5d5e8ff2cf86b67702e48876b1115e23ec8f30352d73276958ea38c53e1"}
EPOCHS, BATCH_SIZE = 20, 8


class Progress:
    def __init__(self, output, log, active, args):
        self.output = output
        self.active = active
        self.last_gpu_sample = 0.
        self.state = dict(arm=args.arm, seed=args.seed, mode="smoke" if args.smoke else "formal",
                          pid=os.getpid(), pid_start=process_start(os.getpid()),
                          started_at=time.time(), log_path=str(log), checkpoint_dir=str(output),
                          gpu_uuid=args.gpu_uuid, total_epochs=1 if args.smoke else EPOCHS,
                          epoch=0, batch=0, updates=0)

    def publish(self, phase, **values):
        self.state.update(values, phase=phase, updated_at=time.time())
        if time.monotonic() - self.last_gpu_sample >= 5:
            self.last_gpu_sample = time.monotonic()
            fields = query_nvidia("--query-gpu=uuid,utilization.gpu,memory.used,memory.total,temperature.gpu")
            selected = next((r for r in (fields or []) if len(r) == 5 and r[0] == self.state["gpu_uuid"]), None)
            if selected:
                path = self.output / "gpu_samples.csv"
                first = not path.exists()
                with path.open("a", newline="") as stream:
                    writer = csv.writer(stream)
                    if first:
                        writer.writerow(("unix_time", "phase", "pid", "gpu_uuid", "utilization_percent",
                                         "used_mib", "total_mib", "temp_c", "nndl_reserved_mib"))
                    writer.writerow((self.state["updated_at"], phase, self.state["pid"], *selected,
                                     self.state.get("cuda_reserved_mib")))
        save_json(self.output / "progress.json", self.state)
        save_json(self.active, self.state)


def cache_datasets(args):
    results = []
    for split, count in (("train", 6345), ("dev", 2484)):
        data = MultiLayerData(args.cache / "reference" / split, args.cache / "layers" / split,
                              arm=BASE, split=split, seed=args.seed, config=model_config(),
                              verify_hashes=True)
        meta = data.ssl_metadata
        if (len(data) != count or manifest_hash(data.uids) != UID_DIGESTS[split]
                or meta["model_id"] != "microsoft/wavlm-base-plus" or meta["revision"] != REVISION
                or meta["model_sha256"] != HISTORICAL_CHECKPOINT_SHA256
                or meta["split"] != split or meta["test_read"] or meta["dtype"] != "float32"
                or data.ssl.dtype != np.float32 or data.channels != (3 if split == "train" else 1)):
            raise ValueError(f"{split}: cache differs from the frozen A0 data protocol")
        results.append(data)
    return results


def training_batch(model, adapter, batch, optimizer, parameters, weights, *, verify=False):
    optimizer.zero_grad(set_to_none=True)
    final, outputs = forward_batch(model, adapter, batch, "cuda")
    loss, parts = training_loss(final, outputs, batch, "cuda", True)
    if any(weights):
        regions = prepare_regions(batch["fine_labels"], batch["fine_weights"],
                                  c1=bool(weights[0]), c2=bool(weights[1]))
        terms = event_losses(final, regions)
        for key, coefficient, term in zip(("C1", "C2"), weights, terms):
            if coefficient:
                parts[key] = float(term.detach())
                loss = loss + coefficient * term
        if verify:
            for key, coefficient, term, region in zip(("C1", "C2"), weights, terms, regions):
                if coefficient and region is not None:
                    gradient, = torch.autograd.grad(term, final, retain_graph=True)
                    if not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
                        raise ValueError(f"Nonfinite or zero {key} gradient on eligible smoke batch")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    optimizer.step()
    if verify and (not final.is_cuda or not loss.is_cuda or not grad_norm.is_cuda
                   or not all(p.is_cuda and p.grad is not None and p.grad.is_cuda for p in parameters)):
        raise RuntimeError("First optimizer batch failed CUDA placement checks")
    return float(loss.detach()), parts


def verify_preflight(args):
    checks = read_json(args.cache / "verified.json")
    audit = args.workspace / "data" / "manifests" / "audit.json"
    if checks["status"] != "verified" or checks["audio_audit_sha256"] != sha256(audit):
        raise ValueError("Verified cache must match the reduced dataset audit")
    for kind in ("reference", "layers"):
        for split in ("train", "dev"):
            key = f"{kind}/{split}"
            if checks["cache_metadata"][key] != sha256(args.cache / key / "complete.json"):
                raise ValueError(f"Cache metadata changed since audit: {key}")
    if Path(sys.prefix).resolve() != (args.workspace / ".venv").resolve():
        raise RuntimeError("Training requires the dedicated NNDL Python environment")


def train(args, output, progress):
    verify_preflight(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device selected by GPU UUID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match the audited GPU UUID")
    setup(args.seed, args.threads)
    train_data, dev_data = cache_datasets(args)
    model = LayerFusionCNN(model_config(), BASE, args.seed).cuda()
    adapter = make_adapter(ORIGINAL_ARMS[BASE], args.seed, "cuda")
    initial_digest = state_digest(model.cnn)
    parameters = list(model.parameters()) + list(adapter.parameters())
    weights = ARMS[args.arm]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    view = TrainingView(train_data, clean_only=False)
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=fine_collate, generator=torch.Generator().manual_seed(args.seed))
    if len(loader) != 794:
        raise ValueError("Batch count did not match historical A0")
    epochs, batches = (1, 12) if args.smoke else (EPOCHS, len(loader))
    total = epochs * batches
    physical_index = int(subprocess.check_output(
        ["nvidia-smi", "-i", args.gpu_uuid, "--query-gpu=index", "--format=csv,noheader"],
        text=True, timeout=10).strip())
    setup_fields = dict(total_batches=batches, total_updates=total, actual_device="cuda:0",
                        gpu_index=physical_index, device_name=torch.cuda.get_device_name(0))
    progress.publish("Preparing", **setup_fields)
    protocol = dict(schema="nndl_event_aware_c_v1", arm=args.arm, mode=progress.state["mode"],
                    seed=args.seed, epochs=epochs, batch_size=BATCH_SIZE, batches_per_epoch=batches,
                    gpu_uuid=args.gpu_uuid, physical_gpu_index=physical_index, device=setup_fields["actual_device"],
                    gpu_name=setup_fields["device_name"], torch_version=torch.__version__,
                    torch_cuda=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                    started_at=progress.state["started_at"], pid=os.getpid(),
                    initial_backbone_sha256=initial_digest, model_config=model_config().to_dict(),
                    adapter_options=ORIGINAL_ARMS[BASE], layers=[6, 9, 12],
                    optimizer="AdamW", learning_rate=.001, weight_decay=.0001, gradient_clip=5.,
                    losses=dict(main20=1., main160=1., coarse_aux=.5, boundary=.1,
                                consistency=.05, C1=weights[0], C2=weights[1]),
                    train_cache_sha256=sha256(train_data.folder / "complete.json"),
                    dev_cache_sha256=sha256(dev_data.folder / "complete.json"),
                    train_ssl=train_data.ssl_metadata, dev_ssl=dev_data.ssl_metadata,
                    train_uid_sha256=manifest_hash(train_data.uids), dev_uid_sha256=manifest_hash(dev_data.uids),
                    selection="Epoch 0 eligible. DEV real-duration FPR <=5%; maximum Event-F1 at IoU 0.5; earliest tie.",
                    augmentation="Three original clean/noise20dB/EQ views, rotated by epoch",
                    historical_base_commit="483c0252cf8b7104376399008d21d414fdced0b6", eval_read=False,
                    implementation_sha256={name: sha256(Path(__file__).with_name(name))
                                           for name in ("trainer.py", "event_losses.py")})
    save_json(output / "protocol.json", protocol)
    (output / "epochs").mkdir()
    updates, history, best = 0, [], None
    started = time.monotonic()
    observed = set()
    dev_view = dev_data
    if args.smoke:
        dev_view = Subset(dev_data, range(16))
        dev_view.uids = dev_data.uids[:16]
        dev_view.fine = dev_data.fine

    for epoch in range(epochs + 1):
        tick = time.monotonic()
        losses, components = [], []
        if epoch:
            model.train()
            adapter.train()
            view.epoch = epoch
            progress.publish("Training", epoch=epoch, batch=0)
            last = time.monotonic()
            for index, batch in enumerate(islice(loader, batches), 1):
                loss, parts = training_batch(model, adapter, batch, optimizer, parameters, weights,
                                             verify=updates == 0 or args.smoke)
                updates += 1
                losses.append(loss)
                components.append(parts)
                observed.update(key for key in ("C1", "C2") if parts.get(key, 0.) > 0)
                progress.state.update(last_batch_at=time.time(), batch=index, updates=updates, loss=loss,
                                      cuda_verified=True, cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20)
                if updates == 1 or index == batches or time.monotonic() - last >= 5:
                    progress.publish("Training")
                    print(f"CUDA epoch={epoch} batch={index}/{batches} updates={updates} loss={loss:.6f}"
                          f" reserved_mib={progress.state['cuda_reserved_mib']:.0f}", flush=True)
                    last = time.monotonic()
        train_seconds = time.monotonic() - tick
        progress.publish("Validating", epoch=epoch, updates=updates)
        records, scores, metadata = infer(model, adapter, dev_view, "cuda")
        metrics = evaluate_dev(records, max_real_fpr=.05)
        metrics.pop("selection_score_j_percent", None)
        if metrics["real_fpr_percent"] > 5 + 1e-10:
            raise RuntimeError("Development threshold violates original duration FPR cap")
        value = float(metrics["event_f1_iou05_percent"])
        selected = dict(epoch=epoch, threshold=float(metrics["threshold"]), event_f1_iou05_percent=value,
                        real_fpr_percent=metrics["real_fpr_percent"])
        prefix = output / "epochs" / f"epoch_{epoch:02d}"
        payload = dict(schema=protocol["schema"], layer_weights=model.layer_weights().detach().cpu().tolist(),
                       backbone_state_dict=_cpu_state(model), adapter_state_dict=_cpu_state(adapter),
                       adapter_options=ORIGINAL_ARMS[BASE], model_config=model_config().to_dict(),
                       protocol=protocol, dev_metrics=metrics, **selected)
        torch.save(payload, prefix.with_suffix(".pt"))
        score_file = prefix.with_name(prefix.name + "_scores20.npz")
        save_scores(score_file, scores, metadata)
        save_json(prefix.with_name(prefix.name + "_metrics.json"), metrics)
        if is_better(value, best):
            best = selected
            shutil.copyfile(prefix.with_suffix(".pt"), output / "best.pt")
            shutil.copyfile(score_file, output / "dev_scores20.npz")
            save_json(output / "dev_metrics.json", metrics)
        row = dict(epoch=epoch, updates=updates, loss=float(np.mean(losses)) if losses else None,
                   loss_components={key: float(np.mean([part[key] for part in components]))
                                    for key in components[0]} if components else {},
                   train_seconds=train_seconds, total_seconds=time.monotonic() - tick,
                   cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, **{
                       k: metrics[k] for k in ("event_f1_iou05_percent", "real_fpr_percent", "threshold")})
        history.append(row)
        save_json(output / "history.json", history)
        progress.publish("Validating", last_event_f1=value, last_real_fpr=metrics["real_fpr_percent"],
                         best_epoch=best["epoch"], best_event_f1=best["event_f1_iou05_percent"],
                         best_real_fpr=best["real_fpr_percent"])
        print(row, flush=True)

    if updates != total or len(history) != epochs + 1:
        raise RuntimeError("Training budget did not complete")
    if args.smoke and any(weight and key not in observed for key, weight in zip(("C1", "C2"), weights)):
        raise RuntimeError("No eligible active auxiliary region was observed in the smoke batches")
    complete = dict(best, mode=protocol["mode"], status="complete", arm=args.arm, seed=args.seed,
                    trained_epochs=epochs, updates=updates, actual_device="cuda:0", gpu_uuid=args.gpu_uuid,
                    checkpoint_sha256=sha256(output / "best.pt"), total_seconds=time.monotonic() - started,
                    dev_scores20_sha256=sha256(output / "dev_scores20.npz"),
                    protocol_sha256=sha256(output / "protocol.json"),
                    active_auxiliary_terms=sorted(observed), gpu_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    optimizer_and_loading_seconds=sum(row["train_seconds"] for row in history))
    save_json(output / "complete.json", complete)
    progress.publish("Completed", finished_at=time.time())
    print(complete, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(666, 667, 668), default=666)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace = args.workspace.resolve()
    args.cache = args.cache.resolve()
    output = args.run.resolve()
    log = args.log.resolve()
    if (not output.is_relative_to(workspace / "experiments" / "event_aware_c")
            or not log.is_relative_to(workspace / "logs")
            or not args.cache.resolve().is_relative_to(workspace / "cache")):
        parser.error("All run, cache and log paths must stay inside the dedicated workspace")
    output.mkdir(parents=True, exist_ok=False)
    progress = Progress(output, log, workspace / "logs" / "active.json", args)
    progress.publish("Preparing")
    try:
        train(args, output, progress)
    except BaseException as error:
        progress.publish("Failed", error=f"{type(error).__name__}: {error}", finished_at=time.time())
        raise


if __name__ == "__main__":
    main()
