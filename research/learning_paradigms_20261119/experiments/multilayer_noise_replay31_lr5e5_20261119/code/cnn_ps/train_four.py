"""Four PS arms trained from the same random CNN initialization, without old weights.

Train exactly one arm per process. Calibration and checkpoint selection use
clean dev only; test inference is a separate, completed-checkpoint operation.
No starting checkpoint or externally estimated normalization file is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .backbone import CNNBiGRU, ModelConfig
from .finetune import fine_collate, save_scores
from .io import manifest_hash, read_json, save_json, sha256
from .metrics20 import evaluate_dev
from .report_native20 import evaluate_report
from .train_dual import forward_batch, training_loss, infer


ARMS = {
    "cnn_clean": None,
    "cnn_aug": None,
    "dual": dict(variant="fusion", normalize_mode="layernorm", learned_pool=False,
                 boundary_head=False, relative_position=False, reliability_gate=False),
    "boundary": dict(variant="fusion", normalize_mode="layernorm", learned_pool=True,
                     boundary_head=True, relative_position=False, reliability_gate=False),
}
SCHEMA = "ps_four_random_cnn_v1"


def model_config():
    """The historical CNN architecture, newly initialized; no historical weights."""
    return ModelConfig(sample_rate=16000, hop_length=320, n_fft=512, win_length=400,
                       n_mels=80, channels=(16, 32, 64), hidden_size=128,
                       gru_layers=2, dropout=.4)


def setup(seed, threads):
    if threads <= 0:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_adapter(options, seed, device):
    if options is None:
        return None
    from .dual_resolution import DualResolutionAdapter
    # Seed only the CPU generator inside its saved state. torch.manual_seed
    # would also reset CUDA dropout RNG, which this extra initialization must
    # not touch. All layer constructors run on CPU before the final transfer.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = DualResolutionAdapter(torch.zeros(256), torch.ones(256), **options)
    return adapter.to(device)


def state_digest(model):
    """Content hash of initialized tensors, independent of torch.save metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class TrainingView(Dataset):
    """Share immutable three-view features; clean never rotates its channel."""
    def __init__(self, dataset, clean_only):
        if dataset.split != "train" or dataset.channels != 3:
            raise ValueError("All arms require the same completed three-view train cache")
        self.dataset, self.clean_only, self.epoch = dataset, bool(clean_only), 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        channel = (int((self.dataset.assignment[index] + self.epoch - 1) % 3)
                   if self.epoch and not self.clean_only else 0)
        return self.dataset.get_record(index, channel)


def is_better(candidate, best):
    """Event F1 only; preserve the earlier epoch within1e-10 percentage points."""
    return best is None or candidate > best["event_f1_iou05_percent"] + 1e-10


def _cpu_state(module):
    return None if module is None else {key: value.detach().cpu().clone()
                                        for key, value in module.state_dict().items()}


def train_arm(args, train, dev):
    arm = args.arm
    if arm not in ARMS or args.epochs < 0 or args.batch_size <= 0:
        raise ValueError("Invalid arm, epoch count or batch size")
    if dev.split != "dev" or dev.channels != 1 or set(train.uids) & set(dev.uids):
        raise ValueError("Require clean dev and disjoint train/dev UIDs")
    output = Path(args.output) / arm
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite arm directory: {output}")
    setup(args.seed, args.threads)
    model = CNNBiGRU(model_config()).to(args.device)
    initial_digest = state_digest(model)
    adapter = make_adapter(ARMS[arm], args.seed, args.device)
    parameters = list(model.parameters()) + ([] if adapter is None else list(adapter.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    view = TrainingView(train, clean_only=arm == "cnn_clean")
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=fine_collate, generator=torch.Generator().manual_seed(args.seed))
    boundary_enabled = arm == "boundary"
    protocol = dict(schema=SCHEMA, arm=arm, adapter_options=ARMS[arm], seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, model_config=model.config.to_dict(),
        initialization="New random CNN; identical initialization seed and tensor values across all four arms",
        initial_backbone_sha256=initial_digest, pretrained_weights_used=False,
        external_normalization_used=False, adapter_normalization="trainable per-frame LayerNorm256" if adapter else None,
        all_backbone_parameters_trainable=True, learning_rate=1e-3, weight_decay=1e-4,
        optimizer="AdamW", gradient_clip=5.,
        train_cache=str(args.train_cache), dev_cache=str(args.dev_cache),
        train_cache_metadata_sha256=sha256(train.folder / "complete.json") if hasattr(train, "folder") else None,
        dev_cache_metadata_sha256=sha256(dev.folder / "complete.json") if hasattr(dev, "folder") else None,
        train_recordings=len(train), dev_recordings=len(dev),
        train_uids_sha256=manifest_hash(train.uids), dev_uids_sha256=manifest_hash(dev.uids),
        backbone_parameters=sum(p.numel() for p in model.parameters()),
        adapter_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
        channels="Always original clean view" if arm == "cnn_clean" else "Balanced per-UID clean/noise20dB/EQ assignment, rotated by epoch",
        data_order="Independent DataLoader generator with the same seed in every arm",
        losses=dict(main20=1., main160=1., coarse_aux=.5 if adapter else 0.,
                    boundary=.1 if boundary_enabled else 0., consistency=.05 if boundary_enabled else 0.),
        final160_pool="Mean of eight stored logits; actual-duration coarse loss; storage-tail logits retained",
        selection="Clean dev genuine-duration FPR<=5%, then maximum dev event F1@IoU0.5; epoch0 eligible; earliest tie",
        test_threshold_selection=False, eval_read=False,
        comparison_scope="Fresh random CNN structure baseline; not the prior pretrained CNN result. Dual includes coarse auxiliary supervision; boundary is a three-component package.",
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                       ("train_four.py", "train_dual.py", "dual_resolution.py", "backbone.py", "finetune.py", "dual_features.py")})
    output.mkdir(parents=True)
    (output / "epochs").mkdir()
    save_json(output / "protocol.json", protocol)
    best, history, updates = None, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs + 1):
        tick, losses, components = time.monotonic(), [], []
        if epoch:
            view.epoch = epoch
            model.train()
            if adapter is not None:
                adapter.train()
            last = time.monotonic()
            for index, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                final, outputs = forward_batch(model, adapter, batch, args.device)
                loss, parts = training_loss(final, outputs, batch, args.device, boundary_enabled)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
                components.append(parts)
                if time.monotonic() - last >= 25:
                    progress = dict(arm=arm, phase="train", epoch=epoch, batches=index+1,
                                    batches_total=len(loader), updates=updates,
                                    loss=float(np.mean(losses)), elapsed_seconds=time.monotonic()-tick)
                    save_json(output / "progress.json", progress)
                    print(progress, flush=True)
                    last = time.monotonic()
        train_seconds = time.monotonic() - tick
        save_json(output / "progress.json", dict(arm=arm, phase="dev", epoch=epoch, updates=updates))
        records, scores, metadata = infer(model, adapter, dev, args.device)
        metrics = evaluate_dev(records, max_real_fpr=.05)
        metrics.pop("selection_score_j_percent", None)
        if metrics["real_fpr_percent"] > 5 + 1e-10:
            raise AssertionError("Development threshold exceeded the declared duration-FPR budget")
        value = float(metrics["event_f1_iou05_percent"])
        selected = dict(epoch=epoch, threshold=float(metrics["threshold"]), event_f1_iou05_percent=value)
        prefix = output / "epochs" / f"epoch_{epoch:02d}"
        payload = dict(schema=SCHEMA, backbone_state_dict=_cpu_state(model),
                       adapter_state_dict=_cpu_state(adapter), adapter_options=ARMS[arm],
                       model_config=model.config.to_dict(), protocol=protocol, dev_metrics=metrics, **selected)
        torch.save(payload, prefix.with_suffix(".pt"))
        save_scores(prefix.with_name(prefix.name + "_scores20.npz"), scores, metadata)
        save_json(prefix.with_name(prefix.name + "_metrics.json"), metrics)
        if is_better(value, best):
            best = selected
            shutil.copyfile(prefix.with_suffix(".pt"), output / "best.pt")
            shutil.copyfile(prefix.with_name(prefix.name + "_scores20.npz"), output / "dev_scores20.npz")
            save_json(output / "dev_metrics.json", metrics)
        row = dict(arm=arm, epoch=epoch, updates=updates,
            loss=float(np.mean(losses)) if losses else None,
            loss_components={key:float(np.mean([item[key] for item in components])) for key in components[0]} if components else {},
            train_seconds=train_seconds, total_seconds=time.monotonic()-tick, best_epoch=best["epoch"],
            **{key: metrics[key] for key in ("threshold", "event_f1_iou05_percent",
                "short_event_recall_le320ms_percent", "real_fpr_percent")})
        history.append(row)
        save_json(output / "history.json", history)
        save_json(output / "progress.json", dict(phase="epoch_complete", **row))
        print(row, flush=True)
    complete = dict(status="complete", schema=SCHEMA, arm=arm, **best,
        checkpoint="best.pt", checkpoint_sha256=sha256(output / "best.pt"),
        initial_backbone_sha256=initial_digest, last_backbone_sha256=state_digest(model),
        trained_epochs=args.epochs, updates=updates, archived_epochs=args.epochs+1,
        protocol_sha256=sha256(output / "protocol.json"), eval_read=False,
        dev_scores20_sha256=sha256(output / "dev_scores20.npz"),
        dev_metrics_sha256=sha256(output / "dev_metrics.json"),
        total_seconds=time.monotonic()-started)
    save_json(output / "complete.json", complete)
    return complete


def freeze_eval_plan(output, arms):
    """Validate all selected artifacts before any caller opens an eval cache."""
    output = Path(output)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Evaluation requires unique known arms")
    frozen, shared_conditions = {}, None
    for arm in arms:
        folder = output / arm
        complete = read_json(folder / "complete.json")
        if (complete.get("status") != "complete" or complete.get("schema") != SCHEMA
                or complete.get("arm") != arm or sha256(folder / "best.pt") != complete["checkpoint_sha256"]
                or sha256(folder / "protocol.json") != complete["protocol_sha256"]
                or sha256(folder / "dev_scores20.npz") != complete["dev_scores20_sha256"]
                or sha256(folder / "dev_metrics.json") != complete["dev_metrics_sha256"]):
            raise ValueError(f"Incomplete or changed selected checkpoint: {arm}")
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        if (payload.get("schema") != SCHEMA or payload["protocol"]["arm"] != arm
                or payload["adapter_options"] != ARMS[arm] or payload["epoch"] != complete["epoch"]
                or payload["event_f1_iou05_percent"] != complete["event_f1_iou05_percent"]
                or complete["trained_epochs"] != payload["protocol"]["epochs"]
                or payload["threshold"] != complete["threshold"]
                or payload["threshold"] != payload["dev_metrics"]["threshold"]
                or not np.isfinite(payload["threshold"]) or payload["dev_metrics"]["real_fpr_percent"] > 5 + 1e-10):
            raise ValueError(f"Selected weight, dev metric and threshold disagree: {arm}")
        conditions = {key: payload["protocol"][key] for key in
            ("seed", "epochs", "batch_size", "model_config", "learning_rate", "weight_decay", "gradient_clip",
             "train_uids_sha256", "dev_uids_sha256", "train_recordings", "dev_recordings",
             "train_cache_metadata_sha256", "dev_cache_metadata_sha256", "source_sha256")}
        if shared_conditions is not None and conditions != shared_conditions:
            raise ValueError("Requested arms differ in paired training data, budget, initialization or source code")
        shared_conditions = conditions
        frozen[arm] = complete
    if len({row["initial_backbone_sha256"] for row in frozen.values()}) != 1:
        raise ValueError("Requested arms did not share the same initial CNN tensors")
    plan = dict(schema=SCHEMA, arms=frozen, calibration="Completed dev calibration only",
                test_threshold_selection=False, eval_previously_inspected=True)
    target = output / "frozen_eval_plan.json"
    if target.exists():
        if read_json(target) != plan:
            raise ValueError("An existing evaluation plan differs; it cannot be replaced")
    else:
        save_json(target, plan)
    return frozen


def evaluate_arms(args, dataset, frozen):
    """Fixed-threshold test reporting only; no calibration helper is called."""
    if dataset.split != "eval" or dataset.channels != 1:
        raise ValueError("Evaluation requires one clean eval view")
    results = {}
    for arm in frozen:
        folder = Path(args.output) / arm
        if (folder / "eval_metrics.json").exists():
            raise FileExistsError(f"Refusing to replace completed eval report: {arm}")
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        if sha256(folder / "best.pt") != frozen[arm]["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed before evaluation")
        model = CNNBiGRU(ModelConfig(**payload["model_config"])).to(args.device)
        model.load_state_dict(payload["backbone_state_dict"], strict=True)
        adapter = make_adapter(payload["adapter_options"], payload["protocol"]["seed"], args.device)
        if adapter is not None:
            adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
        records, scores, metadata = infer(model, adapter, dataset, args.device)
        report = evaluate_report(records, threshold=float(frozen[arm]["threshold"]))
        save_scores(folder / "eval_scores20.npz", scores, metadata)
        save_json(folder / "eval_per_recording.json", report.pop("per_recording"))
        save_json(folder / "eval_metrics.json", report)
        results[arm] = report
        print(dict(arm=arm, event_f1=report["event_f1_iou05_percent"],
                   real_fpr=report["real_duration_fpr_percent"]), flush=True)
    save_json(Path(args.output) / "comparison.json", dict(status="complete", schema=SCHEMA,
        metric_protocol="ps_native20_event_v2", models=results, frozen=frozen,
        test_threshold_selection=False, eval_previously_inspected=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), help="One training arm per process")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), help="Arms to freeze/evaluate; default all four")
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--dev-cache", type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("epochs must be nonnegative; batch-size/threads must be positive")
    if args.evaluate or args.freeze_only:
        if args.arm or args.train_cache or args.dev_cache or (args.freeze_only and args.eval_cache):
            parser.error("Freeze/evaluate cannot accept a training arm or training caches")
        if args.evaluate and not args.eval_cache:
            parser.error("--evaluate requires --eval-cache")
    elif not args.arm or not args.train_cache or not args.dev_cache or args.eval_cache or args.arms:
        parser.error("Training needs --arm, --train-cache and --dev-cache, and cannot receive eval data")
    setup(args.seed, args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; explicitly choose --device cpu")
    from .dual_features import FeatureData
    if args.evaluate or args.freeze_only:
        frozen = freeze_eval_plan(args.output, args.arms or list(ARMS))
        if not args.freeze_only:
            dataset = FeatureData(args.eval_cache, split="eval", seed=args.seed,
                                  config=model_config(), verify_hashes=True)
            evaluate_arms(args, dataset, frozen)
    else:
        train = FeatureData(args.train_cache, split="train", seed=args.seed,
                            config=model_config(), verify_hashes=True)
        dev = FeatureData(args.dev_cache, split="dev", seed=args.seed,
                          config=model_config(), verify_hashes=True)
        train_arm(args, train, dev)


if __name__ == "__main__":
    main()
