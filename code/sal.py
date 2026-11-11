"""SAL on fixed reduced roles; official model/loss preserved. Default check never trains."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from types import MethodType
import sqlite3
import sys
import time
import traceback
import msvcrt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_MODE = "check"  # Importing/clicking never starts training.
from project_config import CONFIG, DATA_ROOT as STORE, RESULT_ROOT, LOG_ROOT, checkpoint_digest, portable_checkpoint_path, artifact_relative
from data import dataset_paths, profile_info, read_manifest, canonical_split, label_split, validate_reduced
ACTIVE_OUT = None
ACTIVE_CONNECTION = None
ACTIVE_LOCK = None
RUN_INFO = {}
RUN_STARTED = None
sys.path.insert(0, str(ROOT / 'vendor' / 'SAL'))
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    for attempt in range(6):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1)


def local_time():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


def begin_run(split, limit, run_name=None, data_profile='reduced'):
    """Reserve a new directory; never overwrite or resume an old experiment."""
    global ACTIVE_OUT, RUN_INFO, RUN_STARTED
    base = f'sal_wavlm_{split}' + (f'_{data_profile}_smoke{limit}' if limit else f'_{data_profile}')
    run = run_name or f'{base}_{local_time():%Y%m%d_%H%M%S_%f}'
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run):
        raise ValueError('Run name may only contain letters, digits, underscores and hyphens')
    out = RESULT_ROOT / run
    out.mkdir(parents=True, exist_ok=False)
    ACTIVE_OUT = out
    RUN_STARTED = time.monotonic()
    RUN_INFO = dict(run=run, split=split, smoke_test=bool(limit), requested_limit=limit, **profile_info(split, data_profile),
                    started_local=local_time().isoformat(timespec='seconds'),
                    completed=0, planned_utterances=None,
                    dataset=('LlamaPartialSpoof crossfade' if split == 'lps' else f'PartialSpoof {split}'),
                    checkpoint=portable_checkpoint_path(), metric_unit='percent',
                    detail_directory=run, predictions_file=f'{run}/predictions.sqlite')
    record_run('initializing')
    return run, out


def record_run(status, **updates):
    """Maintain one readable JSON record at the top level of result."""
    RUN_INFO.update(updates)
    RUN_INFO.update(status=status, updated_local=local_time().isoformat(timespec='seconds'),
                    elapsed_wall_seconds=round(time.monotonic()-RUN_STARTED, 3))
    if status in ('complete', 'failed', 'interrupted'):
        RUN_INFO['finished_local'] = local_time().isoformat(timespec='seconds')
    write_json(RESULT_ROOT / f"{RUN_INFO['run']}.json", RUN_INFO)


def finish_failed_run(status, error):
    """Preserve completed predictions and record a normal error or Ctrl+C."""
    if ACTIVE_OUT is None:
        return
    completed = RUN_INFO.get('completed', 0)
    if ACTIVE_CONNECTION is not None:
        ACTIVE_CONNECTION.commit()
        completed = ACTIVE_CONNECTION.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    details = dict(status=status, run=RUN_INFO['run'], completed=completed,
                   total=RUN_INFO.get('planned_utterances'), error=repr(error),
                   updated_local=local_time().isoformat(timespec='seconds'))
    write_json(ACTIVE_OUT/'status.json', details)
    record_run(status, error=repr(error), completed=completed)


def get_dataset(split, data_profile='reduced'):
    """Load one frozen reduced role using the unchanged vendor dataset/labels."""
    split = canonical_split(split)
    if data_profile != 'reduced':
        raise ValueError('Only the reduced data profile is supported')
    audio_root, label_root = dataset_paths(split, data_profile)
    manifest = read_manifest(split)
    if split == 'lps':
        from src.data.llama_partialspoof import LlamaPartialSpoofDataset
        dataset = LlamaPartialSpoofDataset(root=str(audio_root), split=label_split(split), sample_rate=16000,
            resolution=0.02, input_query='*.wav', label_root=str(label_root),
            max_label_len=200, pad_mode='test', add_label=False)
        resolution = 0.02
    else:
        from src.data.partialspoof import PartialSpoofDataset
        dataset = PartialSpoofDataset(root=str(audio_root), split=label_split(split), sample_rate=16000,
            resolution=0.16, input_query='*.wav', label_root=str(label_root),
            max_label_len=25, pad_mode='test', add_label=False, use_augmentation=False)
        resolution = 0.16
    if (dataset.utt_ids != manifest['utterance_ids'] or set(dataset.labels) != set(dataset.utt_ids)):
        raise ValueError(f'Reduced audio/label/manifest mismatch: {split}')
    return dataset, resolution


def training_identity():
    return dict(training_data_profile='reduced', training_split='train_detector',
        training_manifest_sha256=read_manifest('train_detector')['manifest_sha256'],
        training_dev_manifest_sha256=read_manifest('dev')['manifest_sha256'],
        csm_implementation='aligned_cross_recording_v1')


def validate_training_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Expected a reduced-trained SAL checkpoint; use public explicitly for a reference')
    expected = training_identity()
    if metadata.get('protocol') != 'sal_reduced_v1' or any(metadata.get(k) != v for k,v in expected.items()):
        raise ValueError('SAL checkpoint training role/profile/manifests differ from this reduced experiment')
    return expected


def resolve_checkpoint(value):
    if value == 'public':
        return STORE/'models/SAL_WavLM.ckpt'
    if value == 'latest':
        matches = []
        for file in RESULT_ROOT.glob('sal_train_reduced_*.json'):
            row = json.loads(file.read_text(encoding='utf-8'))
            if row.get('status') != 'complete':
                continue
            try:
                validate_training_metadata(row)
            except ValueError:
                continue
            matches.append((file.name,row))
        if not matches:
            raise FileNotFoundError('No completed SAL training on the current reduced detector/dev manifests')
        row = max(matches,key=lambda item:item[0])[1]
        path = (ROOT/row['checkpoint']).resolve()
        if file_digest(path) != row['checkpoint_sha256']:
            raise ValueError('Trained SAL checkpoint checksum differs')
        return path
    if not value:
        raise ValueError('Specify checkpoint public, latest, or a reduced-trained checkpoint path')
    return (ROOT/value).resolve()


def configured_device():
    device = CONFIG['sal']['device']
    if device not in ('cpu', 'cuda'):
        raise ValueError('SAL device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select cpu explicitly in config.json:sal')
    return device


def load_model(resolution, checkpoint_path=None, public_reference=False, device=None):
    device = configured_device() if device is None else device
    from src.models.net.model import SSL_SeqModel_8Labels_2Loss
    model = SSL_SeqModel_8Labels_2Loss(ssl_encoder='large', mode='s3prl_weighted',
        ckpt=str(STORE/'models/wavlm_large.pt'), hid_dim=1024, seq_model='cf',
        resolution_train=0.16, resolution_test=resolution, pool='avg')
    checkpoint = torch.load(str(checkpoint_path or STORE/'models/SAL_WavLM.ckpt'), map_location='cpu', mmap=True)
    if not public_reference:
        validate_training_metadata(checkpoint.get('project_training'))
    state = checkpoint['state_dict']
    assert all(k.startswith('net.') for k in state)
    loaded = model.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True)
    metadata = dict(epoch=checkpoint['epoch'], global_step=checkpoint['global_step'],
        lightning_version=checkpoint['pytorch-lightning_version'],
        hyper_parameters={k: str(v) for k, v in checkpoint.get('hyper_parameters', {}).items()},
        callbacks=str(checkpoint.get('callbacks', {})), state_dict_keys=len(state), strict_load=str(loaded),
        project_training=checkpoint.get('project_training'))
    del checkpoint, state
    gc.collect()
    return model.float().eval().to(device), metadata


def infer_waveform(model, waveform, device):
    """FP32 whole-utterance inference; retry CUDA OOM on CPU only."""
    if device not in ('cpu', 'cuda'):
        raise ValueError('SAL inference device must be cpu or cuda')
    if device == 'cuda':
        torch.cuda.synchronize()
    try:
        _, logits = model(waveform.unsqueeze(0).to(device=device, dtype=torch.float32))
    except torch.cuda.OutOfMemoryError:
        if device != 'cuda':
            raise
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        _, logits = model(waveform.unsqueeze(0).to(device='cpu', dtype=torch.float32))
        result = logits[0].float().cpu()
        model.to(device)
        torch.cuda.synchronize()
        return result, 'cpu_oom_fallback'
    result = logits[0].float().cpu()
    if device == 'cuda':
        torch.cuda.synchronize()
    return result, device


def calculate_metrics(connection):
    from src.evaluator.evaluator import EERMetric, F1Metric
    rows = connection.execute('SELECT logits, labels FROM predictions ORDER BY ordinal').fetchall()
    logits = torch.from_numpy(np.concatenate([np.frombuffer(r[0], dtype=np.float32).reshape(-1, 2) for r in rows]))
    labels = torch.from_numpy(np.concatenate([np.frombuffer(r[1], dtype=np.uint8) for r in rows]).astype(np.float32))
    eer = EERMetric()
    f1 = F1Metric()
    eer.update(logits[:, 1] - logits[:, 0], labels)
    f1.update(logits, labels)
    eer_value, threshold = eer.compute()
    accuracy, f1_value = f1.compute()
    predicted = logits.argmax(1)
    confusion = torch.bincount((labels.long()*2 + predicted), minlength=4).reshape(2, 2)
    tn, fp, fn, tp = confusion.flatten().tolist()
    return dict(eer_percent=eer_value, f1_bonafide_percent=float(f1_value),
        accuracy_percent=float(accuracy), eer_margin_threshold=threshold,
        f1_spoof_percent=100 * 2*tn/(2*tn+fp+fn),
        frame_count=len(labels), utterance_count=len(rows),
        confusion_true_rows_pred_columns_0_spoof_1_bonafide=confusion.tolist(),
        f1_rule='official F1Metric: binary-head argmax; positive class 1 = bonafide',
        eer_rule='official EERMetric: sort logit_1 - logit_0 descending, nearest FPR/FNR crossing')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Frozen reduced-role SAL prediction export')
    parser.add_argument('--split', choices=['train','train_detector','train_edit','train_review','dev','eval','lps'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--data-profile', choices=['reduced'], default='reduced')
    parser.add_argument('--checkpoint', required=True, help='public, latest or a reduced-trained checkpoint')
    parser.add_argument('--run-name')
    args = parser.parse_args(argv)
    args.split = canonical_split(args.split)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    return args


def evaluate_main(argv=None):
    global ACTIVE_OUT, ACTIVE_CONNECTION, ACTIVE_LOCK
    args = parse_arguments(argv)
    inference_device = configured_device()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    public_reference = args.checkpoint == 'public'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not public_reference and not args.run_name:
        args.run_name = f'sal_wavlm_{args.split}_{args.data_profile}_trained_{local_time():%Y%m%d_%H%M%S_%f}'
    run, out = begin_run(args.split, args.limit, args.run_name, args.data_profile)
    torch.set_num_threads(4)
    torch.manual_seed(666)
    np.random.seed(666)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset, resolution = get_dataset(args.split, args.data_profile)
    indices = list(range(len(dataset)))
    if args.limit:
        indices = np.linspace(0, len(dataset)-1, min(args.limit, len(dataset)), dtype=int).tolist()
        if len(indices) >= 4:
            # Exercise length extremes as well as evenly spaced utterances.
            lengths = [len(dataset.labels[u]) for u in dataset.utt_ids]
            indices[-2] = int(np.argmax(lengths))
            indices[-3] = int(np.argmin(lengths))
            indices = sorted(set(indices))
    run_lock = (out/'run.lock').open('a+b')
    ACTIVE_LOCK = run_lock
    run_lock.seek(0)
    msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
    ACTIVE_OUT = out
    connection = sqlite3.connect(out/'predictions.sqlite')
    ACTIVE_CONNECTION = connection
    connection.execute('CREATE TABLE IF NOT EXISTS predictions (uid TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, logits BLOB, labels BLOB, seconds REAL, runtime REAL, device TEXT)')
    config = dict(run=run, split=args.split, resolution_seconds=resolution, total_dataset=len(dataset),
        **profile_info(args.split, args.data_profile),
        planned_utterances=len(indices), smoke_test=bool(args.limit), precision='float32', batch_size=1,
        checkpoint=artifact_relative(checkpoint_path, ROOT, STORE, RESULT_ROOT),
        checkpoint_kind='released_reference' if public_reference else 'reduced_trained',
        checkpoint_sha256=checkpoint_digest() if public_reference else file_digest(checkpoint_path),
        official_source_commit='b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485',
        packages={x: importlib.metadata.version(x) for x in ['torch','torchaudio','s3prl','lightning','numpy','torchmetrics']},
        gpu=torch.cuda.get_device_name(0) if inference_device == 'cuda' else None,
        inference_device=inference_device, full_utterance=True,
        manifest_sha256=hashlib.sha256('\n'.join(dataset.utt_ids[i] for i in indices).encode()).hexdigest())
    write_json(out/'config.json', config)
    record_run('loading_model', **config)
    model, ckpt_metadata = load_model(resolution, checkpoint_path, public_reference=public_reference,
                                     device=inference_device)
    if not public_reference:
        config.update(validate_training_metadata(ckpt_metadata['project_training']))
    config['prediction_role'] = args.split
    write_json(out/'config.json', config)
    record_run('loading_model', **config)
    write_json(out/'checkpoint_metadata.json', ckpt_metadata)
    print('MODEL_LOADED', run, ckpt_metadata['strict_load'], flush=True)
    started = time.monotonic()
    completed_now = 0
    last_status = 0
    with torch.inference_mode():
        for ordinal in indices:
            uid = dataset.utt_ids[ordinal]
            info = sf.info(dataset.sample_list[ordinal])
            assert info.samplerate == 16000 and info.channels == 1, (uid, info)
            uid2, waveform, labels, nframes = dataset[ordinal]
            assert uid == uid2 and nframes > 0
            assert set(labels.tolist()) <= {0, 1}
            before = time.monotonic()
            logits, device = infer_waveform(model, waveform, inference_device)
            assert logits.shape == (nframes, 2), (uid, logits.shape, nframes)
            assert torch.isfinite(logits).all(), uid
            elapsed = time.monotonic() - before
            connection.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)',
                (uid, ordinal, logits.numpy().astype(np.float32).tobytes(),
                 labels.numpy().astype(np.uint8).tobytes(), info.duration, elapsed, device))
            completed_now += 1
            if completed_now % 25 == 0:
                connection.commit()
            now = time.monotonic()
            if now-last_status >= 30 or completed_now == 1:
                connection.commit()
                count = connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
                status = dict(status='running', run=run, completed=count, total=len(indices),
                    utterances_per_second=completed_now/max(now-started, 1e-9),
                    estimated_remaining_seconds=(len(indices)-count)*(now-started)/completed_now,
                    updated_local=time.strftime('%Y-%m-%dT%H:%M:%S'), last_uid=uid)
                write_json(out/'status.json', status)
                record_run('running', completed=count, last_uid=uid)
                print(json.dumps(status), flush=True)
                last_status = now
    connection.commit()
    assert connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == len(indices)
    model.cpu()
    del model
    if inference_device == 'cuda':
        torch.cuda.empty_cache()
    metrics = calculate_metrics(connection)
    metrics.update(config)
    seconds, runtime, fallbacks = connection.execute("SELECT SUM(seconds), SUM(runtime), SUM(device = 'cpu_oom_fallback') FROM predictions").fetchone()
    metrics.update(audio_seconds=seconds, inference_seconds=runtime, real_time_factor=runtime/seconds,
                   cpu_fallback_utterances=fallbacks, completed_local=time.strftime('%Y-%m-%dT%H:%M:%S'))
    write_json(out/'metrics.json', metrics)
    write_json(out/'status.json', dict(status='complete', run=run, completed=len(indices), total=len(indices)))
    record_run('complete', completed=len(indices), **metrics)
    print('RESULT', json.dumps(metrics), flush=True)
    print('RESULT_FILE', RESULT_ROOT/f'{run}.json', flush=True)
    connection.close()
    ACTIVE_CONNECTION = None
    run_lock.close()


def run(argv=None):
    """Shared entry point so all clickable launch files preserve error records."""
    global ACTIVE_CONNECTION, ACTIVE_LOCK
    try:
        evaluate_main(argv)
        return 0
    except KeyboardInterrupt as error:
        finish_failed_run('interrupted', error)
        print('Interrupted; this run has been recorded.', flush=True)
        return 130
    except Exception as error:
        traceback.print_exc()
        finish_failed_run('failed', error)
        return 1
    finally:
        if ACTIVE_CONNECTION is not None:
            ACTIVE_CONNECTION.close()
            ACTIVE_CONNECTION = None
        if ACTIVE_LOCK is not None:
            ACTIVE_LOCK.close()
            ACTIVE_LOCK = None


def require_no_other_sal_process():
    """Protect existing evaluations when the user clicks the new combined entry."""
    if os.name != 'nt':
        return
    import subprocess
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', command],
                                     creationflags=subprocess.CREATE_NO_WINDOW)
    processes = json.loads(output.decode('utf-8', errors='replace') or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    own = {os.getpid(), os.getppid()}
    names = ('sal.py', 'cnn.py', 'sal/length_ablation.py', 'sal/pipeline.py', 'cnn/baseline.py')
    for process in processes:
        if process['ProcessId'] not in own and any(name in (process.get('CommandLine') or '').replace('\\', '/') for name in names):
            raise RuntimeError('Another SAL evaluation is running; wait until it finishes before starting a new one.')


def click_main(argv=None):
    return experiment_main('reduced', argv)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_config(profile, settings=None):
    """Build official model/datamodule configs without loading weights or starting training."""
    from omegaconf import OmegaConf
    if profile != 'reduced':
        raise ValueError('Only reduced training is supported')
    settings = dict(CONFIG['sal'] if settings is None else settings)
    if not 0 < settings['mixup_ratio'] <= 1 or int(settings['batch_size']*settings['mixup_ratio']) < 1:
        raise ValueError('batch_size * mixup_ratio must be >= 1 to retain SAL cross-segment mixing')
    if settings['batch_size'] < 1 or settings['num_workers'] < 0 or not 1 <= settings['min_epochs'] <= settings['max_epochs']:
        raise ValueError('Invalid training size/epoch settings')
    if settings['learning_rate'] <= 0 or settings['weight_decay'] < 0 or settings['patience'] < 1 or settings['device'] not in ('cuda','cpu'):
        raise ValueError('Invalid training optimizer/device settings')
    audio, _ = dataset_paths('train_detector', profile)
    model = OmegaConf.load(ROOT/'vendor/SAL/configs/model/sal/wavlm_cf.yaml')
    model.net.ckpt = str(STORE/'models/wavlm_large.pt')
    model.optimizer.lr = settings['learning_rate']
    model.optimizer.weight_decay = settings['weight_decay']
    model.mixup2 = True
    model.mixup_ratio = settings['mixup_ratio']
    model.vis = False  # Plotting belongs to the project's external visualization directory.
    data = OmegaConf.load(ROOT/'vendor/SAL/configs/data/partialspoof_rawboost.yaml')
    data.root = str(audio.parent)
    data.batch_size = settings['batch_size']
    data.num_workers = settings['num_workers']
    return settings, model, data


def cross_recording_mixup(self, batch):
    """Repair vendor CSM self-pairing locally, with shared audio/label cuts.

    A cyclic shift of a random batch permutation supplies a different donor for
    every selected row, including when only one row is augmented. Cropped valid
    lengths are clamped to the available label frames before mixing.
    """
    utt_ids, inputs, labels, label_lengths = batch
    batch_size, frames = labels.shape
    if inputs.ndim != 2 or inputs.shape[0] != batch_size or frames < 1:
        raise ValueError('CSM requires matching batched waveform and frame labels')
    if inputs.shape[1] % frames:
        raise ValueError('CSM waveform length must align with whole label frames')
    if not 0 <= self.mixup_ratio <= 1:
        raise ValueError('CSM mixup ratio must be within [0, 1]')
    count = int(batch_size * self.mixup_ratio)
    if batch_size < 2 or count == 0:
        return batch
    samples_per_frame = inputs.shape[1] // frames
    order = torch.randperm(batch_size, device=inputs.device)
    donors = order.roll(-1)
    valid = label_lengths.clamp(min=0, max=frames)
    mixed_inputs, mixed_labels, mixed_lengths = inputs.clone(), labels.clone(), valid.clone()
    for recipient, donor in zip(order[:count].tolist(), donors[:count].tolist()):
        common = min(int(valid[recipient]), int(valid[donor]))
        if common < 2:
            continue  # No interior cut exists when either recording has <2 valid frames.
        fraction = float(torch.rand((), device=inputs.device)) * .6 + .2
        cut = max(1, min(int(fraction * common), common - 1))
        sample_cut = cut * samples_per_frame
        mixed_inputs[recipient, sample_cut:] = inputs[donor, sample_cut:]
        mixed_labels[recipient, cut:] = labels[donor, cut:]
        mixed_lengths[recipient] = valid[donor]
    return utt_ids, mixed_inputs, mixed_labels, mixed_lengths


def train_sal(profile, settings=None):
    """Train the official SAL model on PS train; PS dev alone selects the checkpoint."""
    import hydra
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    settings, model_config, data_config = training_config(profile, settings)
    require_no_other_sal_process()
    if settings['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if not (STORE/'models/wavlm_large.pt').is_file():
        raise FileNotFoundError('Pretrained WavLM backbone is missing')
    identity = training_identity()
    manifests = {}
    for split in ('train_detector','dev'):
        dataset, _ = get_dataset(split, profile)
        manifests[split] = dict(count=len(dataset), sha256=hashlib.sha256('\n'.join(dataset.utt_ids).encode()).hexdigest())
    stamp = local_time().strftime('%Y%m%d_%H%M%S_%f')
    name = f'sal_train_{profile}_{stamp}'
    out = RESULT_ROOT/name
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = STORE/'models'/name
    state = dict(run=name, status='initializing', data_profile=profile, config=settings, manifests=manifests,
                 started_local=local_time().isoformat(), weights_pretrained='WavLM backbone only; fresh SAL head',
                 public_sal_checkpoint_used_for_training=False, test_used_for_selection=False,
                 protocol='sal_reduced_v1', **identity)

    def update(status, **fields):
        state.update(status=status, updated_local=local_time().isoformat(), **fields)
        write_json(RESULT_ROOT/(name+'.json'), state)
        write_json(out/'status.json', state)

    class Progress(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            update('running', phase='train', epoch=trainer.current_epoch+1)

        def on_validation_epoch_start(self, trainer, pl_module):
            update('running', phase='validation', epoch=trainer.current_epoch+1)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint['project_training'] = dict(profile=profile, manifests=manifests, run=name, config=settings,
                protocol='sal_reduced_v1', **identity)

        def on_exception(self, trainer, pl_module, exception):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_dir/'interrupted.ckpt'))

    update('initializing')
    try:
        L.seed_everything(settings['seed'], workers=True)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = hydra.utils.instantiate(model_config)
        # Keep vendor sources intact; explicitly version this implementation repair.
        model._mixup_batch = MethodType(cross_recording_mixup, model)
        data = make_data_module(data_config)
        checkpoint = ModelCheckpoint(dirpath=str(checkpoint_dir), filename='best-{epoch:03d}',
                                     monitor='val/eer', mode='min', save_top_k=1, save_last=True)
        trainer = L.Trainer(accelerator='gpu' if settings['device']=='cuda' else 'cpu', devices=1,
                            precision='32-true', min_epochs=settings['min_epochs'], max_epochs=settings['max_epochs'],
                            deterministic=True, default_root_dir=str(out),
                            logger=CSVLogger(save_dir=str(out), name='logs'),
                            callbacks=[checkpoint, EarlyStopping(monitor='val/eer', mode='min', patience=settings['patience']), Progress()])
        write_json(out/'resolved_config.json', dict(settings=settings, model=OmegaConf.to_container(model_config, resolve=True),
                                                  data=OmegaConf.to_container(data_config, resolve=True),
                                                  csm_implementation=identity['csm_implementation'],
                                                  csm_patch='sal.cross_recording_mixup overrides vendor self-pairing; aligned cuts; two vendor mixup2 rounds'))
        trainer.fit(model, datamodule=data)
        if trainer.interrupted:
            raise KeyboardInterrupt('SAL training interrupted; recovery checkpoint retained')
        if not checkpoint.best_model_path:
            raise RuntimeError('No validation-selected SAL checkpoint was saved')
        weights = Path(checkpoint.best_model_path)
        update('complete', checkpoint=artifact_relative(weights, ROOT, STORE, RESULT_ROOT),
               checkpoint_sha256=file_digest(weights), best_dev_eer_percent=float(checkpoint.best_model_score),
               finished_local=local_time().isoformat())
        return RESULT_ROOT/(name+'.json')
    except BaseException as error:
        update('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=repr(error), finished_local=local_time().isoformat())
        raise


def make_data_module(config):
    """Keep the official datamodule behavior while mapping frozen reduced roles."""
    from omegaconf import OmegaConf
    from src.data.partialspoof import PartialSpoofDataModule, PartialSpoofDataset

    class ReducedDataModule(PartialSpoofDataModule):
        def get_dataset(self, split, resolution, pad_mode):
            role = canonical_split(split)
            audio_root, labels = dataset_paths(role, 'reduced')
            dataset = PartialSpoofDataset(root=str(audio_root), split=label_split(role),
                sample_rate=self.sample_rate, resolution=resolution, input_query='*.wav',
                label_root=str(labels), max_label_len=self.max_label_len, pad_mode=pad_mode,
                add_label=self.add_label, debug_mode=self.debug_mode, max_samples=self.max_samples,
                use_augmentation=self.use_augmentation and role == 'train_detector' and pad_mode == 'train',
                augmentation_algo=self.augmentation_algo, augmentation_prob=self.augmentation_prob)
            if dataset.utt_ids != read_manifest(role)['utterance_ids'] or set(dataset.labels) != set(dataset.utt_ids):
                raise ValueError(f'Reduced datamodule manifest mismatch: {role}')
            return dataset

    settings = OmegaConf.to_container(config, resolve=True)
    settings.pop('_target_')
    return ReducedDataModule(**settings)


def experiment_main(profile='reduced', argv=None):
    if profile != 'reduced':
        raise ValueError('Only reduced is supported')
    parser = argparse.ArgumentParser(description='SAL reduced experiment: explicit training; default check only')
    parser.add_argument('--mode', choices=('check','train','predict','validate','test','all','smoke','ps','lps','dev','full'), default=RUN_MODE)
    parser.add_argument('--split', choices=('train','train_detector','train_edit','train_review','dev','eval','lps'), default='eval')
    parser.add_argument('--checkpoint', help='public, latest, or a reduced-trained checkpoint path')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--run-name')
    parser.add_argument('--data-profile', choices=('reduced',), default='reduced')
    args = parser.parse_args(argv)
    args.split = canonical_split(args.split)
    if args.mode in ('test','all') and args.split not in ('eval','lps'):
        parser.error('test/all accept eval/lps only; use predict for correction-training features')
    if args.mode == 'check':
        settings, model, data = training_config(profile)
        manifests = {role:read_manifest(role)['manifest_sha256'] for role in ('train_detector','dev')}
        print(json.dumps(dict(profile=profile, settings=settings, manifests=manifests,
            model_target=model['_target_'], training_started=False), ensure_ascii=False, indent=2))
        return 0
    if args.mode in ('train','all'):
        if args.checkpoint:
            parser.error('Fresh SAL training uses WavLM backbone only; do not pass a SAL checkpoint')
        record = train_sal(profile)
        if args.mode == 'train':
            print('TRAIN_RESULT_FILE', record)
            return 0
        selected = json.loads(record.read_text(encoding='utf-8'))['checkpoint']
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if not args.checkpoint:
            parser.error('Specify --checkpoint public, latest, or a reduced-trained checkpoint path')
        selected = args.checkpoint
    split = {'validate':'dev','dev':'dev','smoke':'dev','ps':'eval','lps':'lps'}.get(args.mode,args.split)
    splits = ('dev','eval','lps') if args.mode == 'full' else (split,)
    require_no_other_sal_process()
    for split in splits:
        command = ['--split',split,'--data-profile','reduced','--checkpoint',selected,
                   '--limit',str(32 if args.mode == 'smoke' else args.limit)]
        if args.run_name:
            if len(splits)>1:
                parser.error('--run-name is only supported for a single split')
            command.extend(['--run-name',args.run_name])
        result = run(command)
        if result:
            return result
    return 0


if __name__ == '__main__':
    sys.exit(click_main())



