"""点击运行：默认在固定 10% PS 测试集上评估公开 SAL 权重。

ps/lps/dev 默认 fixed10；full 显式运行完整 dev/eval/lps。
FP32 evaluation with a fresh timestamped record for every manual run.
SQLite stores all frame logits and labels, not just aggregate scores.
"""
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
import sqlite3
import sys
import time
import traceback
import msvcrt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_MODE = "ps"  # ps/lps/dev 默认 fixed10；full 显式运行三套完整数据
from project_config import DATA_ROOT as STORE, RESULT_ROOT, LOG_ROOT, checkpoint_digest, portable_checkpoint_path
from tools.data_tools import dataset_paths, profile_info, read_manifest
ACTIVE_OUT = None
ACTIVE_CONNECTION = None
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


def begin_run(split, limit, run_name=None, data_profile='full'):
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


def prepare_lps_labels():
    path = STORE / 'LlamaPartialSpoof/segment_labels/cf_seglab_0.02.npy'
    if path.exists():
        return path
    labels = {}
    # Same flooring and 0=spoof rule as official process_label().
    for line in (STORE / 'downloads/LlamaPartialSpoof/label_R01TTS.0.a.txt').read_text().splitlines():
        uid, duration, _, *segments = line.split()
        if not any(x in uid for x in ('clean', 'full', 'cf')):
            raise ValueError('Unexpected LPS subset: ' + uid)
        label = np.ones(int(float(duration) / 0.02), dtype=np.int64)
        for segment in segments:
            start, end, kind = segment.split('-')
            if kind == 'spoof':
                label[int(float(start) / 0.02):int(float(end) / 0.02)] = 0
        labels[uid] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels, allow_pickle=True)
    return path


def get_dataset(split, data_profile='full'):
    audio_root, label_root = dataset_paths(split, data_profile)
    if data_profile == 'fixed10':
        from tools.data_tools import validate_fixed10
        validate_fixed10(split)
    if split == 'lps':
        from src.data.llama_partialspoof import LlamaPartialSpoofDataset
        if data_profile == 'full':
            prepare_lps_labels()
            root = STORE / 'LlamaPartialSpoof'
            marker = root / 'R01TTS.0.a.tgz.extracted.json'
            assert marker.exists(), 'LPS extraction incomplete'
        dataset = LlamaPartialSpoofDataset(root=str(audio_root), split='cf', sample_rate=16000,
            resolution=0.02, input_query='*.wav', label_root=str(label_root),
            max_label_len=200, pad_mode='test', add_label=False)
        resolution = 0.02
    else:
        from src.data.partialspoof import PartialSpoofDataset
        if data_profile == 'full':
            root = STORE / 'PartialSpoof/database'
            marker = root.parent / f'database_{split}.tar.gz.extracted.json'
            assert marker.exists(), 'PS extraction incomplete'
        dataset = PartialSpoofDataset(root=str(audio_root), split=split, sample_rate=16000,
            resolution=0.16, input_query='*.wav', label_root=str(label_root),
            max_label_len=25, pad_mode='test', add_label=False, use_augmentation=False)
        resolution = 0.16
    assert len(dataset.utt_ids) == len(set(dataset.utt_ids)), 'Duplicate utterance IDs'
    assert set(dataset.utt_ids) == set(dataset.labels), 'Audio / label ID coverage mismatch'
    if data_profile == 'fixed10':
        assert dataset.utt_ids == read_manifest(split)['utterance_ids'], 'Fixed subset manifest mismatch'
    return dataset, resolution


def load_model(resolution, checkpoint_path=None):
    from src.models.net.model import SSL_SeqModel_8Labels_2Loss
    model = SSL_SeqModel_8Labels_2Loss(ssl_encoder='large', mode='s3prl_weighted',
        ckpt=str(STORE/'models/wavlm_large.pt'), hid_dim=1024, seq_model='cf',
        resolution_train=0.16, resolution_test=resolution, pool='avg')
    checkpoint = torch.load(str(checkpoint_path or STORE/'models/SAL_WavLM.ckpt'), map_location='cpu', mmap=True)
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
    return model.eval().cuda(), metadata


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ['--split', 'dev', '--limit', '32']
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'eval', 'lps'], required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--data-profile', choices=['fixed10', 'full'], default='fixed10')
    parser.add_argument('--checkpoint', help='Optional trained checkpoint, relative to code; default is released SAL')
    parser.add_argument('--run-name', help='Optional unique run name; an existing name is rejected')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    return args


def main(argv=None):
    global ACTIVE_OUT, ACTIVE_CONNECTION
    args = parse_arguments(argv)
    checkpoint_path = (ROOT/args.checkpoint).resolve() if args.checkpoint else STORE/'models/SAL_WavLM.ckpt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.checkpoint and not args.run_name:
        args.run_name = f'sal_custom_{args.split}_{args.data_profile}_{local_time():%Y%m%d_%H%M%S_%f}'
    run, out = begin_run(args.split, args.limit, args.run_name, args.data_profile)
    torch.set_num_threads(4)
    torch.manual_seed(666)
    np.random.seed(666)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), 'CUDA required for this run'
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
    run_lock.seek(0)
    msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
    ACTIVE_OUT = out
    connection = sqlite3.connect(out/'predictions.sqlite')
    ACTIVE_CONNECTION = connection
    connection.execute('CREATE TABLE IF NOT EXISTS predictions (uid TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, logits BLOB, labels BLOB, seconds REAL, runtime REAL, device TEXT)')
    config = dict(run=run, split=args.split, resolution_seconds=resolution, total_dataset=len(dataset),
        **profile_info(args.split, args.data_profile),
        planned_utterances=len(indices), smoke_test=bool(args.limit), precision='float32', batch_size=1,
        checkpoint=os.path.relpath(checkpoint_path, ROOT).replace('\\', '/'),
        checkpoint_kind='trained_or_custom' if args.checkpoint else 'released',
        checkpoint_sha256=file_digest(checkpoint_path) if args.checkpoint else checkpoint_digest(),
        official_source_commit='b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485',
        packages={x: importlib.metadata.version(x) for x in ['torch','torchaudio','s3prl','lightning','numpy','torchmetrics']},
        gpu=torch.cuda.get_device_name(0), full_utterance=True,
        manifest_sha256=hashlib.sha256('\n'.join(dataset.utt_ids[i] for i in indices).encode()).hexdigest())
    write_json(out/'config.json', config)
    record_run('loading_model', **config)
    model, ckpt_metadata = load_model(resolution, checkpoint_path)
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
            device = 'cuda'
            before = time.monotonic()
            fallback = False
            try:
                _, logits = model(waveform.unsqueeze(0).cuda())
                logits = logits[0].cpu()
            except torch.cuda.OutOfMemoryError:
                fallback = True
            if fallback:
                # Keep the entire utterance and FP32 computation for long recordings.
                model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                device = 'cpu_oom_fallback'
                _, logits = model(waveform.unsqueeze(0))
                logits = logits[0].cpu()
                model.cuda()
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
                    utterances_per_second=completed_now/(now-started),
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
    torch.cuda.empty_cache()
    metrics = calculate_metrics(connection)
    metrics.update(config)
    seconds, runtime, fallbacks = connection.execute("SELECT SUM(seconds), SUM(runtime), SUM(device != 'cuda') FROM predictions").fetchone()
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
    try:
        main(argv)
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
    names = ('sal/pipeline.py', 'sal/full_data.py', 'sal/fixed10.py', 'cnn/baseline.py', 'cnn/grid_search.py')
    for process in processes:
        if process['ProcessId'] not in own and any(name in (process.get('CommandLine') or '').replace('\\', '/') for name in names):
            raise RuntimeError('Another SAL evaluation is running; wait until it finishes before starting a new one.')


def click_main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--help' in argv or '-h' in argv:
        return run(argv)
    mode = RUN_MODE
    if argv[:1] == ['--mode']:
        if len(argv) != 2:
            raise ValueError('Use --mode smoke/ps/lps/dev/full')
        mode, argv = argv[1], []
    require_no_other_sal_process()
    if argv:
        return run(argv)  # Preserve --split / --limit CLI compatibility.
    if mode == 'full':
        from tools.reporting import summarize, verify_run
        for split in ('dev', 'eval', 'lps'):
            result = run(['--split', split, '--data-profile', 'full'])
            if result:
                return result
            verify_run(ACTIVE_OUT)
        summarize(data_profile="full")
        return 0
    selections = {'smoke': ['--split', 'dev', '--limit', '32'],
                  'ps': ['--split', 'eval'], 'lps': ['--split', 'lps'], 'dev': ['--split', 'dev']}
    if mode not in selections:
        raise ValueError('RUN_MODE must be smoke, ps, lps, dev, or full')
    return run(selections[mode])


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_config(profile, settings=None):
    """Build official model/datamodule configs without loading weights or starting training."""
    from omegaconf import OmegaConf
    settings = settings or json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    if not 0 < settings['mixup_ratio'] <= 1 or int(settings['batch_size']*settings['mixup_ratio']) < 1:
        raise ValueError('batch_size * mixup_ratio must be >= 1 to retain SAL cross-segment mixing')
    if settings['batch_size'] < 1 or settings['num_workers'] < 0 or not 1 <= settings['min_epochs'] <= settings['max_epochs']:
        raise ValueError('Invalid training size/epoch settings')
    if settings['learning_rate'] <= 0 or settings['weight_decay'] < 0 or settings['patience'] < 1 or settings['device'] not in ('cuda','cpu'):
        raise ValueError('Invalid training optimizer/device settings')
    audio, _ = dataset_paths('train', profile)
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
    manifests = {}
    for split in ('train','dev'):
        dataset, _ = get_dataset(split, profile)
        manifests[split] = dict(count=len(dataset), sha256=hashlib.sha256('\n'.join(dataset.utt_ids).encode()).hexdigest())
    stamp = local_time().strftime('%Y%m%d_%H%M%S_%f')
    name = f'sal_train_{profile}_{stamp}'
    out = RESULT_ROOT/name
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = STORE/'models'/name
    state = dict(run=name, status='initializing', data_profile=profile, config=settings, manifests=manifests,
                 started_local=local_time().isoformat(), weights_pretrained='WavLM backbone only; fresh SAL head',
                 public_sal_checkpoint_used_for_training=False, test_used_for_selection=False)

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
            checkpoint['project_training'] = dict(profile=profile, manifests=manifests, run=name, config=settings)

    update('initializing')
    try:
        L.seed_everything(settings['seed'], workers=True)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = hydra.utils.instantiate(model_config)
        data = hydra.utils.instantiate(data_config)
        checkpoint = ModelCheckpoint(dirpath=str(checkpoint_dir), filename='best-{epoch:03d}',
                                     monitor='val/eer', mode='min', save_top_k=1, save_last=True)
        trainer = L.Trainer(accelerator='gpu' if settings['device']=='cuda' else 'cpu', devices=1,
                            precision='32-true', min_epochs=settings['min_epochs'], max_epochs=settings['max_epochs'],
                            deterministic=True, default_root_dir=str(out),
                            logger=CSVLogger(save_dir=str(out), name='logs'),
                            callbacks=[checkpoint, EarlyStopping(monitor='val/eer', mode='min', patience=settings['patience']), Progress()])
        write_json(out/'resolved_config.json', dict(settings=settings, model=OmegaConf.to_container(model_config, resolve=True),
                                                  data=OmegaConf.to_container(data_config, resolve=True)))
        trainer.fit(model, datamodule=data)
        if not checkpoint.best_model_path:
            raise RuntimeError('No validation-selected SAL checkpoint was saved')
        weights = Path(checkpoint.best_model_path)
        update('complete', checkpoint=os.path.relpath(weights,ROOT).replace('\\','/'),
               checkpoint_sha256=file_digest(weights), best_dev_eer_percent=float(checkpoint.best_model_score),
               finished_local=local_time().isoformat())
        return RESULT_ROOT/(name+'.json')
    except BaseException as error:
        update('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=repr(error), finished_local=local_time().isoformat())
        raise


def experiment_main(profile, argv=None):
    """Profile-specific entry used by full_data.py and fixed10.py."""
    parser = argparse.ArgumentParser(description=f'SAL {profile}: train / validate / test. No training on import.')
    parser.add_argument('--mode', choices=('check','train','validate','test','all'), default='check')
    parser.add_argument('--split', choices=('eval','lps'), default='eval', help='Test set; training is always PS')
    parser.add_argument('--checkpoint', help='public, latest, or path relative to code; required for validate/test')
    args = parser.parse_args(argv)
    if args.mode == 'check':
        settings, model, data = training_config(profile)
        print(json.dumps(dict(profile=profile, data_root=str(data.root), settings=settings,
                              model_target=model['_target_'], training_started=False), ensure_ascii=False, indent=2))
        return 0
    if args.mode in ('train','all'):
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
            parser.error('Specify --checkpoint public, latest, or a trained checkpoint path')
        selected = args.checkpoint
        if selected == 'latest':
            completed = []
            for file in RESULT_ROOT.glob(f'sal_train_{profile}_*.json'):
                row = json.loads(file.read_text(encoding='utf-8'))
                if row.get('status') == 'complete':
                    completed.append((file.name,row))
            if not completed:
                raise FileNotFoundError(f'No completed SAL {profile} training run; public inference is a separate option')
            row = max(completed, key=lambda x:x[0])[1]
            selected = row['checkpoint']
            if file_digest((ROOT/selected).resolve()) != row['checkpoint_sha256']:
                raise ValueError('Trained SAL checkpoint checksum differs')
    split = 'dev' if args.mode == 'validate' else args.split
    command = ['--split',split,'--data-profile',profile]
    if selected != 'public':
        command.extend(['--checkpoint',selected])
    require_no_other_sal_process()
    return run(command)


if __name__ == '__main__':
    sys.exit(click_main())



