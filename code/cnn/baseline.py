# 1. Model and signal features
"""New baseline (not SAL): Log-Mel -> time-preserving CNN -> BiGRU.

Native bins are [k*0.02, (k+1)*0.02), with STFT centers at (k+0.5)*0.02.
Internal targets: bonafide=0, spoof=1. SAL metrics use the opposite encoding.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    hop_length: int = 320
    n_fft: int = 512
    win_length: int = 400
    n_mels: int = 80
    channels: tuple = (16, 32, 64)
    hidden_size: int = 128
    gru_layers: int = 2
    dropout: float = 0.2

    def to_dict(self):
        return asdict(self)


class LogMel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if (c.sample_rate, c.hop_length) != (16000, 320):
            raise ValueError('The comparison protocol requires 16 kHz / 20 ms bins')
        if c.n_fft < c.hop_length or (c.n_fft - c.hop_length) % 2:
            raise ValueError('n_fft minus hop_length must be nonnegative and even')
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=c.sample_rate, n_fft=c.n_fft, win_length=c.win_length,
            hop_length=c.hop_length, n_mels=c.n_mels, center=False,
            power=2.0, norm='slaney', mel_scale='htk',
        )

    def forward(self, waveform):
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError('Expected a nonempty mono waveform')
        if not torch.isfinite(waveform).all():
            raise ValueError('Waveform contains non-finite values')
        c = self.config
        frames = math.ceil(waveform.numel() / c.hop_length)
        # Each centered FFT window covers a bin midpoint, including the first bin.
        side = (c.n_fft - c.hop_length) // 2
        waveform = F.pad(waveform, (side, side + frames*c.hop_length-waveform.numel()))
        features = self.mel(waveform).clamp_min(1e-10).log().transpose(0, 1)
        if features.shape != (frames, c.n_mels):
            raise RuntimeError('Unexpected frontend frame alignment')
        return features.unsqueeze(0)  # [1, T, F]


class CNNBiGRU(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.input_norm = nn.LayerNorm(c.n_mels)
        blocks, channels, freq = [], 1, c.n_mels
        for out_channels in c.channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(), nn.MaxPool2d((1, 2)),
            ))
            channels, freq = out_channels, freq // 2
        if freq < 1:
            raise ValueError('Too many CNN frequency-pooling stages')
        self.blocks = nn.ModuleList(blocks)
        self.gru = nn.GRU(channels*freq, c.hidden_size, num_layers=c.gru_layers,
                          bidirectional=True, batch_first=True,
                          dropout=c.dropout if c.gru_layers > 1 else 0)
        self.dropout = nn.Dropout(c.dropout)
        self.classifier = nn.Linear(2*c.hidden_size, 1)

    def forward(self, features, lengths, return_features=False):
        # [B,1,T,F] -> [B,C,T,F'] -> [B,T,C*F'] -> [B,T,2H] -> [B,T]
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError('Expected [B, 1, T, F] features')
        lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
        if len(lengths) != len(features) or (lengths <= 0).any() or (lengths > features.shape[2]).any():
            raise ValueError('Invalid unpadded lengths')
        mask = torch.arange(features.shape[2], device=features.device)[None, :] < lengths.to(features.device)[:, None]
        x = self.input_norm(features) * mask[:, None, :, None]
        for block in self.blocks:
            x = block(x) * mask[:, None, :, None]
        x = x.permute(0, 2, 1, 3).flatten(2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.gru(packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=features.shape[2])
        logits = self.classifier(self.dropout(context)).squeeze(-1) * mask
        return (logits, context) if return_features else logits


def pool_logits(native_logits, lengths, resolution):
    """Mean of 8 native logits for PS 160 ms; identity for LPS 20 ms."""
    factor = round(resolution / 0.02)
    if factor < 1 or not math.isclose(factor*0.02, resolution, abs_tol=1e-9):
        raise ValueError('Resolution must be a multiple of 20 ms')
    lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
    if (lengths % factor).any():
        raise ValueError('Native lengths must match complete evaluation bins')
    if native_logits.ndim != 2 or native_logits.shape[1] % factor:
        raise ValueError('Padded time dimension must be divisible by pooling factor')
    return native_logits.reshape(len(native_logits), -1, factor).mean(-1), lengths // factor


def masked_bce(logits, spoof_targets, lengths):
    if logits.shape != spoof_targets.shape:
        raise ValueError('Prediction and target shapes differ')
    mask = torch.arange(logits.shape[1], device=logits.device)[None] < lengths.to(logits.device)[:, None]
    # Index first: arbitrary padding values (even NaN) must not enter the loss.
    return F.binary_cross_entropy_with_logits(logits[mask], spoof_targets[mask])


def sal_logits(spoof_logits):
    """[spoof score, bonafide score]=[s,0]; SAL EER margin is therefore -s."""
    return torch.stack((spoof_logits, torch.zeros_like(spoof_logits)), dim=-1)


def probabilities_to_intervals(probabilities, resolution, duration, threshold=0.5):
    if not 0 <= threshold <= 1 or resolution <= 0 or duration < 0:
        raise ValueError('Invalid threshold, resolution or duration')
    probabilities = torch.as_tensor(probabilities).detach().cpu().flatten()
    if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite and between 0 and 1')
    flags = (probabilities >= threshold).tolist()
    intervals, start = [], None
    for index, fake in enumerate(flags + [False]):
        if fake and start is None:
            start = index
        elif not fake and start is not None:
            begin, end = start*resolution, min(index*resolution, duration)
            if end > begin:
                intervals.append([round(begin, 8), round(end, 8)])
            start = None
    return intervals


def intervals_to_spoof_labels(intervals, frames, resolution, duration):
    """For new annotated audio only: half-open intervals, frame-center rule.

    PS/LPS evaluation instead uses official labels unchanged.
    """
    centers = (torch.arange(frames, dtype=torch.float64) + 0.5)*resolution
    labels = torch.zeros(frames)
    for begin, end in intervals:
        if not 0 <= begin < end <= duration:
            raise ValueError('Intervals must satisfy 0 <= start < end <= duration')
        labels[(centers >= begin) & (centers < end)] = 1
    return labels


# 2. Dataset and batches
"""Dataset adapter using the SAME SAL data loaders and label files.

Official train/dev/eval splits stay intact; random crops occur only in train.
"""
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Allow VS Code's Run Python File from any terminal directory.
import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from project_config import DATA_ROOT


def check_training_files():
    base = DATA_ROOT / 'PartialSpoof'
    required = [base/'database/train', base/'database/dev',
                base/'database_train.tar.gz.extracted.json',
                base/'database_dev.tar.gz.extracted.json']
    required += [base/'database/segment_labels'/f'{s}_seglab_0.16.npy' for s in ('train', 'dev', 'eval')]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing official PS training resources; never substitute dev/eval for train:\n' + '\n'.join(missing))


def audit_splits(data_profile='full'):
    root = DATA_ROOT/'PartialSpoof/database/segment_labels'
    ids = {split: set(np.load(root/f'{split}_seglab_0.16.npy', allow_pickle=True).item())
           for split in ('train', 'dev', 'eval')}
    for a, b in (('train', 'dev'), ('train', 'eval'), ('dev', 'eval')):
        if ids[a] & ids[b]:
            raise ValueError(f'Utterance ID leakage between {a} and {b}')
    from tools.data_tools import read_manifest
    selected = {s: set(read_manifest(s)['utterance_ids']) if data_profile == 'fixed10' else v for s, v in ids.items()}
    if any(not selected[s] <= ids[s] for s in ids):
        raise ValueError('Subset is not contained in the official split')
    return dict(official_split_utterances={s: len(v) for s, v in ids.items()},
                active_data_profile=data_profile, active_split_utterances={s: len(v) for s, v in selected.items()},
                utterance_ids_disjoint=True,
                source_recording_overlap='Not independently verified: source provenance map is not provided',
                split_policy='Official PS splits unchanged; crop only after splitting')


def manifest_hash(utterance_ids):
    return hashlib.sha256('\n'.join(utterance_ids).encode()).hexdigest()


class BaselineDataset(Dataset):
    def __init__(self, split, model_config, crop_seconds=None, data_profile='fixed10'):
        # Importing the evaluator does not load SAL weights or launch evaluation.
        from sal.pipeline import get_dataset
        self.official, self.resolution = get_dataset(split, data_profile)
        self.data_profile = data_profile
        self.split = split
        self.frontend = LogMel(model_config)
        self.crop_frames = None
        if crop_seconds is not None:
            if split != 'train' or crop_seconds <= 0:
                raise ValueError('Cropping is allowed only on the training split')
            frames = round(crop_seconds/self.resolution)
            if abs(frames*self.resolution-crop_seconds) > 1e-8:
                raise ValueError('Crop must contain a whole number of label frames')
            self.crop_frames = frames
        self.utt_ids = self.official.utt_ids
        self.manifest_sha256 = manifest_hash(self.utt_ids)

    def __len__(self):
        return len(self.official)

    def __getitem__(self, index):
        path = self.official.sample_list[index]
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f'Expected 16 kHz mono audio: {path}')
        uid, waveform, original_labels, nframes = self.official[index]
        if nframes <= 0 or not torch.isin(original_labels, torch.tensor([0., 1.])).all():
            raise ValueError(f'Invalid official labels: {uid}')
        start = 0
        if self.crop_frames is not None and nframes > self.crop_frames:
            start = random.randint(0, nframes-self.crop_frames)
            original_labels = original_labels[start:start+self.crop_frames]
            samples = round(self.resolution*16000)
            waveform = waveform[start*samples:(start+self.crop_frames)*samples]
        started = time.perf_counter()
        features = self.frontend(waveform)
        frontend_seconds = time.perf_counter()-started
        factor = round(self.resolution/0.02)
        if features.shape[1] != len(original_labels)*factor:
            raise ValueError(f'Feature/label length mismatch: {uid}')
        return dict(uid=uid, features=features, labels=1-original_labels.float(),
                    duration=info.duration, crop_start_seconds=start*self.resolution,
                    frontend_seconds=frontend_seconds)


def collate(batch):
    lengths = torch.tensor([x['features'].shape[1] for x in batch], dtype=torch.long)
    label_lengths = torch.tensor([len(x['labels']) for x in batch], dtype=torch.long)
    features = torch.stack([F.pad(x['features'], (0, 0, 0, int(lengths.max())-x['features'].shape[1])) for x in batch])
    labels = torch.stack([F.pad(x['labels'], (0, int(label_lengths.max())-len(x['labels']))) for x in batch])
    return dict(features=features, labels=labels, lengths=lengths,
                label_lengths=label_lengths, uids=[x['uid'] for x in batch],
                durations=[x['duration'] for x in batch],
                frontend_seconds=[x.get('frontend_seconds', 0.) for x in batch])


# 3. Training, validation and evaluation
"""直接点击本文件运行 CNN-BiGRU。默认仅训练；RUN_MODE=full 时训练→PS→LPS→汇总。

默认使用固定 10% 训练/验证/测试数据。导入模块不启动实验。
只执行某一步时修改下方 RUN_MODE；不需要额外的 run 包装文件。
"""
import gc
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader


# Allow VS Code's Run Python File from any terminal directory.
import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from project_config import CODE_ROOT, DATA_ROOT, RESULT_ROOT
from tools.data_tools import profile_info, PROFILES

PROTOCOL = 'cnn_bigru_ps160_lps20_v1'
RUN_MODE = 'train'  # train / full / ps / lps / check / prepare / predict
AUDIO_FILE = ''  # predict 模式：相对路径以 code 为基准，例如 ../data/example.wav
MODULE_ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configuration_signature(config, exclude_search=False):
    keys = ('seed', 'epochs', 'batch_size', 'crop_seconds', 'weight_decay', 'patience',
            'training_data_profile', 'model', 'learning_rate')
    value = {key: copy.deepcopy(config.get(key)) for key in keys}
    if exclude_search:
        value.pop('learning_rate', None)
        value['model'].pop('dropout', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_grid_selection(config):
    """Only a completed, dev-selected search on the current fixed manifests is eligible."""
    pointer = RESULT_ROOT/'cnn_grid_best.json'
    if not pointer.exists():
        return None
    selected = json.loads(pointer.read_text(encoding='utf-8'))
    search = json.loads((RESULT_ROOT/selected['search_file']).read_text(encoding='utf-8'))
    if search.get('status') != 'complete' or not search.get('trials') or any(t.get('status') != 'complete' for t in search['trials']):
        raise ValueError('Grid selection is not from a completed search')
    winner = min(search['trials'], key=lambda t: (t['dev_eer_percent'], t['index']))
    if selected['winner'] != winner:
        raise ValueError('Grid selection differs from the validation-set winner')
    if search['base_signature'] != configuration_signature(config, exclude_search=True):
        raise ValueError('Grid search was run with different fixed settings. Rerun search or disable use_grid_search_best.')
    from tools.data_tools import read_manifest
    for split in ('train', 'dev'):
        if winner[split+'_manifest_sha256'] != read_manifest(split)['manifest_sha256']:
            raise ValueError('Grid search used a different fixed subset')
    checkpoint = (CODE_ROOT/winner['checkpoint']).resolve()
    if sha256(checkpoint) != winner['checkpoint_sha256']:
        raise ValueError('Grid-selected checkpoint checksum differs')
    return selected


def load_config(use_grid=True):
    config = json.loads((MODULE_ROOT/'config.json').read_text(encoding='utf-8'))
    config = validate_config(config)
    selected = read_grid_selection(config) if use_grid and config.get('use_grid_search_best', True) else None
    if selected:
        winner = selected['winner']
        config['learning_rate'] = winner['config']['learning_rate']
        config['model']['dropout'] = winner['config']['model']['dropout']
        config['hyperparameter_source'] = selected['search_file']
        config['grid_selected_checkpoint'] = winner['checkpoint']
    else:
        config['hyperparameter_source'] = 'cnn/config.json'
    return config


def validate_config(config):
    config = copy.deepcopy(config)
    config.setdefault('training_data_profile', 'full')  # Historical configs describe full-data training.
    config.setdefault('evaluation_data_profile', 'full')
    for key in ('training_data_profile', 'evaluation_data_profile'):
        if config[key] not in PROFILES:
            raise ValueError(f'Invalid {key}')
    for key in ('epochs', 'batch_size', 'patience', 'cpu_threads'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['num_workers'] < 0 or config['learning_rate'] <= 0 or config['weight_decay'] < 0:
        raise ValueError('Invalid optimizer or worker settings')
    if config['device'] not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    ModelConfig(**config['model'])
    return config


def configure_runtime(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def other_experiment_processes():
    """Find SAL or CNN experiment processes, excluding this venv process pair."""
    if os.name == 'nt':
        command = (
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        output = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', command],
                                         creationflags=subprocess.CREATE_NO_WINDOW)
        processes = json.loads(output.decode('utf-8', errors='replace') or '[]')
        if isinstance(processes, dict):
            processes = [processes]
        names = ('sal/pipeline.py', 'sal/full_data.py', 'sal/fixed10.py', 'cnn/baseline.py', 'cnn/grid_search.py')
        return [p for p in processes if p['ProcessId'] not in (os.getpid(), os.getppid())
                and any(name in (p.get('CommandLine') or '').replace('\\', '/') for name in names)]
    return []


def require_device_idle(device):
    """Do not accidentally share the GPU with another model experiment."""
    if device == 'cpu':
        return
    if other_experiment_processes():
        raise RuntimeError('Another SAL/CNN experiment is running. Wait until it finishes.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose device="cpu" explicitly if intended')


class RunRecord:
    def __init__(self, purpose, progress=None, **fields):
        self.progress = progress
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        self.name = f'cnn_bigru_{purpose}_{stamp}'
        self.directory = RESULT_ROOT/self.name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = RESULT_ROOT/(self.name+'.json')
        self.started = time.monotonic()
        self.info = dict(run=self.name, model='LogMel-CNN-BiGRU', protocol=PROTOCOL,
                         started_local=now(), **fields)
        self.update('initializing')

    def update(self, status, **fields):
        self.info.update(status=status, updated_local=now(),
                         elapsed_wall_seconds=round(time.monotonic()-self.started, 3), **fields)
        if status in ('complete', 'failed', 'interrupted'):
            self.info['finished_local'] = now()
        write_json(self.path, self.info)
        write_json(self.directory/'status.json', self.info)
        if self.progress is not None:
            self.progress(self.path, dict(self.info))


def official_metrics(spoof_scores, spoof_targets):
    """Use vendor SAL implementations unchanged; do not optimize F1 on test."""
    vendor = str(CODE_ROOT/'vendor/SAL')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from src.evaluator.evaluator import EERMetric, F1Metric
    scores = torch.as_tensor(spoof_scores, dtype=torch.float32).detach().cpu().flatten()
    targets = torch.as_tensor(spoof_targets, dtype=torch.float32).detach().cpu().flatten()
    if not len(scores) or scores.shape != targets.shape or not torch.isfinite(scores).all():
        raise ValueError('Empty, invalid or mismatched metric inputs')
    if not torch.isin(targets, torch.tensor([0., 1.])).all():
        raise ValueError('Metric targets must be binary')
    labels = 1-targets
    logits = sal_logits(scores)
    eer, f1 = EERMetric(), F1Metric()
    eer.update(logits[:, 1]-logits[:, 0], labels)
    f1.update(logits, labels)
    eer_value, threshold = eer.compute()
    accuracy, f1_value = f1.compute()
    predicted = logits.argmax(1)
    matrix = torch.bincount(labels.long()*2+predicted, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = matrix.flatten().tolist()
    return dict(eer_percent=float(eer_value), f1_bonafide_percent=float(f1_value),
                accuracy_percent=float(accuracy), eer_margin_threshold=float(threshold),
                f1_spoof_percent=100*2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.,
                frame_count=len(scores),
                confusion_true_rows_pred_columns_0_spoof_1_bonafide=matrix.tolist(),
                f1_rule='official F1Metric: binary-head argmax; positive class 1 = bonafide',
                eer_rule='official EERMetric: sort logit_1 - logit_0 descending, nearest FPR/FNR crossing',
                single_class_eer_sentinel=bool(targets.min() == targets.max()))


def choose_interval_threshold(scores, targets):
    """Dev only, maximize spoof-positive F1 on a fixed 0.05..0.95 grid.

    This threshold affects intervals ONLY, never SAL-compatible main F1.
    """
    probabilities, truth = scores.sigmoid().cpu(), targets.bool().cpu()
    best = None
    for step in range(1, 20):
        threshold = step/20
        predicted = probabilities >= threshold
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        score = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        key = (score, -abs(threshold-0.5), -threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    return best[1]


def infer_batch(model, batch, resolution, device):
    def forward(target_device):
        native = model(batch['features'].to(target_device), batch['lengths'])
        pooled, lengths = pool_logits(native, batch['lengths'], resolution)
        if not torch.equal(lengths, batch['label_lengths']):
            raise ValueError('Output and label lengths disagree')
        if not torch.isfinite(pooled).all():
            raise ValueError('Model produced non-finite logits')
        return pooled.cpu()
    fallback = False
    try:
        output = forward(device)
    except torch.cuda.OutOfMemoryError:
        if device != 'cuda':
            raise
        fallback = True
    if fallback:
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        output = forward('cpu')
        model.to(device)
    return output, 'cpu_oom_fallback' if fallback else device


@torch.inference_mode()
def validate(model, dataset, device, progress=None):
    model.eval()
    scores, targets = [], []
    loss_sum, frame_count, fallbacks, last = 0., 0, 0, 0.
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=0)
    for index, batch in enumerate(loader):
        output, used_device = infer_batch(model, batch, dataset.resolution, device)
        count = int(batch['label_lengths'][0])
        scores.append(output[0, :count])
        targets.append(batch['labels'][0, :count])
        loss_sum += float(masked_bce(output, batch['labels'], batch['label_lengths']))*count
        frame_count += count
        fallbacks += used_device == 'cpu_oom_fallback'
        if progress and (time.monotonic()-last >= 30 or index+1 == len(dataset)):
            progress(index+1, len(dataset))
            last = time.monotonic()
    scores, targets = torch.cat(scores), torch.cat(targets)
    return dict(metrics=official_metrics(scores, targets), loss=loss_sum/frame_count,
                interval_threshold=choose_interval_threshold(scores, targets), cpu_fallback_utterances=fallbacks)


def train_epoch(model, loader, optimizer, device, progress=None):
    model.train()
    total_loss, total_frames, last = 0., 0, 0.
    for index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        native = model(batch['features'].to(device), batch['lengths'])
        output, lengths = pool_logits(native, batch['lengths'], 0.16)
        if not torch.equal(lengths, batch['label_lengths']):
            raise ValueError('Training target alignment mismatch')
        loss = masked_bce(output, batch['labels'].to(device), lengths)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        count = int(lengths.sum())
        total_loss += float(loss.detach())*count
        total_frames += count
        if progress and (time.monotonic()-last >= 30 or index+1 == len(loader)):
            progress(index+1, len(loader), total_loss/total_frames)
            last = time.monotonic()
    return total_loss/total_frames


def train(config=None, progress=None):
    config = load_config() if config is None else validate_config(config)
    check_training_files()
    require_device_idle(config['device'])
    configure_runtime(config)
    record = RunRecord('train', training_dataset='PartialSpoof train', validation_dataset='PartialSpoof dev',
                       smoke_test=False, config=config, weights_pretrained=False, progress=progress)
    try:
        profile = config['training_data_profile']
        audit = audit_splits(profile)
        model_config = ModelConfig(**config['model'])
        train_data = BaselineDataset('train', model_config, config['crop_seconds'], data_profile=profile)
        dev_data = BaselineDataset('dev', model_config, data_profile=profile)
        record.update('loading_model', data_audit=audit, data_profile=profile, train_utterances=len(train_data),
                      dev_utterances=len(dev_data), train_manifest_sha256=train_data.manifest_sha256,
                      dev_manifest_sha256=dev_data.manifest_sha256,
                      comparability='Same official splits and metric protocol; training/pretraining conditions differ from public SAL checkpoint')
        write_json(record.directory/'config.json', config)
        write_json(record.directory/'data_audit.json', audit)
        model = CNNBiGRU(model_config).to(config['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        generator = torch.Generator().manual_seed(config['seed'])
        loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['num_workers'], collate_fn=collate, generator=generator)
        weights = DATA_ROOT/'models'/f'{record.name}_best.pt'
        weights.parent.mkdir(parents=True, exist_ok=True)
        best_eer, stale, history = float('inf'), 0, []
        for epoch in range(1, config['epochs']+1):
            loss = train_epoch(model, loader, optimizer, config['device'],
                lambda done, total, loss: record.update('running', phase='train', epoch=epoch,
                                                       batches_completed=done, batches_total=total, train_loss=loss))
            validation = validate(model, dev_data, config['device'],
                lambda done, total: record.update('running', phase='validation', epoch=epoch,
                                                  validation_completed=done, validation_total=total))
            entry = dict(epoch=epoch, train_loss=loss, **validation)
            history.append(entry)
            write_json(record.directory/'history.json', history)
            eer = validation['metrics']['eer_percent']
            if validation['metrics']['single_class_eer_sentinel']:
                raise ValueError('Validation must contain both real and spoof frames')
            if eer < best_eer:
                best_eer, stale = eer, 0
                checkpoint = dict(protocol=PROTOCOL, model_config=model_config.to_dict(),
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    train_run=record.name, epoch=epoch, seed=config['seed'], config=config,
                    validation_metrics=validation['metrics'], interval_threshold=validation['interval_threshold'],
                    threshold_source=f'PS dev {profile} 160 ms, grid search for spoof F1; intervals only',
                    training_data_profile=profile,
                    train_manifest_sha256=train_data.manifest_sha256, dev_manifest_sha256=dev_data.manifest_sha256,
                    data_audit=audit, trained=True, internal_labels={'bonafide': 0, 'spoof': 1})
                temporary = weights.with_suffix('.tmp')
                torch.save(checkpoint, temporary)
                temporary.replace(weights)
                record.update('running', best_epoch=epoch, best_dev_eer_percent=eer,
                              checkpoint=os.path.relpath(weights, CODE_ROOT).replace('\\', '/'))
            else:
                stale += 1
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if stale >= config['patience']:
                break
        record.update('complete', epochs_completed=len(history), checkpoint_sha256=sha256(weights),
                      stop_reason='early_stopping' if stale >= config['patience'] else 'epoch_limit')
        print('TRAIN_RESULT_FILE', record.path, flush=True)
        return record.path
    except BaseException as error:
        record.update('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise


def load_trained_model(config):
    config = validate_config(config)
    if config.get('checkpoint'):
        path = (CODE_ROOT/config['checkpoint']).resolve()
    else:
        complete = []
        for result in RESULT_ROOT.glob('cnn_bigru_train_*.json'):
            data = json.loads(result.read_text(encoding='utf-8'))
            if (data.get('status') == 'complete' and data.get('smoke_test') is False
                    and data.get('config', {}).get('training_data_profile', 'full') == config['training_data_profile']
                    and (not config.get('grid_selected_checkpoint') or
                         configuration_signature(validate_config(data['config'])) == configuration_signature(config))):
                complete.append((result.name, data))
        if not complete:
            raise FileNotFoundError('No completed CNN-BiGRU training run. Prepare PS train audio, then run cnn/baseline.py with RUN_MODE="full" or "train".')
        data = max(complete, key=lambda row: row[0])[1]
        path = (CODE_ROOT/data['checkpoint']).resolve()
        if sha256(path) != data['checkpoint_sha256']:
            raise ValueError('Training checkpoint checksum differs from completed run record')
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('config', {}).get('training_data_profile', 'full') != config['training_data_profile']:
        raise ValueError('Checkpoint training-data profile differs from configured profile; old full-data weights cannot stand in for fixed10 training')
    if checkpoint.get('protocol') != PROTOCOL or checkpoint.get('trained') is not True:
        raise ValueError('Expected a trained CNN-BiGRU checkpoint using this evaluation protocol')
    model_config = ModelConfig(**checkpoint['model_config'])
    model = CNNBiGRU(model_config)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint, path


def evaluate(split, checkpoint_path=None, config=None, progress=None):
    if split not in ('dev', 'eval', 'lps'):
        raise ValueError('Choose dev, eval or lps')
    config = validate_config(load_config() if config is None else config)
    if checkpoint_path is not None:
        # Full pipeline pins both tests to this training run, not a later run.
        config['checkpoint'] = str(checkpoint_path)
    # Missing weights should fail before any model evaluation or output directory.
    model, checkpoint, weights = load_trained_model(config)
    require_device_idle(config['device'])
    configure_runtime(config)
    profile = config['evaluation_data_profile']
    record = RunRecord(f'{split}_{profile}', split=split, smoke_test=False, completed=0,
                       **profile_info(split, profile), training_data_profile=config['training_data_profile'],
                       dataset='LlamaPartialSpoof crossfade' if split == 'lps' else f'PartialSpoof {split}',
                       metric_unit='percent', precision='float32', batch_size=1, full_utterance=True,
                       checkpoint=os.path.relpath(weights, CODE_ROOT).replace('\\', '/'), checkpoint_sha256=sha256(weights),
                       train_run=checkpoint['train_run'], interval_threshold=checkpoint['interval_threshold'],
                       threshold_source=checkpoint['threshold_source'], progress=progress)
    connection = None
    try:
        dataset = BaselineDataset(split, ModelConfig(**checkpoint['model_config']), data_profile=profile)
        record.update('loading_model', planned_utterances=len(dataset), resolution_seconds=dataset.resolution,
                      manifest_sha256=dataset.manifest_sha256, detail_directory=record.name,
                      predictions_file=f'{record.name}/predictions.sqlite',
                      native_resolution_seconds=0.02, aggregation='mean of native logits within each evaluation bin',
                      official_labels={'spoof': 0, 'bonafide': 1},
                      interval_rule='Contiguous probabilities >= dev-selected threshold; half-open bins; no smoothing',
                      runtime_scope='LogMel frontend plus CNN/BiGRU/head; excludes disk audio loading',
                      inference_device=config['device'], cpu_fallback_utterances=0)
        write_json(record.directory/'config.json', dict(runtime=config, checkpoint_model=checkpoint['model_config']))
        connection = sqlite3.connect(record.directory/'predictions.sqlite')
        connection.execute('CREATE TABLE predictions (uid TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, logits BLOB, labels BLOB, seconds REAL, runtime REAL, device TEXT)')
        connection.execute('CREATE TABLE intervals (uid TEXT PRIMARY KEY, threshold REAL, resolution REAL, spoof_intervals_json TEXT)')
        model.to(config['device']).eval()
        last, fallbacks, inference_seconds, audio_seconds, completed = 0., 0, 0., 0., 0
        loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)
        with torch.inference_mode():
            for ordinal, batch in enumerate(loader):
                started = time.perf_counter()
                output, used_device = infer_batch(model, batch, dataset.resolution, config['device'])
                runtime = time.perf_counter()-started+batch['frontend_seconds'][0]
                count = int(batch['label_lengths'][0])
                logits = sal_logits(output[0, :count]).numpy().astype(np.float32)
                labels = (1-batch['labels'][0, :count]).numpy().astype(np.uint8)
                uid, duration = batch['uids'][0], batch['durations'][0]
                intervals = probabilities_to_intervals(output[0, :count].sigmoid(), dataset.resolution,
                                                       duration, checkpoint['interval_threshold'])
                connection.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)',
                                   (uid, ordinal, logits.tobytes(), labels.tobytes(), duration, runtime, used_device))
                connection.execute('INSERT INTO intervals VALUES (?,?,?,?)',
                                   (uid, checkpoint['interval_threshold'], dataset.resolution, json.dumps(intervals)))
                completed = ordinal+1
                fallbacks += used_device == 'cpu_oom_fallback'
                inference_seconds += runtime
                audio_seconds += duration
                if completed % 25 == 0:
                    connection.commit()
                if time.monotonic()-last >= 30 or completed == len(dataset):
                    connection.commit()
                    record.update('running', completed=completed, last_uid=uid, cpu_fallback_utterances=fallbacks)
                    print(f'{split}: {completed}/{len(dataset)}', flush=True)
                    last = time.monotonic()
        connection.commit()
        model.cpu()
        if config['device'] == 'cuda':
            torch.cuda.empty_cache()
        record.update('running', phase='calculating_metrics')
        # Same database schema and exact metric function as the SAL experiment.
        from sal.pipeline import calculate_metrics
        metrics = calculate_metrics(connection)
        if metrics['utterance_count'] != len(dataset):
            raise RuntimeError('Incomplete predictions')
        metrics.update(**profile_info(split, profile), manifest_sha256=dataset.manifest_sha256,
                       audio_seconds=audio_seconds, inference_seconds=inference_seconds,
                       real_time_factor=inference_seconds/audio_seconds, cpu_fallback_utterances=fallbacks,
                       planned_utterances=len(dataset), resolution_seconds=dataset.resolution,
                       split=split, run=record.name, smoke_test=False)
        write_json(record.directory/'metrics.json', metrics)
        record.update('complete', **metrics)
        print('RESULT', json.dumps(metrics), flush=True)
        print('RESULT_FILE', record.path, flush=True)
        return record.path
    except BaseException as error:
        if connection is not None:
            connection.commit()
            count = connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        else:
            count = 0
        record.update('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), completed=count)
        raise
    finally:
        if connection is not None:
            connection.close()


def entrypoint(function, *args):
    try:
        value = function(*args)
        return value if isinstance(value, int) else 0
    except KeyboardInterrupt:
        print('Interrupted. Existing run artifacts were retained.', flush=True)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


def full_pipeline(progress=None, evaluation_data_profile=None):
    """Run each stage sequentially; an exception prevents all later stages."""
    if progress:
        progress('training')
    training_file = train()
    training = json.loads(training_file.read_text(encoding='utf-8'))
    if training.get('status') != 'complete':
        raise RuntimeError('Training did not complete; test evaluation was not started')
    checkpoint = training['checkpoint']
    results = dict(training=str(training_file))
    evaluation_config = load_config()
    if evaluation_data_profile is not None:
        evaluation_config['evaluation_data_profile'] = evaluation_data_profile
    for split, name in (('eval', 'ps'), ('lps', 'lps')):
        if progress:
            progress(f'evaluating_{name}', **results)
        results[name] = str(evaluate(split, checkpoint_path=checkpoint, config=evaluation_config))
    from tools.reporting import compare
    if progress:
        progress('comparing', **results)
    comparison = compare(baseline_paths={'eval': results['ps'], 'lps': results['lps']}, data_profile=evaluation_config['evaluation_data_profile'])
    if comparison is not None:
        results['comparison'] = str(comparison)
        report = json.loads(Path(comparison).read_text(encoding='utf-8'))
        if not all(row['comparable'] for row in report['comparisons']):
            raise RuntimeError(f'Evaluation results are not comparable; inspect {comparison}')
    print('PIPELINE_COMPLETE', json.dumps(results, ensure_ascii=False), flush=True)
    return results




def predict():
    import soundfile as sf
    import torchaudio
    if not AUDIO_FILE:
        print('请先填写本文件顶部的 AUDIO_FILE，再点击运行。尚未启动推理。')
        return
    path = (CODE_ROOT/AUDIO_FILE).resolve()
    config = load_config()
    model, checkpoint, weights = load_trained_model(config)
    require_device_idle(config['device'])
    configure_runtime(config)
    waveform, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    duration = len(waveform)/sample_rate
    waveform = torch.from_numpy(waveform.mean(axis=1))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    features = LogMel(ModelConfig(**checkpoint['model_config']))(waveform).unsqueeze(0)
    lengths = torch.tensor([features.shape[2]])
    batch = dict(features=features, lengths=lengths, label_lengths=lengths)
    record = RunRecord('prediction', audio_file=str(path), duration_seconds=duration,
                       checkpoint=str(weights), resolution_seconds=0.02,
                       interval_threshold=checkpoint['interval_threshold'],
                       input_sample_rate=sample_rate, analysis_sample_rate=16000)
    try:
        model.to(config['device']).eval()
        with torch.inference_mode():
            logits, used_device = infer_batch(model, batch, 0.02, config['device'])
        probabilities = logits[0].sigmoid()
        intervals = probabilities_to_intervals(probabilities, 0.02, duration, checkpoint['interval_threshold'])
        write_json(record.directory/'frame_scores.json', dict(spoof_probabilities=probabilities.tolist(),
                   resolution_seconds=0.02, frame_center_rule='(k+0.5)*0.02 seconds'))
        record.update('complete', spoof_intervals=intervals, inference_device=used_device,
                      frame_scores_file=f'{record.name}/frame_scores.json')
        print('SPOOF_INTERVALS', intervals)
        print('RESULT_FILE', record.path)
    except BaseException as error:
        record.update('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise


def check_setup():
    config = load_config()
    print('CONFIG OK: cnn/config.json')
    print('Hyperparameter source:', config.get('hyperparameter_source'),
          '; learning_rate:', config['learning_rate'], '; dropout:', config['model']['dropout'])
    print('Training/validation profile:', config['training_data_profile'], '; test profile:', config['evaluation_data_profile'])
    print('Current SAL results are public-checkpoint evaluation, not matched from-scratch training.')
    print(json.dumps(audit_splits(config['training_data_profile']), ensure_ascii=False, indent=2))
    try:
        check_training_files()
    except FileNotFoundError as error:
        print(str(error))
        print('NOT READY: prepare the official PS train audio before clicking cnn/baseline.py.')
        return 1
    print('TRAINING FILES PRESENT. Full audio/label ID coverage is checked when loading datasets.')
    print('CNN-BiGRU checkpoints:', len(list((DATA_ROOT/'models').glob('cnn_bigru_train_*_best.pt'))))
    print('No experiment was started.')
    return 0


def prepare_training_data():
    from tools.download_resources import download, digest
    from tools.prepare_data import extract
    sources = json.loads((CODE_ROOT/'manifests/resource_sources.json').read_text(encoding='utf-8'))
    item = next(dict(row) for row in sources if row['name'] == 'database_train.tar.gz')
    archive = DATA_ROOT/'downloads/PartialSpoof'/item['name']
    item['path'] = str(archive)
    print(f"Preparing official PS train: {item['size']:,} bytes. No model training will be started.", flush=True)
    result = download(item)
    if result.get('status') != 'verified':
        raise RuntimeError('Download was not verified; no extraction performed')
    if digest(archive, 'sha256') != item['sha256']:
        raise ValueError('SHA256 mismatch; archive retained, no extraction performed')
    extract(('PartialSpoof', item['name']))
    check_training_files()
    dataset = BaselineDataset('train', ModelConfig(**load_config()['model']))
    print(f'PS TRAIN READY: {len(dataset)} files with matching official labels. No training was started.')


def main(mode=None, evaluation_data_profile=None):
    mode = RUN_MODE if mode is None else mode
    if mode == 'check':
        return check_setup()
    if mode == 'prepare':
        return prepare_training_data()
    if mode == 'predict':
        return predict()
    if mode == 'full':
        return full_pipeline(evaluation_data_profile=evaluation_data_profile)
    if mode == 'train':
        return train()
    if mode in ('dev', 'ps', 'lps'):
        config = load_config()
        if evaluation_data_profile is not None:
            config['evaluation_data_profile'] = evaluation_data_profile
        return evaluate('eval' if mode == 'ps' else mode, config=config)
    raise ValueError('RUN_MODE must be train, full, ps, lps, check, prepare, or predict')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'full', 'ps', 'lps', 'dev', 'check', 'prepare', 'predict'), default=RUN_MODE)
    parser.add_argument('--evaluation-data-profile', choices=('fixed10', 'full'), default=None,
                        help='Override evaluation data only; training remains configured fixed10')
    arguments = parser.parse_args()
    raise SystemExit(entrypoint(main, arguments.mode, arguments.evaluation_data_profile))
