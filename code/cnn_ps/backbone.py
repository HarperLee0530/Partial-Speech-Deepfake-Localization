"""Original 20 ms CNN-BiGRU and log-Mel frontend, independent of other models.

Internal labels are 0=real and 1=fake.
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
    """Mean of native logits; use resolution=0.16 for the PS main comparison."""
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



def extract_context(model, features, lengths):
    """Return native logits and BiGRU context without changing model arithmetic."""
    return model(features, lengths, return_features=True)


def load_backbone(checkpoint, device="cpu"):
    """Load a trusted local original CNN checkpoint; return frozen model, CPU frontend.

    PyTorch checkpoint files can contain pickle payloads: load only files from
    trusted experiment runs. No network download or checkpoint guessing occurs.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in payload or "state_dict" not in payload:
        raise ValueError("Expected a CNN checkpoint with model_config and state_dict")
    config = ModelConfig(**payload["model_config"])
    model = CNNBiGRU(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, LogMel(config).eval()
