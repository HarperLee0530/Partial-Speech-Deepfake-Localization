"""Ordered, bounded frequency-knot learning on the original PS power spectrum.

This is a constrained adaptation inspired by learnable filterbank research,
not an exact FastAudio reproduction. Eighty shared interior knot displacements
move the triangular filters' centres and their coupled bandwidths. The nominal
0--8 kHz endpoints remain at torchaudio's original float32 HTK values.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import LogMel, ModelConfig


SCHEMA = 'ps_ordered_frequency_knots_v1'
SHIFT_FRACTION = .2
LOG_FLOOR = 1e-10


def _config(config):
    config = config or ModelConfig()
    values = (config.sample_rate, config.hop_length, config.n_fft,
              config.win_length, config.n_mels)
    if values != (16000, 320, 512, 400, 80):
        raise ValueError('This paired frontend requires 16kHz, hop320, FFT512, window400 and 80 filters')
    return config


class LearnableFilterbank(nn.Module):
    """Float32 [...,257] power -> [...,80] log filterbank representation.

    Both arms execute identical arithmetic. Only raw_shift.requires_grad differs.
    Interior knot i is base_i + .2*min(left_gap,right_gap)*tanh(raw_shift_i).
    These bounds keep neighbouring knots ordered and every triangular filter
    wider than one FFT-bin interval for this declared 80-filter configuration.
    """

    def __init__(self, config=None, learnable=True):
        super().__init__()
        self.config = _config(config)
        if type(learnable) is not bool:
            raise ValueError('learnable must be a bool')
        self.learnable = learnable
        c = self.config
        # Preserve torchaudio 2.2's float32 HTK arithmetic and operation order.
        # Its upper endpoint is 7999.9990234375 Hz on this runtime, not exactly
        # 8000. Replacing it with 8000 would change the initial filter matrix.
        mel_max = 2595.0 * math.log10(1.0 + (c.sample_rate / 2 / 700.0))
        mel_points = torch.linspace(0., mel_max, c.n_mels + 2, dtype=torch.float32)
        base = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        gaps = base[1:] - base[:-1]
        self.register_buffer('base_frequencies', base)
        self.register_buffer('max_displacement', SHIFT_FRACTION * torch.minimum(gaps[:-1], gaps[1:]))
        self.register_buffer('fft_frequencies', torch.linspace(0., c.sample_rate // 2, c.n_fft//2 + 1,
                                                             dtype=torch.float32))
        self.raw_shift = nn.Parameter(torch.zeros(c.n_mels, dtype=torch.float32), requires_grad=learnable)

    def frequencies_hz(self):
        if not torch.isfinite(self.raw_shift).all():
            raise ValueError('Nonfinite frequency parameters')
        inner = self.base_frequencies[1:-1] + self.max_displacement * torch.tanh(self.raw_shift)
        return torch.cat((self.base_frequencies[:1], inner, self.base_frequencies[-1:]))

    def filterbank_matrix(self):
        points = self.frequencies_hz()
        differences = points[1:] - points[:-1]
        slopes = points.unsqueeze(0) - self.fft_frequencies.unsqueeze(1)
        down = (-1.0 * slopes[:, :-2]) / differences[:-1]
        up = slopes[:, 2:] / differences[1:]
        matrix = torch.maximum(points.new_zeros(1), torch.minimum(down, up))
        # Same approximate area normalization as torchaudio's norm='slaney'.
        return matrix * (2.0 / (points[2:] - points[:-2])).unsqueeze(0)

    def forward(self, power):
        if (not isinstance(power, torch.Tensor) or power.ndim < 1
                or power.shape[-1] != self.config.n_fft//2 + 1 or power.numel() == 0):
            raise ValueError('Expected nonempty power with last dimension257')
        if power.dtype != torch.float32 or self.raw_shift.dtype != torch.float32:
            raise ValueError('This experiment requires float32 power and frontend parameters')
        if power.device != self.raw_shift.device:
            raise ValueError('Power and filterbank must be on the same device')
        if not torch.isfinite(power).all() or (power < 0).any():
            raise ValueError('Power must be finite and nonnegative')
        return torch.matmul(power, self.filterbank_matrix()).clamp_min(LOG_FLOOR).log()

    def config_dict(self):
        """State-independent mathematical convention shared by both arms."""
        return dict(schema=SCHEMA, sample_rate=self.config.sample_rate, n_fft=self.config.n_fft,
                    win_length=self.config.win_length, hop_length=self.config.hop_length,
                    n_filters=self.config.n_mels, n_frequency_bins=self.config.n_fft//2 + 1,
                    initial_scale='HTK mel', normalization='Slaney triangular area',
                    nominal_frequency_range_hz=[0.,8000.],
                    endpoint_rule='Fixed original torchaudio float32 HTK endpoints; nominal upper8000Hz',
                    initialization='80 zero raw shifts; same initial matrix as torchaudio Mel80',
                    shift_fraction=SHIFT_FRACTION,
                    parameterization='base_i + .2*min(original adjacent gaps)*tanh(raw_shift_i)',
                    bandwidths='Coupled through adjacent ordered knots; no independent bandwidth parameters',
                    potential_trainable_parameters=80, compression='natural log', log_floor=LOG_FLOOR,
                    window='periodic Hann', power=2.0, stft_center=False,
                    manual_side_padding_samples=(self.config.n_fft-self.config.hop_length)//2,
                    dtype='float32', rng_usage='None')


class PowerSpectrogram(nn.Module):
    """Original PS STFT power with identical window, padding and frame centres.

    The caller pads a recording to its official 160ms storage grid first when
    reproducing training caches. This module also accepts arbitrary nonempty
    waveform lengths and returns ceil(samples/320) frames, exactly like LogMel.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = _config(config)
        self.spectrogram = LogMel(self.config).mel.spectrogram

    def forward(self, waveform):
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError('Expected a nonempty mono waveform')
        if waveform.dtype != torch.float32 or self.spectrogram.window.dtype != torch.float32:
            raise ValueError('This experiment requires float32 waveform and Hann window')
        if waveform.device != self.spectrogram.window.device:
            raise ValueError('Waveform and spectrogram must be on the same device')
        if not torch.isfinite(waveform).all():
            raise ValueError('Waveform contains nonfinite values')
        c = self.config
        frames = math.ceil(waveform.numel()/c.hop_length)
        side = (c.n_fft-c.hop_length)//2
        padded = F.pad(waveform, (side, side+frames*c.hop_length-waveform.numel()))
        result = self.spectrogram(padded).transpose(0,1)
        if result.shape != (frames, c.n_fft//2+1):
            raise RuntimeError('Unexpected power spectrum frame alignment')
        return result.unsqueeze(0)
