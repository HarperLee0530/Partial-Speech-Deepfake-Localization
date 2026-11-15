"""Audio-only energy5 features, exactly the energy columns of prosody v1.

No pitch estimator, labels or detector outputs are read. Extract from the same
unpadded waveform view as the acoustic frontend, then pad storage cells later.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


SAMPLE_RATE = 16000
HOP = 320
WINDOW = 400
VERSION = "energy5_relative_context_v1"
FEATURE_NAMES = (
    "log_rms_recording_centered", "log_rms_delta",
    "log_rms_minus_neighbours_220ms", "log_rms_minus_neighbours_1020ms",
    "log_rms_neighbour_std_220ms",
)
LEGACY_COLUMNS = (3, 5, 8, 9, 13)


def description():
    return dict(version=VERSION, feature_names=list(FEATURE_NAMES),
                legacy_columns=list(LEGACY_COLUMNS), sample_rate=SAMPLE_RATE,
                hop_samples=HOP, energy_window_samples=WINDOW,
                centers="(k + 0.5) * 320; ceil(samples/320) actual frames",
                reference="Median natural-log RMS of actual frames in this recording",
                rms_floor=1e-8, context_windows_frames=[11, 51],
                context="Exclude current frame; count existing neighbours only",
                padding="Zero waveform outside recording; append storage-tail zeros only after extraction",
                validity="Actual frame-duration mask, never feature nonzero status",
                pitch_extraction=False, label_or_transcript_input=False)


def _neighbour_stats(value, size):
    def neighbours(values):
        return uniform_filter1d(values, size=size, mode="constant", cval=0.) * size - values
    count = np.maximum(neighbours(np.ones(len(value), dtype=np.float64)), 0.)
    mean = np.divide(neighbours(value), count, out=np.zeros_like(value), where=count > .5)
    second = np.divide(neighbours(value * value), count,
                       out=np.zeros_like(value), where=count > .5)
    return mean, np.sqrt(np.maximum(second - mean * mean, 0.)), count


def extract(waveform, sample_rate=SAMPLE_RATE):
    """Return raw float32 [ceil(N/320),5], including legitimate all-zero rows."""
    wave = np.asarray(waveform, dtype=np.float64)
    if sample_rate != SAMPLE_RATE or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected finite nonempty 16kHz mono waveform")
    n = (len(wave) + HOP - 1) // HOP
    left = WINDOW // 2 - HOP // 2
    right = (n - 1) * HOP + WINDOW - left - len(wave)
    padded = np.pad(wave, (left, right))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    if not np.any(rms):
        return np.zeros((n, 5), dtype=np.float32)
    log_rms = np.log(np.maximum(rms, 1e-8))
    energy = log_rms - np.median(log_rms)
    delta = np.r_[0., np.diff(energy)]
    short_mean, short_std, short_count = _neighbour_stats(energy, 11)
    long_mean, _, long_count = _neighbour_stats(energy, 51)
    result = np.column_stack((energy, delta,
        np.where(short_count > .5, energy - short_mean, 0.),
        np.where(long_count > .5, energy - long_mean, 0.), short_std))
    if result.shape != (n, 5) or not np.isfinite(result).all():
        raise RuntimeError("Invalid energy5 features")
    return result.astype(np.float32)
