"""The three waveform views used by the PS channel-augmentation experiment.

Transform the complete waveform before feature extraction. All views preserve
sample count and labels; no clipping, splicing, resampling or time warping is
applied. The noise is deterministic for each recording UID.
"""
from __future__ import annotations

import hashlib

import numpy as np


CHANNELS = ("original", "white_noise_20db", "mild_eq")
NOISE_SEED = 20261115


def waveform_sha256(waveform):
    return hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest()


def transform(wave, uid, channel):
    """Return one float32 view, matching the saved experiment's arithmetic."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected nonempty, finite mono waveform")
    if channel == 0:
        return wave.copy()
    if channel == 1:
        seed = int.from_bytes(hashlib.sha256(
            f"{NOISE_SEED}|{uid}|white_noise_20db".encode()).digest()[:8], "little")
        noise = np.random.default_rng(seed).standard_normal(len(wave))
        noise -= noise.mean()
        rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
        nrms = np.sqrt(np.mean(noise ** 2))
        return (wave.astype(np.float64) + noise * (rms / (10 * max(nrms, 1e-30)))).astype(np.float32)
    if channel == 2:
        # Keep this operation order for historical float32 reproducibility.
        # The repeated first sample prevents an artificial initial zero step.
        previous = np.r_[wave[:1], wave[:-1]]
        return (wave + np.float32(.15) * (wave - previous)).astype(np.float32)
    raise ValueError(f"Unknown channel: {channel}; expected 0, 1 or 2")
