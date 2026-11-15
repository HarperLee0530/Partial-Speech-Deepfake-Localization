"""Small residual adapters over full-context CNN-BiGRU features.

This reconstructs the proposed architecture from its written specification; it
is not a reproduction of an unavailable implementation or reported result.
The incoming BiGRU features already contain whole-recording context. Local
cross-attention limits only this adapter's additional connections, not the
backbone receptive field, and is not a streaming/causal mechanism.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DualResolutionAdapter(nn.Module):
    """Adapt one recording using native20 features and masked160 summaries.

    Args:
        mean, std: 256 train-clean feature statistics, estimated using valid
            frame durations only by the caller. They are fixed state buffers.
        variant: residual (fine only), dual (fixed half/half residual mixture),
            or fusion (local cross-attention plus a learned disagreement gate).

    Inputs h[T,256], z[T], weights[T] use T divisible by8. weights are actual
    frame durations, including a fractional final frame and zero padding.
    Gradients through h and z are retained for end-to-end backbone fine-tuning.
    Feature statistics and duration weights remain fixed.
    Returned logits are fake-positive logits, suitable for BCEWithLogitsLoss.
    Invalid frames retain their original storage-grid z with zero residual;
    this preserves historical mean8 final160 supervision. Independent coarse
    summaries, auxiliary logits and attention exclude invalid frames; empty
    coarse blocks return zero. All storage-grid z values must be finite.

    All variants instantiate the fine path first, so the same manual_seed
    gives identical fine initialization. dual/fusion also share the same
    coarse initialization. Zero residual output layers make valid final
    logits exactly z at construction, including for the learned gate.
    """

    block_size = 8
    feature_dim = 256

    def __init__(self, mean, std, variant="fusion", hidden=32, heads=4, radius=2,
                 learned_pool=False, boundary_head=False, relative_position=False,
                 reliability_gate=False):
        super().__init__()
        if variant not in ("residual", "dual", "fusion"):
            raise ValueError("variant must be residual, dual or fusion")
        if (not isinstance(hidden, int) or not isinstance(heads, int)
                or hidden <= 0 or heads <= 0 or hidden % heads):
            raise ValueError("hidden must be positive and divisible by positive heads")
        if not isinstance(radius, int) or radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        if variant == "residual" and learned_pool:
            raise ValueError("learned_pool requires a coarse branch")
        if variant != "fusion" and (relative_position or reliability_gate):
            raise ValueError("relative_position and reliability_gate require fusion")
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        if mean.shape != (self.feature_dim,) or std.shape != mean.shape:
            raise ValueError("mean and std must each contain256 features")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Feature statistics must be finite, with nonnegative std")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-5))
        self.variant, self.hidden, self.heads, self.radius = variant, hidden, heads, radius
        self.learned_pool = bool(learned_pool)
        self.boundary_head = bool(boundary_head)
        self.relative_position = bool(relative_position)
        self.reliability_gate = bool(reliability_gate)

        # Keep this order common to all variants for paired initialization.
        self.fine_projection = nn.Linear(self.feature_dim, hidden)
        self.fine_output = nn.Linear(hidden, 1)
        self._zero_output(self.fine_output)
        if variant != "residual":
            self.coarse_projection = nn.Linear(self.feature_dim, hidden)
            self.coarse_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.coarse_output = nn.Linear(hidden, 1)
            self._zero_output(self.coarse_output)
        if variant == "fusion":
            self.query_projection = nn.Linear(hidden, hidden)
            self.key_projection = nn.Linear(hidden, hidden)
            self.value_projection = nn.Linear(hidden, hidden)
            self.attention_output = nn.Linear(hidden, 1)
            self._zero_output(self.attention_output)
            gate_dim = 10 if reliability_gate else 3
            self.gate_network = nn.Sequential(nn.Linear(gate_dim, 16), nn.GELU(), nn.Linear(16, 1))

        # Optional modules come after the common branch parameters, keeping
        # fine/coarse/attention initialization paired across option ablations.
        if learned_pool:
            self.coarse_downsample = nn.Conv1d(self.feature_dim, self.feature_dim,
                kernel_size=self.block_size, stride=self.block_size,
                groups=self.feature_dim, bias=False)
            nn.init.constant_(self.coarse_downsample.weight, 1 / self.block_size)
        if boundary_head:
            self.boundary_output = nn.Linear(hidden, 1)
            self._zero_output(self.boundary_output)
        if relative_position:
            self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * radius + 1))

    @staticmethod
    def _zero_output(layer):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    @classmethod
    def masked_pool(cls, values, weights):
        """Duration-weighted block8 mean; supports [T] or [T,D]."""
        if values.ndim not in (1, 2) or len(values) % cls.block_size or weights.shape != (len(values),):
            raise ValueError("Masked pool requires [T] or [T,D] and weights[T], with T divisible by8")
        valid = weights > 0
        clean = torch.where(valid if values.ndim == 1 else valid[:, None], values, torch.zeros_like(values))
        grouped_weights = weights.reshape(-1, cls.block_size)
        totals = grouped_weights.sum(1)
        denominator = torch.where(totals > 0, totals, torch.ones_like(totals))
        if values.ndim == 1:
            pooled = (clean.reshape(-1, cls.block_size) * grouped_weights).sum(1) / denominator
        else:
            pooled = (clean.reshape(-1, cls.block_size, values.shape[1])
                      * grouped_weights[:, :, None]).sum(1) / denominator[:, None]
        return pooled, totals

    def _inputs(self, h, z, weights):
        if (h.ndim != 2 or h.shape[1] != self.feature_dim or not len(h)
                or len(h) % self.block_size or z.shape != (len(h),) or weights.shape != (len(h),)):
            raise ValueError("Expected h[T,256], z[T], weights[T], with nonzero T divisible by8")
        if h.device != self.feature_mean.device or z.device != h.device or weights.device != h.device:
            raise ValueError("Inputs and adapter buffers must be on the same device")
        h = h.to(dtype=self.feature_mean.dtype)
        z = z.to(dtype=self.feature_mean.dtype)
        weights = weights.detach().to(dtype=self.feature_mean.dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Duration weights must be finite and nonnegative")
        valid = weights > 0
        # Mask before normalization, pooling and projection, so even arbitrary
        # nonfinite values in padding cannot contaminate genuine frames.
        h = torch.where(valid[:, None], h, self.feature_mean[None, :])
        if not torch.isfinite(h).all() or not torch.isfinite(z).all():
            raise ValueError("Valid features and all storage-grid logits must be finite")
        normalized = ((h - self.feature_mean) / self.feature_std).clamp(-5., 5.)
        return normalized, z, weights, valid

    def _local_attention(self, fine_features, coarse_features, valid, coarse_valid):
        """Gather only2*radius+1 coarse neighbors; never allocate a T by C map."""
        t, c = len(fine_features), len(coarse_features)
        center = torch.arange(t, device=fine_features.device) // self.block_size
        offsets = torch.arange(-self.radius, self.radius + 1, device=fine_features.device)
        indices = center[:, None] + offsets[None, :]
        in_bounds = (indices >= 0) & (indices < c)
        gathered_indices = indices.clamp(0, c - 1)
        allowed = in_bounds & coarse_valid[gathered_indices] & valid[:, None]
        head_dim = self.hidden // self.heads
        query = self.query_projection(fine_features).reshape(t, self.heads, head_dim)
        keys = self.key_projection(coarse_features).reshape(c, self.heads, head_dim)[gathered_indices]
        values = self.value_projection(coarse_features).reshape(c, self.heads, head_dim)[gathered_indices]
        scores = torch.einsum("thd,tkhd->thk", query, keys) / math.sqrt(head_dim)
        if self.relative_position:
            scores = scores + self.relative_bias[None, :, :]
        masked = scores.masked_fill(~allowed[:, None, :], -torch.inf)
        # A recording can contain only padding: softmax(-inf,...) is avoided.
        safe = torch.where(allowed.any(1)[:, None, None], masked, torch.zeros_like(masked))
        attention = torch.softmax(safe, dim=-1) * allowed[:, None, :].to(scores.dtype)
        context = torch.einsum("thk,tkhd->thd", attention, values).reshape(t, self.hidden)
        return context, attention, indices, allowed

    def _coarse_input(self, normalized, weights):
        if not self.learned_pool:
            return self.masked_pool(normalized, weights)[0]
        totals = weights.reshape(-1, self.block_size).sum(1)
        denominator = torch.where(totals > 0, totals, torch.ones_like(totals))
        # A 1/8 depthwise kernel starts at the duration-weighted mean, even
        # for partial blocks. Learned per-channel offsets can subsequently
        # retain distinct within-block cues; invalid cells always stay zero.
        scale = self.block_size * weights / denominator.repeat_interleave(self.block_size)
        weighted = normalized * scale[:, None]
        pooled = self.coarse_downsample(weighted.transpose(0, 1).unsqueeze(0))
        return pooled.squeeze(0).transpose(0, 1)

    @staticmethod
    def _entropy(probability):
        probability = probability.clamp(1e-6, 1 - 1e-6)
        return -(probability * probability.log() + (1-probability) * torch.log1p(-probability)) / math.log(2)

    def _gate_inputs(self, fine_probability, coarse_probability, normalized, valid, boundary_logits):
        features = [fine_probability, coarse_probability, (fine_probability-coarse_probability).abs()]
        if self.reliability_gate:
            # No artificial transition is introduced at a padded neighbor.
            pair_valid = valid[1:] & valid[:-1]
            change = torch.linalg.vector_norm(normalized[1:]-normalized[:-1], dim=1) / math.sqrt(self.feature_dim)
            change = change * pair_valid
            previous = F.pad(change, (1, 0))
            following = F.pad(change, (0, 1))
            boundary_probability = (torch.sigmoid(boundary_logits) if boundary_logits is not None
                                    else torch.zeros_like(fine_probability))
            features.extend([self._entropy(fine_probability), self._entropy(coarse_probability),
                (2*fine_probability-1).abs(), (2*coarse_probability-1).abs(),
                previous, following, boundary_probability])
        return torch.stack(features, dim=-1)

    def forward(self, h, z, weights):
        normalized, base, weights, valid = self._inputs(h, z, weights)
        fine_features = F.gelu(self.fine_projection(normalized)) * valid[:, None]
        fine_residual = self.fine_output(fine_features).squeeze(-1) * valid
        fine_logits = base + fine_residual
        coarse_base, coarse_weights = self.masked_pool(base, weights)
        coarse_valid = coarse_weights > 0
        result = dict(fine_logits=fine_logits, fine_residual=fine_residual,
                      coarse_base=coarse_base, coarse_weights=coarse_weights,
                      valid_mask=valid, coarse_valid_mask=coarse_valid)
        boundary_logits = None
        if self.boundary_head:
            boundary_logits = self.boundary_output(fine_features).squeeze(-1) * valid
            result["boundary_logits"] = boundary_logits
        if self.variant == "residual":
            # No independent coarse branch. Its auxiliary supervision repeats
            # the pooled fine objective, matching the stated loss budget.
            coarse_logits, _ = self.masked_pool(fine_logits, weights)
            coarse_residual, _ = self.masked_pool(fine_residual, weights)
            result.update(final_logits=fine_logits, coarse_logits=coarse_logits,
                          coarse_residual=coarse_residual)
            return result

        coarse_input = self._coarse_input(normalized, weights)
        coarse_features = F.gelu(self.coarse_projection(coarse_input)) * coarse_valid[:, None]
        coarse_features = self.coarse_conv(coarse_features.transpose(0, 1).unsqueeze(0))
        coarse_features = F.gelu(coarse_features.squeeze(0).transpose(0, 1)) * coarse_valid[:, None]
        coarse_residual = self.coarse_output(coarse_features).squeeze(-1) * coarse_valid
        coarse_logits = coarse_base + coarse_residual
        repeated_residual = coarse_residual.repeat_interleave(self.block_size) * valid
        result.update(coarse_logits=coarse_logits, coarse_residual=coarse_residual)
        if self.variant == "dual":
            result["final_logits"] = base + .5 * fine_residual + .5 * repeated_residual
            return result

        context, attention, indices, allowed = self._local_attention(fine_features, coarse_features, valid, coarse_valid)
        attention_residual = self.attention_output(context).squeeze(-1) * valid
        fused_coarse_residual = repeated_residual + attention_residual
        fine_probability = torch.sigmoid(fine_logits)
        coarse_probability = torch.sigmoid(coarse_base.repeat_interleave(self.block_size) + fused_coarse_residual)
        gate_inputs = self._gate_inputs(fine_probability, coarse_probability, normalized, valid, boundary_logits)
        gate = torch.sigmoid(self.gate_network(gate_inputs).squeeze(-1)) * valid
        result.update(final_logits=base + (1 - gate) * fine_residual + gate * fused_coarse_residual,
                      gate=gate, gate_inputs=gate_inputs, attention_residual=attention_residual,
                      fused_coarse_residual=fused_coarse_residual,
                      attention_weights=attention, attention_indices=indices, attention_mask=allowed)
        return result
