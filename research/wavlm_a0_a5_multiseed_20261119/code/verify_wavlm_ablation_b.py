"""Pretrained synthetic forward-only check. Never trains or opens a dataset."""
import copy
from pathlib import Path
import torch
from cnn_ps.io import save_json, sha256
from cnn_ps.wavlm_multilayer import FrozenMultiWavLM
from cnn_ps.wavlm_ablation_b import ARMS, MODEL_SHA256, configure_encoder, encoder_sources
from cnn_ps.train_wavlm_ablation_a import setup, state_digest


@torch.inference_mode()
def verify():
    code = Path(__file__).resolve().parent
    setup(666, 2)
    assert sha256(code.parent/'data/models/wavlm-base-plus/pytorch_model.bin') == MODEL_SHA256
    original = FrozenMultiWavLM(code.parent/'data/models/wavlm-base-plus', 'cpu')
    wave = torch.randn(24*320, generator=torch.Generator().manual_seed(20261119))*.03
    expected = original(wave, 24)
    digest = state_digest(original.encoder)
    rows = {}
    for arm in ARMS:
        changed = copy.deepcopy(original)
        configure_encoder(changed.encoder, arm)
        actual = changed(wave, 24)
        assert state_digest(changed.encoder) == digest
        assert not any(p.requires_grad for p in changed.parameters())
        assert tuple(actual.shape) == (24, 3, 768) and torch.isfinite(actual).all()
        if arm == 'B0_full':
            assert torch.equal(actual, expected)
        else:
            assert not torch.equal(actual, expected)
        rows[arm] = dict(shape=list(actual.shape), frozen=True, parameter_bytes_unchanged=True,
                         max_difference_from_original=float((actual-expected).abs().max()))
        print(arm, rows[arm], flush=True)
        del changed
    root = code.parents[1]/'handoff_audit/ablation_b_20261119_code_checks'
    save_json(root/'pretrained_forward_only.json', dict(status='passed', models=rows,
        encoder_sources=encoder_sources(), input='synthetic noise only; 0.48 seconds',
        device='cpu', training_started=False, optimizer_steps=0, dataset_access=False))


if __name__ == '__main__':
    verify()
