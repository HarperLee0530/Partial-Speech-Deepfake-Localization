"""Small real-audio end-to-end check, separate from the measured experiment."""
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from cnn_ps.io import save_json, read_json
from cnn_ps.wavlm_multilayer import build_cache, MultiLayerData, FrozenMultiWavLM
from cnn_ps.train_wavlm_multilayer import train_arm, freeze_eval_plan, evaluate_arms, setup


def main():
    repo = Path(__file__).resolve().parents[1]
    audit = repo.parent/'handoff_audit'
    original = audit/'smoke_20261117/run1'
    output = audit/'multilayer_20261117/smoke'
    setup(666, 2)
    for split in ('train', 'dev'):
        build_cache(original/'mel'/split, repo/'data/models/wavlm-base-plus', output/'layers'/split)
        previous = np.load(original/'ssl'/split/'features.npy', mmap_mode='r')
        current = np.load(output/'layers'/split/'features.npy', mmap_mode='r')
        np.testing.assert_array_equal(current[:, :, -1, :], previous)
    args = Namespace(output=output/'training', seed=666, threads=2, device='cuda',
                     epochs=1, batch_size=8, train_cache=original/'mel/train', dev_cache=original/'mel/dev')
    for arm in ('layer12', 'fusion6_9_12'):
        args.arm = arm
        train = MultiLayerData(args.train_cache, output/'layers/train', arm=arm)
        dev = MultiLayerData(args.dev_cache, output/'layers/dev', arm=arm)
        train_arm(args, train, dev)
    frozen = freeze_eval_plan(args.output, ['layer12', 'fusion6_9_12'])
    build_cache(original/'mel/eval', repo/'data/models/wavlm-base-plus', output/'layers/eval',
                frozen_plan=args.output/'frozen_eval_plan.json')
    datasets = {arm: MultiLayerData(original/'mel/eval', output/'layers/eval', arm=arm)
                for arm in frozen}
    evaluate_arms(args, datasets, frozen)
    model = FrozenMultiWavLM(repo/'data/models/wavlm-base-plus')
    assert all(not p.requires_grad for p in model.encoder.parameters())
    result = dict(status='passed', last_layer_cache_bitwise_equal=True,
        frozen_encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        training_updates_per_arm=1, real_audio_splits=dict(train=4, dev=4, eval=2),
        purpose='Pipeline verification only; not evidence of performance improvement')
    save_json(output/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
