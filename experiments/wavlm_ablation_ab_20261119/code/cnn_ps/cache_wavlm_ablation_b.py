"""Build one intervention/split cache; no work occurs without --execute."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import soundfile as sf
import torch
from .augmentation import transform
from .dual_features import FeatureData
from .io import read_json, save_json, sha256
from .wavlm_frontend import REVISION
from .wavlm_ablation_b import (ARMS, NEW_ARMS, INTERVENTIONS, CACHE_SCHEMA, MODEL_SHA256,
                                FrozenPositionalWavLM, encoder_sources)


def build_cache(reference, model_folder, output, arm, split, device='cuda', frozen_plan=None):
    reference, model_folder, output = map(Path, (reference, model_folder, output))
    if arm not in ARMS or split not in ('train', 'dev', 'eval'):
        raise ValueError('Unknown arm/split')
    if split == 'eval':
        if frozen_plan is None:
            raise ValueError('All B training must be frozen before opening eval')
        from .train_wavlm_ablation_b import freeze_eval_plan
        plan_path = Path(frozen_plan)
        if plan_path.name != 'frozen_eval_plan.json' or not plan_path.is_file():
            raise ValueError('Expected existing completed B frozen plan')
        freeze_eval_plan(plan_path.parent, list(NEW_ARMS))
    if output.exists():
        raise FileExistsError(output)
    data = FeatureData(reference, split=split, verify_hashes=True)
    shape = (data.channels, int(data.offsets[-1]), 3, 768)
    needed = int(np.prod(shape))*4 + 1024**3
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < needed:
        raise OSError(f'Insufficient disk: need {needed} bytes for cache plus reserve')
    model = FrozenPositionalWavLM(model_folder, arm, device)
    protocol = dict(schema=CACHE_SCHEMA, arm=arm, encoder_intervention=INTERVENTIONS[arm],
        split=split, layers=[6, 9, 12], frozen=True, model_id='microsoft/wavlm-base-plus',
        revision=REVISION, model_sha256=MODEL_SHA256, reference=str(reference.resolve()),
        reference_sha256=sha256(reference/'complete.json'), recordings=len(data),
        channels=data.channels, dtype='float32', encoder_sources=encoder_sources(),
        test_read=split == 'eval', normalization='none; unchanged official preprocessing',
        grid='stride320/receptive400/sidepad40; unchanged full recording storage grid')
    output.mkdir(parents=True)
    save_json(output/'protocol.json', protocol)
    array = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype='float32', shape=shape)
    started = last = time.monotonic()
    try:
        for channel in range(data.channels):
            for index, row in enumerate(data.records):
                wave, sr = sf.read(row['audio'], dtype='float32')
                if sr != 16000 or wave.ndim != 1 or len(wave)/sr != data.durations[index]:
                    raise ValueError('Audio identity/grid mismatch')
                left, right = map(int, data.offsets[index:index+2])
                array[channel, left:right] = model(transform(wave, row['uid'], channel), right-left).cpu().numpy()
                if time.monotonic()-last >= 30:
                    progress = dict(arm=arm, split=split, channel=channel, recordings=index+1,
                                    per_channel=len(data), elapsed_seconds=time.monotonic()-started)
                    save_json(output/'progress.json', progress)
                    print(progress, flush=True)
                    last = time.monotonic()
            array.flush()
        if protocol['encoder_sources'] != encoder_sources():
            raise ValueError('Encoder source changed during caching')
        complete = dict(protocol, status='complete', shape=list(shape), source_unchanged=True,
                        features_sha256=sha256(output/'features.npy'), elapsed_seconds=time.monotonic()-started)
        save_json(output/'complete.json', complete)
        return complete
    finally:
        array.flush()
        del array


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'eval'), required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--frozen-plan', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print('PLAN ONLY: no audio read, no cache created. Add --execute when authorized.')
        return
    from .train_wavlm_ablation_a import setup
    setup(666, 4)
    arguments = vars(args)
    arguments.pop('execute')
    build_cache(**arguments)


if __name__ == '__main__':
    main()
