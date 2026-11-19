"""Portable fresh run of the archived noise replay recipes. Never writes into the archive.

Uses the exact historical training implementation, with configurable paths.
--check validates inputs without training. Full runs require a CUDA GPU.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'noise11': 'multilayer_noise_replay_20261119',
    'noise31': 'multilayer_noise_replay31_20261119',
    'noise51': 'multilayer_noise_replay51_20261119',
    'noise31_lr5e5': 'multilayer_noise_replay31_lr5e5_20261119',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=PROFILES, default='noise31')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='New, nonexistent run directory outside the research archive')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    output = args.output.resolve()
    assert not output.is_relative_to(ROOT), 'Output must be outside the archive.'
    assert not output.exists(), 'Use a new output directory; existing runs are never overwritten.'
    folder = ROOT/'experiments'/PROFILES[args.profile]
    plan = json.loads((folder/'results/plan.json').read_text(encoding='utf-8'))
    for rel, digest in plan['code_sha256'].items():
        assert sha(folder/rel) == digest, ('Archived source modified', rel)
    spec = importlib.util.spec_from_file_location('archived_replay', folder/'run.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DATA = Path(config['data_cache']).resolve()
    m.ENCODER = Path(config['encoder']).resolve()
    m.SOURCE = Path(config['source_checkpoint']).resolve()
    m.RELEASE = Path(config['source_release']).resolve()
    m.OUT = output/'results'
    m.BANK = output/'feature_cache'
    assert sha(m.SOURCE) == m.EXPECTED, 'Wrong original multilayer source checkpoint.'
    assert sha(m.ENCODER/'pytorch_model.bin') == '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
    assert (m.RELEASE/'reports/training/fusion6_9_12/dev_scores20.npz').is_file()
    for split, expected in plan['data_metadata_sha256'].items():
        assert sha(m.DATA/split/'complete.json') == expected, f'{split}: metadata differs; this runner requires the original split/cache metadata.'
    # Resolve raw-audio paths without changing the historical metadata file/hash.
    mappings = config.get('audio_path_map', [])
    original_wave = m.Data.wave
    def mapped_wave(self, i):
        entry = self.meta['records'][i]
        original = entry['audio']
        resolved = original
        for item in mappings:
            prefix = item['from'].replace('\\','/')
            normalized = original.replace('\\','/')
            if normalized.startswith(prefix):
                resolved = str(Path(item['to'])/normalized[len(prefix):].lstrip('/'))
                break
        if resolved == original:
            return original_wave(self, i)
        entry['audio'] = resolved
        try:
            return original_wave(self, i)
        finally:
            entry['audio'] = original
    m.Data.wave = mapped_wave
    old_loader = m.load_backend
    m.load_backend = lambda path=None: old_loader(path if path is not None else m.SOURCE)
    train, dev = m.Data('train'), m.Data('dev')
    assert dev.uids == plan['uids']['clean_full']
    for key in ('target','memory'):
        assert [train.uids[i] for i in plan['indices'][key]] == plan['uids'][key]
    assert [dev.uids[i] for i in plan['indices']['stress']] == plan['uids']['stress']
    train.wave(plan['indices']['target'][0]); dev.wave(0)
    print('PASS: archived source, model/encoder hashes, original metadata/splits and sample audio paths.', flush=True)
    if args.check:
        return
    import torch
    assert torch.cuda.is_available(), 'Full runs require CUDA.'
    m.setup(666,4)
    m.OUT.mkdir(parents=True); m.BANK.mkdir(parents=True)
    fresh = copy.deepcopy(plan)
    fresh.update(source=str(m.SOURCE), arms=['replay'], portable_run=True,
                 historical_plan_sha256=sha(folder/'results/plan.json'),
                 launcher_sha256=sha(__file__), configuration=config,
                 previous_control='Only freshly evaluated original model and two freshly trained replay models. Historical comparisons remain in archive.')
    m.save_json(m.OUT/'plan.json',fresh)
    encoder = m.FrozenMultiWavLM(m.ENCODER,m.DEVICE)
    m.verify_source(encoder,dev,torch.load(m.SOURCE,map_location='cpu',weights_only=False))
    bank = m.prepare_bank(encoder,train,fresh)  # New cache only; original cache reuse checks are not bypassed.
    del encoder; torch.cuda.empty_cache()
    for seed in fresh['seeds']:
        m.train_arm(train,bank,fresh,seed,'replay')
    paths={'original':m.SOURCE}
    paths.update({f'{seed}_replay':m.OUT/f'seed{seed}/replay/last.pt' for seed in fresh['seeds']})
    frozen={name:dict(path=str(path),sha256=sha(path),threshold=fresh['source_threshold']) for name,path in paths.items()}
    m.save_json(m.OUT/'frozen_models.json',frozen)
    del bank;torch.cuda.empty_cache()
    encoder=m.FrozenMultiWavLM(m.ENCODER,m.DEVICE)
    results=m.evaluate_all(encoder,dev,fresh,paths)
    assert sha(m.SOURCE)==m.EXPECTED
    m.save_json(m.OUT/'comparison.json',dict(results=results,source_unchanged=True,formal_test_accessed=False))
    rows=[]
    for model,conditions in results.items():
        for condition,d in conditions.items():
            rows.append(dict(model=model,condition=condition,event_f1=d['event_f1_iou05_percent'],
                frame_f1=d['frame_f1_percent'],real_fpr=d['real_duration_fpr_percent'],
                short_recall=d['short_event_recall_le320ms_percent']))
    with (m.OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    m.progress(status='complete',stage='finished')
    print('Fresh results:',m.OUT,'; historical confidence intervals are not automatically transferred to this new run.')

if __name__=='__main__':
    main()
