import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

"""Fixed data profiles, deterministic 10% subsets, and data audits.

Run with --action prepare/audit/audit-full; default audit never trains a model.
"""
import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from project_config import CODE_ROOT, DATA_ROOT, RESULT_ROOT

ROOT = CODE_ROOT
STORE = DATA_ROOT
PROFILES = ('fixed10', 'full')
SUBSET_ROOT = DATA_ROOT/'fixed10'
SEED = 20261107
FRACTION = 0.1

def manifest_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def read_manifest(split):
    path = SUBSET_ROOT / 'manifests' / (split + '.json')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    ids = manifest['utterance_ids']
    if (manifest['profile'] != 'fixed10' or manifest['split'] != split
            or len(ids) != len(set(ids)) or ids != sorted(ids)
            or manifest_hash(ids) != manifest['manifest_sha256']):
        raise ValueError(f'Invalid fixed subset manifest: {path}')
    return manifest


def profile_info(split, profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown data profile: {profile}')
    info = dict(data_profile=profile, evaluation_role='validation' if split == 'dev' else
                'training' if split == 'train' else 'test',
                dataset_scope='fixed approximately 10% subset' if profile == 'fixed10' else 'complete official split')
    if profile == 'fixed10':
        manifest = read_manifest(split)
        info.update(subset_seed=manifest['seed'], subset_fraction=manifest['actual_fraction'],
                    official_utterances=manifest['official_count'],
                    subset_manifest_sha256=manifest['manifest_sha256'])
    return info


def dataset_paths(split, profile):
    profile_info(split, profile)
    if profile == 'fixed10':
        base = SUBSET_ROOT / ('LlamaPartialSpoof' if split == 'lps' else 'PartialSpoof/database')
        audio = base / ('cf' if split == 'lps' else split)
        return audio, base / 'segment_labels'
    if split == 'lps':
        return DATA_ROOT/'LlamaPartialSpoof', DATA_ROOT/'LlamaPartialSpoof/segment_labels'
    base = DATA_ROOT/'PartialSpoof/database'
    return base/split, base/'segment_labels'


def strata_for(labels, protocol):
    groups = defaultdict(list)
    for uid, values in labels.items():
        fake = np.asarray(values).astype(np.int64) == 0
        ratio = float(fake.mean())
        kind = 'genuine' if ratio == 0 else 'fully_fake' if ratio == 1 else 'partial'
        edges = np.diff(np.r_[False, fake, False].astype(int))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        resolution = 0.02 if protocol is None else 0.16
        shortest = float(lengths.min()*resolution) if len(lengths) else 0
        short_bin = 'none' if not len(lengths) else '<=0.5s' if shortest <= .5 else '<=2s' if shortest <= 2 else '>2s'
        ratio_bin = '0' if ratio == 0 else '<=0.25' if ratio <= .25 else '<=0.5' if ratio <= .5 else '<1' if ratio < 1 else '1'
        attack = protocol.get(uid, 'unknown') if protocol is not None else 'LPS'
        groups['|'.join((attack, kind, ratio_bin, short_bin))].append(uid)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def choose(groups, split):
    target = math.floor(sum(map(len, groups.values()))*FRACTION + .5)
    allocation = {key: math.floor(len(ids)*FRACTION) for key, ids in groups.items()}
    remainder = target-sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(len(groups[key])*FRACTION-allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    selected = []
    for key, ids in groups.items():
        # Hash ranking is deterministic across Python versions and independent of filesystem order.
        ranked = sorted(ids, key=lambda uid: hashlib.sha256(f'{SEED}|{split}|{uid}'.encode()).digest())
        selected.extend(ranked[:allocation[key]])
    return sorted(selected), allocation


def build(split):
    if split == 'lps':
        original = DATA_ROOT/'LlamaPartialSpoof'
        label_file = original/'segment_labels/cf_seglab_0.02.npy'
        source_audio = original/'R01TTS.0.a'
        target = SUBSET_ROOT/'LlamaPartialSpoof'
        audio_target = target/'cf'
        protocol = None
    else:
        original = DATA_ROOT/'PartialSpoof/database'
        label_file = original/f'segment_labels/{split}_seglab_0.16.npy'
        source_audio = original/split
        target = SUBSET_ROOT/'PartialSpoof/database'
        audio_target = target/split
        protocol_path = original/f'protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt'
        protocol = {parts[1]: parts[-2] for line in protocol_path.read_text().splitlines() if (parts := line.split())}
    labels = np.load(label_file, allow_pickle=True).item()
    files = {path.stem: path for path in source_audio.rglob('*.wav')}
    if set(files) != set(labels):
        raise ValueError(f'Original audio / label coverage mismatch: {split}')
    groups = strata_for(labels, protocol)
    selected, allocation = choose(groups, split)
    manifest_path = SUBSET_ROOT/'manifests'/f'{split}.json'
    if manifest_path.exists():
        previous = read_manifest(split)
        if previous['utterance_ids'] != selected:
            raise ValueError('Refusing to replace an existing fixed subset')
    audio_target.mkdir(parents=True, exist_ok=True)
    for uid in selected:
        destination = audio_target/(uid+'.wav')
        if destination.exists():
            if not os.path.samefile(destination, files[uid]):
                raise ValueError(f'Unexpected non-hardlinked audio: {destination}')
        else:
            os.link(files[uid], destination)
    label_target = target/'segment_labels'/label_file.name
    label_target.parent.mkdir(parents=True, exist_ok=True)
    subset_labels = {uid: labels[uid] for uid in selected}
    if label_target.exists():
        actual = np.load(label_target, allow_pickle=True).item()
        if set(actual) != set(selected) or any(not np.array_equal(actual[u], labels[u]) for u in selected):
            raise ValueError(f'Existing labels differ: {label_target}')
    else:
        np.save(label_target, subset_labels, allow_pickle=True)
    manifest = dict(profile='fixed10', split=split, seed=SEED, requested_fraction=FRACTION,
                    actual_fraction=len(selected)/len(labels), official_count=len(labels), count=len(selected),
                    manifest_sha256=manifest_hash(selected), utterance_ids=selected,
                    source_labels_sha256=hashlib.sha256(label_file.read_bytes()).hexdigest(),
                    source_audio=os.path.relpath(source_audio, DATA_ROOT).replace('\\', '/'),
                    audio_storage='NTFS hard links to full originals; do not edit audio in place',
                    sampling='deterministic hash random ranking; proportional largest-remainder stratification',
                    strata='attack where available, utterance authenticity, fake-frame ratio, shortest fake event duration',
                    strata_counts={key: dict(full=len(ids), selected=allocation[key]) for key, ids in groups.items()},
                    source_grouping='Original train/dev/eval retained; source-related variant map unavailable; no source-disjointness claim',
                    statistical_role='train' if split == 'train' else 'validation' if split == 'dev' else 'test; never tune on these results')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (manifest_path.with_suffix('.txt')).write_text('\n'.join(selected)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('split', 'count', 'official_count', 'manifest_sha256')}, ensure_ascii=False), flush=True)
    return manifest


def prepare():
    manifests = [build(split) for split in ('train', 'dev', 'eval', 'lps')]
    for i, left in enumerate(manifests[:3]):
        for right in manifests[i+1:3]:
            assert not set(left['utterance_ids']) & set(right['utterance_ids'])
    rows = ['数据目录说明（固定 10% 实验）', '',
            '完整原始数据：data/PartialSpoof/database/{train,dev,eval}；data/LlamaPartialSpoof。',
            '精简数据：data/fixed10；种子 20261107，按录音固定抽样，每轮不重新抽。',
            'train=训练；dev=验证/早停/调参；eval=PS 测试；lps=LPS 外部测试。',
            '测试子集只能作测试对照，不能据其指标选择模型或阈值。',
            '完整数据保留用于最终评估。10% 按录音条数取最近整数，不是按帧数/磁盘字节。',
            'fixed10 音频为 NTFS 硬链接，不重复占用音频空间。两处共享文件内容，禁止原地改写音频。',
            '因此本操作减少训练/评估处理量，不释放完整原始数据所占磁盘空间。',
            '尚无源录音衍生关系映射，不声称已验证 source-disjoint；三阶段下游训练仍须另外划分。', '']
    for m in manifests:
        rows.append(f"{m['split']}: {m['count']} / {m['official_count']} ({100*m['actual_fraction']:.4f}%)")
    (DATA_ROOT/'数据集说明_完整与固定10%.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8-sig')
    (SUBSET_ROOT/'README.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8-sig')
    import shutil
    target = CODE_ROOT/'manifests/fixed10'
    target.mkdir(parents=True, exist_ok=True)
    for file in (SUBSET_ROOT/'manifests').glob('*'):
        shutil.copy2(file, target/file.name)


def audit_fixed10():
    from cnn.baseline import BaselineDataset
    from cnn.baseline import ModelConfig
    from tools.reporting import find_results, verify_run
    from sal.pipeline import get_dataset
    rows = []
    for split in ('train', 'dev', 'eval', 'lps'):
        manifest = read_manifest(split)
        portable = json.loads((CODE_ROOT/'manifests/fixed10'/f'{split}.json').read_text(encoding='utf-8'))
        assert manifest == portable
        audio, label_dir = dataset_paths(split, 'fixed10')
        _, full_label_dir = dataset_paths(split, 'full')
        label_name = 'cf_seglab_0.02.npy' if split == 'lps' else f'{split}_seglab_0.16.npy'
        original_file = full_label_dir/label_name
        assert hashlib.sha256(original_file.read_bytes()).hexdigest() == manifest['source_labels_sha256']
        original = np.load(original_file, allow_pickle=True).item()
        labels = np.load(label_dir/label_name, allow_pickle=True).item()
        files = {p.stem: p for p in audio.glob('*.wav')}
        sources = {p.stem: p for p in (DATA_ROOT/manifest['source_audio']).rglob('*.wav')}
        assert sorted(files) == sorted(labels) == manifest['utterance_ids']
        assert len(sources) == manifest['official_count']
        for uid in manifest['utterance_ids']:
            assert np.array_equal(labels[uid], original[uid])
            assert os.path.samefile(files[uid], sources[uid])
        dataset, resolution = get_dataset(split, 'fixed10')
        baseline = BaselineDataset(split, ModelConfig(), data_profile='fixed10')
        assert dataset.utt_ids == baseline.utt_ids == manifest['utterance_ids']
        assert baseline.manifest_sha256 == manifest['manifest_sha256']
        for index in (0, len(dataset)-1):
            uid, waveform, target, n = dataset[index]
            example = baseline[index]
            assert uid == example['uid']
            assert np.array_equal(target.numpy(), 1-example['labels'].numpy())
            assert example['features'].shape[1] == n*round(resolution/.02)
        numeric = [np.asarray(x).astype(int) for x in labels.values()]
        counts = dict(bonafide_frames=sum(int((x == 1).sum()) for x in numeric),
                      spoof_frames=sum(int((x == 0).sum()) for x in numeric))
        assert all(counts.values())
        rows.append(dict(split=split, count=len(dataset), full_count=len(sources),
                         manifest_sha256=manifest['manifest_sha256'], **counts,
                         labels_match_full=True, all_audio_hardlinks_verified=True,
                         sal_cnn_manifest_match=True, sampled_feature_alignment=True))
        print(json.dumps(rows[-1]), flush=True)
    manifests = {s: set(read_manifest(s)['utterance_ids']) for s in ('train','dev','eval')}
    assert all(not manifests[a] & manifests[b] for a,b in (('train','dev'),('train','eval'),('dev','eval')))
    verification = []
    summary = ['SAL 固定 10% 测试子集结果', '',
               '作者公开 SAL-WavLM 权重；复用已完成全量推理的逐条 logits，按固定清单重算指标。',
               '没有重新训练 SAL，没有用测试成绩抽样，没有修改原全量结果。',
               'F1 正类为真实帧（bonafide），与论文官方实现一致。',
               '以下结果仅代表固定测试子集，不代替完整测试集指标。', '']
    for split in ('eval', 'lps'):
        result, filename = find_results('sal', split, 'fixed10')
        if result is None:
            raise ValueError(f'Missing SAL fixed10 result: {split}')
        assert result['manifest_sha256'] == read_manifest(split)['manifest_sha256']
        verified = verify_run(RESULT_ROOT/result['run'])
        verification.append(dict(split=split, file=filename, verification=verified))
        summary.extend([f"{split}: {result['utterance_count']} 条；EER={result['eer_percent']:.6f}%；F1={result['f1_bonafide_percent']:.6f}%",
                        '记录：'+filename, ''])
    summary.extend(['CNN-BiGRU 默认 train/dev/eval/LPS 均为对应 fixed10 子集。',
                    '旧完整数据训练权重不会自动作为 fixed10 权重使用。',
                    'CNN-BiGRU 的实际训练和测试状态请查看对应时间戳结果。'])
    (RESULT_ROOT/'SAL_fixed10_结果说明.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8-sig')
    audit = dict(status='passed', datasets=rows, utterance_ids_disjoint=True,
                 source_group_disjointness='Not verified; provenance mapping unavailable', sal_results=verification,
                 note='Read-only data/score audit; no model training or inference launched')
    (RESULT_ROOT/'fixed10_data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('AUDIT PASSED', flush=True)


def audit_full():
    results = {}
    for split in ('dev', 'eval'):
        root = STORE/'PartialSpoof/database'
        labels = np.load(root/f'segment_labels/{split}_seglab_0.16.npy', allow_pickle=True).item()
        audio_ids = [p.stem for p in (root/split).rglob('*.wav')]
        assert len(audio_ids) == len(set(audio_ids))
        assert set(audio_ids) == set(labels)
        lengths = np.array([len(x) for x in labels.values()])
        results[split] = dict(utterances=len(audio_ids), label_frames=int(lengths.sum()),
            resolution_seconds=0.16, max_label_duration_seconds=float(lengths.max()*.16),
            median_label_duration_seconds=float(np.median(lengths)*.16), exact_id_coverage=True)
        print(split, results[split], flush=True)
    root = STORE/'LlamaPartialSpoof'
    audio_ids = [p.stem for p in root.rglob('*.wav')]
    lines = [x.split() for x in (STORE/'downloads/LlamaPartialSpoof/label_R01TTS.0.a.txt').read_text().splitlines()]
    label_ids = [x[0] for x in lines]
    assert len(audio_ids) == len(set(audio_ids))
    assert len(label_ids) == len(set(label_ids))
    assert set(audio_ids) == set(label_ids)
    durations = np.array([float(x[1]) for x in lines])
    results['lps'] = dict(utterances=len(audio_ids), resolution_seconds=.02,
        label_frames=sum(int(float(x[1])/.02) for x in lines),
        max_duration_seconds=float(durations.max()), median_duration_seconds=float(np.median(durations)),
        exact_id_coverage=True)
    print('lps', results['lps'], flush=True)
    (ROOT/'manifests/evaluation_data_audit.json').write_text(json.dumps(results, indent=2))
    extraction = [json.loads(p.read_text()) for family in ('PartialSpoof','LlamaPartialSpoof')
                  for p in (STORE/family).glob('*.extracted.json')]
    (ROOT/'manifests/extracted_evaluation_data.json').write_text(json.dumps(extraction, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'audit', 'audit-full'), default='audit')
    args = parser.parse_args(argv)
    return {'prepare': prepare, 'audit': audit_fixed10, 'audit-full': audit_full}[args.action]()


if __name__ == '__main__':
    main()
