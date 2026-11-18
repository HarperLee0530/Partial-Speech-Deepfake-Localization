"""Frozen reduced datasets: prepare, verify and package; never train or delete data.

``train`` is an alias for train_detector only. Native labels use 1=real, 0=fake.
Preparation reads official data; ordinary training/auditing needs only reduced data.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from project_config import CODE_ROOT, DATA_ROOT

PROFILES = ('reduced',)
ROLES = ('train_detector', 'train_edit', 'train_review', 'dev', 'eval', 'lps')
TRAIN_ROLES = ROLES[:3]
FRACTIONS = dict(train_detector=.20, train_edit=.05, train_review=.05, dev=.10, eval=.05, lps=.05)
SEED = 20261111
SUBSET_ROOT = DATA_ROOT / 'reduced'
_LPS_ID = re.compile(r'(.+?)_(\d+)_(\d+)_(\d+)_(\d+)')


def canonical_split(split):
    role = 'train_detector' if split == 'train' else split
    if role not in ROLES:
        raise ValueError(f'Unknown partition: {split}; choose {ROLES}')
    return role


def label_split(split):
    role = canonical_split(split)
    return 'train' if role in TRAIN_ROLES else 'cf' if role == 'lps' else role


def manifest_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Refusing to overwrite different frozen data: {path}')
    if not path.exists():
        path.write_text(text, encoding='utf-8')


def read_manifest(split):
    role = canonical_split(split)
    path = SUBSET_ROOT / 'manifests' / f'{role}.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    ids = value['utterance_ids']
    if (value['profile'] != 'reduced' or value['split'] != role
            or ids != sorted(set(ids)) or value['count'] != len(ids)
            or value['manifest_sha256'] != manifest_hash(ids)):
        raise ValueError(f'Invalid frozen manifest: {path}')
    portable = CODE_ROOT / 'manifests' / 'reduced' / path.name
    if portable.exists() and json.loads(portable.read_text(encoding='utf-8')) != value:
        raise ValueError(f'Runtime and code manifests disagree: {role}')
    return value


def profile_info(split, profile='reduced'):
    if profile not in PROFILES:
        raise ValueError(f'Only reduced data is enabled; requested {profile!r}')
    role = canonical_split(split)
    value = read_manifest(role)
    return dict(data_profile='reduced', partition_role=role,
                evaluation_role='training' if role in TRAIN_ROLES else 'validation' if role == 'dev' else 'test',
                dataset_scope='frozen course-project reduced official splits',
                subset_seed=value['seed'], subset_fraction=value['actual_fraction'],
                official_utterances=value['official_count'],
                subset_manifest_sha256=value['manifest_sha256'],
                subset_manifest=f'reduced/manifests/{role}.json')


def dataset_paths(split, profile='reduced'):
    if profile not in PROFILES:
        raise ValueError(f'Only reduced data is enabled; requested {profile!r}')
    root = SUBSET_ROOT / canonical_split(split)
    return root / 'audio', root / 'segment_labels'


def label_path(split, resolution=None):
    role = canonical_split(split)
    resolution = resolution or (.02 if role == 'lps' else .16)
    return dataset_paths(role)[1] / f'{label_split(role)}_seglab_{resolution:.2f}.npy'


def validate_reduced(split, deep=False):
    """Validate labels and all file sizes; deep additionally verifies audio hashes."""
    role = canonical_split(split)
    manifest = read_manifest(role)
    audio, labels_dir = dataset_paths(role)
    files = {p.stem: p for p in audio.glob('*.wav')}
    if sorted(files) != manifest['utterance_ids']:
        raise ValueError(f'Audio IDs differ from frozen {role} manifest')
    for name, expected in manifest['label_files'].items():
        path = labels_dir / name
        if file_hash(path) != expected['sha256']:
            raise ValueError(f'Labels changed: {path}')
        values = np.load(path, allow_pickle=True).item()
        if sorted(values) != manifest['utterance_ids']:
            raise ValueError(f'Label IDs differ: {path}')
        if any(x.dtype != np.int8 or not np.isin(x, (0, 1)).all() for x in values.values()):
            raise ValueError(f'Invalid binary numeric labels: {path}')
    for uid, path in files.items():
        expected = manifest['audio_files'][uid]
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'Audio size changed: {path}')
        if deep and file_hash(path) != expected['sha256']:
            raise ValueError(f'Audio hash changed: {path}')
    return dict(split=role, count=len(files), manifest_sha256=manifest['manifest_sha256'],
                audio_bytes=manifest['audio_bytes'], label_hashes_verified=True,
                audio_hashes_verified=bool(deep))


def audit_search(verbose=True):
    result = {role: validate_reduced(role) for role in ('train_detector', 'dev')}
    if verbose:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def audit(deep=False):
    result = {role: validate_reduced(role, deep=deep) for role in ROLES}
    manifests = {role: read_manifest(role) for role in ROLES}
    for i, left in enumerate(ROLES[:-1]):
        for right in ROLES[i + 1:-1]:
            if set(manifests[left]['utterance_ids']) & set(manifests[right]['utterance_ids']):
                raise ValueError(f'Recording overlap: {left} / {right}')
    result['ps_recording_ids_disjoint'] = True
    result['ps_source_disjointness_verified'] = False
    result['total_audio_bytes'] = sum(x['audio_bytes'] for x in manifests.values())
    result['dev_previously_used_for_model_development'] = True
    return result


def _rank(scope, value):
    return hashlib.sha256(f'{SEED}|{scope}|{value}'.encode()).digest()


def _features(values, resolution):
    labels = np.asarray(values).astype(np.int8)
    if not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError('Empty or non-binary labels')
    fake = labels == 0
    ratio = float(fake.mean())
    kind = 'genuine' if ratio == 0 else 'fully_fake' if ratio == 1 else 'partial'
    edges = np.diff(np.r_[False, fake, False].astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    shortest = float(lengths.min() * resolution) if len(lengths) else 0
    return dict(kind=kind,
                fake_ratio='0' if ratio == 0 else '<=.25' if ratio <= .25 else '<=.5' if ratio <= .5 else '<1' if ratio < 1 else '1',
                shortest_fake='none' if not len(lengths) else '<=.16s' if shortest <= .16 + 1e-8 else '<=.5s' if shortest <= .5 else '<=2s' if shortest <= 2 else '>2s',
                duration='<=2s' if len(labels) * resolution <= 2 else '<=5s' if len(labels) * resolution <= 5 else '<=10s' if len(labels) * resolution <= 10 else '>10s',
                label_seconds=len(labels) * resolution, fake_seconds=int(fake.sum()) * resolution)


def _allocate(groups, target):
    """Largest-remainder allocation with exact total and no replacement."""
    size = sum(len(ids) for ids in groups.values())
    if not 0 <= target <= size:
        raise ValueError('Requested count exceeds available records')
    expected = {key: len(ids) * target / size for key, ids in groups.items()}
    counts = {key: math.floor(value) for key, value in expected.items()}
    ranked = sorted(groups, key=lambda key: (-(expected[key] - counts[key]), key))
    for key in ranked[:target - sum(counts.values())]:
        counts[key] += 1
    return counts


def _choose_records(metadata, available, target, role):
    groups = defaultdict(list)
    for uid in sorted(available):
        row = metadata[uid]
        groups['|'.join(str(row[key]) for key in ('speaker', 'attack', 'kind', 'fake_ratio', 'shortest_fake', 'duration'))].append(uid)
    allocation = _allocate(groups, target)
    return sorted(uid for key, ids in groups.items()
                  for uid in sorted(ids, key=lambda u: _rank(role, u))[:allocation[key]])


def _summaries(metadata, ids):
    rows = [metadata[uid] for uid in ids]
    fields = ('kind', 'attack', 'speaker', 'fake_ratio', 'shortest_fake', 'duration')
    return {**{key: dict(sorted(Counter(str(x[key]) for x in rows).items())) for key in fields},
            'label_seconds': sum(x['label_seconds'] for x in rows),
            'fake_seconds': sum(x['fake_seconds'] for x in rows)}


def _read_source(source_root, official):
    if official == 'lps':
        base = source_root / 'LlamaPartialSpoof'
        audio = base / 'R01TTS.0.a'
        label = base / 'segment_labels/cf_seglab_0.02.npy'
        protocol = {}
    else:
        base = source_root / 'PartialSpoof/database'
        audio = base / official / 'con_wav'
        label = base / f'segment_labels/{official}_seglab_0.16.npy'
        protocol_file = base / f'protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{official}.trl.txt'
        protocol = {p[1]: p for line in protocol_file.read_text().splitlines() if (p := line.split())}
    labels = np.load(label, allow_pickle=True).item()
    files = {p.stem: p for p in audio.glob('*.wav')}
    if set(files) != set(labels) or (protocol and set(protocol) != set(labels)):
        raise ValueError(f'Official audio/label/protocol mismatch: {official}')
    metadata = {}
    for uid, values in labels.items():
        row = _features(values, .02 if official == 'lps' else .16)
        if official == 'lps':
            match = _LPS_ID.fullmatch(uid)
            if match is None:
                raise ValueError(f'Unrecognized LPS source ID; do not silently split it: {uid}')
            prefix, speaker, chapter, utterance, clip = match.groups()
            row.update(attack=prefix, speaker=speaker,
                       source_group=f'{speaker}_{chapter}_{utterance}')
        else:
            row.update(speaker=protocol[uid][0], attack=protocol[uid][-2], source_group=None)
        metadata[uid] = row
    return labels, files, metadata, label


def _choose_lps(metadata):
    """Keep inferred parent-utterance groups intact; quotas are approximate."""
    groups = defaultdict(list)
    for uid, row in metadata.items():
        groups[row['source_group']].append(uid)
    strata = defaultdict(list)
    for group, ids in groups.items():
        prefixes = '+'.join(sorted({metadata[uid]['attack'] for uid in ids}))
        typical = float(np.median([metadata[uid]['label_seconds'] for uid in ids]))
        duration = '<=2s' if typical <= 2 else '<=5s' if typical <= 5 else '<=10s' if typical <= 10 else '>10s'
        size = '1' if len(ids) == 1 else '<=13' if len(ids) <= 13 else '<=39' if len(ids) <= 39 else '>39'
        strata[prefixes + '|' + duration + '|' + size].append(group)
    selected = []
    for stratum, group_ids in sorted(strata.items()):
        target = sum(len(groups[g]) for g in group_ids) * FRACTIONS['lps']
        count = 0
        for group in sorted(group_ids, key=lambda g: _rank('lps', g)):
            n = len(groups[group])
            if abs(count + n - target) < abs(count - target):
                selected.extend(groups[group])
                count += n
    selected_groups = {metadata[uid]['source_group'] for uid in selected}
    assert all(set(ids) <= set(selected) for key, ids in groups.items() if key in selected_groups)
    return sorted(selected), len(groups), len(selected_groups)


def _materialize(role, selected, labels, files, metadata, source_label, output_root, provenance):
    role_root = output_root / role
    audio_root = role_root / 'audio'
    labels_root = role_root / 'segment_labels'
    audio_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    if any(p.stem not in selected for p in audio_root.glob('*.wav')):
        raise ValueError(f'Unexpected audio already in destination: {audio_root}')
    for uid in selected:
        target = audio_root / files[uid].name
        if target.exists():
            if target.stat().st_size != files[uid].stat().st_size or file_hash(target) != file_hash(files[uid]):
                raise ValueError(f'Existing destination differs: {target}')
        else:
            try:
                os.link(files[uid], target)
            except OSError:
                shutil.copy2(files[uid], target)
    native_name = source_label.name
    subset = {uid: np.asarray(labels[uid]).astype(np.int8) for uid in selected}
    target_label = labels_root / native_name
    if target_label.exists():
        existing = np.load(target_label, allow_pickle=True).item()
        if sorted(existing) != selected or any(not np.array_equal(existing[uid], subset[uid]) for uid in selected):
            raise ValueError(f'Existing labels differ: {target_label}')
    else:
        np.save(target_label, subset, allow_pickle=True)
    def audio_record(uid):
        path = audio_root / f'{uid}.wav'
        return uid, dict(bytes=path.stat().st_size, sha256=file_hash(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        audio_files = dict(pool.map(audio_record, selected))
    official = 'train' if role in TRAIN_ROLES else role
    manifest = dict(schema_version=1, profile='reduced', split=role, official_split=official,
                    seed=20261107 if role == 'dev' else SEED,
                    requested_fraction=FRACTIONS[role], actual_fraction=len(selected) / len(labels),
                    official_count=len(labels), count=len(selected), utterance_ids=selected,
                    manifest_sha256=manifest_hash(selected), source_labels_sha256=file_hash(source_label),
                    label_resolution_seconds=.02 if role == 'lps' else .16,
                    label_convention='1=real/bonafide; 0=fake/spoof; native temporal grid unchanged',
                    label_files={native_name: dict(sha256=file_hash(target_label), bytes=target_label.stat().st_size)},
                    audio_files=audio_files, audio_bytes=sum(x['bytes'] for x in audio_files.values()),
                    sampling=('reuse original frozen fixed10 dev IDs; original attack/authenticity/fake-ratio/shortest-event strata; no model predictions'
                              if role == 'dev' else 'hash-ranked inferred parent-source groups; approximate quotas by variant-prefix composition, median duration and group size; no speaker quota; no model predictions'
                              if role == 'lps' else 'fixed hash ranking; proportional speaker/attack/authenticity/fake-ratio/shortest-event/duration strata; no model predictions'),
                    provenance_status='heuristic_grouped' if role == 'lps' else 'recording_only',
                    source_grouping=provenance,
                    source_group_by_uid={uid: metadata[uid]['source_group'] for uid in selected} if role == 'lps' else {},
                    speaker_by_uid={uid: metadata[uid]['speaker'] for uid in selected},
                    full_distribution=_summaries(metadata, labels), selected_distribution=_summaries(metadata, selected),
                    statistical_role='detector fit' if role == 'train_detector' else 'edit gain fit' if role == 'train_edit' else 'review gain fit; reserved' if role == 'train_review' else 'validation; already used during development' if role == 'dev' else 'test; never tune on these results',
                    runtime_dependencies='Only this role audio, labels and frozen manifests; full originals are not required',
                    audio_storage='Independent file names; may share NTFS hard-link content with originals. Never edit audio in place.')
    _write_json(output_root / 'manifests' / f'{role}.json', manifest)
    _write_json(CODE_ROOT / 'manifests/reduced' / f'{role}.json', manifest)
    print(json.dumps(dict(split=role, count=len(selected), audio_bytes=manifest['audio_bytes'], manifest_sha256=manifest['manifest_sha256']), ensure_ascii=False), flush=True)
    return manifest


def archive_source_metadata(source_root, manifests):
    """Keep compact original text evidence, without full waveforms or label arrays."""
    destination = CODE_ROOT / 'manifests/reduced/source_metadata'
    destination.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root)
    files = [(source_root / 'PartialSpoof/database/protocols/PartialSpoof_LA_cm_protocols' /
              f'PartialSpoof.LA.cm.{split}.trl.txt') for split in ('train', 'dev', 'eval')]
    files.append(source_root / 'downloads/LlamaPartialSpoof/label_R01TTS.0.a.txt')
    if any(not path.is_file() for path in files):
        raise FileNotFoundError('Required source text evidence is missing')
    archive_path = destination / 'original_protocols_and_lps_labels.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.name)
    expected_files = {path.name: file_hash(path) for path in files}
    with zipfile.ZipFile(archive_path) as archive:
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    if actual != expected_files:
        raise ValueError('Archived original protocol evidence differs')
    provenance = dict(source_text_sha256=expected_files,
                      archive_sha256=file_hash(archive_path),
                      full_label_sha256={role: value['source_labels_sha256'] for role, value in manifests.items()},
                      full_distributions={role: value['full_distribution'] for role, value in manifests.items()},
                      source_relation_limitations='PS mapping unavailable; LPS parent groups inferred from filenames only',
                      selection_uses_model_predictions=False,
                      complete_original_audio_required_at_runtime=False)
    _write_json(destination / 'source_audit.json', provenance)


def prepare(source_root, output_root):
    """Materialize a new dataset. Never overwrite a different frozen selection."""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    manifests = {}
    ps_provenance = ('Official PS splits retained; train roles have disjoint recording IDs and proportional speaker strata. '
                     'Original source/derived-recording map unavailable. Source-recording disjointness is NOT verified; '
                     'speakers intentionally occur in all three training roles.')
    for official in ('train', 'dev', 'eval', 'lps'):
        labels, files, metadata, source_label = _read_source(source_root, official)
        if official == 'train':
            available = set(labels)
            for role in TRAIN_ROLES:
                target = math.floor(len(labels) * FRACTIONS[role] + .5)
                selected = _choose_records(metadata, available, target, role)
                available.difference_update(selected)
                manifests[role] = _materialize(role, selected, labels, files, metadata, source_label, output_root, ps_provenance)
        else:
            if official == 'dev':
                legacy_path = source_root / 'fixed10/manifests/dev.json'
                legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
                selected = legacy['utterance_ids']
                if (selected != sorted(set(selected)) or manifest_hash(selected) != legacy['manifest_sha256']
                        or not set(selected) <= set(labels) or len(selected) != math.floor(len(labels) * .10 + .5)):
                    raise ValueError('Invalid existing frozen dev selection')
                if file_hash(source_label) != legacy['source_labels_sha256']:
                    raise ValueError('Existing dev manifest refers to changed official labels')
                provenance = ps_provenance + ' Dev IDs exactly preserve the 20261107 fixed10 validation selection; this is not an untouched new validation set.'
            elif official == 'lps':
                selected, total_groups, selected_groups = _choose_lps(metadata)
                provenance = (f'Heuristic parent source group = first three of final four numeric filename components '
                              f'(speaker_chapter_original_utterance); fourth treated as clip index. '
                              f'Filename evidence only, not independently verified provenance. All variants/clip indices kept together; '
                              f'{selected_groups}/{total_groups} inferred groups selected; count approximates 5%.')
            else:
                selected = _choose_records(metadata, set(labels), math.floor(len(labels) * FRACTIONS[official] + .5), official)
                provenance = ps_provenance
            manifests[official] = _materialize(official, selected, labels, files, metadata, source_label, output_root, provenance)
    for i, left in enumerate(ROLES[:-1]):
        for right in ROLES[i + 1:-1]:
            if set(manifests[left]['utterance_ids']) & set(manifests[right]['utterance_ids']):
                raise ValueError(f'Unexpected ID overlap: {left}, {right}')
    summary = {role: {k: m[k] for k in ('count', 'official_count', 'actual_fraction', 'manifest_sha256', 'audio_bytes')}
               for role, m in manifests.items()}
    _write_json(output_root / 'summary.json', summary)
    readme = ('缩减数据：固定清单，不在每次运行重新抽样。train 仅是 train_detector 的别名。\n'
              'train_detector=PS train 20%；train_edit=5%；train_review=5%保留；三角色录音ID互斥。\n'
              'dev=原有固定10%验证集；eval=5%；LPS≈5%，按文件名推断的来源组整组抽样。\n'
              'PS缺原始录音衍生映射，不声称source-disjoint。LPS来源分组为启发式，不是已验证的官方映射。\n'
              '已使用dev只能继续用于开发，不能称为全新独立验证集。eval/LPS禁止调参。\n'
              '只保留原生160ms PS/20ms LPS标签，数值转int8但不改时间网格。\n'
              '完整原始数据不是运行依赖；音频可能使用硬链接，禁止原地修改音频。\n')
    (output_root / 'README.txt').write_text(readme, encoding='utf-8')
    archive_source_metadata(source_root, manifests)
    return summary


def package(output_dir):
    """Create one ZIP per role, plus metadata ZIP, all rooted at data/reduced."""
    audit(deep=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = dict(format='ZIP_STORED; extract every ZIP into the project root', files=[])
    items = [(role, sorted((SUBSET_ROOT / role).rglob('*'))) for role in ROLES]
    items.append(('metadata', sorted((SUBSET_ROOT / 'manifests').glob('*.json')) +
                  [SUBSET_ROOT / 'summary.json', SUBSET_ROOT / 'README.txt']))
    for role, paths in items:
        destination = output_dir / f'reduced_{role}.zip'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite package: {destination}')
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in paths:
                if path.is_file():
                    archive.write(path, 'data/reduced/' + path.relative_to(SUBSET_ROOT).as_posix())
        if destination.stat().st_size >= 2_000_000_000:
            raise ValueError(f'Package is too large for a single release asset: {destination}')
        index['files'].append(dict(name=destination.name, bytes=destination.stat().st_size, sha256=file_hash(destination)))
        print(json.dumps(index['files'][-1]), flush=True)
    _write_json(output_dir / 'reduced_package.json', index)
    return index


def main():
    global SUBSET_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'audit', 'package'), default='audit')
    parser.add_argument('--source-data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=SUBSET_ROOT)
    parser.add_argument('--package-dir', type=Path, default=CODE_ROOT.parent / 'release')
    parser.add_argument('--deep', action='store_true')
    args = parser.parse_args()
    SUBSET_ROOT = args.output_root.resolve()
    if args.action == 'prepare':
        result = prepare(args.source_data_root, args.output_root)
    elif args.action == 'package':
        result = package(args.package_dir)
    else:
        result = audit(deep=args.deep)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
