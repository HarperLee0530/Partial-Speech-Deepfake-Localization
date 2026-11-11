"""Full-train statistics with leave-recording/group-out training features.

This subtracts counts, not model weights: the frozen detector was trained on
these recordings. It does not claim out-of-fold detector predictions.
"""
import csv
import json

from project_config import DATA_ROOT

STATUSES = ('tp', 'fn', 'fp', 'tn')


def official_train_ids():
    import numpy as np
    labels = DATA_ROOT/'PartialSpoof/database/segment_labels/train_seglab_0.16.npy'
    return set(np.load(labels, allow_pickle=True).item())


def empty_counts():
    return dict(word={}, primary_type={})


def increment(counts, row):
    for dimension, value in (('word', row['normalized']), ('primary_type', row['primary_type'])):
        entry = counts[dimension].setdefault(value, [0, 0, 0, 0])
        entry[STATUSES.index(row['status'])] += 1


def freeze_full(report, output, source_groups=None):
    from semantic.analysis import read_json, read_jsonl, relative, digest
    meta = read_json(report/'summary.json')
    if meta.get('status') != 'complete' or meta.get('split') != 'train' or meta.get('data_profile') != 'full':
        raise ValueError('Full-train priors require completed full/train diagnostics; never dev/eval/LPS')
    ids = read_json(report/'analyzed_uids.json')
    if len(ids) != len(set(ids)) or set(ids) != official_train_ids():
        raise ValueError('Full-train report must cover every official train recording exactly once (including empty transcripts)')
    if meta.get('missing_transcript_recordings') != 0 or meta.get('prediction_recordings') != len(ids):
        raise ValueError('Full-train predictions/transcripts must have complete coverage')
    id_set = set(ids)
    groups = {uid: uid for uid in ids}
    if source_groups is not None:
        groups = {}
        for row in read_jsonl(source_groups):
            uid, group = row['uid'], row['source_group']
            if uid not in id_set or uid in groups or not isinstance(group, str) or not group:
                raise ValueError('Source-group map requires unique official train IDs and nonempty groups')
            if row.get('split', 'train') != 'train':
                raise ValueError('Only training recordings may contribute statistics')
            groups[uid] = group
        if set(groups) != set(ids):
            raise ValueError('Source-group map must cover the full train split')
    totals, contributions = empty_counts(), {group: empty_counts() for group in set(groups.values())}
    with (report/'word_diagnostics.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['uid'] not in groups:
                raise ValueError('Word diagnostics contain a non-training ID')
            if row['status'] in STATUSES:
                increment(totals, row)
                increment(contributions[groups[row['uid']]], row)
    global_counts = [sum(c[i] for c in totals['primary_type'].values()) for i in range(4)]
    if sum(global_counts) == 0:
        raise ValueError('No eligible training words')
    artifact = dict(schema_version=2, source_split='train', data_profile='full',
                    protocol='full_train_leave_group_out', source_uids=sorted(ids),
                    source_groups=groups, group_contributions=contributions, counts=totals,
                    global_counts=global_counts, config=meta['config'], language=meta['language'],
                    source_report=relative(report), source_summary_sha256=digest(report/'summary.json'),
                    word_diagnostics_sha256=digest(report/'word_diagnostics.csv'),
                    source_groups_sha256=digest(source_groups) if source_groups else None,
                    exclusion_unit='source_group' if source_groups else 'recording_uid',
                    detector_exposure='in-sample frozen detector; not out-of-fold',
                    limitation='Count exclusion removes self-target statistics, not detector/modifier training optimism')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, ensure_ascii=False, allow_nan=False)
    return dict(frozen_statistics=relative(output), training_recordings=len(ids),
                exclusion_unit=artifact['exclusion_unit'])


def reduced_priors(artifact, words, excluded_group=None):
    """Subtract only needed lexical/type cells and global totals: O(words), not O(vocabulary)."""
    removed = artifact['group_contributions'].get(excluded_group, empty_counts())
    global_removed = [sum(c[i] for c in removed['primary_type'].values()) for i in range(4)]

    def subtract(a, b):
        value = [x-y for x, y in zip(a, b)]
        if any(x < 0 for x in value):
            raise ValueError('Invalid negative leave-group-out counts')
        return value

    def smooth(c, parent):
        tp, fn, fp, tn = c
        fake, real = tp+fn, tn+fp
        pairs = ((fake, fake+real), (fn, fake), (fp, real))
        strength = artifact['config']['smoothing_strength']
        return dict(n=fake+real, denominators=[d for _, d in pairs],
                    rates=[(v+strength*p)/(d+strength) for (v, d), p in zip(pairs, parent)],
                    reliable=[d >= artifact['config']['minimum_count'] for _, d in pairs])

    global_feature = smooth(subtract(artifact['global_counts'], global_removed), [.5, .5, .5])
    result = dict(global_statistics=global_feature, word={}, primary_type={})
    for dimension, key in (('word', 'normalized'), ('primary_type', 'primary_type')):
        for value in {w[key] for w in words}:
            c = subtract(artifact['counts'][dimension].get(value, [0]*4),
                         removed[dimension].get(value, [0]*4))
            if sum(c):
                result[dimension][value] = smooth(c, global_feature['rates'])
    return result


def export_features(transcripts, frozen, output, split):
    from semantic.analysis import read_json, read_jsonl, digest, annotate, lookup, relative
    artifact = read_json(frozen)
    if artifact.get('protocol') != 'full_train_leave_group_out' or split not in ('train', 'dev', 'eval', 'lps', 'inference'):
        raise ValueError('Expected full-train frozen artifact and a supported feature split')
    groups = artifact['source_groups']
    frozen_digest, seen = digest(frozen), set()
    with output.open('x', encoding='utf-8') as stream:
        for record in read_jsonl(transcripts):
            uid = record['uid']
            if uid in seen:
                raise ValueError('Duplicate feature recording: '+uid)
            seen.add(uid)
            if split == 'train':
                if uid not in groups:
                    raise ValueError('Training features require an official training recording')
                excluded = groups[uid]
            else:
                if uid in groups or record.get('source_group') in artifact['group_contributions']:
                    raise ValueError('Validation/test/inference recording overlaps fitting recordings/source groups')
                excluded = None
            if 'source_group' in record and uid in groups and record['source_group'] != groups[uid]:
                raise ValueError('Transcript source group differs from frozen map')
            record = annotate(record, artifact['config'])
            if record['language'] != artifact['language']:
                raise ValueError('Prior language differs from transcript language')
            priors = reduced_priors(artifact, record['words'], excluded)
            words = [dict(text=w['text'], start=w['start'], end=w['end'], primary_type=w['primary_type'],
                          **lookup(w, priors, artifact['config']['minimum_word_confidence'])) for w in record['words']]
            # Never forward target labels, prediction outcomes, or measured gains.
            stream.write(json.dumps(dict(uid=uid, words=words, feature_split=split,
                                         excluded_group=excluded, frozen_sha256=frozen_digest), ensure_ascii=False)+'\n')
    return dict(feature_records=len(seen), output=relative(output), feature_split=split)
