"""Auditable narrow disambiguation for completed frequency reports.

Preserves original ASR and annotations. No manual/guessed transcript edits.
Only removes a lexicon-only month tag from an AUX/VERB or conflicting named
entity (e.g. modal may). NER DATE/TIME takes precedence over this rule.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from semantic.analysis import path, read_json, read_jsonl, digest, write_json
from semantic.frequency import summarize


def review_record(record, priority):
    result = copy.deepcopy(record); changes = []
    for w in result['words']:
        sources = set(w['tag_sources']); types = set(w['types'])
        if (w['normalized'] in ('may', 'march') and 'lexicon:time' in sources
                and not sources.intersection({'ner:DATE','ner:TIME'})
                and (w.get('pos') in ('AUX', 'VERB') or types.intersection({'person','place','organization'}))):
            old = w['primary_type']
            sources.discard('lexicon:time'); sources.add('qa:ambiguous_month_non_temporal')
            types.discard('time'); types.discard('other')
            w['types'] = sorted(types) or ['other']; w['tag_sources'] = sorted(sources)
            w['primary_type'] = next(t for t in priority if t in types or t == 'other')
            changes.append(dict(uid=record['uid'], text=w['text'], start=w['start'], end=w['end'],
                pos=w.get('pos'), previous=old, revised=w['primary_type'], reason='Month lexicon conflicts with context; no DATE/TIME entity'))
    return result, changes


def finalize(source, output=None):
    status = read_json(source/'status.json')
    if status['status'] != 'complete' or status['completed'] != status['total']:
        raise ValueError('Only finalize a completed full subset')
    metadata = read_json(source/'run_config.json')
    priority = metadata['annotation_config']['type_priority']
    records = []; changes = []
    for row in read_jsonl(source/'annotated.jsonl'):
        revised, delta = review_record(row, priority); records.append(revised); changes.extend(delta)
    ids = [r['uid'] for r in records]
    from tools.data_tools import read_manifest
    if len(ids) != len(set(ids)) or set(ids) != set(read_manifest(metadata['split'])['utterance_ids']):
        raise ValueError('Final recording IDs differ from the chosen fixed10 split')
    output = output or source/'reviewed'; output.mkdir(exist_ok=False)
    with (output/'annotated.jsonl').open('x', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    write_json(output/'annotation_adjustments.json', changes)
    metadata.update(review_rules_sha256=digest(Path(__file__)),
        summary_code_sha256=digest(Path(__file__).with_name('frequency.py')),
        source_annotations_sha256=digest(source/'annotated.jsonl'),
        rule_adjusted_word_occurrences=len(changes),
        review_scope='Automatic narrow ambiguity correction; not manual transcription/annotation verification')
    summary = summarize(records, metadata, output)
    write_json(output/'status.json', dict(status='complete', recordings=len(ids), adjustments=len(changes)))
    return dict(output=str(output), words=summary['total_word_occurrences'], adjustments=len(changes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--input', required=True)
    parser.add_argument('--output')
    args = parser.parse_args(); print(json.dumps(finalize(path(args.input), path(args.output) if args.output else None), ensure_ascii=False, indent=2))
