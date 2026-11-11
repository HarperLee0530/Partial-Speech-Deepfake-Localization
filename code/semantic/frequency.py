"""Resumable local ASR + NER + frequency report; no SAL/gain training.

Uses the shared fixed10 manifest exactly. Transcripts/annotations are appended
per recording. A complete report is written only after exact ID coverage checks.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from semantic.analysis import (annotate, settings, validate_record, read_jsonl,
                               digest, relative, path, write_json)
from tools.data_tools import read_manifest, dataset_paths

NAMES = dict(negation='否定', time='时间', person='人物', place='地点',
             organization='组织', quantity='数量', other='其他／未归入上述类别')


def summarize(records, metadata, output):
    categories = Counter(); reliable = Counter(); pos = Counter(); all_words = Counter()
    category_words = defaultdict(Counter); category_recordings = defaultdict(set)
    empty = low = missing = total = punctuation = 0
    for record in records:
        empty += not bool(record['words'])
        for w in record['words']:
            if not any(ch.isalnum() for ch in w['normalized']):
                punctuation += 1
                continue
            total += 1
            category = w['primary_type']; token = w['normalized']
            confidence = w.get('confidence')
            categories[category] += 1; category_words[category][token] += 1
            category_recordings[category].add(record['uid'])
            pos[w.get('pos','UNKNOWN')] += 1; all_words[token] += 1
            if confidence is None:
                missing += 1
            elif confidence < .5:
                low += 1
            else:
                reliable[category] += 1
    recognized = total-categories['other']; high = sum(reliable.values())
    rows = []
    for category in sorted(NAMES, key=lambda k: (-categories[k], k)):
        count = categories[category]
        rows.append(dict(category=category, name=NAMES[category], count=count,
            percent_all_words=100*count/total if total else 0,
            percent_classified_words=100*count/recognized if recognized and category != 'other' else None,
            recordings=len(category_recordings[category]),
            percent_recordings=100*len(category_recordings[category])/len(records) if records else 0,
            confidence_ge_05_count=reliable[category], confidence_ge_05_percent=100*reliable[category]/high if high else 0,
            frequent_words=category_words[category].most_common(15)))
    summary = dict(**metadata, recordings=len(records), total_word_occurrences=total,
        unique_normalized_words=len(all_words), empty_transcripts=empty, low_confidence_words=low,
        excluded_punctuation_only_units=punctuation,
        missing_confidence_words=missing, confidence_ge_05_words=high,
        classified_word_occurrences=recognized, categories=rows, pos_counts=pos.most_common(),
        most_frequent_words=all_words.most_common(40),
        count_unit='ASR word-timestamp occurrence containing a letter/digit; not unique word or whole named entity',
        limitations=['ASR/NER predictions, not manually verified transcripts or entity labels.',
                    'other means outside the six selected categories or unrecognized, not meaningless.',
                    f"Only the fixed10 {metadata['split']} subset; no inference about the full corpus or fake-word prevalence.",
                    'Primary categories are mutually exclusive under configured priority; multiword entities count by ASR word.',
                    'ASR confidence is a model score, not a calibrated accuracy guarantee.'])
    write_json(output/'frequency_summary.json', summary)
    split_name = {'train': '训练集', 'dev': '开发集', 'eval': '测试集'}[metadata['split']]
    lines = [f'固定 10% {split_name}：语义类别频率统计', '',
        f"录音：{len(records)} 条；识别词语出现次数：{total}；不同规范化词：{len(all_words)}。",
        f"空转写：{empty} 条；词置信度低于 0.5：{low} 次；置信度缺失：{missing} 次。",
        f"仅含标点、不计入词频的输出单元：{punctuation} 个。",
        '模型：Whisper base / faster-whisper（CPU int8，beam=5）；spaCy en_core_web_sm + 规则。',
        '口径：每个带时间戳的词出现一次计一次。占比以全部识别词为分母。', '',
        '类别\t次数\t占全部词比例\t涉及录音数\t常见词（次数）']
    for r in rows:
        examples = '、'.join(f'{w}({n})' for w,n in r['frequent_words'][:8])
        lines.append(f"{r['name']}\t{r['count']}\t{r['percent_all_words']:.2f}%\t{r['recordings']}\t{examples}")
    if total:
        lines += ['', '词性分布：']+[f'{k}: {v} ({100*v/total:.2f}%)' for k,v in pos.most_common()]
    lines += ['', '注意：这是自动转写及标注的描述性统计，不是人工真值；“其他”不代表没有语义。',
              f'仅统计固定 10% {split_name}，不混入其他划分或此前试跑。',
              '频率高不代表更容易被伪造，也不代表复查收益高。']
    (output/'统计结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    return summary


def run(output, asr_model, ner_model='en_core_web_sm', workers=2, split='train'):
    import spacy
    from faster_whisper import WhisperModel
    if workers not in (1, 2):
        raise ValueError('Use one or two CPU workers to coexist with SAL training')
    config = settings(); manifest = read_manifest(split)
    ids = manifest['utterance_ids']
    reference = json.loads((CODE/'manifests/fixed10'/f'{split}.json').read_text(encoding='utf-8'))
    if manifest != reference:
        raise ValueError('Shared fixed10 manifest mismatch')
    root, _ = dataset_paths(split, 'fixed10')
    audio = {p.stem: p for p in root.rglob('*.wav')}
    if set(audio) != set(ids):
        raise ValueError('Audio IDs differ from the fixed subset manifest')
    output.mkdir(parents=True, exist_ok=True)
    raw_file, annotated_file = output/'transcripts.jsonl', output/'annotated.jsonl'
    metadata = dict(split=split, data_profile='fixed10', expected_recordings=len(ids),
        subset_manifest_sha256=manifest['manifest_sha256'], subset_seed=manifest['seed'],
        asr_model=relative(asr_model), asr_model_sha256=digest(asr_model/'model.bin'),
        ner_model=ner_model, compute_type='int8', device='cpu', beam_size=5, language='en',
        annotation_config=config, annotation_code_sha256=digest(CODE/'semantic/analysis.py'),
        versions={p: importlib.metadata.version(p) for p in ('faster-whisper','ctranslate2','spacy','en-core-web-sm')})
    if (output/'run_config.json').exists():
        if json.loads((output/'run_config.json').read_text(encoding='utf-8')) != metadata:
            raise ValueError('Cannot resume with a different model, subset, or configuration')
    else:
        write_json(output/'run_config.json', metadata)
    raw = {r['uid']: r for r in read_jsonl(raw_file)} if raw_file.exists() else {}
    records = {r['uid']: r for r in read_jsonl(annotated_file)} if annotated_file.exists() else {}
    if not set(records) <= set(raw) <= set(ids):
        raise ValueError('Unexpected IDs in existing outputs')
    nlp = spacy.load(ner_model)
    model = WhisperModel(str(asr_model), device='cpu', compute_type='int8', cpu_threads=2,
                         num_workers=workers, local_files_only=True)
    pending = [uid for uid in ids if uid not in raw]
    start = time.perf_counter(); initial = len(raw)

    def transcribe(uid):
        began = time.perf_counter()
        segments, info = model.transcribe(str(audio[uid]), language='en', word_timestamps=True,
                                          vad_filter=False, beam_size=5)
        words, discarded = [], 0
        for segment in segments:
            for w in segment.words or []:
                if w.end <= w.start or not w.word.strip():
                    discarded += 1; continue
                words.append(dict(text=w.word.strip(), start=w.start, end=w.end, confidence=w.probability))
        row = dict(uid=uid, audio=relative(audio[uid]), language=info.language, words=words,
                   audio_seconds=info.duration, asr_seconds=time.perf_counter()-began,
                   discarded_zero_duration_words=discarded)
        validate_record(row)
        return row

    def progress(state='running', error=None):
        elapsed = time.perf_counter()-start; completed = len(raw)-initial
        write_json(output/'status.json', dict(status=state, completed=len(records), transcribed=len(raw),
            total=len(ids), elapsed_this_session_seconds=elapsed,
            estimated_remaining_seconds=(len(ids)-len(raw))*elapsed/completed if completed else None,
            error=error))

    try:
        with raw_file.open('a', encoding='utf-8') as raw_stream, annotated_file.open('a', encoding='utf-8') as annotated_stream:
            def save_annotation(row):
                item = annotate(row, config, nlp); item['ner_model'] = ner_model
                annotated_stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n'); annotated_stream.flush()
                records[row['uid']] = item
            for uid in ids:
                if uid in raw and uid not in records:
                    save_annotation(raw[uid])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for row in pool.map(transcribe, pending):
                    raw_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); raw_stream.flush()
                    raw[row['uid']] = row; save_annotation(row)
                    if len(records) % 25 == 0 or len(records) == len(ids):
                        progress(); print(f'Semantic {len(records)}/{len(ids)}', flush=True)
        if set(records) != set(ids):
            raise ValueError('Incomplete transcript/annotation coverage')
        summary = summarize([records[uid] for uid in ids], metadata, output)
        progress('complete')
        print(json.dumps(dict(output=relative(output), recordings=summary['recordings'], words=summary['total_word_occurrences']), ensure_ascii=False), flush=True)
    except BaseException as exc:
        progress('failed', str(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--asr-model', required=True)
    parser.add_argument('--ner-model', default='en_core_web_sm'); parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--split', choices=['train', 'dev', 'eval'], default='train')
    args = parser.parse_args()
    run(path(args.output), path(args.asr_model), args.ner_model, args.workers, args.split)
