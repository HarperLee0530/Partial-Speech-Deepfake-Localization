"""Align existing ASR words to official 10 ms truth labels, without SAL scores.

Three mutually exclusive fully-covered groups: spoof, bonafide, boundary.
Any word extending beyond actual audio or label coverage is kept separately.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import numpy as np
import soundfile as sf
from project_config import DATA_ROOT
from semantic.analysis import path, read_jsonl, read_json, write_json, digest, relative
from semantic.frequency import NAMES
from tools.data_tools import read_manifest

GROUPS = dict(spoof='完全位于伪造区间的词', bonafide='完全位于真实区间的词',
              boundary='跨越真假边界的词', unaligned='无法完整对齐的词')
EPS = 1e-7


def align_word(start, end, labels, seconds, dt=.01):
    if not all(math.isfinite(x) for x in (start, end, seconds, dt)) or start < 0 or end <= start or seconds <= 0 or dt <= 0:
        raise ValueError('Invalid interval/grid')
    support = min(seconds, len(labels)*dt)
    lo, hi = max(0, math.floor(start/dt)), min(len(labels), math.ceil(min(end, support)/dt))
    if hi <= lo:
        return dict(group='unaligned', spoof_seconds=0., bonafide_seconds=0., unscored_seconds=end-start,
                    scored_fraction=0., spoof_fraction=0., bonafide_fraction=0.)
    index = np.arange(lo, hi)
    overlap = np.maximum(0., np.minimum(np.minimum((index+1)*dt, support), end)-np.maximum(index*dt, start))
    fake = float(overlap[np.asarray(labels[lo:hi]) == 0].sum())
    real = float(overlap[np.asarray(labels[lo:hi]) == 1].sum())
    uncovered = max(0., end-start-fake-real)
    if uncovered > EPS:
        group = 'unaligned'
    elif fake > EPS and real > EPS:
        group = 'boundary'
    elif fake > EPS:
        group = 'spoof'
    elif real > EPS:
        group = 'bonafide'
    else:
        group = 'unaligned'
    return dict(group=group, spoof_seconds=fake, bonafide_seconds=real, unscored_seconds=uncovered,
                scored_fraction=(fake+real)/(end-start), spoof_fraction=fake/(end-start), bonafide_fraction=real/(end-start))


def run(transcripts, output):
    rows = list(read_jsonl(transcripts)); ids = [r['uid'] for r in rows]
    manifest = read_manifest('train')
    if len(ids) != len(set(ids)) or set(ids) != set(manifest['utterance_ids']):
        raise ValueError('Expected exactly the fixed10 training subset, without dev/eval or duplicates')
    label_file = DATA_ROOT/'PartialSpoof/database/segment_labels/train_seglab_0.01.npy'
    source = np.load(label_file, allow_pickle=True).item()
    labels = {uid: np.asarray(source[uid], dtype=np.uint8) for uid in ids}
    del source
    if any(v.ndim != 1 or not len(v) or not np.isin(v, [0, 1]).all() for v in labels.values()):
        raise ValueError('Invalid official labels')
    output.mkdir(parents=True, exist_ok=False)
    counts = {g: Counter() for g in GROUPS}; high = {g: Counter() for g in GROUPS}
    recording_ids = {g: set() for g in GROUPS}
    examples = {g: defaultdict(Counter) for g in GROUPS}
    audits = []; punctuation = 0
    with (output/'word_alignment.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            info = sf.info(path(row['audio'])); seconds = info.frames/info.samplerate
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError('Expected original mono 16 kHz audio')
            lab = labels[row['uid']]; local = Counter()
            for i, word in enumerate(row['words']):
                if not any(ch.isalnum() for ch in word['normalized']):
                    punctuation += 1; continue
                aligned = align_word(word['start'], word['end'], lab, seconds)
                group, category = aligned['group'], word['primary_type']
                counts[group][category] += 1; local[group] += 1
                recording_ids[group].add(row['uid']); examples[group][category][word['normalized']] += 1
                if word.get('confidence') is not None and word['confidence'] >= .5:
                    high[group][category] += 1
                item = dict(uid=row['uid'], audio=row['audio'], word_index=i, text=word['text'], normalized=word['normalized'],
                    start=word['start'], end=word['end'], primary_type=category, pos=word.get('pos'),
                    confidence=word.get('confidence'), **aligned)
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
            audits.append(dict(uid=row['uid'], audio_seconds=seconds, label_coverage_seconds=min(seconds, len(lab)*.01),
                unlabeled_tail_seconds=max(0., seconds-len(lab)*.01), word_groups=dict(local)))
    total = sum(sum(c.values()) for c in counts.values()); groups = {}
    for group in GROUPS:
        n, reliable_n = sum(counts[group].values()), sum(high[group].values())
        categories = []
        for category in sorted(NAMES, key=lambda k: (-counts[group][k], k)):
            categories.append(dict(category=category, name=NAMES[category], count=counts[group][category],
                percent_within_group=100*counts[group][category]/n if n else 0.,
                confidence_ge_05_count=high[group][category],
                confidence_ge_05_percent_within_group=100*high[group][category]/reliable_n if reliable_n else 0.,
                frequent_words=examples[group][category].most_common(15)))
        groups[group] = dict(name=GROUPS[group], words=n, percent_all_words=100*n/total if total else 0.,
            recordings=len(recording_ids[group]), confidence_ge_05_words=reliable_n, categories=categories)
    summary = dict(status='complete', split='train', data_profile='fixed10', recordings=len(rows),
        word_occurrences=total, excluded_punctuation_only_units=punctuation, resolution_seconds=.01,
        label_convention='0=spoof; 1=bonafide; official truth, not SAL predictions',
        rule='Entire scored word in one class = pure; positive overlap with both = boundary; uncovered time = unaligned',
        positive_duration_tolerance_seconds=EPS, transcripts=relative(transcripts), transcripts_sha256=digest(transcripts),
        labels=relative(label_file), labels_sha256=digest(label_file), subset_manifest_sha256=manifest['manifest_sha256'],
        groups=groups, code_sha256=digest(Path(__file__)),
        limitations=['ASR timing and semantic labels are predictions, not human verification.',
            'Boundary classification is at official 10 ms grid resolution, not sample-accurate edit boundaries.',
            'Counts are ASR word occurrences; multiword entity spans contribute multiple words.',
            'Spoof labels indicate synthetic/converted source, not every splice or inserted segment.',
            'No SAL errors, missed-detection rates, or attack category preferences are inferred.'])
    write_json(output/'region_summary.json', summary); write_json(output/'recording_coverage.json', audits)
    lines = ['固定 10% 训练集：按官方真假区间划分的词类统计', '',
        f'范围：{len(rows)} 条录音，{total} 次有效词出现；排除纯标点 {punctuation} 个。',
        '标签：官方 10 ms 网格，0=伪造，1=真实。没有使用 SAL 预测结果。',
        '归类：词的完整时间范围只覆盖一种标签才归入该类；同时覆盖两类则归为跨边界。',
        '越出录音或标签覆盖范围的词另列，不强行归入三组。', '']
    for key, info in groups.items():
        lines += [f"{info['name']}：{info['words']} 次，占全部词 {info['percent_all_words']:.2f}%；涉及 {info['recordings']} 条录音。",
                  '类别\t次数\t占本组词比例\t常见词']
        for r in info['categories']:
            examples_text = '、'.join(f'{w}({n})' for w, n in r['frequent_words'][:8])
            lines.append(f"{r['name']}\t{r['count']}\t{r['percent_within_group']:.2f}%\t{examples_text}")
        lines.append('')
    lines += ['注意：占比的分母分别是各组总词数，不能仅比较绝对次数判断哪类更偏向伪造。',
              '这是基于自动转写、时间戳和语义标注的描述性统计；未经过人工准确率验证。']
    (output/'真假区间词类统计.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(path(args.transcripts), path(args.output))
    print(json.dumps(dict(recordings=result['recordings'], words=result['word_occurrences'],
        groups={g: v['words'] for g,v in result['groups'].items()}), ensure_ascii=False, indent=2))
