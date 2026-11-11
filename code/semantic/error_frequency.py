"""Descriptive semantic associations with saved SAL errors on fixed10 dev.

No gain network, threshold fitting, or test-derived priors. Errors use the
original SAL 160 ms labels and argmax decision. Touch counts and conditional
duration rates have separate denominators. Recording bootstrap accounts for
within-recording clustering; intervals are exploratory, not multiplicity-adjusted.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import numpy as np
from semantic.analysis import (read_json, read_jsonl, write_json, path, relative, digest,
    database, align, event_ranges, settings)
from semantic.frequency import NAMES
from tools.data_tools import read_manifest, dataset_paths

CATEGORIES = tuple(NAMES)
REGIONS = ('true_spoof', 'false_negative', 'true_bonafide', 'false_positive')


def rate_metrics(values):
    # Final axis: fake duration, missed duration, real duration, false-alarm duration.
    result = np.full(values.shape[:-1]+(4,), np.nan)
    totals = values.sum(axis=-2, keepdims=True)
    for offset, (den, num) in enumerate(((0,1),(2,3))):
        np.divide(values[...,num], values[...,den], out=result[...,offset], where=values[...,den]>0)
        other_den = totals[...,den]-values[...,den]
        other_num = totals[...,num]-values[...,num]
        other_rate = np.full_like(other_den, np.nan)
        np.divide(other_num, other_den, out=other_rate, where=other_den>0)
        result[...,offset+2] = result[...,offset]-other_rate
    return result


def bootstrap(records, repeats=2000):
    rng = np.random.default_rng(20261111)
    samples = np.empty((repeats, len(CATEGORIES), 4))
    for i in range(repeats):
        sample = records[rng.integers(0, len(records), size=len(records))].sum(axis=0)
        samples[i] = rate_metrics(sample)
    output = []
    for c in range(len(CATEGORIES)):
        row = []
        for k in range(4):
            finite = samples[:,c,k][np.isfinite(samples[:,c,k])]
            row.append(dict(low=float(np.quantile(finite,.025)) if len(finite) else None,
                high=float(np.quantile(finite,.975)) if len(finite) else None, valid_resamples=len(finite)))
        output.append(row)
    return output


def matched_difference(strata, category, den, num):
    own = strata[category]; other = strata.sum(axis=0)-own
    valid = (own[:,den]>0) & (other[:,den]>0)
    denominator = float(own[valid,den].sum())
    if not denominator:
        return dict(observed_rate=None, matched_other_rate=None, difference=None, coverage=0.)
    observed = float(own[valid,num].sum()/denominator)
    expected = float(np.sum(own[valid,den]*other[valid,num]/other[valid,den])/denominator)
    return dict(observed_rate=observed, matched_other_rate=expected, difference=observed-expected,
                coverage=min(1.,denominator/max(float(own[:,den].sum()),1e-12)))


def summarize_group(record_values, touched, examples, strata, bootstrap_repeats):
    totals = record_values.sum(axis=0); rates = rate_metrics(totals)
    intervals = bootstrap(record_values, bootstrap_repeats)
    all_totals = totals.sum(axis=0)
    result = dict(words=sum(touched[c]['words'] for c in CATEGORIES),
        touched_false_negative_words=sum(touched[c]['false_negative'] for c in CATEGORIES),
        touched_false_positive_words=sum(touched[c]['false_positive'] for c in CATEGORIES),
        overall_duration_miss_rate=float(all_totals[1]/all_totals[0]) if all_totals[0] else None,
        overall_duration_false_alarm_rate=float(all_totals[3]/all_totals[2]) if all_totals[2] else None,
        categories=[])
    for i, category in enumerate(CATEGORIES):
        c = touched[category]
        row = dict(category=category, name=NAMES[category], words=c['words'],
            words_touching_spoof=c['true_spoof'], words_touching_bonafide=c['true_bonafide'],
            words_touching_miss=c['false_negative'], words_touching_false_alarm=c['false_positive'],
            miss_word_composition=100*c['false_negative']/result['touched_false_negative_words'] if result['touched_false_negative_words'] else 0.,
            false_alarm_word_composition=100*c['false_positive']/result['touched_false_positive_words'] if result['touched_false_positive_words'] else 0.,
            spoof_word_duration_seconds=float(totals[i,0]), missed_word_duration_seconds=float(totals[i,1]),
            bonafide_word_duration_seconds=float(totals[i,2]), false_alarm_word_duration_seconds=float(totals[i,3]),
            recordings_with_miss=int(np.count_nonzero(record_values[:,i,1])),
            recordings_with_false_alarm=int(np.count_nonzero(record_values[:,i,3])),
            frequent_missed_words=examples[category]['false_negative'].most_common(10),
            frequent_false_alarm_words=examples[category]['false_positive'].most_common(10),
            score_duration_boundary_matched_miss=matched_difference(strata,i,0,1),
            score_duration_boundary_matched_false_alarm=matched_difference(strata,i,2,3))
        for j, name in enumerate(('duration_miss_rate','duration_false_alarm_rate','miss_rate_minus_other_categories','false_alarm_rate_minus_other_categories')):
            row[name] = float(rates[i,j]) if np.isfinite(rates[i,j]) else None
            row[name+'_bootstrap_ci95'] = intervals[i][j]
        result['categories'].append(row)
    return result


def run(prediction_run, transcripts, output, bootstrap_repeats=2000):
    meta = read_json(prediction_run/'metrics.json')
    if meta['split'] != 'dev' or meta['resolution_seconds'] != .16 or read_json(prediction_run/'status.json')['status'] != 'complete':
        raise ValueError('Expected completed native .16-second SAL dev predictions')
    records = list(read_jsonl(transcripts)); ids = [r['uid'] for r in records]
    manifest = read_manifest('dev')
    if len(ids) != len(set(ids)) or set(ids) != set(manifest['utterance_ids']):
        raise ValueError('Transcripts must cover exactly fixed10 dev')
    _, label_root = dataset_paths('dev','fixed10')
    official_labels = np.load(label_root/'dev_seglab_0.16.npy', allow_pickle=True).item()
    output.mkdir(parents=True, exist_ok=False)
    kinds = ('all','confidence_ge_05')
    values = {k:np.zeros((len(records),len(CATEGORIES),4),np.float64) for k in kinds}
    touched = {k:{c:Counter() for c in CATEGORIES} for k in kinds}
    examples = {k:{c:{r:Counter() for r in ('false_negative','false_positive')} for c in CATEGORIES} for k in kinds}
    strata = {k:np.zeros((len(CATEGORIES),60,4),np.float64) for k in kinds}
    audits = []; excluded_punctuation = unscored_words = 0
    confusion = np.zeros((2,2),np.int64); config = settings(); db = database(prediction_run)
    try:
        with (output/'word_errors.jsonl').open('x',encoding='utf-8') as words_out, (output/'error_regions.jsonl').open('x',encoding='utf-8') as regions_out:
            for rec_index, record in enumerate(records):
                uid = record['uid']
                saved = db.execute('SELECT logits,labels,seconds FROM predictions WHERE uid=?',(uid,)).fetchone()
                if saved is None:
                    raise ValueError('Missing SAL prediction: '+uid)
                logits = np.frombuffer(saved[0],np.float32).reshape(-1,2)
                labels = np.frombuffer(saved[1],np.uint8)
                if not np.array_equal(labels,np.asarray(official_labels[uid],np.uint8)):
                    raise ValueError('Saved prediction labels differ from official fixed10 dev labels')
                seconds = float(saved[2]); pred = logits.argmax(axis=1)
                np.add.at(confusion,(labels,pred),1)
                selected_words = [w for w in record['words'] if any(ch.isalnum() for ch in w['normalized'])]
                excluded_punctuation += len(record['words'])-len(selected_words)
                aligned, audit = align(selected_words,logits,labels,seconds,.16,config)
                audit['uid'] = uid; audits.append(audit)
                for w in aligned:
                    if w['scored_fraction'] < 1-1e-7:
                        unscored_words += 1; continue
                    c = w['primary_type']; ci = CATEGORIES.index(c)
                    contributions = np.array([w[r+'_fraction']*w['duration'] for r in REGIONS])
                    score_bin = int(np.digitize(w['mean_spoof_score'],[.1,.25,.5,.75,.9]))
                    duration_bin = int(np.digitize(w['duration'],[.16,.32,.64,1.28]))
                    stratum = (score_bin*5+duration_bin)*2+int(w['partial_truth'])
                    for kind in kinds:
                        if kind=='confidence_ge_05' and (w['confidence'] is None or w['confidence'] < .5):
                            continue
                        values[kind][rec_index,ci] += contributions
                        strata[kind][ci,stratum] += contributions
                        touched[kind][c]['words'] += 1
                        for region in REGIONS:
                            if w[region+'_fraction']*w['duration'] > 1e-7:
                                touched[kind][c][region] += 1
                                if region in ('false_negative','false_positive'):
                                    examples[kind][c][region][w['normalized']] += 1
                    w['uid'] = uid; words_out.write(json.dumps(w,ensure_ascii=False,allow_nan=False)+'\n')
                for kind, mask in (('false_negative',(labels==0)&(pred==1)),('false_positive',(labels==1)&(pred==0))):
                    for a,b in event_ranges(mask):
                        start,end = min(float(a)*.16,seconds),min(float(b)*.16,seconds)
                        if end <= start:
                            continue
                        hits = [dict(text=w['text'],start=w['start'],end=w['end'],primary_type=w['primary_type'],confidence=w['confidence'])
                                for w in selected_words if w['start'] < end and w['end'] > start]
                        regions_out.write(json.dumps(dict(uid=uid,audio=record['audio'],error=kind,start=start,end=end,words=hits),ensure_ascii=False)+'\n')
    finally:
        db.close()
    summaries = {k:summarize_group(values[k],touched[k],examples[k],strata[k],bootstrap_repeats) for k in kinds}
    coverage = {key:float(sum(a[key] for a in audits)) for key in audits[0] if key.endswith('_seconds')}
    summary = dict(status='complete',split='dev',data_profile='fixed10',recordings=len(records),
        prediction_run=relative(prediction_run),prediction_metadata_sha256=digest(prediction_run/'metrics.json'),
        prediction_database_sha256=digest(prediction_run/'predictions.sqlite'),
        transcripts=relative(transcripts),transcripts_sha256=digest(transcripts),
        subset_manifest_sha256=manifest['manifest_sha256'],checkpoint_sha256=meta['checkpoint_sha256'],
        resolution_seconds=.16,decision='saved logits argmax; class 0=spoof, ties=spoof; threshold not tuned',
        confusion_true_rows_pred_columns_0_spoof_1_bonafide=confusion.tolist(),
        excluded_punctuation_only_units=excluded_punctuation,unscored_words=unscored_words,
        audio_and_error_coverage=coverage,groups=summaries,
        bootstrap=dict(unit='recording',resamples=bootstrap_repeats,seed=20261111,interval='percentile 95%; exploratory, not multiplicity-adjusted'),
        matched_analysis='Compare other categories within mean SAL-score bins, word-duration bins, and pure/boundary truth; descriptive adjustment, not causal or predictive validation',
        limitations=['Touch counts are not whole-word errors: any positive temporal overlap counts once per word and error kind.',
            'Conditional rates use durations inside ASR words; error time not covered by words is separately reported.',
            'Semantic labels and timestamps are automatic; lower-confidence ASR can confound associations.',
            'Development set diagnostics do not establish held-out gains from semantics or a trained selector.',
            'SAL native 160 ms truth grid differs from the previous 10 ms truth-region table.'])
    write_json(output/'error_summary.json',summary); write_json(output/'recording_coverage.json',audits)
    np.savez_compressed(output/'recording_category_durations.npz',all=values['all'],confidence_ge_05=values['confidence_ge_05'],uids=np.array(ids),categories=np.array(CATEGORIES))
    lines = ['固定 10% 开发集：SAL 错误区域与词语类别', '',
        f'录音：{len(records)} 条；SAL 原生 160 ms 网格，保存分数 argmax 判定，未调阈值。',
        '一个词只要与某类错误区域有正时长交集，就计入该类触及词数；不是整词都判错。',
        '漏检率 = 该类别词内漏掉的伪造时长 / 该类别词内全部伪造时长。',
        '误报率 = 该类别词内被误报的真实时长 / 该类别词内全部真实时长。', '']
    for kind,result in summaries.items():
        lines += [kind, f"词次：{result['words']}；触及漏检：{result['touched_false_negative_words']}；触及误报：{result['touched_false_positive_words']}。",
                  '类别\t触及漏检词数\t触及误报词数\t词内时长漏检率\t词内时长误报率']
        for row in result['categories']:
            def fmt(value): return '无分母' if value is None else f'{100*value:.2f}%'
            lines.append(f"{row['name']}\t{row['words_touching_miss']}\t{row['words_touching_false_alarm']}\t{fmt(row['duration_miss_rate'])}\t{fmt(row['duration_false_alarm_rate'])}")
        lines.append('')
    lines += ['统计关联不能直接证明语义有用；需要独立数据上的语义消融或收益预测验证。',
              '详细分母、按录音 bootstrap 区间、声学分数/时长/边界分层对照见 error_summary.json。']
    (output/'SAL漏检误报词类统计.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--transcripts',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<100: parser.error('Use at least 100 recording bootstrap resamples')
    result=run(path(args.run),path(args.transcripts),path(args.output),args.bootstrap)
    print(json.dumps(dict(recordings=result['recordings'],groups={k:{p:g[p] for p in ('words','touched_false_negative_words','touched_false_positive_words')} for k,g in result['groups'].items()}),ensure_ascii=False,indent=2))
