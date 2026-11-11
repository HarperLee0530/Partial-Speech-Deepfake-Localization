"""Recording-level uncertainty and acoustic-stratified POS error comparisons."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import json

CODE=Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path: sys.path.insert(0,str(CODE))
import numpy as np
from semantic.analysis import path,read_json,read_jsonl,write_json,digest
from semantic.pos_error_ranking import POS_NAMES
from semantic import error_frequency as stats


def run(source):
    records=list(read_jsonl(source/'word_errors.jsonl'))
    metadata=read_json(source/'pos_error_summary.json')
    if metadata['status']!='complete': raise ValueError('Completed word alignment required')
    ids=sorted({r['uid'] for r in records}); positions=sorted({r['pos'] for r in records})
    uid_index={uid:i for i,uid in enumerate(ids)}; pos_index={pos:i for i,pos in enumerate(positions)}
    # Helper mathematics is category-agnostic; this run uses POS only.
    stats.CATEGORIES=tuple(positions)
    result=dict(split='dev',data_profile='fixed10',recordings_with_words=len(ids),
        source_sha256=digest(source/'word_errors.jsonl'),bootstrap_unit='recording',resamples=2000,
        uncertainty_note='Exploratory percentile intervals; not adjusted for comparisons across POS',
        rate_definition='Missed synthetic seconds / synthetic seconds inside words of this POS; analogous for false alarms on genuine seconds',
        strata='Mean SAL-score bins [.1,.25,.5,.75,.9], word-duration bins [.16,.32,.64,1.28], and true-boundary flag',groups={})
    for kind in ('all','confidence_ge_05'):
        values=np.zeros((len(ids),len(positions),4),np.float64)
        strata=np.zeros((len(positions),60,4),np.float64)
        word_counts=Counter(); hit_miss=Counter(); hit_fa=Counter()
        for row in records:
            if kind=='confidence_ge_05' and (row['confidence'] is None or row['confidence']<.5): continue
            i,j=uid_index[row['uid']],pos_index[row['pos']]
            value=np.array([row[k+'_seconds'] for k in ('spoof','miss','bonafide','false_alarm')])
            values[i,j]+=value
            score_bin=int(np.digitize(row['mean_spoof_score'],[.1,.25,.5,.75,.9]))
            duration_bin=int(np.digitize(row['end']-row['start'],[.16,.32,.64,1.28]))
            bucket=(score_bin*5+duration_bin)*2+int(row['crosses_truth_boundary'])
            strata[j,bucket]+=value; word_counts[row['pos']]+=1
            hit_miss[row['pos']]+=int(row['touches_miss']); hit_fa[row['pos']]+=int(row['touches_false_alarm'])
        totals=values.sum(axis=0); rates=stats.rate_metrics(totals); ci=stats.bootstrap(values,2000)
        overall=totals.sum(axis=0); table=[]
        for j,pos in enumerate(positions):
            row=dict(pos=pos,name=POS_NAMES.get(pos,pos),word_count=word_counts[pos],
                touched_miss_words=hit_miss[pos],touched_false_alarm_words=hit_fa[pos],
                spoof_seconds=float(totals[j,0]),missed_seconds=float(totals[j,1]),
                bonafide_seconds=float(totals[j,2]),false_alarm_seconds=float(totals[j,3]),
                recordings_with_miss=int(np.count_nonzero(values[:,j,1]>1e-7)),
                recordings_with_false_alarm=int(np.count_nonzero(values[:,j,3]>1e-7)),
                matched_miss=stats.matched_difference(strata,j,0,1),
                matched_false_alarm=stats.matched_difference(strata,j,2,3))
            for k,name in enumerate(('miss_rate','false_alarm_rate','miss_rate_minus_other_POS','false_alarm_rate_minus_other_POS')):
                row[name]=float(rates[j,k]) if np.isfinite(rates[j,k]) else None
                row[name+'_ci95']=ci[j][k]
            table.append(row)
        result['groups'][kind]=dict(words=sum(word_counts.values()),
            overall_miss_rate=float(overall[1]/overall[0]),overall_false_alarm_rate=float(overall[3]/overall[2]),
            ranking_by_miss_rate=sorted(table,key=lambda r:-(r['miss_rate'] if r['miss_rate'] is not None else -1)),
            ranking_by_false_alarm_rate=sorted(table,key=lambda r:-(r['false_alarm_rate'] if r['false_alarm_rate'] is not None else -1)))
    write_json(source/'pos_error_rates.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); a=p.parse_args()
    r=run(path(a.input))
    print(json.dumps({k:dict(words=v['words'],miss_rate=v['overall_miss_rate'],false_alarm_rate=v['overall_false_alarm_rate']) for k,v in r['groups'].items()},ensure_ascii=False))
