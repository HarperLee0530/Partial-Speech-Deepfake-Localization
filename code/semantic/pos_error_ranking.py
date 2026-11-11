"""Rank POS by counts of ASR word occurrences touching saved SAL FN/FP regions.

No six-category semantic taxonomy, no networks, no threshold fitting.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0,str(CODE))
import numpy as np
from semantic.analysis import path, read_json, read_jsonl, write_json, database, digest, relative
from tools.data_tools import read_manifest, dataset_paths

POS_NAMES = {'NOUN':'名词','PRON':'代词','VERB':'动词','AUX':'助动词','DET':'限定词（含冠词）',
    'ADP':'介词等附置词','ADJ':'形容词','ADV':'副词','PROPN':'专有名词','PART':'小品词',
    'CCONJ':'并列连词','SCONJ':'从属连词','NUM':'数词','INTJ':'感叹词','PUNCT':'标点标注','X':'其他／未识别词性'}


def run(run_dir, transcripts, output):
    meta=read_json(run_dir/'metrics.json'); status=read_json(run_dir/'status.json')
    if meta['split']!='dev' or meta['resolution_seconds']!=.16 or status['status']!='complete':
        raise ValueError('Require completed native .16-second SAL dev predictions')
    records=list(read_jsonl(transcripts)); ids=[r['uid'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(read_manifest('dev')['utterance_ids']):
        raise ValueError('Expected exactly fixed10 dev')
    _, labels_dir=dataset_paths('dev','fixed10')
    official=np.load(labels_dir/'dev_seglab_0.16.npy',allow_pickle=True).item()
    output.mkdir(parents=True,exist_ok=False)
    counts={kind:Counter() for kind in ('all','miss','false_alarm')}
    reliable={kind:Counter() for kind in counts}
    examples={kind:defaultdict(Counter) for kind in counts}
    seconds_by_pos={kind:defaultdict(float) for kind in ('spoof','bonafide','miss','false_alarm')}
    recordings={kind:set() for kind in ('miss','false_alarm')}
    excluded=unscored=0; confusion=np.zeros((2,2),int)
    db=database(run_dir)
    try:
        with (output/'word_errors.jsonl').open('x',encoding='utf-8') as stream:
            for record in records:
                saved=db.execute('SELECT logits,labels,seconds FROM predictions WHERE uid=?',(record['uid'],)).fetchone()
                if saved is None: raise ValueError('Missing prediction')
                labels=np.frombuffer(saved[1],np.uint8); logits=np.frombuffer(saved[0],np.float32).reshape(-1,2)
                if not np.array_equal(labels,np.asarray(official[record['uid']],np.uint8)):
                    raise ValueError('Saved labels differ from official labels')
                pred=logits.argmax(1); np.add.at(confusion,(labels,pred),1)
                spoof_score=1/(1+np.exp(np.clip(logits[:,1].astype(float)-logits[:,0],-700,700)))
                edges=np.minimum(np.arange(len(labels)+1)*.16,float(saved[2]))
                masks={'spoof':labels==0,'bonafide':labels==1,'miss':(labels==0)&(pred==1),'false_alarm':(labels==1)&(pred==0)}
                for index,w in enumerate(record['words']):
                    if not any(c.isalnum() for c in w['normalized']): excluded+=1; continue
                    overlap=np.maximum(0,np.minimum(edges[1:],w['end'])-np.maximum(edges[:-1],w['start']))
                    if overlap.sum()<w['end']-w['start']-1e-7: unscored+=1; continue
                    durations={k:float(overlap[m].sum()) for k,m in masks.items()}
                    pos=w.get('pos','X'); high=w.get('confidence') is not None and w['confidence']>=.5
                    for kind in counts:
                        if kind=='all' or durations[kind]>1e-7:
                            counts[kind][pos]+=1; reliable[kind][pos]+=int(high); examples[kind][pos][w['normalized']]+=1
                            if kind!='all': recordings[kind].add(record['uid'])
                    for kind in durations: seconds_by_pos[kind][pos]+=durations[kind]
                    item=dict(uid=record['uid'],word_index=index,text=w['text'],normalized=w['normalized'],
                        start=w['start'],end=w['end'],pos=pos,confidence=w.get('confidence'),
                        mean_spoof_score=float(np.dot(overlap,spoof_score)/overlap.sum()),
                        crosses_truth_boundary=durations['spoof']>1e-7 and durations['bonafide']>1e-7,
                        touches_miss=durations['miss']>1e-7,touches_false_alarm=durations['false_alarm']>1e-7,**{k+'_seconds':v for k,v in durations.items()})
                    stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    finally: db.close()
    summary=dict(status='complete',split='dev',data_profile='fixed10',recordings=len(ids),
        scored_word_occurrences=sum(counts['all'].values()),excluded_punctuation_only_units=excluded,unscored_words=unscored,
        resolution_seconds=.16,decision='argmax; 0=spoof; any positive overlap counts once per word per error kind',
        grouping='POS only; no negation/time/person/organization/place/quantity categories',
        prediction_run=relative(run_dir),prediction_metadata_sha256=digest(run_dir/'metrics.json'),
        transcripts_sha256=digest(transcripts),code_sha256=digest(Path(__file__)),
        confusion_true_rows_pred_columns=confusion.tolist(),groups={})
    lines=['固定10%开发集：SAL漏检、误报区间词性排名','','按词性出现次数降序，不使用六个语义类别。',
        '一个词与错误区间有正时长重叠，就计一次；不是该词全长都错。',
        '漏检和误报两组分别计数，同一词可能同时进入两组。','']
    for kind,title in (('miss','漏检区间'),('false_alarm','误报区间'),('all','全体已评分词')):
        ranking=[]
        for rank,(pos,n) in enumerate(counts[kind].most_common(),1):
            row=dict(rank=rank,pos=pos,name=POS_NAMES.get(pos,pos),count=n,
                high_confidence_count=reliable[kind][pos],frequent_words=examples[kind][pos].most_common(10),
                corpus_word_count=counts['all'][pos])
            for err,truth in (('miss','spoof'),('false_alarm','bonafide')):
                den=seconds_by_pos[truth][pos]
                row[err+'_duration_rate']=seconds_by_pos[err][pos]/den if den else None
            ranking.append(row)
        summary['groups'][kind]=dict(name=title,words=sum(counts[kind].values()),ranking=ranking,
            recordings=len(recordings[kind]) if kind!='all' else len(ids))
        lines += [f'{title}：{sum(counts[kind].values())}词次','排名\t词性\t词次\t常见词']
        for r in ranking:
            lines.append(f"{r['rank']}\t{r['name']}\t{r['count']}\t"+'、'.join(w+'('+str(n)+')' for w,n in r['frequent_words'][:6]))
        lines.append('')
    lines += ['词性和时间戳是自动预测。数量排名本身不能证明这些词性更易出错或语义能改善SAL。',
              '如要比较风险，应结合各词性本身的出现量、时长、声学分数和独立验证；本次不训练收益网络。']
    write_json(output/'pos_error_summary.json',summary)
    (output/'SAL漏检误报词性排名.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--transcripts',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=run(path(args.run),path(args.transcripts),path(args.output))
    print(json.dumps({k:v['words'] for k,v in result['groups'].items()},ensure_ascii=False))
