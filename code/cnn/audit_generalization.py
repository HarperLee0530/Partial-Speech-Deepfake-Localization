"""Read-only CPU audit of full CNN dev/test discrepancy; saves diagnostics only.

No training, GPU inference, threshold fitting or edits to experiment outputs.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np
import sqlite3
import torch
from cnn import baseline as p
from project_config import DATA_ROOT, RESULT_ROOT, artifact_relative

TRAIN_RUN = 'cnn_bigru_train_20261109_203335_588602'
TEST_RUN = 'cnn_bigru_eval_full_20261110_212857_051038'
SAL_RUN = 'sal_wavlm_eval_full_20261106_115611_680558'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def eer_numpy(scores, labels):
    """Independent ROC calculation, grouping tied scores into one threshold."""
    if not len(labels) or len(np.unique(labels)) < 2:
        return None
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order].astype(np.int64)
    endpoints = np.r_[np.flatnonzero(np.diff(s)), len(y)-1]
    tp = np.cumsum(y)[endpoints]
    fp = endpoints+1-tp
    fpr = np.r_[0., fp/(len(y)-y.sum())]
    fnr = np.r_[1., 1-tp/y.sum()]
    index = int(np.argmin(abs(fpr-fnr)))
    return float(100*(fpr[index]+fnr[index])/2)


def frame_summary(scores, labels):
    fake = labels == 0
    pred_fake = scores <= 0  # SAL binary-head argmax with ties=spoof
    fn, fp = int((fake & ~pred_fake).sum()), int((~fake & pred_fake).sum())
    return dict(frames=len(labels), spoof_frames=int(fake.sum()), real_frames=int((~fake).sum()),
                missed_spoof_frames=fn, false_alarm_real_frames=fp,
                miss_percent=100*fn/int(fake.sum()) if fake.any() else None,
                false_alarm_percent=100*fp/int((~fake).sum()) if (~fake).any() else None,
                eer_numpy_percent=eer_numpy(scores, labels))


def collect_split(split):
    base = DATA_ROOT/'PartialSpoof/database'
    label_file = base/f'segment_labels/{split}_seglab_0.16.npy'
    labels = np.load(label_file, allow_pickle=True).item()
    labels = {uid:np.asarray(y, dtype=np.uint8) for uid,y in labels.items()}
    protocol = {}
    for line in (base/f'protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt').read_text().splitlines():
        fields = line.split()
        if fields[1] in protocol:
            raise ValueError('Duplicate protocol ID')
        protocol[fields[1]] = dict(speaker=fields[0], attack=fields[-2], label=fields[-1])
    files = sorted((base/split).rglob('*.wav'))
    audio_ids = [file.stem for file in files]
    if len(audio_ids) != len(set(audio_ids)) or set(audio_ids) != set(labels) or set(protocol) != set(labels):
        raise ValueError('Audio/protocol/label coverage mismatch: '+split)
    counts = Counter()
    frames = Counter()
    lengths, event_lengths = [], []
    for uid,y in labels.items():
        if not len(y) or not np.isin(y, [0,1]).all():
            raise ValueError('Invalid labels')
        fake = y == 0
        scope = 'genuine' if not fake.any() else 'fully_fake' if fake.all() else 'partial'
        counts[scope] += 1
        frames['spoof'] += int(fake.sum())
        frames['real'] += int((~fake).sum())
        lengths.append(len(y)*.16)
        edge = np.diff(np.r_[False,fake,False].astype(int))
        event_lengths.extend((np.flatnonzero(edge == -1)-np.flatnonzero(edge == 1))*.16)
    result = dict(recordings=len(labels), scopes=dict(counts), frames=dict(frames),
                  spoof_frame_percent=100*frames['spoof']/sum(frames.values()),
                  utterance_duration_quantiles=np.quantile(lengths,[.1,.5,.9,.99]).tolist(),
                  recordings_longer_than_4s_percent=100*float(np.mean(np.array(lengths)>4)),
                  spoof_event_duration_quantiles=np.quantile(event_lengths,[.1,.5,.9,.99]).tolist(),
                  attack_codes=dict(Counter(x['attack'] for x in protocol.values())),
                  speakers=len({x['speaker'] for x in protocol.values()}),
                  manifest_sha256=p.manifest_hash(audio_ids), label_sha256=p.sha256(label_file),
                  exact_audio_label_protocol_coverage=True)
    print('SPLIT_AUDIT',split,json.dumps(result),flush=True)
    return labels, protocol, {file.stem:file for file in files}, result


def scan_database(run, labels, protocol):
    groups = defaultdict(list)
    score_chunks, label_chunks, ids, per_record = [], [], [], []
    boundary_counts, interior_counts = Counter(), Counter()
    with closing(sqlite3.connect((RESULT_ROOT/run/'predictions.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
        for uid,logit_blob,label_blob,seconds in db.execute('SELECT uid,logits,labels,seconds FROM predictions ORDER BY ordinal'):
            y = np.frombuffer(label_blob,dtype=np.uint8)
            logits = np.frombuffer(logit_blob,dtype=np.float32).reshape(-1,2)
            if uid not in labels or not np.array_equal(y,labels[uid]) or len(y) != len(logits) or not np.isfinite(logits).all():
                raise ValueError('Stored prediction/official label mismatch: '+uid)
            margin = logits[:,1]-logits[:,0]
            index = len(ids)
            ids.append(uid); score_chunks.append(margin); label_chunks.append(y)
            fake = y == 0
            scope = 'genuine' if not fake.any() else 'fully_fake' if fake.all() else 'partial'
            duration_group = '<=2s' if seconds<=2 else '2-4s' if seconds<=4 else '4-8s' if seconds<=8 else '>8s'
            groups['scope:'+scope].append(index)
            groups['duration:'+duration_group].append(index)
            groups['speaker:'+protocol[uid]['speaker']].append(index)
            edge_mask = np.zeros(len(y),dtype=bool)
            transitions = np.flatnonzero(y[1:] != y[:-1])+1
            edge_mask[transitions] = True
            edge_mask[transitions-1] = True
            for mask, accumulator in ((edge_mask,boundary_counts),(~edge_mask,interior_counts)):
                accumulator['spoof'] += int((fake & mask).sum())
                accumulator['real'] += int((~fake & mask).sum())
                accumulator['miss'] += int((fake & (margin>0) & mask).sum())
                accumulator['false_alarm'] += int((~fake & (margin<=0) & mask).sum())
            per_record.append(dict(uid=uid,scope=scope,speaker=protocol[uid]['speaker'],seconds=seconds,
                                   frames=len(y),spoof_frames=int(fake.sum()),
                                   missed_spoof_frames=int((fake & (margin>0)).sum()),
                                   false_alarm_real_frames=int((~fake & (margin<=0)).sum())))
    if len(ids)!=len(set(ids)) or set(ids)!=set(labels):
        raise ValueError('Stored predictions do not exactly cover official eval IDs')
    s,y = np.concatenate(score_chunks),np.concatenate(label_chunks)
    summary = frame_summary(s,y)
    summary.update(recordings=len(ids), all_saved_labels_match_official=True,
                   manifest_sha256=p.manifest_hash(ids), boundary=dict(boundary_counts),interior=dict(interior_counts))
    rows = []
    for key,indices in sorted(groups.items()):
        gs=np.concatenate([score_chunks[i] for i in indices])
        gy=np.concatenate([label_chunks[i] for i in indices])
        rows.append(dict(group=key,recordings=len(indices),**frame_summary(gs,gy)))
    return summary,rows,per_record,s,y


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=RESULT_ROOT/('cnn_generalization_audit_'+p.datetime.now(p.ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    data={}
    for split in ('train','dev','eval'):
        data[split]=collect_split(split)
    coverage={s:row[3] for s,row in data.items()}
    overlaps={}
    for a,b in (('train','dev'),('train','eval'),('dev','eval')):
        overlaps[a+'_'+b]=dict(uid_overlap=len(set(data[a][0]) & set(data[b][0])),
                               speaker_overlap=len({v['speaker'] for v in data[a][1].values()} & {v['speaker'] for v in data[b][1].values()}))
    train=read(RESULT_ROOT/(TRAIN_RUN+'.json'))
    history=read(RESULT_ROOT/TRAIN_RUN/'history.json')
    best=next(row for row in history if row['epoch']==train['best_epoch'])
    weights=(CODE/train['checkpoint']).resolve()
    assert p.sha256(weights)==train['checkpoint_sha256']
    checkpoint=torch.load(weights,map_location='cpu')
    assert checkpoint['epoch']==train['best_epoch'] and checkpoint['train_run']==TRAIN_RUN
    assert checkpoint['validation_metrics']==best['metrics']
    assert train['dev_manifest_sha256']==coverage['dev']['manifest_sha256']
    assert train['train_manifest_sha256']==coverage['train']['manifest_sha256']
    result=dict(splits=coverage,overlaps=overlaps,checkpoint=dict(epoch=checkpoint['epoch'],sha256=train['checkpoint_sha256'],
                matches_training_record=True,matches_best_history_metrics=True),
                best_dev_metrics=best['metrics'],history=[dict(epoch=r['epoch'],train_loss=r['train_loss'],
                   validation_loss=r['loss'],dev_eer_percent=r['metrics']['eer_percent']) for r in history],
                limitations=['Full dev scores were not saved; no full dev re-inference in this audit.',
                             'CON is a concatenation protocol code, not a detailed synthesis attack identifier.',
                             'Source-utterance and waveform duplicate relationships are not established by disjoint output IDs.'])
    for name,run in (('cnn',TEST_RUN),('sal',SAL_RUN)):
        summary,groups,records,scores,labels=scan_database(run,data['eval'][0],data['eval'][1])
        saved=read(RESULT_ROOT/run/'metrics.json')
        summary['saved_eer_percent']=saved['eer_percent']
        assert abs(summary['eer_numpy_percent']-saved['eer_percent'])<.001
        if name=='cnn':
            assert saved['checkpoint_sha256']==train['checkpoint_sha256']
            # Recompute saved test scores through the TRAINING-validation metric function.
            summary['via_validation_metric']=p.official_metrics(torch.from_numpy(-scores),torch.from_numpy(1-labels))
            assert abs(summary['via_validation_metric']['eer_percent']-saved['eer_percent'])<.001
        result[name]=summary
        for suffix,rows in (('groups',groups),('recordings',records)):
            with (output/f'{name}_{suffix}.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('MODEL_AUDIT',name,json.dumps(summary),flush=True)
    result['output']=artifact_relative(output)
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AUDIT_REPORT',output,flush=True)


if __name__=='__main__':
    main()
