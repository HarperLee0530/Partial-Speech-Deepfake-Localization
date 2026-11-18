"""Paired fixed-model comparison; no tuning, retraining or threshold selection."""
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent.parent
OLD=Path('D:/SAL_精简实验/result')

def read(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as con:
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        return con.execute('SELECT uid,logits,labels,seconds FROM predictions ORDER BY ordinal').fetchall()

def spans(mask,dt,seconds):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    s=np.minimum(edges.reshape(-1,2)*dt,seconds)
    return s[s[:,1]>s[:,0]+1e-8]

def counts(row,dt):
    _,blob,truth,seconds=row
    logits=np.frombuffer(blob,dtype=np.float32).reshape(-1,2)
    lab=np.frombuffer(truth,dtype=np.uint8)
    assert len(logits)==len(lab) and np.isfinite(logits).all()
    pred=logits.argmax(1)
    cm=np.bincount(lab*2+pred,minlength=4)
    t=spans(lab==0,dt,seconds);p=spans(pred==0,dt,seconds)
    inter=np.maximum(0,np.minimum(t[:,None,1],p[None,:,1])-np.maximum(t[:,None,0],p[None,:,0]))
    union=(t[:,1]-t[:,0])[:,None]+(p[:,1]-p[:,0])[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    hit=0
    if len(t) and len(p):
        valid=iou>=.5-1e-8
        r,c=linear_sum_assignment(-np.where(valid,min(len(t),len(p))+1+iou,0))
        hit=int(valid[r,c].sum())
    return np.r_[cm,hit,len(t),len(p)]

def metrics(total):
    ff,fr,rf,rr,hit,nt,npred=np.moveaxis(total,-1,0)
    return {'frame_f1_spoof_percent':100*2*ff/np.maximum(2*ff+fr+rf,1),
            'event_f1_percent':100*2*hit/np.maximum(nt+npred,1),
            'accuracy_percent':100*(ff+rr)/np.maximum(ff+fr+rf+rr,1)}

def main():
    frozen=json.loads((ROOT/'result/logmel_sal_20261117/frozen_selection.json').read_text(encoding='utf8'))
    output={}
    table=[]
    for split,dt in [('eval',.16),('lps',.02)]:
        old=OLD/f'sal_reduced_{split}_20261113'
        new=ROOT/'result'/f'logmel_sal_20261117_{split}'
        a=read(old/'predictions.sqlite');b=read(new/'predictions.sqlite')
        ma=json.loads((old/'metrics.json').read_text(encoding='utf8'))
        mb=json.loads((new/'metrics.json').read_text(encoding='utf8'))
        assert mb['checkpoint_sha256']==frozen['sha256']
        assert ma['manifest_sha256']==mb['manifest_sha256']
        assert len(a)==len(b)==ma['utterance_count']==mb['utterance_count']
        for aa,bb in zip(a,b):
            assert aa[0]==bb[0] and aa[2]==bb[2] and abs(aa[3]-bb[3])<1e-8
        ca=np.stack([counts(r,dt) for r in a]);cb=np.stack([counts(r,dt) for r in b])
        va=metrics(ca.sum(0));vb=metrics(cb.sum(0))
        assert abs(va['frame_f1_spoof_percent']-ma['f1_spoof_percent'])<1e-4
        assert abs(vb['frame_f1_spoof_percent']-mb['f1_spoof_percent'])<1e-4
        rng=np.random.default_rng(20261117)
        delta={k:[] for k in va}
        for _ in range(2000):
            ii=rng.integers(0,len(a),len(a))
            ra=metrics(ca[ii].sum(0));rb=metrics(cb[ii].sum(0))
            for k in delta:delta[k].append(rb[k]-ra[k])
        va['eer_percent']=ma['eer_percent'];vb['eer_percent']=mb['eer_percent']
        output[split]=dict(recordings=len(a),grid_seconds=dt,wavlm=va,logmel=vb,
            delta_logmel_minus_wavlm_pp={k:float(vb[k]-va[k]) for k in va},
            paired_recording_bootstrap95_pp={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in delta.items()},
            identical_uids_labels_durations=True,confusion_old=ca.sum(0)[:4].tolist(),
            confusion_new=cb.sum(0)[:4].tolist())
        for name,m in [('WavLM + SAL',va),('LogMel + SAL',vb)]:
            table.append(dict(dataset=split,model=name,recordings=len(a),**m))
    (ROOT/'result/comparison.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    with (ROOT/'result/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['SAL前端替换对照：WavLM与原方案Log-Mel','',
        '普通特征直接复用原cnn.LogMel：16kHz、80维、窗长400、FFT512、步长320。',
        '接入LayerNorm(80)+Linear(80,1024)，随后使用原SAL的1024维Conformer及8类/2类输出头。',
        '保留SPL、修正后的CSM两轮、RawBoost、Adam、StepLR、seed888、batch8。',
        '训练5076条，开发2484条；仅开发EER选模，固定argmax测试，不按测试集调层/阈值/参数。',
        f"新模型实际训练轮数：{frozen['epochs_run']}；开发集最佳EER：{frozen['best_dev_eer_percent']:.4f}%。",
        '旧模型实际40轮、选第25轮；两组同用最少40/最多60轮及patience10规则，实际停止轮数可能不同。','']
    for split,item in output.items():
        lines.append(f"{split.upper()}：{item['recordings']}条，原生网格{item['grid_seconds']}秒")
        for name,key in [('WavLM + SAL','wavlm'),('Log-Mel + SAL','logmel')]:
            v=item[key]
            lines.append(f"  {name}: EER {v['eer_percent']:.4f}%，伪造帧F1 {v['frame_f1_spoof_percent']:.4f}%，事件F1 {v['event_f1_percent']:.4f}%。")
        lines.append('  Log-Mel减WavLM差值（百分点）：'+str(item['delta_logmel_minus_wavlm_pp']))
        lines.append('  按录音配对bootstrap 95%区间：'+str(item['paired_recording_bootstrap95_pp']))
    lines+=['','解释限制：',
        '本实验衡量整套WavLM前端（预训练表示、结构及原训练方式）被普通特征替换后的差异；不能把SAL成绩按百分比分摊给前端和后端。',
        '同一训练配方并不保证对随机初始化Log-Mel输入映射最优，尤其原1e-5学习率与每10轮乘0.1衰减可能限制从零学习。',
        '这是单随机种子，bootstrap只覆盖固定模型的录音抽样波动，不包含重训波动。',
        '训练第8轮发生进程退出，从第7轮末完整检查点续训，保留模型/优化器/调度器/选模状态；旧检查点未保存随机数状态，续训重新设seed888，并非逐位等价于不中断训练。详见resume记录。',
        '没有加入同设置的“Log-Mel + 无SPL/CSM后端”组，因此不能单凭本实验量化SAL后续技巧的独立收益。',
        'PS使用160ms标签、LPS使用20ms标签；不要直接与主方案20ms事件F1混为同一评测。',
        '旧WavLM结果已被查看、LPS有历史探索，本次是固定方案的历史对照，不宣称全新盲测。',
        '已核对配对UID、标签、音频时长一致和混淆矩阵；未重新训练WavLM参照组。']
    (ROOT/'result/comparison.txt').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
