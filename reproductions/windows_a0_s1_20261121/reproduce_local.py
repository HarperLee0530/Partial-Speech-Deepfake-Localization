"""Windows relocation for the published four-model S1 protocol, pinned to 50d994ba."""
import csv
import io
import os
import platform
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
REPO=ROOT/'repo'
EX=REPO/'experiments'
CODE=EX/'wavlm_ablation_ab_20261119/code'
EVENT=EX/'event_aware_c'
SUP=EX/'cross_resolution_supervision'
PKG=REPO/'reproductions/s1_20261121'
CACHE=ROOT/'cache/fixed_ps'
DATA=ROOT.parents[1]/'partial_speech_project/data'
sys.path[:0]=[str(CODE),str(EVENT),str(SUP)]
import numpy as np
import torch
from cnn_ps.io import read_json,save_json,sha256,manifest_hash
from cnn_ps.train_wavlm_multilayer import make_adapter,model_config,setup,ARMS
from cnn_ps.wavlm_multilayer import MultiLayerData,LayerFusionCNN
from cnn_ps.train_dual import infer
from cnn_ps.finetune import save_scores
from cnn_ps.report_native20 import evaluate_report
from cnn_ps.metrics20 import decision_mask
from cnn_ps.diagnose_events import diagnose_events
from data_audit import check_fixed

NAMES=('A0_666','S1_666','A0_667','S1_667')
FROZEN=read_json(PKG/'evaluation/FROZEN_MANIFEST_FORMAL.json')


def status(phase,**fields):
    save_json(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**fields))


def published(name):
    return PKG/'artifacts'/(name+'_formal')


def trained(name):
    branch='event_aware_c/plan_c_666' if name=='A0_666' else 'cross_resolution_supervision'
    return ROOT/'experiments'/branch/(name+'_formal')


def audit():
    status('source_and_input_audit')
    for line in (PKG/'PACKAGE_SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        digest,filename=line.split(None,1)
        assert sha256(PKG/filename)==digest,filename
    subprocess.run([sys.executable,'-X','utf8',str(PKG/'verify_package.py')],check=True)
    import test_supervision
    for name in sorted(x for x in dir(test_supervision) if x.startswith('test_')):
        getattr(test_supervision,name)()
    ids=check_fixed(CODE/'manifests/ps_fixed_ids.json')
    published_audit=read_json(PKG/'data_protocol/manifests/audit.json')
    archive=DATA/'downloads/database_segment_labels_v1.2.tar.gz'
    archive_hash=sha256(archive)
    assert archive_hash==published_audit['archive_sha256']
    model_hash=sha256(DATA/'models/wavlm-base-plus/pytorch_model.bin')
    assert model_hash==read_json(published('A0_666')/'protocol.json')['train_ssl']['model_sha256']
    def verify_audio(item):
        rel,meta=item
        if sha256(DATA/'reduced'/rel)!=meta['sha256']:
            raise ValueError(rel)
    with ThreadPoolExecutor(6) as pool:
        for _ in pool.map(verify_audio,published_audit['audio'].items()): pass
    labels_checked={}
    cache_diff={}
    with tarfile.open(archive,'r:gz') as tar:
        for split in ('train','dev','eval'):
            ref=CACHE/'reference'/split
            meta=read_json(ref/'complete.json')
            uids=[r['uid'] for r in ids[split]]
            assert meta['uids']==uids
            for grid,key in [('0.02','labels20'),('0.16','labels160')]:
                member=f'database/segment_labels/{split}_seglab_{grid}.npy'
                official=np.load(io.BytesIO(tar.extractfile(member).read()),allow_pickle=True).item()
                arr=np.load(ref/(key+'.npy'),mmap_mode='r')
                for i,uid in enumerate(uids):
                    off=meta['offsets'][i]
                    off=off if key=='labels20' else off//8
                    n=meta['lengths20' if key=='labels20' else 'lengths160'][i]
                    y=np.asarray(official[uid],dtype=np.float64)
                    assert len(y)==n and np.isin(y,[0,1]).all()
                    np.testing.assert_array_equal(arr[off:off+n],1.-y)
            labels_checked[split]=dict(count=len(uids),order_hash=manifest_hash(uids))
            if split!='eval':
                for kind in ('layers','reference'):
                    remote=read_json(PKG/'data_protocol/cache_metadata'/kind/split/'complete.json')
                    local=read_json(CACHE/kind/split/'complete.json')
                    if kind=='layers':
                        for k in ('layers','grid','normalization','revision','model_sha256','source_sha256','shape','dtype'):
                            assert remote[k]==local[k],(split,k)
                        cache_diff[f'{kind}/{split}']={k:v['features_sha256'] for k,v in [('published',remote),('local',local)]}
                    else:
                        assert remote['durations']==local['durations'] and remote['offsets']==local['offsets']
                        for k in ('labels20','weights20','labels160','weights160'):
                            assert remote['arrays'][k]['sha256']==local['arrays'][k]['sha256']
    local_audit=ROOT/'data/manifests/audit.json'
    save_json(local_audit,dict(official_label_archive=archive_hash,encoder_hash=model_hash,
              audio_verified=len(published_audit['audio']),fixed_uids_and_native_labels=labels_checked,
              feature_digests=cache_diff,environment=dict(python=platform.python_version(),torch=torch.__version__,
              gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,platform=platform.platform()),
              comparability='Local feature files differ from AutoDL; strict byte replay cannot be claimed.'))
    save_json(CACHE/'verified.json',dict(status='verified',scope='local integrity, UID, labels and train/dev audio',
              audio_audit_sha256=sha256(local_audit),cache_metadata={f'{k}/{s}':sha256(CACHE/k/s/'complete.json')
              for k in ('reference','layers') for s in ('train','dev')}))


def pack_rows(scores,data):
    with np.load(scores) as z:
        assert z['uids'].tolist()==data.uids
        for key,field in [('lengths','lengths160'),('lengths20','lengths20'),('durations','durations')]:
            np.testing.assert_array_equal(z[key],getattr(data,field))
        raw=z['scores20']
        assert len(raw)==int(data.offsets[-1])
    return raw,[dict(uid=u,duration=float(data.durations[i]),labels20=data.fine[u],
                     scores20=raw[data.offsets[i]:data.offsets[i+1]]) for i,u in enumerate(data.uids)]


def replay(source):
    refs={r['model']:r for r in csv.DictReader((PKG/'evaluation/EVAL_METRICS.csv').open(encoding='utf8'))}
    out=ROOT/source
    rep=read_json(out/'results.json') if (out/'results.json').exists() else {}
    for split in ('dev','eval'):
        status(source,split=split)
        data=MultiLayerData(CACHE/'reference'/split,CACHE/'layers'/split,arm='fusion6_9_12',
                            split=split,seed=666,config=model_config(),verify_hashes=True)
        for name in NAMES:
            if split in rep.get(name,{}):
                assert (out/name/f'{split}_scores20.npz').is_file() and (out/name/f'{split}_metrics.json').is_file()
                continue
            folder=(published if source=='frozen_replay' else trained)(name)
            complete=read_json(folder/'complete.json')
            assert complete['trained_epochs']==20 and complete['updates']==15880
            assert sha256(folder/'best.pt')==complete['checkpoint_sha256']
            p=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            cut=p['threshold']; seed=int(name[-3:])
            assert p['epoch']==complete['epoch'] and cut==complete['threshold']
            if source=='frozen_replay':
                assert cut==FROZEN['models'][name]['dev_threshold_full_precision']
            setup(seed,2)
            model=LayerFusionCNN(model_config(),'fusion6_9_12',seed)
            ad=make_adapter(ARMS['fusion6_9_12'],seed,'cpu')
            model.load_state_dict(p['backbone_state_dict']);ad.load_state_dict(p['adapter_state_dict'])
            rows,scores,meta=infer(model.cuda(),ad.cuda(),data,'cuda')
            dest=out/name
            dest.mkdir(parents=True,exist_ok=True)
            save_scores(dest/f'{split}_scores20.npz',scores,meta)
            m=evaluate_report(rows,cut)
            save_json(dest/f'{split}_per_recording.json',m.pop('per_recording'))
            save_json(dest/f'{split}_metrics.json',m)
            item=dict(epoch=p['epoch'],threshold=cut,event_f1=m['event_f1_iou05_percent'],
                      precision=m['event_precision_percent'],recall=m['event_recall_percent'],duration_fpr=m['real_fpr_percent'])
            if split=='dev':
                raw,orig=pack_rows(folder/'dev_scores20.npz',data)
                saved=evaluate_report(orig,cut)['event_f1_iou05_percent']
                assert abs(saved-complete['event_f1_iou05_percent'])<1e-9
                item.update(saved_logit_recalculation=saved,logits_bitwise_equal=np.array_equal(raw,scores),
                    max_abs_logit_error=float(np.max(abs(raw.astype('d')-scores))),
                    changed_active_decisions=sum(int(np.count_nonzero(decision_mask(r['scores20'][:len(r['labels20'])],cut)!=
                         decision_mask(s['scores20'][:len(r['labels20'])],cut))) for r,s in zip(rows,orig)))
            else:
                counts=diagnose_events(rows,cut)['counts']
                save_json(dest/'eval_error_structure.json',counts)
            item['published_delta_pp']=item['event_f1']-float(refs[name]['dev_event_f1_percent' if split=='dev' else 'event_f1_percent'])
            rep.setdefault(name,{})[split]=item
            save_json(out/'results.json',rep)
            print(source,split,name,item,flush=True)
        del data
    lines=['# A0/S1 '+source,'','| 模型 | 最佳轮 | DEV F1 (%) | EVAL F1 (%) | EVAL 与发布结果差（百分点） |','|---|---:|---:|---:|---:|']
    for n,r in rep.items():
        d,v=r['dev'],r['eval']
        lines.append(f"| {n} | {d['epoch']} | {d['event_f1']:.6f} | {v['event_f1']:.6f} | {v['published_delta_pp']:+.6f} |")
    for seed in (666,667):
        delta=rep[f'S1_{seed}']['eval']['event_f1']-rep[f'A0_{seed}']['eval']['event_f1']
        lines+=['',f'Seed {seed} EVAL S1−A0 = {delta:+.6f} 个百分点。']
    lines+=['','本地冻结特征与原 AutoDL 特征文件哈希不同；逐帧误差和 DEV 判定差异见 results.json。']
    (out/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def paired_training():
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
             PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join(map(str,(SUP,CODE,EVENT))))
    for name in NAMES:
        folder=trained(name)
        arm,seed=name.split('_')
        script=(ROOT/'entry667' if seed=='667' else EVENT if arm=='A0' else SUP)/'trainer.py'
        if (folder/'complete.json').exists():
            assert read_json(folder/'complete.json')['updates']==15880
            continue
        expected=read_json(published(name)/'protocol.json')['implementation_sha256']
        hashes={file:dict(published_record=digest,used=sha256(script.with_name(file)))
                for file,digest in expected.items()}
        save_json(ROOT/'entry_audit'/f'{name}.json',hashes)
        for file,entry in hashes.items():
            if entry['used']!=entry['published_record']:
                # REPRODUCE.md designates the shipped seed667 trainer for BOTH arms.
                assert (name,file)==('A0_667','trainer.py')
                assert entry['used']==read_json(published('S1_667')/'protocol.json')['implementation_sha256'][file]
        status('from_scratch_training',model=name)
        dest=logs/(name+'.log')
        with dest.open('x',encoding='utf8') as log:
            entry = script.with_name('run_windows.py') if seed == '667' else script
            subprocess.run([sys.executable,'-X','utf8','-u',str(entry),'--workspace',str(ROOT),'--cache',str(CACHE),
                '--run',str(folder),'--log',str(dest),'--arm',arm,'--seed',seed,'--threads','4','--gpu-uuid',gpu],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        actual=read_json(folder/'complete.json')
        assert actual['trained_epochs']==20 and actual['updates']==15880
        if arm=='A0':
            initial=[torch.load(f/'epochs/epoch_00.pt',map_location='cpu',weights_only=False)
                     for f in (folder,published(name))]
            differences={}
            for key in ('backbone_state_dict','adapter_state_dict'):
                aa,bb=(p[key] for p in initial)
                assert set(aa)==set(bb)
                differences[key]=max(float((aa[n]-bb[n]).abs().max()) for n in aa)
            save_json(ROOT/'entry_audit'/f'{name}_initial_tensors.json',differences)
            # Native random initialization is paired locally. Platform-level rounding
            # against the archived initialization is recorded separately.
            assert max(differences.values())<1e-6
    save_json(ROOT/'fresh_frozen.json',{n:read_json(trained(n)/'complete.json') for n in NAMES})
    replay('fresh_replay')


if __name__=='__main__':
    try:
        if sys.argv[1:]!=['--continue-training']:
            audit()
            replay('frozen_replay')
        paired_training()
        status('complete')
    except BaseException:
        status('failed',detail=traceback.format_exc())
        raise
