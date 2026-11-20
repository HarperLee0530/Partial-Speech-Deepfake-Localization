"""Paired A0/A5 full-budget seed667/668 follow-up. All four frozen before eval."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256,manifest_hash
from cnn_ps.dual_features import FeatureData
from cnn_ps.wavlm_multilayer import build_cache,FrozenMultiWavLM,SCHEMA,LAYERS
from cnn_ps.wavlm_frontend import REVISION
from cnn_ps.augmentation import transform
from cnn_ps.wavlm_ablation_a import AblationCNN,AblationData,ARMS,make_adapter
from cnn_ps.train_wavlm_ablation_a import setup,model_config,state_digest,freeze_eval_plan
from cnn_ps.train_dual import forward_batch
from cnn_ps.finetune import fine_collate
DATA=Path('D:/SAL_精简实验/data/cnn_ps_features_v2')
ENCODER=Path('D:/SAL_精简实验/data/models/wavlm_base_plus_hf')
CACHE=Path('D:/SAL_精简实验/data/wavlm_multilayer_full_a0a5_20261119')
OUT=ROOT/'results'
SEEDS=[667,668];NAMES=['A0_full','A5_fixed_pool']
SOURCE_COMMIT='483c0252cf8b7104376399008d21d414fdced0b6'
START=time.monotonic()

def progress(stage,**kw):
    d=dict(stage=stage,elapsed_seconds=round(time.monotonic()-START,1),**kw)
    save_json(OUT/'status.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)

def prepare():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert sha256(ENCODER/'pytorch_model.bin')=='3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
    frozen_references={}
    for name in NAMES:
        folder=ROOT/'reference/seed666'/name
        p=read_json(folder/'protocol.json');c=read_json(folder/'complete.json')
        assert sha256(folder/'best.pt')==c['checkpoint_sha256']
        assert sha256(folder/'dev_scores20.npz')==c['dev_scores20_sha256']
        for f,digest in p['source_sha256'].items():assert sha256(ROOT/'code/cnn_ps'/f)==digest,(name,f)
        frozen_references[name]=dict(checkpoint_sha256=c['checkpoint_sha256'],epoch=c['epoch'],threshold=c['threshold'])
    metadata={split:read_json(DATA/split/'complete.json') for split in ['train','dev']}
    for split in metadata:
        assert manifest_hash(metadata[split]['uids'])==p[f'{split}_uids_sha256']
        assert metadata[split]['recordings']==(6345 if split=='train' else 2484)
    assert not set(metadata['train']['uids'])&set(metadata['dev']['uids'])
    init={}
    for seed in SEEDS:
        init[str(seed)]={}
        for name in NAMES:
            setup(seed,4);model=AblationCNN(model_config(),name,seed);adapter=make_adapter(ARMS[name],seed,'cpu')
            init[str(seed)][name]=dict(cnn=state_digest(model.cnn),projection=state_digest(model.project),adapter=state_digest(adapter))
        assert init[str(seed)][NAMES[0]]==init[str(seed)][NAMES[1]]
    plan=dict(schema='a0_a5_paired_multiseed_v1',source_commit=SOURCE_COMMIT,seeds=SEEDS,arms=NAMES,
        historical_seed666=frozen_references,initial_digests=init,
        source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),*sorted((ROOT/'code').rglob('*.py'))]},
        reference_metadata_sha256={s:sha256(DATA/s/'complete.json') for s in ['train','dev']},
        training='Exact archived train_wavlm_ablation_a.py. Random initialization (not fine-tuning); frozen WavLM6/9/12; seed-specific paired initialization/data order/channel assignment.20epochs,batch8,15880updates,AdamW lr0.001 wd0.0001 clip5,threads4. Two independent trainer processes.',
        selection='Unchanged: each epoch clean dev duration FPR<=5% threshold; highest dev eventF1, epoch0 eligible, earliest tie. No test threshold/epoch selection.',
        evaluation='Do not load eval data until all four completed checkpoints/thresholds frozen. Historical eval already inspected: exploratory follow-up, not new blind test. No LPS.',
        decision='Report per-seed A5-A0 eventF1/FPR and three-seed mean/sampleSD, without selecting the best seed. Per-record bootstrap conditions on each trained pair; three seeds cannot establish general training stability. FPR +0.5pp reported only as pre-existing reference guard, not new tuning criterion.',
        cache='Rebuild full float32 three-layer frozen features on local GPU with unchanged encoder/augmentation/grid. Compare dev predictions of fixed seed666 A0/A5 with their archived scores before training.',
        cache_location=str(CACHE),data_location=str(DATA),source_model_unchanged=True)
    f=OUT/'plan.json'
    if f.exists():assert read_json(f)==plan,'Protocol changed; use another experiment directory.'
    else:save_json(f,plan)
    return plan

def ensure_cache(split):
    path=CACHE/split
    if (path/'complete.json').exists():
        p=read_json(path/'complete.json')
        assert p['reference_sha256']==sha256(DATA/split/'complete.json') and p['status']=='complete'
        assert sha256(path/'features.npy')==p['features_sha256']
        return
    progress('cache',split=split)
    if split!='eval':
        build_cache(DATA/split,ENCODER,path,device='cuda');return
    assert (OUT/'all_four_frozen.json').exists()
    frozen=read_json(OUT/'all_four_frozen.json')
    for seed in SEEDS:
        assert freeze_eval_plan(OUT/f'seed{seed}',NAMES)==frozen['models'][str(seed)]
    # Identical feature-generation loop; authorization uses A-group schema and both seeds.
    data=FeatureData(DATA/'eval',verify_hashes=True)
    assert data.split=='eval' and len(data)==3562
    path.mkdir(parents=True,exist_ok=False)
    encoder=FrozenMultiWavLM(ENCODER,'cuda');shape=(data.channels,int(data.offsets[-1]),3,768)
    import cnn_ps.wavlm_multilayer as module
    p=dict(schema=SCHEMA,status='building',split='eval',layers=list(LAYERS),frozen=True,
        model_id='microsoft/wavlm-base-plus',revision=REVISION,model_sha256=sha256(ENCODER/'pytorch_model.bin'),
        reference=str(DATA/'eval'),reference_sha256=sha256(DATA/'eval/complete.json'),recordings=len(data),channels=data.channels,
        dtype='float32',source_sha256=sha256(module.__file__),source_unchanged=True,test_read=True,
        authorized_by_frozen_plan_sha256=sha256(OUT/'all_four_frozen.json'),shape=list(shape))
    save_json(path/'protocol.json',p)
    x=np.lib.format.open_memmap(path/'features.npy',mode='w+',dtype='float32',shape=shape)
    for i,row in enumerate(data.records):
        wave,sr=sf.read(row['audio'],dtype='float32');assert sr==16000 and wave.ndim==1 and len(wave)/sr==data.durations[i]
        a,b=map(int,data.offsets[i:i+2]);x[0,a:b]=encoder(transform(wave,row['uid'],0),b-a).cpu().numpy()
        if i%256==0:progress('cache',split='eval',completed=i+1,total=len(data))
    x.flush();del x,encoder
    p.update(status='complete',features_sha256=sha256(path/'features.npy'));save_json(path/'complete.json',p)
    torch.cuda.empty_cache()

@torch.inference_mode()
def verify_cache():
    # No fit or selection. Check full dev scores against both historical seed666 models.
    data=AblationData(DATA/'dev',CACHE/'dev','A0_full',split='dev',seed=666)
    models={};archives={};maximum={n:0. for n in NAMES};changed={n:0 for n in NAMES};records={n:[] for n in NAMES}
    for name in NAMES:
        folder=ROOT/'reference/seed666'/name;p=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model=AblationCNN(model_config(),name,666).to('cuda');adapter=make_adapter(ARMS[name],666,'cuda')
        model.load_state_dict(p['backbone_state_dict'],strict=True);adapter.load_state_dict(p['adapter_state_dict'],strict=True)
        models[name]=(model.eval(),adapter.eval(),p['threshold'])
        z=np.load(folder/'dev_scores20.npz');assert list(z['uids'])==data.uids;archives[name]=z
    for start in range(0,len(data),8):
        rows=[data.get_record(i,0) for i in range(start,min(start+8,len(data)))];batch=fine_collate(rows)
        for name,(model,adapter,threshold) in models.items():
            logits,_=forward_batch(model,adapter,batch,'cuda')
            for j,row in enumerate(rows):
                i=start+j;a,b=map(int,data.offsets[i:i+2]);old=archives[name]['scores20'][a:b];score=logits[j,:b-a].cpu().numpy()
                maximum[name]=max(maximum[name],float(np.abs(score-old).max()))
                n=row['length20'];changed[name]+=int(np.count_nonzero((score[:n].astype(float)>=threshold)!=(old[:n].astype(float)>=threshold)))
                records[name].append(dict(uid=row['uid'],scores20=score.copy(),labels20=row['fine_labels'][:n].numpy(),duration=row['duration']))
        if start%256==0:progress('source_cache_check',completed=start+len(rows),total=len(data))
    from cnn_ps.report_native20 import evaluate_report
    report={}
    for name in NAMES:
        assert maximum[name]<0.001,(name,maximum[name])
        metric=evaluate_report(records[name],models[name][2]);old=read_json(ROOT/'reference/seed666'/name/'dev_metrics.json')
        assert abs(metric['event_f1_iou05_percent']-old['event_f1_iou05_percent'])<.01
        report[name]=dict(recordings=len(data),max_raw_score_difference=maximum[name],threshold_decision_changes=changed[name],event_f1=metric['event_f1_iou05_percent'],archived_event_f1=old['event_f1_iou05_percent'])
    save_json(OUT/'source_cache_check.json',report)
    del models,data,records;torch.cuda.empty_cache()

def launch(command,log):
    log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('w',encoding='utf-8')
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    p=subprocess.Popen([sys.executable,'-X','utf8',*command],cwd=ROOT/'code',stdout=handle,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return p,handle

def train_all():
    for seed in SEEDS:
        processes=[]
        for name in NAMES:
            f=OUT/f'seed{seed}'/name/'complete.json'
            if f.exists():continue
            output=OUT/f'seed{seed}'
            command=['-m','cnn_ps.train_wavlm_ablation_a','--arm',name,'--train-cache',str(DATA/'train'),'--dev-cache',str(DATA/'dev'),
                     '--train-ssl',str(CACHE/'train'),'--dev-ssl',str(CACHE/'dev'),'--output',str(output),'--seed',str(seed),
                     '--epochs','20','--batch-size','8','--device','cuda','--threads','4']
            p,h=launch(command,OUT/'logs'/f'seed{seed}_{name}.log');processes.append((name,p,h))
        while processes:
            live=[];state={}
            for name,p,h in processes:
                f=OUT/f'seed{seed}'/name/'progress.json'
                try:state[name]=read_json(f) if f.exists() else dict(phase='loading')
                except (ValueError,OSError):state[name]=dict(phase='writing_progress')
                rc=p.poll()
                if rc is None:live.append((name,p,h))
                else:
                    h.close()
                    if rc:raise RuntimeError(f'Training failed seed{seed}/{name}; see log; no eval accessed.')
            progress('train',seed=seed,models=state)
            processes=live
            if live:time.sleep(25)
    all_frozen={str(seed):freeze_eval_plan(OUT/f'seed{seed}',NAMES) for seed in SEEDS}
    for seed,pairs in all_frozen.items():
        for name,c in pairs.items():assert c['trained_epochs']==20 and c['updates']==15880
    save_json(OUT/'all_four_frozen.json',dict(models=all_frozen,selection='All four complete and frozen before eval cache is opened',plan_sha256=sha256(OUT/'plan.json')))

def evaluate():
    ensure_cache('eval')
    for seed in SEEDS:
        if all((OUT/f'seed{seed}'/name/'eval_metrics.json').exists() for name in NAMES):continue
        command=['-m','cnn_ps.train_wavlm_ablation_a','--evaluate','--arms',*NAMES,'--eval-cache',str(DATA/'eval'),
                 '--eval-ssl',str(CACHE/'eval'),'--output',str(OUT/f'seed{seed}'),'--seed',str(seed),'--device','cuda','--threads','4']
        p,h=launch(command,OUT/'logs'/f'evaluate_seed{seed}.log')
        while p.poll() is None:progress('evaluate',seed=seed);time.sleep(20)
        h.close();assert p.returncode==0,f'Evaluation failed: seed{seed}'

def main():
    setup(666,4);assert torch.cuda.is_available();prepare()
    for split in ['train','dev']:ensure_cache(split)
    if not (OUT/'source_cache_check.json').exists():verify_cache()
    train_all();evaluate();progress('complete',status='complete')

if __name__=='__main__':main()
