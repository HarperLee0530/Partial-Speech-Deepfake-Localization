"""Replay pilot on the actual published multi-layer WavLM checkpoint.

Frozen public encoder; ALL original fusion/projection/CNN/adapter parameters
are adapted. Existing five-term objective and original decision threshold stay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfilt
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256
from cnn_ps.backbone import ModelConfig
from cnn_ps.wavlm_multilayer import FrozenMultiWavLM,LayerFusionCNN
from cnn_ps.train_wavlm_multilayer import setup,make_adapter,state_digest
from cnn_ps.train_dual import forward_batch,training_loss
from cnn_ps.finetune import fine_collate
from cnn_ps.augmentation import transform
from cnn_ps.report_native20 import evaluate_report

RELEASE=ROOT.parents[1]/'tmp/publish_sal_logmel_20261118/experiments/wavlm_multilayer_20261117'
SOURCE=RELEASE/'weights/best.pt'
EXPECTED='160a4b3cacbee48398d11f218e3ca2dff76615670ae05255f6775119dd37b50e'
DATA=Path('D:/SAL_精简实验/data/cnn_ps_features_v2')
ENCODER=Path('D:/SAL_精简实验/data/models/wavlm_base_plus_hf')
BANK=Path('D:/SAL_精简实验/data/multilayer_noise_replay_pilot_20261119')
OUT=ROOT/'results'
PREVIOUS=ROOT.parent/'multilayer_replay_20261119'
SEEDS=[666,667]
ARMS=['old_only','new_only','replay']
DEVICE='cuda'
START=time.monotonic()


def progress(**kw):
    kw['elapsed_seconds']=round(time.monotonic()-START,2)
    save_json(OUT/'status.json',kw);print(json.dumps(kw,ensure_ascii=False),flush=True)


class Data:
    def __init__(self,split):
        assert split in ('train','dev')
        self.folder=DATA/split;self.meta=read_json(self.folder/'complete.json')
        assert self.meta['split']==split and self.meta['status']=='complete'
        self.uids=self.meta['uids'];self.offsets=self.meta['offsets']
        self.arrays={k:np.load(self.folder/f'{k}.npy',mmap_mode='r')
                     for k in ('labels20','labels160','weights20','weights160')}

    def row(self,i,features):
        a,b=self.offsets[i:i+2]
        assert features.shape==(1,b-a,2304)
        tensor=lambda k,l,r:torch.from_numpy(np.array(self.arrays[k][l:r],copy=True))
        return dict(uid=self.uids[i],features=features,labels=tensor('labels160',a//8,b//8),
          weights=tensor('weights160',a//8,b//8),fine_labels=tensor('labels20',a,b),
          fine_weights=tensor('weights20',a,b),duration=float(self.meta['durations'][i]),
          length20=int(self.meta['lengths20'][i]),channel=0)

    def wave(self,i):
        wave,sr=sf.read(self.meta['records'][i]['audio'],dtype='float32')
        assert sr==16000 and wave.ndim==1
        assert abs(len(wave)/sr-float(self.meta['durations'][i]))<1e-9
        return wave


def changed(wave,uid,condition):
    if condition in ('clean','noise20','eq'):
        return transform(wave,uid,('clean','noise20','eq').index(condition))
    if condition in ('lp3400','lp2600'):
        cutoff=3400 if condition=='lp3400' else 2600
        return sosfilt(butter(6,cutoff,fs=16000,output='sos'),wave).astype(np.float32)
    if condition.startswith(('noise','pink')):
        pink=condition.startswith('pink');snr=float(condition[4:] if pink else condition[5:])
        tag=f'replay20261119|{uid}|noise10' if condition=='noise10' else f'noise_pilot20261119|{uid}|{condition}'
        seed=int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8],'little')
        noise=np.random.default_rng(seed).standard_normal(len(wave))
        if pink:
            z=np.fft.rfft(noise);freq=np.fft.rfftfreq(len(noise));z[0]=0
            z[1:]/=np.sqrt(freq[1:]);noise=np.fft.irfft(z,n=len(wave))
        noise-=noise.mean()
        scale=np.sqrt(np.mean(wave.astype(float)**2))/(10**(snr/20)*max(np.sqrt(np.mean(noise**2)),1e-30))
        return (wave.astype(float)+noise*scale).astype(np.float32)
    raise ValueError(condition)


def feature(encoder,data,i,condition):
    a,b=data.offsets[i:i+2]
    states=encoder(changed(data.wave(i),data.uids[i],condition),b-a)
    # Clone outside inference_mode so cached tensors support downstream gradients.
    return states.detach().cpu().reshape(1,b-a,2304).clone()


def load_backend(path=SOURCE):
    p=torch.load(path,map_location='cpu',weights_only=False)
    model=LayerFusionCNN(ModelConfig(**p['model_config']),'fusion6_9_12',666).to(DEVICE)
    adapter=make_adapter(p['adapter_options'],666,DEVICE)
    model.load_state_dict(p['backbone_state_dict'],strict=True)
    adapter.load_state_dict(p['adapter_state_dict'],strict=True)
    return model,adapter,p


def save_backend(path,model,adapter,source,seed,arm,epoch):
    torch.save(dict(schema='ps_multilayer_replay_pilot_v1',source_checkpoint_sha256=EXPECTED,
       source_epoch=18,adaptation_epoch=epoch,seed=seed,arm=arm,threshold=float(source['threshold']),
       model_config=source['model_config'],adapter_options=source['adapter_options'],
       backbone_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
       adapter_state_dict={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
       layer_weights=model.layer_weights().detach().cpu().tolist()),path)


def make_plan(train,dev,source):
    prior=read_json(PREVIOUS/'results/plan.json')
    target,memory,stress=[prior['indices'][k] for k in ('target','memory','stress')]
    assert len(stress)==384 and not set(target)&set(memory)
    assert not set(train.uids)&set(dev.uids)
    return dict(schema='actual_multilayer_noise_replay_pilot_v1',source=str(SOURCE),source_sha256=EXPECTED,
       source_epoch=18,source_threshold=float(source['threshold']),seeds=SEEDS,arms=ARMS,
       indices=dict(target=target,memory=memory,stress=stress),
       uids=dict(target=[train.uids[i] for i in target],memory=[train.uids[i] for i in memory],
                 stress=[dev.uids[i] for i in stress],clean_full=dev.uids),
       training='All arms start from SAME published fusion6/9/12 checkpoint. Seeds vary fine-tuning order/dropout, not original pretraining. Four epochs,32 updates/epoch,batch8,AdamW lr1e-4 wd1e-4,clip5. Fixed final epoch, no model selection.',
       parameter_policy='Public WavLM encoder frozen. Three fusion logits, LayerNorm/projection, original CNN-BiGRU and complete dual-resolution adapter ALL remain trainable.',
       losses='Unchanged main20+main160+0.5coarse_aux+0.1boundary+0.05within_segment_consistency.',
       old_only='Only256 memory recordings, rotating original/noise20dB/EQ views;128 updates.',
       new_only='Only256 target recordings, rotating white noise15dB/noise5dB by (global_index+epoch-1)%2;128 updates.',
       replay='Same target batches as new_only + same memory batches/views as old_only;0.5 new loss+0.5 old loss;128 updates, extra old forward/backward computation. Not compute-matched.',
       source_data_note='All512 selected training recordings were part of original6345 training pool. Newness is channel condition, not unseen training content. Memory and target UIDs disjoint.',
       conditions=dict(clean_full='All2484 dev recordings; source-regression check, historically used for source selection.',
          noise0='Same384 under untrained stronger0dB white noise.', pink10='Same384 under untrained pink noise10dB.',
          lp2600='Same384 under unseen stronger lowpass, never in fine-tuning.',
          noise10='Same384 under10dB white noise; held-out training strength but inspected in previous pilot, exploratory re-test.',
          clean_subset='Original versions of same384, derived from full-clean evaluation.'),
       transforms='Causal sixth-order Butterworth lowpass; no group-delay correction; white noise normalized per-record RMS, fixed UID seed, no clipping/time change.',
       threshold_policy='Original checkpoint logit threshold fixed for EVERY model and EVERY condition. No recalibration, no eval/LPS access.',
       preservation_screen=dict(event_f1_max_drop_pp=.5,frame_f1_max_drop_pp=.5,real_fpr_max_increase_pp=.5,short_recall_max_drop_pp=1.),
       uncertainty='Report per-seed point preservation screen and paired UID bootstrap95% intervals. A point pass does not establish noninferiority; check lower event/frame bounds>=-0.5 and upper FPR bound<=0.5 separately.',
       inference='Frozen encoder shared across9 backends. Known numerical archive error previously0.000679: sanity ceiling0.001, report float64 decision flips; same original checkpoint required.',
       generalization_claim='Training white noise15/5dB. Test interpolation10dB, extrapolation0dB and unseen spectral type pink10dB. Entire384 stress subset reused from previous exploratory pilot; not blind test. LP2600 checks retention. Synthetic noise is not real-world unseen corpus.', previous_control='Previous lowpass+replay checkpoints evaluated unchanged alongside all arms; new arms start from original, not previous replay.', previous_checkpoint_sha256={str(seed):sha256(PREVIOUS/f'results/seed{seed}/replay/last.pt') for seed in SEEDS},
       code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),*sorted((ROOT/'code/cnn_ps').glob('*.py'))]},
       data_metadata_sha256={s:sha256(DATA/s/'complete.json') for s in ('train','dev')},
       original_weights_modified=False,formal_test_accessed=False)


def verify_source(encoder,dev,source):
    model,adapter,_=load_backend();model.eval();adapter.eval()
    z=np.load(RELEASE/'reports/training/fusion6_9_12/dev_scores20.npz')
    uid_index={str(uid):i for i,uid in enumerate(z['uids'])};offsets=np.r_[0,np.cumsum(z['lengths']*8)]
    assert set(uid_index)==set(dev.uids)
    maximum=0.;records=[]
    # Fixed first8 recordings are an implementation identity check, not selection.
    for i in range(8):
        row=dev.row(i,feature(encoder,dev,i,'clean'))
        with torch.inference_mode():y,_=forward_batch(model,adapter,fine_collate([row]),DEVICE)
        j=uid_index[row['uid']];expected=z['scores20'][offsets[j]:offsets[j+1]]
        actual=y[0].cpu().numpy();error=float(np.max(np.abs(expected-actual)))
        assert actual.shape==expected.shape and error<1e-4,(row['uid'],error)
        maximum=max(maximum,error);records.append(row)
    with torch.inference_mode():
        batch,_=forward_batch(model,adapter,fine_collate(records),DEVICE)
        batched_error=0.
        for j,row in enumerate(records):
            single,_=forward_batch(model,adapter,fine_collate([row]),DEVICE)
            batched_error=max(batched_error,float((batch[j,:single.shape[1]]-single[0]).abs().max()))
    assert batched_error<1e-4
    assert abs(float(source['threshold'])+.17905920743942258)<1e-15
    checks=dict(recordings=8,max_archive_score_difference=maximum,batch_single_difference=batched_error,
       source_weights_verified=True,full_original_architecture_loaded=True,
       threshold=float(source['threshold']),layer_weights=model.layer_weights().detach().cpu().tolist())
    save_json(OUT/'source_identity_check.json',checks)
    return checks


class FeatureBank:
    def __init__(self,meta):
        self.meta=meta;self.entries={tuple(r['key']):r for r in meta['entries']}
        self.x=np.load(BANK/'features.npy',mmap_mode='r')
    def get(self,data,i,condition):
        r=self.entries[(data.uids[i],condition)]
        x=torch.from_numpy(np.array(self.x[r['start']:r['stop']],copy=True)).reshape(1,r['stop']-r['start'],2304)
        return data.row(i,x)


def prepare_bank(encoder,train,plan):
    path=BANK/'complete.json'
    if path.exists():
        meta=read_json(path)
        assert meta['plan_sha256']==sha256(OUT/'plan.json')
        assert sha256(BANK/'features.npy')==meta['features_sha256']
        return FeatureBank(meta)
    entries=[];pos=0
    tasks=[(i,v) for i in plan['indices']['memory'] for v in ('clean','noise20','eq')]
    tasks += [(i,v) for i in plan['indices']['target'] for v in ('noise15','noise5')]
    for i,v in tasks:
        n=train.offsets[i+1]-train.offsets[i]
        entries.append(dict(key=[train.uids[i],v],index=i,start=pos,stop=pos+n));pos+=n
    x=np.lib.format.open_memmap(BANK/'features.npy',mode='w+',dtype='float32',shape=(pos,3,768))
    for j,r in enumerate(entries):
        x[r['start']:r['stop']]=feature(encoder,train,r['index'],r['key'][1]).reshape(-1,3,768).numpy()
        if j%64==0:progress(status='running',stage='training_feature_cache',completed=j+1,total=len(entries))
    x.flush();del x
    meta=dict(plan_sha256=sha256(OUT/'plan.json'),entries=entries,shape=[pos,3,768],
              features_sha256=sha256(BANK/'features.npy'),dtype='float32',encoder_frozen=True)
    save_json(path,meta);return FeatureBank(meta)


def train_arm(data,bank,plan,seed,arm):
    folder=OUT/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    setup(seed,4);model,adapter,source=load_backend()
    original_digests=dict(backbone=state_digest(model),adapter=state_digest(adapter))
    params=list(model.parameters())+list(adapter.parameters())
    assert model.layer_logits.requires_grad
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    history=[]
    for epoch in range(1,5):
        model.train();adapter.train()
        target=np.random.default_rng(seed*1000+epoch).permutation(plan['indices']['target']).tolist()
        memory=np.random.default_rng(seed*2000+epoch).permutation(plan['indices']['memory']).tolist()
        totals=[];started=time.monotonic()
        for step,start in enumerate(range(0,256,8)):
            opt.zero_grad(set_to_none=True)
            terms={}
            if arm in ('new_only','replay'):
                torch.manual_seed(seed*100000+epoch*1000+step)
                batch=fine_collate([bank.get(data,i,('noise15','noise5')[(i+epoch-1)%2]) for i in target[start:start+8]])
                y,o=forward_batch(model,adapter,batch,DEVICE)
                terms['new']=training_loss(y,o,batch,DEVICE,True)[0]
            if arm in ('old_only','replay'):
                # Equal old-view dropout stream between old_only and replay.
                torch.manual_seed(seed*100000+epoch*1000+step+50000000)
                batch=fine_collate([bank.get(data,i,('clean','noise20','eq')[(i+epoch-1)%3]) for i in memory[start:start+8]])
                y,o=forward_batch(model,adapter,batch,DEVICE)
                terms['old']=training_loss(y,o,batch,DEVICE,True)[0]
            loss=.5*(terms['new']+terms['old']) if arm=='replay' else next(iter(terms.values()))
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step()
            totals.append({k:float(v.detach()) for k,v in terms.items()})
            if step%16==0:progress(status='running',stage='train',seed=seed,arm=arm,epoch=epoch,step=step+1,total=32)
        history.append(dict(epoch=epoch,updates=epoch*32,seconds=time.monotonic()-started,
                   losses={k:float(np.mean([r[k] for r in totals])) for k in totals[0]},
                   layer_weights=model.layer_weights().detach().cpu().tolist()))
        save_json(folder/'history.json',history)
        save_backend(folder/'last.pt',model,adapter,source,seed,arm,epoch)
    save_json(folder/'complete.json',dict(checkpoint_sha256=sha256(folder/'last.pt'),updates=128,
       initial_digests=original_digests,final_layer_weights=model.layer_weights().detach().cpu().tolist(),
       backbone_trainable=sum(p.numel() for p in model.parameters()),adapter_trainable=sum(p.numel() for p in adapter.parameters())))


@torch.inference_mode()
def evaluate_all(encoder,dev,plan,paths):
    models={name:load_backend(path)[:2] for name,path in paths.items()}
    for model,adapter in models.values():model.eval();adapter.eval()
    threshold=plan['source_threshold'];results={name:{} for name in models}
    archive=np.load(RELEASE/'reports/training/fusion6_9_12/dev_scores20.npz')
    lookup={str(uid):i for i,uid in enumerate(archive['uids'])}
    offsets=np.r_[0,np.cumsum(archive['lengths']*8)]
    max_archive_error=0.; changed_decisions=0; compared_frames=0
    conditions=[('clean_full',list(range(len(dev.uids))),'clean')]
    conditions += [(v,plan['indices']['stress'],v) for v in ('noise10','noise0','pink10','lp2600')]
    for condition,indices,view in conditions:
        all_records={name:[] for name in models}
        for start in range(0,len(indices),8):
            group=indices[start:start+8]
            rows=[dev.row(i,feature(encoder,dev,i,view)) for i in group]
            batch=fine_collate(rows)
            for name,(model,adapter) in models.items():
                y,_=forward_batch(model,adapter,batch,DEVICE)
                for j,row in enumerate(rows):
                    n=int(batch['lengths'][j]);score=y[j,:n].cpu().numpy().copy()
                    if name=='original' and condition=='clean_full':
                        k=lookup[row['uid']];expected=archive['scores20'][offsets[k]:offsets[k+1]]
                        error=float(np.abs(score-expected).max());max_archive_error=max(max_archive_error,error)
                        assert error<0.001,(row['uid'],error)
                        changed_decisions+=int(np.count_nonzero((score[:row['length20']].astype(np.float64)>=threshold)!=(expected[:row['length20']].astype(np.float64)>=threshold)))
                        compared_frames+=row['length20']
                    all_records[name].append(dict(uid=row['uid'],scores20=score,duration=row['duration'],
                            labels20=row['fine_labels'][:row['length20']].numpy()))
            if start%64==0:progress(status='running',stage='evaluate',condition=condition,recordings=start+len(group),total=len(indices))
        for name,records in all_records.items():
            folder=OUT/'evaluation'/name;folder.mkdir(parents=True,exist_ok=True)
            report=evaluate_report(records,threshold)
            save_json(folder/f'{condition}_metrics.json',report)
            np.savez_compressed(folder/f'{condition}_scores.npz',**{r['uid']:r['scores20'] for r in records})
            results[name][condition]={k:v for k,v in report.items() if k!='per_recording'}
            if condition=='clean_full':
                byuid={r['uid']:r for r in records};subset=[byuid[uid] for uid in plan['uids']['stress']]
                subreport=evaluate_report(subset,threshold)
                save_json(folder/'clean_subset_metrics.json',subreport)
                results[name]['clean_subset']={k:v for k,v in subreport.items() if k!='per_recording'}
    save_json(OUT/'full_source_reproduction.json',dict(recordings=len(dev.uids),max_archive_score_difference=max_archive_error, changed_frame_decisions=changed_decisions, compared_frames=compared_frames, original_tolerance_pass=max_archive_error<1e-4, amendment="Report full observed discrepancy and decision changes; numerical ceiling0.001. No training or threshold changes."))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    setup(666,4);OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and sha256(SOURCE)==EXPECTED
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    train_data,dev=Data('train'),Data('dev')
    plan=json.loads(json.dumps(make_plan(train_data,dev,source)))
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan,'Protocol changed; use new run.'
    else:save_json(OUT/'plan.json',plan)
    assert sha256(ENCODER/'pytorch_model.bin')=='3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
    encoder=FrozenMultiWavLM(ENCODER,DEVICE)
    assert not any(p.requires_grad for p in encoder.parameters())
    verify_source(encoder,dev,source)
    if args.check:print('Actual source model reproduction passed.',flush=True);return
    BANK.mkdir(parents=True,exist_ok=True)
    bank=prepare_bank(encoder,train_data,plan)
    del encoder;torch.cuda.empty_cache()
    for seed in SEEDS:
        for arm in ARMS:train_arm(train_data,bank,plan,seed,arm)
    paths={'original':SOURCE}
    paths.update({f'{seed}_{arm}':OUT/f'seed{seed}'/arm/'last.pt' for seed in SEEDS for arm in ARMS})
    paths.update({f'{seed}_previous_replay':PREVIOUS/f'results/seed{seed}/replay/last.pt' for seed in SEEDS})
    frozen={name:dict(path=str(path),sha256=sha256(path),threshold=plan['source_threshold']) for name,path in paths.items()}
    save_json(OUT/'frozen_models.json',frozen)
    del bank;torch.cuda.empty_cache()
    encoder=FrozenMultiWavLM(ENCODER,DEVICE)
    results=evaluate_all(encoder,dev,plan,paths)
    assert sha256(SOURCE)==EXPECTED
    save_json(OUT/'comparison.json',dict(results=results,source_unchanged=True,formal_test_accessed=False,
                  all_models_frozen_before_evaluation=True))
    progress(status='complete',stage='finished')


if __name__=='__main__':main()
