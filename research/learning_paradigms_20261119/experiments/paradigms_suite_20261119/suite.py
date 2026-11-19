"""Six learning-paradigm pilots with preregistered matched controls.

No eval/LPS inputs. Synthetic-channel adaptation is not unseen-attack generalization.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import common as c
from cnn_ps.io import read_json,save_json,sha256
from cnn_ps.backbone import CNNBiGRU,LogMel
from cnn_ps.wavlm_backend import ProjectedCNN
from cnn_ps.metrics20 import choose_threshold
from cnn_ps.report_native20 import evaluate_report

OUT=ROOT/'results'
OLD=ROOT.parent/'paradigms_selftraining_teacher20_20261119'/'results'
WROOT=Path('D:/SAL_精简实验/data/ps_wavlm_base_plus_v1')
WWEIGHTS=Path('D:/SAL_精简实验/data/models/wavlm_base_plus_hf')
SEEDS=[666,667]
ARMS=['mel_control','selftraining_reference','multitask','wavlm_control',
      'cotraining_mel','cotraining_wavlm','coral','continual_source',
      'continual_naive','continual_ewc','continual_replay',
      'meta_erm','meta_fomaml','meta_erm_adapt','meta_fomaml_adapt']
START=time.monotonic()
DEVICE='cuda'


def status(**kw):
    kw['elapsed_seconds']=round(time.monotonic()-START,2)
    save_json(OUT/'status.json',kw)
    print(json.dumps(kw,ensure_ascii=False),flush=True)


class Data(c.Cache):
    def __init__(self,split):
        super().__init__(split)
        wm=read_json(WROOT/split/'complete.json')
        assert wm['reference_sha256']==sha256(self.folder/'complete.json')
        assert wm['frozen'] and wm['status']=='complete' and not wm['probe_only']
        self.wav=np.load(WROOT/split/'features.npy',mmap_mode='r')
        assert self.wav.shape==(self.meta['channels'],self.offsets[-1],768)
        self.band={};self.wavband={}

    def row(self,i,view=0,kind='mel',labeled=True):
        r=self.labeled(i,0) if labeled else self.bare(i,0)
        if view=='band':
            r['features']=(self.wavband if kind=='wav' else self.band)[self.uids[i]]
        elif kind=='wav':
            a,b=self.offsets[i:i+2]
            r['features']=torch.from_numpy(np.array(self.wav[view,a:b],copy=True)).unsqueeze(0)
        else:
            r['features']=self.features(i,view)
        return r


class BoundaryModel(nn.Module):
    def __init__(self,seed):
        super().__init__();self.config=c.model_config();self.cnn=CNNBiGRU(self.config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+999)
            self.boundary=nn.Linear(256,1)

    def forward(self,x,lengths,return_features=False):
        return self.cnn(x,lengths,return_features=return_features)


class MetaHead(nn.Module):
    """Matched framewise heads for the separate synthetic-task meta pilot."""
    def __init__(self):
        super().__init__();self.config=c.model_config()
        self.net=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,x,lengths,return_features=False):
        x=x[:,0];h=self.net[:3](x);y=self.net[3](h).squeeze(-1)
        mask=torch.arange(y.shape[1],device=y.device)[None,:]<lengths.to(y.device)[:,None]
        y=y*mask
        return (y,h) if return_features else y


def build(kind,seed):
    if kind=='wav':return ProjectedCNN(c.model_config(),'wavlm_frozen',seed).to(DEVICE)
    if kind=='boundary':return BoundaryModel(seed).to(DEVICE)
    if kind=='meta':return MetaHead().to(DEVICE)
    return CNNBiGRU(c.model_config()).to(DEVICE)


def save_model(path,model,kind,seed):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind=kind,seed=seed,state_dict={k:v.detach().cpu().clone()
               for k,v in model.state_dict().items()}),path)


def load(path):
    p=torch.load(path,map_location='cpu',weights_only=False)
    model=build(p.get('kind','mel'),p['seed']);model.load_state_dict(p['state_dict'])
    return model


def fresh_warm(seed):
    return load(OLD/f'seed{seed}'/'warm'/'last.pt')


def forward(model,rows):
    batch=c.fine_collate(rows)
    logits,context=model(batch['features'].to(DEVICE),batch['lengths'],return_features=True)
    return logits,context,batch


def boundary_loss(model,context,batch):
    y=batch['fine_labels'].to(DEVICE);w=batch['fine_weights'].to(DEVICE)
    t=torch.zeros_like(y);t[:,1:]=(y[:,1:]!=y[:,:-1]).float()
    pair=torch.zeros_like(w);pair[:,1:]=torch.minimum(w[:,1:],w[:,:-1])
    logits=model.boundary(context).squeeze(-1)
    loss=F.binary_cross_entropy_with_logits(logits,t,pos_weight=logits.new_tensor(10.),reduction='none')
    return (loss*pair).sum()/pair.sum().clamp_min(1e-8)


def coral_loss(x,y):
    """Deep CORAL covariance loss, normalized by 4*d*d."""
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1]
    x=x-x.mean(0);y=y-y.mean(0)
    cx=x.T@x/max(len(x)-1,1);cy=y.T@y/max(len(y)-1,1)
    return (cx-cy).square().mean()/4


def valid_context(h,batch):
    w=batch['fine_weights'].to(DEVICE)
    return h[w>0][::4][:2048]


def ewc_penalty(model,anchor,fisher):
    return .5*sum((fisher[k]*(v-anchor[k]).square()).sum() for k,v in model.named_parameters())


def plan_for(train,dev):
    prior=read_json(OLD/'plan.json');ids=copy.deepcopy(prior['splits'])
    unused=sorted(set(range(len(dev.uids)))-set(ids['calibration'])-set(ids['audit']))
    rng=np.random.default_rng(20261120)
    di=rng.permutation(unused).tolist();ids['calibration']=di[:256];ids['audit']=di[256:640]
    ti=sorted(set(range(len(train.uids)))-set(ids['labeled'])-set(ids['unlabeled']))
    ids['meta_support']=rng.permutation(ti)[:32].tolist()
    assert len(set(ids['labeled'])|set(ids['unlabeled'])|set(ids['meta_support']))==1184
    assert not set(train.uids)&set(dev.uids)
    assert not set(ids['calibration'])&set(ids['audit'])
    return dict(schema='learning_paradigms_suite_v1',seeds=SEEDS,arms=ARMS,splits=ids,
        uids={k:[(dev if k in ('calibration','audit') else train).uids[i] for i in v] for k,v in ids.items()},
        backbone='Same original CNN-BiGRU. Frozen WavLM adds pretrained encoder and 768->80 projection; NOT capacity/pretraining matched.',
        main_training='Reuse independently trained warm20 CNN; 4 epochs AdamW lr3e-4 wd1e-4 batch8, identical labeled order/views/dropout seeds. WavLM gets matched warm20 lr1e-3 then branch4.',
        labeled=384,unlabeled=768,calibration=256,audit=384,
        multitask='Hard-shared CNN-BiGRU with auxiliary transition head; paired valid frames; boundary BCE positive weight10, auxiliary coefficient0.1. No boundary-head decoding.',
        cotraining='One round of two-view consensus co-training-inspired learning: independently trained Mel/WavLM teachers; both confidence>=0.9, agree; exclude +/-2 frames around either predicted transition. Both students use same consensus pseudo labels, balanced BCE coefficient0.3. Views not proven conditionally independent.',
        coral='Source labeled training remains augmented. Additional clean source context and band-limited unlabeled target context aligned with Deep CORAL covariance loss weight100; stride4, cap2048 valid frames. No target labels.',
        continual='Source warm20 has seen clean/noise/EQ. New task is band-limited audio with same 384 labeled contents. Compare 4 epochs target-only fine-tuning, EWC(lambda1000; empirical Fisher64 recordings), replay(0.5 target +0.5 clean). Replay uses extra old-data forward pass.',
        meta='Separate matched WavLM LayerNorm768->Linear128->GELU->Linear1 heads. ERM vs first-order MAML. 200 episodes, 8 support+8 disjoint query recordings, channel task randomly clean/noise/EQ. FOMAML inner SGD lr0.01 one step, outer AdamW lr0.001; ERM same episodes and data. Both adapt5 steps SGD lr0.01 batch8 to same32 labeled band-limited training recordings not in other train roles.',
        band_filter='Causal sixth-order Butterworth lowpass3400Hz, sr16000, SOS filter then float32. Length unchanged; no latency correction to labels. Synthetic channel task, not new spoof generator.',
        thresholds='Main arms each calibrate clean source at FPR<=5%. Continual variants share frozen SOURCE model threshold. Meta adapted heads share corresponding unadapted clean threshold. Never recalibrate on band-limited data.',
        evaluation='Fixed last epoch; clean and band-limited same384 dev recordings. New calibration/audit UIDs exclude previous selftraining pilot dev roles, but historical full-dev reuse remains.',
        primary_metric='event F1 at IoU>=0.5 on native20ms; also frame F1, duration FPR, short<=320ms recall, EER.',
        interpretation='Exploratory multiple comparisons, two seeds; no formal winner selection. Adaptation and nonadaptation have different access; do not rank all arms as one fair leaderboard.',
        pretrained_weights_used='Only public frozen generic WavLM and previous CNN warm/baseline weights trained on identical384 labels.',
        formal_test_accessed=False,original_models_modified=False,
        cache_metadata={f'{kind}_{split}':sha256((c.CACHE if kind=='mel' else WROOT)/split/'complete.json')
                        for kind in ('mel','wav') for split in ('train','dev')},
        references={'MAML':'https://arxiv.org/abs/1703.03400','EWC':'https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101/',
                    'CORAL':'https://arxiv.org/abs/1607.01719','WavLM':'https://arxiv.org/abs/2110.13900'},
        frozen_old_inputs={f'{seed}_{arm}':sha256(OLD/f'seed{seed}'/arm/'last.pt')
                           for seed in SEEDS for arm in ('warm','supervised','selftraining')},
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'common.py',*sorted((ROOT/'code/cnn_ps').glob('*.py'))]})


def prepare_band(train,dev,ids):
    """Only inputs, never outcomes; cache one unseen synthetic channel."""
    folder=OUT/'band_cache';folder.mkdir(exist_ok=True)
    specs=[('train',train,ids['labeled']+ids['unlabeled']+ids['meta_support'],ids['meta_support']),
           ('dev',dev,ids['audit'],ids['audit'])]
    sos=butter(6,3400,fs=16000,output='sos');front=LogMel(c.model_config())
    encoder=None
    for split,data,mel_ids,wav_ids in specs:
        path=folder/f'{split}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            data.band=saved['mel'];data.wavband=saved['wav'];continue
        if encoder is None:
            from cnn_ps.wavlm_frontend import FrozenWavLM
            assert sha256(WWEIGHTS/'pytorch_model.bin')==read_json(WROOT/'train/complete.json')['model_sha256']
            encoder=FrozenWavLM(WWEIGHTS,DEVICE)
        wavset=set(wav_ids)
        for j,i in enumerate(mel_ids):
            wave,sr=sf.read(data.meta['records'][i]['audio'],dtype='float32')
            assert sr==16000 and wave.ndim==1
            assert abs(len(wave)/sr-data.meta['durations'][i])<1e-9
            changed=sosfilt(sos,wave).astype(np.float32)
            n=data.offsets[i+1]-data.offsets[i]
            with torch.inference_mode():
                mel=front(F.pad(torch.from_numpy(changed),(0,n*320-len(wave))))
                if i in wavset:
                    wav=encoder(changed,n).cpu().unsqueeze(0)
            # Ordinary tensors can be saved for parameter gradients during training.
            data.band[data.uids[i]]=mel.clone()
            if i in wavset:data.wavband[data.uids[i]]=wav.clone()
            if j%128==0:status(status='running',stage='band_features',split=split,completed=j+1,total=len(mel_ids))
        torch.save(dict(mel=data.band,wav=data.wavband),path)
    del encoder
    torch.cuda.empty_cache()
    save_json(folder/'complete.json',dict(files={p.name:sha256(p) for p in folder.glob('*.pt')},
                   filter='butter6_lowpass3400_causal_sos',train_count=len(train.band),dev_count=len(dev.band)))


@torch.inference_mode()
def consensus(mel,wav,data,ids,folder):
    path=folder/'consensus.npz'
    if path.exists():
        with np.load(path) as z:return {uid:tuple(z[uid]) for uid in z.files}
    mel.eval();wav.eval();pseudo={};selected=total=0.;classsec=[0.,0.]
    for start in range(0,len(ids),8):
        group=ids[start:start+8]
        a,_,b=forward(mel,[data.row(i,labeled=False) for i in group])
        q,_,_=forward(wav,[data.row(i,kind='wav',labeled=False) for i in group])
        for j,i in enumerate(group):
            n=int(b['lengths'][j]);w=b['fine_weights'][j,:n]
            y,mask=c.pseudo_mask(a[j,:n].sigmoid().cpu(),q[j,:n].sigmoid().cpu(),w)
            pseudo[data.uids[i]]=(y.numpy(),mask.float().numpy())
            total+=float(w.sum());selected+=float(w[mask].sum())
            for k in (0,1):classsec[k]+=float(w[mask&(y==k)].sum())
    np.savez_compressed(path,**{uid:np.stack(pair) for uid,pair in pseudo.items()})
    save_json(folder/'consensus_selection.json',dict(coverage_percent=100*selected/total,
              real_seconds=classsec[0],fake_seconds=classsec[1],hidden_truth_used=False))
    return pseudo


def fisher_diagonal(model,data,ids):
    model.eval();fisher={k:torch.zeros_like(p) for k,p in model.named_parameters()}
    # cuDNN RNN eval kernels do not support backward. Native GRU preserves
    # dropout-free evaluation semantics while permitting empirical Fisher grads.
    with torch.backends.cudnn.flags(enabled=False):
        for i in ids[:64]:
            model.zero_grad(set_to_none=True)
            y,_,b=forward(model,[data.row(i)])
            c.objective(y,b,DEVICE).backward()
            for k,p in model.named_parameters():
                if p.grad is not None:fisher[k]+=p.grad.detach().square()/64
    model.zero_grad(set_to_none=True)
    return fisher


def train(model,data,ids,seed,arm,kind='mel',epochs=4,lr=3e-4,pseudo=None,fisher=None):
    folder=OUT/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return load(folder/'last.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    anchor={k:p.detach().clone() for k,p in model.named_parameters()} if fisher else None
    history=[]
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(seed*1000+epoch).permutation(ids['labeled']).tolist()
        uorder=np.random.default_rng(seed*2000+epoch).permutation(ids['unlabeled']).tolist()
        totals=[];tick=time.monotonic()
        for step,start in enumerate(range(0,len(order),8)):
            torch.manual_seed(seed*100000+epoch*1000+step)
            selected=order[start:start+8]
            target=arm.startswith('continual_')
            rows=[data.row(i,'band' if target else (i+epoch-1)%3,'wav' if kind=='wav' else 'mel') for i in selected]
            optimizer.zero_grad(set_to_none=True)
            y,h,b=forward(model,rows);sup=c.objective(y,b,DEVICE);aux=sup.new_zeros(())
            if arm=='multitask':aux=.1*boundary_loss(model,h,b)
            if pseudo is not None:
                ug=uorder[step*16:(step+1)*16]
                uy,_,ub=forward(model,[data.row(i,(i+epoch-1)%3,'wav' if kind=='wav' else 'mel',False) for i in ug])
                ty=torch.zeros_like(uy);weights=torch.zeros_like(uy)
                for j,i in enumerate(ug):
                    py,pm=pseudo[data.uids[i]];n=len(py)
                    ty[j,:n]=torch.from_numpy(py).to(DEVICE)
                    weights[j,:n]=torch.from_numpy(pm).to(DEVICE)*ub['fine_weights'][j,:n].to(DEVICE)
                aux=.3*c.balanced_pseudo_loss(uy,ty,weights)
            if arm=='coral':
                ug=uorder[step*16:(step+1)*16]
                _,sh,sb=forward(model,[data.row(i,0,labeled=False) for i in selected])
                _,th,tb=forward(model,[data.row(i,'band',labeled=False) for i in ug])
                aux=100*coral_loss(valid_context(sh,sb),valid_context(th,tb))
            if arm=='continual_ewc':aux=1000*ewc_penalty(model,anchor,fisher)
            if arm=='continual_replay':
                sy,_,sb=forward(model,[data.row(i,0) for i in selected])
                sup=.5*sup+.5*c.objective(sy,sb,DEVICE)
            loss=sup+aux
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();totals.append([float(sup.detach()),float(aux.detach())])
            if step%24==0:status(status='running',seed=seed,arm=arm,epoch=epoch,step=step+1,steps=48)
        a=np.mean(totals,axis=0)
        history.append(dict(epoch=epoch,supervised=float(a[0]),weighted_auxiliary=float(a[1]),seconds=time.monotonic()-tick,updates=epoch*48))
        save_json(folder/'history.json',history);save_model(folder/'last.pt',model,kind,seed)
    save_json(folder/'complete.json',dict(checkpoint_sha256=sha256(folder/'last.pt'),updates=epochs*48,
          parameters=sum(p.numel() for p in model.parameters()),kind=kind))
    return model


def meta_train(data,ids,seed,arm):
    folder=OUT/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return load(folder/'last.pt')
    c.setup(seed+880,4);model=build('meta',seed)
    initial=c.state_digest(model);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(seed+900);history=[]
    for episode in range(200):
        selected=rng.choice(ids['labeled'],16,replace=False).tolist();view=int(rng.integers(3))
        support=[data.row(i,view,'wav') for i in selected[:8]]
        query=[data.row(i,view,'wav') for i in selected[8:]]
        opt.zero_grad(set_to_none=True)
        if arm=='meta_fomaml':
            adapted=copy.deepcopy(model)
            sy,_,sb=forward(adapted,support);sl=c.objective(sy,sb,DEVICE)
            grad=torch.autograd.grad(sl,adapted.parameters())
            with torch.no_grad():
                for p,g in zip(adapted.parameters(),grad):p.add_(g,alpha=-.01)
            qy,_,qb=forward(adapted,query);loss=c.objective(qy,qb,DEVICE)
            qgrad=torch.autograd.grad(loss,adapted.parameters())
            for p,g in zip(model.parameters(),qgrad):p.grad=g.detach().clone()
        else:
            y,_,b=forward(model,support+query);loss=c.objective(y,b,DEVICE);loss.backward()
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        history.append(float(loss.detach()))
        if episode%40==0:status(status='running',seed=seed,arm=arm,episode=episode+1,total=200)
    save_model(folder/'last.pt',model,'meta',seed)
    save_json(folder/'complete.json',dict(checkpoint_sha256=sha256(folder/'last.pt'),updates=200,
               initial_digest=initial,history=history,kind='meta'))
    return model


def meta_adapt(model,data,ids,seed,arm):
    folder=OUT/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    order=np.random.default_rng(seed+1001).permutation(ids['meta_support']).tolist()
    for step in range(5):
        selected=[order[(step*8+j)%32] for j in range(8)]
        opt.zero_grad(set_to_none=True)
        y,_,b=forward(model,[data.row(i,'band','wav') for i in selected])
        loss=c.objective(y,b,DEVICE);loss.backward();opt.step()
    save_model(folder/'last.pt',model,'meta',seed)
    save_json(folder/'complete.json',dict(checkpoint_sha256=sha256(folder/'last.pt'),updates=5,kind='meta',target_support=32))


@torch.inference_mode()
def predict(model,data,ids,kind,view=0):
    model.eval();records=[]
    for start in range(0,len(ids),8):
        selected=ids[start:start+8]
        rows=[data.row(i,view,'wav' if kind in ('wav','meta') else 'mel') for i in selected]
        y,_,b=forward(model,rows)
        for j,r in enumerate(rows):
            n=int(b['lengths'][j]);actual=r['length20']
            records.append(dict(uid=r['uid'],duration=r['duration'],labels20=r['fine_labels'][:actual].numpy(),
                                scores20=y[j,:n].cpu().numpy()))
    return records


def checks():
    result=c.checks()
    x=torch.randn(20,6,requires_grad=True)
    assert float(coral_loss(x,x))==0
    loss=coral_loss(x,torch.randn(23,6));loss.backward()
    assert torch.isfinite(x.grad).all()
    m=nn.Linear(3,1);anchor={k:p.detach().clone() for k,p in m.named_parameters()}
    fisher={k:torch.ones_like(p) for k,p in m.named_parameters()}
    assert float(ewc_penalty(m,anchor,fisher))==0
    with torch.no_grad():m.weight.add_(.1)
    assert float(ewc_penalty(m,anchor,fisher))>0
    result.update(coral_identity_zero=True,coral_gradient_finite=True,ewc_anchor_zero_and_displacement_positive=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);c.setup(666,4)
    train_data,dev=Data('train'),Data('dev')
    plan=json.loads(json.dumps(plan_for(train_data,dev)))
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan,'Protocol changed; do not overwrite.'
    else:save_json(OUT/'plan.json',plan)
    save_json(OUT/'checks.json',checks())
    if args.check:print('Checks passed; protocol frozen.',flush=True);return
    integrity={}
    for split in ('train','dev'):
        expected=read_json(WROOT/split/'complete.json')['features_sha256']
        integrity[split]=sha256(WROOT/split/'features.npy')==expected
    assert all(integrity.values());save_json(OUT/'wavlm_cache_integrity.json',integrity)
    ids=plan['splits'];prepare_band(train_data,dev,ids)
    for seed in SEEDS:
        folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
        for arm,old in [('mel_control','supervised'),('selftraining_reference','selftraining'),('continual_source','warm')]:
            model=load(OLD/f'seed{seed}'/old/'last.pt')
            save_model(folder/arm/'last.pt',model,'mel',seed)
            save_json(folder/arm/'complete.json',dict(kind='mel',checkpoint_sha256=sha256(folder/arm/'last.pt'),
                      reused_from=str(OLD/f'seed{seed}'/old/'last.pt')))
        c.setup(seed,4)
        mt=build('boundary',seed);mt.cnn.load_state_dict(fresh_warm(seed).state_dict())
        train(mt,train_data,ids,seed,'multitask','boundary')
        c.setup(seed,4)
        wav=train(build('wav',seed),train_data,ids,seed,'wavlm_warm','wav',epochs=20,lr=.001)
        train(copy.deepcopy(wav),train_data,ids,seed,'wavlm_control','wav')
        pseudo=consensus(fresh_warm(seed),wav,train_data,ids['unlabeled'],folder)
        train(fresh_warm(seed),train_data,ids,seed,'cotraining_mel',pseudo=pseudo)
        train(copy.deepcopy(wav),train_data,ids,seed,'cotraining_wavlm','wav',pseudo=pseudo)
        del wav;torch.cuda.empty_cache()
        train(fresh_warm(seed),train_data,ids,seed,'coral')
        source=fresh_warm(seed);fisher=fisher_diagonal(source,train_data,ids['labeled'])
        torch.save({k:v.cpu() for k,v in fisher.items()},folder/'fisher.pt')
        for arm in ('continual_naive','continual_ewc','continual_replay'):
            train(fresh_warm(seed),train_data,ids,seed,arm,fisher=fisher if arm=='continual_ewc' else None)
        for arm in ('meta_erm','meta_fomaml'):
            model=meta_train(train_data,ids,seed,arm);meta_adapt(model,train_data,ids,seed,arm+'_adapt')
    frozen={}
    for seed in SEEDS:
        for arm in ARMS:
            folder=OUT/f'seed{seed}'/arm;kind=read_json(folder/'complete.json')['kind']
            key=f'{seed}_{arm}'
            if arm.startswith('continual_') and arm!='continual_source':threshold_source=f'{seed}_continual_source'
            elif arm.endswith('_adapt'):threshold_source=f'{seed}_{arm[:-6]}'
            else:threshold_source=key
            if threshold_source==key:
                records=predict(load(folder/'last.pt'),dev,ids['calibration'],kind)
                threshold=choose_threshold(records,.05)['threshold']
            else:threshold=frozen[threshold_source]['threshold']
            frozen[key]=dict(checkpoint_sha256=sha256(folder/'last.pt'),threshold=threshold,threshold_source=threshold_source,kind=kind)
            status(status='running',stage='calibration',model=key)
    save_json(OUT/'frozen_thresholds.json',frozen)
    results={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f'{seed}_{arm}';folder=OUT/f'seed{seed}'/arm
            assert sha256(folder/'last.pt')==frozen[key]['checkpoint_sha256']
            model=load(folder/'last.pt');results[key]={}
            for condition,view in [('clean',0),('band','band')]:
                records=predict(model,dev,ids['audit'],frozen[key]['kind'],view)
                report=evaluate_report(records,frozen[key]['threshold'])
                save_json(folder/f'{condition}_metrics.json',report)
                np.savez_compressed(folder/f'{condition}_scores.npz',**{r['uid']:r['scores20'] for r in records})
                results[key][condition]={k:v for k,v in report.items() if k!='per_recording'}
                status(status='running',stage='audit',model=key,condition=condition,event_f1=report['event_f1_iou05_percent'])
    save_json(OUT/'comparison.json',dict(results=results,formal_test_accessed=False,all_training_frozen_before_audit=True))
    status(status='complete',stage='finished')


if __name__=='__main__':main()
