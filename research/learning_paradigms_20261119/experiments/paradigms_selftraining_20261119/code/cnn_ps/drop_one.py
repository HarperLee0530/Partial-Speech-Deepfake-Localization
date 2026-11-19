"""Dev-only single-factor ablations of the frozen boundary architecture.

No eval/test argument exists. Historical full model is a read-only reference.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .backbone import CNNBiGRU
from .dual_features import FeatureData
from .finetune import fine_collate,objective as baseline_objective,save_scores
from .io import read_json,sha256,manifest_hash
from .metrics20 import evaluate_dev,decision_mask
from .train_dual import forward_batch,infer,boundary_targets,weighted_bce
from .train_four import ARMS,model_config,setup,make_adapter,state_digest,TrainingView,_cpu_state,is_better

REFERENCE=Path('D:/SAL_精简实验/result/ps_four_eventf1_20261115_run1/boundary')
DEFAULT_OUTPUT=Path('D:/SAL_精简实验/result/ps_boundary_drop_one_20261116_run1')
ARM_CONFIG={
    'no_boundary':dict(boundary_weight=0.,consistency_weight=.05,learned_pool=True),
    'no_consistency':dict(boundary_weight=.1,consistency_weight=0.,learned_pool=True),
    'fixed_pool':dict(boundary_weight=.1,consistency_weight=.05,learned_pool=False),
}

def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def save_torch(path,payload):
    path=Path(path);temp=path.with_name(path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)

def adapter_for(arm,seed,device):
    # Construct FULL first: omitting pool construction would change the random
    # initialization of boundary_output, which follows the pool constructor.
    adapter=make_adapter(ARMS['boundary'],seed,device)
    if not ARM_CONFIG[arm]['learned_pool']:
        adapter.learned_pool=False
        del adapter.coarse_downsample
    return adapter

def loss_for(final,outputs,batch,device,boundary_weight=.1,consistency_weight=.05):
    """Same arithmetic/components as train_dual.training_loss; two weights vary."""
    main=baseline_objective(final,batch,device)
    all_coarse=torch.cat([r['coarse_logits'] for r in outputs])
    coarse_targets=torch.cat([batch['labels'][i,:len(r['coarse_logits'])] for i,r in enumerate(outputs)]).to(device)
    coarse_weights=torch.cat([batch['weights'][i,:len(r['coarse_logits'])] for i,r in enumerate(outputs)]).to(device)
    auxiliary=weighted_bce(all_coarse,coarse_targets,coarse_weights)
    logits,labels,weights=[],[],[]
    same_numerator,same_denominator=main.new_zeros(()),main.new_zeros(())
    for i,r in enumerate(outputs):
        n=len(r['final_logits']);y=batch['fine_labels'][i,:n].to(device);w=batch['fine_weights'][i,:n].to(device)
        target,pair=boundary_targets(y,w);logits.append(r['boundary_logits']);labels.append(target);weights.append(pair)
        inside=torch.minimum(w[1:],w[:-1])*(y[1:]==y[:-1]);probability=r['final_logits'].sigmoid()
        same_numerator=same_numerator+((probability[1:]-probability[:-1]).square()*inside).sum()
        same_denominator=same_denominator+inside.sum()
    boundary=weighted_bce(torch.cat(logits),torch.cat(labels),torch.cat(weights))
    consistency=same_numerator/same_denominator.clamp_min(1e-8)
    total=main+.5*auxiliary+boundary_weight*boundary+consistency_weight*consistency
    return total,dict(main=float(main.detach()),coarse_aux=float(auxiliary.detach()),boundary=float(boundary.detach()),consistency=float(consistency.detach()))

def add_frame_metrics(metrics,records):
    cm=np.zeros(4,dtype=np.int64)
    for r in records:
        y=np.asarray(r['labels20'],dtype=bool);n=len(y);t=np.arange(n)*.02;active=t<r['duration']
        y=y[active];p=decision_mask(np.asarray(r['scores20'][:n])[active],metrics['threshold'])
        cm+=[int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum()),int((~p&~y).sum())]
    tp,fp,fn,tn=map(int,cm)
    metrics.update(frame_f1_percent=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,frame_confusion_tp_fp_fn_tn=cm.tolist())

def initialization_audit(model,adapter,arm):
    ref=torch.load(REFERENCE/'epochs/epoch_00.pt',map_location='cpu',weights_only=False)
    for key,value in model.state_dict().items():assert torch.equal(value.cpu(),ref['backbone_state_dict'][key]),key
    for key,value in adapter.state_dict().items():assert torch.equal(value.cpu(),ref['adapter_state_dict'][key]),key
    assert set(ref['adapter_state_dict'])-set(adapter.state_dict())==({'coarse_downsample.weight'} if arm=='fixed_pool' else set())
    return dict(backbone_exact=True,all_common_adapter_tensors_exact=True,removed_state_keys=['coarse_downsample.weight'] if arm=='fixed_pool' else [])

def train_arm(args,train,dev,arm):
    folder=args.output/arm
    if (folder/'complete.json').is_file():
        complete=read_json(folder/'complete.json');assert sha256(folder/'best.pt')==complete['checkpoint_sha256'];return complete
    if folder.exists() and not args.resume:raise FileExistsError(f'Existing unfinished arm: {folder}; use --resume')
    folder.mkdir(parents=True,exist_ok=True);(folder/'epochs').mkdir(exist_ok=True)
    setup(args.seed,args.threads);model=CNNBiGRU(model_config()).to(args.device);adapter=adapter_for(arm,args.seed,args.device)
    audit=initialization_audit(model,adapter,arm);initial=state_digest(model);config=ARM_CONFIG[arm]
    options=dict(ARMS['boundary'],learned_pool=config['learned_pool'])
    reference=read_json(REFERENCE/'protocol.json')
    protocol=copy.deepcopy(reference)
    protocol.update(schema='boundary_drop_one_v1',arm=arm,adapter_options=options,reference=str(REFERENCE),single_change=config,
                    losses=dict(reference['losses'],boundary=config['boundary_weight'],consistency=config['consistency_weight']),
                    adapter_parameters=sum(p.numel() for p in adapter.parameters()),
                    initialization_audit=audit,initial_backbone_sha256=initial,source_drop_one_sha256=sha256(__file__),
                    runtime=dict(python=__import__('sys').version.split()[0],torch=torch.__version__,numpy=np.__version__),
                    boundary_head_note='Retained for identical shared initialization; no_boundary removes only its supervision weight. It does not feed final logits with reliability_gate=False.',
                    epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,threads=args.threads,
                    train_cache=str(args.train_cache),dev_cache=str(args.dev_cache),eval_read=False,test_threshold_selection=False)
    assert initial==reference['initial_backbone_sha256']
    assert len(train)==reference['train_recordings'] and len(dev)==reference['dev_recordings']
    assert manifest_hash(train.uids)==reference['train_uids_sha256'] and manifest_hash(dev.uids)==reference['dev_uids_sha256']
    if (folder/'protocol.json').exists():assert read_json(folder/'protocol.json')==protocol,'Protocol changed on resume'
    else:write(folder/'protocol.json',protocol)
    parameters=list(model.parameters())+list(adapter.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=1e-4)
    view=TrainingView(train,clean_only=False);generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(view,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=fine_collate,generator=generator)
    best=None;history=[];updates=0;first_epoch=0;prior_seconds=0.
    if (folder/'resume.pt').exists():
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['backbone']);adapter.load_state_dict(state['adapter']);optimizer.load_state_dict(state['optimizer'])
        for opt in optimizer.state.values():
            for name,value in opt.items():
                if isinstance(value,torch.Tensor):opt[name]=value.to(args.device) if name!='step' else value
        best=state['best'];history=state['history'];updates=state['updates'];first_epoch=state['epoch']+1;prior_seconds=state['elapsed_seconds']
        random.setstate(state['random']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu_rng']);generator.set_state(state['loader_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.monotonic()
    for epoch in range(first_epoch,args.epochs+1):
        tick=time.monotonic();losses=[];components=[]
        if epoch:
            view.epoch=epoch;model.train();adapter.train();last=time.monotonic()
            for index,batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                final,outputs=forward_batch(model,adapter,batch,args.device)
                loss,parts=loss_for(final,outputs,batch,args.device,config['boundary_weight'],config['consistency_weight'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
                updates+=1;losses.append(float(loss.detach()));components.append(parts)
                if time.monotonic()-last>=30:
                    progress=dict(arm=arm,phase='train',epoch=epoch,batches=index+1,batches_total=len(loader),updates=updates,
                                  loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-tick)
                    write(folder/'progress.json',progress);print(json.dumps(progress),flush=True);last=time.monotonic()
        train_seconds=time.monotonic()-tick
        write(folder/'progress.json',dict(arm=arm,phase='dev',epoch=epoch,updates=updates))
        records,scores,metadata=infer(model,adapter,dev,args.device)
        metrics=evaluate_dev(records,max_real_fpr=.05);metrics.pop('selection_score_j_percent',None);add_frame_metrics(metrics,records)
        assert metrics['real_fpr_percent']<=5+1e-10
        if epoch==0:
            with np.load(REFERENCE/'epochs/epoch_00_scores20.npz',allow_pickle=False) as old:
                assert np.array_equal(scores,old['scores20']),'Initial logits differ from historical full reference'
            write(folder/'epoch_zero_audit.json',dict(full_dev_scores_bitwise_identical=True,recordings=len(dev),**audit))
        value=float(metrics['event_f1_iou05_percent']);selected=dict(epoch=epoch,threshold=float(metrics['threshold']),event_f1_iou05_percent=value)
        prefix=folder/'epochs'/f'epoch_{epoch:02d}'
        payload=dict(schema='boundary_drop_one_v1',backbone_state_dict=_cpu_state(model),adapter_state_dict=_cpu_state(adapter),
                     adapter_options=options,model_config=model.config.to_dict(),protocol=protocol,dev_metrics=metrics,**selected)
        save_torch(prefix.with_suffix('.pt'),payload);save_scores(prefix.with_name(prefix.name+'_scores20.npz'),scores,metadata)
        write(prefix.with_name(prefix.name+'_metrics.json'),metrics)
        if is_better(value,best):
            best=selected;shutil.copyfile(prefix.with_suffix('.pt'),folder/'best.pt')
            shutil.copyfile(prefix.with_name(prefix.name+'_scores20.npz'),folder/'dev_scores20.npz');write(folder/'dev_metrics.json',metrics)
        row=dict(arm=arm,epoch=epoch,updates=updates,loss=float(np.mean(losses)) if losses else None,
                 loss_components={key:float(np.mean([item[key] for item in components])) for key in components[0]} if components else {},
                 train_seconds=train_seconds,total_seconds=time.monotonic()-tick,best_epoch=best['epoch'],
                 **{key:metrics[key] for key in ('threshold','frame_f1_percent','event_f1_iou05_percent','real_fpr_percent','short_event_recall_le320ms_percent')})
        history.append(row);write(folder/'history.json',history);write(folder/'progress.json',dict(phase='epoch_complete',**row))
        save_torch(folder/'resume.pt',dict(backbone=_cpu_state(model),adapter=_cpu_state(adapter),optimizer=optimizer.state_dict(),
                    best=best,history=history,updates=updates,epoch=epoch,elapsed_seconds=prior_seconds+time.monotonic()-started,
                    random=random.getstate(),numpy=np.random.get_state(),cpu_rng=torch.get_rng_state(),loader_rng=generator.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None))
        print(json.dumps(dict(phase='epoch_complete',**row)),flush=True)
    complete=dict(status='complete',arm=arm,**best,trained_epochs=args.epochs,updates=updates,eval_read=False,
                  checkpoint_sha256=sha256(folder/'best.pt'),dev_scores20_sha256=sha256(folder/'dev_scores20.npz'),
                  dev_metrics_sha256=sha256(folder/'dev_metrics.json'),protocol_sha256=sha256(folder/'protocol.json'),
                  last_backbone_sha256=state_digest(model),total_seconds=prior_seconds+time.monotonic()-started)
    write(folder/'complete.json',complete);return complete

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--arms',nargs='+',choices=tuple(ARM_CONFIG),default=list(ARM_CONFIG))
    parser.add_argument('--train-cache',type=Path,default=Path('D:/SAL_精简实验/data/cnn_ps_features_v2/train'))
    parser.add_argument('--dev-cache',type=Path,default=Path('D:/SAL_精简实验/data/cnn_ps_features_v2/dev'))
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--seed',type=int,default=666)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.seed!=666 or args.epochs!=20 or args.batch_size!=8:parser.error('This paired experiment fixes seed666,20epochs,batch8')
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA required but unavailable')
    setup(args.seed,args.threads);reference=read_json(REFERENCE/'protocol.json')
    for key,folder in [('train',args.train_cache),('dev',args.dev_cache)]:
        assert sha256(folder/'complete.json')==reference[f'{key}_cache_metadata_sha256']
    source_check={}
    for name,h in reference['source_sha256'].items():
        path=Path(__file__).with_name(name)
        normalized=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        historical=Path('D:/Partial-Speech-DualResolution-experiment/code/cnn_ps')/name
        assert sha256(historical)==h,('Historical source snapshot mismatch',name)
        assert path.read_bytes().replace(b'\r\n',b'\n')==historical.read_bytes().replace(b'\r\n',b'\n'),('Reference code changed beyond CRLF/LF',name)
        source_check[name]=dict(historical_sha256=h,historical_path=str(historical),current_raw_sha256=sha256(path),normalized_lf_sha256=normalized)
    args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(status='declared_before_training',arms=ARM_CONFIG,execution_order=args.arms,epochs=20,batch_size=8,seed=666,
              reference=str(REFERENCE),reference_checkpoint_sha256=sha256(REFERENCE/'best.pt'),source_sha256=sha256(__file__),source_verification=source_check,
              data='Same immutable train/dev caches and augmentation rotation as full reference',
              selection='Dev true-duration FPR<=5%, max eventF1, earliest ties; no test input accepted',
              initialization='Exact full epoch0 backbone and common adapter tensors; fixed_pool deletes only pool weight after full construction',
              no_boundary='Only boundary loss weight0; inactive head retained to isolate supervision effect',
              no_consistency='Only consistency weight0',fixed_pool='Duration-weighted fixed mean replaces learnable per-channel pooling',
              limitations=['One shared seed; historical full run reused. No independent validation or seed stability claim.'])
    if (args.output/'plan.json').exists():assert read_json(args.output/'plan.json')==plan
    else:write(args.output/'plan.json',plan)
    train=FeatureData(args.train_cache,split='train',seed=args.seed,config=model_config(),verify_hashes=True)
    dev=FeatureData(args.dev_cache,split='dev',seed=args.seed,config=model_config(),verify_hashes=True)
    assert not(set(train.uids)&set(dev.uids))
    done={}
    for arm in args.arms:
        write(args.output/'status.json',dict(status='running',active_arm=arm,completed=list(done)))
        done[arm]=train_arm(args,train,dev,arm)
        if args.device=='cuda':torch.cuda.empty_cache()
    assert sha256(REFERENCE/'best.pt')==plan['reference_checkpoint_sha256']
    write(args.output/'status.json',dict(status='complete',completed=done,test_accessed=False))

if __name__=='__main__':main()
