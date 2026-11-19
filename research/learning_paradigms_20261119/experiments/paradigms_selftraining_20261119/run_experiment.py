"""Small preregistered self-training experiment; never opens eval/LPS data.

Run with D:/SAL_精简实验/code/.venv/Scripts/python.exe -X utf8 run_experiment.py
The unlabeled interface never reads ground truth. Truth is unsealed only in the
post-training diagnostic, after ALL checkpoints and thresholds are frozen.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from cnn_ps.backbone import CNNBiGRU, LogMel
from cnn_ps.train_four import model_config, setup, state_digest
from cnn_ps.finetune import fine_collate, objective
from cnn_ps.augmentation import transform
from cnn_ps.metrics20 import choose_threshold
from cnn_ps.report_native20 import evaluate_report
from cnn_ps.io import read_json, save_json, sha256

CACHE = Path('D:/SAL_精简实验/data/cnn_ps_features_v2')
RESULT = ROOT / 'results'
SEEDS = [666, 667]
WARM_EPOCHS, BRANCH_EPOCHS = 8, 4
BATCH, UBATCH = 8, 16
CONFIDENCE, GUARD, LAMBDA = .9, 2, .3
DEVICE = 'cuda'
START = time.monotonic()


def progress(**kw):
    kw['elapsed_seconds'] = round(time.monotonic() - START, 2)
    save_json(RESULT / 'status.json', kw)
    print(json.dumps(kw, ensure_ascii=False), flush=True)


class Cache:
    def __init__(self, split):
        assert split in ('train', 'dev')
        self.folder = CACHE / split
        self.meta = read_json(self.folder / 'complete.json')
        assert self.meta['split'] == split and self.meta['status'] == 'complete'
        self.uids = self.meta['uids']
        self.x = np.load(self.folder / 'features.npy', mmap_mode='r')
        self.offsets = self.meta['offsets']
        self.labels = None

    def features(self, index, channel=0):
        a, b = self.offsets[index:index+2]
        return torch.from_numpy(np.array(self.x[channel, a:b], copy=True)).unsqueeze(0)

    def bare(self, index, channel=0):
        # Deliberately no label-array access in this interface.
        x = self.features(index, channel)
        n = x.shape[1]
        duration = float(self.meta['durations'][index])
        fine_weights = np.maximum(0., np.minimum((np.arange(n)+1)*.02, duration)-np.arange(n)*.02)
        coarse_weights = np.maximum(0., np.minimum((np.arange(n//8)+1)*.16, duration)-np.arange(n//8)*.16)
        return dict(uid=self.uids[index], features=x, labels=torch.zeros(n//8),
                    fine_labels=torch.zeros(n), fine_weights=torch.tensor(fine_weights, dtype=torch.float32),
                    weights=torch.tensor(coarse_weights, dtype=torch.float32), duration=duration,
                    length20=self.meta['lengths20'][index], channel=channel)

    def labeled(self, index, channel=0):
        if self.labels is None:
            self.labels = {k: np.load(self.folder / f'{k}.npy', mmap_mode='r')
                           for k in ('labels20', 'labels160')}
        row = self.bare(index, channel)
        a, b = self.offsets[index:index+2]
        row['fine_labels'] = torch.tensor(np.array(self.labels['labels20'][a:b], copy=True))
        row['labels'] = torch.tensor(np.array(self.labels['labels160'][a//8:b//8], copy=True))
        return row


def make_plan(train, dev):
    rng = np.random.default_rng(20261119)
    ti = rng.permutation(len(train.uids)).tolist()
    di = rng.permutation(len(dev.uids)).tolist()
    splits = dict(labeled=ti[:384], unlabeled=ti[384:1152], calibration=di[:256], audit=di[256:640])
    assert not set(train.uids) & set(dev.uids)
    assert not set(splits['labeled']) & set(splits['unlabeled'])
    assert not set(splits['calibration']) & set(splits['audit'])
    return dict(schema='course_selftraining_pilot_v1', splits=splits,
        uids={k: [(train if k in ('labeled', 'unlabeled') else dev).uids[i] for i in v]
              for k, v in splits.items()}, seeds=SEEDS, warm_epochs=WARM_EPOCHS,
        branch_epochs=BRANCH_EPOCHS, batch=BATCH, unlabeled_batch=UBATCH,
        confidence=CONFIDENCE, boundary_guard_frames=GUARD, unsupervised_weight=LAMBDA,
        pseudo_rule='Fixed warm teacher; clean/EQ predictions agree, both confidence>=0.9; exclude +/-2 frames at either predicted transition; hard labels; balance real/fake losses.',
        arms=['supervised', 'selftraining'], model_config=model_config().to_dict(),
        optimization='AdamW wd=1e-4; warm lr=1e-3; branches reset AdamW lr=3e-4; grad clip=5',
        paired_control='Identical warm checkpoint, labeled order/views/dropout seeds and optimizer steps. SSL additionally sees unlabeled data and uses extra computation.',
        views='Both arms rotate clean/noise20dB/mildEQ labeled views; SSL rotates same views for unlabeled data.',
        selection='Fixed final epoch, no checkpoint selection; thresholds calibrated on 256 dev recordings at genuine-duration FPR<=5%.',
        evaluation='384 disjoint dev recordings clean and synthetic waveform noise20dB; SAME clean-calibrated threshold for noise; primary event F1@IoU0.5.',
        limitations=['Hidden labels simulate scarcity; no genuinely new unlabeled data.',
                    'UID-disjoint only; source/speaker independence not established.',
                    'Dev historically used in earlier experiments; audit is fresh only for this training run.',
                    'Noise condition is seen as augmentation; robustness check, not unseen-domain proof.',
                    'Small fixed training budget; no comparison to fully supervised full-data model.'],
        test_data_accessed=False, pretrained_weights_used=False,
        cache_metadata_sha256={s: sha256(CACHE/s/'complete.json') for s in ('train','dev')},
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                       [Path(__file__), *sorted((ROOT/'code'/'cnn_ps').glob('*.py'))]})


def infer_batch(model, rows):
    batch = fine_collate(rows)
    return model(batch['features'].to(DEVICE), batch['lengths']), batch


def pseudo_mask(p, q, weights):
    yp, yq = p >= .5, q >= .5
    conf = (torch.maximum(p, 1-p) >= CONFIDENCE) & (torch.maximum(q, 1-q) >= CONFIDENCE)
    mask = conf & (yp == yq) & (weights > 0)
    changes = torch.zeros_like(mask)
    changes[1:] = (yp[1:] != yp[:-1]) | (yq[1:] != yq[:-1])
    excluded = F.max_pool1d(changes.float()[None,None], 2*GUARD+1, stride=1, padding=GUARD)[0,0] > 0
    return yp.float(), mask & ~excluded


def balanced_pseudo_loss(logits, labels, weights):
    active = weights > 0
    pieces = []
    for cls in (0, 1):
        m = active & (labels == cls)
        if m.any():
            terms = F.binary_cross_entropy_with_logits(logits[m], labels[m], reduction='none')
            pieces.append((terms * weights[m]).sum()/weights[m].sum())
    return torch.stack(pieces).mean() if pieces else logits.sum()*0


def checks():
    p = torch.tensor([.01]*8+[.99]*8)
    y, mask = pseudo_mask(p, p, torch.ones(16))
    assert not mask[6:11].any() and mask[:6].all() and mask[11:].all()
    assert not pseudo_mask(p, 1-p, torch.ones(16))[1].any()
    assert not pseudo_mask(p, p, torch.zeros(16))[1].any()
    x = torch.zeros(4, requires_grad=True)
    loss = balanced_pseudo_loss(x, torch.tensor([0.,0.,1.,1.]), torch.tensor([1.,0.,1.,0.]))
    loss.backward()
    assert x.grad[1] == 0 and x.grad[3] == 0 and torch.isfinite(loss)
    assert abs(float(loss)-np.log(2)) < 1e-6
    zero = balanced_pseudo_loss(x, torch.zeros(4), torch.zeros(4))
    assert float(zero) == 0
    return dict(boundary_exclusion=True, disagreement_rejected=True, padding_excluded=True,
                masked_loss_gradient=True, empty_mask_safe=True)


def save_model(path, model, **kw):
    torch.save(dict(model_config=model.config.to_dict(), state_dict={k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()}, **kw), path)


def load_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = CNNBiGRU(model_config()).to(DEVICE)
    model.load_state_dict(payload['state_dict'])
    return model


def train_epochs(model, train, ids, seed, folder, epochs, lr, pseudo=None, uids=None):
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        order = np.random.default_rng(seed*1000+epoch).permutation(ids).tolist()
        uorder = None if uids is None else np.random.default_rng(seed*2000+epoch).permutation(uids).tolist()
        totals = []
        tick = time.monotonic()
        for step, start in enumerate(range(0, len(order), BATCH)):
            # Identical labeled-dropout stream despite the extra SSL forward pass.
            torch.manual_seed(seed*100000+epoch*1000+step)
            rows = [train.labeled(i, (i+epoch-1)%3) for i in order[start:start+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            logits, batch = infer_batch(model, rows)
            supervised = objective(logits, batch, DEVICE)
            unsupervised = supervised.new_zeros(())
            if pseudo is not None:
                selected = uorder[step*UBATCH:(step+1)*UBATCH]
                assert len(selected) == UBATCH
                # No unlabeled ground truth is requested or included in the loss.
                urows = [train.bare(i, (i+epoch-1)%3) for i in selected]
                ul, ub = infer_batch(model, urows)
                targets = torch.zeros_like(ul)
                weights = torch.zeros_like(ul)
                for j, i in enumerate(selected):
                    y, m = pseudo[train.uids[i]]
                    n = len(y)
                    targets[j,:n] = torch.from_numpy(y).to(DEVICE)
                    weights[j,:n] = torch.from_numpy(m).to(DEVICE)*ub['fine_weights'][j,:n].to(DEVICE)
                unsupervised = balanced_pseudo_loss(ul, targets, weights)
            loss = supervised + LAMBDA*unsupervised
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            totals.append([float(supervised.detach()),float(unsupervised.detach())])
            if step % 16 == 0:
                progress(status='running', seed=seed, stage=folder.name, epoch=epoch,
                         step=step+1, steps=len(order)//BATCH)
        values = np.mean(totals, axis=0)
        history.append(dict(epoch=epoch, supervised_loss=float(values[0]), pseudo_loss=float(values[1]),
                            seconds=time.monotonic()-tick, updates=epoch*len(order)//BATCH))
        save_json(folder/'history.json', history)
        save_model(folder/'last.pt', model, seed=seed, epoch=epoch)
        progress(status='running', seed=seed, stage=folder.name, epoch=epoch, epoch_complete=True,
                 supervised_loss=float(values[0]), pseudo_loss=float(values[1]))
    save_json(folder/'complete.json', dict(checkpoint_sha256=sha256(folder/'last.pt'),
                epochs=epochs, updates=epochs*len(ids)//BATCH, state_digest=state_digest(model)))


@torch.inference_mode()
def generate_pseudo(model, train, ids, folder):
    model.eval()
    pseudo = {}
    stats = dict(total_seconds=0., selected_seconds=0., selected_fake_seconds=0., selected_real_seconds=0.)
    for start in range(0,len(ids),BATCH):
        selected = ids[start:start+BATCH]
        clean, batch = infer_batch(model, [train.bare(i,0) for i in selected])
        eq, _ = infer_batch(model, [train.bare(i,2) for i in selected])
        for j,i in enumerate(selected):
            n = int(batch['lengths'][j])
            w = batch['fine_weights'][j,:n]
            y, mask = pseudo_mask(clean[j,:n].sigmoid().cpu(),eq[j,:n].sigmoid().cpu(),w)
            pseudo[train.uids[i]] = (y.numpy(),mask.numpy().astype(np.float32))
            stats['total_seconds'] += float(w.sum())
            stats['selected_seconds'] += float(w[mask].sum())
            stats['selected_fake_seconds'] += float(w[mask & (y==1)].sum())
            stats['selected_real_seconds'] += float(w[mask & (y==0)].sum())
        if start % 128 == 0:
            progress(status='running', stage='pseudo_labels', completed=start+len(selected), total=len(ids))
    stats['coverage_percent'] = 100*stats['selected_seconds']/stats['total_seconds']
    np.savez_compressed(folder/'pseudo.npz', **{uid:np.stack(pair) for uid,pair in pseudo.items()})
    save_json(folder/'pseudo_selection.json',stats)
    return pseudo


@torch.inference_mode()
def predict(model, dev, ids, noise=None):
    model.eval()
    records = []
    for i in ids:
        row = dev.labeled(i)
        if noise is not None:
            row['features'] = noise[dev.uids[i]]
        logits, _ = infer_batch(model,[row])
        records.append(dict(uid=row['uid'], duration=row['duration'],
                    labels20=row['fine_labels'][:row['length20']].numpy(),
                    scores20=logits[0].cpu().numpy()))
    return records


def noise_features(dev, ids):
    frontend = LogMel(model_config())
    output = {}
    for j,i in enumerate(ids):
        path = dev.meta['records'][i]['audio']
        wave, sr = sf.read(path,dtype='float32')
        assert sr==16000 and wave.ndim==1
        assert abs(len(wave)/sr-dev.meta['durations'][i]) < 1e-9
        n = dev.offsets[i+1]-dev.offsets[i]
        padded = F.pad(torch.from_numpy(transform(wave,dev.uids[i],1)),(0,n*320-len(wave)))
        with torch.inference_mode():
            output[dev.uids[i]] = frontend(padded)
        assert output[dev.uids[i]].shape[1] == n
        if j%128==0:
            progress(status='running',stage='noise_check_features',completed=j+1,total=len(ids))
    return output


def pseudo_audit(train, ids, pseudo):
    # ONLY called after all training/threshold freezing; diagnostic, never feedback.
    selected=correct=fake=correct_fake=real=correct_real=0.
    for i in ids:
        row=train.labeled(i)
        y,m=pseudo[train.uids[i]]
        w=row['fine_weights'].numpy()*m
        truth=row['fine_labels'].numpy()
        selected+=float(w.sum());correct+=float(w[y==truth].sum())
        fake+=float(w[y==1].sum());correct_fake+=float(w[(y==1)&(truth==1)].sum())
        real+=float(w[y==0].sum());correct_real+=float(w[(y==0)&(truth==0)].sum())
    return dict(selected_duration_accuracy_percent=100*correct/selected if selected else None,
                fake_pseudo_precision_percent=100*correct_fake/fake if fake else None,
                real_pseudo_precision_percent=100*correct_real/real if real else None,
                diagnostic_only=True, performed_after_all_training=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    setup(666,4)
    assert torch.cuda.is_available()
    RESULT.mkdir(parents=True,exist_ok=True)
    train,dev=Cache('train'),Cache('dev')
    plan=json.loads(json.dumps(make_plan(train,dev)))
    if (RESULT/'plan.json').exists():
        assert read_json(RESULT/'plan.json') == plan, 'Protocol/source changed; use a new output directory.'
    else:
        save_json(RESULT/'plan.json',plan)
    check=checks()
    check.update(train_dev_disjoint=True,labeled_unlabeled_disjoint=True,calibration_audit_disjoint=True,
                 cuda=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    save_json(RESULT/'checks.json',check)
    if args.check:
        print(json.dumps(check),flush=True)
        return
    ids=plan['splits']
    for seed in SEEDS:
        folder=RESULT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        setup(seed,4)
        model=CNNBiGRU(model_config()).to(DEVICE)
        save_json(folder/'initialization.json',dict(seed=seed,state_digest=state_digest(model),pretrained=False))
        if not (folder/'warm'/'complete.json').exists():
            train_epochs(model,train,ids['labeled'],seed,folder/'warm',WARM_EPOCHS,1e-3)
        model=load_model(folder/'warm'/'last.pt')
        if (folder/'pseudo.npz').exists():
            with np.load(folder/'pseudo.npz') as z:
                pseudo={uid:tuple(z[uid]) for uid in z.files}
        else:
            pseudo=generate_pseudo(model,train,ids['unlabeled'],folder)
        for arm in ('supervised','selftraining'):
            if (folder/arm/'complete.json').exists():
                continue
            model=load_model(folder/'warm'/'last.pt')
            assert state_digest(model)==read_json(folder/'warm'/'complete.json')['state_digest']
            train_epochs(model,train,ids['labeled'],seed,folder/arm,BRANCH_EPOCHS,3e-4,
                         pseudo=pseudo if arm=='selftraining' else None,
                         uids=ids['unlabeled'] if arm=='selftraining' else None)
    # Freeze all model identities and thresholds BEFORE any audit-set evaluation.
    frozen={}
    for seed in SEEDS:
        for arm in ('supervised','selftraining'):
            key=f'{seed}_{arm}';path=RESULT/f'seed{seed}'/arm/'last.pt'
            model=load_model(path)
            calibration=predict(model,dev,ids['calibration'])
            frozen[key]=dict(checkpoint_sha256=sha256(path),**choose_threshold(calibration,.05))
            progress(status='running',stage='calibrating',model=key)
    save_json(RESULT/'frozen_thresholds.json',frozen)
    noise=noise_features(dev,ids['audit'])
    all_results={}
    for seed in SEEDS:
        for arm in ('supervised','selftraining'):
            key=f'{seed}_{arm}';folder=RESULT/f'seed{seed}'/arm
            assert sha256(folder/'last.pt')==frozen[key]['checkpoint_sha256']
            model=load_model(folder/'last.pt')
            all_results[key]={}
            for condition,features in [('clean',None),('noise20db',noise)]:
                records=predict(model,dev,ids['audit'],features)
                report=evaluate_report(records,threshold=frozen[key]['threshold'])
                save_json(folder/f'{condition}_metrics.json',report)
                np.savez_compressed(folder/f'{condition}_scores.npz',**{r['uid']:r['scores20'] for r in records})
                all_results[key][condition]={k:v for k,v in report.items() if k!='per_recording'}
                progress(status='running',stage='audit',model=key,condition=condition,
                         event_f1=report['event_f1_iou05_percent'],frame_f1=report['frame_f1_percent'])
        folder=RESULT/f'seed{seed}'
        with np.load(folder/'pseudo.npz') as z:
            pseudo={uid:tuple(z[uid]) for uid in z.files}
        save_json(folder/'pseudo_quality_posthoc.json',pseudo_audit(train,ids['unlabeled'],pseudo))
    save_json(RESULT/'comparison.json',dict(results=all_results,plan_sha256=sha256(RESULT/'plan.json'),
                 test_data_accessed=False,all_training_completed_before_audit=True))
    progress(status='complete',stage='finished',comparison=str(RESULT/'comparison.json'))


if __name__=='__main__':
    main()
