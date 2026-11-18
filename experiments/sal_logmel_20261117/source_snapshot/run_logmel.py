"""Train SAL with the existing LogMel frontend, then evaluate frozen selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import traceback

import sal
import torch
import hydra
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from logmel_model import LogMelSAL

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'result' / 'logmel_sal_20261117'
PROTOCOL = 'sal_logmel_reduced_v1'


def identity():
    return dict(protocol=PROTOCOL, **sal.training_identity())


def validate(metadata):
    expected = identity()
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong LogMel SAL checkpoint identity')
    return expected


def load_model(resolution, checkpoint_path=None, public_reference=False, device=None):
    if public_reference:
        raise ValueError('LogMel requires its own newly trained weights')
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', mmap=True)
    validate(ckpt['project_training'])
    model = LogMelSAL(resolution)
    loaded = model.load_state_dict({k[4:]: v for k, v in ckpt['state_dict'].items()}, strict=True)
    metadata = dict(epoch=ckpt['epoch'], global_step=ckpt['global_step'],
                    strict_load=str(loaded), project_training=ckpt['project_training'])
    return model.float().eval().to(device or 'cuda'), metadata


def build():
    settings, model_cfg, data_cfg = sal.training_config('reduced')
    model_cfg.net = OmegaConf.create({'_target_': 'logmel_model.LogMelSAL'})
    model = sal.configure_runtime_model(hydra.utils.instantiate(model_cfg))
    return settings, model_cfg, data_cfg, model


def smoke():
    L.seed_everything(888, workers=True)
    torch.set_num_threads(4)
    settings, mc, dc, model = build()
    from cnn import LogMel
    x = torch.randn(2, 64000)
    original = LogMel()
    actual = model.net.ssl_encoder.frontend
    torch.testing.assert_close(actual(x[0]), original(x[0]), rtol=0, atol=0)
    assert not hasattr(model.net, 'weight_layer')
    assert not any('wavlm' in k.lower() for k in model.state_dict())
    for n in (1, 319, 320, 321, 2559, 2560, 64000):
        assert actual(torch.zeros(n)).shape == (1, (n+319)//320, 80)
    model.cuda().train()
    y1, y2 = model(x.cuda())
    assert y1.shape == (2,25,8) and y2.shape == (2,25,2)
    (y1.square().mean()+y2.square().mean()).backward()
    assert torch.isfinite(model.net.ssl_encoder.projection.weight.grad).all()
    data = sal.make_data_module(dc)
    data.setup('fit')
    batch = next(iter(data.train_dataloader()))
    batch = tuple(t.cuda() if isinstance(t,torch.Tensor) else t for t in batch)
    model.zero_grad()
    from unittest.mock import patch
    optimizer = torch.optim.Adam(model.net.parameters(), lr=settings['learning_rate'])
    with patch.object(model, 'log'), patch.object(model, 'optimizers', return_value=optimizer):
        loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    result = dict(status='passed', frontend_exact=True, frame_alignment=True,
                  train_count=len(data.train_dataset), dev_count=len(data.valid_dataset),
                  parameters=sum(p.numel() for p in model.net.parameters()),
                  real_batch_loss=float(loss.detach()), no_wavlm=True)
    sal.write_json(ROOT.parent/'result'/'smoke.json', result)
    print(json.dumps(result), flush=True)


def train(resume=False):
    OUT.mkdir(parents=True, exist_ok=resume)
    resume_path = OUT/'checkpoints/last.ckpt' if resume else None
    if resume:
        saved = torch.load(str(resume_path), map_location='cpu', mmap=True)
        validate(saved['project_training'])
        if not saved.get('optimizer_states') or not saved.get('lr_schedulers'):
            raise ValueError('Resume requires complete optimizer and scheduler states')
        sal.write_json(OUT/f'resume_{sal.local_time():%Y%m%d_%H%M%S}.json',
            dict(checkpoint=str(resume_path),sha256=sal.file_digest(resume_path),
                 saved_epoch=saved['epoch'],global_step=saved['global_step'],
                 note='Restore Lightning model/optimizer/scheduler/callback/loop states. Original checkpoint did not save RNG states; reseed 888, not bitwise equivalent to uninterrupted training.'))
        del saved
    state = dict(status='initializing', **identity(), started=sal.local_time().isoformat())
    def update(**kw):
        state.update(**kw, updated=sal.local_time().isoformat())
        sal.write_json(OUT/'status.json',state)
    try:
        L.seed_everything(888, workers=True)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        settings, mc, dc, model = build()
        data = sal.make_data_module(dc)
        # The historical run used these exact limits/early-stop rules (40--60).
        sal.write_json(OUT/'resolved_config.json',dict(settings=settings,
            model=OmegaConf.to_container(mc,resolve=True), data=OmegaConf.to_container(dc,resolve=True),
            frontend='original cnn.LogMel + LayerNorm(80) + Linear(80,1024)',
            **identity(), test_used_for_selection=False,
            historical_reference='sal_baseline_20261112 / sal_frozen_test_20261113'))
        files = list(ROOT.glob('*.py')) + list((ROOT/'vendor/SAL/src').rglob('*.py'))
        sal.write_json(OUT/('resume_source_hashes.json' if resume else 'source_hashes.json'),{str(p.relative_to(ROOT)):sal.file_digest(p) for p in files})
        class Progress(Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                update(status='running', phase='train', epoch=trainer.current_epoch+1)
                print('TRAIN_EPOCH',trainer.current_epoch+1,flush=True)
            def on_validation_epoch_start(self,trainer,pl_module):
                update(status='running',phase='validation',epoch=trainer.current_epoch+1)
            def on_train_epoch_end(self,trainer,pl_module):
                metrics={k:float(v) for k,v in trainer.callback_metrics.items()}
                update(metrics=metrics)
                print('EPOCH_METRICS',trainer.current_epoch+1,json.dumps(metrics),flush=True)
            def on_save_checkpoint(self,trainer,pl_module,checkpoint):
                checkpoint['project_training']=dict(**identity(),settings=settings)
            def on_exception(self,trainer,pl_module,exception):
                trainer.save_checkpoint(str(OUT/'interrupted.ckpt'))
        checkpoint=ModelCheckpoint(dirpath=str(OUT/'checkpoints'),filename='best-{epoch:03d}',
            monitor='val/eer',mode='min',save_top_k=1,save_last=True)
        trainer=L.Trainer(accelerator='gpu',devices=1,precision='32-true',deterministic=True,
            min_epochs=settings['min_epochs'],max_epochs=settings['max_epochs'],
            default_root_dir=str(OUT),logger=CSVLogger(str(OUT),name='logs'),
            enable_progress_bar=False,
            callbacks=[checkpoint,EarlyStopping(monitor='val/eer',mode='min',patience=settings['patience']),Progress()])
        trainer.fit(model,datamodule=data,ckpt_path=str(resume_path) if resume else None)
        if trainer.interrupted or not checkpoint.best_model_path:
            raise RuntimeError('Training interrupted or missing dev-selected checkpoint')
        selected=Path(checkpoint.best_model_path)
        frozen=dict(checkpoint=str(selected),sha256=sal.file_digest(selected),
                    best_dev_eer_percent=float(checkpoint.best_model_score),
                    epochs_run=trainer.current_epoch,**identity())
        sal.write_json(OUT/'frozen_selection.json',frozen)
        update(status='trained',**frozen)
    except BaseException as exc:
        update(status='failed',error=repr(exc))
        raise


def resume():
    train(resume=True)


def evaluate():
    frozen=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf8'))
    if sal.file_digest(frozen['checkpoint']) != frozen['sha256']:
        raise ValueError('Frozen weights changed')
    sal.load_model=load_model
    sal.validate_training_metadata=validate
    for split in ('eval','lps'):
        run_name='logmel_sal_20261117_'+split
        if (sal.RESULT_ROOT/run_name/'metrics.json').exists():
            continue
        code=sal.run(['--split',split,'--checkpoint',frozen['checkpoint'],'--run-name',run_name])
        if code:
            raise RuntimeError(f'{split} evaluation failed: {code}')
    sal.write_json(OUT/'completion.json',dict(status='complete',**frozen))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train','resume','evaluate'])
    args=parser.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        globals()[args.mode]()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
