"""Offline prediction with the experimental frozen WavLM or paired Mel model."""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import torch
from .backbone import LogMel, ModelConfig
from .io import sha256,save_json
from .predict_main import waveform_batch,decode_events
from .train_four import setup,make_adapter
from .train_dual import forward_batch
from .train_wavlm import SCHEMA,ARMS
from .wavlm_backend import ProjectedCNN
from .wavlm_frontend import FrozenWavLM

@torch.inference_mode()
def predict(audio,output,checkpoint,model_folder=None,device='cpu',threads=2):
    if Path(output).exists():
        raise FileExistsError(output)
    setup(666,threads)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    protocol=payload['protocol'];arm=protocol['arm']
    if payload['schema']!=SCHEMA or arm not in ARMS or payload['adapter_options']!=ARMS[arm]:
        raise ValueError('Wrong experiment checkpoint')
    threshold=float(payload['threshold']);dev=payload['dev_metrics']
    if not np.isfinite(threshold) or threshold!=dev['threshold'] or not 0<=dev['real_fpr_percent']<=5+1e-10:
        raise ValueError('Invalid dev threshold')
    config=ModelConfig(**payload['model_config'])
    model=ProjectedCNN(config,arm,protocol['seed']).to(device)
    model.load_state_dict(payload['backbone_state_dict'],strict=True)
    model.eval().requires_grad_(False)
    adapter=make_adapter(payload['adapter_options'],protocol['seed'],device)
    adapter.load_state_dict(payload['adapter_state_dict'],strict=True)
    adapter.eval().requires_grad_(False)
    if any(not torch.isfinite(t).all() for m in (model,adapter) for t in m.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    if arm=='wavlm_frozen':
        if model_folder is None or sha256(Path(model_folder)/'pytorch_model.bin')!=protocol['external_pretraining']['model_sha256']:
            raise ValueError('Require the pinned original WavLM checkpoint')
        ssl=FrozenWavLM(model_folder,device)
        frontend=lambda padded:ssl(padded,len(padded)//320)[None].cpu()
    else:
        frontend=LogMel(config).eval()
    wave,sr=sf.read(audio,dtype='float32')
    batch,duration,actual20=waveform_batch(wave,sr,frontend)
    scores,_=forward_batch(model,adapter,batch,device)
    scores=scores[0,:actual20].cpu().numpy().copy()
    result=dict(arm=arm,checkpoint_sha256=sha256(checkpoint),selected_epoch=payload['epoch'],
        threshold=threshold,threshold_source='Frozen dev calibration',duration_seconds=duration,
        scores20=scores.tolist(),predicted_fake_intervals_seconds=decode_events(scores,threshold,duration).tolist(),
        postprocessing='Consecutive fake frames only',resolution_seconds=.02,
        precision_note='CPU/GPU numerical differences may occur; archived scores define the experiment')
    save_json(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('audio','output','checkpoint'):p.add_argument('--'+arg,required=True)
    p.add_argument('--model-folder')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();result=predict(**vars(args))
    print({'arm':result['arm'],'events':len(result['predicted_fake_intervals_seconds'])})
