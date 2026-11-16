"""Standalone offline inference for paired fixed/learnable filterbank checkpoints.

Loads the entire CNN, boundary adapter, frequency knots and selected dev cutoff.
No labels, training cache or external statistics are required. Audio must be
16 kHz mono; only consecutive native20 positive cells are merged into events.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .backbone import CNNBiGRU, ModelConfig
from .io import save_json, sha256
from .learnable_filterbank import LearnableFilterbank, PowerSpectrogram
from .predict_main import waveform_batch, decode_events
from .train_filterbank import ARMS, BASE_OPTIONS, SCHEMA, forward_batch, json_native
from .train_four import make_adapter, model_config


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2]/'weights'/'boundary_learnable_mel_seed666.pt'


def load_filterbank_checkpoint(checkpoint, device='cpu'):
    """Validate the declared architecture and load its complete trained state."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('Filterbank checkpoint is unavailable; pass --checkpoint for an existing dev-selected .pt file')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = ('schema','protocol','model_config','adapter_options','frontend_config',
                'frontend_learnable','backbone_state_dict','adapter_state_dict',
                'frontend_state_dict','threshold','dev_metrics','epoch')
    if not isinstance(payload,dict) or any(key not in payload for key in required):
        raise ValueError('Expected a complete ps_boundary_learnable_filterbank_v1 checkpoint')
    protocol,dev = payload['protocol'],payload['dev_metrics']
    if not isinstance(protocol,dict) or not isinstance(dev,dict):
        raise ValueError('Missing checkpoint protocol or selected dev metrics')
    arm = protocol.get('arm')
    if (payload['schema'] != SCHEMA or protocol.get('schema') != SCHEMA or arm not in ARMS
            or type(payload['frontend_learnable']) is not bool
            or type(protocol.get('frontend_learnable')) is not bool
            or payload['frontend_learnable'] is not ARMS[arm]
            or protocol['frontend_learnable'] is not ARMS[arm]
            or payload['adapter_options'] != BASE_OPTIONS or protocol.get('adapter_options') != BASE_OPTIONS
            or type(protocol.get('seed')) is not int or protocol['seed'] < 0
            or type(payload['epoch']) is not int or payload['epoch'] < 0):
        raise ValueError('Checkpoint schema, arm, training flag or architecture is inconsistent')
    expected_config = model_config().to_dict()
    if (json_native(payload['model_config']) != json_native(expected_config)
            or json_native(protocol.get('model_config')) != json_native(expected_config)):
        raise ValueError('Checkpoint CNN/frontend configuration differs from this experiment')
    config = ModelConfig(**payload['model_config'])
    frontend = LearnableFilterbank(config,learnable=payload['frontend_learnable'])
    expected_frontend = json_native(frontend.config_dict())
    if (json_native(payload['frontend_config']) != expected_frontend
            or json_native(protocol.get('frontend_config')) != expected_frontend
            or json_native(protocol.get('shared_frontend_config')) != expected_frontend):
        raise ValueError('Frontend mathematical convention differs from this experiment')
    for key in ('backbone_state_dict','adapter_state_dict','frontend_state_dict'):
        state = payload[key]
        if (not isinstance(state,dict) or not state
                or any(not isinstance(value,torch.Tensor) or not torch.isfinite(value).all()
                       for value in state.values())):
            raise ValueError('Missing or nonfinite checkpoint model state')
    expected_buffers = {name:value.clone() for name,value in frontend.named_buffers()}
    frontend.load_state_dict(payload['frontend_state_dict'],strict=True)
    for name,value in frontend.named_buffers():
        source = payload['frontend_state_dict'][name]
        if (source.dtype != torch.float32 or not torch.equal(value,expected_buffers[name])
                or not torch.equal(source,expected_buffers[name])):
            raise ValueError('Checkpoint changed the fixed frontend frequency grid or displacement bounds')
    if payload['frontend_state_dict']['raw_shift'].dtype != torch.float32:
        raise ValueError('Frequency parameters must be float32')
    if not payload['frontend_learnable'] and torch.count_nonzero(frontend.raw_shift):
        raise ValueError('Fixed frontend must retain zero frequency shifts')
    points,matrix = frontend.frequencies_hz(),frontend.filterbank_matrix()
    if (not torch.isfinite(points).all() or not (points[1:]>points[:-1]).all()
            or not torch.isfinite(matrix).all() or (matrix<0).any() or (matrix.sum(0)<=0).any()
            or ((points[1:-1]-frontend.base_frequencies[1:-1]).abs()>frontend.max_displacement+1e-3).any()):
        raise ValueError('Invalid learned frequency geometry')
    threshold = float(payload['threshold'])
    if (not np.isfinite(threshold) or threshold != dev.get('threshold')
            or not np.isfinite(dev.get('real_fpr_percent',np.nan))
            or not 0 <= dev['real_fpr_percent'] <= 5+1e-10):
        raise ValueError('Missing or inconsistent dev-selected threshold/FPR')
    model = CNNBiGRU(config)
    model.load_state_dict(payload['backbone_state_dict'],strict=True)
    model.to(device).eval().requires_grad_(False)
    adapter = make_adapter(BASE_OPTIONS,protocol['seed'],device)
    adapter.load_state_dict(payload['adapter_state_dict'],strict=True)
    adapter.eval().requires_grad_(False)
    frontend.to(device).eval().requires_grad_(False)
    return model,adapter,frontend,PowerSpectrogram(config).eval(),payload,threshold


def power_waveform_batch(wave,sample_rate,power_frontend):
    """Reuse original labels-free storage padding and actual-duration masks."""
    batch,duration,actual20 = waveform_batch(wave,sample_rate,power_frontend)
    batch['power_features'] = batch.pop('features')
    return batch,duration,actual20


@torch.inference_mode()
def predict(audio,output,checkpoint=DEFAULT_CHECKPOINT,device='cpu',threads=4):
    if threads <= 0 or device not in ('cpu','cuda'):
        raise ValueError('threads must be positive and device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Prediction output already exists: {output}')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    wave,sample_rate = sf.read(audio,dtype='float32')
    model,adapter,frontend,power_frontend,payload,threshold = load_filterbank_checkpoint(checkpoint,device)
    batch,duration,actual20 = power_waveform_batch(wave,sample_rate,power_frontend)
    final,_ = forward_batch(model,adapter,frontend,batch,device)
    scores20 = final[0,:actual20].float().cpu().numpy().copy()
    detected = decode_events(scores20,threshold,duration)
    result = dict(audio=str(Path(audio).resolve()),duration_seconds=duration,sample_rate=sample_rate,
                  checkpoint_sha256=sha256(checkpoint),checkpoint_schema=SCHEMA,arm=payload['protocol']['arm'],
                  frontend_learnable=payload['frontend_learnable'],frontend_config=payload['frontend_config'],
                  frequency_knots_hz=frontend.frequencies_hz().detach().cpu().tolist(),
                  selected_epoch=payload['epoch'],threshold=threshold,
                  threshold_source='Checkpoint dev genuine-duration FPR<=5%, selected by dev event F1',
                  score_kind='native20 fake-positive raw logits, not calibrated probabilities',
                  threshold_decision_rule='float64(score) >= float64(threshold)',
                  interval_resolution_seconds=.02,native20_actual_frames=actual20,
                  stored20_frames=int(batch['lengths'][0]),scores20=scores20.tolist(),
                  predicted_fake_intervals_seconds=detected.tolist(),
                  postprocessing='Consecutive positive frames only; no smoothing, gap filling or short-event removal',
                  inference_mode='Offline complete CNN-BiGRU, boundary adapter and checkpoint filterbank',
                  precision_note='Regenerated scores may differ slightly across devices/library versions; archived scores define the original comparison.')
    save_json(output,result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args(argv)
    result = predict(**vars(args))
    print(dict(output=str(args.output),arm=result['arm'],actual20_frames=result['native20_actual_frames'],
               events=len(result['predicted_fake_intervals_seconds']),threshold=result['threshold']))


if __name__=='__main__':
    main()
