import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

"""Check the selected interpreter and resources before any evaluation."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from project_config import CODE_ROOT as PROJECT_ROOT, DATA_ROOT as RESOURCE_ROOT, RESULT_ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--verify-hashes', action='store_true')
parser.add_argument('--environment-only', action='store_true', help='Check installed dependencies and device, without datasets or weights')
parser.add_argument('--data-profile', choices=('fixed10', 'full'), default='fixed10')
parser.add_argument('--task', choices=('cnn', 'sal'), default='cnn', help='CNN does not require SAL pretrained weights')
args = parser.parse_args()
errors = []
print('Python:', sys.executable, flush=True)
print('Resource root:', RESOURCE_ROOT, flush=True)
print('Output root:', RESULT_ROOT, flush=True)
if sys.version_info[:2] != (3, 10):
    errors.append('Use Python 3.10. Run Setup.cmd after installing Python 3.10.')
for name, expected in {'torch':'2.2.1+cu121','torchaudio':'2.2.1+cu121', 'lightning':'2.0.8',
        's3prl':'0.4.18', 'numpy':'1.26.4','torchmetrics':'1.3.2','transformers':'4.51.3'}.items():
    try:
        actual = metadata.version(name)
        print(f'{name}: {actual}')
        if actual != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    except metadata.PackageNotFoundError:
        errors.append(f'Missing package: {name}')
required = []
if not args.environment_only and args.task == 'sal':
    required += ['models/SAL_WavLM.ckpt', 'models/wavlm_large.pt']
if not args.environment_only and args.data_profile == 'full':
    required += [
    'PartialSpoof/database/dev/con_wav', 'PartialSpoof/database/eval/con_wav',
    'PartialSpoof/database/segment_labels/dev_seglab_0.16.npy',
    'PartialSpoof/database/segment_labels/eval_seglab_0.16.npy',
    'PartialSpoof/database_dev.tar.gz.extracted.json',
    'PartialSpoof/database_eval.tar.gz.extracted.json',
    'LlamaPartialSpoof/R01TTS.0.a',
    'LlamaPartialSpoof/R01TTS.0.a.tgz.extracted.json',
    'downloads/LlamaPartialSpoof/label_R01TTS.0.a.txt']
if not args.environment_only and args.data_profile == 'fixed10':
    try:
        from tools.data_tools import validate_fixed10
        for split in ('train', 'dev', 'eval', 'lps'):
            manifest = validate_fixed10(split)
            print(f'FIXED10 {split}: {manifest["count"]} recordings; audio/labels/manifest matched')
    except Exception as exc:
        errors.append('Fixed10 resource check failed: '+str(exc))
for relative in required:
    if not (RESOURCE_ROOT/relative).exists():
        errors.append('Missing resource: '+str(RESOURCE_ROOT/relative))
if args.verify_hashes:
    sources = json.loads((PROJECT_ROOT/'manifests/resource_sources.json').read_text())
    for name in ('SAL_WavLM.ckpt','wavlm_large.pt'):
        path = RESOURCE_ROOT/'models'/name
        if not path.exists():
            continue
        print('Checking SHA256:', name, flush=True)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
        expected = next(x['sha256'] for x in sources if x['name']==name)
        if h.hexdigest() != expected:
            errors.append('SHA256 mismatch: '+name)
try:
    import torch
    model_config = json.loads((PROJECT_ROOT/args.task/'config.json').read_text(encoding='utf-8'))
    if not torch.cuda.is_available() and model_config.get('device', 'cuda') == 'cuda':
        errors.append('CUDA GPU unavailable. This package targets Windows + NVIDIA GPU.')
    elif torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    else:
        print('CPU mode selected in model config.')
except Exception as exc:
    errors.append('PyTorch import failed: '+repr(exc))
if errors:
    print('\nCHECK FAILED:\n'+'\n'.join('- '+x for x in errors))
    print('See code/README.txt. Put datasets and weights inside the adjacent data folder.')
    sys.exit(1)
print('\nCHECK PASSED.' + (' Environment only; datasets and weights were not checked.' if args.environment_only
                            else f' Task={args.task}; data_profile={args.data_profile}.'), flush=True)
