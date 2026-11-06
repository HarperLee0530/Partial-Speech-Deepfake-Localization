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
required = ['models/SAL_WavLM.ckpt', 'models/wavlm_large.pt',
    'PartialSpoof/database/dev/con_wav', 'PartialSpoof/database/eval/con_wav',
    'PartialSpoof/database/segment_labels/dev_seglab_0.16.npy',
    'PartialSpoof/database/segment_labels/eval_seglab_0.16.npy',
    'PartialSpoof/database_dev.tar.gz.extracted.json',
    'PartialSpoof/database_eval.tar.gz.extracted.json',
    'LlamaPartialSpoof/R01TTS.0.a',
    'LlamaPartialSpoof/R01TTS.0.a.tgz.extracted.json',
    'downloads/LlamaPartialSpoof/label_R01TTS.0.a.txt']
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
    if not torch.cuda.is_available():
        errors.append('CUDA GPU unavailable. This package targets Windows + NVIDIA GPU.')
    else:
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
except Exception as exc:
    errors.append('PyTorch import failed: '+repr(exc))
if errors:
    print('\nCHECK FAILED:\n'+'\n'.join('- '+x for x in errors))
    print('See code/README.txt. Put datasets and weights inside the adjacent data folder.')
    sys.exit(1)
print('\nCHECK PASSED. Ready for the smoke test.', flush=True)
