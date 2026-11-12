"""All runtime paths are relative to this code folder and the adjacent data/result."""
import hashlib
import json
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
CONFIG = json.loads((CODE_ROOT/'config.json').read_text(encoding='utf-8-sig'))

def configured_folder(key):
    path = Path(CONFIG[key])
    if path.is_absolute():
        raise ValueError(f'{key} must be relative to code, for example ../data')
    return (CODE_ROOT/path).resolve()

DATA_ROOT = configured_folder('data_root')
RESULT_ROOT = configured_folder('result_root')
LOG_ROOT = RESULT_ROOT/'logs'


def artifact_relative(path, code_root=CODE_ROOT, data_root=DATA_ROOT, result_root=RESULT_ROOT):
    """Serialize logical ../data and ../result paths even when they are junctions."""
    import os
    value = Path(path).resolve()
    for root, setting in ((data_root, 'data_root'), (result_root, 'result_root')):
        try:
            suffix = value.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return (Path(CONFIG[setting])/suffix).as_posix()
    try:
        return Path(os.path.relpath(value, code_root)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drives. External files
        # keep their absolute path; project data/results remain relocatable above.
        return value.as_posix()

def portable_checkpoint_path():
    # Save a relocatable path in new experiment metadata as well.
    return (Path(CONFIG['data_root'])/'models/SAL_WavLM.ckpt').as_posix()

def checkpoint_digest():
    path = DATA_ROOT/'models/SAL_WavLM.ckpt'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    value = digest.hexdigest()
    sources = json.loads((CODE_ROOT/'manifests/resource_sources.json').read_text(encoding='utf-8-sig'))
    expected = next(x['sha256'] for x in sources if x['name'] == path.name)
    if value != expected:
        raise ValueError(f'Checkpoint SHA256 differs from the released research artifact: {path}')
    return value


def check_environment(environment_only=False, task='sal', verify_hashes=False):
    """Check the portable reduced project without requiring old full datasets."""
    import importlib.metadata as metadata
    import sys
    errors = []
    print('Python:', sys.executable)
    if sys.version_info[:2] != (3, 10):
        errors.append('Use Python 3.10 and run Setup.cmd.')
    expected = {'torch': '2.2.1+cu121', 'torchaudio': '2.2.1+cu121',
                'lightning': '2.0.8', 's3prl': '0.4.18', 'numpy': '1.26.4',
                'torchmetrics': '1.3.2', 'transformers': '4.51.3'}
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
            print(f'{name}: {actual}')
            if actual != version:
                errors.append(f'{name}: expected {version}, found {actual}')
        except metadata.PackageNotFoundError:
            errors.append(f'Missing package: {name}')
    try:
        import torch
        if CONFIG[task].get('device', 'cuda') == 'cuda' and not torch.cuda.is_available():
            errors.append('CUDA unavailable; select device=cpu in config.json or install the GPU environment.')
        print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    except Exception as exc:
        errors.append(f'PyTorch import: {exc}')
    if not environment_only:
        try:
            from data import validate_reduced
            for role in ('train_detector', 'train_edit', 'train_review', 'dev', 'eval', 'lps'):
                manifest = validate_reduced(role)
                print(f'{role}: {manifest["count"]} recordings verified')
        except Exception as exc:
            errors.append(f'Reduced data: {exc}')
        if task == 'sal' and not (DATA_ROOT/'models/wavlm_large.pt').is_file():
            errors.append('Missing data/models/wavlm_large.pt (general WavLM pretraining).')
    if verify_hashes:
        sources = json.loads((CODE_ROOT/'manifests/resource_sources.json').read_text(encoding='utf-8-sig'))
        for name in ('wavlm_large.pt', 'SAL_WavLM.ckpt'):
            path = DATA_ROOT/'models'/name
            if not path.is_file():
                continue
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(block)
            expected_hash = next(x['sha256'] for x in sources if x['name'] == name)
            if h.hexdigest() != expected_hash:
                errors.append(f'SHA256 mismatch: {name}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    print('CHECK PASSED' + (' (environment only)' if environment_only else ''))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--environment-only', action='store_true')
    parser.add_argument('--verify-hashes', action='store_true')
    parser.add_argument('--task', choices=('sal', 'cnn'), default='sal')
    args = parser.parse_args()
    if args.check:
        check_environment(args.environment_only, args.task, args.verify_hashes)
    else:
        parser.print_help()
