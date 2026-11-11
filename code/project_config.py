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
    return Path(os.path.relpath(value, code_root)).as_posix()

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
