"""Project-relative locations, independent of drive letter and working directory."""
from pathlib import Path, PureWindowsPath
import json

VISUALIZATION_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = VISUALIZATION_ROOT / 'outputs'
PROJECT_ROOT = VISUALIZATION_ROOT.parent
CODE_ROOT = PROJECT_ROOT / 'code'
config_file = CODE_ROOT / 'config.json'
config = json.loads(config_file.read_text(encoding='utf-8-sig')) if config_file.exists() else {}
result_setting = Path(config.get('result_root', '../result'))
if result_setting.is_absolute() or PureWindowsPath(str(result_setting)).is_absolute():
    raise ValueError('code/config.json result_root must be relative to code for portability')
RESULT_ROOT = (CODE_ROOT / result_setting).resolve()


def project_path(value):
    """CLI relative paths are anchored to the packaged project root."""
    value = Path(value)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def output_path(value):
    """Keep all active visualization exports inside visualization/."""
    candidate = project_path(value)
    if not candidate.is_relative_to(VISUALIZATION_ROOT.resolve()):
        raise ValueError('Visualization output must be inside visualization/')
    return candidate


def analysis_directory(value=None):
    if value is not None:
        candidate = project_path(value)
        if not candidate.is_dir() and (OUTPUT_ROOT/candidate.name).is_dir():
            candidate = OUTPUT_ROOT/candidate.name
    else:
        candidates = [p for p in OUTPUT_ROOT.glob('sal_diagnostics_*')
                      if (p / 'analysis_manifest.json').is_file()
                      and (p / 'PS_eval/summary.json').is_file()
                      and (p / 'LPS_crossfade/summary.json').is_file()]
        if not candidates:
            raise FileNotFoundError('No completed PS/LPS diagnostics. Run diagnose_saved_predictions.py first.')
        candidate = max(candidates, key=lambda p: (p / 'analysis_manifest.json').stat().st_mtime)
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def source_run(value):
    """Rebase legacy absolute paths recorded on the original machine to this package."""
    raw = str(value).replace('\\', '/')
    name = raw.rstrip('/').split('/')[-1]
    # Prefer the current package, even when the original computer path still exists.
    packaged = RESULT_ROOT / name
    if (packaged / 'predictions.sqlite').is_file():
        return packaged
    if not PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
        relative = (RESULT_ROOT / raw).resolve()
        if (relative / 'predictions.sqlite').is_file():
            return relative
    original = Path(value)
    if original.is_absolute() and (original / 'predictions.sqlite').is_file():
        return original
    raise FileNotFoundError(f'Missing saved predictions. Include result/{name}/predictions.sqlite in the package.')
