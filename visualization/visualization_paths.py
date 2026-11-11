"""Project-relative locations, independent of drive letter and working directory."""
from pathlib import Path, PureWindowsPath
import json
import os
import re

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


def safe_relative_path(value, root):
    """Retain an absolute provenance path when Windows volumes differ."""
    try:
        return Path(os.path.relpath(Path(value).resolve(), Path(root).resolve())).as_posix()
    except ValueError:
        return Path(value).resolve().as_posix()


def current_reduced_metrics(metrics, split):
    """Default plots accept only the current detector/dev identity and test IDs."""
    try:
        manifests = {role: json.loads((CODE_ROOT/'manifests/reduced'/f'{role}.json').read_text(encoding='utf-8'))
                     for role in (split, 'train_detector', 'dev')}
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (metrics.get('split') == split and metrics.get('data_profile') == 'reduced'
            and re.fullmatch('[0-9a-fA-F]{64}', str(metrics.get('checkpoint_sha256', ''))) is not None
            and metrics.get('smoke_test') is False
            and metrics.get('checkpoint_kind') == 'reduced_trained'
            and metrics.get('training_data_profile') == 'reduced'
            and metrics.get('training_split') == 'train_detector'
            and metrics.get('training_manifest_sha256') == manifests['train_detector']['manifest_sha256']
            and metrics.get('training_dev_manifest_sha256') == manifests['dev']['manifest_sha256']
            and metrics.get('manifest_sha256') == manifests[split]['manifest_sha256']
            and metrics.get('utterance_count') == manifests[split]['count'])


def latest_reduced_runs():
    selected = []
    for split in ('eval', 'lps'):
        eligible = []
        for candidate in RESULT_ROOT.glob('sal_wavlm_*'):
            try:
                metrics = json.loads((candidate/'metrics.json').read_text(encoding='utf-8-sig'))
                status = json.loads((candidate/'status.json').read_text(encoding='utf-8-sig'))
            except (OSError, json.JSONDecodeError):
                continue
            if (status.get('status') == 'complete' and current_reduced_metrics(metrics, split)
                    and (candidate/'predictions.sqlite').is_file()):
                eligible.append(candidate)
        if not eligible:
            raise FileNotFoundError(f'No completed {split} predictions from current reduced-trained SAL. '
                                    'Run the new model first, or use --runs explicitly for historical diagnostics.')
        selected.append(max(eligible, key=lambda p: p.name))
    hashes = {json.loads((p/'metrics.json').read_text(encoding='utf-8-sig'))['checkpoint_sha256'].lower()
              for p in selected}
    if len(hashes) != 1:
        raise ValueError('Latest PS/LPS predictions use different checkpoints. Use --runs explicitly to choose matching runs or labeled historical results.')
    return selected


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
        candidates = [p for p in candidates if all(current_reduced_metrics(
            json.loads((p/folder/'summary.json').read_text(encoding='utf-8')).get('source_metrics', {}), split)
            for split, folder in (('eval', 'PS_eval'), ('lps', 'LPS_crossfade')))]
        candidates = [p for p in candidates if len({json.loads((p/folder/'summary.json').read_text(encoding='utf-8'))
            ['source_metrics']['checkpoint_sha256'].lower() for folder in ('PS_eval', 'LPS_crossfade')}) == 1]
        if not candidates:
            raise FileNotFoundError('No current reduced PS/LPS diagnostics. Use diagnose_saved_predictions.py --latest, or specify --analysis-dir for history.')
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
