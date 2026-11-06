"""Search CNN hyperparameters on fixed10 train/dev; publish the dev winner for baseline.py.

Default plan only. Use --mode search to train the grid. Test data is never used
to select parameters, and no test evaluation is automatically launched.
"""
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from cnn import baseline as p


def candidate_configs(base, search):
    if base['training_data_profile'] != 'fixed10':
        raise ValueError('This course-project grid is restricted to fixed10 training/validation')
    configs = []
    for lr, dropout in itertools.product(search['learning_rates'], search['dropouts']):
        c = copy.deepcopy(base)
        c.update(learning_rate=float(lr), checkpoint=None, use_grid_search_best=False,
                 hyperparameter_source='current grid trial')
        c.pop('grid_selected_checkpoint', None)
        c['model']['dropout'] = float(dropout)
        if not 0 <= c['model']['dropout'] < 1:
            raise ValueError('Dropout must be in [0, 1)')
        configs.append(p.validate_config(c))
    if not configs or len({json.dumps(c, sort_keys=True) for c in configs}) != len(configs):
        raise ValueError('Grid must contain unique, nonempty configurations')
    return configs


def select_winner(trials):
    if not trials or any(t.get('status') != 'complete' for t in trials):
        raise ValueError('Only complete searches can publish best parameters')
    return min(trials, key=lambda t: (t['dev_eer_percent'], t['index']))


def search_grid(base, configs):
    p.require_device_idle(base['device'])
    p.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = p.datetime.now(p.ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    path = p.RESULT_ROOT/f'cnn_grid_search_{stamp}.json'
    state = dict(status='running', started_local=p.now(), base_config=base,
                 base_signature=p.configuration_signature(base, exclude_search=True),
                 selection_metric='fixed10 PS dev EER; lower is better; ties prefer earlier grid entry',
                 test_data_used=False, trials=[])

    def update(**fields):
        state.update(updated_local=p.now(), **fields)
        p.write_json(path, state)

    try:
        for index, config in enumerate(configs, 1):
            update(current_trial=index, total_trials=len(configs))
            training_path = p.train(config=config, progress=lambda file, info: update(current_training=file.name, current_progress=info))
            row = json.loads(training_path.read_text(encoding='utf-8'))
            if row.get('status') != 'complete' or row.get('data_profile') != 'fixed10':
                raise ValueError('Trial did not complete on fixed10 data')
            if p.sha256((CODE_ROOT/row['checkpoint']).resolve()) != row['checkpoint_sha256']:
                raise ValueError('Trial checkpoint changed')
            trial = dict(index=index, status='complete', config=config, training_file=training_path.name,
                         dev_eer_percent=row['best_dev_eer_percent'], best_epoch=row['best_epoch'],
                         checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'],
                         train_manifest_sha256=row['train_manifest_sha256'], dev_manifest_sha256=row['dev_manifest_sha256'])
            if state['trials'] and any(trial[k] != state['trials'][0][k] for k in ('train_manifest_sha256','dev_manifest_sha256')):
                raise ValueError('Data changed during the search')
            state['trials'].append(trial)
            update(current_progress=None)
            gc.collect()
            if p.torch.cuda.is_available():
                p.torch.cuda.empty_cache()
        winner = select_winner(state['trials'])
        update(status='complete', winner=winner, finished_local=p.now())
        # Published only after all trials succeed. A failed search retains the previous valid winner.
        p.write_json(p.RESULT_ROOT/'cnn_grid_best.json', dict(status='complete', search_file=path.name, winner=winner))
        print(json.dumps(dict(search_file=path.name, winner=winner), ensure_ascii=False, indent=2))
        return path
    except BaseException as error:
        update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), finished_local=p.now())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan','search'), default='plan')
    args = parser.parse_args(argv)
    base = p.load_config(use_grid=False)
    search = json.loads(Path(__file__).with_name('grid_config.json').read_text(encoding='utf-8'))
    configs = candidate_configs(base, search)
    if args.mode == 'plan':
        print(json.dumps(dict(mode='plan', trials=len(configs), learning_rates=search['learning_rates'],
                             dropouts=search['dropouts'], training_profile=base['training_data_profile']), indent=2))
        return 0
    search_grid(base, configs)
    return 0


if __name__ == '__main__':
    raise SystemExit(p.entrypoint(main))
