"""Evaluate full PS and LPS sequentially with one pinned, dev-selected CNN.

Default check only. --mode run performs inference; it never trains a model.
All CLI paths are relative to code/ unless absolute.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from cnn import baseline as p
from project_config import artifact_relative


def prepare(training_result):
    record = json.loads(training_result.read_text(encoding='utf-8'))
    if record.get('status') != 'complete' or record.get('data_profile') != 'full':
        raise ValueError('A completed full-train run is required')
    config = copy.deepcopy(record['config'])
    config.update(checkpoint=record['checkpoint'], use_grid_search_best=False, evaluation_data_profile='full')
    config = p.validate_config(config)
    weights = (CODE/config['checkpoint']).resolve()
    if p.sha256(weights) != record['checkpoint_sha256']:
        raise ValueError('Best checkpoint checksum differs from training record')
    model, checkpoint, _ = p.load_trained_model(config)
    if checkpoint['train_run'] != record['run'] or checkpoint['epoch'] != record['best_epoch']:
        raise ValueError('Checkpoint is not this training run\'s dev-selected best model')
    del model
    return record, config


def run_tests(training_result, record, config):
    p.require_device_idle(config['device'])
    stamp = p.datetime.now(p.ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    status_path = p.RESULT_ROOT/f'cnn_full_tests_{stamp}.json'
    state = dict(status='running', started_local=p.now(), training_result=artifact_relative(training_result),
                 checkpoint=record['checkpoint'], checkpoint_sha256=record['checkpoint_sha256'],
                 best_epoch=record['best_epoch'], config=config, results={})

    def update(**fields):
        state.update(updated_local=p.now(), **fields)
        p.write_json(status_path, state)

    update()
    print('TEST_SUITE_STATUS', status_path, flush=True)
    try:
        for split in ('eval', 'lps'):
            update(current_split=split, current_progress=None)
            if p.sha256((CODE/config['checkpoint']).resolve()) != record['checkpoint_sha256']:
                raise ValueError('Pinned checkpoint changed between tests')
            result_path = p.evaluate(split, config=config, progress=lambda path, info:
                                     update(current_result=artifact_relative(path), current_progress=info))
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if (result['status'] != 'complete' or result['data_profile'] != 'full'
                    or result['checkpoint_sha256'] != record['checkpoint_sha256']
                    or result['utterance_count'] != result['planned_utterances']):
                raise ValueError('Incomplete or incompatible test output')
            state['results'][split] = dict(result_file=artifact_relative(result_path),
                                          utterances=result['utterance_count'], eer_percent=result['eer_percent'],
                                          f1_bonafide_percent=result['f1_bonafide_percent'],
                                          accuracy_percent=result['accuracy_percent'])
            update(current_progress=None)
            gc.collect()
            if p.torch.cuda.is_available():
                p.torch.cuda.empty_cache()
        update(status='complete', finished_local=p.now())
    except BaseException as error:
        update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               error=repr(error), finished_local=p.now())
        raise
    return status_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', required=True)
    parser.add_argument('--mode', choices=('check', 'run'), default='check')
    args = parser.parse_args(argv)
    source = (CODE/args.training_result).resolve()
    record, config = prepare(source)
    print(json.dumps(dict(training_run=record['run'], best_epoch=record['best_epoch'],
                          checkpoint=record['checkpoint'], training_profile=config['training_data_profile'],
                          evaluation_profile=config['evaluation_data_profile'],
                          learning_rate=config['learning_rate'], dropout=config['model']['dropout'],
                          splits=['eval','lps']), ensure_ascii=False), flush=True)
    if args.mode == 'run':
        run_tests(source, record, config)
    return 0


if __name__ == '__main__':
    raise SystemExit(p.entrypoint(main))
