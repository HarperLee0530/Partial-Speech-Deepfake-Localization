"""Default entry routing must preserve the adopted encoder and frozen protocol."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cnn_ps import main_method, predict_main
from cnn_ps.wavlm_frontend import verify_frozen_plan


class WavLMMainTests(unittest.TestCase):
    def test_evaluation_loads_only_selected_arm_dependencies(self):
        training = main_method.train_wavlm
        for arms in (['mel_control'], ['wavlm_frozen'], ['mel_control', 'wavlm_frozen']):
            with self.subTest(arms=arms), \
                 patch.object(training, 'freeze_eval_plan', return_value=dict.fromkeys(arms)) as freeze, \
                 patch('cnn_ps.dual_features.FeatureData') as mel, \
                 patch.object(training, 'WavLMData') as ssl, \
                 patch.object(training, 'evaluate_arms') as evaluate:
                argv = ['--evaluate', '--arms', *arms, '--eval-cache', 'eval',
                        '--output', 'run', '--device', 'cpu']
                if 'wavlm_frozen' in arms:
                    argv += ['--eval-ssl', 'ssl']
                training.main(argv)
                freeze.assert_called_once_with(Path('run'), arms)
                self.assertEqual(mel.call_count, int('mel_control' in arms))
                self.assertEqual(ssl.call_count, int('wavlm_frozen' in arms))
                self.assertEqual(set(evaluate.call_args.args[1]), set(arms))

    def test_missing_wavlm_eval_cache_fails_before_freezing_or_loading_data(self):
        training = main_method.train_wavlm
        with patch.object(training, 'freeze_eval_plan') as freeze, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                training.main(['--evaluate', '--arms', 'wavlm_frozen', '--eval-cache', 'eval',
                               '--output', 'run', '--device', 'cpu'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('--eval-ssl', errors.getvalue())
        freeze.assert_not_called()

    def test_eval_cache_revalidates_single_or_paired_plan_without_expanding_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / 'frozen_eval_plan.json'
            for arms in (['wavlm_frozen'], ['mel_control', 'wavlm_frozen']):
                plan.write_text(json.dumps({'schema': main_method.train_wavlm.SCHEMA,
                                            'arms': {arm: {} for arm in arms}}), encoding='utf8')
                with patch.object(main_method.train_wavlm, 'freeze_eval_plan') as verify:
                    verify_frozen_plan(plan)
                    verify.assert_called_once_with(plan.parent, arms)

    def test_prediction_defaults_to_wavlm_and_requires_its_arm(self):
        with patch.object(predict_main, 'predict_backend', return_value={'ok': True}) as call:
            result = predict_main.predict('audio.wav', 'prediction.json', 'encoder')
        self.assertEqual(result, {'ok': True})
        args, kwargs = call.call_args
        self.assertEqual(args[2].name, 'cnn_wavlm_frozen_seed666.pt')
        self.assertEqual(kwargs['model_folder'], 'encoder')
        self.assertEqual(kwargs['expected_arm'], 'wavlm_frozen')

    def test_train_freeze_and_evaluate_forward_only_main_arm(self):
        cases = [
            (['train', '--train-cache', 'tr', '--dev-cache', 'dv',
              '--train-ssl', 'ssltr', '--dev-ssl', 'ssldv'], ['--arm', 'wavlm_frozen']),
            (['freeze'], ['--arms', 'wavlm_frozen', '--freeze-only']),
            (['evaluate', '--eval-cache', 'ev', '--eval-ssl', 'sslev'],
             ['--arms', 'wavlm_frozen', '--evaluate']),
        ]
        for argv, prefix in cases:
            with self.subTest(command=argv[0]), patch.object(main_method.train_wavlm, 'main') as call:
                main_method.main(argv + ['--output', 'newrun'])
                forwarded = call.call_args.args[0]
                self.assertEqual(forwarded[:len(prefix)], prefix)
                if argv[0] == 'train':
                    self.assertIn('--train-ssl', forwarded)
                    self.assertNotIn('--eval-ssl', forwarded)
                if argv[0] == 'evaluate':
                    self.assertIn('--eval-ssl', forwarded)


if __name__ == '__main__':
    unittest.main()
