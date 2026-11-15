"""Small synthetic checks for fresh-initialization four-arm training.

The archive check performs one optimizer update on two synthetic recordings
inside a temporary directory; no PS data or existing experimental outputs are
read, trained on, calibrated, or changed by these tests.
"""
from argparse import Namespace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cnn_ps.backbone import CNNBiGRU
from cnn_ps.finetune import fine_collate, objective
from cnn_ps.io import read_json, sha256
from cnn_ps import train_four as training


class TinyFeatures:
    def __init__(self, split):
        self.split = split
        self.channels = 3 if split == "train" else 1
        self.assignment = np.array([0, 1])
        marker = {"train":"T", "dev":"D", "eval":"E"}[split]
        self.uids = [f"CON_{marker}_{i:07d}" for i in range(2)]
        self.fine = {self.uids[0]:np.array([0,1,1,0,0,0,0,0,1,1,1], dtype=np.float32),
                     self.uids[1]:np.array([1,1,0,0], dtype=np.float32)}
        self.requested = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.get_record(index, 0)

    def get_record(self, index, channel):
        if not 0 <= channel < self.channels:
            raise IndexError(channel)
        self.requested.append((index, channel))
        generator = torch.Generator().manual_seed(19 + index)
        n = (16,8)[index]
        fine = torch.zeros(n)
        reference = self.fine[self.uids[index]]
        fine[:len(reference)] = torch.from_numpy(reference)
        w = torch.zeros(n)
        w[:len(reference)] = .02
        if index == 0:
            w[len(reference)-1] = .005
        return dict(uid=self.uids[index], features=torch.randn(1,n,80,generator=generator)+channel*.1,
                    labels=torch.tensor([1.,0.]) if index==0 else torch.tensor([1.]),
                    duration=.205 if index==0 else .08, fine_labels=fine, fine_weights=w,
                    weights=torch.tensor([.16,.045]) if index==0 else torch.tensor([.08]),
                    length20=len(reference), channel=channel)


class FourTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_matches_cnn_structure_but_uses_new_random_weights(self):
        config = training.model_config()
        self.assertEqual(config.dropout,.4)
        self.assertEqual(config.channels,(16,32,64))
        self.assertEqual(config.hidden_size,128)
        self.assertEqual(config.gru_layers,2)
        self.assertEqual((config.sample_rate,config.hop_length,config.n_fft,config.win_length,config.n_mels),
                         (16000,320,512,400,80))
        self.assertEqual(sum(p.numel() for p in CNNBiGRU(config).parameters()),911521)

    def test_clean_view_never_rotates_augmented_view_is_paired_and_epoch_zero_clean(self):
        dataset=TinyFeatures('train')
        clean,augmented=training.TrainingView(dataset,True),training.TrainingView(dataset,False)
        for epoch in (0,1,2,3,20):
            clean.epoch=augmented.epoch=epoch
            for index in range(len(dataset)):
                self.assertEqual(clean[index]['channel'],0)
                expected=(dataset.assignment[index]+epoch-1)%3 if epoch else 0
                self.assertEqual(augmented[index]['channel'],expected)
        channels=[]
        for epoch in (1,2,3):
            augmented.epoch=epoch
            channels.append(augmented[0]['channel'])
        self.assertEqual(set(channels),{0,1,2})

    def test_all_arms_share_cnn_initialization_and_adapter_keeps_cpu_cuda_rng(self):
        reference=None
        batch=fine_collate([TinyFeatures('train')[i] for i in range(2)])
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for arm,options in training.ARMS.items():
                training.setup(666,2)
                model=CNNBiGRU(training.model_config()).to(device).eval()
                digest=training.state_digest(model)
                if reference is None:
                    reference=digest
                self.assertEqual(digest,reference)
                cpu_before=torch.get_rng_state().clone()
                cuda_before=torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
                adapter=training.make_adapter(options,666,device)
                self.assertTrue(torch.equal(cpu_before,torch.get_rng_state()),arm)
                if cuda_before is not None:
                    self.assertTrue(torch.equal(cuda_before,torch.cuda.get_rng_state()),arm)
                with torch.no_grad():
                    original=model(batch['features'].to(device),batch['lengths'])
                    final,_=training.forward_batch(model,adapter,batch,device)
                torch.testing.assert_close(final,original,rtol=0,atol=0)
                if adapter is not None:
                    self.assertEqual(adapter.normalize_mode,'layernorm')

    def test_all_arms_backpropagate_main_loss_and_declared_auxiliary_terms(self):
        batch=fine_collate([TinyFeatures('train')[i] for i in range(2)])
        for arm,options in training.ARMS.items():
            training.setup(666,2)
            model=CNNBiGRU(training.model_config()).train()
            adapter=training.make_adapter(options,666,'cpu')
            final,outputs=training.forward_batch(model,adapter,batch,'cpu')
            loss,parts=training.training_loss(final,outputs,batch,'cpu',arm=='boundary')
            self.assertTrue(torch.isfinite(loss))
            if adapter is None:
                torch.testing.assert_close(loss,objective(final,batch,'cpu'),rtol=0,atol=0)
                self.assertEqual(parts['coarse_aux'],0.)
            else:
                self.assertGreater(parts['coarse_aux'],0.)
            self.assertEqual(parts['boundary']>0,arm=='boundary')
            if arm!='boundary':
                self.assertEqual(parts['consistency'],0.)
            loss.backward()
            for parameter in (model.blocks[0][0].weight,model.gru.weight_ih_l0,
                              model.gru.weight_ih_l0_reverse,model.classifier.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(int(torch.count_nonzero(parameter.grad)),0)

    def test_selection_uses_event_f1_and_keeps_earliest_tie(self):
        best=dict(epoch=0,event_f1_iou05_percent=40.)
        self.assertTrue(training.is_better(0.,None))
        self.assertFalse(training.is_better(40.,best))
        self.assertFalse(training.is_better(40.+5e-11,best))
        self.assertTrue(training.is_better(40.+2e-10,best))

    def test_every_epoch_archive_freezing_and_fixed_threshold_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=Namespace(arm='boundary',output=Path(temporary),seed=666,epochs=1,batch_size=2,
                           threads=2,device='cpu',train_cache='synthetic-train',dev_cache='synthetic-dev')
            with contextlib.redirect_stdout(io.StringIO()):
                complete=training.train_arm(args,TinyFeatures('train'),TinyFeatures('dev'))
            folder=Path(temporary)/'boundary'
            self.assertEqual(complete['updates'],1)
            self.assertEqual(complete['archived_epochs'],2)
            for epoch in (0,1):
                prefix=folder/'epochs'/f'epoch_{epoch:02d}'
                self.assertTrue(prefix.with_suffix('.pt').is_file())
                self.assertTrue(prefix.with_name(prefix.name+'_scores20.npz').is_file())
                metric=read_json(prefix.with_name(prefix.name+'_metrics.json'))
                self.assertLessEqual(metric['real_fpr_percent'],5+1e-10)
                self.assertNotIn('selection_score_j_percent',metric)
            history=read_json(folder/'history.json')
            expected=0
            if history[1]['event_f1_iou05_percent']>history[0]['event_f1_iou05_percent']+1e-10:
                expected=1
            self.assertEqual(complete['epoch'],expected)
            self.assertEqual(sha256(folder/'best.pt'),sha256(folder/'epochs'/f'epoch_{expected:02d}.pt'))
            with self.assertRaises(FileExistsError):
                training.train_arm(args,TinyFeatures('train'),TinyFeatures('dev'))
            frozen=training.freeze_eval_plan(Path(temporary),['boundary'])
            freeze_hash=sha256(Path(temporary)/'frozen_eval_plan.json')
            training.freeze_eval_plan(Path(temporary),['boundary'])
            self.assertEqual(freeze_hash,sha256(Path(temporary)/'frozen_eval_plan.json'))
            # Any dev recalibration attempt during eval must fail this test.
            with patch.object(training,'evaluate_dev',side_effect=AssertionError('test calibration forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    reports=training.evaluate_arms(args,TinyFeatures('eval'),frozen)
            self.assertEqual(reports['boundary']['threshold'],complete['threshold'])
            self.assertEqual(reports['boundary']['metric_protocol_version'],'ps_native20_event_v2')
            with self.assertRaises(FileExistsError):
                training.evaluate_arms(args,TinyFeatures('eval'),frozen)
            other=Namespace(**{**vars(args),'arm':'cnn_clean','epochs':0})
            with contextlib.redirect_stdout(io.StringIO()):
                training.train_arm(other,TinyFeatures('train'),TinyFeatures('dev'))
            with self.assertRaisesRegex(ValueError,'paired training'):
                training.freeze_eval_plan(Path(temporary),['boundary','cnn_clean'])

    def test_training_cli_rejects_eval_data_or_old_weight_and_normalization_options(self):
        examples=[['--arm','cnn_clean','--train-cache','train','--dev-cache','dev','--eval-cache','test'],
                  ['--arm','cnn_aug','--train-cache','train','--dev-cache','dev','--checkpoint','old.pt'],
                  ['--arm','dual','--train-cache','train','--dev-cache','dev','--normalization','old.npz'],
                  ['--freeze-only','--eval-cache','test'],['--evaluate','--train-cache','train','--eval-cache','test']]
        for arguments in examples:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exception:
                    training.main(['--output','unused','--device','cpu']+arguments)
            self.assertEqual(exception.exception.code,2)


if __name__=='__main__':
    unittest.main()
