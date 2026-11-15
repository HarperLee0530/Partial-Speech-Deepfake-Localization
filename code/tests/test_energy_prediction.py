"""Energy checkpoint inference: actual conditioning, embedded stats and cutoff."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from cnn_ps.backbone import CNNBiGRU, LogMel
from cnn_ps.energy_cache import NORMALIZATION_SCHEMA
from cnn_ps.energy_features import FEATURE_NAMES
from cnn_ps.predict_energy import energy_waveform_batch, load_energy_checkpoint, predict
from cnn_ps.train_energy import ARMS, BASE_OPTIONS, SCHEMA, make_energy_adapter
from cnn_ps.train_four import model_config


class EnergyPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = model_config()
        cls.frontend = LogMel(cls.config).eval()
        cls.normalization = dict(schema=NORMALIZATION_SCHEMA, mean=[-1.,0.,0.,0.,0.], std=[1.]*5,
                                 fit_split='train', fit_channel=0, feature_names=list(FEATURE_NAMES),
                                 valid_train_frames=100, padded_frames_in_statistics=False)
        cls.payloads = {}
        for arm, enabled in ARMS.items():
            model = CNNBiGRU(cls.config)
            adapter = make_energy_adapter(enabled, cls.normalization, 666, 'cpu')
            with torch.no_grad():
                for parameter in list(model.parameters()) + list(adapter.parameters()):
                    parameter.zero_()
                adapter.fine_output.weight[0,0] = 1.
                if enabled:
                    adapter.energy_projection.weight[0,0] = 2.
            protocol = dict(schema=SCHEMA, arm=arm, energy_enabled=enabled, seed=666,
                            adapter_options=copy.deepcopy(BASE_OPTIONS), model_config=cls.config.to_dict(),
                            energy_normalization=copy.deepcopy(cls.normalization),
                            normalization_file='nonexistent-external-stats-must-not-be-read.json')
            cls.payloads[arm] = dict(schema=SCHEMA, protocol=protocol, energy_enabled=enabled,
                model_config=cls.config.to_dict(), adapter_options=copy.deepcopy(BASE_OPTIONS),
                backbone_state_dict=model.state_dict(), adapter_state_dict=adapter.state_dict(),
                energy_normalization=copy.deepcopy(cls.normalization), epoch=4,
                threshold=.5, dev_metrics=dict(threshold=.5, real_fpr_percent=4.9))

    def test_nonzero_energy_projection_and_embedded_normalization_change_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root/'silence.wav'
            sf.write(audio, np.zeros(3252,np.float32), 16000, subtype='FLOAT')
            outputs = {}
            for arm, payload in self.payloads.items():
                checkpoint = root/(arm+'.pt')
                torch.save(payload, checkpoint)
                result = predict(audio, root/(arm+'.json'), checkpoint, threads=1)
                self.assertEqual(result['arm'], arm)
                self.assertEqual(result['energy_enabled'], ARMS[arm])
                self.assertEqual(result['native20_actual_frames'], 11)
                self.assertEqual(result['stored20_frames'], 16)
                self.assertEqual(result['threshold'], .5)
                outputs[arm] = result
            np.testing.assert_array_equal(outputs['boundary_control']['scores20'], np.zeros(11))
            expected = float(.5*F.gelu(torch.tensor(2.)))
            # Scalar and vector GELU kernels may differ by one float32 ULP.
            np.testing.assert_allclose(outputs['boundary_energy5']['scores20'], np.full(11,expected), rtol=0, atol=1e-7)
            self.assertEqual(outputs['boundary_control']['predicted_fake_intervals_seconds'], [])
            self.assertEqual(outputs['boundary_energy5']['predicted_fake_intervals_seconds'], [[0.,3252/16000]])

    def test_float64_checkpoint_threshold_is_preserved_and_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint,audio,output = root/'model.pt',root/'audio.wav',root/'result.json'
            payload = copy.deepcopy(self.payloads['boundary_energy5'])
            payload['adapter_state_dict']['fine_output.weight'].zero_()
            payload['adapter_state_dict']['fine_output.bias'].fill_(2.)
            expected = 1.  # exact fine residual2 at a gate of0.5
            cut = float(np.nextafter(np.float64(expected),np.float64(np.inf)))
            payload['threshold'] = payload['dev_metrics']['threshold'] = cut
            torch.save(payload,checkpoint)
            sf.write(audio,np.zeros(321,np.float32),16000,subtype='FLOAT')
            result = predict(audio,output,checkpoint,threads=1)
            self.assertEqual(result['threshold'],cut)
            self.assertEqual(result['predicted_fake_intervals_seconds'],[])
            with self.assertRaises(FileExistsError):
                predict(audio,output,checkpoint,threads=1)

    def test_raw_energy_tail_is_padded_only_after_extraction(self):
        for samples in (1,319,320,321,2559,2560,2561):
            wave = np.linspace(.02,.2,samples,dtype=np.float32)
            batch,duration,actual = energy_waveform_batch(wave,16000,self.frontend)
            stored = int(batch['lengths'][0])
            self.assertEqual(batch['energy_features'].shape,(1,stored,5))
            self.assertEqual(actual,(samples+319)//320)
            self.assertEqual(torch.count_nonzero(batch['energy_features'][0,actual:]).item(),0)
            self.assertEqual(int((batch['fine_weights']>0).sum()),actual)
            self.assertAlmostEqual(float(batch['fine_weights'].sum()),duration,places=7)
            self.assertAlmostEqual(float(batch['weights'].sum()),duration,places=7)
        batch,_,_ = energy_waveform_batch(np.ones(1,np.float32),16000,self.frontend)
        self.assertTrue(batch['fine_weights'][0,0]>0)
        self.assertEqual(torch.count_nonzero(batch['energy_features']).item(),0)

    def test_missing_statistics_adapter_and_inconsistent_arm_or_buffers_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'model.pt'
            for key in ('energy_normalization','adapter_state_dict'):
                payload=copy.deepcopy(self.payloads['boundary_energy5']);del payload[key]
                torch.save(payload,file)
                with self.assertRaises(ValueError): load_energy_checkpoint(file)
            payload=copy.deepcopy(self.payloads['boundary_energy5'])
            del payload['adapter_state_dict']['energy_projection.weight']
            torch.save(payload,file)
            with self.assertRaises(RuntimeError): load_energy_checkpoint(file)
            payload=copy.deepcopy(self.payloads['boundary_energy5']);payload['energy_enabled']=False
            torch.save(payload,file)
            with self.assertRaises(ValueError): load_energy_checkpoint(file)
            payload=copy.deepcopy(self.payloads['boundary_energy5'])
            payload['adapter_state_dict']['energy_mean'][0]=50.
            torch.save(payload,file)
            with self.assertRaisesRegex(ValueError,'buffers disagree'): load_energy_checkpoint(file)
            payload=copy.deepcopy(self.payloads['boundary_energy5']);payload['schema']='old'
            torch.save(payload,file)
            with self.assertRaises(ValueError): load_energy_checkpoint(file)

    def test_invalid_audio_is_rejected(self):
        for wave,rate in ((np.zeros(0),16000),(np.zeros((2,2)),16000),
                          (np.array([np.nan]),16000),(np.ones(10),8000)):
            with self.assertRaises(ValueError): energy_waveform_batch(wave,rate,self.frontend)


if __name__ == '__main__':
    unittest.main()
