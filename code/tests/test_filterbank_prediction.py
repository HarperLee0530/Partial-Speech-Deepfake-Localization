"""Standalone filterbank inference uses trained knots and the frozen dev cutoff."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from cnn_ps.backbone import CNNBiGRU
from cnn_ps.learnable_filterbank import LearnableFilterbank, PowerSpectrogram
from cnn_ps.predict_filterbank import load_filterbank_checkpoint, power_waveform_batch, predict
from cnn_ps.train_filterbank import ARMS, BASE_OPTIONS, SCHEMA, json_native
from cnn_ps.train_four import make_adapter, model_config


class FilterbankPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = model_config()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(892)
            model = CNNBiGRU(cls.config)
        adapter = make_adapter(BASE_OPTIONS,666,'cpu')
        cls.payloads = {}
        for arm,enabled in ARMS.items():
            frontend = LearnableFilterbank(cls.config,enabled)
            if enabled:
                with torch.no_grad(): frontend.raw_shift.copy_(torch.linspace(-1.2,1.2,80))
            protocol = dict(schema=SCHEMA,arm=arm,frontend_learnable=enabled,seed=666,
                            adapter_options=copy.deepcopy(BASE_OPTIONS),model_config=cls.config.to_dict(),
                            frontend_config=json_native(frontend.config_dict()),
                            shared_frontend_config=json_native(frontend.config_dict()),
                            train_power_cache='nonexistent-cache-must-not-be-read')
            cls.payloads[arm] = dict(schema=SCHEMA,protocol=protocol,model_config=cls.config.to_dict(),
                adapter_options=copy.deepcopy(BASE_OPTIONS),frontend_config=json_native(frontend.config_dict()),
                frontend_learnable=enabled,frontend_state_dict=copy.deepcopy(frontend.state_dict()),
                backbone_state_dict=copy.deepcopy(model.state_dict()),adapter_state_dict=copy.deepcopy(adapter.state_dict()),
                threshold=.5,dev_metrics=dict(threshold=.5,real_fpr_percent=4.9),epoch=3)

    def test_nonzero_trained_knots_are_loaded_and_change_actual_waveform_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            time = np.arange(4321,dtype=np.float32)/16000
            wave = (.2*np.sin(2*np.pi*671*time)+.15*np.sin(2*np.pi*2317*time)
                    +.03*np.random.default_rng(18).standard_normal(len(time))).astype(np.float32)
            audio = root/'audio.wav'
            sf.write(audio,wave,16000,subtype='FLOAT')
            results = {}
            for arm,payload in self.payloads.items():
                checkpoint = root/(arm+'.pt')
                torch.save(payload,checkpoint)
                _,_,frontend,_,loaded,threshold = load_filterbank_checkpoint(checkpoint)
                self.assertTrue(torch.equal(frontend.raw_shift,payload['frontend_state_dict']['raw_shift']))
                self.assertEqual(threshold,.5)
                self.assertEqual(loaded['protocol']['arm'],arm)
                results[arm] = predict(audio,root/(arm+'.json'),checkpoint,threads=1)
                self.assertEqual(results[arm]['native20_actual_frames'],14)
                self.assertEqual(results[arm]['stored20_frames'],16)
                self.assertEqual(results[arm]['frontend_learnable'],ARMS[arm])
                self.assertEqual(results[arm]['frequency_knots_hz'],frontend.frequencies_hz().tolist())
            self.assertFalse(np.array_equal(results['fixed_mel']['scores20'],results['learnable_mel']['scores20']))
            self.assertGreater(np.max(np.abs(np.asarray(results['fixed_mel']['scores20'])
                                           -results['learnable_mel']['scores20'])),1e-7)

    def test_float64_cutoff_and_fractional_end_are_preserved_without_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = copy.deepcopy(self.payloads['learnable_mel'])
            for key in ('backbone_state_dict','adapter_state_dict'):
                for tensor in payload[key].values(): tensor.zero_()
            payload['backbone_state_dict']['classifier.bias'].fill_(1.)
            checkpoint,audio = root/'model.pt',root/'audio.wav'
            sf.write(audio,np.zeros(3252,np.float32),16000,subtype='FLOAT')
            exact = 1.
            for index,cut in enumerate((exact,float(np.nextafter(np.float64(exact),np.inf)))):
                payload['threshold']=payload['dev_metrics']['threshold']=cut
                torch.save(payload,checkpoint)
                output = root/f'prediction{index}.json'
                result = predict(audio,output,checkpoint,threads=1)
                np.testing.assert_array_equal(result['scores20'],np.ones(11))
                self.assertEqual(result['threshold'],cut)
                expected = [[0.,3252/16000]] if index==0 else []
                self.assertEqual(result['predicted_fake_intervals_seconds'],expected)
                with self.assertRaises(FileExistsError): predict(audio,output,checkpoint,threads=1)

    def test_actual_duration_grid_and_invalid_audio(self):
        power_frontend = PowerSpectrogram(self.config)
        for samples in (1,319,320,321,2559,2560,2561):
            batch,duration,actual = power_waveform_batch(np.ones(samples,np.float32),16000,power_frontend)
            storage = int(batch['lengths'][0])
            self.assertEqual(batch['power_features'].shape,(1,1,storage,257))
            self.assertEqual(storage,8*((samples+2559)//2560))
            self.assertEqual(actual,(samples+319)//320)
            self.assertEqual(int((batch['fine_weights']>0).sum()),actual)
            self.assertAlmostEqual(float(batch['fine_weights'].sum()),duration,places=7)
            self.assertAlmostEqual(float(batch['weights'].sum()),duration,places=7)
            self.assertNotIn('features',batch)
        for wave,sample_rate in ((np.zeros(0),16000),(np.zeros((2,2)),16000),
                                 (np.array([np.nan]),16000),(np.ones(100),8000)):
            with self.assertRaises(ValueError): power_waveform_batch(wave,sample_rate,power_frontend)

    def test_schema_flags_configuration_threshold_and_missing_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'model.pt'
            variants = []
            for key in ('frontend_state_dict','frontend_config','adapter_state_dict'):
                payload=copy.deepcopy(self.payloads['learnable_mel']);del payload[key];variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['schema']='old';variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['frontend_learnable']=False;variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['protocol']['arm']='boundary';variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['frontend_config']['shift_fraction']=.5;variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['threshold']=.4;variants.append(payload)
            payload=copy.deepcopy(self.payloads['learnable_mel']);payload['dev_metrics']['real_fpr_percent']=5.1;variants.append(payload)
            for payload in variants:
                torch.save(payload,checkpoint)
                with self.assertRaises(ValueError): load_filterbank_checkpoint(checkpoint)
            payload=copy.deepcopy(self.payloads['learnable_mel'])
            del payload['frontend_state_dict']['raw_shift']
            torch.save(payload,checkpoint)
            with self.assertRaises(RuntimeError): load_filterbank_checkpoint(checkpoint)

    def test_changed_fixed_grid_bounds_nonfinite_and_nonzero_control_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'model.pt'
            for key in ('base_frequencies','max_displacement','fft_frequencies'):
                payload=copy.deepcopy(self.payloads['learnable_mel'])
                payload['frontend_state_dict'][key][1]+=1.
                torch.save(payload,checkpoint)
                with self.assertRaisesRegex(ValueError,'fixed frontend'): load_filterbank_checkpoint(checkpoint)
            payload=copy.deepcopy(self.payloads['fixed_mel'])
            payload['frontend_state_dict']['raw_shift'][0]=.1
            torch.save(payload,checkpoint)
            with self.assertRaisesRegex(ValueError,'zero frequency'): load_filterbank_checkpoint(checkpoint)
            payload=copy.deepcopy(self.payloads['learnable_mel'])
            payload['frontend_state_dict']['raw_shift'][0]=float('nan')
            torch.save(payload,checkpoint)
            with self.assertRaisesRegex(ValueError,'nonfinite'): load_filterbank_checkpoint(checkpoint)


if __name__=='__main__':
    unittest.main()
