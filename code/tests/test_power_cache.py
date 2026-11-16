"""Verify waveform identity, storage tails and immutable power-cache binding."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from cnn_ps.backbone import ModelConfig
from cnn_ps.dual_features import build_feature_cache
from cnn_ps.learnable_filterbank import LearnableFilterbank
from cnn_ps.power_cache import build_power_cache, PowerFeatureData, power_collate


class PowerCacheTests(unittest.TestCase):
    @staticmethod
    def make_frontend(root,split):
        waves=[np.ones(1,np.float32),np.full(3200,.1,np.float32),
               (.1*np.sin(np.arange(8001)*.073)).astype(np.float32)]
        coarse,fine,records={},{},[]
        labels=root/(split+'_coarse.npy')
        for index,wave in enumerate(waves):
            uid=f'TOY_{split}_{index}'
            audio=root/(uid+'.wav')
            sf.write(audio,wave,16000,subtype='FLOAT')
            fine[uid]=np.arange((len(wave)+319)//320)%2
            coarse[uid]=np.ones((len(wave)+2559)//2560)
            records.append(dict(uid=uid,split=split,audio=str(audio),labels_file=str(labels)))
        np.save(labels,coarse)
        actual=root/(split+'_fine.npz')
        np.savez(actual,**fine)
        manifest=root/(split+'_manifest.json')
        manifest.write_text(json.dumps(dict(split=split,records=records)),encoding='utf-8')
        config=root/'config.json'
        config.write_text(json.dumps(ModelConfig().to_dict()),encoding='utf-8')
        frontend=root/(split+'_frontend')
        build_feature_cache(None,manifest,actual,frontend,split,
                            channels=3 if split=='train' else 1,model_config=config)
        return frontend

    @staticmethod
    def close_reader(data):
        data.fine.clear();data.coarse.clear()
        for array in list(data.arrays.values())+[data.power_array]:
            array._mmap.close()

    def test_waveform_projection_rotation_tail_and_copy_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frontend=self.make_frontend(root,'train')
            with patch('torch.load',side_effect=AssertionError('No learned checkpoint')):
                result=build_power_cache(frontend,root/'power')
            self.assertEqual(result['logmel_exact_matches'],9)
            self.assertEqual(result['fixed_projection_exact_matches'],9)
            self.assertEqual(result['fixed_projection_max_absolute_error'],0.)
            data=PowerFeatureData(frontend,root/'power',split='train',verify_hashes=True)
            fixed=LearnableFilterbank(learnable=False)
            for i in range(len(data)):
                for channel in range(3):
                    row=data.get_record(i,channel)
                    torch.testing.assert_close(fixed(row['power_features']),row['features'],rtol=0,atol=0)
            tail=data.get_record(1)
            self.assertEqual(float(tail['fine_weights'][10:].sum()),0.)
            self.assertGreater(float(tail['power_features'][0,10].sum()),0.)
            batch=power_collate([data.get_record(0),data.get_record(2)])
            self.assertEqual(batch['power_features'].shape,(2,1,32,257))
            self.assertEqual(torch.count_nonzero(batch['power_features'][0,0,8:]),0)
            for epoch in (0,1,2,3):
                data.epoch=epoch
                row=data[2]
                self.assertEqual(row['channel'],0 if epoch==0 else int((data.assignment[2]+epoch-1)%3))
            original=data.get_record(2)['power_features'].clone()
            row=data.get_record(2);row['power_features'].fill_(123)
            torch.testing.assert_close(data.get_record(2)['power_features'],original,rtol=0,atol=0)
            with self.assertRaises(FileExistsError):build_power_cache(frontend,root/'power')
            self.close_reader(data)

    def test_changed_waveform_metadata_and_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frontend=self.make_frontend(root,'dev')
            result=build_power_cache(frontend,root/'power')
            self.assertFalse(result['normalization_fitted'])
            metadata=root/'power/complete.json'
            original=metadata.read_text(encoding='utf-8')
            changed=json.loads(original);changed['durations'][0]=.2
            metadata.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'identity'):PowerFeatureData(frontend,root/'power')
            metadata.write_text(original,encoding='utf-8')
            sf.write(root/'TOY_dev_0.wav',np.zeros(1,np.float32),16000,subtype='FLOAT')
            with self.assertRaisesRegex(ValueError,'cannot be reproduced'):build_power_cache(frontend,root/'changed')
            values=np.load(root/'power/power.npy',mmap_mode='r+')
            values[0,0,0]+=1
            values.flush();values._mmap.close()
            with self.assertRaisesRegex(ValueError,'array changed'):PowerFeatureData(frontend,root/'power',verify_hashes=True)


if __name__=='__main__':
    unittest.main()
