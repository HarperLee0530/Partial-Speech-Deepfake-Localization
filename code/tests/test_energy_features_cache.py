"""Independent signal/grid checks for the pitch-free energy input cache."""
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
from cnn_ps.energy_features import extract
from cnn_ps.energy_cache import build_energy_cache, EnergyFeatureData, energy_collate


def independent_energy(wave):
    """Slow direct windows and neighbour enumeration, without scipy filtering."""
    wave = np.asarray(wave, dtype=np.float64)
    n = (len(wave)+319)//320
    rms = []
    for t in range(n):
        indices = np.arange(t*320-40, t*320+360)
        valid = (indices >= 0) & (indices < len(wave))
        values = np.zeros(400)
        values[valid] = wave[indices[valid]]
        rms.append(np.sqrt(np.mean(values**2)))
    if not np.any(rms):
        return np.zeros((n, 5), dtype=np.float32)
    energy = np.log(np.maximum(rms, 1e-8))
    energy -= np.median(energy)
    columns = [energy, np.r_[0., np.diff(energy)]]
    short_std = []
    for radius in (5, 25):
        deviation = []
        for t in range(n):
            neighbours = [energy[j] for j in range(max(0,t-radius), min(n,t+radius+1)) if j != t]
            deviation.append(energy[t]-np.mean(neighbours) if neighbours else 0.)
            if radius == 5:
                short_std.append(np.std(neighbours) if neighbours else 0.)
        columns.append(deviation)
    columns.append(short_std)
    return np.column_stack(columns).astype(np.float32)


class EnergyFeaturesTests(unittest.TestCase):
    def test_exact_frame_centers_and_neighbours_exclude_current_at_edges(self):
        rng = np.random.default_rng(666)
        for n in (1, 319, 320, 321, 2561, 20000):
            wave = rng.normal(size=n) * np.linspace(.001, .2, n)
            np.testing.assert_allclose(extract(wave), independent_energy(wave), atol=2e-7, rtol=2e-6)

    def test_silence_and_legitimate_zero_vectors_are_retained(self):
        np.testing.assert_array_equal(extract(np.zeros(2561)), np.zeros((9, 5)))
        np.testing.assert_array_equal(extract(np.ones(1)), np.zeros((1, 5)))

    def test_relative_features_are_gain_invariant_without_floor_effects(self):
        t = np.arange(16003)/16000
        wave = (.2+.1*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*180*t)
        np.testing.assert_allclose(extract(wave), extract(wave*.3), atol=2e-6)

    def test_no_pitch_dependency_and_invalid_audio_rejected(self):
        with patch.dict('sys.modules', {'librosa': None}):
            self.assertEqual(extract(np.ones(321)).shape, (2, 5))
        for wave, sr in (([],16000), ([[1,2]],16000), ([np.nan],16000), ([1],8000)):
            with self.assertRaises(ValueError):
                extract(wave, sr)


class EnergyCacheTests(unittest.TestCase):
    @staticmethod
    def close_reader(data):
        # Windows cannot remove the temporary fixture while memmaps are open.
        data.fine.clear()
        data.coarse.clear()
        for array in list(data.arrays.values()) + [data.energy_array]:
            array._mmap.close()

    def make_frontend(self, root, split):
        marker = 'T' if split == 'train' else 'D'
        waves = [np.ones(1, dtype=np.float32), np.zeros(2561, dtype=np.float32),
                 (.1*np.sin(np.arange(8001)*.073)).astype(np.float32)]
        coarse, fine, records = {}, {}, []
        labels = root/(split+'_coarse.npy')
        for i, wave in enumerate(waves):
            uid = f'TOY_{marker}_{i}'
            audio = root/(uid+'.wav')
            sf.write(audio, wave, 16000, subtype='FLOAT')
            n = (len(wave)+319)//320
            fine[uid] = np.arange(n)%2
            coarse[uid] = np.ones((len(wave)+2559)//2560)
            records.append(dict(uid=uid, split=split, audio=str(audio), labels_file=str(labels)))
        np.save(labels, coarse)
        labels20 = root/(split+'_fine.npz')
        np.savez(labels20, **fine)
        manifest = root/(split+'_manifest.json')
        manifest.write_text(json.dumps(dict(split=split,records=records)), encoding='utf-8')
        config = root/'config.json'
        config.write_text(json.dumps(ModelConfig().to_dict()), encoding='utf-8')
        folder = root/(split+'_frontend')
        with patch('torch.load', side_effect=AssertionError('No learned checkpoint may be read')):
            build_feature_cache(None, manifest, labels20, folder, split,
                                channels=3 if split == 'train' else 1, model_config=config)
        return folder

    def test_complete_cache_rotation_masks_clean_statistics_and_copy_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend = self.make_frontend(root, 'train')
            energy = root/'train_energy'
            with patch('torch.load', side_effect=AssertionError('No learned checkpoint may be read')):
                completed = build_energy_cache(frontend, energy)
            data = EnergyFeatureData(frontend, energy, split='train', verify_hashes=True)
            self.assertEqual(completed['logmel_exact_matches'], 9)
            stats = json.loads((energy/'normalization.json').read_text())
            raw = np.concatenate([data.get_record(i)['energy_features'][:data.lengths20[i]].numpy()
                                  for i in range(len(data))]).astype(np.float64)
            self.assertEqual(stats['valid_train_frames'], 1+9+26)
            np.testing.assert_allclose(stats['mean'], raw.mean(0), atol=1e-12)
            np.testing.assert_allclose(stats['std'], raw.std(0).clip(1e-4), atol=1e-12)
            for epoch in (0,1,2,3):
                data.epoch = epoch
                row = data[2]
                channel = 0 if epoch == 0 else int((data.assignment[2]+epoch-1)%3)
                self.assertEqual(row['channel'], channel)
                np.testing.assert_array_equal(row['energy_features'], data.get_record(2,channel)['energy_features'])
            batch = energy_collate([data.get_record(0), data.get_record(2)])
            self.assertEqual(batch['energy_features'].shape, (2,32,5))
            self.assertEqual(float(batch['fine_weights'][0].sum()), np.float32(1/16000))
            self.assertEqual(torch.count_nonzero(batch['energy_features'][0]).item(), 0)
            original = data.get_record(2)['energy_features'].clone()
            row = data.get_record(2)
            row['energy_features'].fill_(999)
            torch.testing.assert_close(data.get_record(2)['energy_features'], original)
            with self.assertRaises(FileExistsError):
                build_energy_cache(frontend, energy)
            changed = json.loads((energy/'complete.json').read_text())
            changed['uids'][0] = 'different'
            (energy/'complete.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                EnergyFeatureData(frontend, energy)
            self.close_reader(data)

    def test_development_never_fits_normalization_and_waveform_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frontend = self.make_frontend(root, 'dev')
            energy = root/'dev_energy'
            complete = build_energy_cache(frontend, energy)
            self.assertIsNone(complete['normalization'])
            self.assertFalse((energy/'normalization.json').exists())
            data = EnergyFeatureData(frontend, energy, split='dev', verify_hashes=True)
            self.assertEqual(data[0]['channel'], 0)
            sf.write(root/'TOY_D_0.wav', np.zeros(1, np.float32), 16000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'waveform/frontend cache mismatch'):
                build_energy_cache(frontend, root/'changed_wave_energy')
            self.close_reader(data)


if __name__ == '__main__':
    unittest.main()
