"""Frontend invariants: original Mel alignment, constrained learning and pairing."""
import unittest

import torch
import torchaudio

from cnn_ps.backbone import LogMel, ModelConfig
from cnn_ps.learnable_filterbank import LearnableFilterbank, PowerSpectrogram
from cnn_ps.train_four import model_config


class LearnableFilterbankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = model_config()

    def test_initial_matrix_matches_torchaudio_without_consuming_rng(self):
        cpu_state = torch.get_rng_state().clone()
        gpu_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        fixed = LearnableFilterbank(self.config, learnable=False)
        learned = LearnableFilterbank(self.config, learnable=True)
        PowerSpectrogram(self.config)
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(gpu_states,torch.cuda.get_rng_state_all())))
        expected = torchaudio.functional.melscale_fbanks(257,0.,8000.,80,16000,norm='slaney',mel_scale='htk')
        self.assertTrue(torch.equal(fixed.filterbank_matrix(),expected))
        self.assertTrue(torch.equal(learned.filterbank_matrix(),expected))
        self.assertEqual(sum(p.numel() for p in learned.parameters() if p.requires_grad),80)
        self.assertEqual(sum(p.numel() for p in fixed.parameters() if p.requires_grad),0)
        self.assertEqual(fixed.config_dict(),learned.config_dict())
        self.assertTrue(all(torch.equal(v,learned.state_dict()[k]) for k,v in fixed.state_dict().items()))

    def test_both_arms_are_bitwise_equal_on_each_available_device_and_rank(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        base_matrix = LearnableFilterbank(self.config).filterbank_matrix().detach()
        for device in devices:
            fixed = LearnableFilterbank(self.config,False).to(device)
            learned = LearnableFilterbank(self.config,True).to(device)
            self.assertTrue(torch.equal(fixed.filterbank_matrix(),learned.filterbank_matrix()))
            # On the installed CUDA runtime this matrix also matches CPU bitwise.
            self.assertTrue(torch.equal(fixed.filterbank_matrix().cpu(),base_matrix))
            generator = torch.Generator(device=device).manual_seed(783)
            for shape in ((257,),(11,257),(3,1,11,257)):
                power = torch.rand(shape, generator=generator, device=device)
                actual = learned(power)
                self.assertEqual(actual.shape,(*shape[:-1],80))
                self.assertTrue(torch.equal(actual,fixed(power)))

    def test_bounded_knots_are_ordered_and_every_filter_stays_nonempty(self):
        module = LearnableFilterbank(self.config)
        original = module.frequencies_hz().detach().clone()
        gen = torch.Generator().manual_seed(561)
        patterns = [torch.full((80,),100.),torch.full((80,),-100.),
                    torch.tensor([100.,-100.]*40),torch.tensor([-100.,100.]*40)]
        patterns.extend(torch.randn((30,80),generator=gen)*10.)
        for values in patterns:
            with torch.no_grad(): module.raw_shift.copy_(values)
            points, matrix = module.frequencies_hz(),module.filterbank_matrix()
            self.assertTrue((points[1:]>points[:-1]).all())
            self.assertEqual(points[0],original[0])
            self.assertEqual(points[-1],original[-1])
            self.assertTrue(((points[1:-1]-original[1:-1]).abs()<=module.max_displacement+1e-3).all())
            self.assertTrue(torch.isfinite(matrix).all())
            self.assertTrue((matrix>=0).all())
            self.assertTrue((matrix.sum(0)>0).all())
            # Every triangle has support wider than the FFT-grid spacing.
            self.assertTrue((points[2:]-points[:-2]>16000/512).all())

    def test_finite_gradients_update_frequencies_but_silence_stays_finite(self):
        module = LearnableFilterbank(self.config)
        generator = torch.Generator().manual_seed(19)
        power = torch.rand((2,13,257),generator=generator).requires_grad_(True)
        original = module.frequencies_hz().detach().clone()
        optimizer = torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.0001)
        output = module(power)
        coefficients = torch.linspace(.1,1.,80)
        (output*coefficients).square().mean().backward()
        self.assertTrue(torch.isfinite(module.raw_shift.grad).all())
        self.assertGreater(torch.count_nonzero(module.raw_shift.grad).item(),0)
        self.assertTrue(torch.isfinite(power.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(module.frequencies_hz(),original))
        silence = module(torch.zeros(3,257))
        self.assertTrue(torch.isfinite(silence).all())
        self.assertTrue(torch.equal(silence,torch.full_like(silence,torch.tensor(1e-10).log())))

    def test_power_grid_and_logmel_are_exact_for_actual_and_storage_lengths(self):
        power_frontend = PowerSpectrogram(self.config)
        filters = LearnableFilterbank(self.config,False)
        original = LogMel(self.config)
        generator = torch.Generator().manual_seed(811)
        for samples in (1,319,320,321,2559,2560,2561,43210):
            wave = torch.rand(samples,generator=generator)*2-1
            actual_power = power_frontend(wave)
            self.assertEqual(actual_power.shape,(1,(samples+319)//320,257))
            self.assertTrue(torch.isfinite(actual_power).all())
            self.assertTrue((actual_power>=0).all())
            self.assertTrue(torch.equal(filters(actual_power),original(wave)))
        # Grid check with a pulse exactly at the first label-cell midpoint.
        wave = torch.zeros(2560)
        wave[160] = 1.
        self.assertTrue(torch.equal(filters(power_frontend(wave)),original(wave)))

    def test_invalid_audio_power_configuration_and_parameters_are_rejected(self):
        module,power_frontend=LearnableFilterbank(self.config),PowerSpectrogram(self.config)
        for value in (torch.zeros(0,257),torch.zeros(2,256),torch.zeros(257,dtype=torch.float64),
                      torch.full((257,),-1.),torch.full((257,),float('nan')),
                      torch.full((257,),float('inf'))):
            with self.assertRaises(ValueError): module(value)
        for value in (torch.zeros(0),torch.zeros(2,300),torch.zeros(300,dtype=torch.float64),
                      torch.full((300,),float('nan')),torch.full((300,),float('inf'))):
            with self.assertRaises(ValueError): power_frontend(value)
        with self.assertRaises(ValueError): LearnableFilterbank(ModelConfig(n_mels=40))
        with self.assertRaises(ValueError): PowerSpectrogram(ModelConfig(hop_length=160))
        with self.assertRaises(ValueError): LearnableFilterbank(learnable=1)
        with torch.no_grad(): module.raw_shift[0]=float('nan')
        with self.assertRaises(ValueError): module.filterbank_matrix()


if __name__=='__main__':
    unittest.main()
