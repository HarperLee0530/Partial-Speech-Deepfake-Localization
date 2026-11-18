"""SAL's original backend and forward method; replace only feature extraction."""
from src.models.net.model import SSLSeq8Bin


class LogMelSAL(SSLSeq8Bin):
    def __init__(self, resolution_test=0.16):
        super().__init__(ssl_encoder='logmel', mode='logmel', hid_dim=1024,
                         seq_model='cf', resolution_train=0.16,
                         resolution_test=resolution_test, pool='avg')

    def _extract_features(self, waveform):
        # Do not apply WavLM's 256-sample padding to the original LogMel.
        return self.ssl_encoder.extract_feat(waveform)
