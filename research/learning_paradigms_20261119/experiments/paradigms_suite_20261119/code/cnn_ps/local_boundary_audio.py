"""Waveform features for the optional local boundary refiner, without labels."""
from dataclasses import replace
import numpy as np
from .local_boundary import LocalBoundaryRefiner,make_candidates,apply_gains

SAMPLE_RATE=16000
HOP=320
WINDOW=400
NFFT=512
BANDS=(0,250,500,1000,1500,2000,3000,4000,8000)
FEATURE_NAMES=('log_rms_db','zero_crossing_rate','spectral_centroid','spectral_bandwidth',
               'spectral_flatness','positive_spectral_flux',
               *[f'power_fraction_{lo}_{hi}hz' for lo,hi in zip(BANDS[:-1],BANDS[1:])])
ADDED_FEATURE_COUNT=9*len(FEATURE_NAMES)


def waveform_features(wave,sample_rate=SAMPLE_RATE):
    """One row per20ms cell. RMS/ZCR use actual cell samples, not padding.

    Spectra use a25ms Hann window centered at each cell's midpoint, zero padded
    at recording edges, and a512-point FFT. All transformations are fixed and
    recording-local; no dataset-wide normalization is fitted.
    """
    x=np.asarray(wave,dtype=np.float64)
    if sample_rate!=SAMPLE_RATE or x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty mono16kHz audio')
    n=(len(x)+HOP-1)//HOP;indices=np.arange(n)[:,None]*HOP+np.arange(HOP)
    valid=indices<len(x);cells=x[np.minimum(indices,len(x)-1)]*valid
    count=valid.sum(1);power=(cells*cells).sum(1)/count
    energy=10*np.log10(np.maximum(power,1e-12))
    crossings=((cells[:,1:]>=0)!=(cells[:,:-1]>=0))&valid[:,1:]
    zcr=crossings.sum(1)/np.maximum(count-1,1)
    centers=np.arange(n)*HOP+HOP//2
    idx=centers[:,None]+np.arange(WINDOW)-WINDOW//2
    inside=(idx>=0)&(idx<len(x))
    frames=x[np.clip(idx,0,len(x)-1)]*inside*np.hanning(WINDOW)
    spectrum=np.abs(np.fft.rfft(frames,n=NFFT,axis=1))**2
    total=spectrum.sum(1);shape=spectrum/np.maximum(total[:,None],1e-20)
    frequency=np.fft.rfftfreq(NFFT,1/SAMPLE_RATE);normfreq=frequency/(SAMPLE_RATE/2)
    centroid=(shape*normfreq).sum(1)
    bandwidth=np.sqrt((shape*(normfreq[None,:]-centroid[:,None])**2).sum(1))
    flatness=np.exp(np.log(np.maximum(spectrum,1e-20)).mean(1))/np.maximum(spectrum.mean(1),1e-20)
    flatness[total<1e-16]=0.
    flux=np.r_[0.,np.sqrt(np.maximum(np.diff(shape,axis=0),0.)**2 @ np.ones(shape.shape[1]))]
    fractions=[]
    for j,(lo,hi) in enumerate(zip(BANDS[:-1],BANDS[1:])):
        selected=(frequency>=lo)&((frequency<=hi) if j==len(BANDS)-2 else (frequency<hi))
        fractions.append(shape[:,selected].sum(1))
    result=np.column_stack([energy,zcr,centroid,bandwidth,flatness,flux,*fractions])
    assert result.shape==(n,len(FEATURE_NAMES)) and np.isfinite(result).all()
    return result


def add_audio_features(candidates,audio_features):
    """Append context at original/proposed boundaries and changed-cell features."""
    af=np.asarray(audio_features,dtype=float);n=len(candidates.original)
    if af.shape!=(n,len(FEATURE_NAMES)) or not np.isfinite(af).all():
        raise ValueError('Waveform features must align exactly with active20ms scores')
    features=[]
    for event,side,offset,lo,hi in candidates.actions:
        edge=int(candidates.spans[event,side]);new=edge+offset
        left=af[max(0,edge-4):edge];right=af[edge:min(n,edge+4)]
        lmean=left.mean(0);rmean=right.mean(0)
        original_delta=af[edge]-af[edge-1]
        deviation=af[max(0,edge-4):min(n,edge+4)].std(0)
        before=af[max(0,new-1)];after=af[min(n-1,new)]
        features.append(np.r_[lmean,rmean,rmean-lmean,original_delta,deviation,
                              before,after,after-before,af[lo:hi].mean(0)])
    additional=np.asarray(features,dtype=float).reshape(-1,ADDED_FEATURE_COUNT)
    return replace(candidates,features=np.column_stack([candidates.features,additional]))


class AudioBoundaryRefiner(LocalBoundaryRefiner):
    def apply(self,scores,duration,threshold,wave,sample_rate=SAMPLE_RATE,margin=.25):
        if abs(len(wave)/sample_rate-duration)>1e-8:raise ValueError('Waveform/score duration mismatch')
        c=add_audio_features(make_candidates(scores,duration,threshold),waveform_features(wave,sample_rate))
        return apply_gains(c,self.predict_gains([c])[0],margin)[0]
