"""Optional score-context boundary refiner; inference never accepts labels.

Predict local accuracy gain for moving each original boundary by -2,-1,+1,+2
20ms cells. Zero shift always has utility zero. No new/deleted/merged events.
This is experimental and is not enabled in the detector's default path.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

STEP = .02
OFFSETS = (-2, -1, 1, 2)
MODEL_OPTIONS = dict(loss='squared_error', learning_rate=.05, max_iter=120,
                     max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=10.,
                     early_stopping=False, random_state=20261116)


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)])


@dataclass
class Candidates:
    scores: np.ndarray
    widths: np.ndarray
    original: np.ndarray
    spans: np.ndarray
    features: np.ndarray
    actions: list  # (event index, side 0=start/1=end, offset, first flipped, stop)


def make_candidates(scores, duration, threshold):
    """Only observable detector scores and recording duration enter features."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Expected finite 1D scores and threshold')
    if not np.isfinite(duration) or duration <= 0 or len(scores)*STEP < duration-1e-9:
        raise ValueError('Scores must cover a positive duration')
    t = np.arange(len(scores))*STEP
    w = np.maximum(0., np.minimum(t+STEP, duration)-t)
    scores = scores[w>0]; w=w[w>0]
    original = scores >= np.float64(threshold)
    spans=intervals(original); n=len(scores)
    logits=np.clip(scores-threshold,-12,12); probabilities=expit(scores)
    features=[]; actions=[]
    for event,(a,b) in enumerate(spans):
        previous = spans[event-1,1] if event else -1
        following = spans[event+1,0] if event+1<len(spans) else n+1
        event_score=logits[a:b]
        context=[np.log1p(b-a), min(a-previous,50), min(following-b,50),
                 float(event_score.mean()),float(event_score.std()),float(event_score.min()),float(event_score.max())]
        for side,edge in enumerate((a,b)):
            if edge in (0,n):continue  # audio-edge boundaries are censored/pinned
            indices=edge+np.arange(-8,8)
            valid=(indices>=0)&(indices<n)
            local=logits[np.clip(indices,0,n-1)]
            lp=probabilities[np.clip(indices,0,n-1)]
            base=[side,*context,*local,*lp,*valid.astype(float),
                  float(local[8]-local[7]),float(local[8:].mean()-local[:8].mean())]
            for offset in OFFSETS:
                new=edge+offset
                allowed=(max(0,previous+1)<=new<b if side==0 else a<new<=min(n,following-1))
                if not allowed:continue
                lo,hi=sorted((edge,new));changed=logits[lo:hi]
                expands=(side==0 and offset<0) or (side==1 and offset>0)
                features.append([*base,offset,abs(offset),int(expands),float(changed.mean()),
                                 float(changed.min()),float(changed.max()),float(w[lo:hi].sum()/STEP)])
                actions.append((event,side,offset,lo,hi))
    # Stable feature width even for an all-real/all-fake recording.
    return Candidates(scores,w,original,spans,np.asarray(features,dtype=float).reshape(-1,65),actions)


def gain_targets(candidates, labels):
    """Label-assisted supervision only; target is exact local net correct cells.

    All original boundaries participate, including misdetections and low-IoU
    candidates. No event matching or truth-boundary selection is involved.
    """
    y=np.asarray(labels,dtype=bool)
    if y.shape!=candidates.original.shape:raise ValueError('Label length mismatch')
    correctness_gain=np.where(candidates.original==y,-1.,1.)*candidates.widths/STEP
    prefix=np.r_[0.,np.cumsum(correctness_gain)]
    return np.array([prefix[hi]-prefix[lo] for _,_,_,lo,hi in candidates.actions])


class LocalBoundaryRefiner:
    def __init__(self):
        self.model=HistGradientBoostingRegressor(**MODEL_OPTIONS)

    def fit(self,candidates,labels):
        if len(candidates)!=len(labels):raise ValueError('Recording count mismatch')
        x=np.concatenate([c.features for c in candidates])
        y=np.concatenate([gain_targets(c,label) for c,label in zip(candidates,labels)])
        with threadpool_limits(limits=2):self.model.fit(x,y)
        self.fit_summary={'recordings':len(candidates),'candidate_actions':len(y),
                          'beneficial_action_fraction':float((y>0).mean()),'feature_count':x.shape[1]}
        return self

    def predict_gains(self,candidates):
        lengths=[len(c.actions) for c in candidates]
        if not sum(lengths):return [np.empty(0) for _ in candidates]
        with threadpool_limits(limits=2):g=self.model.predict(np.concatenate([c.features for c in candidates]))
        return np.split(g,np.cumsum(lengths)[:-1])

    def apply(self,scores,duration,threshold,margin=.25):
        c=make_candidates(scores,duration,threshold)
        return apply_gains(c,self.predict_gains([c])[0],margin)[0]


def apply_gains(c,gains,margin):
    """Choose compatible local moves; margin=None explicitly disables changes."""
    gains=np.asarray(gains,dtype=float)
    if gains.shape!=(len(c.actions),) or not np.isfinite(gains).all():raise ValueError('Invalid action gains')
    if margin is not None and (not np.isfinite(margin) or margin<0):raise ValueError('Invalid margin')
    revised=c.spans.copy();options=[[{0:0.},{0:0.}] for _ in c.spans]
    if margin is not None:
        for (event,side,offset,lo,hi),gain in zip(c.actions,gains):
            if gain>margin+1e-12:options[event][side][offset]=float(gain-margin)
        for event,(a,b) in enumerate(c.spans):
            choices=[(ga+gb,-abs(da)-abs(db),-abs(da),da,db)
                     for da,ga in options[event][0].items() for db,gb in options[event][1].items()
                     if a+da<b+db]
            _,_,_,da,db=max(choices)
            revised[event]=[a+da,b+db]
        # Independently expanding neighbors can collide. Revert both facing
        # boundaries to preserve the original event topology and one real cell.
        for event in range(len(revised)-1):
            if revised[event,1]>=revised[event+1,0]:
                revised[event,1]=c.spans[event,1]
                revised[event+1,0]=c.spans[event+1,0]
    mask=np.zeros_like(c.original)
    for a,b in revised:mask[a:b]=True
    assert len(intervals(mask))==len(c.spans)
    assert np.all(abs(revised-c.spans)<=2)
    return mask,revised
