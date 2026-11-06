"""Read-only SAL diagnostics. Run this file to plot both completed SAL runs.

No inference, GPU use, threshold fitting, or changes to source experiment files.
Requires numpy, scipy, matplotlib. Outputs PNG/PDF/SVG, CSV and provenance JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

from visualization_paths import CODE_ROOT as CODE, RESULT_ROOT, PROJECT_ROOT, OUTPUT_ROOT, project_path, output_path, source_run

COLORS = dict(light='#A7D7E8', cyan='#7EC1D9', blue='#5E9CC5',
              lavender='#B79FD9', purple='#8A74C4', navy='#5B6BA6',
              ink='#4A4A4A', grid='#E8EEF6', white='#FFFFFF',
              background='#FFFFFF', outline='#4A4A4A')
BINS = np.array([0, .16, .32, .64, 1.28, 2.56, np.inf])
BIN_LABELS = ['(0, 0.16]', '(0.16, 0.32]', '(0.32, 0.64]',
              '(0.64, 1.28]', '(1.28, 2.56]', '> 2.56']
CATEGORIES = ['miss', 'false_alarm', 'shrink', 'expand', 'shift', 'fragment', 'merge', 'correct']
TITLES = dict(miss='Complete miss', false_alarm='Spurious interval', shrink='Contracted interval',
              expand='Expanded interval', shift='Shifted interval', fragment='Fragmented prediction',
              merge='Merged true events', correct='Correct control')
EPS = 1e-8


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intervals(mask, dt, seconds):
    """Runs on the saved label grid; clip the padded final cell to real audio."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    runs = edges.reshape(-1, 2).astype(float) * dt
    runs = np.minimum(runs, seconds)
    return runs[(runs[:, 1] - runs[:, 0]) > EPS]


def overlap_and_match(truth, prediction, threshold=.5):
    n, m = len(truth), len(prediction)
    if not n or not m:
        return np.zeros((n, m)), np.zeros((n, m)), {}
    overlap = np.maximum(0, np.minimum(truth[:, None, 1], prediction[None, :, 1])
                         - np.maximum(truth[:, None, 0], prediction[None, :, 0]))
    union = (truth[:, 1] - truth[:, 0])[:, None] + (prediction[:, 1] - prediction[:, 0])[None, :] - overlap
    iou = overlap / union
    eligible = iou >= threshold - EPS
    # Maximize eligible match count first, then summed tIoU. Invalid edges have reward 0.
    reward = np.where(eligible, min(n, m) + 1 + iou, 0)
    rr, cc = linear_sum_assignment(-reward)
    matched = {int(r): int(c) for r, c in zip(rr, cc) if eligible[r, c]}
    return overlap, iou, matched


def bin_index(duration):
    return int(np.searchsorted(BINS[1:], duration - EPS, side='left'))


class Examples:
    """Stable hash reservoir, then median-severity selection (not worst-case selection)."""
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.heaps = {c: [] for c in CATEGORIES}
        self.counts = {c: 0 for c in CATEGORIES}

    def add(self, category, uid, start, end, severity, detail):
        self.counts[category] += 1
        key = f'{uid}|{category}|{start:.8f}|{end:.8f}'
        rank = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
        item = (-rank, key, dict(uid=uid, category=category, start=float(start), end=float(end),
                                severity=float(severity), detail=detail))
        pool = self.heaps[category]
        if len(pool) < self.capacity:
            heapq.heappush(pool, item)
        elif rank < -pool[0][0]:
            heapq.heapreplace(pool, item)

    def selected(self):
        selected, used = {}, set()
        for category in CATEGORIES:
            items = [x[2] for x in self.heaps[category]]
            if not items:
                continue
            median = float(np.median([x['severity'] for x in items]))
            items.sort(key=lambda x: (x['uid'] in used, abs(x['severity'] - median), x['uid'], x['start']))
            selected[category] = items[0]
            used.add(items[0]['uid'])
        return selected


def bootstrap_rates(record_counts, reps, seed=20261106):
    # Columns: truth count, complete miss, matched, fragmentation.
    # Cluster resampling at recording level, keeping all its events together.
    data = np.asarray(record_counts, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.full((reps, len(BIN_LABELS), 3), np.nan)
    for i in range(reps):
        ids = rng.integers(0, len(data), size=len(data))
        total = data[ids].sum(axis=0)
        out[i] = np.divide(total[:, 1:], total[:, :1],
                           out=np.full((len(BIN_LABELS), 3), np.nan), where=total[:, :1] > 0)
    return out


def figure_style():
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'axes.edgecolor': COLORS['ink'], 'axes.linewidth': .75,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.bottom': True, 'axes.spines.left': True, 'axes.titleweight': 'bold',
        'axes.titlesize': 12, 'figure.facecolor': 'white', 'axes.facecolor': COLORS['background'],
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'axes.axisbelow': True})


def save_figure(fig, path):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.' + ext), dpi=210, bbox_inches='tight')
    plt.close(fig)


def timeline(ax, db, example, dt, dataset, short=False):
    row = db.execute('SELECT logits,labels,seconds FROM predictions WHERE uid=?', (example['uid'],)).fetchone()
    logits = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)
    labels = np.frombuffer(row[1], dtype=np.uint8)
    seconds = row[2]
    prob = 1 / (1 + np.exp(np.clip(logits[:, 1].astype(float) - logits[:, 0], -700, 700)))
    truth = intervals(labels == 0, dt, seconds)
    pred = intervals(logits.argmax(axis=1) == 0, dt, seconds)
    start, end = example['start'], example['end']
    pad = max(.6, (end - start) * .7, 3 * dt)
    lo, hi = max(0, start - pad), min(seconds, end + pad, len(labels) * dt)
    edges = np.minimum(np.arange(len(labels) + 1) * dt, seconds)
    ax.stairs(prob, edges, color=COLORS['navy'], linewidth=1.5, baseline=None)
    ax.axhline(.5, color=COLORS['ink'], linestyle='--', linewidth=.85, alpha=.7)
    for s, e in truth:
        ax.broken_barh([(s, e-s)], (-.24, .12), facecolors=COLORS['cyan'], edgecolors=COLORS['blue'], linewidth=.5)
    for s, e in pred:
        ax.broken_barh([(s, e-s)], (-.43, .12), facecolors=COLORS['purple'], edgecolors=COLORS['navy'], linewidth=.5)
    ax.axvspan(start, end, facecolor=COLORS['cyan'], edgecolor=COLORS['blue'],
               alpha=.23, linewidth=.7, zorder=-1)
    ax.set(xlim=(lo, hi), ylim=(-.48, 1.06), xlabel='Time (s)', ylabel='Spoof score')
    ax.set_yticks([-.37, -.18, 0, .5, 1], ['SAL', 'Truth', '0', '0.5', '1'])
    ax.set_ylabel('Spoof score', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='x', color=COLORS['grid'], linewidth=.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['ink'])
        spine.set_linewidth(.75)
    ax.text(.01, .98, example['uid'], transform=ax.transAxes, va='top', fontsize=7.5 if short else 9,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
    ax.text(.01, -.29, example['detail'], transform=ax.transAxes, fontsize=8 if short else 9,
            color=COLORS['ink'])


def plot_examples(db, selected, counts, dt, title, out):
    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    for i, (cat, ax) in enumerate(zip(CATEGORIES, axes.flat)):
        ax.set_title(f'{chr(97+i)}  {TITLES[cat]}', loc='left', pad=15)
        if cat in selected:
            timeline(ax, db, selected[cat], dt, title, short=True)
        else:
            ax.text(.5, .5, 'No eligible case under the stated definition', ha='center', transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle(f'{title} | SAL interval diagnostics', x=.07, ha='left', fontsize=20, fontweight='bold', y=.995)
    fig.text(.07, .968, f'Partial recordings only  |  saved grid: {dt*1000:g} ms  |  fixed argmax decision  |  retrospective test analysis', fontsize=10)
    handles = [Patch(color=COLORS['cyan'], label='True spoof interval'), Patch(color=COLORS['purple'], label='Predicted spoof interval'),
               Line2D([0], [0], color=COLORS['navy'], label='Binary softmax spoof score (uncalibrated)'),
               Line2D([0], [0], color=COLORS['ink'], ls='--', label='Decision threshold: 0.5')]
    legend=fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .95), ncol=4,
                      frameon=True, fontsize=9, edgecolor='#6B6B6B', facecolor='white', framealpha=1)
    legend.get_frame().set_linestyle('--')
    legend.get_frame().set_linewidth(1)
    fig.subplots_adjust(top=.90, bottom=.105, left=.07, right=.985, hspace=.85, wspace=.20)
    fig.text(.07, .012, 'Examples illustrate error shapes, not their prevalence. Stable hash sample, then near-median severity; no smoothing or gap filling.\n'
             'Truth is reconstructed on the saved evaluation grid; boundaries below that resolution are not resolved.', fontsize=9)
    save_figure(fig, out/'figure1_timeline_gallery')
    single_dir = out/'timeline_cases'
    single_dir.mkdir(exist_ok=True)
    for cat, example in selected.items():
        fig, ax = plt.subplots(figsize=(11, 4.2))
        timeline(ax, db, example, dt, title)
        ax.set_title(f'{title} | {TITLES[cat]} | {dt*1000:g} ms grid', loc='left', pad=22)
        fig.subplots_adjust(left=.09, right=.98, bottom=.26, top=.83)
        save_figure(fig, single_dir/cat)


def plot_groups(rows, boundary, total, dt, title, out, bootstrap):
    n = np.array([r['true_events'] for r in rows])
    missed = np.array([r['complete_misses'] for r in rows])
    matched = np.array([r['matched_tiou_0_5'] for r in rows])
    x = np.arange(len(n))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    bars = ax.bar(x, n, color=COLORS['blue'], width=.66,
                  edgecolor=COLORS['outline'], linewidth=.9)
    ax.bar_label(bars, labels=[f'{v:,}' for v in n], padding=4, fontsize=9)
    ax.set_title('a  True spoof-event duration', loc='left')
    ax.set_ylabel('Number of events')
    ax.set_ylim(0, max(n)*1.22 if max(n) else 1)
    ax.grid(axis='y', color=COLORS['grid'])
    ax = axes[0, 1]
    rates = np.divide(np.array([missed, n-matched]), n, out=np.zeros((2,len(n))), where=n>0)*100
    max_upper=0
    for j,(vals,color,label) in enumerate(zip(rates,[COLORS['purple'],COLORS['blue']],
            ['Complete miss: no overlap', 'Unmatched: tIoU < 0.5 or match conflict'])):
        low=[];high=[]
        for i,nn in enumerate(n):
            dist=bootstrap[:,i,0] if j==0 else 1-bootstrap[:,i,1]
            lo,hi=np.nanpercentile(dist,[2.5,97.5])*100 if nn else (0,0)
            low.append(max(0,vals[i]-lo));high.append(max(0,hi-vals[i]))
            max_upper=max(max_upper,hi)
        xx=x+(-.12 if j==0 else .12)
        ax.errorbar(xx,vals,yerr=[low,high],fmt='o-' if j==0 else 's--',color=color,
                    capsize=3,lw=1.4,markersize=5,label=label)
        for i,v in enumerate(vals):
            ax.annotate(f'{v:.2f}%',(xx[i],v),xytext=(0,8 if j==1 else -16),
                        textcoords='offset points',ha='center',fontsize=8,color=color)
    ax.set_title('b  Event miss rates | two definitions', loc='left')
    ax.set(ylabel='Share of true events (%)', ylim=(-max(1,max_upper*.08),max_upper*1.20+1))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.26), frameon=False, fontsize=8, ncol=1)
    ax.grid(axis='y', color=COLORS['grid'])
    boundary_rows=[]
    for j, name in enumerate(['start', 'end']):
        ax = axes[1, j]
        matrix=np.zeros((5,len(n)))
        counts=[]
        for k in x:
            v=np.array(boundary[k][j]);counts.append(len(v))
            if len(v):
                matrix[:,k]=100*np.array([(v < -dt-EPS).sum(),((v>=-dt-EPS)&(v < -EPS)).sum(),
                    (abs(v)<=EPS).sum(),((v>EPS)&(v<=dt+EPS)).sum(),(v>dt+EPS).sum()])/len(v)
                assert abs(matrix[:,k].sum()-100)<1e-6
        cmap=LinearSegmentedColormap.from_list('boundary_'+name,['#FFFFFF',COLORS['light'],COLORS['blue']] if j==0
                                              else ['#FFFFFF',COLORS['lavender'],COLORS['purple']])
        ax.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect='auto')
        for cell_row in range(5):
            for cell_col in x:
                ax.add_patch(Rectangle((cell_col-.5, cell_row-.5), 1, 1, fill=False,
                                       edgecolor=COLORS['outline'], linewidth=.65, zorder=3))
        labels=[f'Earlier > {dt*1000:g} ms',f'Earlier <= {dt*1000:g} ms','Exact on saved grid',
                f'Later <= {dt*1000:g} ms',f'Later > {dt*1000:g} ms']
        for a,label in enumerate(labels):
            for b in x:
                boundary_rows.append(dict(boundary=name,duration_bin=BIN_LABELS[b],offset_group=label,
                                          matched_events=counts[b],percentage=float(matrix[a,b])))
        ax.set_yticks(range(5),labels,fontsize=9)
        ax.set_title(f'{"c" if j==0 else "d"}  {name.capitalize()} boundary offset | matched events',loc='left',pad=28)
        for a in range(5):
            for b in x:
                value=matrix[a,b]
                ax.text(b,a,f'{value:.1f}%' if counts[b] else 'n/a',ha='center',va='center',fontsize=9,
                        color='white' if value>70 else COLORS['ink'])
        for k in x:
            ax.text(k,1.025,f'n={counts[k]:,}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set_xticks(np.arange(-.5,len(n),1),minor=True)
        ax.set_yticks(np.arange(-.5,5,1),minor=True)
        ax.grid(which='minor',visible=False)
        ax.tick_params(which='minor',bottom=False,left=False)
    for ax in axes.flat:
        ax.set_xticks(x, BIN_LABELS, rotation=15)
        ax.set_xlabel('True event duration (s)')
        ax.set_xlim(-.6, len(x)-.4)
    fig.suptitle(f'{title} | Duration-stratified SAL errors', x=.075, ha='left', fontsize=19, fontweight='bold', y=.985)
    fig.text(.075, .94, f'{total["partial"]:,} partial recordings  |  {int(n.sum()):,} true events  |  {dt*1000:g} ms saved grid  |  fixed argmax decision', fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.17, left=.105, right=.98, hspace=.75, wspace=.42)
    fig.text(.075, .025, 'Panel b: 95% recording-bootstrap intervals; unmatched events include complete misses. Panels c/d: column percentages, color scale 0-100%.\n'
             'Boundary statistics condition on one-to-one tIoU >= 0.5 matches. Fully genuine / fully spoofed recordings are excluded here.\n'
             'Durations and boundaries use saved-grid labels (last cell clipped to audio); this is a retrospective analysis, not development-set evidence.', fontsize=9)
    save_figure(fig, out/'figure2_duration_and_boundaries')
    write_csv(out/'boundary_offset_distribution.csv',boundary_rows)


def plot_error_types(stats, counts, title, out):
    names = ['Complete misses / true events', 'Unmatched at tIoU 0.5 / true events',
             'Fragmented / true events', 'Merged / predicted events',
             'No-overlap false alarms / predicted events']
    numer = [stats['miss'], stats['truth']-stats['matched'], stats['fragment'], stats['merge'], stats['false_alarm']]
    denom = [stats['truth']]*3 + [stats['pred']]*2
    rates = [100*a/b if b else 0 for a,b in zip(numer,denom)]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.barh(np.arange(5), rates, color=[COLORS['purple'],COLORS['lavender'],COLORS['blue'],COLORS['navy'],COLORS['cyan']], height=.58,
            edgecolor=COLORS['outline'], linewidth=.9)
    for i,(a,b,r) in enumerate(zip(numer,denom,rates)):
        ax.text(r+.5, i, f'{r:.2f}%  ({a:,}/{b:,})', va='center', fontsize=10)
    ax.set_yticks(np.arange(5), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(12, max(rates)*1.40+8))
    ax.set_xlabel('Rate within the stated denominator (%)')
    ax.grid(axis='x', color=COLORS['grid'])
    ax.set_title(f'{title} | Error structure in partial recordings', loc='left', pad=20)
    fig.subplots_adjust(left=.36, right=.98, top=.85, bottom=.22)
    fig.text(.04, .055, 'Categories are not mutually exclusive and have different denominators; do not sum the bars.\n'
             'Fragment: one true event overlaps >=2 predicted events. Merge: one predicted event overlaps >=2 true events.\n'
             'A no-overlap false alarm is distinct from every unmatched prediction under a tIoU threshold.', fontsize=9)
    save_figure(fig, out/'figure2b_error_types')


def analyze(run, out, reps):
    out.mkdir()
    metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8-sig'))
    dt = metrics['resolution_seconds']
    title = 'PartialSpoof eval' if metrics['split']=='eval' else 'LlamaPartialSpoof crossfade'
    db = sqlite3.connect((run/'predictions.sqlite').as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    confusion = np.zeros((2,2), dtype=np.int64)
    examples = Examples()
    group = np.zeros((len(BIN_LABELS),4), dtype=np.int64)
    boundary = [[[],[]] for _ in BIN_LABELS]
    record_counts = []
    scopes = {kind:dict(recordings=0, truth=0, pred=0, matched=0, miss=0, fragment=0, merge=0,
                       false_alarm=0, false_positive_seconds=0., genuine_seconds=0., audio_seconds=0.)
              for kind in ['partial','genuine','full_spoof']}
    event_fields = ['uid','truth_index','start_s','end_s','duration_s','duration_bin','overlapping_predictions',
                    'covered_fraction','complete_miss','matched_tiou_0_5','matched_prediction_index','matched_tiou',
                    'start_error_ms','end_error_ms']
    event_file = (out/'partial_event_diagnostics.csv').open('w',encoding='utf-8-sig',newline='')
    event_writer=csv.DictWriter(event_file,fieldnames=event_fields)
    event_writer.writeheader()
    record_rows=[]
    nframes=0
    for index,(uid,raw_p,raw_y,seconds) in enumerate(db.execute('SELECT uid,logits,labels,seconds FROM predictions ORDER BY ordinal')):
        logits=np.frombuffer(raw_p,dtype=np.float32).reshape(-1,2)
        y=np.frombuffer(raw_y,dtype=np.uint8)
        assert len(y)==len(logits) and np.isfinite(logits).all() and np.isin(y,[0,1]).all(),uid
        pred_class=logits.argmax(axis=1)
        confusion += np.bincount(y*2+pred_class,minlength=4).reshape(2,2)
        nframes += len(y)
        truth=intervals(y==0,dt,seconds)
        pred=intervals(pred_class==0,dt,seconds)
        edges=np.minimum(np.arange(len(y)+1)*dt,seconds)
        widths=np.maximum(0,np.diff(edges))
        real_mask=widths>EPS
        kind='genuine' if np.all(y[real_mask]==1) else ('full_spoof' if np.all(y[real_mask]==0) else 'partial')
        ov,iou,matched=overlap_and_match(truth,pred)
        tg=(ov>EPS).sum(axis=1)
        pg=(ov>EPS).sum(axis=0)
        s=scopes[kind]
        increment=dict(recordings=1,truth=len(truth),pred=len(pred),matched=len(matched),miss=int((tg==0).sum()),
                       fragment=int((tg>=2).sum()),merge=int((pg>=2).sum()),false_alarm=int((pg==0).sum()),
                       false_positive_seconds=float(widths[(y==1)&(pred_class==0)].sum()),
                       genuine_seconds=float(widths[y==1].sum()),audio_seconds=float(widths.sum()))
        for key,value in increment.items(): s[key]+=value
        record_rows.append(dict(uid=uid,scope=kind,**increment))
        if kind=='partial':
            rc=np.zeros_like(group)
            for ti,(start,end) in enumerate(truth):
                duration=end-start; b=bin_index(duration)
                covered=float(ov[ti].sum()/duration)
                ismatched=ti in matched
                rc[b] += [1,int(tg[ti]==0),int(ismatched),int(tg[ti]>=2)]
                event=dict(uid=uid,truth_index=ti,start_s=start,end_s=end,duration_s=duration,duration_bin=BIN_LABELS[b],
                           overlapping_predictions=int(tg[ti]),covered_fraction=covered,complete_miss=int(tg[ti]==0),
                           matched_tiou_0_5=int(ismatched),matched_prediction_index='',matched_tiou='',start_error_ms='',end_error_ms='')
                if tg[ti]==0:
                    examples.add('miss',uid,start,end,duration,f'True [{start:.2f}, {end:.2f}] s; no predicted overlap')
                if tg[ti]>=2:
                    examples.add('fragment',uid,start,end,float(tg[ti]),f'{int(tg[ti])} predicted pieces overlap one true event')
                if ismatched:
                    pi=matched[ti]; ds=pred[pi,0]-start; de=pred[pi,1]-end
                    boundary[b][0].append(float(ds)); boundary[b][1].append(float(de))
                    event.update(matched_prediction_index=pi,matched_tiou=float(iou[ti,pi]),start_error_ms=ds*1000,end_error_ms=de*1000)
                    if tg[ti]==1 and pg[pi]==1:
                        cat=None
                        if ds>=-EPS and de<=EPS and (ds>EPS or de<-EPS):cat='shrink'
                        elif ds<=EPS and de>=-EPS and (ds<-EPS or de>EPS):cat='expand'
                        elif (ds>EPS and de>EPS) or (ds<-EPS and de<-EPS):cat='shift'
                        if cat:
                            examples.add(cat,uid,min(start,pred[pi,0]),max(end,pred[pi,1]),1-float(iou[ti,pi]),
                                f'Start error {ds*1000:+.0f} ms; end error {de*1000:+.0f} ms; tIoU={iou[ti,pi]:.2f}')
                        elif np.array_equal(y,pred_class):
                            examples.add('correct',uid,start,end,duration,'Whole recording correct on saved grid; shown event is exactly matched')
                event_writer.writerow(event)
            for pi,(start,end) in enumerate(pred):
                if pg[pi]==0:
                    examples.add('false_alarm',uid,start,end,end-start,f'Predicted [{start:.2f}, {end:.2f}] s; no true overlap')
                if pg[pi]>=2:
                    examples.add('merge',uid,start,end,float(pg[pi]),f'One prediction overlaps {int(pg[pi])} distinct true events')
            group += rc
            record_counts.append(rc)
        if (index+1)%15000==0:
            print(f'{title}: scanned {index+1:,} recordings',flush=True)
    event_file.close()
    assert nframes==metrics['frame_count']
    assert len(record_rows)==metrics['utterance_count']
    assert confusion.tolist()==metrics['confusion_true_rows_pred_columns_0_spoof_1_bonafide']
    assert group[:,0].sum()==scopes['partial']['truth']
    assert np.all(group[:,1]+group[:,2]<=group[:,0])
    boot=bootstrap_rates(record_counts,reps)
    rows=[]
    for i,label in enumerate(BIN_LABELS):
        n,miss,match,frag=group[i].tolist()
        r=dict(duration_bin=label,true_events=n,complete_misses=miss,overlap_without_match=n-miss-match,
               matched_tiou_0_5=match,fragmented_events=frag)
        for j,key in enumerate(['complete_miss','matched_recall','fragmentation']):
            r[key+'_rate']=float(group[i,j+1]/n) if n else None
            r[key+'_ci95_low']=float(np.nanpercentile(boot[:,i,j],2.5)) if n else None
            r[key+'_ci95_high']=float(np.nanpercentile(boot[:,i,j],97.5)) if n else None
        for j,key in enumerate(['start','end']):
            vals=np.array(boundary[i][j])*1000
            r[key+'_median_error_ms']=float(np.median(vals)) if len(vals) else None
            r[key+'_mean_abs_error_ms']=float(np.mean(abs(vals))) if len(vals) else None
            r[key+'_p95_abs_error_ms']=float(np.percentile(abs(vals),95)) if len(vals) else None
        rows.append(r)
    selected=examples.selected()
    counts={k:v['recordings'] for k,v in scopes.items()}
    write_csv(out/'duration_groups.csv',rows)
    write_csv(out/'recording_diagnostics.csv',record_rows)
    write_csv(out/'scope_summary.csv',[dict(scope=k,**v) for k,v in scopes.items()])
    save_json(out/'selected_examples.json',dict(selection='Smallest 256 stable SHA256 hashes per category, then near-median severity, prefer distinct recordings',
              eligible_counts=examples.counts,examples=selected))
    summary=dict(dataset=title,source_run=os.path.relpath(run, RESULT_ROOT),resolution_seconds=dt,scope=scopes,
        confusion_verified=confusion.tolist(),frames_verified=nframes,recordings_verified=len(record_rows),
        threshold='argmax on saved 2-class logits; class 0=spoof; ties choose spoof; equivalent spoof softmax >=0.5',
        matching='Maximum-cardinality one-to-one matching at tIoU >=0.5, then maximum summed tIoU',
        bins=BIN_LABELS,bootstrap_replicates=reps,bootstrap_unit='recording (related source variants may remain dependent)',
        limitations=['Retrospective evaluation-set diagnostics; not an independent validation of a subsequently designed method.',
          'Native saved-grid labels, not high-resolution annotation. PS 160 ms and LPS 20 ms are not directly comparable.',
          'Adjacent native-grid fake cells form one event; fine events or real gaps may be lost in coarse labels.',
          'Last grid cell clipped to audio duration. Unlabelled audio tail beyond saved labels excluded.',
          'No ASR, acoustic reinspection, candidate-operation efficacy, or causal mechanism is evaluated.',
          'Binary softmax scores are not calibrated probabilities.',
          'Fully real and fully fake recordings are reported separately from partial-recording event analyses.'],
        source_metrics=metrics,palette=COLORS)
    save_json(out/'summary.json',summary)
    print(f'{title}: verified saved confusion matrix; rendering figures',flush=True)
    plot_examples(db,selected,examples.counts,dt,title,out)
    plot_groups(rows,boundary,counts,dt,title,out,boot)
    plot_error_types(scopes['partial'],counts,title,out)
    db.close()
    return summary


def check_primitives():
    assert np.allclose(intervals(np.array([0,1,1,0],bool),.16,.55),[[.16,.48]])
    assert np.allclose(intervals(np.array([0,0,1],bool),.16,.4),[[.32,.4]])
    assert len(intervals(np.zeros(3,bool),.16,.4))==0
    truth=np.array([[0.,1.],[2.,3.]])
    ov,iou,m=overlap_and_match(truth,np.array([[0.,3.]]))
    assert (ov>0).sum()==2 and not m
    ov,iou,m=overlap_and_match(np.array([[0.,1.]]),np.array([[0.,.4],[.6,1.]]))
    assert (ov>0).sum()==2 and not m
    ov,iou,m=overlap_and_match(truth,truth.copy())
    assert m=={0:0,1:1}
    assert bin_index(.16)==0 and bin_index(.32)==1 and bin_index(.321)==2


def render_existing(output,reps):
    """Regenerate plots from the diagnostic CSVs after a style/layout change."""
    for out in [output/'PS_eval',output/'LPS_crossfade']:
        if not out.exists():continue
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        selected=json.loads((out/'selected_examples.json').read_text(encoding='utf-8'))
        with (out/'duration_groups.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            for key in ['true_events','complete_misses','matched_tiou_0_5']:
                row[key]=int(row[key])
        with (out/'recording_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:
            partial=[r['uid'] for r in csv.DictReader(f) if r['scope']=='partial']
        indices={uid:i for i,uid in enumerate(partial)}
        rc=np.zeros((len(partial),len(BIN_LABELS),4),dtype=np.int32)
        boundary=[[[],[]] for _ in BIN_LABELS]
        with (out/'partial_event_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                b=BIN_LABELS.index(row['duration_bin'])
                rc[indices[row['uid']],b]+=[1,int(row['complete_miss']),int(row['matched_tiou_0_5']),int(row['overlapping_predictions'])>=2]
                if int(row['matched_tiou_0_5']):
                    boundary[b][0].append(float(row['start_error_ms'])/1000)
                    boundary[b][1].append(float(row['end_error_ms'])/1000)
        boot=bootstrap_rates(rc,reps)
        counts={k:v['recordings'] for k,v in summary['scope'].items()}
        db=sqlite3.connect((source_run(summary['source_run'])/'predictions.sqlite').as_uri()+'?mode=ro',uri=True)
        plot_examples(db,selected['examples'],selected['eligible_counts'],summary['resolution_seconds'],summary['dataset'],out)
        plot_groups(rows,boundary,counts,summary['resolution_seconds'],summary['dataset'],out,boot)
        plot_error_types(summary['scope']['partial'],counts,summary['dataset'],out)
        db.close()
        print('Rendered '+out.name,flush=True)
    manifest=json.loads((output/'analysis_manifest.json').read_text(encoding='utf-8'))
    manifest['latest_render_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['latest_render_time']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    save_json(output/'analysis_manifest.json',manifest)
    write_readme(output)


def write_readme(output):
    lines=['SAL 图1与图2 诊断结果说明',
      '',
      '一、这次完成了什么',
      '只读已有两次 SAL 实验的 predictions.sqlite，未重新运行模型，未调整阈值，未修改原始结果。',
      '逐帧样本数、录音数和混淆矩阵与各自 metrics.json 完全一致。',
      '配色按用户提供的蓝紫参考图：真值青蓝，预测区间紫色，分数深蓝，文字灰色。',
      '图中使用英文以方便论文排版；本文档说明统计口径和中文解读。',
      '',
      '二、从哪里开始看',
      '每个数据集文件夹（PS_eval、LPS_crossfade）均有：',
      'figure1_timeline_gallery：8类时间轴案例总览。',
      'figure2_duration_and_boundaries：时长样本量、两种事件漏检定义、起止边界偏差分布。',
      'figure2b_error_types：漏检、碎片化、错误合并、无重叠误报的总体计数与比例。',
      '上述图均提供 PNG（查看）、PDF（矢量排版）、SVG（进一步编辑）。',
      'timeline_cases：总览图中每个案例的单独高清图，同样有三种格式。',
      '',
      '三、图1与六种操作的联系',
      'Complete miss：整段没有预测覆盖，对应研究插入操作。',
      'Spurious interval：预测区间与真值无重叠，对应研究删除操作。',
      'Contracted / Expanded / Shifted interval：区间收缩、外扩、平移，对应边界调整操作。',
      'Fragmented prediction：一个真实事件对应多个预测片段，对应研究合并操作。',
      'Merged true events：一个预测片段覆盖多个真实事件，对应研究拆分操作。',
      'Correct control：整条录音在保存网格上预测正确，对应保留操作。',
      '蓝紫横条分别为真实伪造区间和预测伪造区间；曲线是二分类 softmax 的伪造类分数。',
      '分数未经概率校准，不应解读为真实伪造概率。虚线0.5对应原始二分类argmax决策，未根据测试集选阈值。',
      '各类例子由固定哈希样本池内接近中位严重程度的案例选出，未挑选最严重错误。',
      '这些图说明错误形式存在，不证明相应操作必定有效；修改收益仍需后续实验。',
      '',
      '四、图2怎样读',
      'a：各时长组真实伪造事件数，统计单位是事件，不是帧或录音。',
      'b紫线：完全漏检率，即真实事件与所有预测区间均没有正时长重叠。',
      'b蓝线：没有获得一对一tIoU>=0.5匹配的事件比例，包含完全漏检。',
      'tIoU = 两区间交集时长 / 并集时长。匹配先最大化合格匹配数量，再最大化tIoU之和。',
      '误差线是400次按录音重采样的95%区间；同源变体可能相关，因此不作为严格源独立显著性证明。',
      'c/d：只统计匹配成功的片段，分别显示起点/终点的提前、网格完全一致、延后比例。',
      '每列总和约100%（存在显示舍入），n为该组匹配事件数。深色表示该类偏差占比高。',
      'Exact on saved grid只表示与该分辨率标签一致，不等于精确到原始音频的真实编辑边界。',
      '完全漏检、未匹配、碎片化等可能重叠，图2b的分母也不同，不能把它们相加。',
      '',
      '五、当前结果']
    for folder in ['PS_eval','LPS_crossfade']:
        p=output/folder
        if not p.exists():continue
        s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        v=s['scope']['partial']
        with (p/'duration_groups.csv').open(encoding='utf-8-sig',newline='') as f:
            groups=list(csv.DictReader(f))
        lines.extend(['',f'{s["dataset"]}：',f'全部评估录音 {s["recordings_verified"]:,} 条；保存网格 {s["resolution_seconds"]*1000:g} ms。',
          f'局部伪造录音 {v["recordings"]:,} 条，真实事件 {v["truth"]:,} 个，预测事件 {v["pred"]:,} 个。',
          f'完全漏检 {v["miss"]:,}/{v["truth"]:,} = {100*v["miss"]/v["truth"]:.2f}%。',
          f'tIoU>=0.5的一对一匹配召回 {v["matched"]:,}/{v["truth"]:,} = {100*v["matched"]/v["truth"]:.2f}%。',
          f'碎片化真实事件 {v["fragment"]:,}/{v["truth"]:,} = {100*v["fragment"]/v["truth"]:.2f}%。',
          f'合并多个真实事件的预测 {v["merge"]:,}/{v["pred"]:,} = {100*v["merge"]/v["pred"]:.2f}%。',
          f'无重叠误报预测 {v["false_alarm"]:,}/{v["pred"]:,} = {100*v["false_alarm"]/v["pred"]:.2f}%。',
          '各时长组完全漏检率：'+ '；'.join(f'{r["duration_bin"]} s：{100*float(r["complete_miss_rate"]):.2f}%（n={int(r["true_events"]):,}）' for r in groups if r['true_events']!='0')])
    lines.extend(['',
      'PS：最短网格事件的完全漏检更明显，但不能据此证明是时长本身造成；需排查攻击类型等因素。',
      'PS：匹配成功事件大多在网格上边界重合，不宜据当前结果宣称普遍边界偏移。',
      'LPS：完全无覆盖的漏检少，但一对一匹配召回低、碎片化和误报较多，说明“碰到一点伪造区间”不等于定位成功。',
      '因此，复检后覆盖率提高不能单独证明有效，也可能只是扩大了预测范围，必须同时看误报和匹配质量。',
      '',
      '六、限制与论文表述',
      '1. 这两次已有运行是PS eval和LPS crossfade评估结果，属于事后诊断。不能写成改进前在独立开发集上发现的证据。',
      '2. 主图只统计保存标签网格上有真有假的局部伪造录音；全真、全假录音另列scope_summary.csv。',
      '3. 图中真值由保存的0/1帧标签重建，没有读取更高分辨率标签。PS网格160ms，LPS网格20ms，不能直接比较边界精度。',
      '4. 较粗标签可能合并很近的真实片段或抹去短间隔。≤160ms组主要是PS单网格事件，并非精细标注下的全部短伪造事件。',
      '5. 末尾网格裁切到实际音频时长；标签结束后未标注的极短尾部不纳入事件统计。原始帧混淆矩阵仍按原评估完整核对。',
      '6. 没有做局部复检、ASR或六类操作收益实验，因此没有关于它们有效性的结果。',
      '7. 图1是示例，图2才用于判断普遍程度；保持两者角色区分。',
      '',
      '可用于报告的保守表述：',
      '“对已保存的SAL测试预测进行事后诊断，在原生评价网格上观察到整段漏检、误报、边界偏差、碎片化和错误合并等现象。',
      '这些现象为构造插入、删除、边界调整、合并、拆分及保留候选操作提供了经验动机。候选操作及复检的实际效用仍需独立实验验证。”',
      '',
      '七、数据与复现',
      'duration_groups.csv：每个时长组的数量、三种率、置信区间和边界误差统计。',
      'partial_event_diagnostics.csv：每个真实事件的标识、区间、覆盖情况、匹配和边界误差。',
      'recording_diagnostics.csv：每条录音的类别和计数。',
      'boundary_offset_distribution.csv：图2热力图的精确百分比。',
      'selected_examples.json：每张案例图的uid、区间、选取规则与合格案例数。',
      'summary.json / analysis_manifest.json：来源、口径、验证结果和配色。',
      '',
      '重新生成（自动创建新的时间戳输出文件夹）：',
      '在项目根目录运行：python visualization/diagnose_saved_predictions.py',
      '',
      '只更新现有图的样式（读取已有诊断CSV）：',
      f'python visualization/diagnose_saved_predictions.py --render-existing "{os.path.relpath(output, PROJECT_ROOT)}"',
      ''])
    (output/'阅读说明.txt').write_text('\n'.join(lines),encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,help='Completed SAL run directories; default: latest complete eval and lps')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bootstrap',type=int,default=400)
    parser.add_argument('--render-existing',type=Path,help='Regenerate existing diagnostics using current plot style')
    args=parser.parse_args()
    if args.bootstrap<100: parser.error('--bootstrap must be >=100')
    if args.render_existing:
        check_primitives()
        figure_style()
        render_existing(project_path(args.render_existing),args.bootstrap)
        return
    runs=[project_path(p) for p in args.runs] if args.runs else None
    if not runs:
        runs=[]
        for split in ['eval','lps']:
            eligible=[]
            for path in RESULT_ROOT.glob('sal_wavlm_*'):
                if path.is_dir() and (path/'metrics.json').exists() and (path/'status.json').exists():
                    m=json.loads((path/'metrics.json').read_text(encoding='utf-8-sig'))
                    status=json.loads((path/'status.json').read_text(encoding='utf-8-sig'))
                    if m['split']==split and not m['smoke_test'] and status.get('status')=='complete':eligible.append(path)
            if not eligible:raise RuntimeError(f'No complete SAL run for {split}')
            runs.append(sorted(eligible)[-1])
    output=output_path(args.output) if args.output else OUTPUT_ROOT/f'sal_diagnostics_{datetime.now(ZoneInfo("Asia/Shanghai")):%Y%m%d_%H%M%S}'
    output.mkdir(parents=True,exist_ok=False)
    check_primitives()
    figure_style()
    summaries=[]
    for run in runs:
        metrics=json.loads((run/'metrics.json').read_text(encoding='utf-8-sig'))
        summaries.append(analyze(run,output/('PS_eval' if metrics['split']=='eval' else 'LPS_crossfade'),args.bootstrap))
    save_json(output/'analysis_manifest.json',dict(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        script=os.path.relpath(Path(__file__).resolve(),PROJECT_ROOT),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_runs=[os.path.relpath(x,RESULT_ROOT) for x in runs],palette=COLORS,bootstrap_replicates=args.bootstrap,
        source_files_modified=False,model_inference=False))
    write_readme(output)
    print('OUTPUT_DIRECTORY='+str(output),flush=True)
    for s in summaries:print(json.dumps(dict(dataset=s['dataset'],scope=s['scope']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
