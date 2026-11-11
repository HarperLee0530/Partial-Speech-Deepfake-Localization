from pathlib import Path
import sys, json, csv, sqlite3, argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

from visualization_paths import analysis_directory, project_path, output_path, source_run, PROJECT_ROOT, safe_relative_path
parser=argparse.ArgumentParser(description='Export three Word-ready figures from existing diagnostic results.')
parser.add_argument('--analysis-dir',help='Diagnostics directory, relative to project root; default: latest completed diagnostics')
parser.add_argument('--output-dir',help='Figure destination, relative to project root')
parser.add_argument('--latest',action='store_true',help='Export latest diagnostics of current reduced-trained PS/LPS runs')
args=parser.parse_args()
if not args.analysis_dir and not args.latest:
    parser.print_help()
    raise SystemExit(0)
ROOT=analysis_directory(args.analysis_dir)
OUT=output_path(args.output_dir) if args.output_dir else output_path(ROOT/'导师审阅'/'figures')
OUT.mkdir(parents=True,exist_ok=True)
import diagnose_saved_predictions as source
C=source.COLORS
plt.rcParams.update({'font.family':'Times New Roman','font.size':7.5,'text.color':C['ink'],
 'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],
 'axes.edgecolor':C['ink'],'axes.linewidth':.75,
 'axes.spines.top':True,'axes.spines.right':True,'axes.spines.bottom':True,'axes.spines.left':True,
 'axes.axisbelow':True,'axes.titlesize':8,'axes.titleweight':'bold','axes.facecolor':C['background'],
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name,extra_top=0):
    for ext in ['png','pdf','svg']:
        bounds=Bbox.from_extents(0,0,*[float(fig.get_figwidth()),float(fig.get_figheight())+extra_top]) if extra_top else None
        fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white',bbox_inches=bounds)
    plt.close(fig)

ps=ROOT/'PS_eval'
selection=json.loads((ps/'selected_examples.json').read_text(encoding='utf-8'))
summary=json.loads((ps/'summary.json').read_text(encoding='utf-8'))
dt=float(summary['resolution_seconds'])
scope_title=summary['dataset']
db=sqlite3.connect((source_run(summary['source_run'])/'predictions.sqlite').as_uri()+'?mode=ro',uri=True)
fig,axes=plt.subplots(4,2,figsize=(6.69,6.8))
for i,(cat,ax) in enumerate(zip(source.CATEGORIES,axes.flat)):
    if cat in selection['examples']:
        source.timeline(ax,db,selection['examples'][cat],dt,scope_title,short=True)
    else:
        ax.text(.5,.5,'No eligible example in this run',transform=ax.transAxes,ha='center',va='center')
    if cat in selection['examples']:
        for txt in list(ax.texts):txt.remove()
    ax.set_title(f'({chr(97+i)}) {source.TITLES[cat]}',loc='left',fontsize=8,pad=6)
    ax.set_xlabel('Time (s)',fontsize=7,labelpad=2)
    ax.set_ylabel('Spoof score',fontsize=9,labelpad=3)
    ax.tick_params(labelsize=6.8,pad=2)
    ax.tick_params(axis='y',labelsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
legend=fig.legend(handles=[Patch(color=C['cyan'],label='True spoof interval'),Patch(color=C['purple'],label='SAL interval'),
 Line2D([0],[0],color=C['navy'],label='Spoof score')],ncol=3,frameon=True,loc='upper center',bbox_to_anchor=(.53,1.017),
 fontsize=7.8,edgecolor='#6B6B6B',facecolor='white',framealpha=1,borderpad=.5)
legend.get_frame().set_linestyle('--')
legend.get_frame().set_linewidth(.9)
fig.subplots_adjust(left=.105,right=.985,bottom=.065,top=.93,hspace=.78,wspace=.34)
fig.text(.105,.008,scope_title,fontsize=7)
save(fig,'图1_PS时间轴诊断',extra_top=.16)
db.close()

with (ps/'duration_groups.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
with (ps/'boundary_offset_distribution.csv').open(encoding='utf-8-sig',newline='') as f: boundary=list(csv.DictReader(f))
labels=['≤0.16','0.16–\n0.32','0.32–\n0.64','0.64–\n1.28','1.28–\n2.56','>2.56']
x=np.arange(6)
fig,axes=plt.subplots(2,2,figsize=(6.69,5.55))
ax=axes[0,0]
n=[int(r['true_events']) for r in rows]
bars=ax.bar(x,n,color=C['blue'],width=.65,edgecolor=C['outline'],linewidth=.9)
ax.bar_label(bars,labels=[f'{v:,}' for v in n],fontsize=6.3,padding=3)
ax.set_ylim(0,max(1,max(n)*1.20))
ax.set_title('(a) True-event duration',loc='left')
ax.set_ylabel('Number of events',fontsize=8.5)
ax.grid(axis='y',color=C['grid'])
if max(n)>=1000:
    ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3))
else:
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax=axes[0,1]
def number(value):
    return float(value) if value not in ('',None) else np.nan
for j,(key,col,label) in enumerate([('complete_miss',C['purple'],'Complete miss'),('matched_recall',C['blue'],'Unmatched at tIoU 0.5')]):
    vals=np.array([number(r[key+'_rate']) for r in rows])*100
    low=np.array([number(r[key+'_ci95_low']) for r in rows])*100
    high=np.array([number(r[key+'_ci95_high']) for r in rows])*100
    if j: vals,low,high=100-vals,100-high,100-low
    xx=x+(-.06 if j==0 else .06)
    ax.errorbar(xx,vals,yerr=[np.maximum(0,vals-low),np.maximum(0,high-vals)],fmt='o-' if j==0 else 's--',color=col,lw=1,
                markersize=3,capsize=2,label=label)
    for k,v in enumerate(vals):
        if not np.isfinite(v):continue
        ax.annotate(f'{v:.2f}',(xx[k],v),xytext=(0,7 if j or v<=5 else -12),textcoords='offset points',ha='center',fontsize=6.1,color=col)
ax.set_title('(b) Event miss rates',loc='left')
ax.set_ylabel('True events (%)',fontsize=8.5)
ax.set_ylim(-1.1,101)
ax.grid(axis='y',color=C['grid'])
ax.legend(fontsize=6.2,frameon=False,loc='upper right')
for j,name in enumerate(['start','end']):
    ax=axes[1,j]
    matrix=np.zeros((5,6))
    matched_counts=[]
    for b,group in enumerate(source.BIN_LABELS):
        these=[r for r in boundary if r['boundary']==name and r['duration_bin']==group]
        matched_counts.append(int(these[0]['matched_events']) if these else 0)
        matrix[:,b]=[float(r['percentage']) for r in these]
    cmap=LinearSegmentedColormap.from_list(name,['#FFFFFF',C['light'],C['blue']] if j==0 else ['#FFFFFF',C['lavender'],C['purple']])
    ax.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for cell_row in range(5):
        for cell_col in range(6):
            ax.add_patch(Rectangle((cell_col-.5,cell_row-.5),1,1,fill=False,
                                   edgecolor=C['outline'],linewidth=.65,zorder=3))
    for a in range(5):
        for b in range(6):ax.text(b,a,f'{matrix[a,b]:.1f}' if matched_counts[b] else 'n/a',ha='center',va='center',fontsize=6.4,color='white' if matrix[a,b]>70 else C['ink'])
    grid=f'{dt*1000:g}'
    ax.set_yticks(range(5),[f'Early >{grid}',f'Early ≤{grid}','Exact',f'Late ≤{grid}',f'Late >{grid}'],fontsize=7.5)
    ax.set_title(f'({"c" if j==0 else "d"}) {name.capitalize()} boundary offset',loc='left',pad=7)
    ax.set_ylabel('Offset (ms)',fontsize=8.5)
    ax.set_xticks(np.arange(-.5,6,1),minor=True)
    ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',visible=False)
    ax.tick_params(which='minor',bottom=False,left=False)
for ax in axes.flat:
    ax.set_xticks(x,labels,fontsize=6.5)
    ax.set_xlim(-.55,5.55)
    ax.set_xlabel('True-event duration (s)',fontsize=7,labelpad=3)
    ax.tick_params(axis='y',labelsize=7.5)
fig.subplots_adjust(left=.125,right=.985,top=.95,bottom=.11,hspace=.48,wspace=.43)
fig.text(.125,.012,scope_title,fontsize=7)
save(fig,'图2_PS时长与边界统计')

lps_summary=json.loads((ROOT/'LPS_crossfade'/'summary.json').read_text(encoding='utf-8'))
lps=lps_summary['scope']['partial']
numer=[lps['miss'],lps['truth']-lps['matched'],lps['fragment'],lps['merge'],lps['false_alarm']]
denom=[lps['truth']]*3+[lps['pred']]*2
names=['Complete miss / true events','Unmatched / true events','Fragmented / true events','Merged / predicted events','No-overlap FA / predicted events']
rates=np.divide(numer,denom,out=np.zeros(5),where=np.asarray(denom)>0)*100
fig,ax=plt.subplots(figsize=(6.69,3.05))
ax.barh(range(5),rates,color=[C['purple'],C['lavender'],C['blue'],C['navy'],C['cyan']],height=.58,
        edgecolor=C['outline'],linewidth=.9)
for i,(r,a,b) in enumerate(zip(rates,numer,denom)):
    ax.text(r+.9,i,f'{r:.2f}% ({a:,}/{b:,})',va='center',fontsize=6.7)
ax.set_yticks(range(5),names,fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0,111)
ax.set_xlabel('Rate within stated denominator (%)',fontsize=7.5)
ax.grid(axis='x',color=C['grid'])
ax.tick_params(axis='x',labelsize=7)
fig.subplots_adjust(left=.37,right=.985,bottom=.22,top=.96)
fig.text(.02,.02,lps_summary['dataset'],fontsize=7)
save(fig,'图3_LPS错误类型统计')
(OUT/'来源与配色.json').write_text(json.dumps(dict(source=safe_relative_path(ROOT,PROJECT_ROOT),palette=C,
  ps_scope=scope_title,lps_scope=lps_summary['dataset'],ps_resolution_seconds=dt,lps_resolution_seconds=lps_summary['resolution_seconds'],
  notes='Original diagnostic data, bins, decisions and examples unchanged. Figures reflowed for Word printing.',
  figure1_examples=selection['examples']),ensure_ascii=False,indent=2),encoding='utf-8')
print(str(OUT))
