"""Generate the academic workflow figure from the recorded MSF-HSAT run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
split = json.loads((ROOT / 'splits/main.json').read_text())
manifest = json.loads((ROOT / 'prepared/nsclc/manifest.json').read_text())
run = json.loads((ROOT / 'runs/hybrid_compact_v2_seed42/resolved_config.json').read_text())
protocol = json.loads((ROOT / 'evaluations/hybrid_test/protocol.json').read_text())
counts = {k:len(v) for k,v in split['partitions'].items()}
PINK, BLUE, ORANGE = '#F8D7DA', '#D6E6F5', '#FBE3C8'
PURPLE, GREEN, BLUE2, YELLOW = '#E3D9F0', '#D8EBD5', '#CFE2F3', '#FFF8D6'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
                     'mathtext.fontset':'stix','svg.fonttype':'none','font.size':13,
                     'axes.linewidth':1,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(20,11.4))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,1800),ylim=(1000,0)); ax.axis('off')

def text(x,y,s,size=13,weight='normal',ha='center',va='center',style='normal'):
    return ax.text(x,y,s,fontsize=size,ha=ha,va=va,fontweight=weight,fontstyle=style,linespacing=1.4)

def box(x,y,w,h,s='',color=BLUE,size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',
                  facecolor=color,edgecolor='black',linewidth=1))
    if s: text(x+w/2,y+h/2,s,size)

def arrow(points,dashed=False,head=True):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color='black',lw=1,ls=(0,(5,4)) if dashed else '-')
    a,b=points[-2:]
    if head:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,
                     lw=1,color='black',linestyle=(0,(5,4)) if dashed else '-',shrinkA=0,shrinkB=0))
    else: ax.plot([a[0],b[0]],[a[1],b[1]],color='black',lw=1,ls=(0,(5,4)) if dashed else '-')

def stage(x,num,title):
    ax.add_patch(Rectangle((x,190),256,238,fill=False,edgecolor='black',linewidth=1,linestyle=(0,(5,4))))
    text(x+14,211,f'{num}. {title}',13,weight='bold',ha='left')

def panel(x,title,w=410):
    ax.add_patch(FancyBboxPatch((x,535),w,380,boxstyle='round,pad=0,rounding_size=8',
                    facecolor='white',edgecolor='black',linewidth=1))
    text(x+18,559,title,14,weight='bold',ha='left')

text(900,53,'MSF-HSAT: Methodology for 3D Primary Lung-Tumor Segmentation',23,weight='bold')
text(900,91,'NSCLC-Radiomics CT  ·  expert-defined GTV-1  ·  current compact implementation',14)
text(40,147,'End-to-end experimental workflow',15,weight='bold',ha='left')

xs=[40,332,624,916,1208,1500]
titles=['Dataset and audit','Volume preparation','Patient partition','Learning and selection','Frozen inference','Evaluation and output']
labels=[
    'CT + RTSTRUCT\nor paired CT / mask',
    'Geometry-aware\nHU preprocessing',
    'Grouped train /\nvalidation / test split',
    'MSF-HSAT training\n+ validation selection',
    'Full-volume\nsliding-window prediction',
    'Native-grid metrics\n+ segmentation export',
]
notes=[
    '413 eligible cases\n9 exclusions recorded',
    'LPS; 2.5 × 1 × 1 mm\nLung + soft-tissue windows',
    f"{counts['train']} / {counts['val']} / {counts['test']} cases\nSeed 42; fixed identities",
    'Best validation Dice\nFreeze checkpoint + threshold',
    '50% overlap; Gaussian blend\nNative physical geometry',
    'Probability map + binary mask\nPatient-level report and figures',
]
colors=[PINK,PINK,ORANGE,BLUE,GREEN,GREEN]
for i,x in enumerate(xs):
    stage(x,i+1,titles[i])
    box(x+15,251,226,102,labels[i],colors[i],14)
    text(x+128,391,notes[i],12)
    if i<5: arrow([(x+241,302),(xs[i+1]+15,302)])

# Paired expansion guides link selected overview stages to the detailed panels.
for a,b in [((347,428),(40,535)),((609,428),(450,535)),
            ((931,428),(475,535)),((1172,428),(885,535)),
            ((1223,428),(910,535)),((1464,428),(1320,535))]:
    arrow([a,b],dashed=True,head=False)

panel(40,'(a) Data preparation and cohort control')
box(60,590,370,59,'Check CT / target alignment, HU and slice geometry',PINK,12.5)
arrow([(245,649),(245,669)])
box(60,669,370,59,'Orient to LPS; resample CT and binary GTV-1',PINK,12.5)
arrow([(245,728),(245,748)])
box(60,748,370,59,'Lung [−1000, 400] HU + soft [−160, 240] HU',PINK,12.5)
arrow([(245,807),(245,827)])
box(60,827,370,59,'Freeze patient split; fit acquisition statistics on train',ORANGE,12.3)

panel(475,'(b) Optimization and validation selection')
box(495,590,370,59,'Train: 32 × 96 × 96 patches; 25% foreground sampling',BLUE,12.3)
arrow([(680,649),(680,669)])
box(495,669,370,59,'Spatial / HU augmentation; AdamW optimization',BLUE,12.5)
arrow([(680,728),(680,748)])
box(495,748,370,59,r'$\mathcal{L}=\mathcal{L}_{seg}+0.5\mathcal{L}_{aux1}+0.25\mathcal{L}_{aux2}+0.1\mathcal{L}_{bnd}$',PURPLE,14)
arrow([(680,807),(680,827)])
box(495,827,370,59,'Validation: select best checkpoint; freeze τ = 0.98',ORANGE,12.5)

panel(910,'(c) Inference, reconstruction and assessment')
box(930,590,370,59,'Predict patch logits with the frozen checkpoint',GREEN,12.5)
arrow([(1115,649),(1115,669)])
box(930,669,370,59,'Gaussian-weighted logit blending → sigmoid',GREEN,12.5)
arrow([(1115,728),(1115,748)])
box(930,748,370,59,'Resample probability to native grid; threshold at 0.98',GREEN,12.1)
arrow([(1115,807),(1115,827)])
box(930,827,370,59,'Dice / IoU; HD95 / ASSD; surface Dice at 2 mm',GREEN,12.3)

box(1345,535,410,380,color=YELLOW)
text(1363,559,'(d) Legend and recorded protocol',14,weight='bold',ha='left')
box(1368,591,48,24,color=BLUE)
text(1434,603,'Processing operation',12.5,ha='left')
arrow([(1368,641),(1416,641)])
text(1434,641,'Data or model flow',12.5,ha='left')
ax.add_patch(Rectangle((1368,668),48,24,fill=False,edgecolor='black',linewidth=1,linestyle=(0,(5,4))))
text(1434,680,'Workflow stage',12.5,ha='left')
arrow([(1368,718),(1416,708)],dashed=True,head=False)
text(1434,718,'Expanded process view',12.5,ha='left')
text(1368,765,'Patch dimensions: D × H × W',12.5,ha='left')
text(1368,800,'Segmentation loss: soft Dice + BCE',12.5,ha='left')
text(1368,835,'Validation only: checkpoint / threshold selection',12,ha='left')
text(1368,870,'Locked test: 83 cases; no postprocessing',12.5,ha='left')

text(40,949,'Train / validation / test identities remain grouped by patient; test references are used for final assessment only.',12.5,ha='left')
text(40,978,'Source: lung_ct_segmentation.py; recorded compact run, preparation manifest, fixed split and test protocol.',11.5,ha='left')

png=OUT/'msf_hsat_methodology.png'
svg=OUT/'msf_hsat_methodology.svg'
fig.savefig(png,dpi=300)
fig.savefig(svg)
plt.close(fig)
print(png)
print(svg)
