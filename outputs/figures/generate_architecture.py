"""Academic architecture figure derived from the current single-file MSF-HSAT model."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CFG = json.loads((ROOT / 'runs/hybrid_compact_v2_seed42/resolved_config.json').read_text())
assert CFG['model_config']['widths'] == [16, 32, 64, 128, 192]
assert CFG['patch'] == [32, 96, 96]
plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','Times','DejaVu Serif'],
                     'mathtext.fontset':'stix', 'svg.fonttype':'none', 'font.size':11})
PINK='#F8D7DA'; BLUE='#D6E6F5'; ORANGE='#FBE3C8'; PURPLE='#E3D9F0'
GREEN='#D8EBD5'; BLUE2='#CFE2F3'; YELLOW='#FFF8D6'
W,H=2200,1420
fig,ax=plt.subplots(figsize=(22,14.2))
fig.subplots_adjust(0,0,1,1); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')

def txt(x,y,s,size=11,weight='normal',ha='center',italic=False,**kw):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va='center',
                   fontstyle='italic' if italic else 'normal',zorder=6,**kw)
def block(x,y,w,h,lines,fill=BLUE,size=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',
                              facecolor=fill,edgecolor='black',lw=1,zorder=3))
    if isinstance(lines,str): lines=lines.split('\n')
    step=21
    for i,s in enumerate(lines): txt(x+w/2,y+h/2+(i-(len(lines)-1)/2)*step,s,size)
def line(points,dash=False,arrow=True):
    for i,(a,b) in enumerate(zip(points,points[1:])):
        if arrow and i==len(points)-2:
            ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,
                          color='black',linestyle=(0,(5,4)) if dash else '-',zorder=2,
                          shrinkA=0,shrinkB=0))
        else: ax.plot([a[0],b[0]],[a[1],b[1]],color='black',lw=1,
                       linestyle=(0,(5,4)) if dash else '-',zorder=1)
def stage(x,y,w,h,title):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1,ls=(0,(5,4)),zorder=0))
    txt(x+12,y+20,title,10.5,'bold',ha='left')
def panel(x,y,w,h,title,fill='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',
                              facecolor=fill,edgecolor='black',lw=1,zorder=0))
    txt(x+16,y+26,title,12.5,'bold',ha='left')
def circ(x,y,s='C',r=12):
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor='black',lw=1,zorder=4))
    txt(x,y,s,10)
def cube(x,y,w=72,h=66):
    d=14
    for pts,c in [([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],PINK),
                  ([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],PINK),
                  ([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],ORANGE)]:
        ax.add_patch(Polygon(pts,facecolor=c,edgecolor='black',lw=1,zorder=3))

txt(1100,35,'MSF-HSAT: complete 3D lung-tumor segmentation architecture',21,'bold')
txt(1100,70,'Current hybrid3d implementation  |  compact configuration  |  primary gross tumor volume (GTV-1)',12)

# Main pipeline, in one left-to-right row. Shapes exclude the batch dimension.
stage(200,205,855,220,'Residual encoder')
stage(1073,205,172,220,'Global context')
stage(1264,205,705,220,'Adaptive decoder')
cube(52,282,78,78)
txt(95,245,r'$2\times32\times96\times96$',10)
txt(95,386,'Two-window CT',11,'bold')
txt(95,408,'lung + soft tissue',9.5)

nodes=[
 (220,125,'E0','Residual stem','16 | 32×96×96',PINK),
 (380,125,'E1','Residual stage','32 | 32×48×48',BLUE),
 (540,145,'E2','Residual + MSR','64 | 16×24×24',BLUE),
 (720,145,'E3','Residual + MSR','128 | 8×12×12',BLUE),
 (900,135,'E4','Residual stage','192 | 4×6×6',BLUE),
 (1090,140,'Transformer','2 blocks; 8 heads','192 | 4×6×6',PURPLE),
 (1280,140,'D3 fusion','u3, s3, E2','128 | 8×12×12',GREEN),
 (1460,140,'D2 fusion','u2, s2, E1','64 | 16×24×24',GREEN),
 (1640,140,'D1 fusion','D2, E1, E0','32 | 32×48×48',GREEN),
 (1820,130,'D0 refine','Concat + Conv','16 | 32×96×96',BLUE),
 (2000,165,'Prediction head','1×1×1 Conv','1 | 32×96×96',PINK)]
for x,w,a,b,shape,color in nodes:
    block(x,275,w,112,[a,b],color,10.5)
    txt(x+w/2,254,shape,9)
for (x,w,*_),n in zip(nodes,nodes[1:]): line([(x+w,331),(n[0],331)])
line([(144,321),(220,321)])
for x,s in [(362,'1,2,2'),(522,'2,2,2'),(702,'2,2,2'),(882,'2,2,2')]:
    txt(x,355,s,8)
txt(1160,404,'144 tokens; 3D sin/cos PE',8.5)

# Encoder bypasses into the three-source adaptive fusions and final concatenation.
line([(612,275),(612,174),(1320,174),(1320,275)])
txt(1010,165,'E2',9)
line([(443,275),(443,148),(1490,148),(1490,275)])
line([(1300,148),(1690,148),(1690,275)])
txt(1150,139,'E1',9)
line([(282,275),(282,122),(1865,122),(1865,275)])
line([(1590,122),(1740,122),(1740,275)])
txt(1235,113,'E0',9)

# The two skip modules are separately conditioned by decoder features and metadata.
block(1280,465,140,95,['Wavelet skip','s3 = W(E3,u3,q3)'],ORANGE,9.5)
block(1460,465,140,95,['Wavelet skip','s2 = W(E2,u2,q2)'],ORANGE,9.5)
line([(792,387),(792,435),(1250,435),(1250,512),(1280,512)])
line([(612,387),(612,582),(1530,582),(1530,560)])
line([(1370,465),(1370,387)])
line([(1550,465),(1550,387)])
line([(1260,331),(1260,445),(1310,445),(1310,465)])
txt(1245,451,'u3',9)
line([(1440,331),(1440,445),(1490,445),(1490,465)])
txt(1430,451,'u2',9)
block(900,465,292,72,['Acquisition descriptors q2, q3','9 values each; training-normalized'],PURPLE,10)
line([(1192,484),(1210,484),(1210,452),(1395,452),(1395,465)])
line([(1192,521),(1228,521),(1228,571),(1578,571),(1578,560)])
txt(1046,558,'native spacing, resampling ratio, stage spacing',8.5)

# Output and auxiliary supervision. Heads do not feed back into decoder features.
block(2000,453,165,100,['Sigmoid probability','threshold → mask','GTV-1 segmentation'],GREEN,10)
line([(2082,387),(2082,453)])
block(1650,470,260,78,['Auxiliary heads (training)','D1 + D2: segmentation','D0: boundary prediction'],PURPLE,9.5)
line([(1710,387),(1710,470)])
line([(1600,331),(1620,331),(1620,449),(1755,449),(1755,470)])
line([(1885,387),(1885,449),(1860,449),(1860,470)])
txt(1780,574,'1×1×1 Conv per head; auxiliary logits resized for loss',8.5)

# Zoom cones lie in their own gutter, with no arrows crossing detail panels.
for pts in [[(540,425),(35,651)],[(685,425),(475,651)],
            [(1280,560),(505,651)],[(1600,560),(1175,651)]]:
    line(pts,dash=True,arrow=False)
panel(35,651,440,650,'(a) Multi-scale residual block (MSR)')
panel(505,651,670,650,'(b) Conditioned 3D wavelet skip — E2 and E3')
panel(1205,651,530,650,'(c) Adaptive multi-source fusion')
panel(1765,651,400,650,'(d) Notation and supervision',YELLOW)

# MSR: three explicitly parallel receptive fields, concatenation, residual add.
txt(255,706,r'Input $x$: $C$ channels',11)
block(140,743,230,62,['1×1×1 projection',r'$C\rightarrow C/2$'],BLUE,10.5)
line([(255,721),(255,743)])
centers=[110,255,400]
for cx,d,col in zip(centers,[1,2,3],[PURPLE,GREEN,BLUE2]):
    block(cx-59,852,118,105,['DW Conv 3³',f'dilation = {d}','GN + SiLU'],col,10)
    line([(255,805),(255,828),(cx,828),(cx,852)])
    line([(cx,957),(cx,985),(255,985),(255,1004)])
circ(255,1016)
block(140,1055,230,62,['1×1×1 projection + GN',r'$3C/2\rightarrow C$'],BLUE,10)
line([(255,1028),(255,1055)])
circ(255,1163,'+')
line([(255,1117),(255,1151)])
line([(255,730),(66,730),(66,1163),(243,1163)])
block(160,1200,190,54,'SiLU → output',BLUE,11)
line([(255,1175),(255,1200)])

# Wavelet decomposition; low band attention and seven individually gated details.
txt(840,706,r'Encoder feature $e_i$  +  decoder context $u_i$  +  metadata $q_i$',11)
block(728,741,225,60,'3D Haar analysis',ORANGE,11)
line([(840,722),(840,741)])
block(540,847,225,112,['LLL (low-frequency)','3D window attention','4 heads; window 4×6×6'],PURPLE,10.5)
block(810,847,320,112,['7 detail subbands',r'$g_b\odot[H_b+\Delta_b(H_b)]$','DW Conv → GN → SiLU → 1×1'],GREEN,10.5)
line([(840,801),(840,822),(652,822),(652,847)])
line([(840,822),(970,822),(970,847)])
txt(840,991,r'$g_b=\sigma\{\mathrm{MLP}([\,|H_b|,\;p(u_i),\;h(q_i)\,])\}$',13)
block(728,1040,225,60,'Inverse 3D Haar',ORANGE,11)
line([(652,959),(652,1020),(800,1020),(800,1040)])
line([(970,959),(970,1020),(880,1020),(880,1040)])
block(635,1143,410,72,['Residual correction',r'$s_i=e_i+\kappa\,P(\hat e_i-e_i)$'],BLUE,12)
line([(840,1100),(840,1143)])
txt(840,1250,r'$P$: 1×1×1 Conv;  $\kappa=\sigma(k)$;  $i\in\{2,3\}$',11)
txt(840,1278,'LLL + LLH + LHL + LHH + HLL + HLH + HHL + HHH',9)

# Fusion: true three-source sets, per-channel source weights, residual refinement.
for cx,label_,color in zip([1300,1470,1640],['Source 1','Source 2','Source 3'],[PURPLE,GREEN,BLUE2]):
    block(cx-68,713,136,65,label_,color,10.5)
    block(cx-68,818,136,78,['1×1×1 Conv','resize to target'],BLUE,9.5)
    line([(cx,778),(cx,818)])
    line([(cx,896),(cx,926),(1470,926),(1470,950)])
block(1260,950,420,90,['Global average pooling → MLP','Softmax over 3 sources, per channel',r'$\sum_j\alpha_{j,c}=1$'],PURPLE,11)
block(1300,1083,340,72,[r'$\sum_j\alpha_j\odot\tilde f_j$','Residual refinement → Di'],GREEN,12)
line([(1470,1040),(1470,1083)])
txt(1470,1194,'D3: up(B), W(E3), E2',11)
txt(1470,1225,'D2: up(D3), W(E2), E1',11)
txt(1470,1256,'D1: D2, E1, E0',11)

# Compact legend and implemented composite objective.
block(1789,706,51,33,'',BLUE)
txt(1855,723,'Processing module',10.5,ha='left')
ax.add_patch(Rectangle((1789,764),51,33,fill=False,lw=1,ls=(0,(5,4))))
txt(1855,781,'Logical stage',10.5,ha='left')
circ(1815,840,'C'); txt(1855,840,'Channel concatenation',10.5,ha='left')
circ(1815,894,'+'); txt(1855,894,'Elementwise addition',10.5,ha='left')
line([(1789,948),(1840,948)]); txt(1855,948,'Feature / control flow',10.5,ha='left')
line([(1789,1002),(1840,1002)],True,False); txt(1855,1002,'Module detail link',10.5,ha='left')
txt(1965,1060,r'Shape: $C$ | $D\times H\times W$',11)
txt(1965,1092,'GN: GroupNorm; DW: depthwise',10)
txt(1965,1120,'Up: trilinear resize + projection',10)
txt(1965,1169,'Training objective',11.5,'bold')
txt(1965,1203,r'$L=L_{main}+0.5L_{aux1}$',12)
txt(1965,1231,r'$\quad+0.25L_{aux2}+0.1L_{boundary}$',12)
txt(1965,1267,'Segmentation: soft Dice + BCE',10)

txt(1100,1336,'Early encoder: 1×3×3 kernels; first downsampling: (1,2,2). Later downsampling: (2,2,2).',11)
txt(1100,1361,'D0 concatenates upsampled D1 with E0, then applies a 1×1×1 projection and residual refinement.',10)
txt(1100,1390,'Source: lung_ct_segmentation.py (SegmentationModel and module classes); hybrid_compact_v2_seed42/resolved_config.json.',9.5)

png=OUT/'msf_hsat_architecture.png'; svg=OUT/'msf_hsat_architecture.svg'
fig.savefig(png,dpi=300,facecolor='white')
fig.savefig(svg,facecolor='white')
plt.close(fig)
print(f'Architecture PNG: {png} (6600 x 4260 px, 300 dpi)')
print(f'Editable SVG: {svg}')
