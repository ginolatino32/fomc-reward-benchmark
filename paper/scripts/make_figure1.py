import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE='#2a78d6'; GRAY='#8f8e89'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
A=[('Length only (no text)',0.000,True),('TF–IDF',0.144,False),('Reconstructed rules',0.324,False),('Rules, quadratic',0.378,False),('LLM tone counts',0.490,False),('Frozen embedding',0.501,False)]
B=[('Controls only (no text)',0.123,True),('+ Mention frequency',0.111,False),('+ Reconstructed rules',0.043,False),('+ LLM tone counts',0.109,False),('+ Frozen embedding',0.000,False)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.edgecolor':INK2,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK})
fig,axes=plt.subplots(2,1,figsize=(3.33,3.05),gridspec_kw={'height_ratios':[len(A),len(B)],'hspace':0.62})
for ax,data,title in [(axes[0],A,'(a) Reward: recover the preceding return (99 meetings)'),(axes[1],B,'(b) Reward: predict the next target change (98 meetings)')]:
    labels=[d[0] for d in data][::-1]; vals=[d[1] for d in data][::-1]; base=[d[2] for d in data][::-1]
    y=range(len(data))
    ax.barh(y,vals,height=0.56,color=[GRAY if b else BLUE for b in base],edgecolor='none')
    bl=[v for v,b in zip(vals,base) if b][0]
    ax.axvline(bl,color=GRAY,lw=0.9,ls=(0,(3,2)),zorder=0)
    for yi,v in zip(y,vals):
        ax.text(v+0.008,yi,f'{v:.3f}',va='center',ha='left',fontsize=6.6,color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(0,0.6); ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
    ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['left'].set_linewidth(0.6); ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(length=0)
    ax.set_title(title,loc='left',fontsize=7.4,color=INK,pad=4)
axes[1].set_xlabel('Out-of-sample R² (higher is better)')
fig.savefig('fig/reward_headroom.pdf',bbox_inches='tight'); fig.savefig('fig/reward_headroom.png',dpi=220,bbox_inches='tight')
