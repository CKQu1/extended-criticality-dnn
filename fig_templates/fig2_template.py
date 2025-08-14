import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure(figsize=(24, 25))
gs = mpl.gridspec.GridSpec(850, 740, wspace=0, hspace=0)   
# main plots 
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[290:440, 100:620])
ax5 = fig.add_subplot(gs[440:590,100:620])
ax6 = fig.add_subplot(gs[650:, 0:200])
ax7 = fig.add_subplot(gs[650:, 260:460])
ax8 = fig.add_subplot(gs[650:, 520:720])
# colorbars
ax1_cbar = fig.add_subplot(gs[0:200, 730:740])
#ax2_cbar = fig.add_subplot(gs[650:, 210:220])
#ax3_cbar = fig.add_subplot(gs[650:, 470:480])
ax4_cbar = fig.add_subplot(gs[650:, 730:740])
# inset
#ax1_inset = fig.add_subplot(gs[5:81, 130:206])

# fancy colors
#cmap = mpl.cm.get_cmap("viridis")
naxes = len(fig.axes)
for i, ax in enumerate(fig.axes):
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_facecolor(cmap(float(i)/(naxes-1)))

plt.show()
#fig1_path = "/project/PDLAI/project2_data/figure_ms"
#plt.savefig(f"{fig1_path}/pretrained_template_grid.pdf", bbox_inches='tight')
