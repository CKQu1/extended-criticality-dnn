import matplotlib.pyplot as plt
import matplotlib as mpl

#fig = plt.figure(figsize=(12.5 + 4.5, 9.5/2*3 + 0.5))
fig = plt.figure(figsize=(25, 28))
gs = mpl.gridspec.GridSpec(880, 740, wspace=0, hspace=0)   
# main plots 
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[320:470, 100:620])
ax6 = fig.add_subplot(gs[500:700, 0:200])
ax7 = fig.add_subplot(gs[500:700, 260:460])
ax8 = fig.add_subplot(gs[500:700, 520:720])
ax5 = fig.add_subplot(gs[730:880,100:620])
# colorbars
ax1_cbar = fig.add_subplot(gs[0:200, 730:740])
#ax2_cbar = fig.add_subplot(gs[500:700, 210:220])
#ax3_cbar = fig.add_subplot(gs[500:700, 470:480])
ax4_cbar = fig.add_subplot(gs[500:700, 730:740])
# inset
ax1_inset = fig.add_subplot(gs[5:81, 130:206])

lwidth = 4
x = list(range(11))
ax4.plot(x,x)
ax5.plot(x,x)
ax4.set_xlim(0,10)
ax5.set_xlim(0,10)

# titles
top_row_titles = ["Probablity density", "Probablity density", "Top-5 accuracy"]
for ax_idx, ax in enumerate([ax1,ax2,ax3]):
    ax.set_title(top_row_titles[ax_idx], fontsize=18.5 * 1.5)
#plt.suptitle("Pretrained networks", fontsize=18.5 * 1.5)
ax2.text(.95, 1.3, "Pretrained CNNs", transform=ax2.transAxes, fontweight='bold',
         fontsize=18.5 * 2, va='top', ha='right')
ax4.text(.735, 1.3, "Fully-connected DNNs", transform=ax4.transAxes, fontweight='bold',
         fontsize=18.5 * 2, va='top', ha='right')

# extended lines
for init_epoch in [0,5,10]:
    ax4.axvline(x=init_epoch, c='grey', linestyle="--", linewidth=lwidth-2)
    ax4.axvline(x=init_epoch, ymin=-0.2, ymax=1,
                     c='grey', linestyle="--", linewidth=lwidth-2,
                     clip_on=False)

    ax5.axvline(x=init_epoch, c='grey', linestyle="--", linewidth=lwidth-2)
    ax5.axvline(x=init_epoch, ymin=1, ymax=1.2,
                     c='grey', linestyle="--", linewidth=lwidth-2,
                     clip_on=False)

# fancy colors
naxes = len(fig.axes)
for i, ax in enumerate(fig.axes):
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_facecolor(cmap(float(i)/(naxes-1)))

#plt.show()
fig1_path = "/project/PDLAI/project2_data/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_template_grid2.pdf", bbox_inches='tight')
#plt.show()
