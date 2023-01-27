import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io as sio
import sys

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from os.path import join

sys.path.append(os.getcwd())
from path_names import root_data

plt.switch_backend('agg')

# colorbar scheme
from matplotlib.cm import coolwarm

# colorbar
cm = cm.get_cmap('plasma')

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
#main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
main_path = join(root_data, "trained_mlps")

path = f"{main_path}/fcn_grid/{fcn}_grid"

# post/pre-activation and right/left-eigenvectors
post = 0
reig = 1

assert post == 1 or post == 0, "No such option!"
assert reig == 1 or reig == 0, "No such option!"
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

#dq_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/dq_layerwise"
#dq_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"
dq_path = join(root_data, f"geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}")

# new version phase boundaries
#bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
bound1 = pd.read_csv(f"{root_data}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
boundaries = []
bd_path = join(root_data, "phasediagram")
for i in range(1,102,10):
    boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))

# ----- plot phase transition -----

title_size = 23.5 * 2.5
tick_size = 23.5 * 2.5
label_size = 23.5 * 2.5
axis_size = 23.5 * 2.5
legend_size = 23.5 * 2.5
#c_ls = ["tab:blue", "tab:orange"]
c_ls = ["blue", "red"]

alpha100_ls = [120,200]
#g100 = 150
g100 = 100

missing_data = []
# in the future for ipidx might be needed
depths = np.arange(9)     # not including the final layer since there are only 10 neurons
#for epoch in [0,1] + list(range(50,651,50)):   # all
for epoch in [0, 650]:

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
    fig, ax = plt.subplots(1, 1 ,figsize=(9.5,7.142))
    #fig = plt.figure(figsize=(9.5,7.142))        

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # ticks
    #axs[i].tick_params(axis='both',labelsize=tick_size)

    # ticks
    #if i == 0 or i == 2:
    #axs[i].set_xticks(np.linspace(100,600,6))

    #axs[i].tick_params(axis='both',labelsize=tick_size)
    
    #axs[i].set_yticks(mult_grid)
    #axs[i].set_ylim(0,3.25)

    # set ticks
    ax.set_yticks(np.arange(0,2.1,0.5))
    ax.set_yticklabels(np.arange(0,2.1,0.5))

    # label ticks
    ax.tick_params(axis='x', labelsize=axis_size - 1)
    ax.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # ----- 1. Plot individual dqs -----
    # Figure (a) and (b): plot the D_q vs q (for different alphas same g), chosen a priori
    #axs[0].set_title(r"$\alpha$ = 1.2, $D_w^{1/\alpha}$ = 1.5", fontsize=label_size)
    #axs[1].set_title(r"$\alpha$ = 2.0, $D_w^{1/\alpha}$ = 1.5", fontsize=label_size)

    ylim_lower = 1
    ylim_upper = 1

    for f_idx, alpha100 in enumerate(alpha100_ls):
    
        extension_name = f"alpha{alpha100}_g{g100}_ipidx0_ep{epoch}.txt"
        
        df_mean = np.loadtxt(f"{dq_path}/dqmean_{extension_name}")
        df_std = np.loadtxt(f"{dq_path}/dqstd_{extension_name}")
        #qs = df_mean[:,0]  # first column are q's

        dq_mean_layer, dq_std_layer = [], []
        for layer in depths:
            # D2 (correlation dimension)
            dq_mean_layer.append( df_mean[-1,layer + 1] )
            dq_std_layer.append( df_std[-1,layer + 1] )

        dq_mean_layer = np.array(dq_mean_layer)
        dq_std_layer = np.array(dq_std_layer) 
        lower = dq_mean_layer - dq_std_layer
        upper = dq_mean_layer + dq_std_layer
            
        # averages of dq's with error bars
        ax.plot(depths+1, dq_mean_layer, linewidth=2.5, alpha=1, c = c_ls[f_idx], label=rf"$\alpha$ = {round(alpha100/100,1)}")
        ax.plot(depths+1, lower, linewidth=0.25, alpha=1, c = c_ls[f_idx])
        ax.plot(depths+1, upper, linewidth=0.25, alpha=1, c = c_ls[f_idx])
        ax.fill_between(depths+1, lower, upper, color = c_ls[f_idx], alpha=0.2)

    ylim_lower = min(ylim_lower, min(lower))
    ylim_upper = max(ylim_upper, max(upper))
    #ax.set_ylim(round(ylim_lower,1) - 0.05, round(ylim_upper,1) + 0.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlim(depths[0]+1,depths[-1]+1)

    ax.set_xlabel('Depth', fontsize=axis_size)
    ax.set_ylabel(r'$D_2$', fontsize=axis_size)

    ax.set_title(f"Epoch {epoch}", fontsize=title_size)

    #ax.legend(fontsize = legend_size, frameon=False)
    plt.tight_layout()

    #fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
    fig1_path = join(root_data, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
    if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
    # alleviate memory
    plt.savefig(f"{fig1_path}/jac_d2-vs-depth_{post_dict[post]}_{reig_dict[reig]}_g100={g100}_epoch={epoch}.pdf", bbox_inches='tight')
    #plt.clf()
    #plt.close(fig)
    #plt.show()

    #print(f"Epoch {epoch} layer {layer} done!")
    print(f"Epoch {epoch} done!")

