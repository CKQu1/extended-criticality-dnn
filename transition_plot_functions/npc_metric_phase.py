import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io as sio
import sys

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
#from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from os.path import join

sys.path.append(os.getcwd())
from path_names import root_data
#plt.switch_backend('agg')

# colorbar
#cm_type = 'BrBG'
cm_type = 'PuOr'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

metric = "ED"
fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
#main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
main_path = "/project/PDLAI/project2_data"
path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"

# post/pre-activation and right/left-eigenvectors
post = 0
reig = 1

assert post == 1 or post == 0, "No such option!"
assert reig == 1 or reig == 0, "No such option!"
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

save_path = f"/project/PDLAI/project2_data/geometry_data/npc_layerwise_{post_dict[post]}_{reig_dict[reig]}"

# new version phase boundaries
bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)

# ----- plot phase transition -----

title_size = 23.5 * 2
tick_size = 23.5 * 2
label_size = 23.5 * 2
axis_size = 23.5 * 2
legend_size = 23.5 * 2
linewidth = 0.8 * 2
text_size = 14 * 2

mult_lower = 0.25
mult_upper = 3
#mult_upper = 4
mult_N = int((mult_upper - mult_lower)/0.25 + 1)
mult_grid = np.linspace(mult_lower,mult_upper,mult_N)
mult_incre = round(mult_grid[1] - mult_grid[0],2)

alpha_lower = 1
alpha_upper = 2
alpha_N = int((alpha_upper - alpha_lower)/0.1 + 1)
alpha_grid = np.linspace(alpha_lower,alpha_upper,alpha_N)
alpha_incre = round(alpha_grid[1] - alpha_grid[0],1)

missing_data = []
#for epoch in [0,1] + list(range(50,651,50)):   # all
#for epoch in [0,1,50,100,650]:
#for epoch in [0,1,50,100,650]:
for epoch in [0,650]:
    for layer in range(0,10):
    #for layer in [0,2,4]:

        # ----- Plot grid -----

        good = 0
        alpha_m_ls = []
        metric_mean_ls = []
        metric_std_ls = []

        #for i in range(0,len(net_ls)):
        for alpha100 in range(100, 201, 10):
            for g100 in range(25, 301, 25):
                
                alpha, m = int(alpha100)/100, int(g100)/100
                net_path = join(path, f"{net_type}_id_stable{round(alpha,1)}_{round(m,2)}_epoch650_algosgd_lr=0.001_bs=1024_data_mnist")
                data_path = join(net_path, "ed-dq-batches_pre_r")
                metric_data = np.load(f"{data_path}/{metric}_means_{epoch}.npy")                
                metric_mean = metric_data[0][layer]   
                metric_data = np.load(f"{data_path}/{metric}_stds_{epoch}.npy")
                metric_std = metric_data[0][layer]  

                alpha_m_ls.append((alpha,m))
                metric_mean_ls.append(metric_mean)
                metric_std_ls.append(metric_std)

                good += 1     
         

        # if want to use imshow, convert to grid form
        assert len(alpha_m_ls) == len(metric_mean_ls) and len(alpha_m_ls) == len(metric_std_ls)

        # colorbar bound
        cmap_bd = [ [np.percentile(metric_mean_ls,5), np.percentile(metric_mean_ls,95)], [np.percentile(metric_std_ls,5), np.percentile(metric_std_ls,95)] ]

        mean_mesh = np.zeros((mult_N,alpha_N))
        std_mesh = np.zeros((mult_N,alpha_N))

        for t in range(len(alpha_m_ls)):
            
            alpha,mult = alpha_m_ls[t]

            x_loc = int(round((mult_upper - mult) / mult_incre))
            y_loc = int(round((alpha - alpha_lower) / alpha_incre))

            mean_mesh[x_loc,y_loc] = metric_mean_ls[t]
            std_mesh[x_loc,y_loc] = metric_std_ls[t]

        fig1_path = join(root_data, f"figure_ms/{fcn}_npc")
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)

        # ----- plot template -----
        title_names = ["mean", "standard deviation"]
        save_names = ["mean", "std"]
        metrics_all = {title_names[0]: mean_mesh, title_names[1]: std_mesh}
        #for plot_idx in range(2): # plots mean and std 
        for plot_idx in range(1):   # only plots mean
            fig, ax = plt.subplots(1, 1,figsize=(9.5,7.142 - 0.5))
            # plot boundaries for each axs
            ax.plot(bound1.iloc[:,0], bound1.iloc[:,1], linewidth=2.5, color='k')

            # minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # label ticks
            ax.tick_params(axis='x', labelsize=axis_size - 1)
            ax.tick_params(axis='y', labelsize=axis_size - 1)
            ax.set_xticks([1,2,3])
            ax.set_xticks([1.0,1.5,2.0])

            #-----

            title_name = title_names[plot_idx]
            main_plot = ax.imshow(metrics_all[title_name],extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[plot_idx][0], vmax=cmap_bd[plot_idx][1], 
                                  cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
            #if q_folder == int(q_folder):
            #    ax.set_title(rf"$D_{{{int(q_folder)}}}$ {title_name}", fontsize=label_size)
            #else:
            #    ax.set_title(rf"$D_{{{round(q_folder,1)}}}$ {title_name}", fontsize=label_size)
            cbar = plt.colorbar(main_plot,ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.tick_params(labelsize=axis_size - 1)
            #l, u = round(cmap_bd[plot_idx][0],3), round(cmap_bd[plot_idx][1],3)
            l, u = cmap_bd[plot_idx]
            cbar.ax.set_yticks([round(l,1), round(u,1)])
            cbar.ax.tick_params(labelsize=tick_size)
            plt.tight_layout()
            plt.savefig(f"{fig1_path}/{fcn}_npc_{metric}_{save_names[plot_idx]}_phase_{post_dict[post]}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')

            #plt.show()
            plt.close(fig)
            #quit()

    #print(f"Epoch {epoch} layer {layer} done!")
    print(f"Epoch {epoch} done!")


