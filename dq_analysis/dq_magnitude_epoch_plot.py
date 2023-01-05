import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io as sio
import torch

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from numpy import linalg as la

plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.rcParams["font.family"] = "Helvetica"
plt.switch_backend('agg')

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
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
data_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/{post_dict[post]}jac_layerwise"

# ----- plot phase transition -----

title_size = 23.5 * 1.5
tick_size = 23.5 * 1.5
label_size = 23.5 * 1.5
axis_size = 23.5 * 1.5
legend_size = 23.5 * 1.5
c_ls = ["tab:blue", "tab:orange"]


alpha100_ls = [120,200]
#alpha100_ls = [120]
g100 = 150

q_folder_idx = 25
missing_data = []
# in the future for ipidx might be needed
# test first
#for epoch in [0,1]:
#    for layer in range(0,2):
#for epoch in [0,1] + list(range(50,651,50)):   # all
for epoch in [650]:
    #for layer in range(0,10):
    for f_idx in range(len(alpha100_ls)):
        #for f_idx in range(len(extension_names)):
            alpha100 = alpha100_ls[f_idx]
            extension_name = f"dw_alpha{alpha100}_g{g100}_ipidx0_epoch{epoch}"          
            # load DW's
            DW_all = torch.load(f"{data_path}/{extension_name}")
            DWs_shape = DW_all.shape
            for layer in [4]:

                # set up figure
                fig, ax = plt.subplots(1, 1 ,figsize=(9.5,7.142))      

                # ticks
                #ax1.set_xticks(np.arange(0,2.05,0.5))

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # ticks
                #axs[i].tick_params(axis='both',labelsize=tick_size)

                # ticks
                #if i == 0 or i == 2:
                #axs[i].set_xticks(np.linspace(100,600,6))

                #axs[i].tick_params(axis='both',labelsize=tick_size)
                
                #axs[i].set_yticks(mult_grid)
                #axs[i].set_ylim(0,3.25)

                # set ticks
                ax.set_yticks(np.arange(0,0.151,0.05))
                ax.set_yticklabels(np.round(np.arange(0,0.151,0.05),2))

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

                DW = DW_all[layer].numpy()
                if layer == DWs_shape[0]:
                    DW = DW[:10,:]
                    DW = DW.T @ DW  # final DW is 10 x 784

                # left eigenvector
                if reig == 0:
                    DW = DW.T

                print(f"layer {layer}: {DW.shape}")
                eigvals, eigvecs = la.eig(DW)
                print(f"Max eigenvalue magnitude: {np.max(np.abs(eigvals))}.")

                ax.set_xlim(0,800)
                ax.set_ylim(0,0.15)

                # eigenvector corresponding to the eigenvalue with the largest magnitude
                ax.plot(np.abs(eigvecs[:,np.argmax(np.abs(eigvals))]), c = c_ls[f_idx], linewidth=3)
                
                if f_idx == 1:
                    ax.set_xlabel('Site', fontsize=axis_size)
                ax.set_ylabel('Magnitude', fontsize=axis_size)

                #ax.set_title(f"Layer {layer+1}, Epoch {epoch}", fontsize=title_size)
                ax.set_title(rf"$\alpha$ = {alpha100/100}", fontsize=title_size)

                ax.legend(fontsize = legend_size, frameon=False)
                plt.tight_layout()
                #plt.show()

                fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
                if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
                # alleviate memory
                plt.savefig(f"{fig1_path}/eigvec_jac_single_{post_dict[post]}_{reig_dict[reig]}_alpha100={alpha100}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')
                plt.clf()
                plt.close(fig)

                #plt.show()

    #print(f"Epoch {epoch} layer {layer} done!")
    print(f"Epoch {epoch} done!")

#np.savetxt("/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data", missing_data)
#np.savetxt("/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/missing_data.txt", np.array(missing_data), fmt='%s')

