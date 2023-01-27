import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import scipy.io as sio
import torch

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
#from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from numpy import linalg as la
from os.path import join

sys.path.append(os.getcwd())
from path_names import root_data

def IPR(vec, q):
    return sum(abs(vec)**(2*q)) / sum(abs(vec)**2)**q

def D_q(vec, q):
    return np.log(IPR(vec, q)) / (1-q) / np.log(len(vec))

plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.rcParams["font.family"] = "Helvetica"
plt.switch_backend('agg')

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
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
#data_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/{post_dict[post]}jac_layerwise"
data_path = join(root_data, f"geometry_data/{post_dict[post]}jac_layerwise")

# ----- plot phase transition -----

title_size = 23.5 * 2.5
tick_size = 23.5 * 2.5
label_size = 23.5 * 2.5
axis_size = 23.5 * 2.5
legend_size = 23.5 * 2.5
#c_ls = ["tab:blue", "tab:orange"]
c_ls = ["blue", "red"]


alpha100_ls = [120,200]
g100 = 100
trans_ls = np.linspace(0,1,len(alpha100_ls)+1)[::-1]
max_mag = 0     # maximum magnitude of eigenvalues

missing_data = []
# in the future for ipidx might be needed
# test first
#for epoch in [0,1]:
#    for layer in range(0,2):
#for epoch in [0,1] + list(range(50,651,50)):   # all
for epoch in [0,650]:
    
    # set up figure
    fig, ax = plt.subplots(1, 1 ,figsize=(9.5,7.142))      

    # ticks
    #ax1.set_xticks(np.arange(0,2.05,0.5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set ticks
    #ax.set_yticks(np.arange(0,0.151,0.05))
    #ax.set_yticklabels(np.round(np.arange(0,0.151,0.05),2))

    # label ticks
    ax.tick_params(axis='x', labelsize=axis_size - 1)
    ax.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for f_idx in range(len(alpha100_ls)):
        #for f_idx in range(len(extension_names)):
            alpha100 = alpha100_ls[f_idx]
            extension_name = f"dw_alpha{alpha100}_g{g100}_ipidx0_epoch{epoch}"          
            # load DW's
            DW_all = torch.load(f"{data_path}/{extension_name}")
            DWs_shape = DW_all.shape
            for layer in [4]:

                # ----- 1. Plot eigenvalue norms vs D_2 of eigenvector of Jacobian -----
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

                # order eigenvalues based on magnitudes
                indices = np.argsort(np.abs(eigvals))                
                d2_arr = [ D_q(eigvecs[:,idx],2) for idx in indices ]
                d2_arr = np.array(d2_arr)
                #ax.scatter(np.abs(eigvals[indices]), d2_arr, c = c_ls[f_idx], alpha=trans_ls[f_idx], linewidth=2.5)
                ax.plot(np.abs(eigvals[indices]), d2_arr, c = c_ls[f_idx], linewidth=2.5)
                
                max_mag = max(max_mag, np.max(np.abs(eigvals)))

                if f_idx == 1:
                    ax.set_xlabel('Eigenvalue magnitude', fontsize=axis_size)
                ax.set_ylabel(r'$D_2$', fontsize=axis_size)

    #ax.set_xlim(0,2)
    ax.set_xlim(-0.05, round(max_mag,1) + 0.05)
    ax.set_ylim(0,1)

    # tick labels
    ax.set_xticks([0,0.5,1.0,1.5])
    ax.set_xticklabels([0,0.5,1.0,1.5])

    ax.set_title(f"Layer {layer+1}, Epoch {epoch}", fontsize=title_size)

    #ax.legend(fontsize = legend_size, frameon=False)
    plt.tight_layout()

    #fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
    fig1_path = join(root_data, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
    if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
    # alleviate memory
    plt.savefig(f"{fig1_path}/jac_d2-vs-eigval_jac_{post_dict[post]}_{reig_dict[reig]}_alpha100={alpha100}_g100={g100}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')
    plt.clf()
    plt.close(fig)

    #plt.show()

    #print(f"Epoch {epoch} layer {layer} done!")
    print(f"Epoch {epoch} done!")

