import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import scipy.io as sio
import torch

import pandas as pd
#import seaborn as sns
from ast import literal_eval
from itertools import product
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import linalg as la
from os.path import join
from string import ascii_lowercase

sys.path.append(os.getcwd())
from constants import DROOT
from pubplot import set_size
from utils_dnn import IPR, D_q
from UTILS.mutils import njoin, point_to_path

# ---------- Figure settings ----------
plt.rcParams["font.family"] = "sans-serif"     # set plot font globally
#plt.switch_backend('agg')
MARKERSIZE = 4
#BIGGER_SIZE = 10
BIGGER_SIZE = 8
LEGEND_SIZE = 7
TRANSP = 1  # transparency (corresponding to alpha in plot)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -------------------------------------

global post_dict, reig_dict, c_ls
# pre- or post-activation Jacobian
post_dict = {0:'pre', 1:'post'}
# left or right eigenvectors
reig_dict = {0:'l', 1:'r'}
# color settings
#c_ls = ["tab:blue", "tab:orange"]
c_ls = ["darkblue", "darkred"]


# -------------------- Jacobian quantities --------------------
def single_dw_mfrac(seeds_root, alpha100s=[120,200], g100s=[100], seeds=[0],
             epochs=[0,100], post=0, reig=1):

    """
    Plots quantities over single input and network ensemble.
    """

    global net_paths_dict, Dqs

    # ---------- Figure setup ----------
    fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.5))
    insets = []
    for ii in range(axs.shape[0]):
        insets.append(inset_axes(axs[ii,0], width="35%", height="35%", loc="upper right",
                      bbox_to_anchor=(-.05, .0, 1, 1),  # x0, y0 shift
                      bbox_transform=axs[ii,0].transAxes,
                      borderpad=0.1))

    for ii, ax in enumerate(axs.flatten()):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax.text(-0.1, 1.15, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)
    # -----------------------------------

    # load net_paths
    net_paths_dict= {}
    for alpha100, g100, seed in product(alpha100s, g100s, seeds):
        net_paths_dict[(alpha100, g100, seed)] = point_to_path(seeds_root, alpha100, g100, seed)\
                  
    # colors
    color_dict = {120: "darkblue", 200: "darkred"}

    # inset eigen mode sites
    eig_idxs = np.array(list(range(100)))
    # selected layerwise Jacobian
    l_selected = 5
    for (aidx, alpha100), g100, seed in product(enumerate(alpha100s), g100s, seeds):

        # figure settings
        color = color_dict[alpha100]

        # load data
        data_all = np.load(njoin(net_paths_dict[(alpha100, g100, seed)], 
                                  f'jac_epoch=0_post={post}_reig={reig}.npz'))

        # ---------- ROW 1 (Eigenmode magnitude) ----------
        input_idx = data_all['input_idx']
        DW_l = data_all[f'DW_l={l_selected}_input={input_idx}']
        eigvals, eigvecs = la.eig(DW_l)

        eigvec_selected = np.abs(eigvecs[:,np.argmax(np.abs(eigvals))])

        axs[aidx,0].plot(np.abs(eigvec_selected), color=color)
        if aidx == len(alpha100s) - 1:
            axs[aidx,0].set_xlabel('Site')
        axs[aidx,0].set_ylabel('Magnitude')
        axs[aidx,0].set_ylim([0, 1])

        # inset plot here
        insets[aidx].plot(np.abs(eigvec_selected[eig_idxs]), color=color)
        insets[aidx].set_ylim([0, 0.2]); insets[aidx].set_yticks([0, 0.2])

        for epoch_idx, epoch in enumerate(epochs):
            # load data
            if epoch != 0:
                data_all = np.load(njoin(net_paths_dict[(alpha100, g100, seed)], 
                                          f'jac_epoch={epoch}_post={post}_reig={reig}.npz'))

            # linestyle
            lstyle = '--' if epoch == 0 else '-'

            # ---------- ROW 2 (Jacobian eigenmodes before training) ----------
            qs = data_all['qs']
            Dqs = data_all[f'Dq_l={l_selected}_input={input_idx}']
            Dqs_mean = Dqs.mean(0); Dqs_std = Dqs.std(0)
            axs[epoch_idx, 1].plot(qs, Dqs_mean, color=color, linestyle=lstyle)
            axs[epoch_idx, 1].fill_between(qs, Dqs_mean - Dqs_std, Dqs_mean + Dqs_std, 
                            color = color, alpha=0.2)

            if epoch_idx == len(epochs) - 1:
                axs[epoch_idx, 1].set_xlabel(r'$q$') 
            axs[epoch_idx, 1].set_ylabel(r'$D_q$')
            axs[epoch_idx, 1].set_title(rf'Epoch = {epoch}')
            axs[epoch_idx, 1].set_ylim([0, 1.05])
            axs[epoch_idx, 1].set_yticks(np.arange(0,1.1,0.2))

            # ---------- ROW 3 (Jacobian eigenmodes after training) ----------
            dq_means = data_all['dq_means']; dq_stds = data_all['dq_stds']
            layers = list(range(1, dq_means.shape[1] + 1))
            axs[epoch_idx, 2].plot(layers, dq_means[0,:,-1], color=color, linestyle=lstyle)
            axs[epoch_idx, 2].fill_between(layers, 
                            dq_means[0,:,-1] - dq_stds[0,:,-1], dq_means[0,:,-1] + dq_stds[0,:,-1], 
                            color = color, alpha=0.2)
            
            # featured layer
            axs[epoch_idx, 2].axvline(x=l_selected + 1, c='dimgrey',linestyle=':',lw=1.2)     

            if epoch_idx == len(epochs) - 1:
                axs[epoch_idx, 2].set_xlabel(r'Layer $l$') 
            axs[epoch_idx, 2].set_ylabel(r'$D_2$')
            axs[epoch_idx, 2].set_title(rf'Epoch = {epoch}')
            axs[epoch_idx, 2].set_ylim([0, 1.05])
            axs[epoch_idx, 2].set_xticks([layer for layer in layers if layer % 2 == 1])
            axs[epoch_idx, 2].set_yticks(np.arange(0,1.1,0.2))

    # legend
    for (aidx, alpha100) in enumerate(alpha100s):
        axs[0,0].plot([], [], color=color_dict[alpha100], label=rf'$\alpha$ = {alpha100/100}')
    axs[0,0].legend(frameon=False, ncols=2,
                    loc="upper center", bbox_to_anchor=(0.5, 1.25))

    # Improve layout spacing
    fig.tight_layout()
    fig_path = njoin(DROOT, 'figure_ms', 'pretrained_analysis')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(njoin(fig_path, f'single_jacobian_seeds={seeds}.pdf'), bbox_inches="tight")  # dpi=300, 
    print(f'Figure saved in {fig_path}')


def multi_dw_mfrac(seeds_root, alpha100s=[120,200], g100s=[20, 100, 300], seeds=[0,1,2,3,4],
                   epochs=[0,100], post=0, reig=1):

    """
    Plots quantities over single input and network ensemble.
    seed_root = njoin(DROOT, 'fc10_sgd_mnist')
    """

    global net_paths_dict, Dqs

    # ---------- Figure setup ----------
    nrows, ncols = len(epochs), len(g100s)
    figsize = (2.5 * ncols, 2.25 * len(epochs))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for ii, ax in enumerate(axs.flatten()):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax.text(-0.1, 1.15, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)
    # -----------------------------------

    # load net_paths
    net_paths_dict= {}
    for alpha100, g100, seed in product(alpha100s, g100s, seeds):
        net_paths_dict[(alpha100, g100, seed)] = point_to_path(seeds_root, alpha100, g100, seed)\
                  
    # colors
    color_dict = {120: "darkblue", 200: "darkred"}

    for (aidx, alpha100), (gidx, g100), (sidx, seed), (eidx, epoch) in\
          product(enumerate(alpha100s), enumerate(g100s), enumerate(seeds), enumerate(epochs)):

        # figure settings
        color = color_dict[alpha100]

        # load data
        data_all = np.load(njoin(net_paths_dict[(alpha100, g100, seed)], 
                                  f'jac_epoch={epoch}_post={post}_reig={reig}.npz'))

        # linestyle
        lstyle = '--' if epoch == 0 else '-'

        dq_means = data_all['dq_means']; dq_stds = data_all['dq_stds']
        layers = list(range(1, dq_means.shape[1] + 1))
        d2_means = dq_means[:,:,-1].mean(0)
        d2_stds = dq_means[:,:,-1].std(0)
        axs[eidx, gidx].plot(layers, d2_means, color=color, linestyle=lstyle)
        axs[eidx, gidx].fill_between(layers, 
                        d2_means - d2_stds, d2_means + d2_stds, 
                        color = color, alpha=0.2)
        
        axs[eidx, gidx].set_ylim([0, 1.05])
        axs[eidx, gidx].set_xticks([layer for layer in layers if layer % 2 == 1])
        axs[eidx, gidx].set_yticks(np.arange(0,1.1,0.2))

        if gidx == 0:
            axs[eidx, gidx].set_ylabel(r'$\overline{D}_2$')
        if eidx == 0:
            axs[eidx, gidx].set_title(rf'$\sigma_w$ = {g100/100}')
        if eidx == len(epochs) - 1:
            axs[eidx, gidx].set_xlabel(r'Layer $l$') 

    # legend
    states = ['Initialization', 'Trained']
    lstyles = ['--', '-']
    nrow = 1
    for (aidx, alpha100) in enumerate(alpha100s):
        axs[1,0].plot([], [], color=color_dict[alpha100], label=rf'$\alpha$ = {alpha100/100}')
        axs[1,1].plot([], [], color='k', linestyle=lstyles[aidx], label=states[aidx])
    for ncol in range(2):
        axs[nrow,ncol].legend(frameon=False, ncols=2,
                        loc="upper center", bbox_to_anchor=(0.5, 1.25))

    # Improve layout spacing
    fig.tight_layout()
    fig_path = njoin(DROOT, 'figure_ms', 'pretrained_analysis')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(njoin(fig_path, f'multi_jacobian_seeds={seeds}.pdf'), bbox_inches="tight")
    print(f'Figure saved in {fig_path}')


# -------------------- Neural representation quantities --------------------



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])