import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd
from ast import literal_eval
from itertools import product
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import linalg as la
from string import ascii_lowercase

sys.path.append(os.getcwd())
from constants import DROOT
from UTILS.mutils import njoin, point_to_path
from UTILS.utils_dnn import setting_from_path

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

tick_size = 18.5 * 0.8
label_size = 18.5 * 0.8
axis_size = 18.5 * 0.8
legend_size = 14.1 * 0.8
lwidth = 1.8
msize = 10
text_size = 14
#C_LS_targets = list(mcolors.TABLEAU_COLORS.keys())
cmap = plt.get_cmap('tab10') 
#cmap = plt.get_cmap('PiYG')
C_LS = ["darkblue", "darkred"]
linestyle_ls = ["--", "-"]
# -------------------------------------

# global POST_DICT, REIG_DICT, C_LS
# pre- or post-activation Jacobian
POST_DICT = {0:'pre', 1:'post'}
# left or right eigenvectors
REIG_DICT = {0:'l', 1:'r'}
# color settings
#C_LS = ["tab:blue", "tab:orange"]
C_LS = ["darkblue", "darkred"]


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


def npc_metric(seeds_root, post=0, n_top=2, epochs=[0,100], is_3d=False):
    """
    Plots the class separation for MLPs in the first two columns, plots the ED/localization properties of 
    the neural representation PCs:
        - seed_path (str)
        - post (int): is either 0 or 1, if 1 plots post activation 
    """

    from sklearn.linear_model import LinearRegression

    global epoch_plot, metric_means, metric_data, D2_mean, D2_std, axs
    global depth, X_pca, indices, target_indices, sample_indices
    global colors_all

    """

    Average ED (across minibatches) and 
    D_2 of full batch image (averaged across top n_top PCs) vs depth

    """
    lwidth = 1.8

    post = int(post)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    n_top = int(n_top) if n_top != None else None
    
    alpha100_ls = [120,200]; g100_ls = [25,100,300]
    metriC_LS = ["class_sep", "ED", "D2_hidden", "cum_var"]
    ALL_METRIC_LS = ["class_sep", "ED", "D2_npc", "D2_npd", "D2_hidden", "cum_var", "eigvals_ordered"]
    for metric_name in metriC_LS:
        assert metric_name in ALL_METRIC_LS

    fcn, activation, total_epoch, net_folder = setting_from_path(seeds_root, alpha100_ls[0], g100_ls[0])
    L = int(fcn[2:])    
    if post == 0:
       depth = L
    elif post == 1:
        depth = L - 1
    else:
        depth = 2*L - 1
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls), depth])

    nrows = len(g100_ls)
    ncols = len(metriC_LS) + 1 if "class_sep" in metriC_LS else len(metriC_LS)
    fig, axs = plt.subplots(nrows, ncols, sharex=False,sharey=False,figsize=(12.5/3*ncols,3.1*nrows + 3),constrained_layout=True)   

    # class separation        
    if post == 0:
        xax_limit = 300
        yax_limit = 200  
        zax_limit = 30  
        # second final pre-activation
        #depth_selected = 16         
        depth_selected = 2*L - 4
    elif post == 1:
        xax_limit = 12
        yax_limit = 12
        zax_limit = 15
        # final activation
        #depth_selected = 17  
        depth_selected = 2*L - 3      
    else:
        xax_limit = 50
        yax_limit = 35
        zax_limit = 30
        # output
        #depth_selected = 18
        depth_selected = 2*L - 2
    # schematic figure of PCA
    selected_target_idxs = list(range(10))
    #selected_target_idxs = [1,9]
    # divide the axes
    pca_epoch = epochs[-1]
    N_data = 500
    transparency = 1    
    msize = 10
    marker = 's'
    if "class_sep" in metriC_LS:
        for (gidx, g100), (aidx, alpha100) in product(enumerate(g100_ls), enumerate(alpha100_ls)):  
            fcn, activation, total_epoch, net_folder = setting_from_path(seeds_root, alpha100, g100)   
            # load PCA
            #target_indices = np.load(join(seeds_root, net_folder, "target_indices.npy"), allow_pickle=True)                          
            #X_pca = np.load(join(seeds_root, net_folder, f"npc-depth={depth_selected}.npy"), allow_pickle=True)
            target_indices = np.load(njoin(seeds_root, net_folder, "X_pca_test", "target_indices.npy"), allow_pickle=True)                          
            X_pca = np.load(njoin(seeds_root, net_folder, "X_pca_test", f"npc-depth={depth_selected}-epoch={pca_epoch}.npy"), allow_pickle=True)
            
            # group in classes
            colors_all = np.empty(X_pca.shape[0])
            for iidx, cl_idx in enumerate(selected_target_idxs):
                sample_indices = target_indices[cl_idx]
                #colors_all[np.where(sample_indices==True)[0]] = iidx
                colors_all[sample_indices] = iidx

            if is_3d:
                axs[gidx,aidx].remove()
                axs[gidx,aidx] = fig.add_subplot(nrows, ncols, ncols*gidx+aidx+1, projection='3d')
            if gidx == 0 and aidx == 0:
                # 3D scatter
                if is_3d:
                    im = axs[gidx,aidx].scatter(X_pca[:N_data,0], X_pca[:N_data,1], 
                                                X_pca[:N_data,2],
                                                #vmin=0, vmax=len(selected_target_idxs)-1,
                                                #c=np.full(sample_indices.sum(),iidx), 
                                                c=colors_all[:N_data],
                                                #c=C_LS_targets[iidx],
                                                marker=marker,
                                                s=msize, alpha=transparency, cmap=cmap)  

                # 2D scatter
                else:
                    im = axs[gidx,aidx].scatter(X_pca[:N_data,0], X_pca[:N_data,1], 
                                                vmin=0, vmax=len(selected_target_idxs)-1,
                                                #c=np.full(sample_indices.sum(),iidx), 
                                                #c=C_LS_targets[iidx],
                                                c=colors_all[:N_data],
                                                marker=marker,
                                                s=msize, alpha=transparency, cmap=cmap)  

            else:
                # 3D scatter
                if is_3d:
                    axs[gidx,aidx].scatter(X_pca[:N_data,0], X_pca[:N_data,1], 
                                            X_pca[:N_data,2],
                                            #vmin=0, vmax=len(selected_target_idxs)-1,
                                            #c=np.full(sample_indices.sum(),iidx), 
                                            c=colors_all[:N_data],
                                            #c=C_LS_targets[iidx],
                                            marker=marker,
                                            s=msize, alpha=transparency, cmap=cmap)      
                    
                # 2D scatter 
                else:
                    axs[gidx,aidx].scatter(X_pca[:N_data,0], X_pca[:N_data,1], 
                                        vmin=0, vmax=len(selected_target_idxs)-1,
                                        #c=np.full(sample_indices.sum(),iidx),                                             
                                        #c=C_LS_targets[iidx],
                                        c=colors_all[:N_data],
                                        marker=marker,
                                        s=msize, alpha=transparency, cmap=cmap)                            
                                    
            #print(X_pca[indices,0].shape)  # delete

            axs[gidx,aidx].set_xlim([-xax_limit,xax_limit]); axs[gidx,aidx].set_ylim([-yax_limit,yax_limit])
            if is_3d:
                axs[gidx,aidx].set_zlim([-zax_limit,zax_limit])
            else:
                axs[gidx,aidx].spines['top'].set_visible(False); axs[gidx,aidx].spines['bottom'].set_visible(False)
                axs[gidx,aidx].spines['right'].set_visible(False) ;axs[gidx,aidx].spines['left'].set_visible(False)   
            axs[gidx,aidx].set_xticks([]); axs[gidx,aidx].set_xticklabels([])
            axs[gidx,aidx].set_yticks([]); axs[gidx,aidx].set_yticklabels([])
            if is_3d:
                axs[gidx,aidx].set_zticks([]); axs[gidx,aidx].set_zticklabels([])

            if is_3d:
                axs[gidx,aidx].set_xticks([-xax_limit,0,xax_limit])
                axs[gidx,aidx].set_yticks([-yax_limit,0,yax_limit])
                axs[gidx,aidx].set_zticks([-zax_limit,0,zax_limit])

                if aidx == 0:
                    axs[gidx,aidx].spines['left'].set_visible(True)
                    axs[gidx,aidx].set_yticklabels([-yax_limit,0,yax_limit])
                if gidx == nrows - 1:                    
                    axs[gidx,aidx].spines['bottom'].set_visible(True)
                    axs[gidx,aidx].set_xticklabels([-xax_limit,0,xax_limit]) 
                    if aidx == 0 and is_3d:
                        axs[gidx,aidx].set_zticklabels([-zax_limit,0,zax_limit]) 
            else:
                if aidx == 0:
                    axs[gidx,aidx].spines['left'].set_visible(True)
                    axs[gidx,aidx].set_yticks([-yax_limit,0,yax_limit]); axs[gidx,aidx].set_yticklabels([-yax_limit,0,yax_limit])
                if gidx == nrows - 1:                    
                    axs[gidx,aidx].spines['bottom'].set_visible(True)
                    axs[gidx,aidx].set_xticks([-xax_limit,0,xax_limit]); axs[gidx,aidx].set_xticklabels([-xax_limit,0,xax_limit]) 
                        
    for midx, metric_name in enumerate(metriC_LS[1:]):
        for gidx, g100 in enumerate(g100_ls):  
            # remove spines
            #axs[gidx, midx+2].spines['top'].set_visible(False); axs[gidx,midx+2].spines['right'].set_visible(False) 
            # label ticks
            #axs[gidx, midx+2].tick_params(axis='both', labelsize=axis_size - 3.5)   
            for epoch_plot in epochs:
                for aidx, alpha100 in enumerate(alpha100_ls):       
                    # Extract numeric arguments.
                    alpha, g = int(alpha100)/100., int(g100)/100.
                    # load nets and weights
                    #net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
                    fcn, activation, total_epoch, net_folder = setting_from_path(seeds_root, alpha100, g100)

                    lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                    if metric_name == "ED":
                        metric_means = np.load(njoin(seeds_root, net_folder, f"ed-batches_{POST_DICT[post]}", f"ED_means_{epoch_plot}.npy"))
                        metric_stds = np.load(njoin(seeds_root, net_folder, f"ed-batches_{POST_DICT[post]}", f"ED_stds_{epoch_plot}.npy"))
                        # only includes preactivation    
                        depth = len(metric_means[0])
                        # save ED later
                        ED_all[aidx, gidx, :] = metric_means[0]
                        # mean
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), metric_means[0],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = C_LS[aidx])
                        # standard deviation
                        axs[gidx,midx+2].fill_between(np.arange(1, depth+1), metric_means[0] - metric_stds[0], metric_means[0] + metric_stds[0], color = C_LS[aidx], alpha=0.2) 

                        # leaving extra space for labels
                        #if gidx == len(g100_ls) - 1:
                        #    axs[0,gidx].set_ylim(0,200)

                    elif "D2" in metric_name:
                        D2_mean = []  # D2's corresponding to the eigenvector of the largest eigenvalue of the covariance matrix  
                        D2_std = []
                        for l in range(depth):        
                            if metric_name == "D2_npc":                
                                metric_data = np.load(njoin(seeds_root, net_folder, f"dq_npc-fullbatch_{POST_DICT[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            elif metric_name == "D2_npd":
                                metric_data = np.load(njoin(seeds_root, net_folder, f"dq_npd-fullbatch_{POST_DICT[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            elif metric_name == 'D2_hidden':
                                metric_data = np.load(njoin(seeds_root, net_folder, f"dq_hidden-fullbatch_{POST_DICT[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            #quit()  # delete

                            if metric_name in ["D2_npc", "D2_npd"]:                                  
                                if n_top != None:                          
                                    D2_mean.append(metric_data[:n_top].mean())
                                    D2_std.append(metric_data[:n_top].std())
                                else:
                                    n_top = round(ED_all[aidx, gidx, l])    # top PCs of the mean ED of that layer
                                    D2_mean.append(metric_data[0:5].mean())
                                    D2_std.append(metric_data[0:5].std())    
                            else:
                                D2_mean.append(metric_data.mean())
                                D2_std.append(metric_data.std())                                  

                        D2_mean, D2_std = np.array(D2_mean), np.array(D2_std)
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), D2_mean,linewidth=lwidth,linestyle=lstyle,
                                              alpha=1, c = C_LS[aidx]) 

                        # standard deviation
                        axs[gidx,midx+2].fill_between(np.arange(1, depth+1), D2_mean - D2_std, D2_mean + D2_std, color = C_LS[aidx], alpha=0.2)

                    elif metric_name == "cum_var":
                        var_ls = []
                        for l in range(depth):
                            eigvals = np.load(njoin(seeds_root, net_folder, f"eigvals-fullbatch_{POST_DICT[post]}", f"npc-eigvals_{l}_{epoch_plot}.npy"))                            
                            if n_top != None: 
                                var_ls.append(eigvals[:n_top].sum()/eigvals.sum())
                            else:
                                n_top = round(ED_all[aidx, gidx, l])    # top PCs of the mean ED of that layer
                                var_ls.append(eigvals[:n_top].sum()/eigvals.sum())
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), var_ls,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = C_LS[aidx]) 

                    elif metric_name == "eigvals_ordered" and epoch_plot == epochs[-1]: 
                        l_selected = depth
                        eigvals = np.load(njoin(seeds_root, net_folder, f"eigvals-fullbatch_{POST_DICT[post]}", f"npc-eigvals_{l_selected - 1}_{epoch_plot}.npy"))                       
                        reg = LinearRegression().fit(np.log(np.arange(1, len(eigvals)+1))[0:100].reshape(-1,1), np.log(eigvals)[0:100].reshape(-1,1))
                        # eye guide line
                        b = np.log(eigvals[0])
                        x_guide = np.arange(1, len(eigvals)+1)
                        y_guide = np.exp(reg.coef_[0] * np.log(x_guide) + b)

                        axs[gidx,midx+2].loglog(np.arange(1, len(eigvals)+1), eigvals,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = C_LS[aidx]) 
                        axs[gidx,midx+2].loglog(x_guide, y_guide,linewidth=lwidth,linestyle='--',
                                         alpha=1, c = 'k') 

        print(f"Metric {metric_name}")                 

    # legend
    legend_idx = 0
    selected_col = 2
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    for aidx in range(len(alpha100_ls)):
        axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, c = C_LS[aidx], label=rf"$\alpha$ = {alpha100_ls[aidx]/100}")
    #for cl_idx in range(len(selected_target_idxs)):
    #    axs[legend_idx,selected_col].plot([], [], marker='.', c=C_LS_targets[cl_idx], label=rf"Class {cl_idx + 1}")
    axs[legend_idx,selected_col].legend(fontsize=legend_size+3, ncol=2, loc="lower left", bbox_to_anchor=[0, 1.25],
                                        frameon=True)

    for row in range(nrows):
        for col in range(ncols):
            # label ticks
            axs[row,col].tick_params(axis='both', labelsize=axis_size + 1)

    # metrics plotted against layers
    for row in range(nrows):
        for col in range(2,5):
            axs[row,col].set_xlim(1, depth)
            axs[row,col].set_xticks(list(range(1,depth+1)))
            xtick_ls = []
            for num in range(1,depth+1):
                if num % 2 == 1:
                    xtick_ls.append(str(num))
                else:
                    xtick_ls.append('')
            if row == nrows-1:
                axs[row,col].set_xticklabels(xtick_ls)
            else:
                axs[row,col].set_xticklabels([])
    # d2 and cumulative variance
    for col in [3,4]:
        for row in range(nrows):
            axs[row,col].set_ylim(-0.05,1.05) 
            axs[row,col].set_yticks(np.arange(0,1.01,0.2)) 

    fig.tight_layout(h_pad=2.5, w_pad=-5)
    #fig.tight_layout(h_pad=6)
    plot_path = njoin(root_data, f"figure_ms/{fcn}_npc")
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    epochs_str = [str(epoch) for epoch in epochs]
    epochs_str = "_".join(epochs_str)
    g100_str = [str(g100) for g100 in g100_ls]
    g100_str = "_".join(g100_str)
    metric_str = "_".join(metriC_LS)

    total_classes = 10

    if "fcn_grid" in seeds_root or "gaussian_data" not in seeds_root:
        file_full = f"{plot_path}/{fcn}_mnist_post={post}_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
    else:
        gaussian_data_setting = pd.read_csv(njoin(seeds_root,"gaussian_data_setting.csv"))
        X_dim, Y_classes, cluster_seed, assignment_and_noise_seed = gaussian_data_setting.loc[0,["X_dim", "Y_classes, ,noise_sigma", "cluster_seed,assignment_and_noise_seed"]]
        file_full = f"{plot_path}/{fcn}_gaussian_post={post}"
        file_full += f"_{X_dim}_{Y_classes}_{cluster_seed}_{assignment_and_noise_seed}"
        file_full += f"_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
    print(f"Figure saved as {file_full}")
    #plt.savefig(file_full, bbox_inches='tight')
    plt.savefig(file_full)
    plt.close()     

    # horizontal colorbar
    fig_cbar = plt.figure()
    #cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])  # vertical cbar
    cbar_ax = fig_cbar.add_axes([0.85, 0.20, 0.75, 0.03])  # horizontal cbar
    cbar_ticks = [(total_classes - 1)/total_classes * (0.5 + i) for i in range(total_classes)]
    cbar_tick_labels = list(range(total_classes))
    cbar = fig_cbar.colorbar(im, cax=cbar_ax, ticks=cbar_ticks, orientation='horizontal')
    cbar.outline.set_visible(False)
    
    #colorbar_index(ncolors=len(selected_target_idxs), cmap=cmap)   

    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.set_xticklabels(cbar_tick_labels)
    cbar.ax.tick_params(axis='x', labelsize=tick_size)
    
    plt.savefig(njoin(plot_path, f'pca_cbar_post={post}.pdf'), bbox_inches='tight')           



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])