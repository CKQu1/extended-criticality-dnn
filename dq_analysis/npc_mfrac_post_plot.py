import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
import torch
from ast import literal_eval
from os.path import join
from tqdm import tqdm

sys.path.append(os.getcwd())

from npc_fcn import get_dataset
from path_names import root_data, model_log, id_to_path, get_model_id, get_alpha_g
from utils_dnn import setting_from_path

# plot settings
plt.rcParams["font.family"] = "sans-serif"     # set plot font globally
#plt.switch_backend('agg')
global c_ls, c_ls_targets, figw, figh, axw, axh
global title_size, tick_size, label_size, axis_size, legend_size
#title_size = 23.5 * 2
#tick_size = 23.5 * 2
#label_size = 23.5 * 2
#axis_size = 23.5 * 2
#legend_size = 23.5 * 2 - 12
tick_size = 18.5 * 0.8
label_size = 18.5 * 0.8
axis_size = 18.5 * 0.8
legend_size = 14.1 * 0.8
lwidth = 1.8
msize = 10
text_size = 14
#c_ls_targets = list(mcl.TABLEAU_COLORS.keys())
#cmap = plt.get_cmap('tab10') 
cmap = plt.get_cmap('PiYG')
c_ls = ["darkblue", "darkred"]
linestyle_ls = ["--", "-"]
#marker_ls = ["o", "."]
marker_ls = ["o", "x"]
figw, figh = 9.5, 7.142
axw, axh = figw * 0.7, figh * 0.7

global post_dict, reig_dict, all_metric_ls
post_dict = {0:'pre', 1:'post', 2:'all'}
reig_dict = {0:'l', 1:'r'}    
all_metric_ls = ["class_sep", "ED", "D2_npc", "D2_npd", "D2_hidden", "cum_var", "eigvals_ordered"]

# data_path = /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid
def metrics_vs_depth(data_path, post=0, display=False, n_top=5, epochs=[0,650]):
    """
    Plots the class separation for MLPs in the first two columns, plots the ED/localization properties of 
    the neural representation PCs:
        - data_path (str): /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid
        - post (int): is either 0 or 1, if 1 plots post activation 
    """

    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from NetPortal.models import ModelFactory

    global epoch_plot, metric_means, metric_data, D2_mean, D2_std, axs
    global depth, X_pca, indices, target_indices, sample_indices

    """

    Average ED (across minibatches) and 
    D_2 of full batch image (averaged across top n_top PCs) vs depth

    """

    post = int(post)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    n_top = int(n_top) if n_top != None else None
    display = literal_eval(display) if isinstance(display,str) else display
    
    alpha100_ls = [120,200]
    #alpha100_ls = [120]
    g100_ls = [25,100,300]
    #metric_ls = ["D2", "ED"]
    #metric_ls = ["ED", "D2", "cum_var", "eigvals_ordered"]
    #metric_ls = ["ED", "D2_npc", "D2_npd", "cum_var"]
    #metric_ls = ["class_sep", "ED", "D2_npd", "cum_var"]
    metric_ls = ["class_sep", "ED", "D2_hidden", "cum_var"]
    for metric_name in metric_ls:
        assert metric_name in all_metric_ls

    fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100_ls[0], g100_ls[0])
    L = int(fcn[2:])    
    if post == 0:
       depth = L
    elif post == 1:
        depth = L - 1
    else:
        depth = 2*L - 1
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls), depth])

    nrows = len(g100_ls)
    ncols = len(metric_ls) + 1 if "class_sep" in metric_ls else len(metric_ls)
    #fig, axs = plt.subplots(len(metric_ls), len(g100_ls), sharex = True,sharey=False,figsize=(9.5/2*3,7.142/4*3))
    #fig, axs = plt.subplots(len(metric_ls), len(g100_ls), sharex = True,sharey=False,figsize=(12.5,3.1*len(metric_ls)),constrained_layout=True)
    fig, axs = plt.subplots(nrows, ncols, sharex=False,sharey=False,figsize=(12.5/3*ncols,3.1*nrows),constrained_layout=True)   

    # class separation        
    if post == 0:
        xax_limit = 300
        yax_limit = 200    
        depth_selected = 16         
    elif post == 1:
        xax_limit = 15
        yax_limit = 12
        #depth_selected = 17
        depth_selected = 11
    else:
        xax_limit = 50
        yax_limit = 35
        depth_selected = 18
    # schematic figure of PCA
    selected_target_idxs = list(range(10))
    #selected_target_idxs = [1,9]
    # divide the axes
    pca_epoch = 650
    pca_epochs = [650]
    total_intervals = 3 * len(pca_epochs) - 1
    true_limits = []
    center_shifts = []
    x_scales = [0.5]*len(pca_epochs)
    if "class_sep" in metric_ls:
        for gidx, g100 in enumerate(g100_ls):
            for aidx, alpha100 in enumerate(alpha100_ls):  
                fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)   
                # load PCA
                #target_indices = np.load(join(data_path, net_folder, "target_indices.npy"), allow_pickle=True)                          
                #X_pca = np.load(join(data_path, net_folder, f"npc-depth={depth_selected}.npy"), allow_pickle=True)
                target_indices = np.load(join(data_path, net_folder, "X_pca_test", "target_indices.npy"), allow_pickle=True)                          
                X_pca = np.load(join(data_path, net_folder, "X_pca_test", f"npc-depth={depth_selected}-epoch={pca_epoch}.npy"), allow_pickle=True)
                
                # group in classes
                for iidx, cl_idx in enumerate(selected_target_idxs):
                    sample_indices = target_indices[cl_idx]
                    #plot_points = 5000
                    #sample_indices = list(target_indices[cl_idx][0:plot_points]) + [False] * (60000 - plot_points)
                    # old version
                    #axs[gidx,aidx].scatter(X_pca[sample_indices,0], X_pca[sample_indices,1], c=c_ls_targets[iidx],
                    #                       s=1.5, alpha=0.2)   
                    # new version

                    if gidx == 0 and aidx == 0:
                        im = axs[gidx,aidx].scatter(X_pca[sample_indices,0], X_pca[sample_indices,1], 
                                                    vmin=0, vmax=len(selected_target_idxs)-1,
                                                    c=np.full(sample_indices.sum(),iidx), 
                                                    #c=c_ls_targets[iidx],
                                                    s=1.5, alpha=0.2, cmap=cmap)  
                    else:
                        axs[gidx,aidx].scatter(X_pca[sample_indices,0], X_pca[sample_indices,1], 
                                               vmin=0, vmax=len(selected_target_idxs)-1,
                                               c=np.full(sample_indices.sum(),iidx), 
                                               #c=c_ls_targets[iidx],
                                               s=1.5, alpha=0.2, cmap=cmap)                       
                    #print(X_pca[indices,0].shape)  # delete

                axs[gidx,aidx].set_xlim([-xax_limit,xax_limit]); axs[gidx,aidx].set_ylim([-yax_limit,yax_limit])
                axs[gidx,aidx].spines['top'].set_visible(False); axs[gidx,aidx].spines['bottom'].set_visible(False)
                axs[gidx,aidx].spines['right'].set_visible(False) ;axs[gidx,aidx].spines['left'].set_visible(False)   
                axs[gidx,aidx].set_xticks([]); axs[gidx,aidx].set_xticklabels([])
                axs[gidx,aidx].set_yticks([]); axs[gidx,aidx].set_yticklabels([])
                # label ticks
                axs[gidx,aidx].tick_params(axis='both', labelsize=axis_size - 3.5)
                if aidx == 0:
                    axs[gidx,aidx].spines['left'].set_visible(True)
                    axs[gidx,aidx].set_yticks([-yax_limit,0,yax_limit]); axs[gidx,aidx].set_yticklabels([-yax_limit,0,yax_limit])
                if gidx == nrows - 1:                    
                    axs[gidx,aidx].spines['bottom'].set_visible(True)
                    axs[gidx,aidx].set_xticks([-xax_limit,0,xax_limit]); axs[gidx,aidx].set_xticklabels([-xax_limit,0,xax_limit])                                                                       
                           
    for midx, metric_name in enumerate(metric_ls[1:]):
        for gidx, g100 in enumerate(g100_ls):  
            # remove spines
            axs[gidx, midx+2].spines['top'].set_visible(False); axs[gidx,midx+2].spines['right'].set_visible(False) 
            # label ticks
            axs[gidx, midx+2].tick_params(axis='both', labelsize=axis_size - 3.5)   
            for epoch_plot in epochs:
                for aidx, alpha100 in enumerate(alpha100_ls):       
                    # Extract numeric arguments.
                    alpha, g = int(alpha100)/100., int(g100)/100.
                    # load nets and weights
                    #net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
                    fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)

                    lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                    if metric_name == "ED":
                        metric_means = np.load(join(data_path, net_folder, f"ed-batches_{post_dict[post]}", f"ED_means_{epoch_plot}.npy"))
                        metric_stds = np.load(join(data_path, net_folder, f"ed-batches_{post_dict[post]}", f"ED_stds_{epoch_plot}.npy"))
                        # only includes preactivation    
                        depth = len(metric_means[0])
                        # save ED later
                        ED_all[aidx, gidx, :] = metric_means[0]
                        # mean
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), metric_means[0],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx])
                        # standard deviation
                        axs[gidx,midx+2].fill_between(np.arange(1, depth+1), metric_means[0] - metric_stds[0], metric_means[0] + metric_stds[0], color = c_ls[aidx], alpha=0.2) 

                        # leaving extra space for labels
                        #if gidx == len(g100_ls) - 1:
                        #    axs[0,gidx].set_ylim(0,200)

                    elif "D2" in metric_name:
                        D2_mean = []  # D2's corresponding to the eigenvector of the largest eigenvalue of the covariance matrix  
                        D2_std = []
                        for l in range(depth):        
                            if metric_name == "D2_npc":                
                                metric_data = np.load(join(data_path, net_folder, f"dq_npc-fullbatch_{post_dict[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            elif metric_name == "D2_npd":
                                metric_data = np.load(join(data_path, net_folder, f"dq_npd-fullbatch_{post_dict[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            elif metric_name == 'D2_hidden':
                                metric_data = np.load(join(data_path, net_folder, f"dq_hidden-fullbatch_{post_dict[post]}", f"D2_{l}_{epoch_plot}.npy"))
                            #n_top = round(ED_all[aidx, gidx, l])    # top PCs of the mean ED of that layer  
                            if n_top != None:                          
                                D2_mean.append(metric_data[:n_top].mean())
                                D2_std.append(metric_data[:n_top].std())
                            else:
                                D2_mean.append(metric_data[0:5].mean())
                                D2_std.append(metric_data[0:5].std())

                        D2_mean, D2_std = np.array(D2_mean), np.array(D2_std)
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), D2_mean,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 

                        # standard deviation
                        axs[gidx,midx+2].fill_between(np.arange(1, depth+1), D2_mean - D2_std, D2_mean + D2_std, color = c_ls[aidx], alpha=0.2)

                    elif metric_name == "cum_var":
                        var_ls = []
                        for l in range(depth):
                            eigvals = np.load(join(data_path, net_folder, f"eigvals-fullbatch_{post_dict[post]}", f"npc-eigvals_{l}_{epoch_plot}.npy"))
                            #n_top = round(ED_all[aidx, gidx, l])    # variance explained by top PCs
                            if n_top != None: 
                                var_ls.append(eigvals[:n_top].sum()/eigvals.sum())
                            else:
                                var_ls.append(eigvals[:5].sum()/eigvals.sum())
                        axs[gidx,midx+2].plot(np.arange(1, depth+1), var_ls,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 

                    elif metric_name == "eigvals_ordered" and epoch_plot == epochs[-1]: 
                        l_selected = depth
                        eigvals = np.load(join(data_path, net_folder, f"eigvals-fullbatch_{post_dict[post]}", f"npc-eigvals_{l_selected - 1}_{epoch_plot}.npy"))                       
                        reg = LinearRegression().fit(np.log(np.arange(1, len(eigvals)+1))[0:100].reshape(-1,1), np.log(eigvals)[0:100].reshape(-1,1))
                        # eye guide line
                        b = np.log(eigvals[0])
                        x_guide = np.arange(1, len(eigvals)+1)
                        y_guide = np.exp(reg.coef_[0] * np.log(x_guide) + b)

                        axs[gidx,midx+2].loglog(np.arange(1, len(eigvals)+1), eigvals,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                        axs[gidx,midx+2].loglog(x_guide, y_guide,linewidth=lwidth,linestyle='--',
                                         alpha=1, c = 'k') 

                    

        print(f"Metric {metric_name}")     

    # adjust gaps between subplots
    #plt.subplots_adjust(hspace=0.3)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.25)

    # legend
    legend_idx = 0
    selected_col = 2
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    for aidx in range(len(alpha100_ls)):
        axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, c = c_ls[aidx], label=rf"$\alpha$ = {alpha100_ls[aidx]/100}")
    #for cl_idx in range(len(selected_target_idxs)):
    #    axs[legend_idx,selected_col].plot([], [], marker='.', c=c_ls_targets[cl_idx], label=rf"Class {cl_idx + 1}")
    axs[legend_idx,selected_col].legend(fontsize=legend_size, ncol=2, loc="lower left", bbox_to_anchor=[0, 1.25],
                                        frameon=True)

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
            axs[row,col].set_xticklabels(xtick_ls)
    # d2 and cumulative variance
    for col in [3,4]:
        for row in range(nrows):
            axs[row,col].set_ylim(-0.05,1.05) 
            axs[row,col].set_yticks(np.arange(0,1.01,0.2)) 

    #fig.tight_layout(h_pad=2, w_pad=2)
    plot_path = join(root_data, f"figure_ms/{fcn}_npc")
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    epochs_str = [str(epoch) for epoch in epochs]
    epochs_str = "_".join(epochs_str)
    g100_str = [str(g100) for g100 in g100_ls]
    g100_str = "_".join(g100_str)
    metric_str = "_".join(metric_ls)

    if not display:
        if "fcn_grid" in data_path or "gaussian_data" not in data_path:
            file_full = f"{plot_path}/{fcn}_mnist_post={post}_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
        else:
            gaussian_data_setting = pd.read_csv(join(data_path,"gaussian_data_setting.csv"))
            X_dim, Y_classes, cluster_seed, assignment_and_noise_seed = gaussian_data_setting.loc[0,["X_dim", "Y_classes, ,noise_sigma", "cluster_seed,assignment_and_noise_seed"]]
            file_full = f"{plot_path}/{fcn}_gaussian_post={post}"
            file_full += f"_{X_dim}_{Y_classes}_{cluster_seed}_{assignment_and_noise_seed}"
            file_full += f"_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
        print(f"Figure saved as {file_full}")
        plt.savefig(file_full, bbox_inches='tight')
        plt.close()

        # horizontal colorbar (can check geometry_analysis/greatcircle_proj2.py)
        fig_cbar = plt.figure()
        #cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])  # vertical cbar
        cbar_ax = fig_cbar.add_axes([0.85, 0.20, 0.75, 0.03])  # horizontal cbar
        cbar_ticks = list(range(10))
        cbar = fig_cbar.colorbar(im, cax=cbar_ax, ticks=cbar_ticks, orientation='horizontal')
        cbar.ax.set_xticklabels(cbar_ticks)
        cbar.ax.tick_params(axis='x', labelsize=tick_size)
        
        plt.savefig(join(plot_path, f'pca_cbar_post={post}.pdf'), bbox_inches='tight')           

    else:
        plt.show()

# examines the microscopic statistics of the covariance matrix
def micro_stats(data_path, metric, post=0, display=False, n_top=200, epochs=[0,650]):
    """

    microscopic statistics of the covariance matrix    

    """    
    assert metric in ["var", "var_cumsum", "d2_npc", "d2_npd"], "metric not in list"
    
    post = int(post)
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    n_top = int(n_top)
    display = literal_eval(display) if isinstance(display,str) else display
    
    alpha100_ls = [120,200]
    #alpha100_ls = [200]
    g100_ls = [25,100,300]
    #layers = [0,3,6]
    #layers = [0,4,8]
    layers = [0,3,6]
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls)])

    fig, axs = plt.subplots(len(layers), 3, sharex=True,sharey=False,figsize=(12.5,3.1*len(layers)),constrained_layout=True)    
    
    for lidx, l in enumerate(layers):
        for gidx, g100 in enumerate(g100_ls):
            alpha_m_ls = []
            good = 0           
            for epoch_plot in epochs:
                for aidx, alpha100 in enumerate(alpha100_ls):               
                    # Extract numeric arguments.
                    alpha, g = int(alpha100)/100., int(g100)/100.
                    # load nets and weights
                    fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
                    L = int(fcn[2:])

                    ED_means = np.load(join(data_path, net_folder, f"ed-batches_{post_dict[post]}", f"ED_means_{epoch_plot}.npy"))
                    # only includes preactivation
                    ED_all[aidx, gidx] = ED_means[0,l]    
                    if metric == "d2_npc":               
                        D2s = np.load(join(data_path, net_folder, f"dq_npc-fullbatch_{post_dict[post]}", f"D2_{l}_{epoch_plot}.npy"))
                    elif metric == "d2_npd":               
                        D2s = np.load(join(data_path, net_folder, f"dq_npd-fullbatch_{post_dict[post]}", f"D2_{l}_{epoch_plot}.npy"))

                    eigvals = np.load(join(data_path, net_folder, f"eigvals-fullbatch_{post_dict[post]}", f"npc-eigvals_{l}_{epoch_plot}.npy"))
                    var_percentage = eigvals/eigvals.sum()
                    var_cum = np.cumsum(var_percentage)

                    lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                    if metric == "var_cumsum":
                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1)[:n_top], var_cum[:n_top],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                    elif "d2" in metric:
                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1)[:n_top], D2s[:n_top],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                    elif metric == "var":
                        # add powerlaw fit to the log-log plot of the covariance eigenspectrum density
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(np.log(np.arange(1, len(eigvals)+1))[0:100].reshape(-1,1), np.log(eigvals)[0:100].reshape(-1,1))
                        # eye guide line
                        b = np.log(eigvals[0])
                        x_guide = np.arange(1, len(eigvals)+1)
                        y_guide = np.exp(reg.coef_[0] * np.log(x_guide) + b)

                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1), eigvals,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                        axs[lidx,gidx].plot(x_guide, y_guide,linewidth=lwidth,linestyle='--',
                                         alpha=1, c = 'k') 

                        axs[lidx,gidx].set_title(f"{-reg.coef_[0]}")
                        axs[lidx,gidx].set_xscale('log')                        
                        axs[lidx,gidx].set_yscale('log')
                        # set log-log

                    # plot ED 
                    # presentation type 1
                    #axs[0,gidx].plot(np.arange(1, n_top+1), [0.1]*n_top, linewidth=lwidth,linestyle="-",
                    #                 alpha=1, c = "gray")
                    #axs[0,gidx].axvline(x = ED_all[aidx, gidx], linewidth=lwidth-1,linestyle=lstyle,
                    #                    alpha=0.75, c = c_ls[aidx], 
                    #                    ymin=0.05, ymax=0.15) 

                    # presentation type 2
                    #axs[1,gidx].axvline(x = ED_all[aidx, gidx], linewidth=lwidth,linestyle=lstyle,
                    #                    alpha=0.4, c = c_ls[aidx]) 

                    #mstyle = marker_ls[0] if epoch_plot == 0 else marker_ls[1]
                    #axs[0,gidx].plot([ED_all[aidx, gidx]], [1], marker=mstyle, markersize=msize, c=c_ls[aidx])
                    #axs[1,gidx].plot([ED_all[aidx, gidx]], [1], marker=mstyle, markersize=msize, c=c_ls[aidx])

    # adjust gaps between subplots
    plt.subplots_adjust(hspace=0.2)

    # legend
    legend_idx = 1
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    #axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
    #axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
    #axs[legend_idx,-1].legend(fontsize=legend_size, ncol=1, loc="lower right", frameon=False)

    for row in range(len(layers)):
        if metric != "var":
            axs[row,0].set_xlim(0, n_top + 1)
            for col in range(3):
                # remove spines
                axs[row,col].spines['top'].set_visible(False); axs[row,col].spines['right'].set_visible(False) 
                # label ticks
                axs[row,col].tick_params(axis='both', labelsize=axis_size - 1) 
                # ylim
                axs[row,col].set_ylim(-0.05,1.05) 
                axs[row,col].set_yticks(np.arange(0,1.01,0.2))   
        else:
            #axs[row,0].set_xlim(0, 1000)
            pass

    #plt.tight_layout()
    
    if not display:
        plot_path = join(root_data, f"figure_ms/{fcn}_npc")
        if not os.path.isdir(plot_path): os.makedirs(plot_path)    
        epochs_str = [str(epoch) for epoch in epochs]
        epochs_str = "_".join(epochs_str)
        g100_str = [str(g100) for g100 in g100_ls]
        g100_str = "_".join(g100_str)
        epoch_str = [str(epoch) for epoch in epochs]
        epoch_str = "_".join(epoch_str)
        layer_str = [str(l) for l in layers]
        layer_str = "_".join(layer_str)
        if "fcn_grid" in data_path or "gaussian_data" not in data_path:
            file_full = f"{plot_path}/{fcn}_mnist_post={post}_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_micro_stats_{metric}.pdf"
        else:
            gaussian_data_setting = pd.read_csv(join(data_path,"gaussian_data_setting.csv"))
            X_dim, Y_classes, cluster_seed, assignment_and_noise_seed = gaussian_data_setting.loc[0,["X_dim", "Y_classes, ,noise_sigma", "cluster_seed,assignment_and_noise_seed"]]
            file_full = f"{plot_path}/{fcn}_gaussian_post={post}"
            file_full += f"_{X_dim}_{Y_classes}_{cluster_seed}_{assignment_and_noise_seed}"
            file_full += f"_layer={layer_str}_epoch={epoch_str}_g100={g100_str}_micro_stats_{metric}.pdf"
        print(f"Figure saved as {file_full}")
        plt.savefig(file_full, bbox_inches='tight')
    else:
        plt.show()

# relationship between multifractality of Jacobian eigenvectors and NPC (of the same layer)
def jac_npc(data_path, n_top=200, post=0, display=False, epochs=[0,650]):
    global jac_df_mean, xnames, ynames, metrics_all
    
    alpha100_ls = [120,200]
    #g100_ls = [25,100,300]
    g100_ls = [25]
    layers = range(9)
    reig = 1

    post = int(post)
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    n_top = int(n_top)
    display = literal_eval(display) if isinstance(display,str) else display    

    # metric pairs
    metric_names = ["jac_d2", "npc_ed", "npc_d2"]
    xnames = ["jac_d2", "jac_d2", "npc_ed",]
    ynames = ["npc_ed", "npc_d2", "npc_d2"]
    
    fig, axs = plt.subplots(len(epochs), 3, sharex=False,sharey=False,figsize=(12.5/3*len(epochs),3.1*1),constrained_layout=True)
    #axs = axs.flat
    for eidx, epoch_plot in enumerate(epochs):
        jac_D2_all =[[], []]
        ED_all = [[], []]
        npc_D2_all = [[], []]
        metrics_all = {}
        
        for aidx, alpha100 in enumerate(alpha100_ls):        
            for gidx, g100 in enumerate(g100_ls):            
                               
                alpha, g = int(alpha100)/100., int(g100)/100.
                # load nets and weights
                fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
                L = int(fcn[2:])

                # Jacobian metrics
                extension_name = f"alpha{alpha100}_g{g100}_ipidx0_ep{epoch_plot}.txt"  
                dq_path = join(root_data, f"geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}")              
                jac_df_mean = np.loadtxt(f"{dq_path}/dqmean_{extension_name}")                

                # NPC metrics
                ED_means = np.load(join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_r", f"ED_means_{epoch_plot}.npy"))

                for lidx, l in enumerate(range(L-1)):
                    # only includes preactivation             
                    npc_d2s = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"D2_{l}_{epoch_plot}.npy"))

                    jac_D2_all[aidx].append(jac_df_mean[-1,l+1])
                    ED_all[aidx].append(ED_means[0,l])
                    npc_D2_all[aidx].append(npc_d2s[:n_top].mean())
                
        metrics_all["jac_d2"] = jac_D2_all
        metrics_all["npc_ed"] = ED_all
        metrics_all["npc_d2"] = npc_D2_all
                    
        for aidx, alpha100 in enumerate(alpha100_ls):
            for metric_pair in range(len(xnames)):
                xname, yname = xnames[metric_pair], ynames[metric_pair]
                axs[eidx,metric_pair].scatter(metrics_all[xname][aidx], metrics_all[yname][aidx], c=c_ls[aidx], s=6)
                axs[eidx,metric_pair].set_xlabel(xname)
                axs[eidx,metric_pair].set_ylabel(yname)

    if display:
        plt.show()
        

# examines the decay of the ordered covariance matrix eigenvalues
def cov_spectrum(data_path, post=0, epochs=[0,650], display=False):
    """

    powerlaw fit of the covariance spectrum (log-log)

    """    
    
    post = int(post)
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    display = literal_eval(display) if isinstance(display,str) else display
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]

    fig, axs = plt.subplots(len(epochs), len(g100_ls), sharex = True,sharey=True,figsize=(12.5,3.1*2),constrained_layout=True)

    for gidx, g100 in enumerate(g100_ls):
        alpha_m_ls = []
        good = 0           
        for eidx, epoch_plot in enumerate(epochs):
            mstyle = marker_ls[0] if epoch_plot == 0 else marker_ls[1]
            for aidx, alpha100 in enumerate(alpha100_ls):    
                coefs = []           
                # Extract numeric arguments.
                alpha, g = int(alpha100)/100., int(g100)/100.
                # load nets and weights
                fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
                L = int(fcn[2:])
                layers = list(range(L - 1))
                for lidx, l in enumerate(layers):
                    eigvals = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"npc-eigvals_{l}_{epoch_plot}.npy"))

                    # add powerlaw fit to the log-log plot of the covariance eigenspectrum density
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression().fit(np.log(np.arange(1, len(eigvals)+1))[0:100].reshape(-1,1), np.log(eigvals)[0:100].reshape(-1,1))
                    coefs.append( -reg.coef_[0] )

                axs[eidx,gidx].plot([1,2,3,4,5], [1,2,3,4,5], linewidth=lwidth,linestyle='--',
                                 alpha=0.5, c = 'k') 
                axs[eidx,gidx].scatter([1 + 2/784]*len(coefs), coefs, marker=mstyle,
                                    alpha=1, c = c_ls[aidx]) 

    # adjust gaps between subplots
    plt.subplots_adjust(hspace=0.2)

    # legend
    legend_idx = 1
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    #axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
    #axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
    #axs[legend_idx,-1].legend(fontsize=legend_size, ncol=1, loc="lower right", frameon=False)

    """
    for row in range(len(layers)):
        if metric != "var":
            axs[row,0].set_xlim(0, n_top + 1)
            for col in range(3):
                # remove spines
                axs[row,col].spines['top'].set_visible(False); axs[row,col].spines['right'].set_visible(False) 
                # label ticks
                axs[row,col].tick_params(axis='both', labelsize=axis_size - 1) 
                # ylim
                axs[row,col].set_ylim(-0.05,1.05) 
                axs[row,col].set_yticks(np.arange(0,1.01,0.2))   
        else:
            #axs[row,0].set_xlim(0, 1000)
            pass
    """

    #plt.tight_layout()
    
    if not display:
        plot_path = join(root_data, f"figure_ms/{fcn}_npc")
        if not os.path.isdir(plot_path): os.makedirs(plot_path)    
        epochs_str = [str(epoch) for epoch in epochs]
        epochs_str = "_".join(epochs_str)
        g100_str = [str(g100) for g100 in g100_ls]
        g100_str = "_".join(g100_str)
        epoch_str = [str(epoch) for epoch in epochs]
        epoch_str = "_".join(epoch_str)
        layer_str = [str(l) for l in layers]
        layer_str = "_".join(layer_str)
        if "fcn_grid" in data_path or "gaussian_data" not in data_path:
            file_full = f"{plot_path}/{fcn}_mnist_post={post}_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_cov_spec_pl.pdf"
        else:
            gaussian_data_setting = pd.read_csv(join(data_path,"gaussian_data_setting.csv"))
            X_dim, Y_classes, cluster_seed, assignment_and_noise_seed = gaussian_data_setting.loc[0,["X_dim", "Y_classes, ,noise_sigma", "cluster_seed,assignment_and_noise_seed"]]
            file_full = f"{plot_path}/{fcn}_gaussian_post={post}"
            file_full += f"_{X_dim}_{Y_classes}_{cluster_seed}_{assignment_and_noise_seed}"
            file_full += f"_layer={layer_str}_epoch={epoch_str}_g100={g100_str}_cov_spec_pl.pdf"
        print(f"Figure saved as {file_full}")
        plt.savefig(file_full, bbox_inches='tight')
    else:
        plt.show()


def class_separation_plot(data_path, post, display=False, epochs=[650]):
    global axs, depth, depth_idxs

    from NetPortal.models import ModelFactory
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    display = literal_eval(display) if isinstance(display,str) else display 
    
    post = int(post)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    display = literal_eval(display) if isinstance(display,str) else display
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]

    fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100_ls[0], g100_ls[0])
    L = int(fcn[2:])    
    if post == 0:
        depth = L
        depth_idxs = list(range(0,2*L - 1,2))    
    elif post == 1:
        depth = L - 1
        depth_idxs = list(range(1,2*L - 1,2))
    else:
        depth = 2*L - 1
        depth_idxs = list(range(0,2*L - 1))
    nrows = int(np.ceil(np.sqrt(len(depth_idxs))))
    ncols = nrows

    # create save dir
    if not display:
        plot_path = join(root_data, f"figure_ms/{fcn}_class_separation")
        if not os.path.isdir(plot_path): os.makedirs(plot_path)   

    # class separation        
    if post == 0:
        ax_limit = 250        
    elif post == 1:
        ax_limit = 15
    else:
        ax_limit = 50
    # schematic figure of PCA
    selected_target_idxs = list(range(10))
    #selected_target_idxs = [1,9]
    for pca_epoch in epochs:
        for gidx, g100 in enumerate(g100_ls):
            for aidx, alpha100 in enumerate(alpha100_ls): 
                fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(12.5/3*ncols,3.1*nrows), 
                                        subplot_kw=dict(projection='3d'), constrained_layout=True)    
                axs = axs.flat
                fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)  
                for depth_iidx, depth_selected in enumerate(depth_idxs): 
                    # load PCA
                    
                    target_indices = np.load(join(data_path, net_folder, "X_pca", "target_indices.npy"), allow_pickle=True)                          
                    X_pca = np.load(join(data_path, net_folder, "X_pca", f"npc-depth={depth_selected}-epoch={pca_epoch}.npy"), allow_pickle=True)

                    # group in classes
                    for iidx, cl_idx in enumerate(selected_target_idxs):
                        #sample_indices = target_indices[cl_idx]
                        plot_points = 1000
                        sample_indices = list(target_indices[cl_idx][0:plot_points]) + [False] * (60000 - plot_points)
                        axs[depth_iidx].scatter(X_pca[sample_indices,0], X_pca[sample_indices,1], X_pca[sample_indices,2],
                                                c=c_ls_targets[iidx], s=1.5, alpha=0.3)   

                    axs[depth_iidx].set_xlim([-ax_limit,ax_limit]); axs[depth_iidx].set_ylim([-ax_limit,ax_limit]); axs[depth_iidx].set_zlim([-ax_limit,ax_limit])
                    axs[depth_iidx].set_xticks([-ax_limit,0,ax_limit]); axs[depth_iidx].set_xticklabels([-ax_limit,0,ax_limit])  
                    axs[depth_iidx].set_yticks([-ax_limit,0,ax_limit]); axs[depth_iidx].set_yticklabels([-ax_limit,0,ax_limit])
                    axs[depth_iidx].set_zticks([-ax_limit,0,ax_limit]); axs[depth_iidx].set_zticklabels([-ax_limit,0,ax_limit])
                        
                plt.suptitle(f"({alpha100}, {g100}), epoch = {pca_epoch}")
                if display:
                    plt.show()
                else:
                    file_full = f"{plot_path}/{fcn}-mnist-post={post}-alpha100={alpha100}-g100={g100}-epoch={pca_epoch}-class_separation.pdf"
                    print(f"Figure saved as {file_full}")
                    plt.savefig(file_full, bbox_inches='tight')
                    


def class_separation_final_layer(data_path, display=False, epochs=[650]):
    global trainloader, gaussian_data_kwargs, eigvecs, hidden_layer, targets_indices, X_pca
    global centers, cluster_class_label
    global net, hidden_N

    import random
    from NetPortal.models import ModelFactory
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    display = literal_eval(display) if isinstance(display,str) else display
    
    #alpha100_ls = [100,150,200]
    alpha100_ls = [120, 200]
    g100_ls = [25,100,300]

    if "fcn_grid" in data_path or "gaussian_data" not in data_path:
        # load MNIST
        image_type = 'mnist'
        batch_size = 60000
        trainloader = get_dataset(image_type, batch_size)
    else:
        image_type = "gaussian"
        gaussian_data_setting = pd.read_csv(join(data_path, "gaussian_data_setting.csv"))
        gaussian_data_kwargs = {}
        for param_name in gaussian_data_setting.columns:
            gaussian_data_kwargs[param_name] = gaussian_data_setting.loc[0,param_name]
        batch_size = int(gaussian_data_kwargs["num_train"])  # full batch
        trainloader, centers, cluster_class_label = get_dataset(image_type, batch_size, **gaussian_data_kwargs)

    trainloader = next(iter(trainloader))
    targets = trainloader[1].unique()    
    selected_targets_indices = [6,9]
    #selected_targets_indices = random.sample(range(len(targets)), 2)
    #selected_targets_indices = list(range(len(targets)))
    targets_indices = []
    # group in classes
    for cl_idx, cl in enumerate(targets):
        #targets_indices.append( torch.where(trainloader[1]==cl) )
        targets_indices.append( trainloader[1] == cl)

    # number of PCs
    n_components = 2
    figsize = (12.5/3*len(g100_ls), 3.1*len(alpha100_ls))                       
    epoch_plot = 650
    if n_components == 2:
        fig, axs = plt.subplots(len(alpha100_ls), len(g100_ls), sharex=True,sharey=True,figsize=figsize,constrained_layout=True)
    elif n_components == 3:
        fig, axs = plt.subplots(len(alpha100_ls), len(g100_ls), figsize=figsize, subplot_kw=dict(projection='3d') ,constrained_layout=True)

    for aidx, alpha100 in enumerate(alpha100_ls):
        for gidx, g100 in enumerate(g100_ls):
            # Network setting
            alpha, g = int(alpha100)/100., int(g100)/100.
            # load nets and weights
            fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
            L = int(fcn[2:])    
            hidden_N = [784]*L + [len(targets)]

            # load nets and weights
            with_bias = False
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch_plot,
                      "activation": 'tanh', "with_bias": with_bias,
                      "architecture": 'fc'}
            net = ModelFactory(**kwargs)

            with torch.no_grad():
                # to PCA for input first
                hidden_layer = trainloader[0]
                for lidx in range(len(net.sequential) - 2):
                #for lidx in range(4):
                    hidden_layer = net.sequential[lidx](hidden_layer)   
                 
                # only for the second final layer
                hidden_layer_centered = hidden_layer - hidden_layer.mean(0)
                pca = PCA(n_components)
                pca.fit(hidden_layer_centered.detach().numpy())
                X_pca = pca.fit_transform(hidden_layer_centered.detach().numpy())
                
            # group in classes
            for cl_idx in selected_targets_indices:
                #indices = targets_indices[cl_idx][:min(X_pca.shape[0], hidden_layer.shape[0])]
                indices = targets_indices[cl_idx]
                if n_components == 2:
                    axs[aidx, gidx].scatter(X_pca[indices,0], X_pca[indices,1], c=c_ls_targets[cl_idx], label=f"{targets[cl_idx]}", s=5)
                elif n_components == 3:
                    axs[aidx, gidx].scatter(X_pca[indices,0], X_pca[indices,1], X_pca[indices,2], c=c_ls_targets[cl_idx], label=f"{targets[cl_idx]}", s=5)
                               
    axs[0,0].legend()
    if display:
        plt.show()


"""
def npc_angle(l, post=0, epochs=[0,650]):

    ### microscopic statistics of the covariance matrix ###   

    global npc_angles, selected_angles, alpha, g

    l, post = int(l), int(post)
    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls)])

    fig, axs = plt.subplots(2, 3, sharex = True,sharey=False,figsize=(12.5,3.1*2),constrained_layout=True)    
    
    n_top = 20
    for gidx, g100 in enumerate(g100_ls):
        alpha_m_ls = []
        good = 0           
        for epoch_plot in epochs:
            for aidx, alpha100 in enumerate(alpha100_ls):               
                # Extract numeric arguments.
                alpha, g = int(alpha100)/100., int(g100)/100.
                # load nets and weights
                net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  

                npc_angles = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"eigvec-angles_{l}_{epoch_plot}.npy"))
                #N_angles = (npc_angles.shape[0]**2 - npc_angles.shape[0])/2
                selected_angles = npc_angles[0,1:]
                # flatten angles in upper trangular matrix
                for eidx in range(1,npc_angles.shape[0]):
                    selected_angles = np.hstack([selected_angles, npc_angles[eidx,eidx+1:]])
                print(f"No. of angles: {len(selected_angles)}")
                    
                lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                axs[0,gidx].hist(selected_angles, color = c_ls[aidx]) 

    # legend
    legend_idx = 1
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
    #axs[legend_idx,-1].legend(fontsize=legend_size, ncol=1, loc="lower right", frameon=False)

    for row in range(2):
        axs[row,0].set_xlim(-2*np.pi, 2*np.pi)
        for col in range(3):
            # remove spines
            axs[row,col].spines['top'].set_visible(False); axs[row,col].spines['right'].set_visible(False) 
            # label ticks
            axs[row,col].tick_params(axis='both', labelsize=axis_size - 1) 
            # ylim
            #axs[row,col].set_ylim(-0.05,1.05) 
            #axs[row,col].set_yticks(np.arange(0,1.01,0.2))   

    #plt.tight_layout()
    plot_path = join(root_data, f"figure_ms/{fcn}_npc")
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    epochs_str = [str(epoch) for epoch in epochs]
    epochs_str = "_".join(epochs_str)
    g100_str = [str(g100) for g100 in g100_ls]
    g100_str = "_".join(g100_str)
    epoch_str = [str(epoch) for epoch in epochs]
    epoch_str = "_".join(epoch_str)
    #plt.savefig(f"{plot_path}/{fcn}_mnist_layer={l}_epoch={epoch_str}_g100={g100_str}_npc_angle.pdf", bbox_inches='tight')
    plt.show()
"""

# plots either D_2 and ED_mean w.r.t. the depth (single metric)
def metric_vs_depth(data_path, metric, post=1, epochs=[0,650], method="batches", display=False):
    global net_folder, ED_means, epoch_plot, L
    
    assert metric == "ED" or metric == "D2", "metric does not exist!"
    assert method == "batches" or metric == "fullbatch", "method does not exist!"
    post = int(post)
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs
    display = literal_eval(display) if isinstance(display,str) else display
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]

    for gidx in range(len(g100_ls)):
        g100 = g100_ls[gidx]
        alpha_m_ls = []
        good = 0

        # set up plot
        fig = plt.figure(figsize=(figw, figh))
        fig.set_size_inches(figw, figh)

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')

        #if gidx == 0:
        #    plt.ylabel(r'$D_2$', fontsize=label_size)
        #if gidx != 0:
        #plt.xlabel(r'$l$', fontsize=label_size)

        plt.tick_params(bottom=True, top=False, left=True, right=False)
        plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

        plt.tick_params(axis="x", direction="out", labelsize=tick_size)
        plt.tick_params(axis="y", direction="out", labelsize=tick_size)                    

        for epoch_plot in epochs:
            for aidx in range(len(alpha100_ls)):
                alpha100 = alpha100_ls[aidx]
                
                # Extract numeric arguments.
                alpha, g = int(alpha100)/100., int(g100)/100.
                # load nets and weights
                fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
                L = int(fcn[2:])

                metric_means = np.load(join(data_path, net_folder, f"ed-dq-{method}_{post_dict[post]}_r", f"{metric}_means_{epoch_plot}.npy"))
                metric_stds = np.load(join(data_path, net_folder, f"ed-dq-{method}_{post_dict[post]}_r", f"{metric}_stds_{epoch_plot}.npy"))

                """
                try:
                    ED_means = np.load(join(data_path, net_folder, f"ed-dq-{method}_{post_dict[post]}_r", f"ED_means_{epoch_plot}.npy"))

                    #print((alpha100,g100))
                    #print(net_path)
                    alpha_m_ls.append((alpha, g))
                    good += 1     

                except (FileNotFoundError, OSError) as error:
                    # use the following to keep track of what to re-run
                    #print((alpha100, g100))
                    print(net_folder)
                """

                # only includes preactivation    
                L = len(metric_means[0])  

                if epoch_plot == 0:  
                    plt.plot(np.arange(1, L+1), metric_means[0],linewidth=lwidth,linestyle=linestyle_ls[0],
                             alpha=1, c = c_ls[aidx])
                else:
                    plt.plot(np.arange(1, L+1), metric_means[0],linewidth=lwidth,linestyle=linestyle_ls[1],
                             alpha=1, c = c_ls[aidx])

                # standard deviation
                plt.fill_between(np.arange(1, L+1), metric_means[0] - metric_stds[0], metric_means[0] + metric_stds[0], color = c_ls[aidx], alpha=0.2) 
                
                print(f"Epoch {epoch_plot}")     

        if gidx == 2 and metric == "ED":
            plt.plot([], [] ,linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
            plt.plot([], [] ,linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
            plt.plot([], [],linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
            plt.plot([], [],linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
            plt.legend(fontsize=legend_size, ncol=1, loc="upper left", frameon=False)

        plt.xlim(1, L)
        plt.xticks(list(range(2,L+1,2)))
        if metric == 'D2':
            plt.ylim(0,1) 
            plt.yticks(np.arange(0,1.01,0.2))

        plt.tight_layout()

        if not display:
            plot_path = join(root_data, f"figure_ms/{fcn}_npc")
            if not os.path.isdir(plot_path): os.makedirs(plot_path)    
            epochs_str = [str(epoch) for epoch in epochs]
            epochs_str = "_".join(epochs_str)

            if "fcn_grid" in data_path or "gaussian_data" not in data_path:
                file_full = f"{plot_path}/{fcn}_mnist_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
            else:
                gaussian_data_setting = pd.read_csv(join(data_path,"gaussian_data_setting.csv"))
                X_dim, Y_classes, cluster_seed, assignment_and_noise_seed = gaussian_data_setting.loc[0,["X_dim", "Y_classes, ,noise_sigma", "cluster_seed,assignment_and_noise_seed"]]
                file_full = f"{plot_path}/{fcn}_gaussian"
                file_full += f"_{X_dim}_{Y_classes}_{cluster_seed}_{assignment_and_noise_seed}"
                file_full += f"_epoch={epochs_str}_g100={g100}_{method}_{metric.lower()}-vs-depth.pdf"
            print(f"Figure saved as {file_full}")
            plt.savefig(file_full, bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
    
