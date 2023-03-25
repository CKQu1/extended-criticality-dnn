import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
from ast import literal_eval
from os.path import join
from tqdm import tqdm

sys.path.append(os.getcwd())

from path_names import root_data, model_log, id_to_path, get_model_id, get_alpha_g

# plot settings
plt.rcParams["font.family"] = "sans-serif"     # set plot font globally
#plt.switch_backend('agg')
global c_ls, figw, figh, axw, axh
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
#c_ls = list(mcl.TABLEAU_COLORS.keys())
c_ls = ["darkblue", "darkred"]
linestyle_ls = ["--", "-"]
marker_ls = ["o", "."]
figw, figh = 9.5, 7.142
axw, axh = figw * 0.7, figh * 0.7

global post_dict, reig_dict
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

def metrics_vs_depth(post=0, epochs=[0,650]):
    """

    Average ED (across minibatches) and 
    D_2 of full batch image (averaged across top ED PCs) vs depth

    """

    post = int(post)
    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]
    #metric_ls = ["D2", "ED"]
    metric_ls = ["ED", "D2", "eigvals"]

    # ED means for each layer (needed to compute the average top X PCs for D_2)
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls), 10])

    #fig, axs = plt.subplots(len(metric_ls), len(g100_ls), sharex = True,sharey=False,figsize=(9.5/2*3,7.142/4*3))
    fig, axs = plt.subplots(len(metric_ls), len(g100_ls), sharex = True,sharey=False,figsize=(12.5,3.1*len(metric_ls)),constrained_layout=True)    
    
    for midx, metric_name in enumerate(metric_ls):
        for gidx, g100 in enumerate(g100_ls):
            alpha_m_ls = []
            good = 0           
            # remove spines
            axs[midx,gidx].spines['top'].set_visible(False); axs[midx,gidx].spines['right'].set_visible(False) 
            # label ticks
            axs[midx,gidx].tick_params(axis='both', labelsize=axis_size - 1)   
            for epoch_plot in epochs:
                for aidx, alpha100 in enumerate(alpha100_ls):               
                    # Extract numeric arguments.
                    alpha, g = int(alpha100)/100., int(g100)/100.
                    # load nets and weights
                    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  

                    lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                    if metric_name == "ED":
                        metric_means = np.load(join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_r", f"ED_means_{epoch_plot}.npy"))
                        metric_stds = np.load(join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_r", f"ED_stds_{epoch_plot}.npy"))
                        # only includes preactivation    
                        L = len(metric_means[0])    
                        # save ED later
                        ED_all[aidx, gidx, :] = metric_means[0]
                        # mean
                        axs[0,gidx].plot(np.arange(1, L+1), metric_means[0],linewidth=lwidth,linestyle=lstyle,
                                 alpha=1, c = c_ls[aidx])
                        # standard deviation
                        axs[0,gidx].fill_between(np.arange(1, L+1), metric_means[0] - metric_stds[0], metric_means[0] + metric_stds[0], color = c_ls[aidx], alpha=0.2) 

                        # leaving extra space for labels
                        #if gidx == len(g100_ls) - 1:
                        #    axs[0,gidx].set_ylim(0,200)

                    elif metric_name == "D2":
                        D2_mean = []  # D2's corresponding to the eigenvector of the largest eigenvalue of the covariance matrix  
                        D2_std = []
                        for l in range(L):                        
                            metric_data = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"D2_{l}_{epoch_plot}.npy"))
                            #n_top = round(ED_all[aidx, gidx, l])    # top PCs of the mean ED of that layer
                            n_top = 5
                            D2_mean.append(metric_data[:n_top].mean())
                            D2_std.append(metric_data[:n_top].std())

                        D2_mean, D2_std = np.array(D2_mean), np.array(D2_std)
                        axs[1,gidx].plot(np.arange(1, L+1), D2_mean,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 

                        # standard deviation
                        axs[1,gidx].fill_between(np.arange(1, L+1), D2_mean - D2_std, D2_mean + D2_std, color = c_ls[aidx], alpha=0.2) 

                    elif metric_name == "eigvals":
                        var_ls = []
                        for l in range(L):
                            eigvals = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"npc-eigvals_{l}_{epoch_plot}.npy"))
                            #n_top = round(ED_all[aidx, gidx, l])    # variance explained by top PCs
                            n_top = 5
                            var_ls.append(eigvals[:n_top].sum()/eigvals.sum())
                        axs[2,gidx].plot(np.arange(1, L+1), var_ls,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 

        print(f"Metric {metric_name}")     

    # adjust gaps between subplots
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.3)

    # legend
    legend_idx = 0
    selected_col = 0
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[0], c='k', label="Before training")
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, linestyle=linestyle_ls[1], c='k', label="After training")   
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
    axs[legend_idx,selected_col].plot([], [], linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
    axs[legend_idx,selected_col].legend(fontsize=legend_size, ncol=2, loc="lower left", bbox_to_anchor=[0, 1.25],
                              frameon=True)

    for idx in range(2):
        axs[idx,0].set_xlim(1, L)
        axs[idx,0].set_xticks(list(range(1,L+1)))
        xtick_ls = []
        for num in range(1,L+1):
            if num % 2 == 1:
                xtick_ls.append(str(num))
            else:
                xtick_ls.append('')
        axs[idx,0].set_xticklabels(xtick_ls)
    for row in [1,2]:
        for col in range(3):
            axs[row,col].set_ylim(-0.05,1.05) 
            axs[row,col].set_yticks(np.arange(0,1.01,0.2)) 
    axs[1,0].set_yticks(np.arange(0,1.01,0.2))

    #fig.tight_layout(h_pad=2, w_pad=2)
    plot_path = join(root_data, f"figure_ms/{fcn}_npc")
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    epochs_str = [str(epoch) for epoch in epochs]
    epochs_str = "_".join(epochs_str)
    g100_str = [str(g100) for g100 in g100_ls]
    g100_str = "_".join(g100_str)
    metric_str = "_".join(metric_ls)

    file_full = f"{plot_path}/{fcn}_mnist_epoch={epochs[0]}_{epochs[1]}_g100={g100_str}_{metric_str}-vs-depth.pdf"
    print(f"Figure saved as {file_full}")
    plt.savefig(file_full, bbox_inches='tight')
    #plt.show()

def micro_stats(metric, n_top=200, post=0, epochs=[0,650]):
    """

    microscopic statistics of the covariance matrix    

    """    
    assert metric in ["var", "var_cumsum", "d2"], "metric not in list"
    
    post = int(post)
    n_top = int(n_top)
    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")
    
    alpha100_ls = [120,200]
    g100_ls = [25,100,300]
    layers = [0,3,6]
    ED_all = np.zeros([len(alpha100_ls), len(g100_ls)])

    fig, axs = plt.subplots(len(layers), 3, sharex = True,sharey=False,figsize=(12.5,3.1*len(layers)),constrained_layout=True)    
    
    for lidx, l in enumerate(layers):
        for gidx, g100 in enumerate(g100_ls):
            alpha_m_ls = []
            good = 0           
            for epoch_plot in epochs:
                for aidx, alpha100 in enumerate(alpha100_ls):               
                    # Extract numeric arguments.
                    alpha, g = int(alpha100)/100., int(g100)/100.
                    # load nets and weights
                    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  

                    ED_means = np.load(join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_r", f"ED_means_{epoch_plot}.npy"))
                    # only includes preactivation
                    ED_all[aidx, gidx] = ED_means[0,l]                   
                    D2s = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"D2_{l}_{epoch_plot}.npy"))

                    eigvals = np.load(join(data_path, net_folder, f"ed-dq-fullbatch_{post_dict[post]}_r", f"npc-eigvals_{l}_{epoch_plot}.npy"))
                    var_percentage = eigvals/eigvals.sum()
                    var_cum = np.cumsum(var_percentage)

                    lstyle = linestyle_ls[0] if epoch_plot == 0 else linestyle_ls[1]
                    if metric == "var_cumsum":
                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1)[:n_top], var_cum[:n_top],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                    elif metric == "d2":
                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1)[:n_top], D2s[:n_top],linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
                    elif metric == "var":
                        axs[lidx,gidx].plot(np.arange(1, len(eigvals)+1), eigvals,linewidth=lwidth,linestyle=lstyle,
                                         alpha=1, c = c_ls[aidx]) 
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
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[0], label=rf"$\alpha$ = {alpha100_ls[0]/100}")
    axs[legend_idx,-1].plot([], [], linewidth=lwidth, c = c_ls[1], label=rf"$\alpha$ = {alpha100_ls[1]/100}")    
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
    plt.savefig(f"{plot_path}/{fcn}_mnist_layer={layer_str}_epoch={epoch_str}_g100={g100_str}_micro_stats_{metric}.pdf", bbox_inches='tight')
    #plt.show()

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

# plots either D_2 and ED_mean w.r.t. the depth
def metric_vs_depth(metric, post=1, epochs=[0,650], method="batches"):
    global net_folder, ED_means, epoch_plot, L
    
    assert metric == "ED" or metric == "D2", "metric does not exist!"
    post = int(post)
    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")
    
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
                net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  

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
        plot_path = join(root_data, f"figure_ms/{fcn}_npc")
        if not os.path.isdir(plot_path): os.makedirs(plot_path)    
        epochs_str = [str(epoch) for epoch in epochs]
        epochs_str = "_".join(epochs_str)
        plt.savefig(f"{plot_path}/{fcn}_mnist_epoch={epochs_str}_g100={g100}_{method}_{metric.lower()}-vs-depth.pdf", bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
    
