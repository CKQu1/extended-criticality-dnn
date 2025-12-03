import matplotlib.pyplot as plt
import numpy as np
import os, sys
import torch

from ast import literal_eval
from matplotlib.ticker import AutoMinorLocator
from numpy import linalg as la
from os.path import join

sys.path.append(os.getcwd())
from constants import DROOT
from UTILS.fig_utils import set_size
from UTILS.utils_dnn import IPR, D_q

plt.rcParams["font.family"] = "sans-serif"     # set plot font globally
#plt.switch_backend('agg')
global title_size, tick_size, label_size, axis_size, legend_size, axw, axh, figw, figh
title_size = 23.5 * 1.5
tick_size = 23.5 * 1.5
label_size = 23.5 * 1.5
axis_size = 23.5 * 1.5
legend_size = 23.5 * 1.5 - 10
figw, figh = 9.5, 7.142 - 0.25
axw, axh = figw * 0.7, figh * 0.7

global post_dict, reig_dict, c_ls
# pre- or post-activation Jacobian
post_dict = {0:'pre', 1:'post'}
# left or right eigenvectors
reig_dict = {0:'l', 1:'r'}
# color settings
#c_ls = ["tab:blue", "tab:orange"]
c_ls = ["darkblue", "darkred"]

# original dq_magnitude_epoch_plot.py
"""
Plots absolute value of principal eigenvector sites for one input
"""
def eigvec_magnitude(alpha100_ls = [120,200], g100 = 100, post=0, reig=1, inset=False):
    post, reig = int(post), int(reig)

    # re-adjusted
    figw, figh = 9.75, 7.142 - 0.25
    axw, axh = figw * 0.7, figh * 0.7
    lwidth = 4.5 if not inset else 6.5

    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    # main_path = join(DROOT, "trained_mlps")
    # path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    #dq_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"
    #data_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/{post_dict[post]}jac_layerwise"
    data_path = join(DROOT, f"geometry_data/{post_dict[post]}jac_layerwise")

    missing_data = []
    # in the future for ipidx might be needed
    # test first
    #for epoch in [0,1]:
    #    for layer in range(0,2):
    #for epoch in [0,1] + list(range(50,651,50)):   # all
    #for epoch in [0,650]:
    for epoch in [0]:
        #for layer in range(0,10):
        #lstyle = "--" if epoch == 0 else "-"
        lstyle = "-"  
        for f_idx in range(len(alpha100_ls)):
            #for f_idx in range(len(extension_names)):
                alpha100 = alpha100_ls[f_idx]
                extension_name = f"dw_alpha{alpha100}_g{g100}_ipidx0_epoch{epoch}"          
                # load DW's
                DW_all = torch.load(f"{data_path}/{extension_name}")
                DWs_shape = DW_all.shape
                for layer in [4]:

                    # set up figure
                    fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))     
                    set_size(axw, axh, ax) 

                    # ticks
                    #ax1.set_xticks(np.arange(0,2.05,0.5))

                    # set ticks
                    if inset:
                        xticks = [0,25,50,75,100]
                        yticks = np.arange(0,0.16,.05)
                        yticks = np.round(yticks,2)
                    else:
                        xticks = [0,200,400,600,800]
                        yticks = np.arange(0,1.01,0.2)
                        yticks = np.round(yticks,1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticks)

                    # label ticks
                    if inset:
                        ax.tick_params(axis='x', labelsize=(axis_size - 1)*1.5)
                        ax.tick_params(axis='y', labelsize=(axis_size - 1)*1.5)

                    else:
                        ax.tick_params(axis='x', labelsize=axis_size - 1)
                        ax.tick_params(axis='y', labelsize=axis_size - 1)

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)                        

                    # minor ticks
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                    ax.yaxis.set_minor_locator(AutoMinorLocator())

                    # ----- 1. Plot individual dqs -----
                    # Figure (a) and (b): plot the D_q vs q (for different alphas same g), chosen a priori

                    DW = DW_all[layer].numpy()

                    # left eigenvector
                    if reig == 0:
                        DW = DW.T

                    # this is also documented in dq_analysis/jac_fcn.py
                    if layer == DWs_shape[0]:
                        # old method
                        #DW = DW[:10,:]
                        #DW = DW.T @ DW  # final DW is 10 x 784
                        _, eigvals, eigvecs = la.svd(DW)
                    else:
                        eigvals, eigvecs = la.eig(DW)

                    print(f"Epoch {epoch} layer {layer}: {DW.shape}")
                    eigvals, eigvecs = la.eig(DW)
                    print(f"Max eigenvalue magnitude: {np.max(np.abs(eigvals))}.")
                    print(f"Max eigenvector site: {np.max(np.abs(eigvecs[:,np.argmax(np.abs(eigvals))]))}")

                    # x, y axis limit
                    ax.set_xlim(0,np.max(xticks))
                    ax.set_ylim(0,np.max(yticks))

                    # eigenvector corresponding to the eigenvalue with the largest magnitude
                    ax.plot(np.abs(eigvecs[:,np.argmax(np.abs(eigvals))]), c = c_ls[f_idx], linestyle=lstyle, linewidth=lwidth)
                    
                    #if f_idx == 1:
                    if not inset:
                        ax.set_xlabel('Site', fontsize=axis_size)
                        ax.set_ylabel('Magnitude', fontsize=axis_size)

                    #ax.set_title(f"Layer {layer+1}, Epoch {epoch}", fontsize=title_size)
                    #if epoch == 0:
                    #    ax.set_title(rf"$\alpha$ = {alpha100/100}", fontsize=title_size)

                    #ax.legend(fontsize = legend_size, frameon=False)
                    plt.tight_layout()
                    #plt.show()

                    #fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
                    fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
                    if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
                    # alleviate memory
                    fig_name = f"jac_eigvec_mag_{post_dict[post]}_{reig_dict[reig]}_alpha100={alpha100}_g100={g100}_l={layer}_epoch={epoch}"
                    print(f'Saved in {fig1_path} as {fig_name}')
                    if inset:
                        fig_name += "_inset"
                    plt.savefig(join(fig1_path, fig_name + ".pdf"), bbox_inches='tight')
                    plt.clf()
                    plt.close(fig)

                    #plt.show()

        print(f"Epoch {epoch} done!")

    # missing data due to simulation errors or jobs not submitted properly
    #np.savetxt(join(DROOT,"geometry_data/missing_data.txt"), np.array(missing_data), fmt='%s')


# original dq_single_epoch_plot.py
"""
Plots D_q vs q for one input
"""
def dq_vs_q(alpha100_ls = [120,200], g100 = 100, post=0, reig=1):
    post, reig = int(post), int(reig)

    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    main_path = join(DROOT, "trained_mlps")
    path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    dq_path = join(DROOT, f"geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}")

    q_folder_idx = 25
    missing_data = []
    # in the future for ipidx might be needed
    # test first
    #for epoch in [0,1]:
    #    for layer in range(0,2):
    #for epoch in [0,1] + list(range(50,651,50)):   # all
    for epoch in [0,650]:
        for layer in range(0,10):

            #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
            fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))
            set_size(axw, axh, ax)
            #axs = [ax1, ax2, ax3, ax4]
            #fig = plt.figure(figsize=(9.5,7.142))   
            lstyle = "--" if epoch == 0 else "-"   

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # set ticks
            ax.set_yticks(np.arange(0,2.1,0.5))
            ax.set_yticklabels(np.arange(0,2.1,0.5))

            # label ticks
            ax.tick_params(axis='x', labelsize=axis_size - 1)
            ax.tick_params(axis='y', labelsize=axis_size - 1)

            # minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ylim_lower = 1
            ylim_upper = 1
            for f_idx, alpha100 in enumerate(alpha100_ls):
                extension_name = f"alpha{alpha100}_g{g100}_ipidx0_ep{epoch}.txt"
                
                df_mean = np.loadtxt(f"{dq_path}/dqmean_{extension_name}")
                df_std = np.loadtxt(f"{dq_path}/dqstd_{extension_name}")
                qs = df_mean[:,0]

                dq_means = df_mean[:,layer + 1]
                dq_stds = df_std[:,layer + 1]
                lower = dq_means - dq_stds
                upper = dq_means + dq_stds
                
                # averages of dq's with error bars
                ax.plot(qs, dq_means, linewidth=3.5, linestyle=lstyle, alpha=1, c = c_ls[f_idx], label=rf"$\alpha$ = {round(alpha100/100,1)}")
                #ax.plot(qs, lower, linewidth=0.25, alpha=1, c = c_ls[f_idx])
                #ax.plot(qs, upper, linewidth=0.25, alpha=1, c = c_ls[f_idx])
                ax.fill_between(qs, lower, upper, color = c_ls[f_idx], alpha=0.2)

                ylim_lower = min(ylim_lower, min(lower))
                ylim_upper = max(ylim_upper, max(upper))

                #for q_idx in range(len(qs)):
                    #axs[f_idx].axvline(qs[q_idx], ymin=lower[q_idx]/1.1, ymax=upper[q_idx]/1.1, alpha=0.75)

            #ax.set_ylim(round(ylim_lower,1) - 0.05, round(ylim_upper,1) + 0.05)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(0,2)

            ax.set_xlabel(r'$q$', fontsize=axis_size)
            ax.set_ylabel(r'$D_q$', fontsize=axis_size)

            #ax.set_title(f"Layer {layer+1}, Epoch {epoch}", fontsize=title_size)

            #if epoch == 0:
            #    ax.legend(fontsize = legend_size, frameon=False)
            plt.tight_layout()
            #plt.show()

            #fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
            fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
            if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
            # alleviate memory
            plt.savefig(f"{fig1_path}/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')
            plt.clf()
            plt.close(fig)

        print(f"Epoch {epoch} done!")

    #np.savetxt("/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data", missing_data)
    #np.savetxt("/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/missing_data.txt", np.array(missing_data), fmt='%s')


# original d2-vs-depth.py
"""
Plots d2 (averaged across all eigenvectors) against network depth
"""
def d2_vs_depth(alpha100_ls=[120,200], g100=25, post=0, reig=1, appendix=False):

    post, reig = int(post), int(reig)
    appendix = literal_eval(appendix) if isinstance(appendix, str) else appendix
    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    main_path = join(DROOT, "trained_mlps")

    path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    dq_path = join(DROOT, f"geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}")

    missing_data = []
    # in the future for ipidx might be needed
    #depths = np.arange(9)     # not including the final layer since there are only 10 neurons
    depths = np.arange(10)
    #epochs = [0, 650]
    #epochs = [50]
    epochs = [1]
    #for epoch in [0,1] + list(range(50,651,50)):   # all
    for epoch in epochs:

        #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
        fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))
        set_size(axw, axh, ax)
        # linestyle
        lstyle = "--" if epoch == 0 else "-"
        # vertical line
        if g100 == 100:
            ax.axvline(x=5, c='dimgrey',linestyle=':',lw=2.5)     

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # set ticks
        ax.set_yticks(np.arange(0,2.1,0.5))
        ax.set_yticklabels(np.arange(0,2.1,0.5))

        # label ticks
        ax.tick_params(axis='x', labelsize=axis_size - 1)
        ax.tick_params(axis='y', labelsize=axis_size - 1)

        # minor ticks
        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

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
            ax.plot(depths+1, dq_mean_layer, linewidth=3.5, linestyle=lstyle, alpha=1, c = c_ls[f_idx])
            #ax.plot(depths+1, lower, linewidth=0.25, alpha=1, c = c_ls[f_idx])
            #ax.plot(depths+1, upper, linewidth=0.25, alpha=1, c = c_ls[f_idx])
            ax.fill_between(depths+1, lower, upper, color = c_ls[f_idx], alpha=0.2)

        # extra labels
        if epoch == 0 and g100 == 25:
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[0], label=rf"$\alpha$ = {round(alpha100_ls[0]/100,1)}")
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[1], label=rf"$\alpha$ = {round(alpha100_ls[1]/100,1)}")
            ax.plot([], [], linewidth=3.5, linestyle="--", c="k", label="Before training")
            ax.plot([], [], linewidth=3.5, linestyle="-", c="k", label="After training")

        ylim_lower = min(ylim_lower, min(lower))
        ylim_upper = max(ylim_upper, max(upper))
        #ax.set_ylim(round(ylim_lower,1) - 0.05, round(ylim_upper,1) + 0.05)
        ax.set_ylim(-0.1,1.1)
        ax.set_xlim(depths[0]+1,depths[-1]+1)
        ax.set_xticks(list(range(1,11)))
        xtick_ls = []
        for num in range(1,11):
            if num % 2 == 1:
                xtick_ls.append(str(num))
            else:
                xtick_ls.append('')
        ax.set_xticklabels(xtick_ls)

        if not appendix:
            ax.set_xlabel(r'Layer $l$', fontsize=axis_size)
            ax.set_ylabel(r'$D_2$', fontsize=axis_size)

        #ax.set_title(f"Epoch {epoch}", fontsize=title_size)

        # bbox_to_anchor=(-0.05, 1.2),
        #ax.legend(fontsize = legend_size, ncol=2, loc="lower left", 
        #          frameon=True)
        plt.tight_layout()

        fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
        # alleviate memory
        plt.savefig(f"{fig1_path}/jac_d2-vs-depth_{post_dict[post]}_{reig_dict[reig]}_g100={g100}_epochs={epochs}.pdf", bbox_inches='tight')
        #plt.clf()
        #plt.close(fig)
        #plt.show()

        #print(f"Epoch {epoch} layer {layer} done!")
        print(f"Epoch {epoch} done!")


"""
Plots d2 (averaged across all eigenvectors AND different inputs) against network depth
"""
def d2mean_vs_depth(alpha100_ls=[120, 200], g100=300, post=0, reig=1, appendix=False, display=False):
    global ms, stds

    post, reig = int(post), int(reig)
    appendix = literal_eval(appendix) if isinstance(appendix, str) else appendix
    display = literal_eval(display) if isinstance(display, str) else display
    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    main_path = join(DROOT, "trained_mlps")

    path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    trans_ls = [1, 0.5]
    dq_path = join(DROOT, f"geometry_data/d2_layerwise_navg=1000_{post_dict[post]}_{reig_dict[reig]}")

    missing_data = []
    # in the future for ipidx might be needed
    #depths = np.arange(9)     # not including the final layer since there are only 10 neurons
    depths = np.arange(10)
    l = 4
    #epochs = [0, 650]
    epochs = [0,1]
    for epoch in epochs:

        #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
        fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))
        set_size(axw, axh, ax)
        # linestyle
        lstyle = "--" if epoch == 0 else "-" 

        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)           
        # label ticks
        ax.tick_params(axis='x', labelsize=axis_size - 1)
        ax.tick_params(axis='y', labelsize=axis_size - 1)
        # set ticks
        ax.set_yticks(np.arange(0,2.1,0.5))
        ax.set_yticklabels(np.arange(0,2.1,0.5))

        # minor ticks
        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ylim_lower = 1
        ylim_upper = 1
        for f_idx, alpha100 in enumerate(alpha100_ls):
            extension_name = f"d2means-navg=1000-alpha{alpha100}-g{g100}-ep{epoch}.npy"            
            d2_means = np.load(join(dq_path, extension_name))
            ms = d2_means.mean(1)
            stds = d2_means.std(1)
            ax.plot(depths+1, ms, linewidth=3.5, linestyle=lstyle, alpha=1, c = c_ls[f_idx])
            ax.fill_between(depths+1, ms-stds, ms+stds, color = c_ls[f_idx], alpha=0.2)

        # extra labels
        if epoch == 0 and g100 == 25:
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[0], label=rf"$\alpha$ = {round(alpha100_ls[0]/100,1)}")
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[1], label=rf"$\alpha$ = {round(alpha100_ls[1]/100,1)}")
            ax.plot([], [], linewidth=3.5, linestyle="--", c="k", label="Before training")
            ax.plot([], [], linewidth=3.5, linestyle="-", c="k", label="After training")

        ax.set_ylim(-0.1,1.1)
        ax.set_xlim(depths[0]+1,depths[-1]+1)
        ax.set_xticks(list(range(1,11)))
        xtick_ls = []
        for num in range(1,11):
            if num % 2 == 1:
                xtick_ls.append(str(num))
            else:
                xtick_ls.append('')
        ax.set_xticklabels(xtick_ls)

        if not appendix:
            ax.set_xlabel(r'Layer $l$', fontsize=axis_size)
            ax.set_ylabel(r'$D_2$', fontsize=axis_size)

        plt.tight_layout()

        fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
        # alleviate memory
        if display:
            plt.show()
        else:
            plt.savefig(f"{fig1_path}/jac_d2mean-vs-depth_{post_dict[post]}_{reig_dict[reig]}_g100={g100}_epoch={epochs}.pdf", bbox_inches='tight')
        #plt.clf()
        #plt.close(fig)
        #plt.show()

        #print(f"Epoch {epoch} layer {layer} done!")
        print(f"Epoch {epoch} done!")


# distribution of mean d2 for different inputs
def d2_dist(alpha100_ls=[120,200], g100=100, post=0, reig=1, appendix=False, display=False):
    from scipy.stats import gaussian_kde

    post, reig = int(post), int(reig)
    appendix = literal_eval(appendix) if isinstance(appendix, str) else appendix
    display = literal_eval(display) if isinstance(display, str) else display
    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    main_path = join(DROOT, "trained_mlps")

    path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    trans_ls = [0.5,1]
    dq_path = join(DROOT, f"geometry_data/d2_layerwise_navg=1000_{post_dict[post]}_{reig_dict[reig]}")

    missing_data = []
    # in the future for ipidx might be needed
    #depths = np.arange(9)     # not including the final layer since there are only 10 neurons
    depths = np.arange(10)
    l = 4
    for eidx, epoch in enumerate([0,650]):

        #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
        fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))
        set_size(axw, axh, ax)
        # linestyle
        lstyle = "--" if epoch == 0 else "-" 

        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)           
        # label ticks
        ax.tick_params(axis='x', labelsize=axis_size - 1)
        ax.tick_params(axis='y', labelsize=axis_size - 1)

        # minor ticks
        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ylim_lower = 1
        ylim_upper = 1
        for f_idx, alpha100 in enumerate(alpha100_ls):
            extension_name = f"d2means-navg=1000-alpha{alpha100}-g{g100}-ep{epoch}.npy"            
            d2_means = np.load(join(dq_path, extension_name))
            #ax.hist(2_means, 1000, linewidth=3.5, linestyle=lstyle, alpha=1, c = c_ls[f_idx])
            ax.hist(d2_means[l,:], 50, alpha=trans_ls[eidx], color=c_ls[f_idx], density=True)
            # kde
            #density = gaussian_kde(d2_means[l,:])            
            #xs = np.linspace(d2_means[l,:].min(),d2_means[l,:].max(),1000)
            #xs = np.linspace(0,0.6,1000) if f_idx == 0 else np.linspace(0.8,1,1000)
            #ax.plot(xs, density(xs), linewidth=3.5, linestyle=lstyle, alpha=1, c = c_ls[f_idx] )

        # extra labels
        if epoch == 0 and g100 == 25:
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[0], label=rf"$\alpha$ = {round(alpha100_ls[0]/100,1)}")
            ax.plot([], [], linewidth=3.5, alpha=1, c = c_ls[1], label=rf"$\alpha$ = {round(alpha100_ls[1]/100,1)}")
            ax.plot([], [], linewidth=3.5, linestyle="--", c="k", label="Before training")
            ax.plot([], [], linewidth=3.5, linestyle="-", c="k", label="After training")

        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(0,200)
        #ax.set_yscale('log')

        if not appendix:
            ax.set_ylabel('Density', fontsize=axis_size)
            ax.set_xlabel(r'$D_2$', fontsize=axis_size)

        plt.tight_layout()

        fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
        # alleviate memory
        if display:
            plt.show()
        else:
            plt.savefig(f"{fig1_path}/jac_d2mean-dist_{post_dict[post]}_{reig_dict[reig]}_g100={g100}_epoch={epoch}.pdf", bbox_inches='tight')
        #plt.clf()
        #plt.close(fig)
        #plt.show()

        #print(f"Epoch {epoch} layer {layer} done!")
        print(f"Epoch {epoch} done!")



# original: d2-vs-eigvals.py
def d2_vs_eigvals(alpha100_ls = [120,200], g100 = 100, post=0, reig=1):

    post, reig = int(post), int(reig)
    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    #net_type = f"{fcn}_mnist_tanh_2"
    main_path = join(DROOT, "trained_mlps")
    path = f"{main_path}/fcn_grid/{fcn}_grid"

    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    #data_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/{post_dict[post]}jac_layerwise"
    data_path = join(DROOT, f"geometry_data/{post_dict[post]}jac_layerwise")

    trans_ls = np.linspace(0,1,len(alpha100_ls)+1)[::-1]
    max_mag = 0     # maximum magnitude of eigenvalues

    missing_data = []
    # in the future for ipidx might be needed
    #for epoch in [0,1] + list(range(50,651,50)):   # all
    for epoch in [0,650]:
        
        # set up figure
        fig, ax = plt.subplots(1, 1 ,figsize=(figw, figh))      
        set_size(axw, axh, ax)

        # ticks
        #ax1.set_xticks(np.arange(0,2.05,0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # label ticks
        ax.tick_params(axis='x', labelsize=axis_size - 1)
        ax.tick_params(axis='y', labelsize=axis_size - 1)

        # minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        for f_idx, alpha100 in enumerate(alpha100_ls):
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

        #ax.set_title(f"Layer {layer+1}, Epoch {epoch}", fontsize=title_size)

        #ax.legend(fontsize = legend_size, frameon=False)
        plt.tight_layout()

        #fig1_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots"
        fig1_path = join(DROOT, f"figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_plots")
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
        # alleviate memory
        plt.savefig(f"{fig1_path}/jac_d2-vs-eigval_{post_dict[post]}_{reig_dict[reig]}_alpha100={alpha100}_g100={g100}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')
        plt.clf()
        plt.close(fig)

        #plt.show()

        #print(f"Epoch {epoch} layer {layer} done!")
        print(f"Epoch {epoch} done!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

