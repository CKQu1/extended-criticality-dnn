#import cmcrameri as cmc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import sys
sys.path.append(f'{os.getcwd()}')

from ast import literal_eval
from os import makedirs
from os.path import join, isdir
from tqdm import tqdm
from constants import DROOT
from constants import id_to_path, model_log
from constants import get_model_id, get_alpha_g

# colorbar setting
cm_lib = "sns"
cmaps = []
if cm_lib == "plt":
    for cm_type in ['Spectral', 'gist_stern', 'RdGy']:
        cmaps.append(plt.cm.get_cmap(cm_type))
elif cm_lib == "sns":
    for cm_type in ['Spectral', 'RdBu', 'rocket_r']:
    #for cm_type in ['plasma', 'vlag', 'icefire']:
    #for cm_type in ['batlow', 'cividis', 'thermal']:
        #cmaps.append(sns.color_palette(cm_type, as_cmap=True))
        #cmaps.append(sns.color_palette(cm_type))
        cmaps.append(plt.cm.get_cmap(cm_type))
# custom
# else:
#     """
#     for cm_type in [[500,200], [600,100], [20000,25]]:
#         cmaps.append(sns.diverging_palette(cm_type[0], cm_type[1], as_cmap=True))
#     """
#     #cmaps = [cmc.cm.batlow, plt.cm.get_cmap("cividis"), plt.cm.get_cmap("vlag")]
#     cmaps = [cmc.cm.fes, cmc.cm.oleron, cmc.cm.bukavu]

#interp = "quadric"
interp = None
plt.rcParams["font.family"] = 'sans-serif'     # set plot font globally
# plot settings
tick_size = 14.5
label_size = 15.5
axis_size = 15.5
legend_size = 14
linewidth = 0.8
# phase diagram setting
# alpha100_ls = list(range(100,201,10))
# g100_ls = list(range(25,301,25))
alpha100_ls = list(range(100,201,5))
g100_ls = list(range(20,301,20))
alpha100_incre = alpha100_ls[1] - alpha100_ls[0]
g100_incre = g100_ls[1] - g100_ls[0]

alpha_lower, alpha_upper = int(alpha100_ls[0]/100), int(alpha100_ls[-1]/100)
mult_lower, mult_upper = g100_ls[0]/100, g100_ls[-1]/100
alpha_N, mult_N = len(alpha100_ls), len(g100_ls)

centers = [alpha_lower, alpha_upper, mult_lower, mult_upper]
dx, = np.diff(centers[:2])/(alpha_N-1)
dy, = -np.diff(centers[2:])/(mult_N-1)
extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

def load_accloss(net_path, acc_type):
    acc_loss = pd.read_csv(f"{net_path}/acc_loss")
    metrics = acc_loss.iloc[:,[0,1]]*100 if acc_type == "train" else acc_loss.iloc[:,[2,3]]*100

    return metrics


def get_net_ls(nets_dir, net_type):
    #global net_ls

    # if net_type == "alexnet":
    #     nets_dir = join(root_data, "trained_cnns", "alexnet_htcw_ufw_tanh")
    # elif net_type == "resnet14_ht":
    #     nets_dir = join(root_data, "trained_cnns", "resnet14_ht_new")
    # elif 'cnn' in net_type:
    #     nets_dir = join(root_data, 'trained_cnns', f'{net_type}_fc_default_mnist_sgd_epochs=50')

    net_ls = [join(nets_dir, dirname) for dirname in next(os.walk(nets_dir))[1] if net_type in dirname]
    return net_ls, nets_dir


# 1 by 3 figure including phase transition for accuracy, loss and stopping epoch (train/test)
"""
python -i transition_plot_functions/cnn_acc_phase.py accloss_phase .droot/wide-cnns/cnn7_seed\=0/ cnn7 test 85 99
"""
def accloss_phase(nets_dir, net_type="alexnet", acc_type="train", acc_threshold=70, epoch=500, display=False):
    global model_info, net_ls, alpha100, g100, metrics
    global acc_mesh, loss_mesh, early_mesh
    global acc_ls, loss_ls, early_ls

    acc_threshold, epoch = int(acc_threshold), int(epoch)
    display = literal_eval(display) if isinstance(display,str) else display

    net_ls, nets_dir = get_net_ls(nets_dir, net_type)
    if 'cnn' in net_type:
        #epoch_last = int(net_ls[0][net_ls[0].index("epochs="):])
        epoch_last = 50
    else:
        epoch_last = int(net_ls[0][net_ls[0].index("epochs=")+7:])
    print(f'nets_dir: {nets_dir}')
    print(f"Total of networks: {len(net_ls)}.")

    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # with text
    #fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(9.5/2*3,7.142/2 - 0.1))     # without text
    #fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3,7.142/2 - 0.1))     # without text
    fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3+0.45,7.142/2 - 0.1))     # without text
    axs = axs.flat

    # plot points which computations where executed
    a_cross, m_cross = np.meshgrid(alpha100_ls, g100_ls)
    for i in range(len(axs)):
        #axs[i].plot(a_cross/100, g_cross/100, c='k', linestyle='None',marker='.',markersize=5)
        pass

    xticks = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    xtick_ls = []
    for xidx, xtick in enumerate(xticks):
        if xidx % 2 == 0:
            xtick_ls.append(str(xtick))
        else:
            xtick_ls.append('')
    title_ls = [f'{acc_type[0].upper() + acc_type[1:]} accuracy', 'Earliest epoch reaching' + '\n' + f'{acc_type} acc. threshold ']
    for i in range(len(axs)):
        #axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
        #if i == 0: axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)
        #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)
        #axs[i].text(-0.1, 1.2, f'({string.ascii_lowercase[i]})', transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')   # fontweight='bold'
        # setting ticks
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xtick_ls)

        axs[i].tick_params(bottom=True, top=False, left=True, right=False)
        axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        axs[i].tick_params(axis="x", direction="out")
        axs[i].tick_params(axis="y", direction="out")
        axs[i].tick_params(axis='both',labelsize=tick_size)

    #acc_threshold = 95
    good = 0
    alpha_m_ls = []
    acc_ls = []
    loss_ls = []
    early_ls = []

    acc_mesh, loss_mesh, early_mesh = np.zeros((len(g100_ls),len(alpha100_ls))), np.zeros((len(g100_ls),len(alpha100_ls))), np.zeros((len(g100_ls),len(alpha100_ls)))
    acc_mesh[:], loss_mesh[:], early_mesh[:] = np.nan, np.nan, np.nan
    for i in range(len(net_ls)):
        net_path = net_ls[i]
        model_id = get_model_id(net_path)
        try:            
            #alpha100, g100 = model_info.loc[model_info.index[0],['alpha100','g100']]
            if 'cnn' not in net_path:
                alpha100, g100 = get_alpha_g(net_path)
                model_info = model_log(model_id)
            else:
                s = net_path.split('/')[-1].split('_')
                alpha100, g100 = int(s[1]), int(s[2])
            alpha, g = int(alpha100)/100, int(g100)/100

            if 'cnn' not in net_path:
                metrics = load_accloss(net_path, acc_type)
            else:
                metrics = pd.read_csv(join(net_path, 'acc_loss'))
                metrics = metrics.iloc[:,1:]
                metrics.iloc[:,1] = metrics.iloc[:,1] * 100
                metrics.iloc[:,3] = metrics.iloc[:,3] * 100
            
            if acc_type == 'train':
                acc = metrics.iloc[epoch,1]
                loss = metrics.iloc[epoch,0]      
                good_ls = [x for x,y in enumerate(metrics.iloc[:,1]) if y > acc_threshold]
            elif acc_type == 'test':
                acc = metrics.iloc[epoch,3]
                loss = metrics.iloc[epoch,2]   
                good_ls = [x for x,y in enumerate(metrics.iloc[:,3]) if y > acc_threshold]               

            epoch_stop = epoch_last if len(good_ls) == 0 else min(good_ls)

            alpha_m_ls.append((alpha, g))

            acc_ls.append(acc)
            early_ls.append(epoch_stop)
            loss_ls.append(loss)

            x_loc = int((max(g100_ls) - g100) / g100_incre)
            y_loc = int((alpha100 - min(alpha100_ls)) / alpha100_incre)
            acc_mesh[x_loc,y_loc] = acc
            loss_mesh[x_loc,y_loc] = loss
            early_mesh[x_loc,y_loc] = epoch_stop

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            #early_ls.append(epoch_last)
            #acc = 10
            
            #print((alpha100, g100))
            print(net_path)

    #assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls), print("alpha_m_ls and acc_ls have different lengths!")

    cmap_bd = []        
    p_lowers, p_uppers = [15,1,10], [75,90,90]
    for ii, ls in enumerate([acc_ls, loss_ls, early_ls]):        
        cmap_bd.append([np.percentile(ls,p_lowers[ii]),np.percentile(ls,p_uppers[ii])]) 

    mesh_ls = [acc_mesh, loss_mesh, early_mesh]
    for pidx in range(3):
        #extent=[min(alpha100_ls)/100,max(alpha100_ls)/100,min(g100_ls)/100,max(g100_ls)/100]
        plot = axs[pidx].imshow(mesh_ls[pidx],extent=extent, 
                                vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=cmaps[pidx], 
                                interpolation=interp, aspect='auto',
                                origin='upper')

        cbar = plt.colorbar(plot,ax=axs[pidx])
        cbar.ax.tick_params(labelsize=tick_size)
        """
        if pidx == 1:
            cbar.set_ticks(list(range(20,71,10)))
            cbar.set_ticklabels(list(range(20,71,10)))
        if pidx == 2:
            cbar.set_ticks(list(range(200,451,50)))
            cbar.set_ticklabels(list(range(200,451,50)))        
        cbar.ax.tick_params(labelsize=tick_size)
        """

    plt.tight_layout()
    #plt.show()

    #net_type = "alexnet"
    #net_type = model_info.loc[model_info.index[0],'net_type']
    #dataname = model_info.loc[model_info.index[0],'name']
    dataname = "cifar10" if 'alexnet' in net_type else 'mnist'
    #fig1_path = "/project/PDLAI/project2_data/figure_ms/trained_dnn_performance"
    fig1_path = "/project/phys_DL/extended-criticality-dnn/.droot/figure_ms"
    if not isdir(fig1_path):  makedirs(fig1_path)  
    fname =  join(fig1_path, f"{net_type}_{dataname}_{acc_type}_epoch={epoch}_grid_all.pdf")    
    plt.savefig(fname, bbox_inches='tight')

    print(f"Loaded networks: {good}")
    print(f"Existing networks folders: {len(net_ls)}")
    print(f"Successfully data initialisations loaded: {len(alpha_m_ls)}")
    print(f'Figure saved as {fname}')

# 1 by 3 plot for either accuracy or loss (alexnet)
def pure_phase(acc_type="train", epoch_ls=[10,50,200], display=False):   
    display = literal_eval(display) if isinstance(display,str) else display

    metrics = ["acc", "loss"]

    net_path = join(root_data, "trained_cnns", "alexnet_htcw_ufw_tanh")
    net_ls = [net[0] for net in os.walk(net_path) if "epochs=" in net[0]]
    epoch_last = int(net_ls[0][net_ls[0].index("epochs=")+7:])

    nrows, ncols = len(metrics), len(epoch_ls)
    #fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3,7.142/2 - 0.1))     # without text
    fig, axs = fig, axs = plt.subplots(nrows, ncols,sharex=True,sharey=True,figsize=(8.4/2*3 + 1,7.142/2 * len(metrics)))     # without text

    xticks = [1.0,1.2,1.4,1.6,1.8,2.0]
    yticks = [0.5,1.0,1.5,2.0,2.5,3.0]

    good = 0
    alpha_m_ls = []

    acc_mesh = np.zeros((len(epoch_ls),len(g100_ls),len(alpha100_ls)))
    loss_mesh = np.zeros((len(epoch_ls),len(g100_ls),len(alpha100_ls)))
    for i in range(len(net_ls)):
        for eidx, epoch in enumerate(epoch_ls):
            net_path = net_ls[i]
            model_id = get_model_id(net_path)
            try:
                acc_loss = pd.read_csv(f"{net_path}/acc_loss")
                model_info = model_log(model_id)
                alpha100, g100 = get_alpha_g(net_path)
                alpha, g = int(alpha100)/100, int(g100)/100

                metrics_data = acc_loss.iloc[:,[0,1]]*100 if acc_type == "train" else acc_loss.iloc[:,[2,3]]*100
                acc = metrics_data.iloc[epoch,1]       
                loss= metrics_data.iloc[epoch,0]
                alpha_m_ls.append((alpha, g))
                good += 1     

            except (FileNotFoundError, OSError) as error:
                # use the following to keep track of what to re-run
                acc = np.nan
                loss = np.nan
                #print((alpha100, g100))
                print(net_path)

            x_loc = int((max(g100_ls) - g100) / g100_incre)
            y_loc = int((alpha100 - min(alpha100_ls)) / alpha100_incre)
            acc_mesh[eidx,x_loc,y_loc] = acc
            loss_mesh[eidx,x_loc,y_loc] = loss

    alpha_min, alpha_max = min(alpha100_ls)/100, max(alpha100_ls)/100
    g_min, g_max = min(g100_ls)/100, max(g100_ls)/100
    cmap_bds = np.zeros((2, len(epoch_ls), 2))
    for eidx, epoch in enumerate(epoch_ls):
        for metric_idx, metric_name in enumerate(["acc_mesh", "loss_mesh"]):
            axis = axs[metric_idx, eidx]
            metric_ls = locals()[metric_name][eidx,:,:].flatten()
            cmap_bds[metric_idx,eidx,:] = [np.percentile(metric_ls,5),np.percentile(metric_ls,95)]
            #assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

            # plot results
            metric_plot = axis.imshow(locals()[metric_name][eidx,:,:],
                                               extent=[alpha_min,alpha_max,g_min,g_max], 
                                               vmin=cmap_bds[metric_idx,eidx,0], vmax=cmap_bds[metric_idx,eidx,1], cmap=cmaps[metric_idx], 
                                               interpolation=interp, aspect='auto')
            cbar = plt.colorbar(metric_plot,ax=axis)
            #l, u = np.ceil(cmap_bds[metric_idx,eidx,0]), np.floor(cmap_bds[metric_idx,eidx,1])
            #cbar.ax.set_yticks([l, u])
            cbar.ax.tick_params(labelsize=tick_size)

            # axis ticks
            axis.tick_params(axis='both',labelsize=label_size)

            axis.set_xticks(xticks)
            axis.set_yticks(yticks)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    #plt.show()

    #net_type = "alexnet"
    net_type = model_info.loc[model_info.index[0],'net_type']
    dataname = model_info.loc[model_info.index[0],'name']
    epoch_str = [str(epoch) for epoch in epoch_ls]
    epoch_str = "_".join(epoch_str)
    fig1_path = "/project/PDLAI/project2_data/figure_ms/trained_dnn_performance"
    if not isdir(fig1_path):  makedirs(fig1_path)  
    fname =  join(fig1_path, f"{net_type}_{dataname}_{acc_type}_accloss_epoch={epoch_str}.pdf")
    print(fname)
    plt.savefig(fname, bbox_inches='tight')

    print(f"Loaded networks: {good}")
    print(f"Existing networks folders: {len(net_ls)}")
    print(f"Successfully data initialisations loaded: {len(alpha_m_ls)}")


def epochs_all(alpha100s, g100s, acc_type, net_type='alexnet', display=False):
    """
    Plots the all epoch accuracy/loss for specified (\alpha, \sigma_w)
    """

    global net_ls, dataname, epoch_last, fcn, net_paths, alpha, g, metrics_all

    assert acc_type in ["test", "train"], "acc_type does not exist!"
    alpha100s = literal_eval(alpha100s); g100s = literal_eval(g100s)
    display = literal_eval(display) if isinstance(display,str) else display

    net_ls, nets_dir = get_net_ls(net_type)
    print(f'nets_dir: {nets_dir}')
    print(f"Total networks {len(net_ls)}")

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    nrows, ncols = 1, 2
    fig, axs = plt.subplots(nrows,ncols,sharex = True,sharey=True,figsize=(8.4/2*ncols+0.45,3.471*nrows))     # without text
    axs = axs.flat

    metrics_all = []
    title_ls = ['Test accuracy', 'Test loss' + '\n' + 'test acc. threshold ']
    for aidx, alpha100 in enumerate(alpha100s):
        axis = axs[aidx]
        for g100 in g100s:
            alpha, g = alpha100/100, g100/100

            net_paths = [net_path for net_path in net_ls if f'{net_type}_{alpha100}_{g100}_' in net_path]
            net_path = net_paths[0]
            metrics = load_accloss(net_path, acc_type)
            metrics_all.append(metrics)


            #axis.plot(metrics.iloc[:,1], label=rf'$\sigma_w$ = {g}')
            axis.plot(metrics.iloc[:501,1], label=rf'$\sigma_w$ = {g}')

            #axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)             

            # # ticks
            # axis.tick_params(axis='both',labelsize=tick_size)
            # # major ticks
            # axis.set_xticks(xticks)
            # axis.set_xticklabels(xtick_ls)

            #if i == 2 or i == 3:
            #axis.set_xlabel(r'$\alpha$', fontsize=axis_size)
            #axis.set_title(f"{title_ls[i]}", fontsize=axis_size)

            # # setting ticks
            # axs[i].tick_params(bottom=True, top=False, left=True, right=False)
            # axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

            # axs[i].tick_params(axis="x", direction="out")
            # axs[i].tick_params(axis="y", direction="out")

            # minor ticks
            #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            #axs[i].yaxis.set_minor_locator(AutoMinorLocator())

    axs[0].legend(frameon=False)
    #axs[1].set_yticklabels([])

    plt.tight_layout()
    if display:
        plt.show()
    else:
        fig1_path = "/project/PDLAI/project2_data/figure_ms/trained_dnn_performance"
        if not isdir(fig1_path):  makedirs(fig1_path)
        file_name = f'{net_type}_cifar10_sgd_{acc_type}_alpha100={alpha100s}_g100={g100s}.pdf'
        plt.savefig(join(fig1_path, file_name), bbox_inches='tight')

        print(f"Figure saved: {join(fig1_path, file_name)}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
