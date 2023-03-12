import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import sys
sys.path.append(f'{os.getcwd()}')
import path_names
from os.path import join
from path_names import root_data, id_to_path, model_log
from path_names import get_model_id, get_alpha_g

# colorbar setting
cm_lib = "sns"
cmaps = []
if cm_lib == "plt":
    for cm_type in ['Spectral', 'gist_stern', 'RdGy']:
        cmaps.append(plt.cm.get_cmap(cm_type))
elif cm_lib == "sns":
    for cm_type in ['Spectral', 'RdBu', 'rocket_r']:
        cmaps.append(sns.color_palette(cm_type, as_cmap=True))
# custom
else:
    for cm_type in [[500,200], [600,100], [20000,25]]:
        cmaps.append(sns.diverging_palette(cm_type[0], cm_type[1], as_cmap=True))

interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally
# plot settings
tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8
# phase diagram setting
alpha100_ls = list(range(100,201,10))
g100_ls = list(range(25,301,25))
alpha100_incre = alpha100_ls[1] - alpha100_ls[0]
g100_incre = g100_ls[1] - g100_ls[0]

def load_accloss(net_path, acc_type):
    acc_loss = pd.read_csv(f"{net_path}/acc_loss")
    metrics = acc_loss.iloc[:,[0,1]]*100 if acc_type == "train" else acc_loss.iloc[:,[2,3]]*100

    return metrics

# 1 by 3 figure including phase transition for accuracy, loss and stopping epoch (train/test)
def cnn_accloss_phase(net_type="alexnet", acc_type="train", acc_threshold=70, epoch=500):
    acc_threshold, epoch = int(acc_threshold), int(epoch)

    if net_type == "alexnet":
        net_path = join(root_data, "trained_cnns", "alexnet_htcw_ufw_tanh")
    #elif net_type == "resnet14_ht":
    #    net_path = join(root_data, "trained_cnns", "resnet14_ht_new")
    #net_ls = [net[0] for net in os.walk(net_path) if "epochs=" in net[0]]
    net_ls = [join(net_path, dirname) for dirname in next(os.walk(net_path))[1]]
    epoch_last = int(net_ls[0][net_ls[0].index("epochs=")+7:])
    print(net_path)
    print(f"Total of networks: {len(net_ls)}.")

    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # with text
    #fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(9.5/2*3,7.142/2 - 0.1))     # without text
    fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3,7.142/2 - 0.1))     # without text
    axs = axs.flat

    # plot points which computations where executed
    a_cross, m_cross = np.meshgrid(alpha100_ls, g100_ls)
    for i in range(len(axs)):
        #axs[i].plot(a_cross/100, g_cross/100, c='k', linestyle='None',marker='.',markersize=5)
        pass

    title_ls = [f'{acc_type[0].upper() + acc_type[1:]} accuracy', 'Earliest epoch reaching' + '\n' + f'{acc_type} acc. threshold ']
    for i in range(len(axs)):
        #axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
        #if i == 0: axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)
        #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)
        #axs[i].text(-0.1, 1.2, f'({string.ascii_lowercase[i]})', transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')   # fontweight='bold'
        # setting ticks
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
    for i in range(len(net_ls)):
        net_path = net_ls[i]
        model_id = get_model_id(net_path)
        try:
            model_info = model_log(model_id)
            #alpha100, g100 = model_info.loc[model_info.index[0],['alpha100','g100']]
            alpha100, g100 = get_alpha_g(net_path)
            alpha, g = int(alpha100)/100, int(g100)/100
            metrics = load_accloss(net_path, acc_type)

            good_ls = [x for x,y in enumerate(metrics.iloc[:,1]) if y > acc_threshold]
            acc = metrics.iloc[epoch,1]
            loss = metrics.iloc[epoch,0]      

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
    for ls in [acc_ls, loss_ls, early_ls]:
        cmap_bd.append([np.percentile(ls,5),np.percentile(ls,95)])

    mesh_ls = [acc_mesh, loss_mesh, early_mesh]
    for pidx in range(3):
        plot = axs[pidx].imshow(mesh_ls[pidx],extent=[min(alpha100_ls)/100,max(alpha100_ls)/100,min(g100_ls)/100,max(g100_ls)/100], 
                                vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=cmaps[pidx], 
                                interpolation=interp, aspect='auto')
        cbar = plt.colorbar(plot,ax=axs[pidx])
        cbar.ax.tick_params(labelsize=tick_size)

    plt.tight_layout()
    #plt.show()

    #net_type = "alexnet"
    net_type = model_info.loc[model_info.index[0],'net_type']
    dataname = model_info.loc[model_info.index[0],'name']
    plt.savefig(f"{root_data}/figure_ms/{net_type}_{dataname}_{acc_type}_epoch={epoch}_grid_all.pdf", bbox_inches='tight')

    print(f"Loaded networks: {good}")
    print(f"Existing networks folders: {len(net_ls)}")
    print(f"Successfully data initialisations loaded: {len(alpha_m_ls)}")

# 1 by 3 plot for either accuracy or loss (alexnet)
def cnn_pure_phase(acc_type="train", epoch_ls=[10,50,200]):   

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
            axis.tick_params(axis='both',labelsize=tick_size)

            axis.set_xticks(xticks)
            axis.set_yticks(yticks)

    plt.tight_layout()
    #plt.show()

    #net_type = "alexnet"
    net_type = model_info.loc[model_info.index[0],'net_type']
    dataname = model_info.loc[model_info.index[0],'name']
    epoch_str = [str(epoch) for epoch in epoch_ls]
    epoch_str = "_".join(epoch_str)
    fname = f"{root_data}/figure_ms/{net_type}_{dataname}_{acc_type}_accloss_epoch={epoch_str}.pdf"
    print(fname)
    plt.savefig(fname, bbox_inches='tight')

    print(f"Loaded networks: {good}")
    print(f"Existing networks folders: {len(net_ls)}")
    print(f"Successfully data initialisations loaded: {len(alpha_m_ls)}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
