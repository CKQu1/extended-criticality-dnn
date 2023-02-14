import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import string
import sys
sys.path.append(f'{os.getcwd()}')
import path_names
from os.path import join
from path_names import root_data, id_to_path, model_log
from path_names import get_model_id, get_alpha_g

# colorbar
cm_type1 = 'CMRmap'
cm_type2 = 'gist_stern'
cm_types = [cm_type1, cm_type2]
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

net_path = join(root_data, "trained_cnns", "alexnet_htcw_ufw_tanh")
net_ls = [net[0] for net in os.walk(net_path) if "epochs=" in net[0]]
epoch_last = int(net_ls[0][net_ls[0].index("epochs=")+7:])
print(net_path)
print(f"Total of networks: {len(net_ls)}.")

tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

#fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # with text
fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 - 0.1))     # with text
axs = [ax1, ax2]

alpha100_ls = sorted(list(set([int(get_alpha_g(net_ls[nidx])[0]) for nidx in range(len(net_ls))])))
g100_ls = sorted(list(set([int(get_alpha_g(net_ls[nidx])[1]) for nidx in range(len(net_ls))])))
alpha100_incre = alpha100_ls[1] - alpha100_ls[0]
g100_incre = g100_ls[1] - g100_ls[0]
# plot points which computations where executed
a_cross, m_cross = np.meshgrid(alpha100_ls, g100_ls)
for i in range(len(axs)):
    #axs[i].plot(a_cross/100, g_cross/100, c='k', linestyle='None',marker='.',markersize=5)
    pass

acc_type = "train"
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

acc_threshold = 70
#acc_threshold = 95
epoch = 200
good = 0
alpha_m_ls = []
acc_ls = []
early_ls = []

acc_mesh, early_mesh = np.zeros((len(g100_ls),len(alpha100_ls))), np.zeros((len(g100_ls),len(alpha100_ls)))
for i in range(len(net_ls)):
    net_path = net_ls[i]
    model_id = get_model_id(net_path)
    try:
        acc_loss = pd.read_csv(f"{net_path}/acc_loss")
        model_info = model_log(model_id)
        #alpha100, g100 = model_info.loc[model_info.index[0],['alpha100','g100']]
        alpha100, g100 = get_alpha_g(net_path)
        alpha, g = int(alpha100)/100, int(g100)/100

        metrics = acc_loss.iloc[:,[0,1]]*100 if acc_type == "train" else acc_loss.iloc[:,[2,3]]*100
        good_ls = [x for x,y in enumerate(metrics.iloc[:,1]) if y > acc_threshold]
        acc = metrics.iloc[epoch,1]      
        if len(good_ls) == 0:
            early_ls.append(epoch_last)
        else:
            early_ls.append( min(good_ls) )  

        alpha_m_ls.append((alpha, g))
        good += 1     

    except (FileNotFoundError, OSError) as error:
        # use the following to keep track of what to re-run

        early_ls.append(epoch_last)
        acc = 10
        
        #print((alpha100, g100))
        print(net_path)

    acc_ls.append(acc)
    x_loc = int((max(g100_ls) - g100) / g100_incre)
    y_loc = int((alpha100 - min(alpha100_ls)) / alpha100_incre)
    acc_mesh[x_loc,y_loc] = acc
    early_mesh[x_loc,y_loc] = early_ls[i]

# resnet14
# train
#cmap_bd = [[75, 100], [5, 50]]  # epoch 75
# test
#cmap_bd = [[65, 72], [5, 35]]  # epoch 200

# alexnet
# test
#cmap_bd = [[45, 58], [50, 150]]  # epoch 100
#cmap_bd = [[54, 61], [50, 150]]  # epoch 200
#cmap_bd = [[20, 56], [50, 150]]  # epoch 200
# train
#cmap_bd = [[80, 95], [100, 220]]  # epoch 200

#assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls), print("alpha_m_ls and acc_ls have different lengths!")

cmap_bd = [[np.percentile(acc_ls,5),np.percentile(acc_ls,95)], [np.percentile(early_ls,5),np.percentile(early_ls,95)]]

mesh_ls = [acc_mesh, early_mesh]
for pidx in range(2):
    plot = axs[pidx].imshow(mesh_ls[pidx],extent=[min(alpha100_ls)/100,max(alpha100_ls)/100,min(g100_ls)/100,max(g100_ls)/100], 
                            vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=plt.cm.get_cmap(cm_types[pidx]), 
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
