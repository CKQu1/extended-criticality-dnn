import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import string
import sys
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from os.path import join
from path_names import root_data

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
# plot settings
import pubplot as ppt
plt.rc('font', **ppt.pub_font)
plt.rcParams.update(ppt.plot_sizes(False))

tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

trained_path = join(root_data, "trained_cnns", "alexnets_nomomentum")
net_ls = [ f.path for f in os.scandir(trained_path) if f.is_dir() and "epochs=100" in f.path ]
epoch_last = int(net_ls[0][net_ls[0].index("epochs=")+7:])
epoch = 100
print(trained_path)
print(f"Total of networks: {len(net_ls)}.")

def id_to_path(model_id, path):   # tick
    for subfolder in os.walk(path):
        if model_id in subfolder[0]:
            model_path = subfolder[0]
            break
    assert 'model_path' in locals(), "model_id does not exist!"

    return model_path


def model_log(model_id):    # tick
    log_book = pd.read_csv(f"{root_data}/net_log.csv")
    #print(f"length {len(log_book)}")     # delete
    #if model_id in log_book['model_id'].item():
    if model_id in list(log_book['model_id']):
        model_info = log_book.loc[log_book['model_id']==model_id]        
    else:
        #if f"{id_to_path(model_id)}"
        print("Update model_log() function in tamsd_analysis/tamsd.py!")
        model_info = None
    #model_info = log_book.iloc[0,log_book['model_id']==model_id]
    return model_info

# transfer full path to model_id
def get_model_id(full_path):
    str_ls = full_path.split('/')[-1].split('_')
    for s in str_ls:
        if len(s) == 36:
            return s    

# get (alpha,g) when the pair is not saved
def get_alpha_g(full_path):
    str_ls = full_path.split('/')
    str_alpha_g = str_ls[-1].split("_")
    if str_alpha_g[2].isnumeric() and str_alpha_g[3].isnumeric():
        return (int(str_alpha_g[2]), int(str_alpha_g[3]))
    else:
        return (int(str_alpha_g[1]), int(str_alpha_g[2]))

fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))
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
    axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
    if i == 0: axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)
    #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)
    axs[i].text(-0.1, 1.2, f'({string.ascii_lowercase[i]})', transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')   # fontweight='bold'
    # setting ticks
    axs[i].tick_params(bottom=True, top=False, left=True, right=False)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")
    axs[i].tick_params(axis='both')

good = 0
alpha_m_ls = []
prototype_snr_ls = []
exemplar_snr_ls = []

ms = np.arange(2,50,2)
ks = np.arange(1,100,2)
midx = 1
kidx = -1
print(f"m = {ms[midx]}, D = {ks[kidx]}")

prototype_snr_mesh, exemplar_snr_mesh = np.zeros((len(g100_ls),len(alpha100_ls))), np.zeros((len(g100_ls),len(alpha100_ls)))
subfolder_name = "proto_vs_NN_experiments/navg=500_bs=10_K=64_P=50_N=2048"
for i in range(0,len(net_ls)):
    net_path = net_ls[i]
    model_id = get_model_id(net_path)
    model_info = model_log(model_id)
    #alpha100, g100 = model_info.loc[model_info.index[0],['alpha100','g100']]
    alpha100, g100 = get_alpha_g(net_path)

    try:
        alpha, g = int(alpha100)/100, int(g100)/100

        # load SNR
        data_path = join(net_path, subfolder_name)
        means = np.load(join(data_path, 'NNmeans.npy'))
        stds = np.load(join(data_path, 'NNstds.npy'))
        means_proto = np.load(join(data_path, "proto_means.npy"))
        stds_proto = np.load(join(data_path, "proto_stds.npy"))
        SNR = means/stds
        SNR_proto = means_proto/stds_proto

        prototype_snr_ls.append(SNR_proto[kidx,midx])
        exemplar_snr_ls.append(SNR[kidx,midx])

        alpha_m_ls.append((alpha, g))
        good += 1     

    except (FileNotFoundError, OSError) as error:
        # use the following to keep track of what to re-run

        prototype_snr_ls.append(None)
        exemplar_snr_ls.append(None)

        print((alpha100, g100))
        #print(net_path)

    x_loc = int((max(g100_ls) - g100) / g100_incre)
    y_loc = int((alpha100 - min(alpha100_ls)) / alpha100_incre)
    prototype_snr_mesh[x_loc,y_loc] = prototype_snr_ls[i]
    exemplar_snr_mesh[x_loc,y_loc] = exemplar_snr_ls[i]

#assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls), print("alpha_m_ls and acc_ls have different lengths!")

cmap_bd = [[np.percentile(prototype_snr_ls,5),np.percentile(prototype_snr_ls,95)], [np.percentile(exemplar_snr_ls,5),np.percentile(exemplar_snr_ls,95)]]

mesh_ls = [prototype_snr_mesh, exemplar_snr_mesh]
for pidx in range(2):
    plot = axs[pidx].imshow(mesh_ls[pidx],extent=[min(alpha100_ls)/100,max(alpha100_ls)/100,min(g100_ls)/100,max(g100_ls)/100], 
                            vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=plt.cm.get_cmap(cm_type), 
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
