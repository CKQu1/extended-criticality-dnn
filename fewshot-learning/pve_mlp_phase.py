import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio

import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
# colorbar scheme
from matplotlib.cm import coolwarm
from ast import literal_eval
from os.path import join

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
path = f"{main_path}/fcn_grid/{fcn}_grid"
#path = f"{main_path}/fcn_grid/{fcn}_grid128"
#net_ls = [net[0] for net in os.walk(path)]
net_ls = [ f.path for f in os.scandir(path) if f.is_dir() and "epoch650" in f.path ]

print(path)
#print(net_ls)

# epoch network was trained till
epoch_last = 650

# phase boundaries

# new version
bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
boundaries = []
bd_path = "/project/phys_DL/phasediagram"
for i in range(1,92,10):
#for i in range(1,102,10):
    boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))

# plot phase transition 
tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.15))
#fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))
axs = [ax1, ax2]

# plot boundaries for each axs
for i in range(len(axs)):
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')

    # new version
    for j in range(len(boundaries)):
        bd = boundaries[j]
        axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k--')  


mult_lower = 0.25
mult_upper = 3
#mult_upper = 4
mult_N = int((mult_upper - mult_lower)/0.25 + 1)
mult_grid = np.linspace(mult_lower,mult_upper,mult_N)
mult_incre = round(mult_grid[1] - mult_grid[0],2)

alpha_lower = 1
alpha_upper = 2
alpha_N = int((alpha_upper - alpha_lower)/0.1 + 1)
alpha_grid = np.linspace(alpha_lower,alpha_upper,alpha_N)
alpha_incre = round(alpha_grid[1] - alpha_grid[0],1)

# plot points which computations where executed
a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)

for i in range(len(axs)):
    #axs[i].plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=5)

    # major ticks
    axs[i].tick_params(bottom=True, top=True, left=True, right=True)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")

    # minor ticks
    #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    #axs[i].yaxis.set_minor_locator(AutoMinorLocator())

label_ls = ['(a)', '(b)']
title_ls = ["Prototype SNR", "Exemplar SNR"]
for i in range(len(axs)):
    # ticks
    axs[i].tick_params(axis='both',labelsize=tick_size)
    
    #axs[i].set_yticks(mult_grid)
    
    if i == 0:
        axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

    #axs[i].axes.xaxis.set_ticklabels([])   # delete

    #if i == 2 or i == 3:
    
    #axs[i].set_xticks(alpha_grid)
    #axs[i].set_xlim(0.975,2.025)

    #if i == 2 or i == 3:
    #axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
    axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

    # adding labels
    label = label_ls[i] 
    axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')

    # setting ticks
    axs[i].tick_params(bottom=True, top=False, left=True, right=False)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")

#axs[r].set_title(f"Epoch {epoch}", fontsize=axis_size)

alpha_m_ls = []
prototype_snr_ls = []
exemplar_snr_ls = []
good = 0

# training example number and dimension (always double check)
ms = np.arange(2,50,2)
ks = np.arange(1,100,2)
midx = -1
kidx = 1
print(f"m = {ms[midx]}, D = {ks[kidx]}")

#subfolder_name = "proto_vs_NN_experiments/navg=500_omniglot=test_ep=650_bs=10_K=128_P=50_N=784"
fn = "navg=500_omniglot=test_ep=0_bs=10_K=128_P=50_N=784"
subfolder_name = join("proto_vs_NN_experiments", fn)
for i in range(len(net_ls)):

    net_path = net_ls[i]
    #acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
    
    net_params = sio.loadmat(net_path + "/net_params_all.mat")            
    alpha = list(net_params['net_init_params'][0][0])[1][0][0]
    m = list(net_params['net_init_params'][0][0])[2][0][0]

    try:
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

        alpha_m_ls.append((alpha,m))
        good += 1     

    except (FileNotFoundError, OSError) as error:
        # use the following to keep track of what to re-run

        #print(net_path)
        print("error")
        print((alpha,m))
        #print("\n")

cmap_bd = [[np.percentile(prototype_snr_ls,5),np.percentile(prototype_snr_ls,95)], [np.percentile(exemplar_snr_ls,5),np.percentile(exemplar_snr_ls,95)]]

#assert len(alpha_m_ls) == len(test_acc_ls) and len(alpha_m_ls) == len(ratio_ls)
assert len(alpha_m_ls) == len(prototype_snr_ls) and len(alpha_m_ls) == len(exemplar_snr_ls)

prototype_snr_mesh = np.zeros((mult_N,alpha_N))
exemplar_snr_mesh = np.zeros((mult_N,alpha_N))
for t in range(len(alpha_m_ls)):
    
    alpha,mult = alpha_m_ls[t]
    #x_loc = int((alpha - alpha_lower)/alpha_incre)   
    #y_loc = int((mult - mult_lower)/mult_incre)   

    x_loc = int(round((mult_upper - mult) / mult_incre))
    y_loc = int(round((alpha - alpha_lower) / alpha_incre))

    prototype_snr_mesh[x_loc,y_loc] = prototype_snr_ls[t]
    exemplar_snr_mesh[x_loc,y_loc] = exemplar_snr_ls[t]    

# plot results
prototype_snr_plot = axs[0].imshow(prototype_snr_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                              vmin=cmap_bd[0][0], vmax=cmap_bd[0][1], cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
cbar = plt.colorbar(prototype_snr_plot,ax=axs[0])
cbar.ax.tick_params(labelsize=tick_size)

exemplar_snr_plot = axs[1].imshow(exemplar_snr_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                           vmin=cmap_bd[1][0], vmax=cmap_bd[1][1], cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
cbar = plt.colorbar(exemplar_snr_plot,ax=axs[1])
cbar.ax.tick_params(labelsize=tick_size)

print(f"Good: {good}")

plt.tight_layout()
plt.show()

fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
#plt.savefig(f"{fig1_path}/{net_type}_grid_snr_m={ms[midx]}_D={ks[kidx]}_{fn}_pve.pdf", bbox_inches='tight')

print("Figure 1")
print("\n")
print(len(net_ls))
print(len(alpha_m_ls))

