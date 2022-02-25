import argparse
import math
import numpy as np
import os
import pandas as pd
import random
import scipy.io as sio
import seaborn as sns
import sys
import time
import torch

from numpy import dot
from scipy.stats import levy_stable

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from nporch.np_forward import wm_np_sim
from nporch.geometry import kappa, gbasis, hidden_fixed, vel_acc

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm 

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import axes3d

# colorbar scheme
from matplotlib.cm import coolwarm

t0 = time.time()

# colorbar
cm = cm.get_cmap('plasma')
# font size
tick_size = 16.5
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

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

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
axs = [ax1, ax2, ax3, ax4]

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"

# phase boundaries
boundaries = []
boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None))
for i in list(range(1,10,2)):
    boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_{i}_line_2.csv", header=None))
bound1 = boundaries[0]

# plot boundaries for each axs
for i in range(len(axs)):
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')
    for j in range(2,len(boundaries)):
        bd = boundaries[j]
        axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')

# plot points which computations where executed
a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)
for i in range(len(axs)):
    axs[i].plot(a_cross, m_cross, 'kx')

label_ls = ['(a)', '(b)', '(c)', '(d)']
for i in range(len(axs)):
    #axs[i].spines['top'].set_visible(False)
    #axs[i].spines['right'].set_visible(False)

    # ticks
    #axs[i].tick_params(axis='both',labelsize=tick_size)

    # ticks
    #if i == 0 or i == 2:
    #axs[i].set_xticks(np.linspace(100,600,6))

    #axs[i].tick_params(axis='both',labelsize=tick_size)
    
    #axs[i].set_yticks(mult_grid)
    #axs[i].set_ylim(0,3.25)
    
    axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

    #if i == 2 or i == 3:
    
    #axs[i].set_xticks(alpha_grid)
    #axs[i].set_xlim(0.975,2.025)
    

    if i == 2 or i == 3:
        axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)

    # adding labels
    label = label_ls[i] 
    axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, fontweight='bold', va='top', ha='right')

# --------------------------------------------------------------------

alpha_m_ls = []
#cmap_bd = [[0,np.log(2)]] + [[0,np.log(70)]]*3
cmap_bd = [[0,1]] + [[0,10], [0,25], [0,30]]  
#cmap_bd = []
flip = "postact"
layer_ls = [1,10,25,40]
#metric_names = ["kappas_std", "gE_bars_std", "L_gs_std", "kappas", "gE_bars", "L_gs"]
metric_names = ["gE_bars"]
L = 40
xx = list(range(int(L) + 1))

for metric_idx in range(len(metric_names)):
    metric_data = torch.empty((len(layer_ls), mult_N * alpha_N))
    metric_name = metric_names[metric_idx]
    grid_idx = 0
    for alpha_idx in range(len(alpha_grid)):
        for m_idx in range(len(mult_grid)):

            alpha = alpha_grid[alpha_idx]
            mult = mult_grid[m_idx]

            # data path
            randnets_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_{flip}"
            #net_ls = [net[0] for net in os.walk(randnets_path)][1:]
            #net_ls = next(os.walk(f'{randnets_path}'))[1]
            net_path = f"fc{L}_{round(alpha,1)}_{round(mult,2)}"
            net_path_full = f"{randnets_path}/{net_path}"
                      
            init_params = pd.read_csv(f"{randnets_path}/{net_path}/net_init.csv")  
            _, w_alpha, w_mult, N_0, _, _ = init_params.iloc[0,:]    
            N_0 = int(N_0)                

            data_raw = torch.load(f"{net_path_full}/{metric_name}") 
            metric_data[:,grid_idx] = data_raw[layer_ls,:].T/N_0
            alpha_m_ls.append((w_alpha,w_mult))

            grid_idx += 1

    metric_data = metric_data.detach().numpy()
    for layer_idx in range(len(layer_ls)):
        layer_id = layer_ls[layer_idx]
        axs[layer_idx].set_title(f"Layer {layer_id}", fontsize=axis_size)

        metric_mesh = np.zeros((mult_N,alpha_N))
        for t in range(len(alpha_m_ls)):
            
            alpha,mult = alpha_m_ls[t]
            x_loc = int(round((mult_upper - mult) / mult_incre))
            y_loc = int(round((alpha - alpha_lower) / alpha_incre))
            metric_layer = metric_data[layer_idx,:]
            metric_mesh[x_loc,y_loc] = np.log(metric_layer[t] + 1)
            #metric_mesh[x_loc,y_loc] = metric_layer[t]

            #cmap_bd.append([min(metric_layer), max(metric_layer)])
            #cmap_bd.append([np.log(min(metric_layer) + 1),  np.log(max(metric_layer) + 1)])

        # plot results
        # 
        metric_plot = axs[layer_idx].imshow(metric_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                                            vmin=cmap_bd[layer_idx][0], vmax=cmap_bd[layer_idx][1], cmap=cm, interpolation='quadric', aspect='auto')
        #main_plot = axs[r].imshow(acc_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], cmap=cm, interpolation='quadric', aspect='auto')
        plt.colorbar(metric_plot,ax=axs[layer_idx])

        #print(f"{metric_name} {layer_id}: {good}")

    plt.tight_layout()
    print(f"{time.time() - t0}")
    #plt.show()
    fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
    plt.savefig(f"{fig1_path}/{metric_name}_grid.pdf", bbox_inches='tight')


# ---------------------------------------------
   


        









