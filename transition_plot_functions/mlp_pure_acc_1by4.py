import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
sys.path.append(f'{os.getcwd()}')

from ast import literal_eval
from os.path import join
from matplotlib.ticker import AutoMinorLocator
from matplotlib.cm import coolwarm

from path_names import root_data, id_to_path, model_log
from utils import load_transition_lines

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/project2_data"
if fcn == "fc10" or fcn == "fc5":
    path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"
else:
    path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid128"
net_ls = [join(path, dirname) for dirname in os.listdir(path)]
print(path)

# accuracy type
acc_type = "test"
# epoch network was trained till
epoch_last = 650

# phase transition lines
bound1, boundaries = load_transition_lines()

# plot phase transition 
"""
tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8
"""

title_size = 14.5
tick_size = 14.5
label_size = 14.5
axis_size = 14.5
legend_size = 8.5

#fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
#axs = [ax1, ax2, ax3, ax4]

# 1 by 4
epoch_ls = [10, 50, 200]
nrows, ncols = 1, len(epoch_ls)
fig, axs = plt.subplots(nrows, ncols,sharex = True,sharey=True,figsize=(9.5 + 0.5,7.142/len(epoch_ls)))

# plot phase transition lines for each axs
for i in range(len(axs)):
    # ordered regime separation
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')

    # new version
    for j in range(len(boundaries)):
        bd = boundaries[j]
        axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k--')  

mult_lower = 0.25
mult_upper = 3
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

title_ls = [f"Epoch {epoch}" for epoch in epoch_ls]
#label_ls = ['(a)', '(b)']
for i in range(len(axs)):
    # ticks
    axs[i].tick_params(axis='both',labelsize=tick_size)
    
    #axs[i].set_yticks(mult_grid)
    
    #if i == 0:
    #    axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

    #if i == 2 or i == 3:
    #axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
    #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

    # adding labels
    #label = label_ls[i] 
    #axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')

    # setting ticks
    axs[i].tick_params(bottom=True, top=False, left=True, right=False)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")

alpha_m_ls = []
acc_ls = []
good = 0

# convert to grid form for imshow
acc_mesh = np.zeros((len(epoch_ls),mult_N,alpha_N))
for i in range(len(net_ls)):
    for eidx, epoch in enumerate(epoch_ls):
        try:
            net_path = net_ls[i]
            matches = [join(net_path, fname) for fname in os.listdir(net_path) if fcn in fname and "loss_log.mat" in fname]
            acc_loss = sio.loadmat(matches[0])
                          
            if fcn == "fc10":      
                net_params = sio.loadmat(net_path + "/net_params_all.mat")     
                alpha = list(net_params['net_init_params'][0][0])[1][0][0]
                mult = list(net_params['net_init_params'][0][0])[2][0][0]
            else:
                net_params_all = pd.read_csv(f"{net_path}/net_params_all.csv")
                alpha, mult = literal_eval( net_params_all.loc[0,'init_params'] )

            if mult_lower <= mult <= mult_upper:
                acc = acc_loss['training_history'][epoch - 1,1] if acc_type == "train" else acc_loss['testing_history'][epoch - 1,1]      

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            print(net_path)
            #print("\n")

        x_loc = int(round((mult_upper - mult) / mult_incre))
        y_loc = int(round((alpha - alpha_lower) / alpha_incre))
        acc_mesh[eidx,x_loc,y_loc] = acc

        alpha_m_ls.append((alpha,mult))

cmap_bds = np.zeros((len(epoch_ls), 2))
for eidx, epoch in enumerate(epoch_ls):
    acc_ls = acc_mesh[eidx,:,:].flatten()
    cmap_bds[eidx,:] = [np.percentile(acc_ls,5),np.percentile(acc_ls,95)]
    #assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

    # plot results
    acc_plot = axs[eidx].imshow(acc_mesh[eidx,:,:],extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                                  vmin=cmap_bds[eidx][0], vmax=cmap_bds[eidx][1], cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
    cbar = plt.colorbar(acc_plot,ax=axs[eidx])
    l, u = np.ceil(cmap_bds[eidx][0]), np.ceil(cmap_bds[eidx][1])
    cbar.ax.set_yticks([l, u])
    cbar.ax.tick_params(labelsize=tick_size)

print(f"Good: {good}")

plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
epochs = [str(epoch) for epoch in epoch_ls]
epochs = "_".join(epochs)
plt.savefig(f"{fig1_path}/{net_type}_grid_{acc_type}_epoch_{epochs}.pdf", bbox_inches='tight')

print("Figure saved!")
print("\n")
print(len(net_ls))
print(len(alpha_m_ls))

