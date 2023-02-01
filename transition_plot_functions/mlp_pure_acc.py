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


# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/project2_data"
path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"
#path = f"{main_path}/fcn_grid/{fcn}_grid128"
net_ls = [net[0] for net in os.walk(path)]

print(path)
#print(net_ls)

# epoch network was trained till
epoch_last = 650

# phase boundaries
# old version
"""
boundaries = []
boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None))
for i in list(range(1,10,2)):
    boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_{i}_line_2.csv", header=None))
bound1 = boundaries[0]
"""

# new version
bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
boundaries = []
bd_path = "/project/phys_DL/phasediagram"
for i in range(1,92,10):
#for i in range(1,102,10):
    boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))

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

# 2 by 1
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

    # old version
    #for j in range(2,len(boundaries)):
    #    bd = boundaries[j]
    #    axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')


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


epoch_ls = [10, 50]
#epoch_ls = [200, 500]
title_ls = [f"Epoch {epoch}" for epoch in epoch_ls]
#label_ls = ['(a)', '(b)']
#label_ls = ['(c)', '(d)']
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

# accuracy grid

#epoch_ls = [5, 50, 200, epoch_last]
#epoch_ls = [5, 50, 200, 500]
#cmap_bd = [[50,85], [75,95], [80,95], [85,96]] # fc10

#cmap_bd = [[0.85,1], [0.85,1], [0.85,0.99], [0.85,0.99]] # fc10

#cmap_bd = [[85,96], [0.85,0.99]]    # for test acc 650 and test/train acc ratio

#cmap_bd = [[50,85], [75,95]]
#cmap_bd = [[80,95], [85,96]]

#acc_threshold = 93
acc_threshold = 85

#axs[r].set_title(f"Epoch {epoch}", fontsize=axis_size)

alpha_m_ls = []
test_acc_ls = []
#ratio_ls = []
early_ls = []
good = 0

for i in range(1,len(net_ls)):
    try:
        net_path = net_ls[i]
        acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
        
        net_params = sio.loadmat(net_path + "/net_params_all.mat")            
        alpha = list(net_params['net_init_params'][0][0])[1][0][0]
        m = list(net_params['net_init_params'][0][0])[2][0][0]

        #net_params_all = pd.read_csv(f"{net_path}/net_params_all.csv")
        #alpha, m = literal_eval( net_params_all.loc[0,'init_params'] )

        train_loss_all = acc_loss['training_history']
        test_loss_all = acc_loss['testing_history']  

        #train_loss = train_loss_all[epoch - 1,1] 
        #test_loss = test_loss_all[epoch - 1,1]

        alpha_m_ls.append((alpha,m))
        test_acc_ls.append(test_loss_all[epoch_ls[0] - 1,1])
        early_ls.append(test_loss_all[epoch_ls[1] - 1,1])
        #ratio_ls.append(test_loss/train_loss)
        #test_loss_ls.append(train_loss)

        #plt.scatter(alpha, m, c=test_loss, vmin=90, vmax=97,s=55, cmap=cm)   

        good += 1     

    except (FileNotFoundError, OSError) as error:
        # use the following to keep track of what to re-run

        print(net_path)
        #print("\n")

cmap_bd = [[np.percentile(test_acc_ls,5),np.percentile(test_acc_ls,95)], [np.percentile(early_ls,5),np.percentile(early_ls,95)]]

# if want to use imshow
# convert to grid form
#assert len(alpha_m_ls) == len(test_acc_ls) and len(alpha_m_ls) == len(ratio_ls)
assert len(alpha_m_ls) == len(test_acc_ls) and len(alpha_m_ls) == len(early_ls)

#acc_mesh = np.zeros((alpha_N,mult_N))
test_acc_mesh = np.zeros((mult_N,alpha_N))
#ratio_mesh = np.zeros((mult_N,alpha_N))
early_mesh = np.zeros((mult_N,alpha_N))
#print(acc_mesh.shape)
for t in range(len(alpha_m_ls)):
    
    alpha,mult = alpha_m_ls[t]
    #x_loc = int((alpha - alpha_lower)/alpha_incre)   
    #y_loc = int((mult - mult_lower)/mult_incre)   

    x_loc = int(round((mult_upper - mult) / mult_incre))
    y_loc = int(round((alpha - alpha_lower) / alpha_incre))
    """ 
    if epoch == 650:
        print((alpha,mult,test_loss))
        print((alpha - alpha_lower,alpha_incre))
        print((y_loc,test_loss))
        print("\n")
    """

    test_acc_mesh[x_loc,y_loc] = test_acc_ls[t]
    #ratio_mesh[x_loc,y_loc] = ratio_ls[t]
    early_mesh[x_loc,y_loc] = early_ls[t]
    #acc_mesh[y_loc,x_loc] = test_loss_ls[i]

# plot results
test_acc_plot = axs[0].imshow(test_acc_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                              vmin=cmap_bd[0][0], vmax=cmap_bd[0][1], cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
cbar = plt.colorbar(test_acc_plot,ax=axs[0])
cbar.ax.tick_params(labelsize=tick_size)

early_plot = axs[1].imshow(early_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                           vmin=cmap_bd[1][0], vmax=cmap_bd[1][1], cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
cbar = plt.colorbar(early_plot,ax=axs[1])
cbar.ax.tick_params(labelsize=tick_size)

print(f"Good: {good}")

plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
plt.savefig(f"{fig1_path}/{net_type}_grid_testacc_epoch_{epoch_ls[0]}_{epoch_ls[1]}.pdf", bbox_inches='tight')

print("Figure 1")
print("\n")
print(len(net_ls) - 1)
print(len(alpha_m_ls))
#print(len(alpha_ls))
#print(len(m_ls))
print("\n")
#print((len(acc_mesh),len(acc_mesh[0])))

