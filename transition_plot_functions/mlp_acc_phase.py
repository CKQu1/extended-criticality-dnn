import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import sys
from matplotlib.ticker import AutoMinorLocator
sys.path.append(f'{os.getcwd()}')
import path_names
from os.path import join
from path_names import root_data, id_to_path, model_log

# colorbar
#cm = cm.get_cmap('gist_heat')
cm_type1 = 'CMRmap'
cm_type2 = 'gist_stern'
#cm_type2 = "terrain"
interp = "quadric"
#cm = cm.get_cmap('plasma')
plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.rcParams["font.family"] = "Helvetica"

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/project2_data"
path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"
#path = f"{main_path}/fcn_grid/{fcn}_grid128"
net_ls = [join(path,net) for net in os.listdir(path) if "epoch650" in net]
# epoch network was trained till
#epoch_last = int(net_ls[0][net_ls[0].index("epochs")+7:])
epoch_last = 650
print(path)

tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

# phase boundaries
bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
boundaries = []
bd_path = "/project/phys_DL/phasediagram"
for i in range(1,92,10):
#for i in range(1,102,10):
    boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))

#fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
#axs = [ax1, ax2, ax3, ax4]
#fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2))
axs = [ax1, ax2]

# plot boundaries for each axs
for i in range(len(axs)):
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')
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
    # network realizations
    if i == 0:
        axs[i].plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=5)

    # major ticks
    axs[i].tick_params(bottom=True, top=True, left=True, right=True)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")

    # minor ticks
    #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    #axs[i].yaxis.set_minor_locator(AutoMinorLocator())


#title_ls = ['Test accuracy', r'$\frac{{{}}}{{8}}$'.format("Test accuracy", "Train accuracy")]
title_ls = ['Test accuracy', 'Earliest epoch reaching' + '\n' + 'test acc. threshold ']
label_ls = ['(a)', '(b)', '(c)', '(d)']
for i in range(len(axs)):
    # ticks
    axs[i].tick_params(axis='both',labelsize=tick_size)

    # ticks
    #if i == 0 or i == 2:
    #axs[i].set_xticks(np.linspace(100,600,6))
    
    #axs[i].set_yticks(mult_grid)
    #axs[i].set_ylim(0,3.25)
    
    if i == 0:
        axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

    #if i == 2 or i == 3:
    
    #axs[i].set_xticks(alpha_grid)
    #axs[i].set_xlim(0.975,2.025)

    #if i == 2 or i == 3:
    #axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
    #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

    # adding labels
    label = label_ls[i] 
    #axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')   # fontweight='bold'

    # setting ticks
    axs[i].tick_params(bottom=True, top=False, left=True, right=False)
    axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    axs[i].tick_params(axis="x", direction="out")
    axs[i].tick_params(axis="y", direction="out")

# accuracy grid

epoch_ls = [5, 50, 200, epoch_last]
#epoch_ls = [5, 50, 200, 500]
#cmap_bd = [[50,85], [75,95], [80,95], [85,96]] # fc10
#cmap_bd = [[0,100], [0,100], [0,100], [0,100]] 
#cmap_bd = [[50,85], [75,95], [80,95], [85,97]] # fc5
#cmap_bd = [[65,85], [82,92], [92,95], [94,96]] # fc3
#cmap_bd = [[10,85], [10,92], [10,95], [10,96]] # fc3

#cmap_bd = [[0.99,1], [0.97,1], [0.94,1], [0.94,1]] # fc3
#cmap_bd = [[0.85,1], [0.85,1], [0.85,0.99], [0.85,0.99]] # fc10

acc_threshold = 93
#acc_threshold = 85
epoch = epoch_last

alpha_m_ls = []
test_acc_ls = []
early_ls = []
good = 0

for i in range(len(net_ls)):
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

        good_ls = [x for x,y in enumerate(test_loss_all[:,1]) if y > acc_threshold]
        if len(good_ls) == 0:
            early_ls.append(epoch_last)
        else:
            early_ls.append( min(good_ls) )       

        train_loss = train_loss_all[epoch - 1,1] 
        test_loss = test_loss_all[epoch - 1,1]

        alpha_m_ls.append((alpha,m))
        test_acc_ls.append(test_loss)
        #ratio_ls.append(test_loss/train_loss)
        #test_loss_ls.append(train_loss)

        #plt.scatter(alpha, m, c=test_loss, vmin=90, vmax=97,s=55, cmap=cm)   
        good += 1     

    except (FileNotFoundError, OSError) as error:
        # use the following to keep track of what to re-run
        print(net_path)

# colourmap bounds
cmap_bd = [[np.percentile(test_acc_ls,5),np.percentile(test_acc_ls,95)], [np.percentile(early_ls,5),np.percentile(early_ls,95)]]

# if want to use imshow convert to grid form
assert len(alpha_m_ls) == len(test_acc_ls) and len(alpha_m_ls) == len(early_ls)

test_acc_mesh = np.zeros((mult_N,alpha_N))
early_mesh = np.zeros((mult_N,alpha_N))
#print(acc_mesh.shape)
for t in range(len(alpha_m_ls)):   
    alpha,mult = alpha_m_ls[t]
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

#if epoch == 650:
#    print(acc_mesh)

# plot results
#axs[r].scatter(alpha_ls, m_ls, c=test_loss_ls, vmin=test_min, vmax=test_max,s=55, cmap=cm)
#main_plot = axs[r].scatter(alpha_ls, m_ls, c=test_loss_ls, vmin=cmap_bd[r][0], vmax=cmap_bd[r][1],s=55, cmap=cm)

test_acc_plot = axs[0].imshow(test_acc_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                              vmin=cmap_bd[0][0], vmax=cmap_bd[0][1], cmap=plt.cm.get_cmap(cm_type1), interpolation=interp, aspect='auto')
cbar = plt.colorbar(test_acc_plot,ax=axs[0])
cbar.ax.tick_params(labelsize=tick_size)

early_plot = axs[1].imshow(early_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                           vmin=cmap_bd[1][0], vmax=cmap_bd[1][1], cmap=plt.cm.get_cmap(cm_type2), interpolation=interp, aspect='auto')
cbar = plt.colorbar(early_plot,ax=axs[1])
cbar.ax.tick_params(labelsize=tick_size)

plt.tight_layout()
#plt.show()
fig1_path = "/project/PDLAI/project2_data/figure_ms"
plt.savefig(f"{fig1_path}/{net_type}_grid_testacc_early_{epoch}.pdf", bbox_inches='tight')

print(f"Good: {good}")
print("Figure 1")
print("\n")
print(len(net_ls))
print(len(alpha_m_ls))
print("\n")

