import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from mpl_toolkits.axes_grid.inset_locator import inset_axes
# colorbar scheme
from matplotlib.cm import coolwarm

from ast import literal_eval

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
path = f"{main_path}/fcn_grid/{fcn}_grid"
#path = f"{main_path}/fcn_grid/{fcn}_grid128"
net_ls = [net[0] for net in os.walk(path)]

print(path)
#print(net_ls)

# epoch network was trained till
#epoch_last = 500
epoch_last = 650

# phase boundaries
boundaries = []
boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None))
for i in list(range(1,10,2)):
    boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_{i}_line_2.csv", header=None))

bound1 = boundaries[0]

# plot phase transition 

"""
tick_size = 12
label_size = 12
axs_size = 12
legend_size = 12
text_size = 12
linewidth = 0.8
"""
tick_size = 16.5
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
axs = [ax1, ax2, ax3, ax4]

# colorbar
cm = cm.get_cmap('RdYlBu')

# plot boundaries for each axs
for i in range(len(axs)):
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')

    for j in range(2,len(boundaries)):
        bd = boundaries[j]
        axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')

# text
# silent
#ax1.text(1.5, 0.23, 'silent', fontsize=text_size)
# localised chaos
#ax1.text(1.4, 1.1, 'persistent chaos', fontsize=text_size)
# global chaos
#ax1.text(1.75, 2.5, 'active chaos', fontsize=text_size)

# x/y label
#ax1.set_xlabel(r'$\alpha$', fontsize=axis_size)
#ax1.set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

# ticks
#ax1.set_xticks(list(range(0,6)))
#ax1.set_xticks(np.linspace(1,2,6))

#ax1.set_xlim(1.0,2.0)

#ax1.set_xticks(np.linspace(1.0,3.0,30))

#ax1.set_ylim(0,4.0)


# tick size
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

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

# accuracy grid

epoch_ls = [5, 50, 200, epoch_last]
#epoch_ls = [5, 50, 200, 500]
#cmap_bd = [[50,85], [75,95], [80,95], [85,96]] # fc10
#cmap_bd = [[0,100], [0,100], [0,100], [0,100]] 
#cmap_bd = [[50,85], [75,95], [80,95], [85,97]] # fc5
#cmap_bd = [[65,85], [82,92], [92,95], [94,96]] # fc3
#cmap_bd = [[10,85], [10,92], [10,95], [10,96]] # fc3

#cmap_bd = [[0.99,1], [0.97,1], [0.94,1], [0.94,1]] # fc3
cmap_bd = [[0.85,1], [0.85,1], [0.85,0.99], [0.85,0.99]] # fc10

for r in range(len(epoch_ls)):

    #epoch = 650
    epoch = epoch_ls[r]

    axs[r].set_title(f"Epoch {epoch}", fontsize=axis_size)

    #alpha_ls = []
    #m_ls = []
    alpha_m_ls = []
    test_loss_ls = []
    good = 0

    test_min = 100
    test_max = 0

    for i in range(1,len(net_ls)):
    #for i in range(45,46):
        try:
            net_path = net_ls[i]
            acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
            
            net_params = sio.loadmat(net_path + "/net_params_all.mat")            
            alpha = list(net_params['net_init_params'][0][0])[1][0][0]
            m = list(net_params['net_init_params'][0][0])[2][0][0]

            #net_params_all = pd.read_csv(f"{net_path}/net_params_all.csv")
            #alpha, m = literal_eval( net_params_all.loc[0,'init_params'] )

            train_loss = acc_loss['training_history'][epoch - 1,1] 
            test_loss = acc_loss['testing_history'][epoch - 1,1]

            test_min = min(test_min,test_loss)
            test_max = max(test_max,test_loss)

            #alpha_ls.append(alpha)
            #m_ls.append(m)
            alpha_m_ls.append((alpha,m))
            #test_loss_ls.append(test_loss)
            test_loss_ls.append(test_loss/train_loss)
            #test_loss_ls.append(train_loss)

            #plt.scatter(alpha, m, c=test_loss, vmin=90, vmax=97,s=55, cmap=cm)   

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            print(net_path)
            #print("\n")

    # if want to use imshow
    # convert to grid form
    assert len(alpha_m_ls) == len(test_loss_ls)

    #acc_mesh = np.zeros((alpha_N,mult_N))
    acc_mesh = np.zeros((mult_N,alpha_N))
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

        acc_mesh[x_loc,y_loc] = test_loss_ls[t]
        #acc_mesh[y_loc,x_loc] = test_loss_ls[i]

    #if epoch == 650:
    #    print(acc_mesh)

    # plot results
    #axs[r].scatter(alpha_ls, m_ls, c=test_loss_ls, vmin=test_min, vmax=test_max,s=55, cmap=cm)
    #main_plot = axs[r].scatter(alpha_ls, m_ls, c=test_loss_ls, vmin=cmap_bd[r][0], vmax=cmap_bd[r][1],s=55, cmap=cm)
    main_plot = axs[r].imshow(acc_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[r][0], vmax=cmap_bd[r][1], cmap=cm, interpolation='quadric', aspect='auto')
    #main_plot = axs[r].imshow(acc_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], cmap=cm, interpolation='quadric', aspect='auto')
    plt.colorbar(main_plot,ax=axs[r])
    
    print(f"Good: {good}")

    #axs[r].colorbar()

plt.tight_layout()
plt.show()

fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
# ratio
#plt.savefig(f"{fig1_path}/{net_type}_grid_ratio.pdf", bbox_inches='tight')
# test
#plt.savefig(f"{fig1_path}/{net_type}_grid_acc.pdf", bbox_inches='tight')

print("Figure 1")
print("\n")
print(len(net_ls) - 1)
print(len(alpha_m_ls))
#print(len(alpha_ls))
#print(len(m_ls))
print("\n")
#print((len(acc_mesh),len(acc_mesh[0])))

##########################################################################################################################################################

"""

figure(figsize=(6.5,5.4))

# plot 1 - order
ax1 = plt.axes([0, 0, 1, 1])

# phase transition
ax1.plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')

for j in range(2,len(boundaries)):
    bd = boundaries[j]
    ax1.plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')

layer_width = [784]*10 + [10]
#layer_width = [784]*10 + [10]
depth = len(layer_width) - 1
#markers = ['o','x','^','.','*']
markers = ['.','.','.','.','.']
colors = ['r','b','k','g','c']
incre = 50


for i in range(1,len(net_ls)):
#for i in range(1,30):

    try: 
        net_path = net_ls[i]
        #stable_temp = sio.loadmat(f"{net_path}/W_stable_params_1-500.mat")
        stable_temp = sio.loadmat(f"{net_path}/W_stable_params_1-650.mat")

        init_params = sio.loadmat(net_path + "/net_params_all.mat")
        alpha = list(init_params['net_init_params'][0][0])[1][0][0]
        m = list(init_params['net_init_params'][0][0])[2][0][0]

        if alpha <= 1.5 and m <= 1:

            alphas = stable_temp['alphas']
            deltas = stable_temp['deltas']

            for l in range(depth):

                alpha_f = [alpha] + list(alphas[l,::incre])
                m_f = [m] + list((2*np.sqrt(layer_width[l]*layer_width[l+1]))**(1/alphas[l,::incre])*deltas[l,::incre])
             
                ax1.plot(alpha_f,m_f,marker=markers[i%3],label=str(l+1),color=colors[l%3])
                #ax1.plot(alpha_f,m_f,label=str(l+1))
    
    except FileNotFoundError:
        print(net_path)

ax1.set_ylim([0,4])

plt.show()

#plt.legend()

#plt.savefig(f"{fig1_path}/fc10_grid_transition.pdf", bbox_inches='tight')

print("Figure 2")
print("\n")
print(len(net_ls))
print(len(alpha_ls))
print(len(m_ls))

"""

