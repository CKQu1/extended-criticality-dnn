import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')


import path_names
from ast import literal_eval
from tamsd_analysis.tamsd import id_to_path, model_log

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

selected_path = path_names.fc_path2
net_ls = [net[0] for net in os.walk(selected_path) if "epochs=650" in net[0]]
#net_ls.pop(0)

print(selected_path)
print(f"Total of networks: {len(net_ls)}.")

# transfer full path to model_id
def get_model_id(full_path):

    str_ls = full_path.split('/')

    #return str_ls[-1].split("_")[4]
    return str_ls[-1].split("_")[3]

# get (alpha,g) when the pair is not saved
def get_alpha_g(full_path):
    str_ls = full_path.split('/')
    str_alpha_g = str_ls[-1].split("_")
    #return (int(str_alpha_g[2]), int(str_alpha_g[3]))
    return (int(str_alpha_g[1]), int(str_alpha_g[2]))

# epoch network was trained till
epoch_last = 650

# plot phase transition 

title_size = 26.5
tick_size = 26.5
label_size = 26.5
axis_size = 26.5
legend_size = 23.5

# plot boundaries for each axs
"""
for i in range(len(axs)):
    axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], 'k')

    for j in range(2,len(boundaries)):
        bd = boundaries[j]
        axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], 'k-.')
"""

mult_lower = 0.25
#mult_upper = 2
#mult_lower = 0.25
mult_upper = 3
#mult_upper = 4
mult_N = int((mult_upper - mult_lower)/0.25 + 1)
#mult_N = 20
mult_grid = np.linspace(mult_lower,mult_upper,mult_N)
#mult_incre = round(mult_grid[1] - mult_grid[0],2)
mult_incre = 0.25

alpha_lower = 1
alpha_upper = 2
alpha_N = int((alpha_upper - alpha_lower)/0.1 + 1)
alpha_grid = np.linspace(alpha_lower,alpha_upper,alpha_N)
alpha_incre = round(alpha_grid[1] - alpha_grid[0],1)

# plot points which computations where executed
a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)

#title_ls = ['Test accuracy', r'$\frac{{{}}}{{8}}$'.format("Test accuracy", "Train accuracy")]
#title_ls = ['Test accuracy', 'Test/train accuracy ratio']
title_ls = ['Test accuracy', 'Earliest epoch reaching' + '\n' + 'test acc. threshold ']
label_ls = ['(a)', '(b)', '(c)', '(d)']

# accuracy grid

epoch_ls = [20]

#cmap_bd = [[0.85,1], [0.85,1], [0.85,0.99], [0.85,0.99]] # fc10

#cmap_bd = [[0,100], [0,200]]
min_acc = 100
max_acc = 0
min_epoch = epoch_last
max_epoch = 0

#epoch = 500
epoch = 500
#epoch = 10

#ed_mesh -= 10

#alpha_ls = np.arange(1,2.01,0.1)
#g_ls = np.linspace(0.25,3.01,0.25)

#for epoch_plot in [650]:
#for epoch_plot in [0,5]:

hidden_type = "all"      # take ed average over all hidden layers or pre/post activation layers
for epoch_plot in [0,1,5,10,50,100,200,250,500,600,650]:
#for epoch_plot in [650]:
    #fig, axs_matrix = plt.subplots(nrows, ncols,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))
    #axs = [axs_matrix[row,col] for row in range(nrows) for col in range(ncols) ]

    ed_mean_mesh = np.zeros((mult_N,alpha_N))
    ed_var_mesh = np.zeros((mult_N,alpha_N))
    cmap_bd = [[np.inf, -np.inf], [np.inf, -np.inf]]    # colormap

    # set up plot
    nrows, ncols = 1,1
    fig, ax1 = plt.subplots(nrows, ncols,sharex = True,sharey=True,figsize=(9.5,7.142))
    axs = [ax1]

    for i in range(len(axs)):

        axs[i].plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=5)
        
        if i == 0:
            axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

        #if i == 2 or i == 3:
        
        #axs[i].set_xticks(alpha_grid)
        #axs[i].set_xlim(0.975,2.025)
        #axs[i].set_yticks(np.arange(0.4,2.01,0.4))

        #if i == 2 or i == 3:
        axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
        #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

        # adding labels
        #label = label_ls[i] 
        #axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')

        # setting ticks
        axs[i].tick_params(bottom=True, top=False, left=True, right=False)
        axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        axs[i].tick_params(axis="x", direction="in")
        axs[i].tick_params(axis="y", direction="in")
        axs[i].tick_params(axis='both',labelsize=tick_size)

    alpha_m_ls = []
    ed_means = []
    ed_vars = []
    good = 0

    for i in range(len(net_ls)):

        net_path = net_ls[i]
        model_id = get_model_id(net_path)
        try:
            ed_all = np.loadtxt(f"{net_path}/ed_train")
            model_info = model_log(model_id)
            alpha100, g100 = get_alpha_g(net_path)
            alpha, g = int(alpha100)/100, int(g100)/100

            x_loc = int(round((mult_upper - g) / mult_incre))
            y_loc = int(round((alpha - alpha_lower) / alpha_incre))

            alpha_m_ls.append((alpha, g))
            if hidden_type == "all":
                mean, var = ed_all[:, epoch_plot].mean(), ed_all[:, epoch_plot].var()                
            elif hidden_type == "postact":
                mean, var = ed_all[1::2, epoch_plot].mean(), ed_all[1::2, epoch_plot].var()                
            elif hidden_type == "preact":
                mean, var = ed_all[0::2, epoch_plot].mean(), ed_all[0::2, epoch_plot].var()                
            else:
                raise TypeError("Choose the right hidden_type!")

            ed_means.append(mean)
            #ed_vars.append(var)
            ed_vars.append(np.sqrt(var)/mean)

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            ed_means.append(0)
            ed_vars.append(0)

            #print((alpha100, g100))
            print(net_path)

        ed_mean_mesh[x_loc,y_loc] = ed_means[i]
        ed_var_mesh[x_loc,y_loc] = ed_vars[i]

    cmap_bd[0][0], cmap_bd[0][1] = np.percentile(ed_means,5),np.percentile(ed_means,95)
    #cmap_bd[1][0], cmap_bd[1][1] = min(cmap_bd[1][0], min(ed_vars)), max(cmap_bd[1][1], max(ed_vars))

    # if want to use imshow
    # convert to grid form
    #assert len(alpha_m_ls) == len(test_acc_ls) and len(alpha_m_ls) == len(early_ls), print("alpha_m_ls and test_acc_ls have different lengths!")

    ed_mean_plot = axs[0].imshow(ed_mean_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[0][0], vmax=cmap_bd[0][1], 
                                 cmap=plt.cm.get_cmap(cm_type), interpolation=interp, aspect='auto')
    cbar = plt.colorbar(ed_mean_plot,ax=axs[0])
    cbar.ax.tick_params(labelsize=tick_size)

    #axs[0].set_title(f"Epoch {epoch_plot}, ED mean")
    #axs[1].set_title(f"Epoch {epoch_plot}, ED variance")  

    print(f"Epoch {epoch_plot}")
    print(f"Good: {good}")
    #print("\n")
    #print(len(net_ls))
    #print(len(alpha_m_ls))

    plt.tight_layout()

    net_type = model_info.loc[model_info.index[0],'net_type']
    depth = int(model_info.loc[model_info.index[0],'depth'])
    plot_path = f"{path_names.log_path}/fig_path/{net_type}{depth}_ed"
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    plt.savefig(f"{plot_path}/{net_type}{depth}_mnist_epoch={epoch_plot}_edmean_{hidden_type}_phase.pdf", bbox_inches='tight')

    #plt.show()
    plt.close()


