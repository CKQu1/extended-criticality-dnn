import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
#from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf
from mpl_toolkits.axes_grid.inset_locator import inset_axes
# colorbar scheme
from matplotlib.cm import coolwarm
#from matplotlib.cm import inferno

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

from ast import literal_eval

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
main_path = "/project/PDLAI/project2_data"
path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"
dq_path = "/project/phys_DL/grad_plots"

net_data_path = "/project/phys_DL/grad_plots"
net_ls = [net[0] for net in os.walk(net_data_path)][1:]
net_ls.remove(f"{net_data_path}/job")

# phase boundaries
boundaries = []
boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None))
for i in list(range(1,10,2)):
    boundaries.append(pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_{i}_line_2.csv", header=None))

bound1 = boundaries[0]

# plot phase transition 

tick_size = 16.5
label_size = 16.5
axis_size = 16.5
legend_size = 12

#epoch = 100
for epoch in range(0,601,50):
    for layer in range(0,10):

        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = False,sharey=False,figsize=(9.5,7.142))
        axs = [ax1, ax2, ax3, ax4]

        # plot boundaries for each axs
        #for i in range(len(axs)):
        for i in range(2,4):
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

        for i in range(2,4):
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

            if i == 0 or i == 1:
            
                #axs[i].set_xticks(alpha_grid)
                axs[i].set_ylim(0,1.1)
                axs[i].set_xlim(0,2)
            

            if i == 2 or i == 3:
                axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)
                axs[i].set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

            # adding labels
            label = label_ls[i] 
            axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, fontweight='bold', va='top', ha='right')

        axs[0].set_title(r"$\alpha$ = 1.2, $m$ = 1.5", fontsize=label_size)
        axs[1].set_title(r"$\alpha$ = 2.0, $m$ = 1.5", fontsize=label_size)

        # Figure (a) and (b): plot the D_q vs q

        net_names = ["fc10_mnist_tanh_1.2_1.5_grads", "fc10_mnist_tanh_2.0_1.5_grads"]
        data_names = [f"grad_ep_{epoch}_l_{layer}.txt", f"grad_ep_{epoch}_l_{layer}.txt"]

        for f_idx in range(len(data_names)):
            
            df = np.loadtxt(f"{dq_path}/{net_names[f_idx]}/{data_names[f_idx]}")
            qs = df[:,0]
            dq_means = df[:,1]
            dq_stds = df[:,2]
            lower = dq_means - dq_stds
            upper = dq_means + dq_stds
            
            # averages of dq's
            axs[f_idx].plot(qs, dq_means)
            #axs[f_idx].plot(qs, lower)
            #axs[f_idx].plot(qs, upper)
            # error bars
            for q_idx in range(len(qs)):
                axs[f_idx].axvline(qs[q_idx], ymin=lower[q_idx]/1.1, ymax=upper[q_idx]/1.1, alpha=0.75)

            axs[f_idx].set_xlabel(r'$q$', fontsize=axis_size)
            axs[f_idx].set_ylabel(r'$D_q$', fontsize=axis_size)

        # accuracy grid

        #cmap_bd = [[0.24,1], [6.5e-11,0.26]]
        #cmap_bd = [[0.24,1], [0,0.26]]
        #cmap_bd = [[0.3,0.85], [0,0.25]]    # epoch 650
        #cmap_bd = [[0.1,0.10002], [0,0.09]]      # epoch 0

        #epoch = 650
        #epoch = 150
        #layer = 0
        good = 0

        #axs[r].set_title(f"Epoch {epoch}", fontsize=axis_size)

        alpha_m_ls = []
        d2_mean_ls = []
        d2_std_ls = []

        def get_alpha_m(net_path):

            f = net_path.split("/")[-1]
            f_strs = f.split("_")
            alpha, m = f_strs[3:5]

            return float(alpha), float(m)

        for i in range(0,len(net_ls)):
            try:
                net_path = net_ls[i]
                alpha, m = get_alpha_m(net_path)
                dq_data = np.loadtxt(f"{net_path}/grad_ep_{epoch}_l_{layer}.txt")
                
                d2_mean = dq_data[-1,1]    
                d2_std = dq_data[-1,2]    

                alpha_m_ls.append((alpha,m))
                d2_mean_ls.append(d2_mean)
                d2_std_ls.append(d2_std)

                good += 1     

            except (FileNotFoundError, OSError) as error:
                # use the following to keep track of what to re-run

                print(net_path)
                #print("\n")

        # if want to use imshow
        # convert to grid form
        assert len(alpha_m_ls) == len(d2_mean_ls) and len(alpha_m_ls) == len(d2_std_ls)

        cmap_bd = [[min(d2_mean_ls),max(d2_mean_ls)], [min(d2_std_ls),max(d2_std_ls)]]

        mean_mesh = np.zeros((mult_N,alpha_N))
        std_mesh = np.zeros((mult_N,alpha_N))

        for t in range(len(alpha_m_ls)):
            
            alpha,mult = alpha_m_ls[t]
            #x_loc = int((alpha - alpha_lower)/alpha_incre)   
            #y_loc = int((mult - mult_lower)/mult_incre)   

            x_loc = int(round((mult_upper - mult) / mult_incre))
            y_loc = int(round((alpha - alpha_lower) / alpha_incre))

            mean_mesh[x_loc,y_loc] = d2_mean_ls[t]
            std_mesh[x_loc,y_loc] = d2_std_ls[t]

        # plot results
        #axs[r].scatter(alpha_ls, m_ls, c=d2_ls, vmin=test_min, vmax=test_max,s=55, cmap=plt.cm.get_cmap(cm_type))
        #main_plot = axs[r].scatter(alpha_ls, m_ls, c=d2_ls, vmin=cmap_bd[r][0], vmax=cmap_bd[r][1],s=55, cmap=plt.cm.get_cmap(cm_type))
        mean_plot = axs[2].imshow(mean_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[0][0], vmax=cmap_bd[0][1], cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
        #main_plot = axs[r].imshow(mean_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
        plt.colorbar(mean_plot,ax=axs[2])

        std_plot = axs[3].imshow(std_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[1][0], vmax=cmap_bd[1][1], cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
        #main_plot = axs[r].imshow(mean_mesh,extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
        plt.colorbar(std_plot,ax=axs[3])

        #axs[r].colorbar()

        axs[2].set_title(r"$D_2$ mean", fontsize=label_size)
        axs[3].set_title(r"$D_2$ standard deviation", fontsize=label_size)

        plt.tight_layout()
        #plt.show()

        fig1_path = "/project/PDLAI/project2_data/figure_ms/dq_grad_plots"
        plt.savefig(f"{fig1_path}/dq_transition_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')

    print(f"{epoch} done!")

