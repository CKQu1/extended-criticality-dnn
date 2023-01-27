import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io as sio

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
#from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes

plt.switch_backend('agg')

# colorbar
cm_type = 'CMRmap'
interp = "quadric"
plt.rcParams["font.family"] = "serif"     # set plot font globally

fcn = "fc10"
net_type = f"{fcn}_mnist_tanh"
#net_type = f"{fcn}_mnist_tanh_2"
#main_path = "/project/PDLAI/Anomalous-diffusion-dynamics-of-SGD"
main_path = "/project/PDLAI/project2_data"
path = f"{main_path}/trained_mlps/fcn_grid/{fcn}_grid"

# post/pre-activation and right/left-eigenvectors
post = 0
reig = 1

assert post == 1 or post == 0, "No such option!"
assert reig == 1 or reig == 0, "No such option!"
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

#dq_path = f"/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/dq_layerwise"
dq_path = f"/project/PDLAI/project2_data/geometry_data/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"

# new version phase boundaries
bound1 = pd.read_csv(f"{main_path}/phase_bound/phasediagram_pow_1_line_1.csv", header=None)
boundaries = []
#bd_path = "/project/PDLAI/project2_data/phasediagram"
#for i in range(1,102,10):
#    boundaries.append(pd.read_csv(f"{bd_path}/pow_{i}.csv"))

# ----- plot phase transition -----

#tick_size = 16.5
#label_size = 16.5
#axis_size = 16.5
#legend_size = 12

title_size = 23.5
tick_size = 23.5
label_size = 23.5
axis_size = 23.5
legend_size = 23.5
linewidth = 0.8
text_size = 14

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

q_folder_idx = 49
missing_data = []
# in the future for ipidx might be needed
# test first
#for epoch in [0,1]:
#    for layer in range(0,2):
#for epoch in [0,1] + list(range(50,651,50)):   # all
for epoch in [0,650]:
    for layer in range(0,10):

        if epoch == 0 and layer == 0:
            extension_name = f"alpha120_g150_ipidx0_ep{epoch}.txt"                
            df_mean = np.loadtxt(f"{dq_path}/dqmean_{extension_name}")
            qs = df_mean[:,0]
            q_folder = qs[q_folder_idx]
            print(q_folder)

        # ----- Plot grid -----

        good = 0
        #axs[r].set_title(f"Epoch {epoch}", fontsize=axis_size)

        alpha_m_ls = []
        d2_mean_ls = []
        d2_std_ls = []

        def get_alpha_m(str_name):

            if all(s in str_name for s in ('alpha', 'g')):
                alpha, m = tuple(re.findall('\d+', str_name)[:2]) 

            return float(alpha)/100, float(m)/100

        #for i in range(0,len(net_ls)):
        for alpha100 in range(100, 201, 10):
            for g100 in range(25, 301, 25):
                try:
                    ext_name = f"alpha{alpha100}_g{g100}_ipidx0_ep{epoch}"
                    alpha, m = get_alpha_m(ext_name)
                    #dq_data = np.loadtxt(f"{net_path}/grad_ep_{epoch}_l_{layer}.txt")
                    dq_data = np.loadtxt(f"{dq_path}/dqmean_{ext_name}.txt")                
                    d2_mean = dq_data[q_folder_idx, layer + 1]    
                    dq_data = np.loadtxt(f"{dq_path}/dqstd_{ext_name}.txt")
                    d2_std = dq_data[q_folder_idx, layer + 1]    

                    alpha_m_ls.append((alpha,m))
                    d2_mean_ls.append(d2_mean)
                    d2_std_ls.append(d2_std)

                    good += 1     

                except (FileNotFoundError, OSError) as error:
                    # use the following to keep track of what to re-run

                    if all(s in ext_name for s in ('alpha', 'g', 'ipidx', 'ep')):
                        #missing_data.append( tuple(re.findall('\d+', ext_name)[:4]) )
                        line = re.findall('\d+', ext_name)[:4]
                        #for idx in range(len(line)):
                        #    line[idx] = int(line[idx])
                        missing_data.append( re.findall('\d+', ext_name)[:4] )

        # if want to use imshow, convert to grid form
        assert len(alpha_m_ls) == len(d2_mean_ls) and len(alpha_m_ls) == len(d2_std_ls)

        # colorbar bound
        cmap_bd = [[min(d2_mean_ls),max(d2_mean_ls)], [min(d2_std_ls),max(d2_std_ls)]]
        #cmap_bd = [[0.5, 1], [min(d2_std_ls),max(d2_std_ls)]]

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

        fig1_path = f"/project/PDLAI/project2_data/figure_ms/dq_jac_single_{post_dict[post]}_{reig_dict[reig]}_q={round(q_folder,1)}_plots"
        if not os.path.isdir(fig1_path): os.makedirs(fig1_path)

        # ----- plot template -----
        title_names = ["mean", "standard deviation"]
        save_names = ["mean", "std"]
        metrics_all = {title_names[0]: mean_mesh, title_names[1]: std_mesh}
        for plot_idx in range(2):
            fig, ax = plt.subplots(1, 1,figsize=(9.5,7.142))
            # plot boundaries for each axs
            ax.plot(bound1.iloc[:,0], bound1.iloc[:,1], linewidth=2.5, color='k')
            #for j in range(len(boundaries)):
            #    bd = boundaries[j]
            #    ax.plot(bd.iloc[:,0], bd.iloc[:,1], linewidth=2.5, color='k', linestyle='-.')

            # plot points which computations where executed
            #a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)
            #ax.plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=12)
                
            ax.set_xlabel(r'$\alpha$', fontsize=axis_size)
            ax.set_ylabel(r'$D_w^{1/\alpha}$', fontsize=axis_size)

            # minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # label ticks
            ax.tick_params(axis='x', labelsize=axis_size - 1)
            ax.tick_params(axis='y', labelsize=axis_size - 1)

            #-----

            title_name = title_names[plot_idx]
            main_plot = ax.imshow(metrics_all[title_name],extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], vmin=cmap_bd[plot_idx][0], vmax=cmap_bd[plot_idx][1], 
                                  cmap=plt.cm.get_cmap(cm_type), interpolation='quadric', aspect='auto')
            if q_folder == int(q_folder):
                ax.set_title(rf"$D_{{{int(q_folder)}}}$ {title_name}", fontsize=label_size)
            else:
                ax.set_title(rf"$D_{{{round(q_folder,1)}}}$ {title_name}", fontsize=label_size)
            cbar = plt.colorbar(main_plot,ax=ax)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.tick_params(labelsize=axis_size - 3)
            plt.tight_layout()
            plt.savefig(f"{fig1_path}/dq_jac_{save_names[plot_idx]}_transition_{post_dict[post]}_{reig_dict[reig]}_l={layer}_epoch={epoch}.pdf", bbox_inches='tight')

            #plt.show()
            plt.close(fig)

    #print(f"Epoch {epoch} layer {layer} done!")
    print(f"Epoch {epoch} done!")

#np.savetxt("/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data", missing_data)
np.savetxt("/project/PDLAI/project2_data/geometry_data/missing_data.txt", np.array(missing_data), fmt='%s')

