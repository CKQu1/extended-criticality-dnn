import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import seaborn as sns
import sys
from matplotlib.ticker import AutoMinorLocator
sys.path.append(os.getcwd())
import path_names
from os.path import join
from path_names import root_data, id_to_path, model_log
from utils_dnn import load_transition_lines

# colorbar setting
cm_lib = "sns"
cmaps = []
if cm_lib == "plt":
    for cm_type in ['Spectral', 'gist_stern', 'RdGy']:
        cmaps.append(plt.cm.get_cmap(cm_type))
elif cm_lib == "sns":
    for cm_type in ['Spectral', 'RdBu', 'rocket_r']:
        cmaps.append(sns.color_palette(cm_type, as_cmap=True))
# custom
else:
    for cm_type in [[500,200], [600,100], [20000,25]]:
        cmaps.append(sns.diverging_palette(cm_type[0], cm_type[1], as_cmap=True))

interp = "quadric"
#interp = "spline16"
#cm = cm.get_cmap('plasma')
plt.rcParams["font.family"] = "serif"     # set plot font globally

# plot settings
tick_size = 13
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8

# (alpha, sigma_w) grid setting
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

# includes accuracy, loss and stopping epoch for MLP (main text plot)
def mlp_accloss_phase(acc_type, acc_threshold=93):
    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"

    dataname = "mnist"
    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    path = f"{root_data}/trained_mlps/fcn_grid/{fcn}_grid"
    net_ls = [join(path,net) for net in os.listdir(path) if "epoch650" in net]
    # epoch network was trained till
    epoch_last = 650
    print(path)

    # phase transition lines
    bound1, boundaries = load_transition_lines()

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3,7.142/2 - 0.1))     # without text
    axs = axs.flat

    line_cols_2 = ["white", "white", "white"]
    line_cols_1 = ['k', 'k', 'k']
    # plot boundaries for each axs
    for i in range(len(axs)):
        axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], line_cols_1[i])
        for j in range(len(boundaries)):
            bd = boundaries[j]
            axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], c=line_cols_2[i], linestyle='--')  

    # plot points which computations where executed
    a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)

    for i in range(len(axs)):
        # network realizations
        if i == 0:
            axs[i].plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=5)
        #    axs[i].grid()

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

    epoch = epoch_last

    alpha_m_ls = []
    acc_ls = []
    loss_ls = []
    early_ls = []
    good = 0

    # accuracy grid
    acc_mesh = np.zeros((mult_N,alpha_N))
    loss_mesh = np.zeros((mult_N,alpha_N))
    early_mesh = np.zeros((mult_N,alpha_N))
    for i in range(len(net_ls)):
        try:
            net_path = net_ls[i]
            acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
            
            net_params = sio.loadmat(net_path + "/net_params_all.mat")            
            alpha = list(net_params['net_init_params'][0][0])[1][0][0]
            mult = list(net_params['net_init_params'][0][0])[2][0][0]

            #net_params_all = pd.read_csv(f"{net_path}/net_params_all.csv")
            #alpha, m = literal_eval( net_params_all.loc[0,'init_params'] )

            metrics = acc_loss['training_history'] if acc_type=="train" else acc_loss['testing_history']

            good_ls = [x for x,y in enumerate(metrics[:,1]) if y > acc_threshold]
            early_epoch = epoch_last if len(good_ls) == 0 else min(good_ls)    

            loss, acc = metrics[epoch-1,:]

            alpha_m_ls.append((alpha,mult))
            acc_ls.append(acc)
            loss_ls.append(loss)
            early_ls.append(early_epoch)    

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            acc, loss = np.nan, np.nan
            early_epoch = np.nan
            print(net_path)

        x_loc = int(round((mult_upper - mult) / mult_incre))
        y_loc = int(round((alpha - alpha_lower) / alpha_incre))

        acc_mesh[x_loc,y_loc] = acc
        loss_mesh[x_loc,y_loc] = loss
        early_mesh[x_loc,y_loc] = early_epoch

    # colourmap bounds
    cmap_bd = []
    for ls in [acc_ls, loss_ls, early_ls]:
        cmap_bd.append([np.percentile(ls,5),np.percentile(ls,95)])

    # if want to use imshow convert to grid form
    assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

    # plot results
    mesh_ls = [acc_mesh, loss_mesh, early_mesh]
    for pidx in range(3):
        plot = axs[pidx].imshow(mesh_ls[pidx],extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                                vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=cmaps[pidx], 
                                interpolation=interp, aspect='auto')
        cbar = plt.colorbar(plot,ax=axs[pidx])
        cbar.ax.tick_params(labelsize=tick_size)

    plt.tight_layout()
    #plt.show()
    fig1_path = "/project/PDLAI/project2_data/figure_ms"
    plt.savefig(f"{fig1_path}/{fcn}_{dataname}_{acc_type}_epoch={epoch}_grid_all.pdf", bbox_inches='tight')

    print(f"Good: {good}")
    print("Figure 1")
    print("\n")
    print(len(net_ls))
    print(len(alpha_m_ls))
    print("\n")

def mlp_pure_metric(acc_type="test", epoch_ls = [10, 50, 200]):
  
    assert acc_type == "test" or acc_type == "train"    

    fcn = "fc10"
    net_type = f"{fcn}_mnist_tanh"
    if fcn == "fc10" or fcn == "fc5":
        path = f"{root_data}/trained_mlps/fcn_grid/{fcn}_grid"
    else:
        path = f"{root_data}/trained_mlps/fcn_grid/{fcn}_grid128"
    net_ls = [join(path, dirname) for dirname in os.listdir(path)]
    print(path)
 
    # epoch network was trained till
    epoch_last = 650
    title_ls = [f"Epoch {epoch}" for epoch in epoch_ls]

    # phase transition lines
    bound1, boundaries = load_transition_lines()
    metrics = ["acc", "loss"]

    # 1 by n
    nrows, ncols = len(metrics), len(epoch_ls)
    #fig, axs = plt.subplots(nrows, ncols,sharex=False,sharey=False,figsize=(9.5 + 0.5,7.142/3*len(metrics)))
    fig, axs = fig, axs = plt.subplots(nrows, ncols,sharex=True,sharey=True,figsize=(8.4/2*3 + 1,7.142/2 * len(metrics)))     # without text

    xticks = [1.0,1.2,1.4,1.6,1.8,2.0]
    yticks = [0.5,1.0,1.5,2.0,2.5,3.0]
    # plot phase transition lines for each axs
    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            # ordered regime separation
            axis = axs[row,col]
            axis.plot(bound1.iloc[:,0], bound1.iloc[:,1], c='k')

            # new version
            for j in range(len(boundaries)):
                bd = boundaries[j]
                axis.plot(bd.iloc[:,0], bd.iloc[:,1], c="white", linestyle='--')  

            axis.tick_params(axis='both',labelsize=tick_size)

            # adding labels
            #label = label_ls[i] 
            #axis.text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')

            # setting ticks
            #axis.tick_params(bottom=True, top=False, left=True, right=False)
            #axis.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            #axis.tick_params(axis="x", direction="out")
            #axis.tick_params(axis="y", direction="out")

            axis.set_xticks(xticks)
            axis.set_yticks(yticks)


    alpha_m_ls = []
    acc_ls = []
    loss_ls = []
    good = 0

    # convert to grid form for imshow
    acc_mesh = np.zeros((len(epoch_ls),mult_N,alpha_N))
    loss_mesh = np.zeros((len(epoch_ls),mult_N,alpha_N))
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
                    loss = acc_loss['training_history'][epoch - 1,0] if acc_type == "train" else acc_loss['testing_history'][epoch - 1,0]      

                good += 1     

            except (FileNotFoundError, OSError) as error:
                # use the following to keep track of what to re-run

                acc, loss = np.nan, np.nan
                print(net_path)

            x_loc = int(round((mult_upper - mult) / mult_incre))
            y_loc = int(round((alpha - alpha_lower) / alpha_incre))
            acc_mesh[eidx,x_loc,y_loc] = acc
            loss_mesh[eidx,x_loc,y_loc] = loss

            alpha_m_ls.append((alpha,mult))

    cmap_bds = np.zeros((2, len(epoch_ls), 2))
    for eidx, epoch in enumerate(epoch_ls):
        for metric_idx, metric_name in enumerate(["acc_mesh", "loss_mesh"]):
            axis = axs[metric_idx, eidx]
            metric_ls = locals()[metric_name][eidx,:,:].flatten()
            cmap_bds[metric_idx,eidx,:] = [np.percentile(metric_ls,5),np.percentile(metric_ls,95)]
            #assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

            # plot results
            metric_plot = axis.imshow(locals()[metric_name][eidx,:,:],extent=[alpha_lower,alpha_upper,mult_lower,mult_upper], 
                                          vmin=cmap_bds[metric_idx,eidx,0], vmax=cmap_bds[metric_idx,eidx,1], cmap=cmaps[metric_idx], 
                                          interpolation=interp, aspect='auto')
            cbar = plt.colorbar(metric_plot,ax=axis)
            #l, u = np.ceil(cmap_bds[metric_idx,eidx,0]), np.floor(cmap_bds[metric_idx,eidx,1])
            #cbar.ax.set_yticks([l, u])
            cbar.ax.tick_params(labelsize=tick_size)

    print(f"Good: {good}")

    plt.tight_layout()
    #plt.show()

    fig1_path = "/project/PDLAI/project2_data/figure_ms"
    epochs = [str(epoch) for epoch in epoch_ls]
    epochs = "_".join(epochs)
    fname = f"{fig1_path}/{net_type}_grid_{acc_type}_accloss_epoch={epochs}.pdf"
    print(fname)
    plt.savefig(fname, bbox_inches='tight')

    print("Figure saved!")
    print("\n")
    print(len(net_ls))
    print(len(alpha_m_ls))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

