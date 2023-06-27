import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import sys
from ast import literal_eval
from matplotlib.ticker import AutoMinorLocator
from time import time
from tqdm import tqdm
sys.path.append(os.getcwd())
import path_names
from os.path import join
from path_names import root_data, id_to_path, model_log

c_ls = ["darkblue", "darkred"]
plt.rcParams["font.family"] = 'sans-serif'     # set plot font globally

# plot settings
tick_size = 14.5
label_size = 15.5
axis_size = 15.5
legend_size = 14
linewidth = 0.8

# includes accuracy, loss and stopping epoch for MLP (main text plot)
def accloss_control(path, acc_type, control, display=False):
    t0 = time()
    global acc_all, acc_arr, net_ensembles, net_folders, dataset_folders, acc_load

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    assert control == "num_train" or control == "X_clusters"
    display = literal_eval(display) if isinstance(display,str) else display

    # common settings
    fcn = "fc10"
    epoch = 650
    #alpha100_ls = [100, 200]
    alpha100_ls = [100, 200]
    g100_ls = [25, 100, 300]
    X_dim = 784 
    Y_classes = 10
    num_test = 1000
    fig, axs = plt.subplots(1,len(g100_ls),sharex = True,sharey=True,figsize=(8.4/2*len(g100_ls)+0.45,7.142/2 - 0.1))     # without text
    axs = axs.flat

    # plot 1: effect of training samples number
    if control == "num_train":                
        #num_train_ls = [5,10,20,30,40,50] + list(range(100,801,100))
        num_train_ls = [5,10,20,30,40,50] + list(range(100,501,100))
        X_clusters_ls = [60,120]   
        # plot settings
        xticks = [5, 100, 200, 300, 400]
        xtick_ls = xticks
    else:
        num_train_ls = [10, 400]
        #X_clusters_ls = list(range(20,241,20))
        X_clusters_ls = list(range(20,121,20))
        xticks = X_clusters_ls
        xtick_ls = xticks

    # failed sims
    failed_sims = []

    print("Computing over ensembles.")
    acc_all = np.zeros([len(alpha100_ls), len(g100_ls), len(X_clusters_ls), len(num_train_ls), 2])
    net_ensembles = np.zeros([len(alpha100_ls), len(g100_ls), len(X_clusters_ls), len(num_train_ls)])   # number of ensembles for each setting        
    for idx, X_clusters in enumerate(X_clusters_ls):
        for iidx, num_train in tqdm(enumerate(num_train_ls)):        
            dataset_folders = [join(path,dataset_folder) for dataset_folder in next(os.walk(path))[1] if f"{fcn}_tanh_gaussian_data_{num_train}_{num_test}_{X_dim}_{Y_classes}_{X_clusters}" in dataset_folder]
            for aidx, alpha100 in enumerate(alpha100_ls):
                for gidx, g100 in enumerate(g100_ls):    
                    net_folders = []                           
                    for dataset_folder in dataset_folders: 
                        net_folders += [join(dataset_folder,net) for net in next(os.walk(dataset_folder))[1] if f"{fcn}_{alpha100}_{g100}_" in net]      
                    net_folders = net_folders[:20]                                        
                    net_ensembles[aidx, gidx, idx, iidx] = len(net_folders)
                    acc_arr = np.zeros([len(net_folders)])
                    #for net_idx, net_folder in enumerate(net_folders):
                    for net_idx, net_folder in enumerate(net_folders):
                        if os.path.isfile(join(net_folder,"acc_loss")):
                            acc_load = pd.read_csv(join(net_folder,"acc_loss"))
                            acc_arr[net_idx] = acc_load.iloc[epoch,3] if acc_type=="test" else acc_load.iloc[epoch,1]
                        else:
                            # failed ones
                            net_ensembles[aidx, gidx, idx, iidx] -= 1
                            acc_arr[net_idx] = np.nan
                            failed_sims.append(net_folders[net_idx])

                    acc_all[aidx,gidx,idx,iidx,0] = np.nanmean(acc_arr)
                    acc_all[aidx,gidx,idx,iidx,1] = np.nanstd(acc_arr)    
    acc_all *= 100
    print(f"Failed sims: {failed_sims}")


    # plot
    for aidx, alpha100 in enumerate(alpha100_ls):
        for gidx, g100 in enumerate(g100_ls):  

            if control == "num_train":  
                #for idx in range(len(X_clusters_ls[1:2])):    
                for idx in range(len(X_clusters_ls[0:1])):    
                    # mean                            
                    axs[gidx].plot(num_train_ls, acc_all[aidx,gidx,idx,:,0], c=c_ls[aidx])
                    # standard deviation
                    axs[gidx].fill_between(num_train_ls, acc_all[aidx,gidx,idx,:,0] - acc_all[aidx,gidx,idx,:,1], acc_all[aidx,gidx,idx,:,0] + acc_all[aidx,gidx,idx,:,1], 
                                           color = c_ls[aidx], alpha=0.2)                    
            else:
                #for idx in range(len(num_train_ls[1:2])):    
                for idx in range(len(num_train_ls[0:1])):    
                    # mean                            
                    axs[gidx].plot(X_clusters_ls, acc_all[aidx,gidx,:,iidx,0], c=c_ls[aidx])
                    # standard deviation
                    axs[gidx].fill_between(X_clusters_ls, acc_all[aidx,gidx,:,iidx,0] - acc_all[aidx,gidx,:,iidx,1], acc_all[aidx,gidx,:,iidx,0] + acc_all[aidx,gidx,:,iidx,1], 
                                           color = c_ls[aidx], alpha=0.2)                                  

        
    # general plot setting
    for i in range(len(axs)):
        # ticks
        """
        axs[i].tick_params(axis='both',labelsize=tick_size)
        # major ticks
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xtick_ls)
        """

        # setting ticks
        axs[i].tick_params(bottom=True, top=False, left=True, right=False)
        axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

        axs[i].tick_params(axis="x", direction="out")
        axs[i].tick_params(axis="y", direction="out")


    print(f"Time to plot: {time() - t0}s")
    dataname = "gaussian"
    plt.tight_layout()
    if display:
        plt.show()
    else:
        fig1_path = "/project/PDLAI/project2_data/figure_ms"
        plt.savefig(f"{fig1_path}/{fcn}_{dataname}_{acc_type}_epoch={epoch}_control={control}.pdf", bbox_inches='tight')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

