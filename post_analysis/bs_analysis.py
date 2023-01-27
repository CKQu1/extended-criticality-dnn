import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
from os.path import join

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

import path_names
from path_names import root_data, model_log

plt.rcParams["font.family"] = "serif"     # set plot font globally
plt.switch_backend('agg')

"""
tick_size = 12
label_size = 12
axs_size = 12
legend_size = 12
text_size = 12
linewidth = 0.8
"""

"""
tick_size = 16.5
label_size = 16.5
axis_size = 16.5
legend_size = 14
linewidth = 0.8
"""

"""
tick_size = 18.5
label_size = 18.5
axis_size = 18.5
legend_size = 14
linewidth = 0.8
text_size = 14
"""

title_size = 23.5
tick_size = 23.5
label_size = 23.5
axis_size = 23.5
legend_size = 23.5

linestyle_ls = ["-", "--", ":"]
marker_ls = ["o","^","+"]
label_ls = ['(a)', '(b)', '(c)', '(d)']

selected_path = join(root_data, "trained_mlps", "bs_analysis")
#net_ls = [net[0] for net in os.walk(selected_path) if "epochs=650" in net[0]]
net_ls = os.listdir(selected_path)
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

alpha100_ls = [100,200]
g100_ls = [25,100,300]
trans_ls = np.linspace(0,1,len(g100_ls)+1)[::-1]

acc_type = "test"
#bs_ls = [2**p for p in range(3,11)]
bs_ls = [2**p for p in range(4,11)]
#for epoch_plot in [0,1,5,10,50,100,200,250,500,600,650]:
#for epoch_plot in [1,5,10,20,100,250,500,650]:
for epoch_plot in [50]:

    # set up plot
    nrows, ncols = 1,2
    fig, (ax1,ax2) = plt.subplots(nrows, ncols,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))
    axs = [ax1, ax2]

    for i in range(len(axs)):
        
        if i == 0:
            axs[i].set_ylabel(f'{acc_type[0].upper() + acc_type[1:]} accuracy', fontsize=axis_size)

        #if i == 2 or i == 3:
        
        #axs[i].set_xticks(alpha_grid)
        #axs[i].set_xlim(0.975,2.025)
        #axs[i].set_yticks(np.arange(0.4,2.01,0.4))

        #if i == 2 or i == 3:
        axs[i].set_xlabel('Batch size', fontsize=axis_size)
        #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

        # adding labels
        #label = label_ls[i] 
        #axs[i].text(-0.1, 1.2, '%s'%label, transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')       # fontweight='bold'

        # setting ticks
        axs[i].tick_params(bottom=True, top=True, left=True, right=True)
        axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

        axs[i].tick_params(axis="x", direction="in", labelsize=axis_size - 1)
        axs[i].tick_params(axis="y", direction="in", labelsize=axis_size - 1)

        # set log axis for x
        axs[i].set_xscale('log')
        #axs[i].set_yscale('log')

    alpha_m_ls = []
    good = 0

    #accs = np.zeros([len(alpha100_ls), len(bs_ls)])
    accs = np.zeros([5, len(alpha100_ls), len(bs_ls)])  # each setting was ran 5 times
    for aidx in range(len(alpha100_ls)):
        alpha100 = alpha100_ls[aidx]
        alpha = int(alpha100/100)
        for gidx in range(len(g100_ls)):
            g100 = g100_ls[gidx]
            g = int(g100/100)
            
            for bidx in range(len(bs_ls)):
                bs = bs_ls[bidx]
                #net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath and f"bs={bs}" in npath][0]
                net_paths = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath and f"bs={bs}" in npath]
                print(f"alpha100 = {alpha100}, g100 = {g100}, bs = {bs} trained networks: {len(net_paths)}")
                if len(net_paths) == 0:
                    print(f"({alpha100},{g100}) not trained!")

                for repeat, net_path in enumerate(net_paths):
                    model_id = get_model_id(net_path)
                    acc_path = join(selected_path, net_path, "acc_loss")
                    if os.path.isfile(acc_path):
                        acc_loss = pd.read_csv(acc_path)
                        model_info = model_log(model_id)
                        alpha100, g100 = get_alpha_g(net_path)
                        alpha, g = int(alpha100)/100, int(g100)/100

                        #print((alpha100,g100))
                        #print(net_path)
                        alpha_m_ls.append((alpha, g))

                        acc = acc_loss.iloc[epoch_plot,[1]]*100 if acc_type == "train" else acc_loss.iloc[epoch_plot,[3]]*100
                        accs[repeat,aidx,bidx] = acc

                        good += 1     

                    else:
                        # use the following to keep track of what to re-run
                        #print((alpha100, g100))
                        print(net_path)

            acc_mean = accs[:,aidx,:].mean(0)
            acc_std = accs[:,aidx,:].std(0)
            if aidx == 0:
                axs[aidx].plot(bs_ls,acc_mean,marker=marker_ls[gidx],markersize=10,label=rf"$g$ = {g}",linestyle=linestyle_ls[gidx],alpha=trans_ls[gidx])
            else:
                axs[aidx].plot(bs_ls,acc_mean,marker=marker_ls[gidx],markersize=10,linestyle=linestyle_ls[gidx],alpha=trans_ls[gidx])
        
            # standard deviation
            #axs[aidx].plot(bs_ls,acc_mean - acc_std,marker=marker_ls[gidx],markersize=10,linestyle=linestyle_ls[gidx])
            #axs[aidx].plot(bs_ls,acc_mean + acc_std,marker=marker_ls[gidx],markersize=10,linestyle=linestyle_ls[gidx])
            axs[aidx].errorbar(bs_ls, acc_mean, yerr=acc_std,alpha=trans_ls[gidx])

    
    axs[0].set_title(rf"$\alpha$ = {int(alpha100_ls[0]/100)}", fontsize=axis_size)
    axs[1].set_title(rf"$\alpha$ = {int(alpha100_ls[1]/100)}", fontsize=axis_size)  
    #axs[0].legend(fontsize=legend_size, loc="center right", frameon=False) 

    print(f"Epoch {epoch_plot}")
    print(f"Good: {good}")
    #print("\n")
    #print(len(net_ls))
    #print(len(alpha_m_ls))

    plt.tight_layout()

    net_type = model_info.loc[model_info.index[0],'net_type']
    depth = int(model_info.loc[model_info.index[0],'depth'])
    plot_path = f"{path_names.root_data}/figure_ms/{net_type}{depth}_bs"
    if not os.path.isdir(plot_path): os.makedirs(plot_path)    
    plt.savefig(f"{plot_path}/{net_type}{depth}_mnist_epoch={epoch_plot}_bs_analysis.pdf", bbox_inches='tight')

    #plt.show()
    #plt.clf()


