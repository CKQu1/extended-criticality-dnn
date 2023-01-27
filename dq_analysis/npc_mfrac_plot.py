import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.io as sio
import sys
from ast import literal_eval
from os.path import join

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from path_names import root_data, model_log, id_to_path

plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.switch_backend('agg')

"""

Functions for demonstrating multifractality and associated quantities of 
the neural representation principal components (NPCs).
Note that all the figures generated are 1 by 1, this is for the convenience of adjusting the figures to the manuscript.

"""

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

"""

i_pc   : the ith PC
pc_type: the type of input data passed into the MLPs for analysis, i.e. either train or test data

"""

def dq_vs_q(i_pc=0, train=True):

    i_pc = int(i_pc)
    train = train if isinstance(train, bool) else literal_eval(train)
    pc_type = "train" if train else "test"

    title_size = 23.5 * 1.5
    tick_size = 23.5 * 1.5
    label_size = 23.5 * 1.5
    axis_size = 23.5 * 1.5
    legend_size = 23.5 * 1.5 - 5

    c_ls = list(mcl.TABLEAU_COLORS.keys())
    linestyle_ls = ["-", "--", ":"]
    marker_ls = ["o","^","+"]

    #selected_path = "/project/dnn_maths/project_qu3/fc10_pcdq"
    #selected_path = "/project/dnn_maths/project_qu3/fc10_momentum"
    selected_path = join(root_data, "trained_mlps/fc10_pcdq")
    net_ls = [net[0] for net in os.walk(selected_path) if "epochs=650" in net[0]]
    #net_ls.pop(0)

    print(selected_path)
    print(f"Total of networks: {len(net_ls)}.")

    # epoch network was trained till
    epoch_last = 650

    alpha100_ls = [100,200]
    g100_ls = [25,100,300]
    q_ls = np.linspace(0,2,50)

    #for epoch_plot in [1,5,10,20,100,250,500,650]:
    #for epoch_plot in [1,5,10] + list(range(50,651,50)):
    for epoch_plot in [1,50,100,650]:
        Dqss = np.zeros([len(alpha100_ls),len(g100_ls),len(q_ls)])
        Dqss_std = np.zeros([len(alpha100_ls),len(g100_ls),len(q_ls)])
        alpha_m_ls = []
        good = 0
        for gidx in range(len(g100_ls)):
            # set up plot
            fig, ax1 = plt.subplots(1, 1 ,figsize=(9.5,7.142))
            axs = [ax1]

            for i in range(len(axs)):            
                if i == 0:
                    axs[i].set_ylabel(r'$D_q$', fontsize=label_size)

                #if gidx != 0:
                axs[i].set_xlabel(r'$q$', fontsize=label_size)
                #axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)

                # setting ticks
                axs[i].set_xlim(0,2)
                axs[i].set_xticks(np.arange(0,2.1,0.5))
                axs[i].set_xticklabels(np.arange(0,2.1,0.5))

                axs[i].tick_params(bottom=True, top=False, left=True, right=False)
                axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                #axs[i].tick_params(axis='both',labelsize=tick_size)

                axs[i].tick_params(axis="x", direction="out", labelsize=tick_size)
                axs[i].tick_params(axis="y", direction="out", labelsize=tick_size)            

                # set log axis for x
                #axs[i].set_xscale('log')
                #axs[i].set_yscale('log')

            g100 = g100_ls[gidx]
            g = int(g100/100)
            for aidx in range(len(alpha100_ls)):
                alpha100 = alpha100_ls[aidx]
                alpha = int(alpha100/100)
                
                net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath in npath][0]
                if len(net_path) == 0:
                    print(f"({alpha100},{g100}) not trained!")

                #model_id = get_model_id(net_path)
                try:
                    C_dims = np.loadtxt(f"{net_path}/C_dims")
                    pc_dqss = np.loadtxt(f"{net_path}/pc_dqss_{pc_type}_{epoch_plot}")
                    start = i_pc
                    total_layers = 0
                    # exclude final layer due to only 10 neurons are present
                    for cidx in range(len(C_dims) - 1): 
                        end = start + int(C_dims[cidx])                            
                        Dqss[aidx,gidx,:] += np.mean(pc_dqss[start:start + 1,:], axis=0)
                        Dqss_std[aidx,gidx,:] += np.mean(pc_dqss[start:start + 1,:]**2 , axis=0)
                        start = end
                        total_layers += 1

                    #model_info = model_log(model_id)
                    alpha100, g100 = get_alpha_g(net_path)
                    alpha, g = int(alpha100)/100, int(g100)/100

                    #print((alpha100,g100))
                    #print(net_path)
                    alpha_m_ls.append((alpha, g))
                    good += 1     

                except (FileNotFoundError, OSError) as error:
                    # use the following to keep track of what to re-run
                    #print((alpha100, g100))
                    print(net_path)

                dq_mean = Dqss[aidx,gidx,:]/total_layers
                lower = dq_mean - np.sqrt( Dqss_std[aidx,gidx,:]/total_layers - dq_mean**2 )
                upper = dq_mean*2 - lower         
                axs[0].plot(q_ls,dq_mean,linewidth=2.5,alpha=1,c = c_ls[aidx],label=rf"$\alpha$ = {alpha}")

                # standard deviation
                axs[0].plot(q_ls,lower,linewidth=0.25, alpha=1,c = c_ls[aidx])
                axs[0].plot(q_ls,upper,linewidth=0.25, alpha=1,c = c_ls[aidx])
                axs[0].fill_between(q_ls, lower, upper, color = c_ls[aidx], alpha=0.2)               

            axs[0].set_ylim(-0.05,1.05)
            #axs[0].set_title(r"$D_w^{1 / \alpha}$" + f" = {g}", fontsize=title_size)  
            if gidx == 0 or gidx == 1:
                axs[0].legend(fontsize=legend_size, loc="lower left", frameon=False) 

            print(f"Epoch {epoch_plot}")
            print(f"Good: {good}")
            #print("\n")
            #print(len(net_ls))
            #print(len(alpha_m_ls))

            plt.tight_layout()
            #net_type = model_info.loc[model_info.index[0],'net_type']
            #depth = int(model_info.loc[model_info.index[0],'depth'])
            net_type = "fc"
            depth = 10
            plot_path = join(root_data, f"figure_ms/{net_type}{depth}_pcmfrac")
            if not os.path.isdir(plot_path): os.makedirs(plot_path)    
            plt.savefig(f"{plot_path}/{net_type}{depth}_mnist_epoch={epoch_plot}_g100={g100}_pcmfrac={i_pc}_{pc_type}_single.pdf", bbox_inches='tight')
            #plt.show()


def d2_vs_depth(i_pc=0, train=True):

    i_pc = int(i_pc)
    train = train if isinstance(train, bool) else literal_eval(train)
    pc_type = "train" if train else "test"

    title_size = 23.5 * 1.5
    tick_size = 23.5 * 1.5
    label_size = 23.5 * 1.5
    axis_size = 23.5 * 1.5
    legend_size = 23.5 * 1.5 - 5

    #c_ls = ["tab:blue", "tab:orange", "tab:green"]
    c_ls = list(mcl.TABLEAU_COLORS.keys())
    linestyle_ls = ["-", "--", ":"]
    marker_ls = ["o","^","+"]
    label_ls = ['(a)', '(b)', '(c)', '(d)']

    #selected_path = "/project/dnn_maths/project_qu3/fc10_pcdq"
    #selected_path = "/project/dnn_maths/project_qu3/fc10_momentum"
    selected_path = join(root_data, "trained_mlps/fc10_pcdq")
    net_ls = [net[0] for net in os.walk(selected_path) if "epochs=650" in net[0]]
    #net_ls.pop(0)

    print(selected_path)
    print(f"Total of networks: {len(net_ls)}.")

    # epoch network was trained till
    epoch_last = 650

    alpha100_ls = [100,200]
    g100_ls = [25,100,300]
    q_ls = np.linspace(0,2,50)

    # get depth of network
    alpha100, g100 = 100, 100
    net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath in npath][0]
    total_depth = len(np.loadtxt(f"{net_path}/C_dims")) - 1   # excluding final layer as it only has 10 neurons for classification, this includes pre and post activation
    #for epoch_plot in [1,50,100,650]:
    for epoch_plot in [0,650]:
        D2ss = np.zeros([len(alpha100_ls),len(g100_ls),total_depth])
        #D2ss_std = np.zeros([len(alpha100_ls),len(g100_ls),total_depth])
        alpha_m_ls = []
        good = 0
        for gidx in range(len(g100_ls)):
            # set up plot
            fig = plt.figure(figsize=(9.5,7.142))

            plt.ylabel(r'$D_2$', fontsize=label_size)

            #if gidx != 0:
            plt.xlabel('Depth', fontsize=label_size)

            # setting ticks
            plt.xlim(1,total_depth)
            #plt.xticks(np.arange(0,2.1,0.5))

            plt.tick_params(bottom=True, top=False, left=True, right=False)
            plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            #plt.tick_params(axis='both',labelsize=tick_size)

            plt.tick_params(axis="x", direction="out", labelsize=tick_size)
            plt.tick_params(axis="y", direction="out", labelsize=tick_size)            

            g100 = g100_ls[gidx]
            g = int(g100/100)
            for aidx in range(len(alpha100_ls)):
                alpha100 = alpha100_ls[aidx]
                alpha = int(alpha100/100)
                
                net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath in npath][0]
                if len(net_path) == 0:
                    print(f"({alpha100},{g100}) not trained!")

                #model_id = get_model_id(net_path)
                try:
                    C_dims = np.loadtxt(f"{net_path}/C_dims")
                    pc_dqss = np.loadtxt(f"{net_path}/pc_dqss_{pc_type}_{epoch_plot}")
                    start = i_pc
                    total_layers = 0
                    # exclude final layer due to only 10 neurons are present
                    for l in range(len(C_dims) - 1): 
                        end = start + int(C_dims[l])                            
                        D2ss[aidx,gidx,l] += np.mean(pc_dqss[start:start + 1,-1], axis=0)
                        #D2ss_std[aidx,gidx,l] += np.mean(pc_dqss[start:start + 1,-1]**2 , axis=0)
                        start = end
                        total_layers += 1

                    #model_info = model_log(model_id)
                    alpha100, g100 = get_alpha_g(net_path)
                    alpha, g = int(alpha100)/100, int(g100)/100

                    #print((alpha100,g100))
                    #print(net_path)
                    alpha_m_ls.append((alpha, g))
                    good += 1     

                except (FileNotFoundError, OSError) as error:
                    # use the following to keep track of what to re-run
                    #print((alpha100, g100))
                    print(net_path)

                #lower = dq_mean - np.sqrt( Dqss_std[aidx,gidx,:]/total_layers - dq_mean**2 )
                #upper = dq_mean*2 - lower         
                plt.plot(np.arange(1, total_depth+1) ,D2ss[aidx,gidx,:],linewidth=2.5,alpha=1,c = c_ls[aidx],label=rf"$\alpha$ = {alpha}")

                # standard deviation
                #plt.plot(q_ls,lower,linewidth=0.25, alpha=1,c = c_ls[aidx])
                #plt.plot(q_ls,upper,linewidth=0.25, alpha=1,c = c_ls[aidx])
                #plt.fill_between(q_ls, lower, upper, color = c_ls[aidx], alpha=0.2)               

            plt.ylim(-0.05,1.05)
            #plt.title(r"$D_w^{1 / \alpha}$" + f" = {g}", fontsize=title_size)  
            if gidx == 0 or gidx == 1:
                plt.legend(fontsize=legend_size, ncol=2, loc="upper right", frameon=False) 

            print(f"Epoch {epoch_plot}")
            print(f"Good: {good}")
            #print("\n")
            #print(len(net_ls))
            #print(len(alpha_m_ls))

            plt.tight_layout()
            #net_type = model_info.loc[model_info.index[0],'net_type']
            #depth = int(model_info.loc[model_info.index[0],'depth'])
            net_type = "fc"
            depth = 10
            plot_path = join(root_data, f"figure_ms/{net_type}{depth}_pcmfrac")
            if not os.path.isdir(plot_path): os.makedirs(plot_path)    
            plt.savefig(f"{plot_path}/{net_type}{depth}_mnist_epoch={epoch_plot}_g100={g100}_pcmfrac={i_pc}_{pc_type}_d2-vs-depth.pdf", bbox_inches='tight')
            #plt.show()


def d2_vs_ed(i_pc=0, train=True):

    i_pc = int(i_pc)
    train = train if isinstance(train, bool) else literal_eval(train)
    pc_type = "train" if train else "test"

    #title_size = 26.5
    #tick_size = 26.5
    #label_size = 26.5
    #axis_size = 26.5
    #legend_size = 23.5
    title_size = 23.5 * 1.5
    tick_size = 23.5 * 1.5
    label_size = 23.5 * 1.5
    axis_size = 23.5 * 1.5
    legend_size = 23.5 * 1.5 - 5

    #c_ls = ["tab:blue", "tab:orange", "tab:green"]
    c_ls = list(mcl.TABLEAU_COLORS.keys())
    linestyle_ls = ["-", "--", ":"]
    marker_ls = ["o","^","+"]
    label_ls = ['(a)', '(b)', '(c)', '(d)']

    #selected_path = path_names.fc_path
    #selected_path = "/project/dnn_maths/project_qu3/fc10_pcdq"
    #selected_path = "/project/dnn_maths/project_qu3/fc10_momentum"
    selected_path = join(root_data, "trained_mlps/fc10_pcdq")
    net_ls = [net[0] for net in os.walk(selected_path) if "epochs=650" in net[0]]
    #net_ls.pop(0)

    print(selected_path)
    print(f"Total of networks: {len(net_ls)}.")

    # epoch network was trained till
    epoch_last = 650

    alpha100_ls = [100,200]
    g100_ls = [25,100,300]
    q_ls = np.linspace(0,2,50)

    # get depth of network
    alpha100, g100 = 100, 100
    net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath in npath][0]
    total_depth = len(np.loadtxt(f"{net_path}/C_dims")) - 1   # excluding final layer as it only has 10 neurons for classification, this includes pre and post activation
    #for epoch_plot in [1,50,100,650]:
    for epoch_plot in [0,650]:
        D2ss = np.zeros([len(alpha100_ls),len(g100_ls),total_depth])
        #D2ss_std = np.zeros([len(alpha100_ls),len(g100_ls),total_depth])
        alpha_m_ls = []
        good = 0
        for gidx in range(len(g100_ls)):
            # set up plot
            fig = plt.figure(figsize=(9.5,7.142))

            plt.ylabel(r'$D_2$', fontsize=label_size)

            #if gidx != 0:
            plt.xlabel('ED', fontsize=label_size)

            # setting ticks
            plt.xlim(1,total_depth)
            #plt.xticks(np.arange(0,2.1,0.5))

            plt.tick_params(bottom=True, top=False, left=True, right=False)
            plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            #plt.tick_params(axis='both',labelsize=tick_size)

            plt.tick_params(axis="x", direction="out", labelsize=tick_size)
            plt.tick_params(axis="y", direction="out", labelsize=tick_size)            

            g100 = g100_ls[gidx]
            g = int(g100/100)
            for aidx in range(len(alpha100_ls)):
                alpha100 = alpha100_ls[aidx]
                alpha = int(alpha100/100)
                
                net_path = [npath for npath in net_ls if f"_{alpha100}_{g100}_" in npath in npath][0]
                if len(net_path) == 0:
                    print(f"({alpha100},{g100}) not trained!")

                #model_id = get_model_id(net_path)
                try:
                    C_dims = np.loadtxt(f"{net_path}/C_dims")
                    EDs = np.loadtxt(join(net_path, f"ed_{pc_type}"))
                    pc_dqss = np.loadtxt(f"{net_path}/pc_dqss_{pc_type}_{epoch_plot}")
                    start = i_pc
                    total_layers = 0
                    # exclude final layer due to only 10 neurons are present
                    for l in range(len(C_dims) - 1): 
                        end = start + int(C_dims[l])                            
                        D2ss[aidx,gidx,l] += np.mean(pc_dqss[start:start + 1,-1], axis=0)
                        #D2ss_std[aidx,gidx,l] += np.mean(pc_dqss[start:start + 1,-1]**2 , axis=0)
                        start = end
                        total_layers += 1

                    #model_info = model_log(model_id)
                    alpha100, g100 = get_alpha_g(net_path)
                    alpha, g = int(alpha100)/100, int(g100)/100

                    #print((alpha100,g100))
                    #print(net_path)
                    alpha_m_ls.append((alpha, g))
                    good += 1     

                except (FileNotFoundError, OSError) as error:
                    # use the following to keep track of what to re-run
                    #print((alpha100, g100))
                    print(net_path)

                #lower = dq_mean - np.sqrt( Dqss_std[aidx,gidx,:]/total_layers - dq_mean**2 )
                #upper = dq_mean*2 - lower         
                plt.scatter(EDs[:-1, epoch_plot] ,D2ss[aidx,gidx,:],linewidth=2.5,alpha=1,c = c_ls[aidx],label=rf"$\alpha$ = {alpha}")

                # standard deviation
                #plt.plot(q_ls,lower,linewidth=0.25, alpha=1,c = c_ls[aidx])
                #plt.plot(q_ls,upper,linewidth=0.25, alpha=1,c = c_ls[aidx])
                #plt.fill_between(q_ls, lower, upper, color = c_ls[aidx], alpha=0.2)               

            plt.ylim(-0.05,1.05)
            #plt.title(r"$D_w^{1 / \alpha}$" + f" = {g}", fontsize=title_size)  
            if gidx == 0 or gidx == 1:
                plt.legend(fontsize=legend_size, ncol=2, loc="upper right", frameon=False) 

            print(f"Epoch {epoch_plot}")
            print(f"Good: {good}")
            #print("\n")
            #print(len(net_ls))
            #print(len(alpha_m_ls))

            plt.tight_layout()
            #net_type = model_info.loc[model_info.index[0],'net_type']
            #depth = int(model_info.loc[model_info.index[0],'depth'])
            net_type = "fc"
            depth = 10
            plot_path = join(root_data, f"figure_ms/{net_type}{depth}_pcmfrac")
            if not os.path.isdir(plot_path): os.makedirs(plot_path)    
            #plt.savefig(f"{plot_path}/{net_type}{depth}_mnist_epoch={epoch_plot}_g100={g100}_pcmfrac={i_pc}_{pc_type}_d2-vs-ed.pdf", bbox_inches='tight')
            plt.show()



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])