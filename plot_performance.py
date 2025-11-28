# import cmcrameri as cmc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import scipy.io as sio
import sys
from ast import literal_eval
from itertools import product
# from matplotlib.ticker import AutoMinorLocator
from string import ascii_lowercase
from tqdm import tqdm
sys.path.append(os.getcwd())
import constants
from os.path import join, isdir
from os import makedirs
from constants import DROOT, id_to_path, model_log, HYP_CNORM, HYP_CMAP
from utils_dnn import load_transition_lines
from UTILS.mutils import njoin

# colorbar setting
cm_lib = "sns"
cmaps = []
if cm_lib == "plt":
    for cm_type in ['Spectral', 'gist_stern', 'RdGy']:
        cmaps.append(plt.cm.get_cmap(cm_type))
elif cm_lib == "sns":
    import seaborn as sns
    for cm_type in ['Spectral', 'RdBu', 'rocket_r']:
    #for cm_type in ['plasma', 'vlag', 'icefire']:
    #for cm_type in ['batlow', 'cividis', 'thermal']:
        #cmaps.append(sns.color_palette(cm_type, as_cmap=True))
        #cmaps.append(sns.color_palette(cm_type))
        cmaps.append(plt.cm.get_cmap(cm_type))
# custom
else:
    """
    for cm_type in [[500,200], [600,100], [20000,25]]:
        cmaps.append(sns.diverging_palette(cm_type[0], cm_type[1], as_cmap=True))
    """
    #cmaps = [cmc.cm.batlow, plt.cm.get_cmap("cividis"), plt.cm.get_cmap("vlag")]
    #cmaps = [cmc.cm.fes, cmc.cm.oleron, cmc.cm.bukavu]
    #cmaps = [cmc.cm.lapaz, cmc.cm.imola, cmc.cm.acton]
    pass

#interp = "quadric"
interp = "none"
#interp = "spline16"
#cm = cm.get_cmap('plasma')
plt.rcParams["font.family"] = 'sans-serif'     # set plot font globally

# plot settings
tick_size = 7
label_size = 7
axis_size = 7
legend_size = 7
linewidth = 0.8

# (alpha, sigma_w) grid setting
mult_delta = 0.2
mult_lower = 0.2
mult_upper = 3
mult_N = int(round((mult_upper - mult_lower)/mult_delta)) + 1
mult_grid = np.linspace(mult_lower,mult_upper,mult_N)
#mult_incre = round(mult_grid[1] - mult_grid[0],2)
mult_incre = mult_delta
alpha_delta = 0.05
alpha_lower = 1
alpha_upper = 2
alpha_N = int((alpha_upper - alpha_lower)/alpha_delta + 1)
alpha_grid = np.linspace(alpha_lower,alpha_upper,alpha_N)
#alpha_incre = round(alpha_grid[1] - alpha_grid[0],1)
alpha_incre = alpha_delta

centers = [alpha_lower, alpha_upper, mult_lower, mult_upper]
dx, = np.diff(centers[:2])/(alpha_N-1)
dy, = -np.diff(centers[2:])/(mult_N-1)
extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

# get batch of networks trained initialized based on the phase transition diagram
def get_mlp_ls(path):
    net_folder = path.split("/")[-1] if len(path.split("/")[-1]) != 0 else path.split("/")[-2]
    dataname = "gaussian_data" if "gaussian_data" in net_folder else "mnist"
    fcn = net_folder.split("_")[0]

    net_ls = [join(path,net) for net in next(os.walk(path))[1] if fcn in net and "epochs=" in net]
    net_setup = pd.read_csv(njoin(net_ls[0],"log"))
    activation = net_setup.loc[0,"activation"]
    optimizer = net_setup.loc[0,"optimizer"]
    epoch_last = int(net_setup.loc[0,"epochs"])
    
    return net_ls, activation, dataname, epoch_last, optimizer, fcn


def get_cnn_ls(path, net_type='cnn5'):

    net_ls = [join(path,net) for net in next(os.walk(path))[1] if net_type in net]
    
    return net_ls


def get_accloss(net_path, acc_type):
    global acc_loss

    #net_params = pd.read_csv(f"{net_path}/net_log.csv")
    # net_params = pd.read_csv(f"{net_path}/log")
    # alpha100 = int(net_params.loc[0,'alpha100'])
    # g100 = int(net_params.loc[0,'g100'])

    acc_loss = pd.read_csv(f"{net_path}/acc_loss", index_col=0)
    metrics = acc_loss.iloc[:,0:2] if acc_type=="train" else acc_loss.iloc[:,2:]           
    losses = metrics.iloc[:,0]   
    accs = metrics.iloc[:,1] * 100      

    return accs, losses


def get_net_paths(net_ls, alpha100, g100):
    net_paths = []
    for net_path in net_ls:
        if f'_{alpha100}_{g100}_' in net_path:
            net_paths.append(net_path)

    return net_paths

# get the phase transition for accuracy, loss and earliest epoch reaching accuracy threhold
def get_accloss_phase(path, net_ls, epoch, epoch_last, acc_type, acc_threshold):

    global acc_mesh

    alpha_m_ls, acc_ls, loss_ls, early_ls = [], [], [], []
    good = 0
    # failed jobs
    failed_jobs = []
    # accuracy grid
    acc_mesh = np.zeros((mult_N,alpha_N))
    loss_mesh = np.zeros((mult_N,alpha_N))
    early_mesh = np.zeros((mult_N,alpha_N))
    for i in tqdm(range(len(net_ls))):
        try:
            net_path = net_ls[i]  

            #net_params = pd.read_csv(f"{net_path}/net_log.csv")
            net_params = pd.read_csv(f"{net_path}/log")
            alpha100 = int(net_params.loc[0,'alpha100'])
            g100 = int(net_params.loc[0,'g100'])
            alpha = alpha100/100
            mult = g100/100

            acc_loss = pd.read_csv(f"{net_path}/acc_loss")
            metrics = acc_loss.iloc[:,0:2] if acc_type=="train" else acc_loss.iloc[:,2:]          
            good_ls = [x for x,y in enumerate(100*metrics.iloc[:,1]) if y > acc_threshold]
            early_epoch = epoch_last if len(good_ls) == 0 else min(good_ls)    
            loss = metrics.iloc[epoch-1,0]   
            acc = metrics.iloc[epoch-1,1] * 100  

            good += 1     

        except (FileNotFoundError, OSError) as error:
            # use the following to keep track of what to re-run

            acc, loss = np.nan, np.nan
            early_epoch = np.nan
            failed_jobs.append((alpha100,g100))

        alpha_m_ls.append((alpha,mult))
        acc_ls.append(acc)
        loss_ls.append(loss)
        early_ls.append(early_epoch) 

        x_loc = int(round((mult_upper - mult) / mult_incre))
        y_loc = int(round((alpha - alpha_lower) / alpha_incre))

        #print(f'{(alpha, mult)}: acc = {acc}')  # delete
        acc_mesh[x_loc,y_loc] = acc
        loss_mesh[x_loc,y_loc] = loss
        early_mesh[x_loc,y_loc] = early_epoch

    return alpha_m_ls, acc_ls, loss_ls, early_ls, acc_mesh, loss_mesh, early_mesh, failed_jobs

def combined_phase(acc_type, model_paths, seeds='0,1,2,3,4', acc_threshold=93, display=False):
    '''
    Includes accuracy, loss and stopping epoch for MLP (main text plot)
        - acc_type (str): test or train
        - path: /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid
        - acc_threshold (int): the accuracy threshold to determine speed of training
    '''

    global acc_ls, alpha_m_ls, acc_mesh, cmap_bd, axs
    global net_ls, dataname, epoch_last, fcn, early_ls, mesh_ls

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    acc_threshold = float(acc_threshold)
    display = literal_eval(display) if isinstance(display,str) else display

    seeds = [int(seed) for seed in seeds.split(',')]

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    fig, axs = plt.subplots(2, 3,sharex = True,sharey=True,figsize=(7.5,4))     # without text

    model_paths =model_paths.split(',')
    fig_ii = 0
    for model_ii, model_path in enumerate(model_paths):
        paths = []
        for subdir in os.listdir(model_path):
            if int(subdir.split('seed=')[-1]) in seeds:
                paths.append(njoin(model_path, subdir))
        net_lss = []
        for path in paths:
            net_ls, activation, dataname, epoch_last, optimizer, fcn = get_mlp_ls(path)
            net_lss.append(net_ls)
        
        print(f"Total networks {len(net_ls)}, activation: {activation}, dataname: {dataname}, epoch_last: {epoch_last}, optimizer: {optimizer}, fcn: {fcn}")

        # phase transition lines
        #bound1, boundaries = load_transition_lines()

        # axs = axs.flat
        axs = axs[model_ii]

        # line_cols_2 = ["gray", "gray", "gray"]
        # line_cols_1 = ['k', 'k', 'k']
        # # plot boundaries for each axs
        # for i in range(len(axs)):
        #     axs[i].plot(bound1.iloc[:,0], bound1.iloc[:,1], line_cols_1[i])
        #     for j in range(len(boundaries)):
        #         bd = boundaries[j]
        #         axs[i].plot(bd.iloc[:,0], bd.iloc[:,1], c=line_cols_2[i], linestyle='--')  

        # plot points which computations where executed
        a_cross, m_cross = np.meshgrid(alpha_grid, mult_grid)

        xticks = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        xtick_ls = []
        for xidx, xtick in enumerate(xticks):
            if xidx % 2 == 0:
                xtick_ls.append(str(xtick))
            else:
                xtick_ls.append('')


        # title_ls = ['Test accuracy', 'Earliest epoch reaching' + '\n' + 'test acc. threshold ']
        title_ls = ['Top-1 acc.', 'Loss', 'Stopping epoch']
        for i in range(len(axs)):
            # network realizations
            #if i == 0:
            #    axs[i].plot(a_cross, m_cross, c='k', linestyle='None',marker='.',markersize=5)
            #    axs[i].grid()

            # ticks
            axs[i].tick_params(axis='both',labelsize=tick_size)
            # major ticks
            axs[i].set_xticks(xticks)
            axs[i].set_xticklabels(xtick_ls)

            if model_ii == 0:
                axs[i].set_title(f"{title_ls[i]}", fontsize=axis_size)
            if model_ii == len(model_paths) - 1:
                axs[i].set_xlabel(r'$\alpha$', fontsize=axis_size)

            # adding labels
            label = ascii_lowercase[fig_ii]
            fig_ii += 1
            axs[i].text(-0.1, 1.2, rf'$\mathbf{{{label}}}$', transform=axs[i].transAxes, fontsize=label_size, va='top', ha='right')   # fontweight='bold'

            # setting ticks
            axs[i].tick_params(bottom=True, top=False, left=True, right=False)
            axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

            axs[i].tick_params(axis="x", direction="out")
            axs[i].tick_params(axis="y", direction="out")

            # minor ticks
            #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            #axs[i].yaxis.set_minor_locator(AutoMinorLocator())

        epoch = epoch_last
        #epoch = 5

        acc_meshs, loss_meshs, early_meshs = [], [], []
        for path_idx, path in enumerate(paths):

            alpha_m_ls, acc_ls, loss_ls, early_ls, acc_mesh, loss_mesh, early_mesh, failed_jobs = get_accloss_phase(path, net_lss[path_idx], epoch, epoch_last, acc_type, acc_threshold)
            good = np.prod(acc_mesh.shape) - len(failed_jobs)

            # print failed jobs
            print(f"{path} Failed jobs: {len(failed_jobs)}")
            if len(failed_jobs) > 0:
                print(failed_jobs)

            # if want to use imshow convert to grid form
            assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

            acc_meshs.append(acc_mesh)
            loss_meshs.append(loss_mesh)
            early_meshs.append(early_mesh)

        acc_meshs = np.stack(acc_meshs)
        loss_meshs = np.stack(loss_meshs)
        early_meshs = np.stack(early_meshs)

        # colourmap bounds
        #perc_l, perc_u = 5, 95
        perc_l, perc_u = 10, 90
        #perc_l, perc_u = 15, 85
        #perc_l, perc_u = 30, 70
        cmap_bd = []
        for ls_idx, ls in enumerate([acc_ls, loss_ls, early_ls]):
            #map_bd.append([np.percentile(ls,10),np.percentile(ls,90)])
            #cmap_bd.append([np.percentile(ls,5),np.percentile(ls,95)])
            if ls_idx < 2:
                cmap_bd.append([np.nanpercentile(ls,perc_l),np.nanpercentile(ls,perc_u)])        
            else:
                #cmap_bd.append([0,epoch_last])
                cmap_bd.append([np.nanpercentile(ls,perc_l),np.nanpercentile(ls,perc_u)])

        # plot results
        mesh_ls = [acc_meshs, loss_meshs, early_meshs]
        for pidx in range(3):
            #extent=[alpha_lower,alpha_upper,mult_lower,mult_upper]
            plot = axs[pidx].imshow(np.median(mesh_ls[pidx], axis=0),extent=extent, 
                                    vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=cmaps[pidx], 
                                    interpolation=interp, aspect='auto',
                                    origin='upper')

            # axs[pidx].xticks(np.arange(0, mesh_ls[pidx].shape[1]+1, 1))
            # axs[pidx].yticks(np.arange(0, mesh_ls[pidx].shape[0]+1, 1))


            cbar = plt.colorbar(plot,ax=axs[pidx])
            cbar.ax.tick_params(labelsize=tick_size)
            # Make border invisible
            cbar.outline.set_visible(False)

            """
            if pidx == 2:
                if fcn == "fc10":
                    cbar.set_ticks(list(range(100,601,100)))
                    cbar.set_ticklabels(list(range(100,601,100)))
                else:
                    cbar.set_ticks(list(range(5,21,5)))
                    cbar.set_ticklabels(list(range(5,21,5)))
            """

    plt.tight_layout()
    if display:
        plt.show()
    else:
        fig1_path = join(DROOT, 'figure_ms', 'performance_phase')
        if not isdir(fig1_path): makedirs(fig1_path)
        plt.savefig(f"{fig1_path}/combined_median_performance.pdf", bbox_inches='tight')

    print("Figure 1")
    print(f"Trained networks: {len(net_ls)}")
    print(f"(alpha, g) pair: {len(alpha_m_ls)}")
    print(f"Good: {good}")    


def epochs_all(alpha100s, g100s, acc_type, path, display=False):
    """
    Plots the all epoch accuracy/loss for specified (\alpha, \sigma_w)
    """

    global net_ls, dataname, epoch_last, fcn, net_paths, net_path, alpha, g, accs_all, losses_all

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    alpha100s = literal_eval(alpha100s); g100s = literal_eval(g100s)
    display = literal_eval(display) if isinstance(display,str) else display

    lstyles = ['-', '-.', '--']
    net_ls = get_cnn_ls(path)
    print(f'path: {path}')

    nrows, ncols = 1, 3
    figsize = (5,2)
    fig, axs = plt.subplots(nrows,ncols,sharex = True,sharey=True,figsize=figsize)     # without text
    axs = axs.flat

    accs_all, losses_all = [], []
    for (aidx, alpha100), (gidx, g100) in product(enumerate(alpha100s), enumerate(g100s)):
        
        alpha, g = alpha100/100, g100/100

        net_paths = get_net_paths(net_ls, alpha100, g100)
        net_path = net_paths[0]
        accs, losses = get_accloss(net_path, acc_type)
        accs_all.append(accs); losses_all.append(losses)

        axis = axs[gidx]
        axis.plot(accs, 
                  c=HYP_CMAP(HYP_CNORM(alpha)), linestyle=lstyles[gidx], 
                  label=rf'$\alpha$ = {alpha}')  # label=rf'$\sigma_w$ = {g}'
        axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)             

    axs[0].legend(frameon=False)
    #axs[1].set_yticklabels([])

    plt.tight_layout()
    if display:
        plt.show()
    else:
        fig1_path = njoin(DROOT, 'figure_ms')
        if not isdir(fig1_path):  makedirs(fig1_path)
        file_name = f'epochs_{acc_type}.pdf'
        plt.savefig(join(fig1_path, file_name), bbox_inches='tight')

        print(f"Figure saved: {join(fig1_path, file_name)}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])