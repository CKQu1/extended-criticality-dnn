# import cmcrameri as cmc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import scipy.io as sio
import sys
from ast import literal_eval
# from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
sys.path.append(os.getcwd())
import constants
from os.path import join, isdir
from os import makedirs
from constants import DROOT, id_to_path, model_log
from utils_dnn import load_transition_lines

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
tick_size = 14.5
label_size = 15.5
axis_size = 15.5
legend_size = 14
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

# get the 132 batch of networks trained initialized based on the phase transition diagram
def get_net_ls(path):
    net_folder = path.split("/")[-1] if len(path.split("/")[-1]) != 0 else path.split("/")[-2]
    dataname = "gaussian_data" if "gaussian_data" in net_folder else "mnist"
    fcn = net_folder.split("_")[0]
    #path = f"{DROOT}/trained_mlps/fcn_grid/{fcn}_grid"
    if "fcn_grid" in path:
        net_ls = [join(path,net) for net in os.listdir(path) if "epoch650" in net]
        net_type = f"{fcn}_mnist_tanh"
        activation = "tanh"
        epoch_last = 650    # epoch network was trained till
        optimizer = "sgd"
    else:
        net_ls = [join(path,net) for net in next(os.walk(path))[1] if fcn in net and "epochs=" in net]
        net_setup = pd.read_csv(join(net_ls[0],"log"))
        activation = net_setup.loc[0,"activation"]
        optimizer = net_setup.loc[0,"optimizer"]
        epoch_last = int(net_setup.loc[0,"epochs"])
    
    return net_ls, activation, dataname, epoch_last, optimizer, fcn


def get_accloss(net_path, acc_type, net_type):
    if "fcn_grid" in net_path:         
        net_params = sio.loadmat(net_path + "/net_params_all.mat") 
        alpha = list(net_params['net_init_params'][0][0])[1][0][0]
        mult = list(net_params['net_init_params'][0][0])[2][0][0]

        acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
        metrics = acc_loss['training_history'] if acc_type=="train" else acc_loss['testing_history'] 
        losses = metrics[:,0]
        accs = metrics[:,1]
    else:
        #net_params = pd.read_csv(f"{net_path}/net_log.csv")
        net_params = pd.read_csv(f"{net_path}/log")
        alpha100 = int(net_params.loc[0,'alpha100'])
        g100 = int(net_params.loc[0,'g100'])
        alpha = alpha100/100
        mult = g100/100

        acc_loss = pd.read_csv(f"{net_path}/acc_loss")
        metrics = acc_loss.iloc[:,0:2] if acc_type=="train" else acc_loss.iloc[:,2:]           
        losses = metrics.iloc[:,0]   
        accs = metrics.iloc[:,1] * 100      

    return accs, losses


def get_net_paths(net_ls, alpha100, g100):
    net_paths = []
    is_fcn_grid = ("fcn_grid" in net_ls[0])
    for net_path in net_ls:
        if is_fcn_grid:
            alpha, g = alpha100/100, g100/100
            if f'stable{alpha}_{g}_' in net_path:
                net_paths.append(net_path)
        else:
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
            if "fcn_grid" in path:         
                net_params = sio.loadmat(net_path + "/net_params_all.mat") 
                alpha = list(net_params['net_init_params'][0][0])[1][0][0]
                mult = list(net_params['net_init_params'][0][0])[2][0][0]

                acc_loss = sio.loadmat(f"{net_path}/{net_type}_loss_log.mat")
                metrics = acc_loss['training_history'] if acc_type=="train" else acc_loss['testing_history']
                good_ls = [x for x,y in enumerate(metrics[:,1]) if y > acc_threshold]
                early_epoch = epoch_last if len(good_ls) == 0 else min(good_ls)    
                loss, acc = metrics[epoch-1,:]
            else:
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


def accloss_ensembles(acc_type, paths, acc_threshold=93, display=False):
    '''
    Includes accuracy, loss and stopping epoch for MLP (main text plot)
        - acc_type (str): test or train
        - path: /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid
        - acc_threshold (int): the accuracy threshold to determine speed of training
    '''

    global acc_ls, alpha_m_ls, acc_mesh, cmap_bd
    global net_ls, dataname, epoch_last, fcn, early_ls, mesh_ls

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    acc_threshold = float(acc_threshold)
    display = literal_eval(display) if isinstance(display,str) else display

    paths = paths.split(',')    
    print(paths)
    net_ls, activation, dataname, epoch_last, optimizer, fcn = get_net_ls(paths[0])
    net_lss = [net_ls]
    for path in paths[1:]:
        net_ls, _, _, _, _, _ = get_net_ls(path)
        net_lss.append(net_ls)
    
    print(f"Total networks {len(net_ls)}, activation: {activation}, dataname: {dataname}, epoch_last: {epoch_last}, optimizer: {optimizer}, fcn: {fcn}")

    # phase transition lines
    #bound1, boundaries = load_transition_lines()

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3+0.45,7.142/2 - 0.1))     # without text
    axs = axs.flat

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


    #title_ls = ['Test accuracy', r'$\frac{{{}}}{{8}}$'.format("Test accuracy", "Train accuracy")]
    title_ls = ['Test accuracy', 'Earliest epoch reaching' + '\n' + 'test acc. threshold ']
    label_ls = ['(a)', '(b)', '(c)', '(d)']
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
        plt.savefig(f"{fig1_path}/{fcn}_{dataname}_{optimizer}_{acc_type}_epoch={epoch}_grid_all_median.pdf", bbox_inches='tight')

    print("Figure 1")
    print(f"Trained networks: {len(net_ls)}")
    print(f"(alpha, g) pair: {len(alpha_m_ls)}")
    print(f"Good: {good}")    

def accloss_phase(acc_type, path, acc_threshold=93, display=False):
    '''
    Includes accuracy, loss and stopping epoch for MLP (main text plot)
        - acc_type (str): test or train
        - path: /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid
        - acc_threshold (int): the accuracy threshold to determine speed of training
    '''

    global acc_ls, alpha_m_ls, acc_mesh, cmap_bd
    global net_ls, dataname, epoch_last, fcn, early_ls

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    acc_threshold = float(acc_threshold)
    display = literal_eval(display) if isinstance(display,str) else display

    net_ls, activation, dataname, epoch_last, optimizer, fcn = get_net_ls(path)
    print(path)
    print(f"Total networks {len(net_ls)}, activation: {activation}, dataname: {dataname}, epoch_last: {epoch_last}, optimizer: {optimizer}, fcn: {fcn}")

    # phase transition lines
    #bound1, boundaries = load_transition_lines()

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    fig, axs = plt.subplots(1, 3,sharex = True,sharey=True,figsize=(8.4/2*3+0.45,7.142/2 - 0.1))     # without text
    axs = axs.flat

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


    #title_ls = ['Test accuracy', r'$\frac{{{}}}{{8}}$'.format("Test accuracy", "Train accuracy")]
    title_ls = ['Test accuracy', 'Earliest epoch reaching' + '\n' + 'test acc. threshold ']
    label_ls = ['(a)', '(b)', '(c)', '(d)']
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

        # minor ticks
        #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
        #axs[i].yaxis.set_minor_locator(AutoMinorLocator())

    epoch = epoch_last
    #epoch = 5

    alpha_m_ls, acc_ls, loss_ls, early_ls, acc_mesh, loss_mesh, early_mesh, failed_jobs = get_accloss_phase(path, net_ls, epoch, epoch_last, acc_type, acc_threshold)
    good = np.prod(acc_mesh.shape) - len(failed_jobs)

    # print failed jobs
    print(f"Failed jobs: {len(failed_jobs)}")
    if len(failed_jobs) > 0:
        print(failed_jobs)

    # colourmap bounds
    #perc_l, perc_u = 5, 95
    perc_l, perc_u = 15, 85
    cmap_bd = []
    for ls_idx, ls in enumerate([acc_ls, loss_ls, early_ls]):
        #map_bd.append([np.percentile(ls,10),np.percentile(ls,90)])
        #cmap_bd.append([np.percentile(ls,5),np.percentile(ls,95)])
        if ls_idx < 2:
            cmap_bd.append([np.nanpercentile(ls,perc_l),np.nanpercentile(ls,perc_u)])        
        else:
            cmap_bd.append([0,epoch_last])

    # if want to use imshow convert to grid form
    assert len(alpha_m_ls) == len(acc_ls) and len(alpha_m_ls) == len(early_ls)

    # plot results
    mesh_ls = [acc_mesh, loss_mesh, early_mesh]
    for pidx in range(3):
        #extent=[alpha_lower,alpha_upper,mult_lower,mult_upper]
        plot = axs[pidx].imshow(mesh_ls[pidx],extent=extent, 
                                vmin=cmap_bd[pidx][0], vmax=cmap_bd[pidx][1], cmap=cmaps[pidx], 
                                interpolation=interp, aspect='auto',
                                origin='upper')

        # axs[pidx].xticks(np.arange(0, mesh_ls[pidx].shape[1]+1, 1))
        # axs[pidx].yticks(np.arange(0, mesh_ls[pidx].shape[0]+1, 1))


        cbar = plt.colorbar(plot,ax=axs[pidx])
        cbar.ax.tick_params(labelsize=tick_size)
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
        plt.savefig(f"{fig1_path}/{fcn}_{dataname}_{optimizer}_{acc_type}_epoch={epoch}_grid_all.pdf", bbox_inches='tight')

    print("Figure 1")
    print(f"Trained networks: {len(net_ls)}")
    print(f"(alpha, g) pair: {len(alpha_m_ls)}")
    print(f"Good: {good}")        

def pure_metric(acc_type="test", fcn = "fc10", epoch_ls = [10, 50, 200], display = False):
  
    assert acc_type == "test" or acc_type == "train"   
    display = literal_eval(display) if isinstance(display,str) else display 

    net_type = f"{fcn}_mnist_tanh"
    if fcn == "fc10" or fcn == "fc5":
        path = f"{DROOT}/trained_mlps/fcn_grid/{fcn}_grid"
    else:
        path = f"{DROOT}/trained_mlps/fcn_grid/{fcn}_grid128"
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
    #fig, axs = fig, axs = plt.subplots(nrows, ncols,sharex=True,sharey=True,figsize=(8.4/2*3 + 1,7.142/2 * len(metrics)))     # without text
    fig, axs = plt.subplots(nrows, ncols,sharex=True,sharey=True,figsize=(8.4/2*3,7.142/2 * len(metrics)))     # without text

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

            axis.tick_params(axis='both',labelsize=label_size)

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
    good_ls = []
    failed_jobs_ls = []

    # convert to grid form for imshow
    acc_mesh = np.zeros((len(epoch_ls),mult_N,alpha_N))
    loss_mesh = np.zeros((len(epoch_ls),mult_N,alpha_N))
    for eidx, epoch in enumerate(epoch_ls):
        alpha_m_ls, acc_ls, loss_ls, _, acc_mesh_epoch, loss_mesh_epoch, _, failed_jobs = get_accloss_phase(path, net_ls, epoch, epoch_last, acc_type, 10)
        good_ls.append(np.prod(acc_mesh.shape) - len(failed_jobs))
        acc_mesh[eidx,:,:] = acc_mesh_epoch
        loss_mesh[eidx,:,:] = loss_mesh_epoch

    for goodidx, good in enumerate(good_ls):
        if good != 0:
            print(failed_jobs_ls[goodidx])

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

    print("Figure 1")
    print(f"Trained networks: {len(net_ls)}")
    print(f"(alpha, g) pair: {len(alpha_m_ls)}")
    print(f"Good: {good}")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)

    if display:
        fig1_path = "/project/PDLAI/project2_data/figure_ms"
        epochs = [str(epoch) for epoch in epoch_ls]
        epochs = "_".join(epochs)
        fname = f"{fig1_path}/{net_type}_grid_{acc_type}_accloss_epoch={epochs}.pdf"
        print(fname)
        plt.savefig(fname, bbox_inches='tight')
        print("Figure saved!")
    else:
        plt.show()


def epochs_all(alpha100s, g100s, acc_type, path, display=False):
    """
    Plots the all epoch accuracy/loss for specified (\alpha, \sigma_w)
    """

    global net_ls, dataname, epoch_last, fcn, net_paths, net_path, alpha, g, accs_all, losses_all

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    alpha100s = literal_eval(alpha100s); g100s = literal_eval(g100s)
    display = literal_eval(display) if isinstance(display,str) else display

    net_ls, activation, dataname, epoch_last, optimizer, fcn = get_net_ls(path)
    net_type = f"{fcn}_mnist_tanh"
    print(path)
    print(f"Total networks {len(net_ls)}, activation: {activation}, dataname: {dataname}, epoch_last: {epoch_last}, optimizer: {optimizer}, fcn: {fcn}")

    #fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True,figsize=(9.5,7.142))
    #fig, ((ax1,ax2)) = plt.subplots(1, 2,sharex = True,sharey=True,figsize=(9.5,7.142/2 + 0.5))     # window size with text
    nrows, ncols = 1, 2
    fig, axs = plt.subplots(nrows,ncols,sharex = True,sharey=True,figsize=(8.4/2*ncols+0.45,3.471*nrows))     # without text
    axs = axs.flat

    title_ls = ['Test accuracy', 'Test loss' + '\n' + 'test acc. threshold ']

    accs_all, losses_all = [], []
    for aidx, alpha100 in enumerate(alpha100s):
        axis = axs[aidx]
        for g100 in g100s:
            alpha, g = alpha100/100, g100/100

            # '/project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid/fc10_mnist_tanh_id_stable1.5_3.0_epoch650_algosgd_lr=0.001_bs=1024_data_mnist'
            net_paths = get_net_paths(net_ls, alpha100, g100)
            net_path = net_paths[0]
            accs, losses = get_accloss(net_path, acc_type, net_type)
            accs_all.append(accs); losses_all.append(losses)

            axis.plot(accs, label=rf'$\sigma_w$ = {g}')
            #axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)             

            # # ticks
            # axis.tick_params(axis='both',labelsize=tick_size)
            # # major ticks
            # axis.set_xticks(xticks)
            # axis.set_xticklabels(xtick_ls)

            #if i == 2 or i == 3:
            #axis.set_xlabel(r'$\alpha$', fontsize=axis_size)
            #axis.set_title(f"{title_ls[i]}", fontsize=axis_size)

            # # setting ticks
            # axs[i].tick_params(bottom=True, top=False, left=True, right=False)
            # axs[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

            # axs[i].tick_params(axis="x", direction="out")
            # axs[i].tick_params(axis="y", direction="out")

            # minor ticks
            #axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            #axs[i].yaxis.set_minor_locator(AutoMinorLocator())

    axs[0].legend(frameon=False)
    #axs[1].set_yticklabels([])

    plt.tight_layout()
    if display:
        plt.show()
    else:
        fig1_path = "/project/PDLAI/project2_data/figure_ms/trained_dnn_performance"
        if not isdir(fig1_path):  makedirs(fig1_path)
        file_name = f'{fcn}_{dataname}_{optimizer}_{acc_type}_alpha100={alpha100s}_g100={g100s}.pdf'
        plt.savefig(join(fig1_path, file_name), bbox_inches='tight')

        print(f"Figure saved: {join(fig1_path, file_name)}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])

