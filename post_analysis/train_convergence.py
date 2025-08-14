#import cmcrameri as cmc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import seaborn as sns
import sys
from ast import literal_eval
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
sys.path.append(os.getcwd())
import path_names
from os import makedirs
from os.path import join, isdir, isfile
from path_names import root_data, id_to_path, model_log
from utils_dnn import load_transition_lines
from transition_plot_functions.mlp_acc_phase import get_net_ls

# colorbar setting
cm_lib = "sns"
cmaps = []
if cm_lib == "plt":
    for cm_type in ['Spectral', 'gist_stern', 'RdGy']:
        cmaps.append(plt.cm.get_cmap(cm_type))
elif cm_lib == "sns":
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

interp = "quadric"
#interp = "none"
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

# from NetPortal/architectures.py
def load_pretrained_weights(init_path, init_epoch):
    #print(f"Loading pretrained weights from {init_path} at epoch {init_epoch}!")
    #with torch.no_grad():
    widx = 0
    if 'fcn_grid' in init_path:     # (by main_last_epoch_2.py)
        #print('fcn_grid')
        w_all = sio.loadmat(f"{init_path}/model_{init_epoch}_sub_loss_w.mat")    
        w_all = w_all['sub_weights'][0]
    else:                   # (by train_supervised.py)
        w_all = np.load(f"{init_path}/epoch_{init_epoch}/weights.npy")       

    return w_all

# for MLPs only
def train_convergence(acc_type, path, display=False):
    global acc_ls, alpha_m_ls, acc_mesh, cmap_bd
    global net_ls, dataname, epoch_last, fcn, early_ls
    global net_path, weights_t1, weights_t0, delta_weight, delta_weights, final_delta_weights

    assert acc_type == "test" or acc_type == "train", "acc_type does not exist!"
    display = literal_eval(display) if isinstance(display,str) else display

    net_ls, activation, dataname, epoch_last, optimizer, fcn = get_net_ls(path)

    # net_idx = 0
    # net_path = net_ls[net_idx]
    # alpha, g = 1.0, 1.0
    # net_path = [path for path in net_ls if f'stable{alpha}_{g}_' in path][0]

    
    # epochs = list(range(50, epoch_last, 50))
    # delta_weights = []
    # for epoch in epochs: 
    #     weights_t1 = load_pretrained_weights(net_path, epoch) 
    #     weights_t0 = load_pretrained_weights(net_path, epoch-50)        
    #     if "fcn_grid" in net_path:
    #         delta_weight = 0
    #         for widx in range(len(weights_t1)):
    #             delta_weight += np.sum((weights_t1[widx] - weights_t0[widx])**2)

    #         delta_weight = np.sqrt(delta_weight)
    #     delta_weights.append(delta_weight)   

    # plt.plot(epochs, delta_weights)
    # if display:
    #     plt.show()
    # else:
    #     fig_path = join(root_data, 'figure_ms', 'fc10_train')
    #     if not isdir(fig_path): makedirs(fig_path)
    #     plt.savefig(join(fig_path, 'fc10_train_convergence.py'))

    epochs = list(range(50, epoch_last, 50))
    final_delta_weights = []
    for net_path in net_ls:
        weights_t1 = load_pretrained_weights(net_path, epoch_last) 
        weights_t0 = load_pretrained_weights(net_path, epoch_last-50)        
        if "fcn_grid" in net_path:
            delta_weight = 0
            for widx in range(len(weights_t1)):
                delta_weight += np.sum((weights_t1[widx] - weights_t0[widx])**2)

            delta_weight = np.sqrt(delta_weight)
        final_delta_weights.append(delta_weight)   

    final_delta_weights = np.array(final_delta_weights)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])    