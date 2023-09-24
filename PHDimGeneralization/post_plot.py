import matplotlib.pyplot as plt
import  numpy as np
import os
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import makedirs, mkdir
from os.path import join, isdir
from pylab import cm
from tqdm import tqdm

# PLOT SETTINGS
SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_nn_dir_all(main_path):  
    ls = []
    for x in next(os.walk(main_path))[1]:
        ls.append(join(main_path, x))        
    return sorted(ls)

def load_nn_dir(main_path, *args):  
    ls = []
    if args[0] != 'not':
        for x in next(os.walk(main_path))[1]:
            keywords_exist = True
            for keyword in args:
                keywords_exist = keywords_exist and (keyword in x)
            if keywords_exist:
                ls.append(join(main_path, x))
    else:
        for x in next(os.walk(main_path))[1]:
            keywords_exist = False
            for keyword in args[1:]:
                keywords_exist = keywords_exist or (keyword in x)
            if not keywords_exist:
                ls.append(join(main_path, x))        
    return sorted(ls)

def load_g100_nn_dir(main_path, g100):  
    ls = []
    if isinstance(int(g100),int):
        keyword = f"g100={g100}"
    else:
        keywrod = g100
    for x in next(os.walk(main_path))[1]:
        keywords_exist = (keyword in x)
        if keywords_exist:
            ls.append(join(main_path, x))        
    return sorted(ls)    

def phdim_plot(main_path, *args):    
    global phdim_file, phd_data

    # plot settings
    msize = 26    

    g100_exists = False
    for keyword in args:
        g100_exists = g100_exists or ("g100" in keyword)
        if g100_exists:
            if "=" in keyword:
                g100_plot = int(keyword.split("=")[-1])
            elif keyword == "g100":
                g100_plot = keyword

    nn_dirs = load_nn_dir(main_path, *args)    
    fig, axs = plt.subplots(1,3, figsize=(24,8), constrained_layout=True)    

    phdims = []
    test_accs = []
    ges = []
    alphas = []
    gs = []
    total_networks = 0
    for nn_dir in tqdm(nn_dirs):
        dnn_setup = pd.read_csv(join(nn_dir, "dnn_setup.csv"), dtype=object)
        model, depth, dataset = dnn_setup.loc[0, "model"], dnn_setup.loc[0, "depth"], dnn_setup.loc[0, "dataset"]
        phdim_file = join(nn_dir, "dim.txt")
        if os.path.isfile(phdim_file):
            phd_data = np.loadtxt(phdim_file, delimiter=',', usecols=range(1,4))
            alpha100, g100 = dnn_setup.loc[0, ["alpha100", "g100"]]
            alpha, g = int(alpha100)/100, int(g100)/100
            alphas.append(alpha)
            gs.append(g)

            phdim, test_acc, ge = phd_data[2], phd_data[1], phd_data[0] - phd_data[1]
            phdims.append(phdim)
            test_accs.append(test_acc)
            ges.append(ge)

            total_networks += 1
    print(f"Total networks analyzed: {total_networks}!")

    metric_names = ["test_accs", "ges", "alphas"]
    for metric_idx, metric_name in enumerate(metric_names):

        metric = locals()[metric_name]
        axis = axs[metric_idx]
        if metric_idx == 0 or metric_idx == 1:
            divider = make_axes_locatable(axis)
            cmap = cm.get_cmap('PiYG', len(alphas))    # 11 discrete colors
            im = axis.scatter(metric, phdims, c=alphas, s=msize, cmap=cmap)  
            if metric_idx == 0:          
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')      
            else:
                axis.set_yticklabels([])
        else:
            im = axis.scatter(metric, phdims, s=msize)
            axis.set_yticklabels([])

    xlabels = ["Test acc.", "Generalization error", r"$\alpha$"]
    ylabels = ["PH-Dim"] * 2 + ["PH-Dim"]
    for i in range(3):
        axs[i].set_xlabel(xlabels[i])
    axs[0].set_ylabel(ylabels[0])

    suptitle_name = model[0].upper() + model[1:] + f", {dataset}"
    if g100_plot == "g100":
        suptitle_name += rf", $g$ = {list(set(gs))}"        
    else:
        suptitle_name += rf", $g$ = {int(g100_plot)/100}"        
    plt.suptitle(suptitle_name)

    plt.subplots_adjust(hspace=0.2)
    # save file
    fig_dir = join(os.getcwd(), "phdim_analysis", "figures")
    if not isdir(fig_dir): makedirs(fig_dir)
    if model == 'fc':
        plt.savefig(join(fig_dir, f"{model}{depth}_dataset={dataset}_g100={g100_plot}_phdim.pdf"))
    else:
        plt.savefig(join(fig_dir, f"{model}_dataset={dataset}_g100={g100_plot}_phdim.pdf"))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])
