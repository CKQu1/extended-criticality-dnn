import numpy as np
import math
import os
import pandas as pd
import random
import scipy.io as sio
import scipy.stats as sst
#import seaborn as sns
import sys
import time

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from ast import literal_eval
from os.path import join, isfile
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions 
from tqdm import tqdm

from path_names import root_data
from pretrained_wfit import fit_and_test

t0 = time.time()

# Stable fit function
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

# fit the entire distribution
def fit_dist_full(net_path, epoch, wmat_idx):

    #epochs = epochs if isinstance(epochs,list) else literal_eval(epochs)
    epoch, wmat_idx = int(epoch), int(wmat_idx)

    df_setup = pd.read_csv(join(net_path, "log"))
    model_dims, N = df_setup.loc[0, ["model_dims", "depth"]]
    model_dims = literal_eval(model_dims)
    assert wmat_idx <= len(model_dims) - 1, "wmat_idx exceeding length"
    # these networks are all trained without biases
    total_weights = 0
    for idx in range(len(model_dims) - 1):
        total_weights += model_dims[idx] * model_dims[idx+1]
    
    col_names = ['wmat_idx','w_size', 'fit_size',
                'alpha','beta','delta','sigma', 'logl_stable', 'ad sig level stable', 'ks stat stable', 'ks pvalue stable',                        # stable(stability, skewness, location, scale), 3 - 10
                'mu', 'sigma_norm', 'logl_norm', 'ad sig level normal','ks stat normal', 'ks pvalue normal', 'shap stat', 'shap pvalue',           # normal(mean, std), 11 - 18
                'nu', 'sigma_t', 'mu_t', 'logl_t', 'ad sig level tstudent','ks stat tstudent', 'ks pvalue tstudent',                               # tstudent(dof, scale, location), 19 - 25
                'shape_lognorm', 'loc_lognorm', 'scale_lognorm', 'logl_lognorm', 'ad sig level lognorm','ks stat lognorm', 'ks pvalue lognorm'     # lognormal(loc, scale), 26 - 32                                                                                  
                ]

    #df = pd.DataFrame(np.zeros((len(model_dims)-1,len(col_names))))
    df = pd.DataFrame(np.zeros((1,len(col_names)))) 
    df = df.astype('object')    
    row = 0
    with_logl = True
    #index_dict = {'levy_stable': [4, 8, 12], 'normal': [12, 14, 20], 'tstudent': [20, 23, 27], 'lognorm': [27, 30, 34]}
    index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
    dist_types = list(index_dict.keys())

    # load weight matrix
    print(f"Loading and fitting weight matrix {wmat_idx + 1} at epoch {epoch}!")
    weights_all = np.load(join(net_path,f"epoch_{epoch}","weights.npy"))
    assert len(weights_all) == total_weights, "The MLPs must be trained without biases"
    start = 0
    for idx in range(wmat_idx + 1):
        start += model_dims[idx] * model_dims[idx+1]
    wmat_size = model_dims[idx+1] * model_dims[idx+2]
    end = start + wmat_size
    wmat = weights_all[start:end]
    del weights_all

    w_size = len(wmat)
    # values significantly smaller than zero are filtered out         
    #weights = weights[np.abs(weights) >0.00001]

    # save params
    df.iloc[row,0:3] = [wmat_idx, w_size, len(wmat)]
    
    for dist_type in dist_types:
        idxs = index_dict[dist_type]
        if dist_type == "normal":
            params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue, is_logl_defined = fit_and_test(wmat, dist_type, with_logl)
            df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
        else:
            params, logl, ad_siglevel, ks_stat, ks_pvalue, is_logl_defined = fit_and_test(wmat, dist_type, with_logl) 
            df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue]   

        df.iloc[0,idxs[0]:idxs[1]] = list(params)   

    # save df
    save_path = join(net_path, "wmat-fit-all")
    if not os.path.exists(save_path): os.makedirs(save_path)  
    df.to_csv(join(save_path, f"wfit-epoch={epoch}-wmat_idx={wmat_idx}.csv"))


# fit the distribution tails
def fit_dist_tail(net_path, epoch, wmat_idx):
    import matplotlib.pyplot as plt
    import powerlaw as plaw
    #global weights_all, df_setup

    #epochs = epochs if isinstance(epochs,list) else literal_eval(epochs)
    epoch, wmat_idx = int(epoch), int(wmat_idx)

    df_setup = pd.read_csv(join(net_path, "log"))
    model_dims, N = df_setup.loc[0, ["model_dims", "depth"]]
    model_dims = literal_eval(model_dims)
    assert wmat_idx <= len(model_dims) - 2, "wmat_idx exceeding length"
    # these networks are all trained without biases
    total_weights = 0
    for idx in range(len(model_dims) - 1):
        total_weights += model_dims[idx] * model_dims[idx+1]
    
    col_names = ['layer','fit_size','alpha','xmin','xmax', "R_plaw_ln", "p_plaw_ln", "R_plaw_exp", "p_plaw_exp", "R_plaw_trun", "p_plaw_trun", 
                 'R_trun_ln', 'p_trun_ln', 'R_trun_exp', 'p_trun_exp',  
                 "stable_alpha", "w_size",
                 "xmin_lower", "xmin_upper"]  

    #df = pd.DataFrame(np.zeros((len(model_dims)-1,len(col_names)))) 
    df = pd.DataFrame(np.zeros((1,len(col_names))))
    df = df.astype('object')    

    # load weight matrix    
    weights_all = np.load(join(net_path,f"epoch_{epoch}","weights.npy"))
    assert len(weights_all) == total_weights, "The MLPs must be trained without biases"
    start = 0
    for ii in range(wmat_idx):
        start += model_dims[ii] * model_dims[ii+1]
    if wmat_idx == 0:
        ii = 0
    wmat_size = model_dims[ii+1] * model_dims[ii+2]
    end = start + wmat_size
    wmat = weights_all[start:end]
    print(f"Loading and fitting weight matrix {wmat_idx+1} with size {len(wmat)} at epoch {epoch}!")
    del weights_all

    w_size = len(wmat)
    # values significantly smaller than zero are filtered out     
    wmat = np.abs(wmat)    
    #wmat = wmat[wmat >0.00001]

    #xmin = np.percentile(wmat, 50)
    xmin = np.percentile(wmat, 75)
    #plaw_fit = plaw.Fit(wmat, xmin=xmin, verbose=False)
    
    #plaw_fit = plaw.Power_Law(wmat, xmin=xmin, verbose=False)
    scinote = "{:e}".format(wmax)
    e_idx = scinote.find("e")
    integer = float(scinote[:e_idx])
    power = int(scinote[e_idx+1:])    

    xmin, xmax = integer*10**(power-3), integer*10**(power-1)
    plaw_fit = plaw.Power_Law(wmat, xmin=xmin, xmax=xmax, verbose=False)
    
    # 1. Power law vs Lognormal
    R_plaw_ln, p_plaw_ln = plaw_fit.distribution_compare('power_law', 'lognormal')
    # 2. Power law vs exponential
    R_plaw_exp, p_plaw_exp = plaw_fit.distribution_compare('power_law', 'exponential')
    # 3. Power law vs truncated powerlaw
    R_plaw_trun, p_plaw_trun = plaw_fit.distribution_compare('power_law', 'truncated_power_law')      
    # 4. Truncated plaw vs lognormal
    R_trun_ln, p_trun_ln = plaw_fit.distribution_compare('truncated_power_law', 'lognormal')
    # 5. Truncated plaw vs exponential  
    R_trun_exp, p_trun_exp = plaw_fit.distribution_compare('truncated_power_law', 'exponential')    

    if plaw_fit.xmax == None:
        xmax = None
    else:
        xmax = plaw_fit.xmax

    # save params
    df.iloc[0,:-2] = [wmat_idx, len(wmat), plaw_fit.alpha, plaw_fit.xmin, xmax, R_plaw_ln, p_plaw_ln, 
                      R_plaw_exp, p_plaw_exp, R_plaw_trun, p_plaw_trun, 
                      R_trun_ln, p_trun_ln, R_trun_exp, p_trun_exp,
                      0, w_size]
    df.iloc[0,-2:] = [None, None]

    # Plots
    fig, axs = plt.subplots(1, 1, figsize=(17/3, 5.67))
    if isinstance(axs,np.ndarray):
        axis = axs[ii]
    else:
        axis = axs

    plaw_fit.plot_ccdf(ax=axis, linewidth=3, label='Empirical Data')
    plaw_fit.power_law.plot_ccdf(ax=axis, color='r', linestyle='--', label='Power law fit')
    plaw_fit.lognormal.plot_ccdf(ax=axis, color='g', linestyle='--', label='Lognormal fit')
    plaw_fit.exponential.plot_ccdf(ax=axis, color='b', linestyle='--', label='Exponential')
    plaw_fit.truncated_power_law.plot_ccdf(ax=axis, color='c', linestyle='--', label='Truncated powerlaw')

    # semilog
    #axis.set_xscale('linear'); axis.set_yscale('log')

    weights_tails = ["wmat"]
    if len(weights_tails) == 2:
        if weights_tail == "weights_lower":
            axis.set_title("Lower tail")
        else:
            axis.set_title("Upper tail")
    elif len(weights_tails) == 1:
        axis.set_title("Absolute values")  

    # save df
    print(df)
    save_path = join(net_path, "wmat-fit-tail")
    if not os.path.exists(save_path): os.makedirs(save_path)  
    df.to_csv(join(save_path, f"wfit-epoch={epoch}-wmat_idx={wmat_idx}.csv"))

    # save plot
    plt.legend(loc = 'lower left')
    plt.savefig(join(save_path, f"wfit-epoch={epoch}-wmat_idx={wmat_idx}.pdf"), bbox_inches='tight')       


def get_net_ls(nets_dir):

    #net_ls = [join(nets_dir, folder) for folder in os.listdir(nets_dir) if "epochs=" in folder]
    net_ls = [join(nets_dir, folder) for folder in os.listdir(nets_dir) if "epochs=" in folder]
    return net_ls


def submit(*args):
    #global net_ls

    from qsub import qsub, job_divider, project_ls

    nets_dir = join(root_data, "trained_mlps", "fc3_sgd_fig1")
    net_ls = get_net_ls(nets_dir)
    net_ls = [f for f in net_ls if "e32328ba-5e86-11ee-9b71-b083feccf35c" in f or "e32318ca-5e86-11ee-ab8f-b083feccfcbb" in f]
    #epochs = list(range(21))
    epochs = list(range(1))
    wmat_idxs = list(range(2))
    #wmat_idxs = [0,1] 

    # raw submissions    
    pbs_array_data = [(net_path, epoch, wmat_idx)                      
                      for net_path in net_ls
                      for epoch in epochs
                      for wmat_idx in wmat_idxs
                      ]   

    print(pbs_array_data[0])
    print(len(pbs_array_data))             
        
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=nets_dir,
             P=project_ls[pidx],
             #source="/project/phys_DL/extended-criticality-dnn/virt-test-qu/bin/activate",
             source="virt-test-qu/bin/activate",
             ncpus=1,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='2GB')     


def plot_dist_full(net_path, display=True):
    
    epochs = list(range(2))
    #wmat_idxs = list(range(5))
    wmat_idxs = list(range(4))
    for wmat_idx in wmat_idxs:
        wmat_alphas = []
        for epoch in epochs:
            df = pd.read_csv(join(net_path, "wmat-fit-all", f"wfit-epoch={epoch}-wmat_idx={wmat_idx}.csv"))
            wmat_alphas.append(df.loc[0,"alpha"])

        plt.plot(epochs, wmat_alphas, label=r'$\mathbf{{W}}^{{{}}}$'.format(wmat_idx + 1))

    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig(join(net_path, "wmat_epoch_fit.pdf"))


def plot_dist_tail(net_path, display=True):
    
    epochs = list(range(2))
    #wmat_idxs = list(range(5))
    wmat_idxs = list(range(4))
    for wmat_idx in wmat_idxs:
        wmat_alphas = []
        for epoch in epochs:
            df = pd.read_csv(join(net_path, "wmat-fit-tail", f"wfit-epoch={epoch}-wmat_idx={wmat_idx}.csv"))
            wmat_alphas.append(df.loc[0,"alpha"])

        plt.plot(epochs, wmat_alphas, label=r'$\mathbf{{W}}^{{{}}}$'.format(wmat_idx + 1))

    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig(join(net_path, "wmat_epoch_fit.pdf"))        


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])    