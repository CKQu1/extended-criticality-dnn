import numpy as np
import math
import os
import pandas as pd
import powerlaw as plaw
import random
import scipy.io as sio
import scipy.stats as sst
import sys
import time
import torch

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from ast import literal_eval
from os.path import join, isfile
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

from path_names import root_data

t0 = time.time()

# ----------------------------

# Stable fit function
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

def get_name(weight_name):    
    return '_'.join(weight_name.split("_")[:-3])

def replace_name(weight_name,other):
    assert isinstance(other,str)
    ls = weight_name.split("_")
    ls[-3] = other
    #ls += other
    return '_'.join(ls)

# fitting and testing goodness of fit
def fit_and_test(data, dist_type):
    
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        params = pconv(*levy_stable._fitstart(data))
        r = levy_stable.rvs(*params, size=len(data))
        logl = np.sum(np.log(levy_stable.pdf(data, *params)))
    elif dist_type == 'normal':
        params = distributions.norm.fit(data)
        r = norm.rvs(*params, len(data))
        logl = np.sum(np.log(norm.pdf(data, *params)))
    elif dist_type == 'tstudent':
        params = sst.t.fit(data)
        r = sst.t.rvs(*params, len(data))
        logl = np.sum(np.log(sst.t.pdf(data, *params)))
    elif dist_type == 'lognorm':
        params = lognorm.fit(data)
        r = lognorm.rvs(*params, size=len(data))
        logl = np.sum(np.log(lognorm.pdf(data, *params)))

    # statistical tests    
    # AD test
    try:
        ad_test = anderson_ksamp([r, data])
        ad_siglevel = ad_test.significance_level
    except:
        ad_siglevel = None
        pass

    # KS test
    try:
        ks_test = ks_2samp(r, data, alternative='two-sided')
        ks_stat = ks_test.statistic
        ks_pvalue = ks_test.pvalue
    except:
        ks_stat, ks_pvalue = None, None
        pass

    if dist_type == 'normal':
        shapiro_test = shapiro(data)
        shapiro_stat = shapiro_test[0]
        shapiro_pvalue = shapiro_test[1]
        return params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue
    else:
        return params, logl, ad_siglevel, ks_stat, ks_pvalue


# powerlaw fit (to absolute value of the weights, i.e. not two-sided)
def pretrained_plfit(weight_path, save_dir, n_weight, pytorch=True):
    global model_path, df_pl
    #global weights_all, thing, weights, plaw_fit, fits, compare_ls

    pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)

    t0 = time.time()
    n_weight = int(n_weight)

# Loading weight matrix ----------------------

    col_names = ['layer','fit_size','alpha','xmin','xmax', "R_ln", "p_ln", "R_exp", "p_exp", "R_trun", "p_trun","stable_alpha", "w_size"]  

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Powerlaw fitting ----------------------

    df_pl = pd.DataFrame(np.zeros((1,len(col_names)))) 
    df_pl = df_pl.astype('object')
    df_pl.columns = col_names

    weights = torch.load(f"{weight_path}/{weight_name}")
    weights = weights.detach().numpy()
    w_size = len(weights)

    # 1. values much smaller than zero are filtered out
    weights = np.abs(weights)            
    weights = weights[weights >0.00001]

    # 2. split into cases for fitting speed
    print("Directory set up, start fitting.")
    if len(weights) <= 2e5:
        print("Low range.")

        #plaw_fit = plaw.Fit(weights, verbose=False)
        plaw_fit = plaw.Fit(weights[weights > np.quantile(weights, 0.99)], verbose=False)
    elif 2e5 < len(weights) <= 4e5:
        print("Medium range.")

        q1 = 0.85
        q2 = 0.95
        xmin_range = ( np.quantile(weights, q1), np.quantile(weights, q2) )
        weights = weights[weights > np.quantile(weights, q1)]

        plaw_fit = plaw.Fit(weights, xmin=xmin_range, xmax=max(weights), verbose=False)
    
    else:
        print("High range.")

        """
        q_ls = np.arange(0.9, 0.999, 0.005)
        xmin_ls = []
        fits = []
        compare_ls = []
        for q_idx in tqdm(range(len(q_ls))):
            xmin_cur = np.quantile(weights, q_ls[q_idx])
            print(xmin_cur)
            xmin_ls.append(xmin_cur)
            fit = plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, xmax=max(weights), verbose=False)
            # lognormal
            R_1, p_1 = fit.distribution_compare('power_law', 'lognormal')
            # exponential
            R_2, p_2 = plaw_fit.distribution_compare('power_law', 'exponential')
            compare_ls.append([R_1, p_1, R_2, p_2])
            fits.append(fit)   
        """    
        q_large = 0.9
        xmin_cur = np.quantile(weights, q_large)
        #plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, xmax=max(weights), verbose=False)
        plaw_fit = plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, verbose=False)
    
    print(f"True size: {w_size}")
    print(f"Fit size: {len(weights)}")

    # dist comparison
    # 1. Lognormal
    R_ln, p_ln = plaw_fit.distribution_compare('power_law', 'lognormal')
    # 2. exponential
    R_exp, p_exp = plaw_fit.distribution_compare('power_law', 'exponential')
    # 3. truncated powerlaw
    R_trun, p_trun = plaw_fit.distribution_compare('power_law', 'truncated_power_law')
    
    # save params
    wmat_idx = int( weight_name.split("_")[-1] )
    if plaw_fit.xmax == None:
        xmax = 0
    else:
        xmax = plaw_fit.xmax
    df_pl.iloc[0,:] = [wmat_idx, len(weights), plaw_fit.alpha, plaw_fit.xmin, xmax, R_ln, p_ln, R_exp, p_exp, R_trun, p_trun, 0, w_size]
    data_name = replace_name(weight_name,'plfit')
    df_pl.to_csv(f'{model_path}/{data_name}.csv', index=False)

    print(df_pl)

    # Plots
    fig = plaw_fit.plot_ccdf(linewidth=3, label='Empirical Data')
    plaw_fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law fit')
    plaw_fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
    plaw_fit.exponential.plot_ccdf(ax=fig, color='b', linestyle='--', label='Exponential')
    plaw_fit.truncated_power_law.plot_ccdf(ax=fig, color='c', linestyle='--', label='Truncated powerlaw')

    plt.legend(loc = 'lower left')
    plot_name = replace_name(weight_name,'plot')
    plt.savefig(f"{model_path}/{plot_name}.pdf", bbox_inches='tight')     
    #plt.clf()
    #plt.show()

    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!")        

# -----------------------------------------------------------------------
    
# fitting to stable, Gaussian, Student-t, lognormal distribution
def pretrained_allfit(weight_path, save_dir, n_weight, pytorch=True):
    global weights, params, df, plot_title, x, params
    global weight_name

    pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    #global weights_all, thing, weights, plaw_fit, fits, compare_ls

    t0 = time.time()
    n_weight = int(n_weight)

# Loading weight matrix ----------------------

    col_names = ['wmat_idx','w_size', 'fit_size',
                 'alpha','beta','delta','sigma', 'logl_stable', 'ad sig level stable', 'ks stat stable', 'ks pvalue stable',                        # stable(stability, skewness, location, scale), 3 - 10
                 'mu', 'sigma_norm', 'logl_norm', 'ad sig level normal','ks stat normal', 'ks pvalue normal', 'shap stat', 'shap pvalue',           # normal(mean, std), 11 - 18
                 'nu', 'sigma_t', 'mu_t', 'logl_t', 'ad sig level tstudent','ks stat tstudent', 'ks pvalue tstudent',                               # tstudent(dof, scale, location), 19 - 25
                 'shape_lognorm', 'loc_lognorm', 'scale_lognorm', 'logl_lognorm', 'ad sig level lognorm','ks stat lognorm', 'ks pvalue lognorm'     # lognormal(loc, scale), 26 - 32                                                                                  
                 ]           

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    #weight_path = join(main_path, "weights_all")

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)
    if not os.path.exists(model_path): os.makedirs(model_path)        
    print(f"{model_path} directory set up, fitting now!")
    print("\n")

    df = pd.DataFrame(np.zeros((1,len(col_names)))) 
    df = df.astype('object')
    df.columns = col_names    

    weights = torch.load(f"{weight_path}/{weight_name}")
    weights = weights.detach().numpy()
    w_size = len(weights)

    # 1. values much smaller than zero are filtered out         
    weights = weights[np.abs(weights) >0.00001]

    print(f"True size: {w_size}")
    print(f"Fit size: {len(weights)}")

    # save params
    df.iloc[0,0:3] = [wmat_idx, w_size, len(weights)]

    # Fitting
    index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
    for dist_type in tqdm(["levy_stable", "normal", "tstudent", "lognorm"]):
        idxs = index_dict[dist_type]
        if dist_type == "normal":
            params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue = fit_and_test(weights, dist_type)
            df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
        else:
            params, logl, ad_siglevel, ks_stat, ks_pvalue = fit_and_test(weights, dist_type) 
            df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue]   

        df.iloc[0,idxs[0]:idxs[1]] = list(params)      
        #print(f"{dist_type} done!")      
        #print('\n')

# Save params ----------------------
    data_name = replace_name(weight_name,'allfit')
    df.to_csv(f'{model_path}/{data_name}.csv', index=False)
    print("df saved!")
    print(df)

# Plots ----------------------    
    fig, axs = plt.subplots(3, 1, sharex = False,sharey=False,figsize=(12.5 + 4.5, 9.5/2 + 0.5))
    # plot 1 (full distribution)
    axs[0].hist(weights, bins=200, density=True)
    wmin, wmax = weights.min(), weights.max()
    bd = max(np.abs(wmin), np.abs(wmax))
    #x = np.linspace(-bd, bd, 1000)
    x = np.linspace(-1, 1, 1000)
    axs[0].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
    axs[0].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
    axs[0].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
    axs[0].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)

    # plot 2 (log-log hist right tail)
    axs[1].hist(weights, bins=1000, histtype="step")

    x = np.linspace((mean_gaussian + wmax)/2, wmax, 500)
    mean_gaussian = df.iloc[0,11]
    axs[1].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
    axs[1].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
    axs[1].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
    axs[1].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
    axs[1].set_xlim((mean_gaussian + wmax)/2, wmax)
    axs[1].set_xscale('log'); axs[1].set_yscale('log')

    # plot 3 (left tail)
    axs[2].hist(weights, bins=1000, histtype="step")

    x = np.linspace(wmin, (wmin + mean_gaussian)/2, 500)
    axs[2].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
    axs[2].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
    axs[2].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
    axs[2].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
    axs[1].set_xlim(wmin, (wmin + mean_gaussian)/2)
    axs[2].set_xscale('log'); axs[2].set_yscale('log')

    print("Starting plot")
    plot_title = str(list(df.iloc[0,0:3])) + '\n'
    plot_title += "Levy" + str(["{:.2e}".format(num) for num in df.iloc[0,3:7]]) + '  '
    plot_title += "Normal" + str(["{:.2e}".format(num) for num in df.iloc[0,11:13]]) + '\n'
    plot_title += "TStudent" + str(["{:.2e}".format(num) for num in df.iloc[0,19:22]]) + '  '
    plot_title += "Lognorm" + str(["{:.2e}".format(num) for num in df.iloc[0,26:29]])
    plt.title(plot_title)
    plot_name = replace_name(weight_name,'plot')
    plt.legend(loc = 'upper right')
    plt.savefig(f"{model_path}/{plot_name}.pdf", bbox_inches='tight', format='pdf')     
    #plt.clf()
    # Time
    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!") 

def submit(*args):

    pytorch = True
    #is_stablefit = False
    #global df, model_name, weight_name, i, wmat_idx, main_path, fit_path, root_path
    
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)    

    if pytorch:
        # --- Pytorch ---
        root_path = join(main_path, "weights_all")
        fit_path = join(main_path, "allfit_all")
        df = pd.read_csv(join(main_path, "weight_info.csv"))
    else:
        # ---TensorFlow ---
        root_path = join(main_path, "weights_all_tf")
        fit_path = join(main_path, "allfit_all_tf")
        df = pd.read_csv(join(main_path, "weight_info_tf.csv"))

    print(fit_path)
    #total_weights = len([weight_ii for weight_ii in os.walk(p)][1:])
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    
    print(df.shape)
    assert total_weights == df.shape[0]

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]

    pbs_array_data = []

    total_weights = 20
    for n_weight in list(range(total_weights)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        #if not (plot_exist or fit_exist):
        #if not fit_exist:
        pbs_array_data.append( (root_path, fit_path, n_weight, pytorch) )
 
    #pbs_array_data = pbs_array_data[:1000] 
    print(len(pbs_array_data))    
    """        
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=main_path,  
             P=project_ls[pidx], 
             mem="4GB")              
    """

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])



