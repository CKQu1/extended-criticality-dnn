import numpy as np
import math
import os
import pandas as pd
import random
import scipy.io as sio
import scipy.stats as sst
import seaborn as sns
import sys
import time

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from ast import literal_eval
from os.path import join, isfile
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions 
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

def list_str_divider(ls, chunks):
    start = 0
    n = len(ls)
    lss = []
    while start + chunks < n:
        lss.append(str( ls[start:start+chunks] ).replace(" ",""))
        start += chunks
    if start <= n - 1:
        lss.append(str( ls[start:] ).replace(" ",""))    
    return lss

# convert torch saved weight matrices into numpy
def wmat_torch_to_np(weight_path, n_weight):
    import torch

    pytorch = False if "_tf" in weight_path else True
    n_weight = int(n_weight)

# Loading weight matrix ----------------------   

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight, "model_name"]
    print(f"{n_weight}: {weight_name}")

    weights = torch.load(f"{weight_path}/{weight_name}")
    weights = weights.detach().numpy()    

    # save weights
    if pytorch:
        np_weight_path = join(main_path, "np_weights_all")
    else:
        np_weight_path = join(main_path, "np_weights_all_tf")
    if not os.path.isdir(np_weight_path): os.makedirs(np_weight_path)
    np.save(join(np_weight_path, weight_name), weights)
    print("Weights saved in numpy!")

def ensemble_wmat_torch_to_np(weight_path, n_weights):
    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights,list), "n_weights is not a list!"
    for n_weight in n_weights:
        wmat_torch_to_np(weight_path, n_weight)

def w_conversion_submit(*args):
    pytorch = False
    
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
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    total_weights_idxs = list(range(total_weights))
    
    print(df.shape)
    assert total_weights == df.shape[0]

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]

    chunks = 15  # number of elements in each list
    nweightss = list_str_divider(total_weights_idxs, chunks)

    pbs_array_data = []    
    for n_weights in nweightss: 
        pbs_array_data.append( (root_path, n_weights) )
 
    #pbs_array_data = pbs_array_data[:1000] 
    print(len(pbs_array_data))    
                
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=main_path,  
             P=project_ls[pidx], 
             mem="4GB")        

# check if number is nan or inf
def is_num_defined(num):
    return not ( np.isnan(num) or np.isposinf(num) or np.isneginf(num) )

def logl_from_params(data, params, dist_type):
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        params = pconv(*levy_stable._fitstart(data))
        logl = np.sum(np.log(levy_stable.pdf(data, *params)))
    elif dist_type == 'normal':
        params = distributions.norm.fit(data)
        logl = np.sum(np.log(norm.pdf(data, *params)))
    elif dist_type == 'tstudent':
        params = sst.t.fit(data)
        logl = np.sum(np.log(sst.t.pdf(data, *params)))
    elif dist_type == 'lognorm':
        params = lognorm.fit(data)
        logl = np.sum(np.log(lognorm.pdf(data, *params)))
    return logl

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

    is_logl_defined = is_num_defined(logl)
    if not is_logl_defined:
        logl = np.nan
    #assert is_logl_defined, f"Log likelihood fitting {dist_type} from fit_and_test() ill-defined, i.e. inf or nan"

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
        return params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue, is_logl_defined
    else:
        return params, logl, ad_siglevel, ks_stat, ks_pvalue, is_logl_defined


# powerlaw fit (to absolute value of the weights, i.e. not two-sided)
def pretrained_plfit(weight_path, save_dir, n_weight):
    import powerlaw as plaw
    global model_path, df_pl

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

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

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
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

        
        #q_ls = np.arange(0.9, 0.999, 0.005)
        #xmin_ls = []
        #fits = []
        #compare_ls = []
        #for q_idx in tqdm(range(len(q_ls))):
        #    xmin_cur = np.quantile(weights, q_ls[q_idx])
        #    print(xmin_cur)
        #    xmin_ls.append(xmin_cur)
        #    fit = plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, xmax=max(weights), verbose=False)
        #    # lognormal
        #    R_1, p_1 = fit.distribution_compare('power_law', 'lognormal')
        #    # exponential
        #    R_2, p_2 = plaw_fit.distribution_compare('power_law', 'exponential')
        #    compare_ls.append([R_1, p_1, R_2, p_2])
        #    fits.append(fit)   
           
        #q_large = 0.9
        #xmin_cur = np.quantile(weights, q_large)
        ##plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, xmax=max(weights), verbose=False)
        #plaw_fit = plaw.Fit(weights[weights > xmin_cur], xmin=xmin_cur, verbose=False)
    
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
    
# -------------------- Single pretrained weight matrix fitting --------------------

def load_single_wmat(weight_path, weight_name, if_torch_weights):
    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
    return weights

def plot_weight_fit(df, weights, model_path, weight_name):
    # Percentiles of the weights 
    print(f"Min weight: {weights.min()}, Max weight: {weights.max()}")
    percs = [5e-6, 50, 50, 99.999995]
    percentiles = [np.percentile(weights, per) for per in percs]
    pl1, pl2, pu1, pu2 = percentiles
    print(f"Percentiles at {percs}: {percentiles}")

    # Plots ----------------------    

    # x-axis bounds for 3 plots
    bd = min(np.abs(pl1), np.abs(pu2))
    x = np.linspace(-bd, bd, 1000)
    xbds = [[-bd,bd], [pu1, pu2], [pl1, pl2]]

    fig, axs = plt.subplots(1, 3, sharex = False,sharey=False,figsize=(12.5 + 4.5, 9.5/3 + 2.5))
    # plot 1 (full distribution); # plot 2 (log-log hist right tail); # plot 3 (left tail)
    for aidx in range(len(axs)):
        axs[aidx].hist(weights, bins=2000, density=True)
        sns.kdeplot(weights, fill=False, color='blue', ax=axs[aidx])

        x = np.linspace(xbds[i][0], xbds[i][1], 1000)
        axs[aidx].plot(x, levy_stable.pdf(x, *df.iloc[0,3:7]), label = 'Stable fit', alpha=1)
        axs[aidx].plot(x, norm.pdf(x, *df.iloc[0,11:13]), label = 'Normal fit', linestyle='dashdot', alpha=0.85)
        axs[aidx].plot(x, sst.t.pdf(x, *df.iloc[0,19:22]), label = "Student-t",  linestyle='dashed', alpha=0.7)
        axs[aidx].plot(x, lognorm.pdf(x, *df.iloc[0,26:29]), label = "Lognormal",  linestyle='dotted', alpha=0.7)
        axs[aidx].set_xlim(xbds[aidx][0], xbds[aidx][1])
        if aidx == 0:
            axs[aidx].legend(loc = 'upper right')
        if aidx == 1 or aidx == 2:
            axs[aidx].set_xscale('symlog'); axs[aidx].set_yscale('log')

    print("Starting plot")
    plot_title = str(list(df.iloc[0,0:3])) + '\n'
    plot_title += "Levy" + str(["{:.2e}".format(num) for num in df.iloc[0,3:7]]) + '  '
    plot_title += "Normal" + str(["{:.2e}".format(num) for num in df.iloc[0,11:13]]) + '\n'
    plot_title += "TStudent" + str(["{:.2e}".format(num) for num in df.iloc[0,19:22]]) + '  '
    plot_title += "Lognorm" + str(["{:.2e}".format(num) for num in df.iloc[0,26:29]])
    plt.suptitle(plot_title)
    plot_name = replace_name(weight_name,'plot')
    plt.savefig(f"{model_path}/{plot_name}.pdf", bbox_inches='tight', format='pdf')       
    
# fitting to stable, Gaussian, Student-t, lognormal distribution
def pretrained_allfit(weight_path, n_weight):
    #global weights, params, df, plot_title, x, params
    #global weight_name

    t0 = time.time()
    n_weight = int(n_weight)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

# Loading weight matrix ----------------------       

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    #weight_path = join(main_path, "weights_all")

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]
    print(f"{n_weight}: {weight_name}")

    # dir for potentially previously fitted params
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"
    model_path1 = join(os.path.dirname(weight_path), allfit_folder1, model_name)
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    model_path2 = join(os.path.dirname(weight_path), allfit_folder2, model_name)   
    if not os.path.exists(model_path1): os.makedirs(model_path1)
    if not os.path.exists(model_path2): os.makedirs(model_path2)

    # check if previously trained
    df_name = replace_name(weight_name,'allfit')
    plot_name = df_name = replace_name(weight_name,'plot')  
    plot_exists = isfile( join(model_path1, plot_name) )
    fit_exists1 = isfile( join(model_path1, df_name) )
    fit_exists2 = isfile( join(model_path2, df_name) )

    # ---------- 1. fit and test ----------

    if not (fit_exists1 or fit_exists2):
        col_names = ['wmat_idx','w_size', 'fit_size',
                    'alpha','beta','delta','sigma', 'logl_stable', 'ad sig level stable', 'ks stat stable', 'ks pvalue stable',                        # stable(stability, skewness, location, scale), 3 - 10
                    'mu', 'sigma_norm', 'logl_norm', 'ad sig level normal','ks stat normal', 'ks pvalue normal', 'shap stat', 'shap pvalue',           # normal(mean, std), 11 - 18
                    'nu', 'sigma_t', 'mu_t', 'logl_t', 'ad sig level tstudent','ks stat tstudent', 'ks pvalue tstudent',                               # tstudent(dof, scale, location), 19 - 25
                    'shape_lognorm', 'loc_lognorm', 'scale_lognorm', 'logl_lognorm', 'ad sig level lognorm','ks stat lognorm', 'ks pvalue lognorm'     # lognormal(loc, scale), 26 - 32                                                                                  
                    ]            

        df = pd.DataFrame(np.zeros((1,len(col_names)))) 
        df = df.astype('object')
        df.columns = col_names    

        weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
        w_size = len(weights)
        # values significantly smaller than zero are filtered out         
        weights = weights[np.abs(weights) >0.00001]
        print(f"True size: {w_size}")
        print(f"Fit size: {len(weights)}")

        # save params
        df.iloc[0,0:3] = [wmat_idx, w_size, len(weights)]

        # Fitting
        all_logl_defined = True
        index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
        for dist_type in tqdm(["levy_stable", "normal", "tstudent", "lognorm"]):
            idxs = index_dict[dist_type]
            if dist_type == "normal":
                params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue, is_logl_defined = fit_and_test(weights, dist_type)
                df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
            else:
                params, logl, ad_siglevel, ks_stat, ks_pvalue, is_logl_defined = fit_and_test(weights, dist_type) 
                df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue]   

            all_logl_defined = all_logl_defined and is_logl_defined
            df.iloc[0,idxs[0]:idxs[1]] = list(params)      

        # save path based on all_logl_defined
        model_path = model_path1 if all_logl_defined else model_path2
        if not os.path.exists(model_path): os.makedirs(model_path)  

        print(f"{model_path} directory set up, fitting now!")
        print("\n")

    # Save params ----------------------
        df.to_csv(f'{model_path}/{data_name}.csv', index=False)
        print("df saved!")
        pd.set_option('display.max_columns', df.shape[1])
        print(df)

    elif fit_exists2:
        df = pd.read_csv(join(model_path2, f"{data_name}.csv"))
        # check entries 7, 13, 22, 29
        logl_idxs = [7, 13, 22, 29]
        dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']
        weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
        # values significantly smaller than zero are filtered out         
        weights = weights[np.abs(weights) >0.00001]

        for idx, log_idx in enumerate(logl_idxs):
            logl = df.iloc[0, logl_idx]
            if is_num_defined(logl):
                logl = logl_from_params(weights, params, dist_type)  

        all_logl_defined = True
        for idx, log_idx in enumerate(logl_idxs):
            all_logl_defined = all_logl_defined and is_num_defined(df.iloc[0, logl_idx])     

        if all_logl_defined:
            df.to_csv(f'{model_path1}/{data_name}.csv', index=False)
            # delete original file from the nan verion
            os.remove(join(model_path2, f"{data_name}.csv"))
            print(f"no nan logls, old df deleted from {model_path2}")
        else:
            df.to_csv(f'{model_path2}/{data_name}.csv', index=False)
            print("nan logls still exist")

    # plot hist and fit
    if not plot_exists:
        if fit_exists1:
            weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
            # values significantly smaller than zero are filtered out         
            weights = weights[np.abs(weights) >0.00001]            
            df = pd.read_csv(join(model_path1, f"{data_name}.csv"))
        plot_weight_fit(df, weights, model_path1, weight_name)
        print("Plot done!")
    else:
        print("Plot already created!")

    # Time
    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!")    

def pre_submit(pytorch: bool):

    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)    

    if pytorch:
        # --- Pytorch ---
        #root_path = join(main_path, "weights_all")
        root_path = join(main_path, "np_weights_all")
        df = pd.read_csv(join(main_path, "weight_info.csv"))
    else:
        # ---TensorFlow ---
        #root_path = join(main_path, "weights_all_tf")
        root_path = join(main_path, "np_weights_all_tf")
        df = pd.read_csv(join(main_path, "weight_info_tf.csv"))

    print(root_path)
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    
    print(df.shape)
    assert total_weights == df.shape[0]   

    return main_path, root_path, df, weights_all, total_weights

def submit(*args):

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder) 

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]
    pbs_array_data = []
    
    #for n_weight in list(range(10, total_weights)):
    for n_weight in list(range(10)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )        
        if not (plot_exist or fit_exist):
        #if not fit_exist:
            pbs_array_data.append( (root_path, n_weight) )
 
    #pbs_array_data = pbs_array_data[:1000] 
    print(len(pbs_array_data))
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=main_path,  
             P=project_ls[pidx], 
             mem="2GB")            

# -------------------- Single pretrained weight matrix fitting --------------------

def batch_pretrained_allfit(weight_path, save_dir, n_weights):
    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights, list), "n_weights is not a list!"

    for n_weight in tqdm(n_weights):
        pretrained_allfit(weight_path, save_dir, n_weight)

    print(f"Batch completed for {weight_path} for n_weights: {n_weights}")

def batch_submit(*args):

    pytorch = False
    chunks = 4

    main_path, root_path, fit_path, df, weights_all, total_weights = pre_submit(pytorch)

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]    
    
    n_weights = []
    #for n_weight in list(range(total_weights)):
    for n_weight in list(range(10,total_weights)):
    #for n_weight in list(range(10)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        #if not (plot_exist or fit_exist):
        if not fit_exist:            
            n_weights.append(n_weight)
     
    n_weightss = list_str_divider(n_weights, chunks)
    pbs_array_data = []
    for n_weights in n_weightss:
        pbs_array_data.append( (root_path, fit_path, n_weights) )

    #pbs_array_data = pbs_array_data[:1000] 
    print(len(pbs_array_data))    
        
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=main_path,  
             P=project_ls[pidx], 
             mem="2GB")            


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])



