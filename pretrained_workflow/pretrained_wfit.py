import numpy as np
import math
import os
import pandas as pd
import powerlaw as plaw
import random
import scipy.io as sio
import sys
import time
import torch

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from ast import literal_eval
from os.path import join, isfile
from scipy.stats import levy_stable
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions, norm, entropy
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm 

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import axes3d

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

# powerlaw fit
def pretrained_plfit(weight_path, save_dir, n_weight, pytorch=True):

    pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    #global weights_all, thing, weights, plaw_fit, fits, compare_ls

    t0 = time.time()
    n_weight = int(n_weight)

# Loading weight matrix ----------------------

    col_names = ['layer','fit_size','alpha','xmin','xmax', "R_ln", "p_ln", "R_exp", "p_exp", "R_trun", "p_trun","stable_alpha", "w_size"]  

    # path for loading the weights
    main_path = "/project/PDLAI/project2_data/pretrained_workflow"

    # old method
    """
    #main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
    weight_path = f"{main_path}/weights_all"
    weights_all = next(os.walk(f'{weight_path}'))[2]
    weights_all.sort()

    weight_name = weights_all[n_weight]
    print(f"{n_weight}: {weight_name}")
    model_name = get_name(weight_name)

    model_path = f"{main_path}/plfit_all/{model_name}"
    """

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Powerlaw fitting ----------------------

    #df_pl = pd.DataFrame(columns=col_names)
    df_pl = pd.DataFrame(np.zeros((1,len(col_names)))) 
    df_pl = df_pl.astype('object')
    df_pl.columns = col_names

    weights = torch.load(f"{weight_path}/{weight_name}")
    weights = weights.detach().numpy()
    w_size = len(weights)

    # 1. values much smaller than zero are filtered out
    weights = np.abs(weights)            
    weights = weights[weights >0.00001]

    # alpha stable fit
    #params = pconv(*levy_stable._fitstart(weights))
    #try: params
    #except NameError: params = [None]

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

    # numpy save
    #df_pl = df_pl.to_numpy()
    #print(df_pl)
    #np.savetxt(f'{model_path}/{data_name}.csv', df_pl, delimiter=",")
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
    plt.clf()
    #plt.show()

    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!")        

# -----------------------------------------------------------------------

# levy alpha stable fit
def pretrained_stablefit(weight_path, save_dir, n_weight, pytorch=False):

    pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    #global weights_all, thing, weights, plaw_fit, fits, compare_ls

    t0 = time.time()
    n_weight = int(n_weight)

# Loading weight matrix ----------------------

    col_names = ['wmat_idx','w_size', 'fit_size','alpha','beta','delta','sigma', 'mu', 'sigma_norm',
                 'ad sig level stable','ks stat stable', 'ks pvalue stable', 'cst',                               # stable
                 'ad sig level normal','ks stat normal', 'ks pvalue normal', 'shap stat', 'shap pvalue']          # normal 

    main_path = "/project/PDLAI/project2_data/pretrained_workflow"
    #weight_path = join(main_path, "weights_all")

    # old method

    """
    # path for loading the weights
    #main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
    #weight_path = join(main_path, "weights_all")
    weights_all = next(os.walk(f'{weight_path}'))[2]
    weights_all.sort()

    weight_name = weights_all[n_weight]
    i, wmat_idx =  weight_name.split("_")[-2:]
    i, wmat_idx = int(i), int(wmat_idx)
    print(f"{n_weight}: {weight_name}")
    model_name = get_name(weight_name)
    """

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(f"{model_path} directory set up, fitting now!")
    print("\n")

# Stable fitting ----------------------

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
    
    params = pconv(*levy_stable._fitstart(weights))
    print("Stable fit done, testing now!")

    # save params
    df.iloc[0,0:3] = [wmat_idx, w_size, len(weights)]
    df.iloc[0,3:7] = list(params)

    # generate sample from same dist
    r = levy_stable.rvs(*params, size=len(weights))

    # AD test
    try:
        ad_test = anderson_ksamp([r, weights])
        ad_siglevel = ad_test.significance_level
        df.iloc[0,9] = ad_siglevel
    except:
        #df.iloc[i,0] = None
        pass

    # KS test
    try:
        ks_test = ks_2samp(r,weights, alternative='two-sided')
        ks_stat = ks_test.statistic
        ks_pvalue = ks_test.pvalue
        df.iloc[0,10:12] = [ks_stat, ks_pvalue]
    except:
        #df.iloc[i,1:3] = [None, None]
        pass

    # stable test
    try:
        cst = find_condition_number(weights)     
        df.iloc[0,12] = cst
               
    except:
        #df.iloc[i,3] = None
        pass

    # Normal fitting -------
    print("Starting Gaussian fit now!")
    mu, sigma_norm = distributions.norm.fit(weights)
    df.iloc[0,7:9] = [mu, sigma_norm]

    # generate sample from same dist
    r = np.random.normal(mu, sigma_norm, len(weights))

    # AD test
    try:
        ad_test = anderson_ksamp([r, weights])
        ad_siglevel = ad_test.significance_level
        df.iloc[0,13] = ad_siglevel
    except:
        #df.iloc[i,0] = None
        pass

    # KS test
    try:
        ks_test = ks_2samp(r,weights, alternative='two-sided')
        ks_stat = ks_test.statistic
        ks_pvalue = ks_test.pvalue
        df.iloc[0,14:16] = [ks_stat, ks_pvalue]
    except:
        #df.iloc[i,5:7] = [None, None]
        pass
        
    # Wilkinson
    try:
        shapiro_test = shapiro(weights)
        shapiro_stat = shapiro_test[0]
        shapiro_pvalue = shapiro_test[1]
        df.iloc[0,16:18] = [shapiro_stat, shapiro_pvalue]
    except:
        #df.iloc[i,7:9] = [None, None]
        pass
    
    # save params
    data_name = replace_name(weight_name,'stablefit')
    df.to_csv(f'{model_path}/{data_name}.csv', index=False)

    print(df)

    # Plots
    plt.hist(weights, bins=200, density=True)
    plt.title([i, wmat_idx, len(weights), params])
    x = np.linspace(-1, 1, 1000)
    plt.plot(x, levy_stable.pdf(x, *params), label = 'Stable fit')
    plt.plot(x, norm.pdf(x, mu, sigma_norm), label = 'Normal fit')
    plot_name = replace_name(weight_name,'plot')
    plt.savefig(f"{model_path}/{plot_name}.pdf", bbox_inches='tight')     
    plt.legend(loc = 'upper right')
    plt.clf()
    # Time
    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!") 

def submit(*args):

    #global df, model_name, weight_name, i, wmat_idx, main_path, stablefit_path, root_path
    
    #root_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow/weights_all"
    main_path = "/project/PDLAI/project2_data/pretrained_workflow"

    pytorch = False

    if pytorch:
        # --- Pytorch ---
        root_path = join(main_path, "weights_all")
        stablefit_path = join(main_path, "stablefit_all")
        df = pd.read_csv(join(main_path, "weight_info.csv"))
    else:
        # ---TensorFlow ---
        root_path = join(main_path, "weights_all_tf")
        stablefit_path = join(main_path, "stablefit_all_tf")
        df = pd.read_csv(join(main_path, "weight_info_tf.csv"))

    print(stablefit_path)
    #total_weights = len([weight_ii for weight_ii in os.walk(p)][1:])
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    
    print(df.shape)
    assert total_weights == df.shape[0]

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson"]

    pbs_array_data = []

    for n_weight in list(range(total_weights)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(stablefit_path,model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        stablefit_exist = isfile( join(stablefit_path,model_name, f"{replace_name(weight_name,'stablefit')}.csv") )
        if not (plot_exist or stablefit_exist):
            pbs_array_data.append( (root_path, stablefit_path, n_weight, pytorch) )

    #qsub(f'python geometry_preplot.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/geometry_data/metrics_postact/', P='phys_DL')
    #qsub(f'python pretrained_workflow/pretrained_wfit.py {" ".join(args)}', pbs_array_data, path='/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow', P='phys_DL', mem="3GB")

    print(len(pbs_array_data))
    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             #path='/project/phys_DL/project2_data/pretrained_workflow',
             path='/project/PDLAI/project2_data/pretrained_workflow',  
             P=project_ls[pidx], 
             mem="2GB")  
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])



