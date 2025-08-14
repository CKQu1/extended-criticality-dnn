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
from pretrained_wfit import get_name, replace_name, list_str_divider, fit_and_test, 

t0 = time.time()

# ----------------------------

# Stable fit function
pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)
    
# -------------------- Single pretrained weight matrix fitting --------------------
    
# fitting to stable, Gaussian, Student-t, lognormal distribution
def remainder_nonstablefit(weight_path, save_dir, n_weight):
    #global weights, params, df, plot_title, x, params
    #global weight_name

    t0 = time.time()
    n_weight = int(n_weight)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

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

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
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
    pd.set_option('display.max_columns', df.shape[1])
    print(df)

    

def pre_submit(pytorch: bool):

    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)    

    if pytorch:
        # --- Pytorch ---
        #root_path = join(main_path, "weights_all")
        root_path = join(main_path, "np_weights_all")
        fit_path = join(main_path, "allfit_all")
        df = pd.read_csv(join(main_path, "weight_info.csv"))
    else:
        # ---TensorFlow ---
        #root_path = join(main_path, "weights_all_tf")
        root_path = join(main_path, "np_weights_all_tf")
        fit_path = join(main_path, "allfit_all_tf")
        df = pd.read_csv(join(main_path, "weight_info_tf.csv"))

    print(fit_path)
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    
    print(df.shape)
    assert total_weights == df.shape[0]   

    return main_path, root_path, fit_path, df, weights_all, total_weights

def submit(*args):

    pytorch = False
    main_path, root_path, fit_path, df, weights_all, total_weights = pre_submit(pytorch)

    from qsub import qsub, job_divider
    project_ls = ["phys_DL", "PDLAI", "dnn_maths", "ddl", "dyson", "vortex_dl"]

    pbs_array_data = []
    
    #for n_weight in list(range(total_weights)):
    for n_weight in list(range(10, total_weights)):
    #for n_weight in list(range(10)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist = isfile( join(fit_path, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        #if not (plot_exist or fit_exist):
        if not fit_exist:
            pbs_array_data.append( (root_path, fit_path, n_weight) )
 
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
    pbss = pbss[:-1]  # delete
    #for idx, pidx in enumerate(perm):  
    for idx, pidx in enumerate(perm[:-1]):  # delete
        pbs_array_true = pbss[idx]
        pbs_array_true = pbs_array_true[:2]  # delete
        print(project_ls[pidx])

        qsub(f'python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=join(main_path,"jobs_all","stablefit"),  
             P=project_ls[pidx], 
             mem="2GB")            


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])



