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

from ast import literal_eval
from os import makedirs
from os.path import join, isfile, isdir
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions 
from tqdm import tqdm

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from qsub import list_str_divider
from weightwatcher.constants import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from constants import root_data

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

def nets_summary(pytorch=True):
    """
    Get a summary of the how many weight matrices/tensors each CNN type has
    """

    global df, counter_dict, saved_wmat

    pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)

    main_path = join(root_data,"pretrained_workflow")

    # method 1
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    if pytorch:
        net_types = ['alexnet', 'convnext', 'densenet', 'efficientnet', 'googlenet', 'mnasnet', 'mobilenet',
                     'regnet', 'resnet', 'resnext', 'shufflenet', 'squeezenet', 'vgg', 'WRN']

        counter_dict = {}
        for row in tqdm(range(df.shape[0])):
            model_name = df.loc[row,'model_name']            
            
            if 'wide_resnet' in model_name:
                net_type = 'WRN'
            elif 'resnet' in model_name:
                net_type = 'resnet'
            else:
                for net_type in net_types:
                    if net_type in model_name:
                        break

            if net_type not in counter_dict.keys():
                counter_dict[net_type] = 1
            else:
                counter_dict[net_type] += 1

            del net_type
    
    # method 2
    # df = pd.read_csv(join(main_path, "net_names_all.csv")) if pytorch else pd.read_csv(join(main_path, "net_names_all_tf.csv"))
    # if pytorch:
    #     net_types = ['alexnet', 'convnext', 'densenet', 'efficientnet', 'googlenet', 'mnasnet', 'mobilenet',
    #                  'regnet', 'resnet', 'resnext', 'shufflenet', 'squeezenet', 'vgg', 'WRN']

    #     counter_dict = {}
    #     #for row in tqdm(range(df.shape[0])):
    #     for row in tqdm(range(df.shape[0])):
    #         model_name = df.loc[row,'model_name'] 
    #         saved_wmat = df.loc[row,'saved_wmat']            
            
    #         if 'wide_resnet' in model_name:
    #             net_type = 'WRN'
    #         elif 'resnet' in model_name:
    #             net_type = 'resnet'
    #         else:
    #             for net_type in net_types:
    #                 if net_type in model_name:
    #                     break

    #         # print(f'row {row}: {model_name} is {net_type} has {saved_wmat}')  # delete
    #         # if net_type == 'alexnet':  # delete
    #         #     break

    #         if net_type not in counter_dict.keys():
    #             counter_dict[net_type] = saved_wmat
    #         else:
    #             counter_dict[net_type] += saved_wmat

    #         del net_type    

    print('\n')
    print(counter_dict)
    print(f'Total wmat = {sum(counter_dict.values())}')
        

# convert torch saved weight matrices into numpy
def wmat_torch_to_np(weight_path, n_weight):
    """
    Converting saved torch weights (via torch.save) into np arrays and saving them.
    """

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

    if not os.path.isfile(join(np_weight_path, f'{weight_name}.npy')):
        np.save(join(np_weight_path, f'{weight_name}.npy'), weights)
        print("Weights saved in numpy!")
    else:
        print("Weights already saved!")

def ensemble_wmat_torch_to_np(weight_path, n_weights):
    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights,list), "n_weights is not a list!"
    for n_weight in n_weights:
        wmat_torch_to_np(weight_path, n_weight)

# function: ensemble_wmat_torch_to_np()
def w_conversion_submit(*args):
    """
    Converting saved torch weights (via torch.save) into np arrays and saving them.
    This was initially needed since a singularity container was not used.
    """
    global total_weights_idxs

    pytorch = True

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    if pytorch:
        np_weight_path = join(main_path, "np_weights_all")
    else:
        np_weight_path = join(main_path, "np_weights_all_tf")            

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
    #total_weights_idxs = list(range(total_weights))
    total_weights_idxs = []
    for n_weight in range(total_weights):
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        model_name = df.loc[n_weight, "model_name"]
        #print(f"{n_weight}: {weight_name}")

        if not os.path.isfile(join(np_weight_path, f'{weight_name}.npy')):
            total_weights_idxs.append(n_weight)
    
    print(f'Remaining weights to be converted: {len(total_weights_idxs)}')

    print(df.shape)
    assert total_weights == df.shape[0]    

    from qsub import qsub, job_divider, command_setup, project_ls
    from constants import SPATH, BPATH

    # number of elements in each list
    #chunks = 15  
    chunks = 1
    nweightss = list_str_divider(total_weights_idxs, chunks)

    pbs_array_data = []    
    for n_weights in nweightss: 
        pbs_array_data.append( (root_path, n_weights) )
 
    #pbs_array_data = pbs_array_data[:1000] 
    print(len(pbs_array_data))    

    #quit()

    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)    
                
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}', 
               pbs_array_true, 
               path=join(main_path,'jobs_all/w_conversion_submit'),  
               P=project_ls[pidx], 
               ncpus=ncpus,
               ngpus=ngpus,
               mem="6GB")        

# check if number is nan or inf
def is_num_defined(num):
    return not ( np.isnan(num) or np.isposinf(num) or np.isneginf(num) or (num==None) )

# for alleviating the computaton pressure
def log_of_normal_pdf(x, params):
    return -np.log(params[1] * np.sqrt(2*np.pi)) - 0.5 * ((x - params[0])/params[1])**2 

# manually written log-likelihood function for log-normal distribution
def log_of_lognormal_pdf(x, params):
    return -np.log(params[0] * (x - params[1])/params[2] * np.sqrt(2*np.pi)) - 0.5 * np.log((x - params[1])/params[2])**2/params[0]**2 - np.log(params[2])

# computes logl with specified params for 4 types of distribution
def logl_from_params(data, params, dist_type):
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        logl = np.sum( np.log(levy_stable.pdf(data, *params)) )
    elif dist_type == 'normal':
        #logl = np.sum(np.log(norm.pdf(data, *params)))
        logl = np.sum( log_of_normal_pdf(data, params) )
    elif dist_type == 'tstudent':
        logl = np.sum(np.log(sst.t.pdf(data, *params)))
    elif dist_type == 'lognorm':
        #logl = np.sum(np.log(lognorm.pdf(data, *params)))
        logl = np.sum( log_of_lognormal_pdf(data, params) )
        
    return logl

# performs 1-sample KS test with specified params for 4 types of distribution
def kstest_from_params(data, params, dist_type):
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        ks_stat, ks_pvalue = sst.kstest(rvs=data, cdf='levy_stable', args=params)
    elif dist_type == 'normal':
        ks_stat, ks_pvalue = sst.kstest(rvs=data, cdf='norm', args=params)
    elif dist_type == 'tstudent':
        ks_stat, ks_pvalue = sst.kstest(rvs=data, cdf='t', args=params)
    elif dist_type == 'lognorm':
        ks_stat, ks_pvalue = sst.kstest(rvs=data, cdf='lognorm', args=params)
        
    return ks_stat, ks_pvalue    

# performs 1-sample AD test similar to above
def adtest_from_params(data, params, dist_type):
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        ad_stat, ad_pvalue = sst.anderson(rvs=data, cdf='levy_stable', args=params)
    elif dist_type == 'normal':
        ad_stat, ad_pvalue = sst.anderson(rvs=data, cdf='norm', args=params)
    elif dist_type == 'tstudent':
        ad_stat, ad_pvalue = sst.anderson(rvs=data, cdf='t', args=params)
    elif dist_type == 'lognorm':
        ad_stat, ad_pvalue = sst.anderson(rvs=data, cdf='lognorm', args=params)
        
    return ad_stat, ad_pvalue       

# fitting and testing goodness of fit
def fit_and_test(data, dist_type, with_logl):
    
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        params = pconv(*levy_stable._fitstart(data))
        r = levy_stable.rvs(*params, size=len(data))
        if with_logl:
            logl = np.sum(np.log(levy_stable.pdf(data, *params)))
    elif dist_type == 'normal':
        params = distributions.norm.fit(data)
        r = norm.rvs(*params, len(data))
        if with_logl:
            #logl = np.sum(np.log(norm.pdf(data, *params)))
            logl = np.sum( log_of_normal_pdf(data, params) )
    elif dist_type == 'tstudent':
        params = sst.t.fit(data)
        r = sst.t.rvs(*params, len(data))
        if with_logl:
            logl = np.sum(np.log(sst.t.pdf(data, *params)))
    elif dist_type == 'lognorm':
        params = lognorm.fit(data)
        r = lognorm.rvs(*params, size=len(data))
        if with_logl:
            logl = np.sum(np.log(lognorm.pdf(data, *params)))
            if not np.isnan(logl):
                logl = np.sum( log_of_lognormal_pdf(data, params) )

    if with_logl:
        is_logl_defined = is_num_defined(logl)
    else:
        #logl, is_logl_defined = np.nan, False
        logl, is_logl_defined = None, False
    #assert is_logl_defined, f"Log likelihood fitting {dist_type} from fit_and_test() ill-defined, i.e. inf or nan"

    # statistical tests    
    # 2-sided AD test
    try:
        ad_test = anderson_ksamp([r, data])
        ad_siglevel = ad_test.significance_level
    except:
        ad_siglevel = None
        pass

    # 2-sided KS test
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


def fast_plfit(data):
    import powerlaw as plaw

    """
    # split into cases for fitting speed
    if len(data) <= 2e5:
        print("Low range.")

        #plaw_fit = plaw.Fit(data, verbose=False)
        plaw_fit = plaw.Fit(data[data > np.quantile(data, 0.99)], verbose=False)
    elif 2e5 < len(data) <= 4e5:
        print("Medium range.")

        q1 = 0.85
        q2 = 0.95
        xmin_range = ( np.quantile(data, q1), np.quantile(data, q2) )
        data = data[data > np.quantile(data, q1)]

        plaw_fit = plaw.Fit(data, xmin=xmin_range, xmax=max(data), verbose=False)
    
    else:
        print("High range.")

        
        #q_ls = np.arange(0.9, 0.999, 0.005)
        #xmin_ls = []
        #fits = []
        #compare_ls = []
        #for q_idx in tqdm(range(len(q_ls))):
        #    xmin_cur = np.quantile(data, q_ls[q_idx])
        #    print(xmin_cur)
        #    xmin_ls.append(xmin_cur)
        #    fit = plaw.Fit(data[data > xmin_cur], xmin=xmin_cur, xmax=max(data), verbose=False)
        #    # lognormal
        #    R_1, p_1 = fit.distribution_compare('power_law', 'lognormal')
        #    # exponential
        #    R_2, p_2 = plaw_fit.distribution_compare('power_law', 'exponential')
        #    compare_ls.append([R_1, p_1, R_2, p_2])
        #    fits.append(fit)   
           
        #q_large = 0.9
        #xmin_cur = np.quantile(data, q_large)
        ##plaw.Fit(data[data > xmin_cur], xmin=xmin_cur, xmax=max(data), verbose=False)
        #plaw_fit = plaw.Fit(data[data > xmin_cur], xmin=xmin_cur, verbose=False)

        print(f"True size: {w_size}")
        print(f"Fit size: {len(weights)}")   
    """

    # hack by extending up to 3 orders of magnitudes below the max for the upper tail (and vice versa for the lower tail)
    wmax = data.max()
    #scinote = "{:.4e}".format(wmax)  # scientific notation up to 4 decimal points
    integer, power = get_int_power(wmax)

    # 1. fixed xmin and xmax
    """
    xmin_range = ( integer*10**(power-3), integer*10**(power-2) )
    data_inbetween = data[xmin_range[0] < data]
    data_inbetween = data_inbetween[data_inbetween < xmin_range[1]]
    data_inbetween = len(data_inbetween)
    print(f"xmin_range: {xmin_range}, data in between {data_inbetween}")
    #data = data[data > np.quantile(data, integer*10**(power-3))]
    plaw_fit = plaw.Fit(data, xmin=xmin_range, verbose=False)    
    """

    # 2. fixed xmin and xmax
    
    xmin_range = ( integer*10**(power-6), None )
    #xmax = integer*10**(power-5)
    #xmax = integer*10**(power-1)
    data_inbetween = data[xmin_range[0] < data]
    #data_inbetween = data_inbetween[data_inbetween < xmax]
    data_inbetween = len(data_inbetween)
    print(f"xmin_range: {xmin_range}, data in between {data_inbetween}")
    #data = data[data > np.quantile(data, integer*10**(power-3))]
    #plaw_fit = plaw.Fit(data, xmin=xmin_range[0], xmax=xmax, verbose=False)
    plaw_fit = plaw.Fit(data, xmin=xmin_range[0], verbose=False)
    

    # 3. fixed quantile
    """
    #q1 = 0.50
    q1 = 0.75
    #q2 = 0.95
    xmin_range = ( np.quantile(data, q1), None )    
    #xmax = np.quantile(data, 0.9)
    plaw_fit = plaw.Fit(data, xmin=xmin_range[0], verbose=False)
    #plaw_fit = plaw.Fit(data, xmin=xmin_range[0], xmax=xmax, verbose=False)
    """

    return plaw_fit, xmin_range


def get_int_power(num):
    scinote = "{:e}".format(num)
    e_idx = scinote.find("e")
    integer = float(scinote[:e_idx])
    power = int(scinote[e_idx+1:])

    return integer, power

def convert_sigfig(num, sigfigs=None):
    integer, power = get_int_power(num)
    if sigfigs == None:
        return integer * 10**power
    else:
        assert isinstance(sigfigs, int)
        return round(integer, sigfigs-1) * 10**power


# two-sided powerlaw fit (based on weightwatcher: https://github.com/CalculatedContent/WeightWatcher/blob/master/weightwatcher/WW_powerlaw.py)
def pretrained_moments(weight_path, save_dir, n_weight, replace=True):
    """
    Computes the mean, std, skewness and kurtosis before and after entry-removal
    """
    import powerlaw
    import sys
    lib_path = os.getcwd()
    sys.path.append(f'{lib_path}')   

    global model_path, df_mmt, df_path, col_names, weights

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    replace = replace if isinstance(replace,bool) else literal_eval(replace)


# Loading weight matrix ----------------------

    col_names = ['layer_idx', 'wmat_idx',
                 'size_1', 'mean_1', 'std_1', 'skewness_1', 'kurtosis_1',   # before removal
                 'wmin_lower_1', 'wmax_lower_1', 'wmin_upper_1', 'wmax_upper_1',
                 'jb_stat_1', 'jb_pval_1',
                 'size_2', 'mean_2', 'std_2', 'skewness_2', 'kurtosis_2',   # after removal
                 'wmin_lower_2', 'wmax_lower_2', 'wmin_upper_2', 'wmax_upper_2',
                 'jb_stat_2', 'jb_pval_2'
                 ]

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Setting up dir ----------------------

    data_name = replace_name(weight_name,'mmt')
    df_path = f'{model_path}/{data_name}.csv'

# Powerlaw fitting ----------------------

    if not os.path.isfile(df_path) or replace:
        if if_torch_weights:
            import torch
            weights = torch.load(f"{weight_path}/{weight_name}")
            weights = weights.detach().numpy()
        else:
            weights = np.load(f"{weight_path}/{weight_name}.npy")
        
        print(f"Directory set up, start computing moments.")
        df_mmt = pd.DataFrame(np.zeros((1,len(col_names)))) 
        df_mmt = df_mmt.astype('object')
        df_mmt.columns = col_names    


        df_mmt.iloc[0,:2] = [layer_idx, wmat_idx]

        # before removal
        wmin_lower = weights[weights < 0].min()
        wmax_lower = weights[weights < 0].max()
        wmin_upper = weights[weights > 0].min()
        wmax_upper = weights[weights > 0].max()
        res = sst.jarque_bera(weights)
        res_stat, res_pval = res.statistic, res.pvalue
        df_mmt.iloc[0,2:13] = [len(weights), weights.mean(), weights.std(), sst.skew(weights), sst.kurtosis(weights),
                               wmin_lower, wmax_lower, wmin_upper, wmax_upper,
                               res_stat, res_pval]
        # after removal
        weights = weights[np.abs(weights) >0.00001]
        if len(weights) > 0:
            try:
                wmin_lower = weights[weights < 0].min()
            except:
                wmin_lower = np.nan
            try:
                wmax_lower = weights[weights < 0].max()
            except:
                wmax_lower = np.nan    
            try:
                wmin_upper = weights[weights > 0].min()
            except:
                wmin_upper = np.nan                 
            try:
                wmax_upper = weights[weights > 0].max()     
            except:
                wmax_upper = np.nan                   
            res = sst.jarque_bera(weights)  
            res_stat, res_pval = res.statistic, res.pvalue
        else:
            wmin_lower, wmax_lower, wmin_upper, wmax_upper = [np.nan] * 4
            res_stat, res_pval = [np.nan] * 2

        df_mmt.iloc[0,13:] = [len(weights), weights.mean(), weights.std(), sst.skew(weights), sst.kurtosis(weights),
                              wmin_lower, wmax_lower, wmin_upper, wmax_upper,
                              res_stat, res_pval]

        #quit()  # delete

        # save data                
        df_mmt.to_csv(df_path, index=False)
        pd.set_option('display.max_columns', None)
        print(df_mmt)

        t_last = time.time()
        print(f'Saved under: {df_path}')
        print(f"{weight_name} done in {t_last - t0} s!")    

    else:
        print(f"{weight_name} has moments computed!")


# two-sided powerlaw fit (based on weightwatcher: https://github.com/CalculatedContent/WeightWatcher/blob/master/weightwatcher/WW_powerlaw.py)
"""
python -i pretrained_workflow/pretrained_wfit.py pretrained_ww_plfit\
 /project/PDLAI/project2_data/pretrained_workflow/weights_all\
 /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_v2/ 6
"""
def pretrained_ww_plfit(weight_path, save_dir, n_weight, replace=True,
                        plot=True):
                        #**kwargs):  # ,remove_weights=False
    """
    Fit tails of each weight matrix to powerlaw (adopted from WeightWatcher):
        - weigh_path (str): path of stored weights (/project/PDLAI/project2_data/pretrained_workflow/weights_all)
        - save_dir (str): path of saved fitted parameters 
            1. /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all
            2. /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_v2
        - n_model (int): the index of the model from get_pretrained_names()
        - replace (bool): refit even if already fitted previously
    """
    import powerlaw
    import sys
    lib_path = os.getcwd()
    sys.path.append(f'{lib_path}')
    from weightwatcher.WW_powerlaw import fit_powerlaw    

    global model_path, df_pl, plfit, df_path, col_names, col_names_all, fit_types
    global alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, Rs_all, ps_all
    global bestfit, unfitted_dists, weights
    global dist_compare, fit_compare, fit_type, ps_all, FIT

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    #remove_weights = remove_weights if isinstance(remove_weights,bool) else literal_eval(remove_weights)
    replace = replace if isinstance(replace,bool) else literal_eval(replace)
    plot = plot if isinstance(plot,bool) else literal_eval(plot)
    #fit_type = kwargs.get('fit_type', 'power_law')
    #fit_type = POWER_LAW

    fit_types = [POWER_LAW]
    #fit_types = [POWER_LAW, TRUNCATED_POWER_LAW]

# Loading weight matrix ----------------------

    #col_names = ['layer','fit_size','alpha','xmin','xmax', "R_plaw_trun", "p_plaw_trun", "R_plaw_ln", "p_plaw_ln", "R_plaw_exp", "p_plaw_exp",  
    #             "stable_alpha", "w_size"]  

    col_names_all = ['layer_idx', 'wmat_idx', 'total_entries', 'total_is', 'skewness', 'kurtosis',    # 0 - 5
                     'fit_entries', 'alpha', 'Lambda', 'xmin', 'xmax', 'D', 'sigma', 
                     'num_pl_spikes', 'num_fingers', 'raw_alpha', 'status', 'warning',
                     'bestfit', 'Rs_all', 'ps_all',
                     'alpha1_alt', 'Lambda1_alt',  # 6 - 22
                     'fit_entries_tpl', 'alpha_tpl', 'Lambda_tpl', 'xmin_tpl', 'xmax_tpl', 'D_tpl', 'sigma_tpl', 
                     'num_pl_spikes_tpl', 'num_fingers_tpl', 'raw_alpha_tpl', 'status_tpl', 'warning_tpl',
                     'bestfit_tpl', 'Rs_all_tpl', 'ps_all_tpl',
                     'alpha2_alt', 'Lambda2_alt'  # 23 - 39
                     ]
    colidxss = [[0, 6], [6, 23], [23, 40]]

    col_names = col_names_all[:colidxss[1][1]]
    if TRUNCATED_POWER_LAW in fit_types:
        col_names += col_names_all[colidxss[2][0]:]
    
    all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL]    

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name} \n")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Setting up dir ----------------------

    data_name = replace_name(weight_name,'plfit')
    df_path = f'{model_path}/{data_name}.csv'

# Powerlaw fitting ----------------------

    if not os.path.isfile(df_path) or replace:
        if if_torch_weights:
            import torch
            weights = torch.load(f"{weight_path}/{weight_name}")
            weights = weights.detach().numpy()
        else:
            weights = np.load(f"{weight_path}/{weight_name}.npy")
        w_size = len(weights)

        # values significantly smaller than zero are filtered out 
        #weights = np.abs(weights)
        """
        if remove_weights:
            weights = weights[np.abs(weights) >0.00001]    

        # 1. split into upper and lower tail
            #weights_upper = weights[weights > weights.mean()]
            weights_upper = np.abs(weights[weights > weights.mean()])
            weights_lower = np.abs(weights[weights < weights.mean()])

        else:
            weights_upper = np.abs(weights[weights >= 0])
            weights_lower = np.abs(weights[weights <= 0])
        """

        upper = np.abs(weights[weights>=0])
        lower = np.abs(weights[weights<=0])    

        # added for speed up and PL fit does not deal well with small values
        upper = upper[upper > 0.00001]
        lower = lower[lower > 0.00001]        

        # for entries exceeding certain threshold, start from the 50 percentile
        pctage = 75
        size_threshold = 5e6
        if len(upper) >= size_threshold:
            upper = upper[upper > np.percentile(upper, pctage)]
            upper_triggered = True
        else:
            upper_triggered = False
        if len(lower) >= size_threshold:
            lower = lower[lower > np.percentile(lower, pctage)]  
            lower_triggered = True     
        else:
            lower_triggered = False

        skewness = sst.skew(weights)
        kurtosis = sst.kurtosis(weights)
        del weights    

        #total_is = 1000  # divided intervals for xmin    
        figdir = join(root_data, 'pretrained_workflow', DEF_SAVE_DIR, model_name)       
        if not os.path.isdir(figdir): os.makedirs(figdir)

        # 2. direct fit
        print(f"Directory set up, start fitting weight matrix of sizes {len(lower)} (lower) and {len(upper)} (upper).")
        weights_tails = ["lower", "upper"]
        #weights_tails = ["weights"]
        df_pl = pd.DataFrame(np.zeros((len(weights_tails),len(col_names)))) 
        df_pl = df_pl.astype('object')
        df_pl.columns = col_names    

        #fig, axs = plt.subplots(2, len(weights_tails), figsize=(17/3*2, 5.67))
        fig, axs = plt.subplots(1, len(weights_tails), figsize=(17/3*2, 5.67))
        for ii, tail_name in enumerate(weights_tails):

            # full iteration
            #total_is = len(locals()[tail_name]) - 1

            # more efficient iteration
            if len(locals()[tail_name]) < 1.2e6:
                ratio = 1
            elif len(locals()[tail_name]) >= 1.2e6 and len(locals()[tail_name]) < 2.4e6:
                ratio = 2
            elif len(locals()[tail_name]) >= 2.4e6 and len(locals()[tail_name]) < 5e6:
                ratio = 5
            elif len(locals()[tail_name]) >= 5e6 and len(locals()[tail_name]) < 1e7:
                ratio = 10
            elif len(locals()[tail_name]) >= 1e7 and len(locals()[tail_name]) < 5e8:                
                ratio = 500
            else:
                ratio = 1000
            # elif len(locals()[tail_name]) >= 1e7 and len(locals()[tail_name]) < 1e8:
            #     ratio = 50
            # else:
            #     ratio = 100

            total_is = int((len(locals()[tail_name]) - 1) / ratio)
            print(f'Tail {tail_name}')
            print(f'total_is = {total_is}, ratio = {ratio}')
            
            # 1. pl_fit direct
            #plfit = pl_fit(locals()[tail_name])
            #pl_fits.append( plfit )          

            # ---------- 1. Fit PL first ----------
            # ---------- 2. Fit TPL first ----------            

            for fit_ii, fit_type in enumerate(fit_types):
                print(f'---------- Fitting {fit_type} first ----------')

                if fit_type == POWER_LAW:
                    dist_compare = TRUNCATED_POWER_LAW
                elif fit_type == TRUNCATED_POWER_LAW:
                    dist_compare = POWER_LAW

                # remove the distributed that is already fitted
                unfitted_dists = [dist for dist in all_dists if dist != dist_compare]                  

                # 2. save plot
                if fit_type == POWER_LAW:
                    plot_id = f"{model_name}_{layer_idx}_{wmat_idx}_{tail_name}"
                    plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot_id=plot_id, savedir=figdir,
                                        plot=plot)
                else:
                    plot_id = f"{fit_type}_{model_name}_{layer_idx}_{wmat_idx}_{tail_name}"
                    plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot_id=plot_id,
                                        fit_type=fit_type, savedir=figdir,
                                        plot=plot)  
                # 3. don't save plot
                #plfit = fit_powerlaw(locals()[tail_name], total_is=total_is, plot=False)            

                # output of fit_powerlaw()
                #alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, best_fit_1, Rs_all, ps_all = plfit            
                alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, FIT = plfit
                if warning == '':
                    if locals()[tail_name + '_triggered'] == True:
                        warning = f'pctage={pctage}' + f'ratio={ratio}'
                    else:
                        warning = f'ratio={ratio}'
                else:
                    if locals()[tail_name + '_triggered'] == True:
                        warning = str(warning) + f'pctage={pctage}' + f'_ratio={ratio}'
                    else:
                        warning = str(warning) + f'_ratio={ratio}'

                if status == 'success':                                    
                    Rs = [0.0]
                    Rs_all = []
                    ps_all = []
                    #dists = [POWER_LAW]
                    #dists = [TRUNCATED_POWER_LAW]
                    dists = [dist_compare]
                    #for dist in all_dists[1:]:
                    for dist in unfitted_dists:
                        #R, p = pl_compare(fit, dist)  # the dist is always being compare to TRUNCATED_POWER_LAW, check distribution_compare()
                        R, p = FIT.distribution_compare(dist, dist_compare, normalized_ratio=True)
                        Rs_all.append(R)
                        ps_all.append(p)
                        #if R > 0.1 and p > 0.05:
                        if p > 0.05:
                            dists.append(dist)
                            Rs.append(R)
                            #logger.debug("compare dist={} R={:0.3f} p={:0.3f}".format(dist, R, p))
                    bestfit = dists[np.argmax(Rs)]  
                    
                    fit_compare = powerlaw.Fit(locals()[tail_name], xmin=xmin, xmax=xmax, 
                                               verbose=False, distribution=dist_compare, xmin_distribution=dist_compare,
                                               fit_method='KS')                    

                    if dist_compare == TRUNCATED_POWER_LAW:
                        alpha_alt, Lambda_alt = [fit_compare.truncated_power_law.alpha, fit_compare.truncated_power_law.Lambda]
                    elif dist_compare == POWER_LAW:
                        alpha_alt, Lambda_alt = fit_compare.power_law.alpha, -1

                    if fit_ii == 0:
                        df_pl.iloc[ii,colidxss[0][0]:colidxss[0][1]] = [layer_idx, wmat_idx, len(locals()[tail_name]), 
                                                                        total_is, skewness, kurtosis]

                    fit_entries = sum( (xmin<=locals()[tail_name]) & (locals()[tail_name]<=xmax) )          

                    print(f"{plot_id} {tail_name} tail")
                    print(f'Fit entries {fit_entries}, total entries {len(locals()[tail_name])}')
                    print(f'best fit: {bestfit}') 
                    print(f'Distributions for comparison: {unfitted_dists}')
                    print(f'Ratios: {Rs_all}')
                    print(f'Pvals: {ps_all} \n')

                    df_pl.iloc[ii,colidxss[fit_ii+1][0]:colidxss[fit_ii+1][1]] = [fit_entries, alpha, Lambda, xmin, xmax, D, sigma,  
                                                                                  num_pl_spikes, num_fingers, raw_alpha, status, warning,
                                                                                  bestfit, Rs_all, ps_all,
                                                                                  alpha_alt, Lambda_alt
                                                                                  ]            

                else:
                    print(f"{plot_id} {tail_name} tail")
                    print(f"Status: {status} \n")
                    return False    

            print('\n')

        #quit()

        # save data                
        df_pl.to_csv(df_path, index=False)
        pd.set_option('display.max_columns', None)
        print(df_pl)

        t_last = time.time()
        print(f'Saved under: {df_path}')
        print(f"{weight_name} done in {t_last - t0} s!")    

    else:
        print(f"{weight_name} has tails already powerlaw fitted!")


# for load PL fits which already exist
"""
python -i pretrained_workflow/pretrained_wfit.py load_fitted_data /project/PDLAI/project2_data/pretrained_workflow/weights_all\
/project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all 0 False False
"""
def load_fitted_data(weight_path, save_dir, n_weight, replace=True,
                     plot=True):
                        #**kwargs):  # ,remove_weights=False
    """
    Fit tails of each weight matrix to powerlaw (adopted from WeightWatcher):
        - weight_path (str): path of stored weights (/project/PDLAI/project2_data/pretrained_workflow/weights_all)
        - save_dir (str): path of saved fitted parameters (/project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all)
        - n_model (int): the index of the model from get_pretrained_names()
        - replace (bool): refit even if already fitted previously
    """
    import sys
    lib_path = os.getcwd()
    sys.path.append(f'{lib_path}')
    from weightwatcher.WW_powerlaw import fit_powerlaw    

    global model_path, df_pl, plfit, df_path
    global alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, Rs_all, ps_all
    global best_fit_1, best_fit_2, unfitted_dists, weights

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    #remove_weights = remove_weights if isinstance(remove_weights,bool) else literal_eval(remove_weights)
    replace = replace if isinstance(replace,bool) else literal_eval(replace)
    plot = plot if isinstance(plot,bool) else literal_eval(plot)
    #fit_type = kwargs.get('fit_type', 'power_law')
    fit_type = 'power_law'

# Loading weight matrix ----------------------

    #col_names = ['layer','fit_size','alpha','xmin','xmax', "R_plaw_trun", "p_plaw_trun", "R_plaw_ln", "p_plaw_ln", "R_plaw_exp", "p_plaw_exp",  
    #             "stable_alpha", "w_size"]  

    col_names = ['layer_idx', 'wmat_idx', 'total_entries', 'fit_entries', 
                 'alpha', 'Lambda', 'xmin', 'xmax', 'D', 'sigma', 'num_pl_spikes', 'num_fingers', 'raw_alpha', 'status', 'warning',
                 'best_fit_1', 'best_fit_2', 'Rs_all', 'ps_all']
    all_dists = [TRUNCATED_POWER_LAW, POWER_LAW, LOG_NORMAL, EXPONENTIAL]
    # remove the distributed that is already fitted
    #unfitted_dists = [fit_type] + [dist for dist in all_dists if dist != fit_type]    

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    model_path = join(save_dir, model_name)

    if not os.path.exists(model_path): os.makedirs(model_path)
    print(f"{model_path} directory set up!")
    print("\n")

# Setting up dir ----------------------

    data_name = replace_name(weight_name,'plfit')
    df_path = f'{model_path}/{data_name}.csv'

# Powerlaw fitting ----------------------

    if os.path.isfile(df_path) or replace:
        df_pl = pd.read_csv(df_path)

        if if_torch_weights:
            import torch
            weights = torch.load(f"{weight_path}/{weight_name}")
            weights = weights.detach().numpy()
        else:
            weights = np.load(f"{weight_path}/{weight_name}.npy")
        w_size = len(weights)

        # values significantly smaller than zero are filtered out 
        #weights = np.abs(weights)
        """
        if remove_weights:
            weights = weights[weights >0.00001]    

        # 1. split into upper and lower tail
            #weights_upper = weights[weights > weights.mean()]
            weights_upper = np.abs(weights[weights > weights.mean()])
            weights_lower = np.abs(weights[weights < weights.mean()])

        else:
            weights_upper = np.abs(weights[weights >= 0])
            weights_lower = np.abs(weights[weights <= 0])
        """

        upper = np.abs(weights[weights>=0])
        lower = np.abs(weights[weights<=0])    

        #del weights    

        total_is = 1000  # divided intervals for xmin    
        figdir = join(DEF_SAVE_DIR, model_name)        




# powerlaw fit (both tails are fitted individually)
"""
python -i pretrained_workflow/pretrained_wfit.py pretrained_ww_plfit /project/PDLAI/project2_data/pretrained_workflow/weights_all /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all 2751
"""
def pretrained_plfit(weight_path, save_dir, n_weight, remove_weights=False):    
    global model_path, df_pl, xmin_range, plaw_fit, weights, xtick_ls, xmin_range, axs

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    remove_weights = remove_weights if isinstance(remove_weights,bool) else literal_eval(remove_weights)

# Loading weight matrix ----------------------

    col_names = ['layer','fit_size','alpha','xmin','xmax', "R_plaw_ln", "p_plaw_ln", "R_plaw_exp", "p_plaw_exp", "R_plaw_trun", "p_plaw_trun", 
                 'R_trun_ln', 'p_trun_ln', 'R_trun_exp', 'p_trun_exp',  
                 "stable_alpha", "w_size",
                 "xmin_lower", "xmin_upper"]  

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

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
    w_size = len(weights)

    # values significantly smaller than zero are filtered out 
    #weights = np.abs(weights)
    if remove_weights:
        weights = weights[weights >0.00001]    

    # 1. split into upper and lower tail
    #weights_upper = weights[weights > weights.mean()]
    weights_upper = np.abs(weights[weights > weights.mean()])
    weights_lower = np.abs(weights[weights < weights.mean()])
    #del weights    

    # 2. direct fit
    print("Directory set up, start fitting.")
    weights_tails = ["weights_lower", "weights_upper"]
    #weights_tails = ["weights"]
    df_pl = pd.DataFrame(np.zeros((len(weights_tails),len(col_names)))) 
    df_pl = df_pl.astype('object')
    df_pl.columns = col_names    

    #fig, axs = plt.subplots(2, len(weights_tails), figsize=(17/3*2, 5.67))
    fig, axs = plt.subplots(1, len(weights_tails), figsize=(17/3*2, 5.67))
    for ii, weights_tail in enumerate(weights_tails):

        # brute force
        #plaw_fit = plaw.Fit(locals()[weights_tail], verbose=False)
        # efficient way
        plaw_fit, xmin_range = fast_plfit(locals()[weights_tail])
        
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

        # save params
        #wmat_idx = int( weight_name.split("_")[-1] )
        if plaw_fit.xmax == None:
            xmax = None
        else:
            xmax = plaw_fit.xmax

        df_pl.iloc[ii,:-2] = [wmat_idx, len(locals()[weights_tail]), plaw_fit.alpha, plaw_fit.xmin, xmax, R_plaw_ln, p_plaw_ln, 
                              R_plaw_exp, p_plaw_exp, R_plaw_trun, p_plaw_trun, 
                              R_trun_ln, p_trun_ln, R_trun_exp, p_trun_exp,
                              0, w_size]
        df_pl.iloc[ii,-2:] = [xmin_range[0], xmin_range[1]]

        # Plots
        if isinstance(axs,np.ndarray):
            if axs.ndim == 1:
                axis = axs[ii]
            else:    
                axis = axs[0,ii]
        else:
            axis = axs[ii]

        plaw_fit.plot_ccdf(ax=axis, linewidth=3, label='Empirical Data')
        
        plaw_fit.power_law.plot_ccdf(ax=axis, color='r', linestyle='--', label='Power law fit')
        plaw_fit.lognormal.plot_ccdf(ax=axis, color='g', linestyle='--', label='Lognormal fit')
        plaw_fit.exponential.plot_ccdf(ax=axis, color='b', linestyle='--', label='Exponential')
        plaw_fit.truncated_power_law.plot_ccdf(ax=axis, color='c', linestyle='--', label='Truncated powerlaw')      

        if xmax == None:
            xtick_ls = [convert_sigfig(plaw_fit.xmin,2), convert_sigfig(locals()[weights_tail].max(),2)]
        else:
            xtick_ls = [convert_sigfig(plaw_fit.xmin,2), convert_sigfig(xmax,2)]            

        #axis.set_xticks(xtick_ls)
        #axis.set_xticklabels(xtick_ls)
        #axis.ticklabel_format(style='sci')

        # semilog
        #axis.set_xscale('linear'); axis.set_yscale('log')

        if len(weights_tails) == 2:
            if weights_tail == "weights_lower":
                axis.set_title("Lower tail")
            else:
                axis.set_title("Upper tail")
        elif len(weights_tails) == 1:
            axis.set_title("Absolute values")
          
    # full distribution fit
    """
    lwidth = 2.5
    c_ls_1 = ["red", "blue"]

    weight_info = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = weight_info.loc[n_weight,"weight_file"]
    i, wmat_idx = int(weight_info.loc[n_weight,"idx"]), int(weight_info.loc[n_weight,"wmat_idx"])
    model_name = weight_info.loc[n_weight,"model_name"]

    dirname = "allfit_all" if pytorch else "allfit_all_tf"
    weight_file = weight_info.loc[n_weight, 'weight_file']
    weight_foldername = replace_name(weight_file, "allfit") + ".csv"
    full_fit_path = join(main_path, dirname, model_name, weight_foldername)
    if not os.path.isfile(full_fit_path):
        dirname = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
        full_fit_path = join(main_path, dirname, model_name, weight_foldername)
    df_full = pd.read_csv( full_fit_path )

    params_stable = df_full.iloc[0,3:7]
    params_normal = df_full.iloc[0,11:13]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(17/3*2, 5.67*2))

    x = np.linspace(weights.min(), weights.max(), 1000)
    y_stable = levy_stable.pdf(x, *params_stable)
    y_normal = norm.pdf(x, params_normal[0], params_normal[1])

    axs[1,1].hist(weights, 1500, density=True)
    axs[1,1].plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
    axs[1,1].plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')    
    #axs[1,1].set_ylim(0,1)    
    """
    
    # save data and figure
    data_name = replace_name(weight_name,'plfit')
    df_pl.to_csv(f'{model_path}/{data_name}.csv', index=False)
    print(df_pl)

    plt.legend(loc = 'lower left')
    plot_name = replace_name(weight_name,'plot')
    plt.savefig(f"{model_path}/{plot_name}.pdf", bbox_inches='tight')     
    #plt.show()


    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!")     

# functions: pretrained_plfit(), pretrained_ww_plfit
def plfit_submit(*args):
    global df, pbs_array_data, n_weights, pbss

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    if pytorch:
        #save_dir = join(main_path, "ww_plfit_all")
        save_dir = join(main_path, "ww_plfit_v2")
    else:
        save_dir = join(main_path, "ww_plfit_all_tf")

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH    
    pbs_array_data = []
    n_weights = []  # just for noting
    
    #for n_weight in list(range(total_weights)):     
    for n_weight in [2751, 4973]:  # first one is dummy since at least 2 jobs are required
        
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        #plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(save_dir, model_name, f"{replace_name(weight_name,'plfit')}.csv") )
        #fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )        
        #if not (plot_exist or fit_exist1):
        #if not (fit_exist1 or fit_exist2):
        if not fit_exist1:
            pbs_array_data.append( (root_path, save_dir, n_weight) )
            n_weights.append(n_weight)
 
    #pbs_array_data = pbs_array_data[:4]  # delete
    print(len(pbs_array_data))

    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)                
        
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    #quit()  # delete
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
        #qsub(f'singularity exec dist_fit.sif python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             #path=join(main_path,'jobs_all/ww_plfit_all'),  
             #path=join(main_path,'jobs_all', args[0]),
             #path=join(main_path,'jobs_all', args[0], '20240326'),
             #path=join(main_path,'jobs_all', args[0], '20240326_48h'),
             path=join(main_path,'jobs_all', args[0], 'final_sets'),
             P=project_ls[pidx], 
             #source="virt-test-qu/bin/activate",
             #walltime='23:59:59',
             walltime='47:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="6GB")   


def plfit_submit_terminal():
    global df, pbs_array_data, n_weights

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    if pytorch:
        save_dir = join(main_path, "ww_plfit_all")
    else:
        save_dir = join(main_path, "ww_plfit_all_tf")

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH    
    pbs_array_data = []
    n_weights = []  # just for noting
    
    #for n_weight in range(50):
    #for n_weight in [699,700,701,702]:
    for n_weight in list(range(total_weights)):
    #for n_weight in list(range(10)): 
        
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        #plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(save_dir, model_name, f"{replace_name(weight_name,'plfit')}.csv") )
        #fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )        
        #if not (plot_exist or fit_exist1):
        #if not (fit_exist1 or fit_exist2):
        if not fit_exist1:
            pbs_array_data.append( (root_path, save_dir, n_weight) )
            n_weights.append(n_weight)
 
    #pbs_array_data = pbs_array_data[:500]
    print(f"Total jobs: {len(pbs_array_data)}")

    for ii, args in enumerate(pbs_array_data):
        pretrained_ww_plfit(*args, True, False)


# functions: pretrained_moments()
def mmt_submit(*args):
    global df, pbs_array_data, n_weights, pbss

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    if pytorch:
        save_dir = join(main_path, "moments_v2")

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH    
    pbs_array_data = []
    n_weights = []  # just for noting
    
    for n_weight in list(range(total_weights)):
    #for n_weight in list(range(4500, 6000)): 
        
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        fit_exist1 = isfile( join(save_dir, model_name, f"{replace_name(weight_name,'mmt')}.csv") )
        #fit_exist1 = False
        if not fit_exist1:
            pbs_array_data.append( (root_path, save_dir, n_weight) )
            n_weights.append(n_weight)
 
    #pbs_array_data = pbs_array_data[:500]
    print(len(pbs_array_data))

    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)                
        
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    quit()  # delete
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
             pbs_array_true, 
             path=join(main_path,'jobs_all', args[0]),
             P=project_ls[pidx], 
             #source="virt-test-qu/bin/activate",
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="8GB")   


# -------------------- Batched tail fitting for pretrained weights --------------------

# batch version of pretrained_moments
def batch_pretrained_mmt(weight_path, n_weights, *args):
    """
    Batches arg n_weight for pretrained_allfit()
    """
    global n_weights_copy, data_type  

    print(f'weight_path = {weight_path}')
    print(f'n_weights = {n_weights}')
    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    print(n_weights)
    n_weights_copy = n_weights
    data_type = type(n_weights)
    assert isinstance(n_weights, list), "n_weights is not a list!"

    #save_dir1 = join(root_data, "pretrained_workflow", "moments")
    save_dir1 = join(root_data, "pretrained_workflow", "moments_v2")
    if not isdir(save_dir1): makedirs(save_dir1)
    for n_weight in tqdm(n_weights):
        pretrained_moments(weight_path, save_dir1, n_weight)

    print(f"Batch completed for {weight_path} for n_weights: {n_weights}")

# batch version of pretrained_ww_plfit
def batch_pretrained_plfit(weight_path, n_weights, *args):

    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights, list), "n_weights is not a list!"

    #save_dir1 = join(root_data, "pretrained_workflow", "ww_plfit_all")
    save_dir1 = join(root_data, "pretrained_workflow", "ww_plfit_v2")
    for n_weight in tqdm(n_weights):
        pretrained_ww_plfit(weight_path, save_dir1, n_weight)

    print(f"Batch completed for {weight_path} for n_weights: {n_weights}")


# batch version of pretrained_1samp_test()
def batch_pretrained_1samp(weight_path, n_weights, *args):

    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights, list), "n_weights is not a list!"

    save_dir2 = join(root_data, "pretrained_workflow", "all_onesamp_all")
    for n_weight in tqdm(n_weights):
        pretrained_1samp_test(weight_path, save_dir2, n_weight)

    print(f"Batch completed for {weight_path} for n_weights: {n_weights}")    


# functions: batch_pretrained_allfit(), batch_pretrained_mmt(), batch_pretrained_plfit(), batch_pretrained_1samp()
def batch_submit(*args):
    global pbss, n_weightss, pbs_array_data, total_weights, root_path, perm

    pytorch = True
    #chunks = 60
    #chunks = 30
    chunks = 10
    #chunks = 5
    #chunks = 1

    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)
    #allfit_folder = "ww_plfit_all" if pytorch else "ww_plfit_all_tf"
    #allfit_folder = 'ww_plfit_v2'
    #allfit_folder = 'moments'
    allfit_folder = "moments_v2"
    fit_path1 = join(os.path.dirname(root_path), allfit_folder) 

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH    
        
    is_first_time = True
    if is_first_time:
        n_weights = list(range(total_weights))
    else:
        n_weights = []
        for n_weight in list(range(total_weights)):
            model_name = df.loc[n_weight,"model_name"]
            weight_name = df.loc[n_weight,"weight_file"]
            i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
            #plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
            fit_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plfit')}.csv") )
            #fit_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'mmt')}.csv") )
            #if not (plot_exist and fit_exist):
            if not fit_exist:            
                n_weights.append(n_weight)
            #n_weights.append(n_weight)

    #n_weights = n_weights[10:]  # delete
    print(f'Total weights: {len(n_weights)}')

    n_weightss = list_str_divider(n_weights, chunks)
    pbs_array_data = []
    for n_weights in n_weightss:
        pbs_array_data.append( (root_path, n_weights) )
    
    print(f'Total jobs: {len(pbs_array_data)}')    
    
    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)    

    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    # debug
    #pbss = pbss[:1]  # delete
    #print(len(pbss))   
    # perm = [0]
    # pbss = [[("/project/PDLAI/project2_data/pretrained_workflow/weights_all", "[0,1,2]"), 
    #          ("/project/PDLAI/project2_data/pretrained_workflow/weights_all", "[3,4,5]")]]

    quit()  # delete
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=join(main_path, "jobs_all", args[0]),
             P=project_ls[pidx], 
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             #mem="1GB")
             mem="4GB")
             #mem="5GB")
             #mem="6GB")
             #mem="10GB")        
    
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
    #print(f"Min weight: {weights.min()}, Max weight: {weights.max()}")
    percs = [5e-6, 50, 50, 99.999995]
    percentiles = [np.percentile(weights, per) for per in percs]
    pl1, pl2, pu1, pu2 = percentiles
    #print(f"Percentiles at {percs}: {percentiles}")

    # Plots ----------------------    

    # x-axis bounds for 3 plots
    bd = min(np.abs(pl1), np.abs(pu2))
    x = np.linspace(-bd, bd, 1000)
    xbds = [[-bd,bd], [pu1, pu2], [pl1, pl2]]

    fig, axs = plt.subplots(1, 3, sharex = False,sharey=False,figsize=(12.5 + 4.5, 9.5/3 + 2.5))
    # plot 1 (full distribution); # plot 2 (log-log hist right tail); # plot 3 (left tail)
    for aidx in range(len(axs)):
        axs[aidx].hist(weights, bins=2000, density=True)
        sns.kdeplot(weights, color='blue', ax=axs[aidx])

        x = np.linspace(xbds[aidx][0], xbds[aidx][1], 1000)
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


def remove_repeated_files(weight_path):
    """
    Remove fitted files in model_path2 if they both exist in model_path1 and model_path2
    """

    weight_path = weight_path if weight_path[-1] != '/' else weight_path[:-1]    
    pytorch = False if "_tf" in weight_path else True
    upper_weight_path = os.path.dirname(weight_path)

    main_path = join(root_data,"pretrained_workflow")
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))

    # dir for potentially previously fitted params
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"      

    n = 0
    exists1_ls = []; exists2_ls = []
    for n_weight in tqdm(range(df.shape[0])):

        weight_name = df.loc[n_weight,"weight_file"]
        model_name = df.loc[n_weight,"model_name"]
        model_path1 = join(upper_weight_path, allfit_folder1, model_name)
        model_path2 = join(upper_weight_path, allfit_folder2, model_name) 

        data_name = replace_name(weight_name,'allfit') 
        df_name = data_name + ".csv"
        fit_exists1 = isfile( join(model_path1, df_name) )
        fit_exists2 = isfile( join(model_path2, df_name) )

        if fit_exists1 and fit_exists2:
            os.remove(join(model_path2, df_name))
            n += 1

    print(f"{n} files removed from {allfit_folder2}!")

# fitting to stable, Gaussian, Student-t, lognormal distribution
"""
Initial version filters out very small values, i.e. weights = weights[np.abs(weights) >0.00001]
"""
def pretrained_allfit(weight_path, n_weight, with_logl, remove_weights=True):
    """
    Fit the the entire entries of a single pretrained weight matrix to one of the 4 distributions: "levy_stable", "normal", "tstudent", "lognorm".
        - weight_path (str): dir of the stored pretrained weights
            1. 
        - n_weight (int): index of the pretrained weight matrix in the folder/recorded csv file
        - with_logl (bool): whether to compute the log-likelihood
        - remove_weights (bool): whether to remove weights smaller or equal than 0.00001
    """

    global weights, params, df, plot_title, x, params

    t0 = time.time()
    n_weight = int(n_weight)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    with_logl = with_logl if isinstance(with_logl,bool) else literal_eval(with_logl)
    remove_weights = remove_weights if isinstance(remove_weights,bool) else literal_eval(remove_weights)

# Loading weight matrix ----------------------       

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    #weight_path = join(main_path, "weights_all")

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]
    print(f"n_weight = {n_weight}: {weight_name}")

    # dir for potentially previously fitted params
    """
    first 'all' refers to full dist, second 'all' refers to fitted to all types of distribution, i.e. normal, stable, T-student etc
    """
    prefix = "all" if remove_weights else "noremove"  # no weights removed

    allfit_folder1 = f"{prefix}fit_all" if pytorch else f"{prefix}fit_all_tf"
    model_path1 = join(os.path.dirname(weight_path), allfit_folder1, model_name)
    allfit_folder2 = f"nan_{prefix}fit_all" if pytorch else f"nan_{prefix}fit_all_tf"
    model_path2 = join(os.path.dirname(weight_path), allfit_folder2, model_name)   
    if not os.path.exists(model_path1): os.makedirs(model_path1)
    if not os.path.exists(model_path2): os.makedirs(model_path2)
    print(f"model_path1: {model_path1}" + "\n" + f"model_path2: {model_path2}")

    # check if previously fitted
    data_name = replace_name(weight_name,'allfit') 
    df_name = data_name + ".csv"
    plot_name = replace_name(weight_name,'plot') + ".pdf"
    print(f"df_name: {df_name}")
    print(f"plot_name: {plot_name}")
    plot_exists = isfile( join(model_path1, plot_name) )
    fit_exists1 = isfile( join(model_path1, df_name) )
    fit_exists2 = isfile( join(model_path2, df_name) )
    print(f"plot_exists: {plot_exists}, fit_exists1: {fit_exists1}, fit_exists2: {fit_exists2}")

    # ---------- 1. fit and test ----------

    if not (fit_exists1 or fit_exists2):
        print("Start fitting from scratch!")

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
        print(f"Weight min: {min(weights)} and max: {max(weights)}")
        if remove_weights:
            # values significantly smaller than zero are filtered out         
            weights = weights[np.abs(weights) >0.00001]            

        print(f"True size: {w_size}")
        print(f"Fit size: {len(weights)}")        

        if len(weights) < 100:
            if remove_weights:
                print(f"After removing weights, there are {len(weights)} entries! \n")
            else:
                print(f"Even keeping all weights, there are {len(weights)} entries! \n")
            return False

        # save params
        df.iloc[0,0:3] = [wmat_idx, w_size, len(weights)]

        # Fitting
        all_logl_defined = True
        index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
        for dist_type in tqdm(["levy_stable", "normal", "tstudent", "lognorm"]):
            idxs = index_dict[dist_type]
            if dist_type == "normal":
                params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue, is_logl_defined = fit_and_test(weights, dist_type, with_logl)
                df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
            else:
                params, logl, ad_siglevel, ks_stat, ks_pvalue, is_logl_defined = fit_and_test(weights, dist_type, with_logl) 
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

    elif fit_exists2 and (not fit_exists1):
        print("Fix ill logls!")

        df = pd.read_csv(join(model_path2, f"{data_name}.csv"))
        # check entries 7, 13, 22, 29
        logl_idxs = [7, 13, 22, 29]
        bd_idxs = [[3,7,11], [11,13,19], [19,22,26], [26,29,33]]
        dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']
        weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
        if remove_weights:
            # values significantly smaller than zero are filtered out         
            weights = weights[np.abs(weights) >0.00001]

        # refit the whole distribution
        ill_logl_dists = []
        for dist_idx, logl_idx in enumerate(logl_idxs):
            logl = df.iloc[0, logl_idx]            
            dist_type = dist_types[dist_idx]
            idxs = bd_idxs[dist_idx]
            if not is_num_defined(logl):                
            #if not np.isnan(logl):
                """
                if dist_type == "normal":
                    params, logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue, is_logl_defined = fit_and_test(weights, dist_type, with_logl)
                    df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue, shapiro_stat, shapiro_pvalue]
                else:
                    params, logl, ad_siglevel, ks_stat, ks_pvalue, is_logl_defined = fit_and_test(weights, dist_type, with_logl) 
                    df.iloc[0,idxs[1]:idxs[2]] = [logl, ad_siglevel, ks_stat, ks_pvalue]  

                df.iloc[0,idxs[0]:idxs[1]] = list(params)
                """
                params = list(df.iloc[0,idxs[0]:idxs[1]])
                logl = logl_from_params(weights, params, dist_type)
                is_logl_defined = is_num_defined(logl)

                if is_logl_defined:
                    df.iloc[0,idxs[1]] = logl
                    print(f"{dist_type} logl updated!")
                else:
                    ill_logl_dists.append(dist_type)
                    print(f"{dist_type} logl still ill-defined!")        
                print('\n')

        all_logl_defined = True
        for dist_idx, logl_idx in enumerate(logl_idxs):
            all_logl_defined = all_logl_defined and is_num_defined(df.iloc[0, logl_idx])
            #all_logl_defined = all_logl_defined and np.isnan(df.iloc[0, logl_idx])     

        if all_logl_defined:
            df.to_csv(f'{model_path1}/{data_name}.csv', index=False)
            # delete original file from the nan verion
            os.remove(join(model_path2, f"{data_name}.csv"))
            print(f"no ill logls, old df deleted from {model_path2}")
        else:
            df.to_csv(f'{model_path2}/{data_name}.csv', index=False)
            print(f"ill logls in distributions fits from {ill_logl_dists} still exist")

    else:
        df = pd.read_csv(join(model_path1, df_name))
        print("Fitting already done!")

    # plot hist and fit
    """
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
    """

    # Time
    t_last = time.time()
    print(f"{weight_name} done in {t_last - t0} s!")    

# extra 1-sample KS test
def pretrained_1samp_test(weight_path, save_dir, n_weight, remove_weights=True):
    """
    Perform 1-sample KS test to one of the 4 distributions: "levy_stable", "normal", "tstudent", "lognorm".
        args same as pretrained_allfit
    """

    global weights, params, df, df_fitted, res_ad, params, model_path2, df_name2, model_path1, df_name, if_torch_weights

    t0 = time.time()
    n_weight = int(n_weight)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")
    remove_weights = remove_weights if isinstance(remove_weights,bool) else literal_eval(remove_weights)

# Loading weight matrix ----------------------       

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)

    # new method
    df_info = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df_info.loc[n_weight,"weight_file"]
    i, wmat_idx = int(df_info.loc[n_weight,"idx"]), int(df_info.loc[n_weight,"wmat_idx"])
    model_name = df_info.loc[n_weight,"model_name"]
    print(f"n_weight = {n_weight}: {weight_name}")

    # dir for potentially previously fitted params
    """
    first 'all' refers to full dist, second 'all' refers to fitted to all types of distribution, i.e. normal, stable, T-student etc
    """
    prefix = "all" if remove_weights else "noremove"  # no weights removed

    # dir to be saved in
    #ks_folder = f"{prefix}_onesamp_all" if pytorch else f"{prefix}_onesamp_tf"
    model_path1 = join(os.path.dirname(weight_path), save_dir, model_name)   

    # dir which already has fitting data
    allfit_folder2 = f"{prefix}fit_all" if pytorch else f"{prefix}fit_all_tf"
    model_path2 = join(os.path.dirname(weight_path), allfit_folder2, model_name)

    if not os.path.exists(model_path1): os.makedirs(model_path1)
    print(f"model_path1: {model_path1}" + "\n" + f"model_path2: {model_path2}" + "\n")

    # check if previously fitted
    data_name1 = replace_name(weight_name,'kstest') 
    df_name1 = data_name1 + ".csv"
    file_exists1 = isfile( join(model_path1, df_name1) )
    print(f"{join(model_path1, df_name1)} exists: {file_exists1} \n")

    data_name2 = replace_name(weight_name,'allfit') 
    df_name2 = data_name2 + ".csv"
    file_exists2 = isfile( join(model_path2, df_name2) )
    print(f"{join(model_path2, df_name2)} exists: {file_exists2} \n")    
    
    # ---------- 1. fit and test ----------

    assert isfile( join(model_path2, df_name2) ), f'weight_path = {weight_path}; n_weight = {n_weight} not fitted!'
    df_fitted = pd.read_csv(join(model_path2, df_name2))

    if not file_exists1:
        print("Getting one-sample tests!")

        col_names = ['wmat_idx','w_size', 'fit_size', 'skewness', 'kurtosis',         # 0 - 4  
                     'stat_jb', 'pval_jb',                                            # 5 - 6                  
                     'ks stat stable', 'ks pvalue stable',     # 7 - 8         
                     'ks stat normal', 'ks pvalue normal',     # 9 - 10
                     'ks stat tstudent', 'ks pvalue tstudent', # 11 - 12                             
                     'ks stat lognorm', 'ks pvalue lognorm'    # 13 - 14                                                                                  
                     ]  
        col_idxss = [[0, 5], [5, 7], [7, 9], [9, 11], [11, 13], [13, 15]]

        index_dict2 = {'levy_stable': [7, 9], 'normal': [9, 11], 'tstudent': [11, 13], 'lognorm': [13, 15]}


        df = pd.DataFrame(np.zeros((1,len(col_names)))) 
        df = df.astype('object')
        df.columns = col_names    

        weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
        w_size = len(weights)
        print(f"Weight min: {min(weights)} and max: {max(weights)}")
        if remove_weights:
            # values significantly smaller than zero are filtered out         
            weights = weights[np.abs(weights) >0.00001]            

        print(f"True size: {w_size}")
        print(f"Fit size: {len(weights)}")        

        # if len(weights) < 100:
        #     if remove_weights:
        #         print(f"After removing weights, there are {len(weights)} entries! \n")
        #     else:
        #         print(f"Even keeping all weights, there are {len(weights)} entries! \n")
        #     return False
        
        df.iloc[0,col_idxss[0][0]:col_idxss[0][1]] = [wmat_idx, w_size, len(weights), sst.skew(weights), sst.kurtosis(weights)]
        res_jb = sst.jarque_bera(weights)        
        df.iloc[0,col_idxss[1][0]:col_idxss[1][1]] = [res_jb.statistic, res_jb.pvalue]

        # Fitting
        index_dict1 = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
        for dist_type in tqdm(["levy_stable", "normal", "tstudent", "lognorm"]):
            idxs = index_dict1[dist_type]

            params = list(df_fitted.iloc[0,idxs[0]:idxs[1]])  
            # ----- 1. KS test -----              
            ks_stat, ks_pvalue = kstest_from_params(weights, params, dist_type)  
            print(f'KS test for {dist_type} done!')
            # # ----- 2. AD test -----
            # res_ad = adtest_from_params(weights, params, dist_type)
            # #ad_stat, ad_pvalue = 
            # print(f'AD test for {dist_type} done!')

            idxs2 = index_dict2[dist_type]
            df.iloc[0,idxs2[0]:idxs2[1]] = [ks_stat, ks_pvalue]

        print(f"{model_path1} directory set up, fitting now!")
        print("\n")

        quit()

        # Save params ----------------------
        df.to_csv(f'{model_path1}/{data_name1}.csv', index=False)
        print("df saved!")
        pd.set_option('display.max_columns', df.shape[1])
        print(df)

    else:
        print("1-sample KS and AD test already done!")


    # Time
    t_last = time.time()
    print(f"{weight_name} 1-sample KS test done in {t_last - t0} s!")    


def divided_logl(weight_path, n_weight, batch_idx, B):

    """
    Assuming the fitting is all done, 
    this is for getting the log-likelihood via dividing the weight matrix in to several batches
    """

    t0 = time.time()
    n_weight, batch_idx, B = int(n_weight), int(batch_idx), int(B)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

    assert "allfit" or "noremovefit" in weight_path, "weight_path incorrect"
    remove_weights = True
    if "noremovefit" in weight_path:
        remove_weights = False    

# Loading weight matrix ----------------------       

    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    #weight_path = join(main_path, "weights_all")

    # new method
    df = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    model_name = df.loc[n_weight,"model_name"]
    print(f"n_weight = {n_weight}: {weight_name}")

    # dir for potentially previously fitted params
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"
    model_path1 = join(os.path.dirname(weight_path), allfit_folder1, model_name)
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    model_path2 = join(os.path.dirname(weight_path), allfit_folder2, model_name)   
    if not os.path.exists(model_path1): os.makedirs(model_path1)
    if not os.path.exists(model_path2): os.makedirs(model_path2)
    print(f"model_path1: {model_path1}" + "\n" + f"model_path2: {model_path2}")

    # check if previously fitted
    data_name = replace_name(weight_name,'allfit') 
    df_name = data_name + ".csv"
    plot_name = replace_name(weight_name,'plot') + ".pdf"
    print(f"df_name: {df_name}")
    print(f"plot_name: {plot_name}")
    plot_exists = isfile( join(model_path1, plot_name) )  # if the distribution hist was plotted
    fit_exists1 = isfile( join(model_path1, df_name) )  # if the fitted distribution log-likelihood does not have nan values
    fit_exists2 = isfile( join(model_path2, df_name) )  # if the fitted distribution log-likelihood has nan values
    print(f"plot_exists: {plot_exists}, fit_exists1: {fit_exists1}, fit_exists2: {fit_exists2}")

    # there are no nan-valued logl for all fitted distributions
    if fit_exists1:
        print("Fitting done already!") 
        
    # nan-valued logl for all fitted distributions exist
    elif fit_exists2:
        # create appropriate dir
        partialfit_folder = "partialfit_all" if pytorch else "partialfit_all_tf"
        wmat_path = join(main_path, partialfit_folder, model_name, f"{n_weight}_{layer_idx}_{wmat_idx}")
        if not os.path.isdir(wmat_path): os.makedirs(wmat_path)

        df = pd.read_csv(join(model_path2, f"{data_name}.csv"))
        # check entries 7, 13, 22, 29
        logl_idxs = [7, 13, 22, 29]
        bd_idxs = [[3,7,11], [11,13,19], [19,22,26], [26,29,33]]
        dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']
        weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
        if remove_weights:
            # values significantly smaller than zero are filtered out         
            weights = weights[np.abs(weights) >0.00001]

        # evenly divide weights       
        nsize = int(np.ceil(len(weights)/B)) 
        idx_pairs = [[i, min(i+nsize, len(weights))] for i in range(0, len(weights), nsize)]
        #assert idx_pairs[-1][0] < idx_pairs[-1][1], "Zero chunk!"

        if idx_pairs[-1][0] < idx_pairs[-1][1]:  # zero chunk
            weights = weights[idx_pairs[batch_idx][0]:idx_pairs[batch_idx][1]]

        # refit the whole distribution
        #ill_logl_dists = []
        for dist_idx, logl_idx in enumerate(logl_idxs):
            logl = df.iloc[0, logl_idx]            
            dist_type = dist_types[dist_idx]
            idxs = bd_idxs[dist_idx]
            if not is_num_defined(logl):      
                df_divided_file = join(wmat_path, f"{dist_type}_batch={batch_idx}_B={B}.csv")
                if os.path.isfile(df_divided_file):
                    df_divided = pd.read_csv(df_divided_file)
                    is_partial_logl_defined = df_divided.iloc[0,1]
                else:
                    is_partial_logl_defined = False

                if bool(is_partial_logl_defined) == False:

                    # load fitted params
                    if idx_pairs[-1][0] < idx_pairs[-1][1]:
                        params = list(df.iloc[0,idxs[0]:idxs[1]])
                        logl = logl_from_params(weights, params, dist_type)
                        is_partial_logl_defined = is_num_defined(logl)
                    else:
                        logl = 0
                        is_partial_logl_defined = True

                    # ----- version 1 -----

                    # # create df if not exist
                    # df_divided_file = join(wmat_path, f"{dist_type}_{B}.csv")
                    # if os.path.isfile(df_divided_file):
                    #     df_divided = pd.read_csv(df_divided_file)
                    # else:
                    #     df_divided = pd.DataFrame({'partial_logl': [0]*B, 'is_defined': [False]*B, 'entries': [0]*B})                  

                    # df_divided.iloc[batch_idx,0] = logl
                    # df_divided.iloc[batch_idx,1] = is_partial_logl_defined
                    # df_divided.iloc[batch_idx,2] = len(weights)                

                    # ----- version 2 (2024-01-17) -----

                    if os.path.isfile(df_divided_file):
                        df_divided = pd.read_csv(df_divided_file)
                    else:
                        df_divided = pd.DataFrame({'partial_logl': [0]*1, 'is_defined': [False]*1, 'entries': [0]*1})                  

                    df_divided.iloc[0,0] = logl
                    df_divided.iloc[0,1] = is_partial_logl_defined
                    df_divided.iloc[0,2] = len(weights)

                    # save dataframe
                    df_divided.to_csv(df_divided_file, index=False)                

                    print(f"n_weight = {n_weight} with batch_idx {batch_idx} has partial LL {logl}, which is {is_partial_logl_defined}")

                    """
                    if is_partial_logl_defined:
                        df.iloc[0,idxs[1]] = logl
                        print(f"{dist_type} logl updated!")
                    else:
                        ill_logl_dists.append(dist_type)
                        print(f"{dist_type} logl still ill-defined!")        
                    print('\n')
                    """

                else:
                    print(f"Partial logl for {dist_type} done already!")

    # Time
    t_last = time.time()
    print(f"{weight_name} at batch_idx {batch_idx}/{B} done in {t_last - t0} s!")        


def group_divided_logl(weight_path, n_weight, B=15):
    """
    Grouping together the divided logl's
    """

    global df_info, df

    t0 = time.time()
    n_weight, B = int(n_weight), int(B)
    #pytorch = pytorch if isinstance(pytorch,bool) else literal_eval(pytorch)
    pytorch = False if "_tf" in weight_path else True
    print(weight_path.split("/")[-1][:3])
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

    #assert "allfit" or "noremovefit" in weight_path, "weight_path incorrect"
    remove_weights = True
    #if "noremovefit" in weight_path:
    #    remove_weights = False    

# Loading weight matrix ----------------------       

    main_path = join(root_data,"pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)
    #weight_path = join(main_path, "weights_all")

    # new method
    df_info = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = df_info.loc[n_weight,"weight_file"]
    layer_idx, wmat_idx = int(df_info.loc[n_weight,"idx"]), int(df_info.loc[n_weight,"wmat_idx"])
    model_name = df_info.loc[n_weight,"model_name"]
    print(f"n_weight = {n_weight}: {weight_name}")

    # dir for potentially previously fitted params
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"
    model_path1 = join(os.path.dirname(weight_path), allfit_folder1, model_name)
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    model_path2 = join(os.path.dirname(weight_path), allfit_folder2, model_name)   
    if not os.path.exists(model_path1): os.makedirs(model_path1)
    if not os.path.exists(model_path2): os.makedirs(model_path2)
    print(f"model_path1: {model_path1}" + "\n" + f"model_path2: {model_path2}")

    # check if previously fitted
    data_name = replace_name(weight_name,'allfit') 
    df_name = data_name + ".csv"
    plot_name = replace_name(weight_name,'plot') + ".pdf"
    print(f"df_name: {df_name}")
    print(f"plot_name: {plot_name}")
    plot_exists = isfile( join(model_path1, plot_name) )  # if the distribution hist was plotted
    fit_exists1 = isfile( join(model_path1, df_name) )  # if the fitted distribution log-likelihood does not have nan values
    fit_exists2 = isfile( join(model_path2, df_name) )  # if the fitted distribution log-likelihood has nan values
    print(f"plot_exists: {plot_exists}, fit_exists1: {fit_exists1}, fit_exists2: {fit_exists2}")

    # there are no nan-valued logl for all fitted distributions
    if fit_exists1:
        print("Fitting done already!") 
        return True
        
    # nan-valued logl for all fitted distributions exist
    elif fit_exists2:        
        partialfit_folder = "partialfit_all" if pytorch else "partialfit_all_tf"
        wmat_path = join(main_path, partialfit_folder, model_name, f"{n_weight}_{layer_idx}_{wmat_idx}")  
                
        df = pd.read_csv(join(model_path2, df_name))
        # check entries 7, 13, 22, 29
        logl_idxs = [7, 13, 22, 29]
        bd_idxs = [[3,7,11], [11,13,19], [19,22,26], [26,29,33]]  # not useful
        dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']

        is_all_logl_defined = True  # if logl for one dist is defined
        is_all_dist_defined = True  # if logl for all dists are defined

        # refit the whole distribution
        #ill_logl_dists = []
        for dist_idx, logl_idx in enumerate(logl_idxs):
            logl_og = df.iloc[0, logl_idx]  # original log            
            dist_type = dist_types[dist_idx]
            idxs = bd_idxs[dist_idx]
            if not is_num_defined(logl_og):                                 

                # ----- version 2 (2024-02-08) -----

                logl_new = 0
                for batch_idx in range (B):
                    df_divided_file = join(wmat_path, f"{dist_type}_batch={batch_idx}_B={B}.csv")
                    if os.path.isfile(df_divided_file):
                        df_divided = pd.read_csv(df_divided_file)
                        logl_partial = df_divided.iloc[0,0]
                        logl_new += logl_partial
                        is_partial_logl_defined = bool(df_divided.iloc[0,1])
                        is_all_dist_defined = is_all_dist_defined and is_partial_logl_defined
                        is_all_logl_defined = is_all_logl_defined and is_partial_logl_defined
                        if not is_partial_logl_defined:                            
                            break                     
                    else:
                        is_all_dist_defined = False
                        is_all_logl_defined = False
                        break               
                
                if is_all_logl_defined:
                    df.iloc[0,idxs[1]] = logl_new
                    print(f"{dist_type} logl updated!")
                else:
                    #ill_logl_dists.append(dist_type)
                    print(f"{dist_type} logl still ill-defined!")        
                print('\n')                

        # save df
        if is_all_dist_defined:
            df.to_csv(join(model_path1, df_name))
            return True
        else:
            df.to_csv(join(model_path2, df_name))
            return False

    else:
        return False

    # Time
    t_last = time.time()
    print(f"{weight_name} at batch_idx {batch_idx}/{B} done in {t_last - t0} s!")        



def pre_submit(pytorch: bool, prefix="weights_all"):
    assert prefix == "weights_all" or prefix == "np_weights_all"

    main_path = join(root_data, "pretrained_workflow")
    if not os.path.isdir(main_path): os.makedirs(main_path)    

    if pytorch:
        # --- Pytorch ---
        root_path = join(main_path, prefix)
        #root_path = join(main_path, "np_weights_all")
        df = pd.read_csv(join(main_path, "weight_info.csv"))
    else:
        # ---TensorFlow ---
        #root_path = join(main_path, "weights_all_tf")
        root_path = join(main_path, f"{prefix}_tf")
        df = pd.read_csv(join(main_path, "weight_info_tf.csv"))

    print(root_path)
    weights_all = next(os.walk(root_path))[2]
    weights_all.sort()
    total_weights = len(weights_all)
    
    print(df.shape)
    assert total_weights == df.shape[0]   

    return main_path, root_path, df, weights_all, total_weights

# pretrained_allfit() or pretrained_1samp_test()
def submit(*args):
    global pbs_array_data

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2) 

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH
    pbs_array_data = []
    
    # pretrained_allfit()    
    # first time
    with_logl = True
    for n_weight in [0,1]:
        pbs_array_data.append( (root_path, n_weight, with_logl) )      

    # second time   
    # with_logl = False
    # for n_weight in list(range(total_weights)):
    # #for n_weight in list(range(10)): 
    #     model_name = df.loc[n_weight,"model_name"]
    #     weight_name = df.loc[n_weight,"weight_file"]
    #     i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    #     plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
    #     fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
    #     fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )        
    #     #if not (plot_exist or fit_exist1):
    #     if not (fit_exist1 or fit_exist2):
    #         pbs_array_data.append( (root_path, n_weight, with_logl) )  

    ### OR ###

    # pretrained_1samp_test() 
    # for n_weight in list(range(total_weights)):
    # #for n_weight in list(range(10)): 
    #     model_name = df.loc[n_weight,"model_name"]
    #     weight_name = df.loc[n_weight,"weight_file"]
    #     i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
    #     fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
    #     fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'kstest')}.csv") )        
    #     if fit_exist1 and not fit_exist2:     
    #         pbs_array_data.append( (root_path, n_weight) )            
 
    #pbs_array_data = pbs_array_data[:10] 
    print(len(pbs_array_data))

    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)       

    #quit()

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
        #qsub(f'singularity exec dist_fit.sif python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             #path=join(main_path,"jobs_all","stablefit_nologl"),  
             path=join(main_path,"jobs_all","alexnet_allfit"),
             P=project_ls[pidx], 
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="6GB")       


# for running remaining jobs from pretrained_allfit() directly in the terminal
def submit_terminal():
    global pbs_array_data

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2) 

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH
    pbs_array_data = []
    
    with_logl = False
    for n_weight in list(range(total_weights)):
    #for n_weight in list(range(10)): 
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )        
        #if not (plot_exist or fit_exist1):
        if not (fit_exist1 or fit_exist2):
            pbs_array_data.append( (root_path, n_weight, with_logl) )
    
    print(f"Total jobs: {len(pbs_array_data)}")

    status_all = []
    for ii, args in enumerate(pbs_array_data):
        #pretrained_allfit(args[0], args[1], args[2])
        status = pretrained_allfit(*args)
        status_all.append(status)

    print(f"Failed statuses: {len(status_all)}")

#  functions: divided_logl()
def divided_submit(*args):
    global df_divided, wmat_path_dir_files, wmat_path_dirs, wmat_paths, partialfit_folder, wmat_info, batch_idxs, pbs_array_data 
    global n_weights    

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2)  
    partialfit_folder = "partialfit_all" if pytorch else "partialfit_all_tf"
    partialfit_folder = join(main_path, partialfit_folder)    

    dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH
    pbs_array_data = []
    
    B = 15
    # initial submit    
    n_weights = []
    for n_weight in list(range(total_weights)):
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )                  
        if fit_exist2:  
            n_weights.append(n_weight)
            for batch_idx in range(B):
                pbs_array_data.append( (root_path, n_weight, batch_idx, B) )          

    print(f"Total weight matrices: {len(n_weights)}")

    # remainder submit
    
    # neworks paths that have nan logl values
    wmat_paths = [join(partialfit_folder,f) for f in os.listdir(partialfit_folder) if os.path.isdir(join(partialfit_folder,f))]
    for wmat_path in wmat_paths:

        wmat_path_dirs = [join(wmat_path,f) for f in os.listdir(wmat_path) if os.path.isdir(join(wmat_path,f))]
        for wmat_path_dir in wmat_path_dirs:
            wmat_info = wmat_path_dir.split("/")[-1]
            if wmat_info == "":
                wmat_info = wmat_path_dir.split("/")[-2]
            n_weight, i, wmat_idx = wmat_info.split("_")
            wmat_path_dir_files = [join(wmat_path_dir, f) for f in os.listdir(wmat_path_dir) if f'_{B}.csv' in f]
            batch_idxs = []
            for wmat_path_dir_file in wmat_path_dir_files:
                df_divided = pd.read_csv(wmat_path_dir_file)
                for batch_idx in range(df_divided.shape[0]):
                    if df_divided.iloc[batch_idx, 1] == False:
                        batch_idxs.append( batch_idx )
                batch_idxs = list(set(batch_idxs))
                for batch_idx in batch_idxs:
                    pbs_array_data.append( (root_path, n_weight, batch_idx, B) )    
 
    #pbs_array_data = pbs_array_data[B:]
    print(len(pbs_array_data))
    #print(pbs_array_data)    

    quit()    
    
    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)    

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
        #qsub(f'singularity exec dist_fit.sif python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=join(main_path, "jobs_all", "divided_logl_jobs"),  
             P=project_ls[pidx], 
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="4GB")     
    

# group_divided_logl()
def group_divided_submit(*args):
    global df_divided, wmat_path_dir_files, wmat_path_dirs, wmat_paths, partialfit_folder, wmat_info, batch_idxs, pbs_array_data 
    global n_weights    

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2)  
    partialfit_folder = "partialfit_all" if pytorch else "partialfit_all_tf"
    partialfit_folder = join(main_path, partialfit_folder)    

    dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH
    pbs_array_data = []
    
    B = 15
    # initial submit    
    n_weights = []
    for n_weight in list(range(total_weights)):
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )                  
        if not fit_exist1 and fit_exist2:  
            n_weights.append(n_weight)
            pbs_array_data.append( (root_path, n_weight, B) )                      
 
    #pbs_array_data = pbs_array_data[B:]
    print(len(pbs_array_data))
    #print(pbs_array_data)    

    #quit()    
    
    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)    

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',
        #qsub(f'singularity exec dist_fit.sif python {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             path=join(main_path, "jobs_all", "group_logl_jobs"),  
             P=project_ls[pidx], 
             walltime='23:59:59',
             ncpus=ncpus,
             ngpus=ngpus,
             mem="1GB")     

def group_divided_run():
    global df_divided, wmat_path_dir_files, wmat_path_dirs, wmat_paths, partialfit_folder, wmat_info, batch_idxs, pbs_array_data 
    global n_weights, completions    

    pytorch = True
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2)  
    partialfit_folder = "partialfit_all" if pytorch else "partialfit_all_tf"
    partialfit_folder = join(main_path, partialfit_folder)    

    dist_types = ['levy_stable', 'normal', 'tstudent', 'lognorm']

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH
    pbs_array_data = []
    
    B = 15
    # initial submit    
    n_weights = []
    completions = []
    for n_weight in list(range(total_weights)):
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist1 = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        fit_exist2 = isfile( join(fit_path2, model_name, f"{replace_name(weight_name,'allfit')}.csv") )                  
        if not fit_exist1 and fit_exist2:  
            n_weights.append(n_weight)
            pbs_array_data.append( (root_path, n_weight, B) )                      
 
            completion = group_divided_logl(root_path, n_weight, B)
            completions.append(completion)

        else:
            completions.append(True)                           

# -------------------- Grouping batched pretrained weights matrix fitting --------------------             

def group_batched_fit(partialfit_folder):
    """
    Grouping log-likelihood from divided batches of weights into the full log-likelihood.
    """    

    global df, df_divided

    pytorch = False if "_tf" in partialfit_folder else True    
    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)       
    allfit_folder1 = "allfit_all" if pytorch else "allfit_all_tf"       
    fit_path1 = join(os.path.dirname(root_path), allfit_folder1) 
    allfit_folder2 = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
    fit_path2 = join(os.path.dirname(root_path), allfit_folder2)     

    B = 15
    logl_idxs = [7, 13, 22, 29]
    dist_dict = {'levy_stable':0, 'normal':1, 'tstudent':2, 'lognorm':3}

    wmat_paths = [join(partialfit_folder,f) for f in os.listdir(partialfit_folder) if os.path.isdir(join(partialfit_folder,f))]
    for wmat_path in wmat_paths:
        wmat_path_dirs = [join(wmat_path,f) for f in os.listdir(wmat_path) if os.path.isdir(join(wmat_path,f))]
        for wmat_path_dir in wmat_path_dirs:
            wmat_info = wmat_path_dir.split("/")[-1]
            model_name = wmat_path_dir.split("/")[-2]
            if wmat_info == "":
                wmat_info = wmat_path_dir.split("/")[-2]
                model_name = wmat_path_dir.split("/")[-3]
            n_weight, layer_idx, wmat_idx = wmat_info.split("_")
            wmat_path_dir_files = [join(wmat_path_dir, f) for f in os.listdir(wmat_path_dir) if f'_{B}.csv' in f]
            batch_idxs = []
            for wmat_path_dir_file in wmat_path_dir_files:
                dist_type = wmat_path_dir_file.split("/")[-1]
                dist_type = dist_type[:-7]
                df_divided = pd.read_csv(wmat_path_dir_file)
                if df_divided.iloc[:,1].sum() == df_divided.shape[0]:
                    df_fname = join(fit_path2, model_name, f"{model_name}_allfit_{layer_idx}_{wmat_idx}.csv")
                    df = pd.read_csv(df_fname)
                    #print(df)   # delete
                    col = logl_idxs[dist_dict[dist_type]]
                    df.iloc[0,col] = df_divided.iloc[:,0].sum()
                    print(f"Corrected {dist_type}")

                    # save df
                    #df.to_csv(df_fname)                    

# -------------------- Batched full distribution fitting for pretrained weights --------------------

def batch_pretrained_allfit(weight_path, n_weights, with_logl=True):
    """
    Batches arg n_weight for pretrained_allfit()
    """

    if isinstance(n_weights,str):
        n_weights = literal_eval(n_weights)
    assert isinstance(n_weights, list), "n_weights is not a list!"

    for n_weight in tqdm(n_weights):
        pretrained_allfit(weight_path, n_weight, with_logl=with_logl)

    print(f"Batch completed for {weight_path} for n_weights: {n_weights}")

# functions: batch_pretrained_allfit()
def batch_allfit_submit(*args):
    global pbss

    pytorch = True
    chunks = 2

    main_path, root_path, df, weights_all, total_weights = pre_submit(pytorch)
    allfit_folder = "allfit_all" if pytorch else "allfit_all_tf"
    fit_path1 = join(os.path.dirname(root_path), allfit_folder) 

    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH 
    
    n_weights = []
    for n_weight in list(range(total_weights)):
        model_name = df.loc[n_weight,"model_name"]
        weight_name = df.loc[n_weight,"weight_file"]
        i, wmat_idx = int(df.loc[n_weight,"idx"]), int(df.loc[n_weight,"wmat_idx"])
        plot_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'plot')}.pdf") )
        fit_exist = isfile( join(fit_path1, model_name, f"{replace_name(weight_name,'allfit')}.csv") )
        #if not (plot_exist and fit_exist):
        if not fit_exist:            
            n_weights.append(n_weight)
     
    print(f"Remaining n_weight: {len(n_weights)}")

    n_weightss = list_str_divider(n_weights, chunks)
    pbs_array_data = []
    for n_weights in n_weightss:
        pbs_array_data.append( (root_path, n_weights) )

    print(len(pbs_array_data))    
    
    ncpus, ngpus = 1, 0
    command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)    

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    #pbss = pbss[:-1]  # delete
    for idx, pidx in enumerate(perm):
    #for idx, pidx in enumerate(perm[:-1]):  # delete
        pbs_array_true = pbss[idx]
        #pbs_array_true = pbs_array_true[:2]  # delete
        print(project_ls[pidx])

        qsub(f'{command} {sys.argv[0]} {" ".join(args)}', 
             pbs_array_true, 
             #path=join(main_path,"jobs_all","new_stablefit"),  
             path=join(main_path,"jobs_all","new_stablefit_remainder"),
             P=project_ls[pidx], 
             ncpus=ncpus,
             ngpus=ngpus,
             mem="4GB")        

# -------------------- For quick visualizations --------------------

def compare_tails(weight_path, n_weight):
    import powerlaw as plaw

    global weight_file, weight_info, weight_foldername, full_fit_path, df_full, weights
    global y_normal, y_stable

    c_ls_1 = ['green', 'red']
    lwidth = 1.5

    t0 = time.time()
    n_weight = int(n_weight)    
    pytorch = False if "_tf" in weight_path else True
    if_torch_weights = (weight_path.split("/")[-1][:3] != "np_")

# Loading weight matrix ----------------------

    # path for loading the weights
    main_path = join(root_data, "pretrained_workflow")

    weight_info = pd.read_csv(join(main_path, "weight_info.csv")) if pytorch else pd.read_csv(join(main_path, "weight_info_tf.csv"))
    weight_name = weight_info.loc[n_weight,"weight_file"]
    i, wmat_idx = int(weight_info.loc[n_weight,"idx"]), int(weight_info.loc[n_weight,"wmat_idx"])
    model_name = weight_info.loc[n_weight,"model_name"]

    print(f"{n_weight}: {weight_name}")

    print("\n")

# Powerlaw fitting ----------------------

    if if_torch_weights:
        import torch
        weights = torch.load(f"{weight_path}/{weight_name}")
        weights = weights.detach().numpy()
    else:
        weights = np.load(f"{weight_path}/{weight_name}.npy")
    w_size = len(weights)

    # values significantly smaller than zero are filtered out         
    weights = weights[np.abs(weights) >0.00001]    

    # 1. split into upper and lower tail
    weights_upper = weights[weights > weights.mean()]
    weights_lower = np.abs(weights[weights < weights.mean()])
    
    # 2. full distribution fit
    dirname = "allfit_all" if pytorch else "allfit_all_tf"
    weight_file = weight_info.loc[n_weight, 'weight_file']
    weight_foldername = replace_name(weight_file, "allfit") + ".csv"
    full_fit_path = join(main_path, dirname, model_name, weight_foldername)
    if not os.path.isfile(full_fit_path):
        dirname = "nan_allfit_all" if pytorch else "nan_allfit_all_tf"
        full_fit_path = join(main_path, dirname, model_name, weight_foldername)
    df_full = pd.read_csv( full_fit_path )

    params_stable = df_full.iloc[0,3:7]
    params_normal = df_full.iloc[0,11:13]

    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(17/3*2, 5.67*2))

    x = np.linspace(weights.min(), weights.max(), 1000)
    y_stable = levy_stable.pdf(x, *params_stable)
    y_normal = norm.pdf(x, params_normal[0], params_normal[1])
    for ncol in range(ncols):

        axs[0,ncol].hist(weights, 1500, density=True)
        axs[0,ncol].plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axs[0,ncol].plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')    
        axs[0,ncol].set_ylim(0,1)

    axs[0,0].legend()        

    axs[0,0].set_xlim(weights.min(), np.percentile(weights,1.5))
    axs[0,1].set_xlim(np.percentile(weights,98.5), weights.max())    
    
    # 3. tail fit
    """
    dirname = "plfit_all" if pytorch else "plfit_all_tf"
    weight_file = weight_info.loc[n_weight, 'weight_file']
    weight_foldername = replace_name(weight_file, "plfit") + ".csv"
    tail_fit_path = join(main_path, dirname, model_name, weight_foldername)
    if os.path.isfile(tail_fit_path):    
        df_tail = pd.read_csv( tail_fit_path )
        alpha = df_tail.loc[0,'alpha']
    """
    plaw.plot_pdf(data=np.abs(weights),ax=axs[1,0])        


    for nrow in [0]:
        for ncol in range(ncols):
            axs[nrow,ncol].set_xscale("symlog")
            axs[nrow,ncol].set_yscale("symlog")

    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])



