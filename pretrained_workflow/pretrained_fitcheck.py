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
from os.path import join, isfile
from scipy.stats import levy_stable, norm, lognorm
from scipy.stats import anderson_ksamp, ks_2samp, shapiro, distributions 
from tqdm import tqdm

from pretrained_wfit import replace_name, load_single_wmat

lib_path = os.getcwd()
sys.path.append(f'{lib_path}')

from weightwatcher.constants import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from path_names import root_data

t0 = time.time()

def test(data, dist_type, params):
    
    assert dist_type in ['levy_stable', 'normal', 'tstudent', 'lognorm']

    # fitting plus log likelihood
    if dist_type == 'levy_stable':
        r = levy_stable.rvs(*params, size=len(data))
    elif dist_type == 'normal':
        r = norm.rvs(*params, len(data))
    elif dist_type == 'tstudent':
        r = sst.t.rvs(*params, len(data))
    elif dist_type == 'lognorm':
        r = lognorm.rvs(*params, size=len(data))

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

    return ad_test, ad_siglevel, ks_stat, ks_pvalue

def check(weight_path, n_weight, with_logl, remove_weights=True):
    """
    Fit the the entire entries of a single pretrained weight matrix to one of the 4 distributions: "levy_stable", "normal", "tstudent", "lognorm".
        - weight_path (str): dir of the stored pretrained weights
            1. 
        - n_weight (int): index of the pretrained weight matrix in the folder/recorded csv file
        - with_logl (bool): whether to compute the log-likelihood
        - remove_weights (bool): whether to remove weights smaller or equal than 0.00001
    """

    global df, ad_test, ad_siglevel, ks_stat, ks_pvalue, stat_test_all, fit_exists1, params, weights

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

    weights = load_single_wmat(weight_path, weight_name, if_torch_weights)
    index_dict = {'levy_stable': [3, 7, 11], 'normal': [11, 13, 19], 'tstudent': [19, 22, 26], 'lognorm': [26, 29, 33]}
    dists_all = list(index_dict.keys())
    print(f'weight size: {len(weights)} \n')

    N_total = 20
    stat_test_all = np.full([4, N_total, len(dists_all)], np.nan)
    if fit_exists1:
        df = pd.read_csv(join(model_path1, df_name))
        print("Fitting already done!")                

        for dist_ii, dist_type in tqdm(enumerate(dists_all)):
            idxs = index_dict[dist_type]
            params = df.iloc[0,idxs[0]:idxs[1]]
            print(f'{dist_type} params: {list(params)}')
            for ii in range(N_total):
                ad_test, ad_siglevel, ks_stat, ks_pvalue = test(weights, dist_type, params)
                stat_test_all[:, ii, dist_ii] = ad_test.statistic, ad_siglevel, ks_stat, ks_pvalue

        for dist_ii, dist_type in enumerate(dists_all):
            #for type_ii in range(shape(stat_test_all.shape[1])):
            for type_ii in [1,3]:
                print(f'{dist_type}, pvalue {type_ii}: mean: {stat_test_all[type_ii,:,dist_ii].mean()} and std: {stat_test_all[type_ii,:,dist_ii].std()} \n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])                