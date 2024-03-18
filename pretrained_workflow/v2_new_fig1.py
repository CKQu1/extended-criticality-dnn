import seaborn as sns
import matplotlib.colors as mcl
import numpy as np
import time
import torch
import os
import pandas as pd
import scipy.stats as sst

from ast import literal_eval
from os.path import join, isfile, isdir
from scipy.stats import levy_stable, norm, distributions, lognorm
from string import ascii_lowercase as alphabet
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name

pub_font = {'family' : 'sans-serif'}
plt.rc('font', **pub_font)

"""
Plots the fitting results for PyTorch pretrained CNNs (image classification).
"""

t0 = time.time()

# ---------------------------

# 3 by 3 template

# colour schemes
cm_type_1 = 'coolwarm'
cm_type_2 = 'RdGy'
c_ls_1 = ['k', 'k']
c_ls_2 = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:green", "tab:brown", "tab:olive"] * 3
#c_ls_2 = ["peru", "dodgerblue", "limegreen"]
c_ls_3 = ["red", "blue"]
#c_hist_1 = "tab:blue"
#c_hist_1 = "dimgrey"
#c_hist_2 = "dimgrey"
c_hist_1 = "tab:blue"
c_hist_2 = "dimgrey"
opacity = 0.4
rwidth = 0.8

linestyle_ls = ["solid", "dashed","dashdot"]

lwidth = 18
lwidth2 = 18
axis_size_1 = 18.5 * 1.5 * 4
label_size_1 = 18.5 * 1.5 * 4
legend_size_1 = 14 * 1.5 * 4
param_sizes_1 = {'legend.fontsize': legend_size_1,
                 'axes.labelsize': label_size_1,
                 'axes.titlesize': label_size_1,
                 'xtick.labelsize': label_size_1,
                 'ytick.labelsize': label_size_1}

axis_size_2 = 18.5 * 1.5 * 8
label_size_2 = 18.5 * 1.5 * 8
legend_size_2 = 14 * 1.5 * 3
param_sizes_2 = {'legend.fontsize': legend_size_2,
                 'axes.labelsize': label_size_2,
                 'axes.titlesize': label_size_2,
                 'xtick.labelsize': label_size_2,
                 'ytick.labelsize': label_size_2}
tick_size = 30                                  

sep_str = '-'*75
# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"
grouped_stats_path = join(prepath, "grouped_stats_v2")
if not isdir(grouped_stats_path): os.makedirs(grouped_stats_path)

df_full_file = "full_dist_grouped.csv"
df_tail_file = "tail_dist_grouped.csv"
df_mmt_file = 'mmt_grouped.csv'
df_files = [df_full_file, df_tail_file, df_mmt_file]

"""
allfit_paths = [f"{prepath}/allfit_all", f"{prepath}/nan_allfit_all", 
                #f"{prepath}/noremovefit_all", f"{prepath}/nan_noremovefit_all",
                f"{prepath}/allfit_all_tf", f"{prepath}/nan_allfit_all_tf"]
"""
allfit_paths = [f"{prepath}/allfit_all"]

metric_names = ['alpha', 'sigma',    # full distribution fit  
                'shap pvalue', 
                'ks pvalue stable', 'ks pvalue normal',    # KS test
                'ks pvalue tstudent', 'ks pvalue lognorm',
                'ad sig level stable', 'ad sig level normal',
                'ad sig level tstudent', 'ad sig level lognorm',
                'nu', 'sigma_t', 'mu_t',      
                'logl_stable', 'logl_norm', 'logl_t', 'logl_lognorm',          
                'model_name', 'param_shape', 'weight_num', 'fit_size', 
                'wmat_idx', 'idx', 'weight_file',
                'fit_done', 'dirname']

ks_cols = ['ks pvalue stable', 'ks pvalue normal', 'ks pvalue tstudent', 'ks pvalue lognorm']
ad_cols = ['ad sig level stable', 'ad sig level normal', 'ad sig level tstudent', 'ad sig level lognorm']                

#'wmin', 'wmax'
metric_names_tail = ['alpha_lower', 'alpha_upper', 
                     'tpl_alpha_lower', 'tpl_alpha_upper',
                    #'total_entries', 'fit_entries', 'xmin', 'xmax', 
                    'xmin_lower', 'xmax_lower', 'xmin_upper', 'xmax_upper',
                    'entries_lower', 'entries_upper',
                    'xlogrange_lower', 'xlogrange_upper',
                    'warning_lower', 'warning_upper',
                    #'best_fit_1', 'best_fit_2',
                    #'bestfit',
                    #'bf1_lower', 'bf1_upper', 'bf2_lower', 'bf2_upper',
                    'bf1_lower', 'bf1_upper',
                    'model_name', 'param_shape', 'weight_num',                     
                    'wmat_idx', 'idx', 'weight_file',
                    'fit_done', 'dirname']

mmt_names1 = ['size_1','mean_1','std_1','skewness_1','kurtosis_1','wmin_lower_1', 'wmax_lower_1', 'wmin_upper_1', 'wmax_upper_1']
mmt_names2 = ['size_2','mean_2','std_2','skewness_2','kurtosis_2','wmin_lower_2', 'wmax_lower_2', 'wmin_upper_2', 'wmax_upper_2']
mmt_names = mmt_names1 + mmt_names2
metric_names_mmt = mmt_names + ['model_name', 'param_shape', 'weight_num',                     
                                'wmat_idx', 'idx', 'weight_file',
                                'fit_done', 'dirname']                    

replace = True  # replace even if created

param_selection_base = 'ad'  # or 'ks'
pass_ad_test = True  # if AD test is considered
pass_ks_test = False  # if 2-sided KS test is considered
pass_fit_size = True

#min_weight_num = 1000
min_weight_num = 500
max_removed_perc = 0.1

do_compute = True
for df_file in df_files:
    do_compute = do_compute and isfile(join(grouped_stats_path, df_file))
do_compute = (not do_compute)
if do_compute or replace:
    print("At least one file does not exist, creating now!")

    metrics_all = {}
    #metrics_all['sample_w'] = torch.load(w_path).detach().numpy()    

    """
    Distribution of fitted stable distribution parameters
    """

    # load two files of fitted params from Pytorch and Tensorflow
    files = []
    #allfit_paths = [f"{prepath}/allfit_all"]
    for mpath in allfit_paths:
        for f1 in os.listdir(mpath):
            f1_path = os.path.join(mpath, f1)
            for f2 in os.listdir(f1_path):
                #if 'plfit' in f2:
                if 'allfit' in f2 and '.csv' in f2:
                    plaw_data_path = os.path.join(f1_path, f2)
                    files.append(plaw_data_path)

    print(f"Total number of weight matrices and convolution tensors: {len(files)}")

    # full distribution fit
    for metric_name in metric_names:
        metrics_all[metric_name] = []                

    # tail fit
    metrics_tail = {}
    for metric_name in metric_names_tail:
        metrics_tail[metric_name] = []    

    # moments
    metrics_mmt = {}
    for metric_name in metric_names_mmt:
        metrics_mmt[metric_name] = []                                 

    # weight tensor information 
    for weight_info_name in ["weight_info.csv"]:
    #for weight_info_name in ["weight_info.csv", "weight_info_tf.csv"]:
        weight_info = pd.read_csv( join(prepath,weight_info_name) )    

        # ---------- full distribution fit ----------
        for metric_idx in range(metric_names.index('model_name'),len(metric_names)-2):
            metric_name = metric_names[metric_idx]            
            metrics_all[metric_name] += list(weight_info.loc[:,metric_name])
        # getting 'fit_done', 'dirname'
        for ii in tqdm(range(weight_info.shape[0])):
            fit_done = False
            weight_foldername = replace_name(weight_info.loc[ii,'weight_file'], "allfit") + ".csv"    
            model_name = weight_info.loc[ii,'model_name']        
            #for dirname in ["allfit_all", "nan_allfit_all", "allfit_all_tf", "nan_allfit_all_tf"]:        
            for dirname in ["allfit_all", "nan_allfit_all"]:
                fit_done = fit_done or os.path.isfile(join(prepath, dirname, model_name, weight_foldername))
                if fit_done:
                    break
    
            metrics_all['fit_done'].append(fit_done)
            if fit_done:
                metrics_all['dirname'].append(dirname)
            else:
                metrics_all['dirname'].append(np.nan)

        # ---------- tail fit ----------
        for metric_idx in range(metric_names_tail.index('model_name'),len(metric_names_tail)-2):
            metric_name = metric_names_tail[metric_idx]
            metrics_tail[metric_name] += list(weight_info.loc[:,metric_name])     
        
        for ii in tqdm(range(weight_info.shape[0])):
            fit_done = False
            weight_foldername = replace_name(weight_info.loc[ii,'weight_file'], "plfit") + ".csv"    
            model_name = weight_info.loc[ii,'model_name']        
            for dirname in ["ww_plfit_v2"]:  # newer version     
                fit_done = fit_done or os.path.isfile(join(prepath, dirname, model_name, weight_foldername))

            metrics_tail['fit_done'].append(fit_done)
            if fit_done:
                metrics_tail['dirname'].append(dirname)
            else:
                metrics_tail['dirname'].append(np.nan)

        # ---------- moments ----------
        for metric_idx in range(metric_names_mmt.index('model_name'),len(metric_names_mmt)-2):
            metric_name = metric_names_mmt[metric_idx]
            metrics_mmt[metric_name] += list(weight_info.loc[:,metric_name])     
        
        for ii in tqdm(range(weight_info.shape[0])):
            fit_done = False
            weight_foldername = replace_name(weight_info.loc[ii,'weight_file'], "mmt") + ".csv"    
            model_name = weight_info.loc[ii,'model_name']        
            #for dirname in ["moments"]:    
            for dirname in ["moments_v2"]:
                fit_done = fit_done or os.path.isfile(join(prepath, dirname, model_name, weight_foldername)) 

            metrics_mmt['fit_done'].append(fit_done)
            if fit_done:
                metrics_mmt['dirname'].append(dirname)
            else:
                metrics_mmt['dirname'].append(np.nan)                

    # important messages (currently just pytorch weights)        
    total_wmat = sum(metrics_all['fit_done'])
    print(f"{total_wmat} out of {weight_info.shape[0]} have been analyzed for the full distribution! \n")

    total_wmat_tail = sum(metrics_tail['fit_done'])
    print(f"{total_wmat_tail} out of {weight_info.shape[0]} have been analyzed for the distribution tail! \n")

    total_wmat_mmt = sum(metrics_mmt['fit_done'])
    print(f"{total_wmat_mmt} out of {weight_info.shape[0]} have been analyzed for moments computation! \n")    

    # top-1 and top-5 acc
    #net_names_all = [pd.read_csv(join(prepath,fname)) for fname in ["net_names_all.csv", "net_names_all_tf.csv"]]
    #metrics_all['top-1'], metrics_all['top-5'] = [], []

    files_failed, files_failed_tail, files_failed_mmt  = [], [], []
    
    for ii in tqdm(range(len(metrics_all['param_shape']))):        

        # -------------------- 1. FULL DIST --------------------
        weight_foldername = replace_name(metrics_all['weight_file'][ii], "allfit") + ".csv"    
        model_name = metrics_all['model_name'][ii]
        fit_done = metrics_all['fit_done'][ii]
        dirname = metrics_all['dirname'][ii]
        if fit_done:
            df = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )
            # stablefit params
            alpha, sigma = df.loc[0,'alpha'].item(), df.loc[0,'sigma'].item()
            # student t params
            nu, sigma_t, mu_t = df.loc[0,'nu'].item(), df.loc[0,'sigma_t'].item(), df.loc[0,'mu_t'].item()
            # fitting stats
            shap_stat, ks_pvalue_stable, ks_pvalue_normal, ks_pvalue_tstudent, ks_pvalue_lognorm = df.loc[0,["shap pvalue","ks pvalue stable","ks pvalue normal", \
                                                                                                             "ks pvalue tstudent", "ks pvalue lognorm"]]
            # AD test
            ad_pvalue_stable, ad_pvalue_normal, ad_pvalue_tstudent, ad_pvalue_lognorm = df.loc[0,['ad sig level stable','ad sig level normal', \
                                                                                                  'ad sig level tstudent','ad sig level lognorm']]
            # log-likelihood
            logl_stable, logl_norm, logl_t, logl_lognorm = df.loc[0,["logl_stable","logl_norm","logl_t","logl_lognorm"]]
        else:          
            alpha, sigma, nu, sigma_t, mu_t, shape_stat = [np.nan] * 6
            # KS teste
            ks_pvalue_stable, ks_pvalue_normal = [np.nan] * 2
            ks_pvalue_tstudent, ks_pvalue_lognorm = [np.nan] * 2
            # AD test
            ad_pvalue_stable, ad_pvalue_normal = [np.nan] * 2
            ad_pvalue_tstudent, ad_pvalue_lognorm = [np.nan] * 2  
            # log-likelihood          
            logl_stable, logl_norm, logl_t, logl_lognorm = [np.nan] * 4

            files_failed.append( weight_foldername )

        metrics_all['alpha'].append( alpha )
        metrics_all['sigma'].append( sigma )     
        metrics_all['nu'].append( nu )
        metrics_all['sigma_t'].append( sigma_t )    
        metrics_all['mu_t'].append( mu_t )      
        
        metrics_all['shap pvalue'].append(shap_stat)  # shapiro-wilk test    

        metrics_all['ks pvalue stable'].append(ks_pvalue_stable)  # 2-sided KS test
        metrics_all['ks pvalue normal'].append(ks_pvalue_normal)   
        metrics_all['ks pvalue lognorm'].append(ks_pvalue_lognorm)
        metrics_all['ks pvalue tstudent'].append(ks_pvalue_tstudent)      
        
        metrics_all['ad sig level stable'].append(ad_pvalue_stable)  # 2-sided AD test
        metrics_all['ad sig level normal'].append(ad_pvalue_normal)   
        metrics_all['ad sig level lognorm'].append(ad_pvalue_lognorm)
        metrics_all['ad sig level tstudent'].append(ad_pvalue_tstudent)        

        metrics_all['logl_stable'].append(logl_stable)  # LL
        metrics_all['logl_norm'].append(logl_norm)
        metrics_all['logl_t'].append(logl_t)
        metrics_all['logl_lognorm'].append(logl_lognorm)
 

        # -------------------- 2. TAIL DIST --------------------   
        weight_foldername = replace_name(metrics_tail['weight_file'][ii], "plfit") + ".csv"    
        model_name = metrics_tail['model_name'][ii]
        fit_done = metrics_tail['fit_done'][ii]
        dirname = metrics_tail['dirname'][ii]
        if fit_done:
            df_pl = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )
            # PL fit params
            alpha_lower = df_pl.loc[0,'alpha'].item(); alpha_upper = df_pl.loc[1,'alpha'].item()
            #bf1_lower = df_pl.loc[0,'best_fit_1']; bf1_upper = df_pl.loc[1,'best_fit_1']   
            
            # TPL fit params
            tpl_alpha_lower = df_pl.loc[0,'alpha1_alt'].item(); tpl_alpha_upper = df_pl.loc[1,'alpha1_alt'].item()

            # Best fit                             
            bf1_lower, bf1_upper = df_pl.loc[:,'bestfit']

            # xmin/xmax
            xmin_lower, xmin_upper = df_pl.loc[:,'xmin']
            xmax_lower, xmax_upper = df_pl.loc[:,'xmax']

            # fitted entry number
            entries_lower, entries_upper = df_pl.loc[:,'fit_entries']    

            xlogrange_lower = (np.log(xmax_lower) - np.log(xmin_lower)) / np.log(10)
            xlogrange_upper = (np.log(xmax_upper) - np.log(xmin_upper)) / np.log(10)

            # warning message
            warning_lower, warning_upper = df_pl.loc[:,'warning']
        else:          
            alpha_lower, alpha_upper = [np.nan] * 2
            #bf1_lower, bf1_upper, bf2_lower, bf2_upper = [np.nan] * 4
            bf1_lower, bf1_upper = [np.nan] * 2
            xmin_lower, xmin_upper, xmax_lower, xmax_upper = [np.nan] * 4
            warning_lower, warning_upper = [np.nan] * 2

            files_failed_tail.append( weight_foldername )

        #for prefix in ['alpha', 'bf1', 'bf2', 'xmin', 'xmax', 'xlogrange']:
        for prefix in ['alpha', 'tpl_alpha', 'bf1', 'xmin', 'xmax', 'entries', 'xlogrange', 'warning']:
            for tail_type in ['lower', 'upper']:
                metrics_tail[f'{prefix}_{tail_type}'].append( locals()[f'{prefix}_{tail_type}'] )


        # -------------------- 3. MOMENTS --------------------
        weight_foldername = replace_name(metrics_mmt['weight_file'][ii], "mmt") + ".csv"    
        model_name = metrics_mmt['model_name'][ii]
        fit_done = metrics_mmt['fit_done'][ii]
        dirname = metrics_mmt['dirname'][ii]
        if fit_done:
            df_m = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )

            size_1,mean_1,std_1,skewness_1,kurtosis_1,wmin_lower_1,wmax_lower_1,wmin_upper_1,wmax_upper_1 = df_m.loc[0,mmt_names1]   
            size_2,mean_2,std_2,skewness_2,kurtosis_2,wmin_lower_2,wmax_lower_2,wmin_upper_2,wmax_upper_2 = df_m.loc[0,mmt_names2]          

        else:
            size_1,mean_1,std_1,skewness_1,kurtosis_1 = [np.nan] * 9   
            size_2,mean_2,std_2,skewness_2,kurtosis_2 = [np.nan] * 9        

            files_failed_mmt.append( weight_foldername )

        for name_ in mmt_names:
            metrics_mmt[name_].append( locals()[name_] )
        
    # ----- saving data -----
    df_full = pd.DataFrame(data=metrics_all)
    df_full.to_csv(join(grouped_stats_path, df_full_file)) 

    df_tail = pd.DataFrame(data=metrics_tail)
    df_tail.to_csv(join(grouped_stats_path, df_tail_file)) 

    df_mmt = pd.DataFrame(data=metrics_mmt)
    df_mmt.to_csv(join(grouped_stats_path, df_mmt_file))     
    print("Summary csv saved!")

else:
    print("Files already created!")
    df_full = pd.read_csv(join(grouped_stats_path, df_full_file), index_col=0)
    df_tail = pd.read_csv(join(grouped_stats_path, df_tail_file), index_col=0)
    df_mmt = pd.read_csv(join(grouped_stats_path, df_mmt_file), index_col=0)


# -------------------- add AIC and BIC for df_full --------------------

full_dist_names = [col.split('_')[1] for col in df_full.columns if "logl_" in col]  # the columns that contain log_l
params_dict = {'stable':4, 'norm': 2, 't': 3, 'lognorm':3}  # number of params in each type of distribution
for ii, dist_name in enumerate(full_dist_names):
    # AIC
    df_full[f"aic_{dist_name}"] = 2 * params_dict[dist_name] - 2 * df_full.loc[:,f'logl_{dist_name}']

for ii, dist_name in enumerate(full_dist_names):                   
    # BIC
    #df_full[f"bic_{dist_name}"] = params_dict[dist_name] * np.log(df_full.loc[:,'weight_num']) - 2 * df_full.loc[:,f'logl_{dist_name}']
    df_full[f"bic_{dist_name}"] = params_dict[dist_name] * np.log(df_full.loc[:,'fit_size']) - 2 * df_full.loc[:,f'logl_{dist_name}']

# merged df
df_tail = df_tail.rename(columns={"fit_done": "fit_done_tail"})
df_mmt = df_mmt.rename(columns={"fit_done": "fit_done_mmt"})

df_merge = pd.concat([df_full, df_tail.iloc[:,:df_tail.columns.get_loc('model_name')], df_mmt.iloc[:,:df_mmt.columns.get_loc('model_name')]], axis=1)
df_merge = pd.concat([df_merge, df_tail.loc[:,'fit_done_tail'], df_mmt.loc[:,'fit_done_mmt']], axis=1)

# add removed entries percentage
df_merge.loc[:,'removed_perc'] = (df_merge['weight_num'] - df_merge['fit_size']) / df_merge['weight_num']

# general setting
print(f'{sep_str} \n')
print('General setting \n')
print(f'pass_ad_test = {pass_ad_test} \n')
print(f'pass_ks_test = {pass_ks_test} \n')
print(f'{sep_str} \n')

# -------------------- Weight set filter --------------------

if pass_fit_size:
    cond_1 = (df_merge.loc[:,'fit_size'] >= min_weight_num)
else:
    cond_1 = (df_merge.loc[:,'removed_perc'] <= max_removed_perc)  # removed entry cannot exceed 10%
if pass_ad_test:
    cond_2 = (df_merge.loc[:,'ad sig level stable']>=0.05)
if pass_ks_test:
    cond_2 = (df_merge.loc[:,'ks pvalue stable']>=0.05)  

# full dist
dict_best = {'ks_best':'ks pvalue', 'ad_best':'ad sig level',
             'logl_best':'logl_',
             'aic_best':'aic_','bic_best':'bic_'}

# if the row contains nan, set value to nan  
criteria_cond = cond_1
if pass_ad_test:
    pass_cond = df_merge[ks_cols].isna().any(axis=1)
    criteria_cond = cond_1 & pass_cond
if pass_ad_test:            
    pass_cond = df_merge[ad_cols].isna().any(axis=1)
    criteria_cond = cond_1 & pass_cond
nan_idxs = df_merge[criteria_cond].index    
# deal with the remaining index
nonan_idxs = df_merge.index.difference(nan_idxs) 

for metric_best in dict_best.keys():
    for dist_name in full_dist_names:

        cols = [col for col in df_merge.columns if dict_best[metric_best] in col]      
        df_merge.loc[:,metric_best] = np.nan

        if 'aic' in metric_best or 'bic' in metric_best:
            df_merge.loc[nonan_idxs,metric_best] = df_merge.loc[nonan_idxs,cols].idxmin(axis=1)
        else:
            df_merge.loc[nonan_idxs,metric_best] = df_merge.loc[nonan_idxs,cols].idxmax(axis=1)

# -------------------- (a) Featured CNN convolution tensor --------------------

# full distribution best fitted by stable dist and tails best fitted by tpl
# df_pl_tpl_stable = df_merge[(df_merge.loc[:,'bf1_upper']=='truncated_power_law') \
#                      & (df_merge.loc[:,'ks pvalue stable']>=0.05)
#                      & (df_merge.loc[:,'alpha_upper']<=2)].reset_index()

# condition for Fig. 1(a)
cond_a = cond_1 & cond_2 & ((df_merge.loc[:,'bf1_upper'] == 'truncated_power_law') | (df_merge.loc[:,'bf1_upper'] == 'power_law'))
passed_idxs = df_merge[cond_a].index     
df_pl_tpl_stable = df_merge.loc[passed_idxs,:]

# find indices where alpha matchese to PL/TPL fit
match_idxs = df_pl_tpl_stable.index[np.abs(df_pl_tpl_stable.loc[:,'alpha'] + 1 - df_pl_tpl_stable.loc[:,'alpha_upper']) < 0.2]

# -------------------------------------- Comment out ------------------------------------------

"""
print(f'{sep_str} \n')
print(f'Check tail \n')

for miidx, match_idx in enumerate(match_idxs):
    # featured weight matrix
    l = df_pl_tpl_stable.loc[match_idx,'idx']
    wmat_idx = df_pl_tpl_stable.loc[match_idx,'wmat_idx']
    net = df_pl_tpl_stable.loc[match_idx,'model_name']

    w_type = f"{net}_layer_{l}_{wmat_idx}"
    wmat_file = f"{net}_allfit_{l}_{wmat_idx}"
    #w_path = f"{prepath}/weights_all/{w_type}"
    w_path = join(prepath, 'np_weights_all', f'{w_type}.npy')

    for allfit_path_dir in allfit_paths:
        allfit_path = f"{allfit_path_dir}/{net}/{wmat_file}.csv"
        print(allfit_path + "\n")
        if isfile(allfit_path):        
            break

    # print(f'{sep_str} \n')
    # print(f'Featured networks: {net} \n')
    # print(f'{wmat_file} \n')
    # print(f"Featured weight matrix: {net}_{l}_{wmat_idx} \n")
    # print(f'{sep_str} \n')

    # tail fit paths
    wmat_file_tail = f"{net}_plfit_{l}_{wmat_idx}"
    tailfit_path = f"{prepath}/ww_plfit_all/{net}/{wmat_file_tail}.csv"


    fig = plt.figure(figsize=(26, 21))
    axis = plt.subplot(111)

    # distribution tail
    df_tailfit = pd.read_csv(tailfit_path)
    layer_idx, wmat_idx, total_entries, fit_entries, \
    k, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, \
    best_fit_1, best_fit_2, Rs_all, ps_all = df_tailfit.iloc[1,:].values

    print(f"Plaw to upper tail: k = {k}, xmin = {xmin}, xmax = {xmax} \n")

    sample_w = np.load(w_path)
    min_evals_to_plot = (xmin/100)
    evals = np.abs(sample_w[sample_w>=0])
    evals_to_plot = evals[evals>min_evals_to_plot]

    num_bins = 250
    counts, binEdges = np.histogram(evals_to_plot, bins=num_bins, density=True)     
    
    sns.distplot(evals_to_plot, hist=False, color='b', ax=axis, kde_kws={'linewidth':lwidth2})                         

    # --- eye guide 1 --- (from PL)
    binEdges = binEdges[1:]
    xc = binEdges[binEdges>=xmin].min()
    yc = counts[binEdges>=xmin].max()
    #b = yc + k * xc
    #b = (np.log(yc) + k * np.log(xc))/np.log(10)  # original y-intercept
    b = (np.log(yc) + k * np.log(xc))/np.log(10) + 1  # lift higher
    #xs = np.linspace(np.exp(xmin), np.exp(xmax), 500)
    #xs = np.linspace(xmin, xmax, 500)  # original range
    #xs = np.linspace(xmin - (xmax - xmin)/4, xmax + (xmax - xmin)/4, 500)  # make longer
    xs = np.linspace(xmin/10, xmax * 10, 500)  # even longer
    #ys = -k * xs + b
    #axs[0,0].plot(np.exp(xs), np.exp(ys), c='g')
    ys = 10**b * xs**(-k)
    axis.plot(xs, ys, c='g', linestyle='dotted', label=rf'PL fit ($k$ = {round(k,2)})')

    axis.axvline(xmin, color='r', linestyle='dashed')  # label=rf'$x_{{\min}}$ = {round(xmin,2)}'

    axis.set_xlim(evals_to_plot.min(), evals_to_plot.max())        
    axis.set_xscale('log'); axis.set_yscale('log')      

    fig_file = f'_{match_idx}_net={net}_l={l}.pdf'

    fig1_path = "/project/PDLAI/project2_data/figure_ms/pretrained_fitting"
    if not isdir(fig1_path): os.makedirs(fig1_path)
    #axis.subplots_adjust(hspace=0.2)
    plt.tight_layout()    
    plt.savefig(join(fig1_path, fig_file), bbox_inches='tight')        

print(f'{sep_str} \n')
"""

# ------------------------------------- Comment out -------------------------------------------

#l, wmat_idx = 96, 260
#net = 'efficientnet_v2_m'

# featured weight matrix
# feat_i = 6053  # original
# l = df_merge.loc[feat_i,'idx']
# wmat_idx = df_merge.loc[feat_i,'wmat_idx']
# net = df_merge.loc[feat_i,'model_name']

# best
# feat_i = 149

feat_i = match_idxs[2]  # best
l = df_pl_tpl_stable.loc[feat_i,'idx']
wmat_idx = df_pl_tpl_stable.loc[feat_i,'wmat_idx']
net = df_pl_tpl_stable.loc[feat_i,'model_name']

w_type = f"{net}_layer_{l}_{wmat_idx}"
wmat_file = f"{net}_allfit_{l}_{wmat_idx}"
#w_path = f"{prepath}/weights_all/{w_type}"
w_path = join(prepath, 'np_weights_all', f'{w_type}.npy')

for allfit_path_dir in allfit_paths:
    allfit_path = f"{allfit_path_dir}/{net}/{wmat_file}.csv"
    print(allfit_path + "\n")
    if isfile(allfit_path):        
        break

print(f'{sep_str} \n')
print(f'Featured networks: {net} \n')
print(f'{wmat_file} \n')
print(f"Featured weight matrix: {net}_{l}_{wmat_idx} \n")
print(f'{sep_str} \n')

# tail fit paths
wmat_file_tail = f"{net}_plfit_{l}_{wmat_idx}"
tailfit_path = f"{prepath}/ww_plfit_all/{net}/{wmat_file_tail}.csv"

params_stable = pd.read_csv(allfit_path).iloc[0,3:7]
params_normal = pd.read_csv(allfit_path).iloc[0,11:13]
params_tstudent = pd.read_csv(allfit_path).iloc[0,19:22]
params_lognorm = pd.read_csv(allfit_path).iloc[0,26:29]
#sample_w = torch.load(w_path).detach().numpy()
sample_w = np.load(w_path)

# -------------------- Plot --------------------
print("Start plotting")

plotted_metrics = {}
#bin_ls = [1000,50,2500]
#bin_ls = [1000] * 6
#bin_ls = [1000, 75, 500, 1000, 1000, 250]
bin_ls = [250] * 6
bin_ls[1] = 50
bin_ls[2] = 50
bin_ls[3] = 500
bin_ls[4] = 50

# xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})"] \
#             + [r"$\alpha$ (Stable)", r"$\nu$ (Student t)"] \
#             + [r"$\mathbf{{W}}^{{{}}}$ upper tail".format(l + 1), r"$k$ (TPL)", r"Range (TPL)"]     

xlabel_ls = ["Weights"] \
            + [r"$\alpha$ (Stable)", r"$\nu$ (Student t)"] \
            + ["Upper tail", r"$k$ (TPL)", r"Range (TPL)"]                        
            
ylabel_ls = ["Probability density", "Probability density", "Probability density", 
             "Probability density", "Probability density", "Probability density"]

# metric_names_plot = ["sample_w", "alpha", "nu", "sample_w", "alpha_tail_pl", "xlogrange_pl"]
# xlim_ls = [[-0.3, 0.3], [0.5,2], [0,3e-3], [1e-4,500], [0,15], [0,5]]
# ylim_ls = [[0,12.5], [0,5], [0,1267035], [1e-2, 1], [0,0.3], [0,2]]

metric_names_plot = ["sample_w", "sample_w_tail", "alpha", "alpha_tail_tpl", 
                     "kurtosis_1", "kurtosis_2", 
                     "skewness_1", "skewness_2",
                     "std_1", "std_2",
                     "mean_1", "mean_2",
                     "alpha_tail_pl", "nu", "xlogrange"]
xlim_ls = [[-0.25, 0.25], [1e-2,1], [0.5,2.1], [0.5,8], 
           [-3, 30], [-3, 30], 
           [-5,5], [-5,5],
           [-5,5], [-5,5],
           [-5,5], [-5,5],
           [0,10], [0,18], [0,4]
           ]

ylim_ls = [[0,25], [1e-1, 50], [0,2], [0,0.4], 
           [0,0.15], [0,0.15], 
           [0,5], [0,5],
           [0,5], [0,5],
           [0,5], [0,5],
           [0,1], [0,0.25], [0,1.75]]

metric_labels = []
metrics_all_plot = {}
top_n = 'top-5'     # choose Top-1 or Top-5 accuracy

color_ii = 0
for i, metric_name in enumerate(metric_names_plot):

    fig = plt.figure(figsize=(24, 21))
    axis = plt.subplot(111)
    # set up individual figures
    if i != 1:
        #plt.rcParams.update(param_sizes_2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)             
    else:
        #plt.rcParams.update(param_sizes_1)    
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # full distribution
    if metric_name in ['alpha','nu']:
        # filter
        best_ks = "ks pvalue stable" if metric_name == 'alpha' else "ks pvalue tstudent"
        best_ad = "ad sig level stable" if metric_name == 'alpha' else "ad sig level tstudent"       
        # if param_selection_base == 'ad':
        #     cond = cond & (df_merge.loc[:,'ad_best'] == best_ad)
        # elif param_selection_base == 'ks':
        #     cond = cond & (df_merge.loc[:,'ks_best'] == best_ks)            
        cond = cond_1 & cond_2
        selected_idxs = df_merge[cond].index

        metric = df_merge.loc[selected_idxs,metric_name]
        #print(f'plotted {metric_name}: {len(metric)} \n')

        if metric_name == 'nu':
            metric = metric[metric < 18]  # remove outliers

        metrics_all_plot[metric_name] = metric
    
    elif metric_name in ['alpha_tail_tpl', 'alpha_tail_pl']:        

        best_tailfit = "truncated_power_law" if metric_name == 'alpha_tail_tpl' else 'power_law'        

        # filter
        cond = cond_1 & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        if best_tailfit == 'power_law':
            alpha_lower = df_merge.loc[selected_idxs,'alpha_lower']
        else:
            alpha_lower = df_merge.loc[selected_idxs,'tpl_alpha_lower']

        cond = cond_1 & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        selected_idxs = df_merge[cond].index   
        if best_tailfit == 'power_law':     
            alpha_upper = df_merge.loc[cond,'alpha_upper']
        else:
            alpha_upper = df_merge.loc[selected_idxs,'tpl_alpha_upper']            

        metric = pd.concat([alpha_lower, alpha_upper])

        metrics_all_plot[metric_name] = metric

    elif metric_name in ['xlogrange']:

        #best_tailfit = "truncated_power_law"
        best_tailfit = 'power_law'

        # filter
        cond = cond_1 & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        xlogrange_lower = df_merge.loc[selected_idxs,'xlogrange_lower']

        cond = cond_1 & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        xlogrange_upper = df_merge.loc[cond,'xlogrange_upper']

        metric = pd.concat([xlogrange_lower, xlogrange_upper])       

        metrics_all_plot[metric_name] = metric 

    elif metric_name in mmt_names:
        cond = cond_1
        selected_idxs = df_merge[cond].index 
        metric = df_merge.loc[cond, metric_name]

        metrics_all_plot[metric_name] = metric

    # plotting the histogram
    if i == 0:
        axis.hist(sample_w[np.abs(sample_w) > 1e-5], bin_ls[i], density=True, 
                  color=c_ls_2[color_ii], rwidth=rwidth)       
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, *params_normal)
        #y_tstudent = sst.t.pdf(x, *params_tstudent)
        #y_lognorm = lognorm.pdf(x, *params_lognorm)
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], alpha=opacity, linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], alpha=opacity, linestyle='dotted', label = 'Stable')
        #axis.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[2], linestyle='dashdot', label = 'Student t')
        #axis.plot(x, y_lognorm, linewidth=lwidth, c=c_ls_1[3], linestyle='dotted', label = 'Log-normal')  

        #xlim_ls[0][0], xlim_ls[0][1] = sample_w[np.abs(sample_w) > 1e-5].min(), sample_w[np.abs(sample_w) > 1e-5].max()
        #ylim_ls[0][1] = max(y_stable.max(), y_normal.max()) + 2

        #color_ii += 1        

    elif i == 1 and isfile(tailfit_path):
        # distribution tail
        df_tailfit = pd.read_csv(tailfit_path)
        layer_idx, wmat_idx, total_entries, fit_entries, \
        k, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, \
        best_fit_1, best_fit_2, Rs_all, ps_all = df_tailfit.iloc[1,:].values

        print(f"Plaw to upper tail: k = {k}, xmin = {xmin}, xmax = {xmax} \n")

        # ----- method 1 -----
        """
        # directly do a power-law fit
        from weightwatcher.WW_powerlaw import fit_powerlaw
        plfit = fit_powerlaw(sample_w[sample_w>=0], xmin=xmin, total_is=None,
                             plot=False, savefig=False)
        plfit.plot_ccdf(ax=axis, linewidth=3, label='Empirical Data')
        plfit.power_law.plot_ccdf(ax=axis, color='r', linestyle='--', label='Power law fit')
        plfit.lognormal.plot_ccdf(ax=axis, color='g', linestyle='--', label='Lognormal fit')
        plfit.exponential.plot_ccdf(ax=axis, color='b', linestyle='--', label='Exponential')
        plfit.truncated_power_law.plot_ccdf(ax=axis, color='c', linestyle='--', label='Truncated powerlaw')    
        """

        # ----- method 2 -----
        """
        import powerlaw as plaw
        plfit = plaw.Power_Law(sample_w[sample_w>=0], xmin=xmin, xmax=xmax, verbose=False)      

        plfit.plot_ccdf(ax=axis, linewidth=3, label='Empirical Data')
        plfit.power_law.plot_ccdf(ax=axis, color='r', linestyle='--', label='Power law fit')
        plfit.lognormal.plot_ccdf(ax=axis, color='g', linestyle='--', label='Lognormal fit')
        plfit.exponential.plot_ccdf(ax=axis, color='b', linestyle='--', label='Exponential')
        plfit.truncated_power_law.plot_ccdf(ax=axis, color='c', linestyle='--', label='Truncated powerlaw')
        """

        # ----- method 3 -----
        # upper tail
        min_evals_to_plot = (xmin/100)
        evals = np.abs(sample_w[sample_w>=0])
        evals_to_plot = evals[evals>min_evals_to_plot]

        # adapted from weightwatcher/WW_powerlaw.py (fit_powerlaw()) 
        # from weightwatcher.RMT_Util import ax_plot_loghist       
        # num_bins = 500
        # counts, binEdges = ax_plot_loghist(axis, evals_to_plot, bins=num_bins, xmin=None)  

        num_bins = 50
        counts, binEdges = np.histogram(evals_to_plot, bins=num_bins, density=True)
        #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))        
        
        # ----- method 4 -----
        sns.distplot(evals_to_plot, hist=False, color=c_ls_2[color_ii], ax=axis, kde_kws={'linewidth':lwidth2})        
        color_ii += 1                  

        """
        counts, binEdges = np.histogram(evals_to_plot, bins=num_bins, density=True)
        logbins = np.logspace(np.log10(binEdges[0]),np.log10(binEdges[-1]),len(binEdges))
        axis.hist(x, bins=logbins, density=True)
        axis.axvline(xmin, color='r', label=r"$x_{{{}}}$".format("min"))
        """

        # --- eye guide 1 --- (from PL)
        binEdges = binEdges[1:]
        xc = binEdges[binEdges>=xmin].min()
        yc = counts[binEdges>=xmin].max()
        #b = yc + k * xc
        #b = (np.log(yc) + k * np.log(xc))/np.log(10)  # original y-intercept
        b = (np.log(yc) + k * np.log(xc))/np.log(10) + 1  # lift higher
        #xs = np.linspace(np.exp(xmin), np.exp(xmax), 500)
        #xs = np.linspace(xmin, xmax, 500)  # original range
        #xs = np.linspace(xmin - (xmax - xmin)/4, xmax + (xmax - xmin)/4, 500)  # make longer
        xs = np.linspace(xmin/10, xmax * 10, 500)  # even longer
        #ys = -k * xs + b
        #axs[0,0].plot(np.exp(xs), np.exp(ys), c='g')
        ys = 10**b * xs**(-k)
        axis.plot(xs, ys, c=c_ls_1[0], linestyle='dotted', label=rf'PL fit ($k$ = {round(k,2)})')

        #xlim_ls[i] = [xmin/10, xmax*10]
        #xlim_ls[i] = [0.01,10]
        xlim_ls[i] = [0.01,1]
        #ylim_ls[i] = [1e-2, 50]

        axis.tick_params(axis='both', which='major', length=5, width=1, size=label_size_2)
        axis.set_xticks([0.1,1])
        axis.set_xticklabels([0.1,1])
        axis.set_xticklabels([])

        # --- eye guide 2 --- (from stable)   
        # k_stable = params_stable[0] + 1               
        # b_stable = (np.log(yc) + k_stable * np.log(xc))/np.log(10) + 1
        # ys_stable = 10**b_stable * xs**(-k_stable)
        # axis.plot(xs, ys_stable, c='b', linestyle='dotted', label=rf'Stable fit ($k$ = {round(params_stable[0] + 1,2)})')

        # mark xmin
        #axis.plot([], [], c='r', linestyle='dashed', label=rf'$x_{{\min}}$ = {round(xmin,2)}') 
        #axis.axvline(xmin, color='r', linestyle='dashed')  # label=rf'$x_{{\min}}$ = {round(xmin,2)}'

        # kde
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,500) 
        #axis.plot(x, density(x))

        axis.set_xlim(evals_to_plot.min(), evals_to_plot.max())        
        axis.set_xscale('log'); axis.set_yscale('log')             

        #axis.minorticks_off()
        #axis.tick_params(left=False, right=False , labelleft=False,
        #                 labelbottom=False, bottom=False)       

    elif i in list(range(2, len(metric_names_plot))):

        if metric_name in ["alpha_tail_pl", "alpha", "xlogrange", "skewness_1", "skewness_2", "std_1", "std_2", "mean_1", "mean_2"]:
            sns.distplot(metric, hist=False, color=c_ls_2[color_ii], ax=axis, kde_kws={'linewidth':lwidth2})  # 'linestyle':'--', 
            #sns.barplot(metric, color=c_ls_2[color_ii], ax=axis)
            axis.hist(metric, density=True, 
                      color=c_ls_2[color_ii], alpha=opacity,
                      rwidth=rwidth)
        else:
            # remove outliers
            sns.distplot(metric[metric < np.percentile(metric, 95)], hist=False, color=c_ls_2[color_ii], ax=axis, kde_kws={'linewidth':lwidth2})
            #sns.barplot(metric[metric < np.percentile(metric, 95)], color=c_ls_2[color_ii], ax=axis)
            axis.hist(metric[metric < np.percentile(metric, 95)], density=True, 
                     color=c_ls_2[color_ii], alpha=opacity,
                     rwidth=rwidth)
        #bar = axis.containers[0]
        #bar.set_alpha(0.5)
        

        color_ii += 1
           
    # labels and legend
    if i == 0:
        axis.legend(loc = 'upper left', fontsize = legend_size_1, frameon=False)

    if i > 1:
        plotted_metrics[metric_name] = metric
        print(f"{metric_name} size: {len(metric)} \n")
        if i == 1:
            print(print(f"{metric_name} smaller than 1.9: {len(metric[metric <= 1.9])} \n"))

    axis.set_xlim(xlim_ls[i])
    axis.set_ylim(ylim_ls[i])    

    # tick labels
    if i != 1:
        axis.tick_params(axis='x', labelsize=axis_size_1)
        axis.tick_params(axis='y', labelsize=axis_size_1)
    # else:
    #     axis.tick_params(axis='x', labelsize=axis_size_2)
    #     axis.tick_params(axis='y', labelsize=axis_size_2)  
    #     #axis.set_yticklabels([])      

    # minor ticks
    #axis.xaxis.set_minor_locator(AutoMinorLocator())
    #axis.yaxis.set_minor_locator(AutoMinorLocator())

    #axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)

    #if i == 0 or i == 1:
    #axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
    #axis.set_title(f"{ylabel_ls[i]}", fontsize=axis_size)

    # -------------------- Save fig --------------------
    if i == 0:
        fig_file = f'fig1_{alphabet[i]}_net={net}_l={l}.pdf'
    elif i == 1:
        fig_file = 'fig1_a_inset.pdf'
    else:
        fig_file = f'fig1_{alphabet[i-1]}.pdf'
        metric_labels.append((alphabet[i-1], metric_names_plot[i]))

    fig1_path = "/project/PDLAI/project2_data/figure_ms/pretrained_fitting"
    if not isdir(fig1_path): os.makedirs(fig1_path)
    #axis.subplots_adjust(hspace=0.2)
    plt.tight_layout()    
    plt.savefig(join(fig1_path, fig_file), bbox_inches='tight')    

print(f'metric_labels: {metric_labels} \n')

print(f"{sep_str} \n")

# -------------------- Statistics summary --------------------

print(f"Total number of weight matrices: {df_merge.shape[0]} \n")

# -------------------- Full dist --------------------

print("Full fit \n")

pval_thresh = 0.05
cond = (df_merge.loc[:,'fit_done'] == True) & cond_1
total_legit_wmat = df_merge[cond].shape[0]

if pass_fit_size:
    print(f"Total entries greater than {1e-5} and with fit_size at least {min_weight_num}: {total_legit_wmat} \n")
else:
    print(f"Proportion of entries greater than {1e-5} removed at most {max_removed_perc*100}%: {total_legit_wmat} \n")

# AD p-value > 0.05
cond_ad = ((df_merge.loc[:,'ad sig level stable'] >= pval_thresh) | (df_merge.loc[:,'ad sig level normal'] >= pval_thresh) | \
           (df_merge.loc[:,'ad sig level tstudent'] >= pval_thresh) | (df_merge.loc[:,'ad sig level lognorm'] >= pval_thresh))
# KS p-value > 0.05
cond_ks = ((df_merge.loc[:,'ks pvalue stable'] >= pval_thresh) | (df_merge.loc[:,'ks pvalue normal'] >= pval_thresh) | \
            (df_merge.loc[:,'ks pvalue tstudent'] >= pval_thresh) | (df_merge.loc[:,'ks pvalue lognorm'] >= pval_thresh))
if pass_ad_test:
    cond = cond & cond_ad            
if pass_ks_test:
    cond = cond & cond_ks

full_others_best = total_legit_wmat - df_merge[cond].shape[0]

print(f"Wmats best fit by others: {full_others_best} \n")

criterion = ['ks', 'ad', 'logl', 'aic', 'bic']
dist_names1 = [col_name.split(" ")[-1] for col_name in df_merge.columns if 'ks pvalue' in col_name]
dist_names2 = [col_name.split("_")[-1] for col_name in df_merge.columns if 'logl_' in col_name]
for crit in criterion:
    if crit == 'ks' or crit == 'ad':
        dist_names = dist_names1
    else:
        dist_names = dist_names2

    print(f"criteria {crit}:")
    s1 = ''
    for dist_name in dist_names:
        if crit == 'ks':
            amount = (df_merge.loc[cond,f'{crit}_best'] == f'ks pvalue {dist_name}').sum()
        elif crit == 'ad':
            amount = (df_merge.loc[cond,f'{crit}_best'] == f'ad sig level {dist_name}').sum()
        else:
            amount = (df_merge.loc[cond,f'{crit}_best'] == f'{crit}_{dist_name}').sum()
        s1 += '   ' + f'{dist_name}: {amount}'
    print(s1 + "\n")

print(f"{sep_str} \n")
print("Tail fit \n")

# -------------------- Tail dist --------------------

cond = (df_merge.loc[:,'fit_done_tail'] == True) & cond_1
total_legit_wmat = df_merge[cond].shape[0]
if pass_fit_size:
    print(f"Total wmat that with most entries greater than {1e-5} and with entries at least {min_weight_num}: {total_legit_wmat} \n")
else:
    print(f"Proportion of entries greater than {1e-5} removed at most {max_removed_perc*100}%: {total_legit_wmat} \n")

#criterion_pl = ['bf1', 'bf2']
criterion_pl = ['bf1']
tail_types = ['lower', 'upper']
dist_names_tail = df_merge.loc[:,'bf1_upper'].unique()
for crit in criterion_pl:
    for tail_type in tail_types:
        print(f"criteria {crit} for {tail_type} tail: \n")
        s1 = ''
        for dist_name in dist_names_tail:
            amount = (df_merge.loc[cond,f'{crit}_{tail_type}'] == dist_name).sum()
            s1 += '   ' + f'{dist_name}: {amount}'
        print(s1 + "\n")        


print(f"Time: {time.time() - t0}")