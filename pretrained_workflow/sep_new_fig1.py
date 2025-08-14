import argparse
#import seaborn as sns
import scipy.io as sio
import math
import matplotlib as mpl
import matplotlib.colors as mcl
import numpy as np
import time
import torch
import os
import pandas as pd
import scipy.stats as sst
import string

from ast import literal_eval
from os.path import join, isfile, isdir
from scipy.stats import levy_stable, norm, distributions, lognorm
from string import ascii_lowercase as alphabet
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
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
c_ls_2 = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]
#c_ls_2 = ["peru", "dodgerblue", "limegreen"]
c_ls_3 = ["red", "blue"]
#c_hist_1 = "tab:blue"
#c_hist_1 = "dimgrey"
#c_hist_2 = "dimgrey"
c_hist_1 = "tab:blue"
c_hist_2 = "dimgrey"

label_ls = [f"({letter})" for letter in list(string.ascii_lowercase)]
linestyle_ls = ["solid", "dashed","dashdot"]

lwidth = 4.2
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

# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"
grouped_stats_path = join(prepath, "grouped_stats")
if not isdir(grouped_stats_path): os.makedirs(grouped_stats_path)

df_full_file = "full_dist_grouped.csv"
df_tail_file = "tail_dist_grouped.csv"

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
                #'wmin', 'wmax',
                'wmat_idx', 'idx', 'weight_file',
                'fit_done', 'dirname']

metric_names_tail = ['alpha_lower', 'alpha_upper', 
                    #'total_entries', 'fit_entries', 'xmin', 'xmax', 
                    'xmin_lower', 'xmax_lower', 'xmin_upper', 'xmax_upper',
                    'xlogrange_lower', 'xlogrange_upper',
                    #'best_fit_1', 'best_fit_2',
                    'bf1_lower', 'bf1_upper', 'bf2_lower', 'bf2_upper',
                    'model_name', 'param_shape', 'weight_num', 'fit_size', 
                    #'wmin', 'wmax',
                    'wmat_idx', 'idx', 'weight_file',
                    'fit_done', 'dirname']

param_selection_base = 'ad'  # or 'ks'
pass_ad_test = True  # if AD test is considered
pass_ks_test = False  # if 2-sided KS test is considered
replace = False      # replace even if created
#min_weight_num = 1000
min_weight_num = 300
max_removed_perc = 0.1

if not (isfile(join(grouped_stats_path, df_full_file)) and isfile(join(grouped_stats_path, df_tail_file))) or replace:
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

    # load stable fit params, and weight tensor size and dimension
    # sigma_scaled corresponds to D_w^(1/alpha)
    # full distribution fit
    for metric_name in metric_names:
        metrics_all[metric_name] = []                

    # tail fit
    metrics_tail = {}
    for metric_name in metric_names_tail:
        metrics_tail[metric_name] = []                        

    # weight tensor information 
    for weight_info_name in ["weight_info.csv"]:
    #for weight_info_name in ["weight_info.csv", "weight_info_tf.csv"]:
        weight_info = pd.read_csv( join(prepath,weight_info_name) )    

        # full distribution fit
        for metric_idx in range(metric_names.index('model_name'),len(metric_names)-2):
            metric_name = metric_names[metric_idx]            
            metrics_all[metric_name] += list(weight_info.loc[:,metric_name])
        # this is for getting 'fit_done', 'dirname'
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

        # tail fit
        #for metric_idx in range(metric_names_tail.index('model_name'),len(metric_names_tail)):
        
        for metric_idx in range(metric_names_tail.index('model_name'),len(metric_names_tail)-2):
            metric_name = metric_names_tail[metric_idx]
            metrics_tail[metric_name] += list(weight_info.loc[:,metric_name])     
        
        for ii in tqdm(range(weight_info.shape[0])):
            fit_done = False
            weight_foldername = replace_name(weight_info.loc[ii,'weight_file'], "plfit") + ".csv"    
            model_name = weight_info.loc[ii,'model_name']        
            for dirname in ["ww_plfit_all"]:        
                fit_done = fit_done or os.path.isfile(join(prepath, dirname, model_name, weight_foldername))
                if fit_done:
                    break  

            metrics_tail['fit_done'].append(fit_done)
            if fit_done:
                metrics_tail['dirname'].append(dirname)
            else:
                metrics_tail['dirname'].append(np.nan)

    # important messages (currently just pytorch weights)        
    total_wmat = sum(metrics_all['fit_done'])
    print(f"{total_wmat} out of {weight_info.shape[0]} have been analyzed for the full distribution! \n")

    total_wmat_tail = sum(metrics_tail['fit_done'])
    print(f"{total_wmat_tail} out of {weight_info.shape[0]} have been analyzed for the distribution tail! \n")

    # top-1 and top-5 acc
    #net_names_all = [pd.read_csv(join(prepath,fname)) for fname in ["net_names_all.csv", "net_names_all_tf.csv"]]
    #metrics_all['top-1'], metrics_all['top-5'] = [], []

    files_failed = []
    # load stablefit alpha and sigma
    for ii in tqdm(range(len(metrics_all['param_shape']))):
        # ---------- full dist ----------
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

        # shapiro-wilk test
        metrics_all['shap pvalue'].append(shap_stat)    
        # 2-sided KS test
        metrics_all['ks pvalue stable'].append(ks_pvalue_stable)
        metrics_all['ks pvalue normal'].append(ks_pvalue_normal)   
        metrics_all['ks pvalue lognorm'].append(ks_pvalue_lognorm)
        metrics_all['ks pvalue tstudent'].append(ks_pvalue_tstudent)      
        # AD test
        metrics_all['ad sig level stable'].append(ad_pvalue_stable)
        metrics_all['ad sig level normal'].append(ad_pvalue_normal)   
        metrics_all['ad sig level lognorm'].append(ad_pvalue_lognorm)
        metrics_all['ad sig level tstudent'].append(ad_pvalue_tstudent)        

        metrics_all['logl_stable'].append(logl_stable)
        metrics_all['logl_norm'].append(logl_norm)
        metrics_all['logl_t'].append(logl_t)
        metrics_all['logl_lognorm'].append(logl_lognorm)

        # ---------- dist tail ----------
        weight_foldername = replace_name(metrics_tail['weight_file'][ii], "plfit") + ".csv"    
        model_name = metrics_tail['model_name'][ii]
        fit_done = metrics_tail['fit_done'][ii]
        dirname = metrics_tail['dirname'][ii]
        if fit_done:
            df_pl = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )
            # stablefit params
            alpha_lower = df_pl.loc[0,'alpha'].item(); alpha_upper = df_pl.loc[1,'alpha'].item()
            bf1_lower = df_pl.loc[0,'best_fit_1']; bf1_upper = df_pl.loc[1,'best_fit_1']            
            bf2_lower = df_pl.loc[0,'best_fit_2']; bf2_upper = df_pl.loc[1,'best_fit_2']                   
                         
            xmin_lower, xmin_upper = df_pl.loc[:,'xmin']
            xmax_lower, xmax_upper = df_pl.loc[:,'xmax']

            xlogrange_lower = (np.log(xmax_lower) - np.log(xmin_lower)) / np.log(10)
            xlogrange_upper = (np.log(xmax_upper) - np.log(xmin_upper)) / np.log(10)
        else:          
            alpha_lower, alpha_upper = [np.nan] * 2
            bf1_lower, bf1_upper, bf2_lower, bf2_upper = [np.nan] * 4
            xmin_lower, xmin_upper, xmax_lower, xmax_upper = [np.nan] * 4

            files_failed.append( weight_foldername )

        for prefix in ['alpha', 'bf1', 'bf2', 'xmin', 'xmax', 'xlogrange']:
            for tail_type in ['lower', 'upper']:
                metrics_tail[f'{prefix}_{tail_type}'].append( locals()[f'{prefix}_{tail_type}'] )


    df_full = pd.DataFrame(data=metrics_all)
    df_full.to_csv(join(grouped_stats_path, df_full_file)) 

    df_tail = pd.DataFrame(data=metrics_tail)
    df_tail.to_csv(join(grouped_stats_path, df_tail_file)) 
    print("Summary csv saved!")

else:
    print("Files already created!")
    df_full = pd.read_csv(join(grouped_stats_path, df_full_file), index_col=0)
    df_tail = pd.read_csv(join(grouped_stats_path, df_tail_file), index_col=0)


# -------------------- remove non-existent files for df_full and df_tail (delete later) --------------------



# ---------------------------------------------------------------------


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
df_merge = pd.concat([df_full, df_tail.iloc[:,:df_tail.columns.get_loc('model_name')]], axis=1)
df_merge = pd.concat([df_merge, df_tail.loc[:,'fit_done_tail']], axis=1)

# add removed entries percentage
df_merge.loc[:,'removed_perc'] = (df_merge['weight_num'] - df_merge['fit_size']) / df_merge['weight_num']

# full dist
dict_best = {'ks_best':'ks pvalue', 'ad_best':'ad sig level',
             'logl_best':'logl_',
             'aic_best':'aic_','bic_best':'bic_'}
for metric_best in dict_best.keys():
    for dist_name in full_dist_names:
        # idxmax(axis=1), agg(lambda x : x.idxmax()), apply(lambda x: x.max())

        # if any of the distributions yield 

        cols = [col for col in df_merge.columns if dict_best[metric_best] in col]      

        df_merge.loc[:,metric_best] = np.nan
        # if the row contains nan, set value to nan  
        nan_idxs = df_merge[df_merge[['ks pvalue stable', 'ks pvalue normal']].isna().any(axis=1)].index        

        # deal with the remaining index
        nonan_idxs = df_merge.index.difference(nan_idxs)
        if 'aic' in metric_best or 'bic' in metric_best:
            df_merge.loc[nonan_idxs,metric_best] = df_merge.loc[nonan_idxs,cols].idxmin(axis=1)
        else:
            df_merge.loc[nonan_idxs,metric_best] = df_merge.loc[nonan_idxs,cols].idxmax(axis=1)

"""
# -------------------- full dist conditions --------------------

# best distributions based on KS p-value
pval_names = ['ks pvalue stable', 'ks pvalue normal', 'ks pvalue tstudent', 'ks pvalue lognorm']
logl_names = ['logl_stable', 'logl_norm', 'logl_t', 'logl_lognorm']

pval_idxss = {}; logl_idxss = {}
for idx, pval_name in enumerate(pval_names):    
    pval_idxss[pval_name] = df_full.index[df_full[pval_names].idxmax(axis=1)==pval_name]
    logl_idxss[logl_names[idx]] = df_full.index[df_full[logl_names].idxmax(axis=1)==logl_names[idx]]

    print(f"{pval_name}: {len(pval_idxss[pval_name])}")
    print(f"{logl_names[idx]}: {len(logl_idxss[logl_names[idx]])}")

print("\n")
# -------------------- tail dist conditions --------------------

# lower tail best fitted by plaw
dist_names = ['power_law', 'truncated_power_law', 'lognormal', 'exponential']
lower1_idxss = {}; upper1_idxss = {}
lower2_idxss = {}; upper2_idxss = {}
for dist_name in dist_names:
    lower1_idxss[dist_name] = df_tail.index[df_tail.loc[:,'bf1_lower']==dist_name]
    lower2_idxss[dist_name] = df_tail.index[df_tail.loc[:,'bf2_lower']==dist_name]
    upper1_idxss[dist_name] = df_tail.index[df_tail.loc[:,'bf1_upper']==dist_name]
    upper2_idxss[dist_name] = df_tail.index[df_tail.loc[:,'bf2_upper']==dist_name]

    print(f"{dist_name} (lower 1): {len(lower1_idxss[dist_name])}")
    print(f"{dist_name} (lower 2): {len(lower2_idxss[dist_name])}")
    print(f"{dist_name} (upper 1): {len(upper1_idxss[dist_name])}")
    print(f"{dist_name} (upper 2): {len(upper2_idxss[dist_name])}")    
    print('\n')    
"""

# -------------------- (a) Featured CNN convolution tensor --------------------

# full distribution best fitted by stable dist and tails best fitted by tpl
# df_pl_tpl_stable = df_merge[(df_merge.loc[:,'bf1_upper']=='truncated_power_law') \
#                      & (df_merge.loc[:,'ks pvalue stable']>=0.05)
#                      & (df_merge.loc[:,'alpha_upper']<=2)].reset_index()

# condition for Fig. 1(a)
#cond_a = (df_merge.loc[:,'bf1_upper'] == 'power_law')
#cond_a = (df_merge.loc[:,'weight_num'] > min_weight_num)
#cond_a = (df_merge.loc[:,'fit_size'] > min_weight_num)
cond_a = (df_merge.loc[:,'removed_perc'] <= max_removed_perc)
cond_a = cond_a & ((df_merge.loc[:,'bf1_upper'] == 'truncated_power_law') | (df_merge.loc[:,'bf1_upper'] == 'power_law'))
if pass_ad_test:
    cond_a = cond_a & (df_merge.loc[:,'ad sig level stable']>=0.05)
if pass_ks_test:
    cond_a = cond_a & (df_merge.loc[:,'ks pvalue stable']>=0.05)  
if param_selection_base == 'ad':
    cond_a = cond_a & (df_merge.loc[:,'bf1_upper'] == 'truncated_power_law')
passed_idxs = df_merge[cond_a].index     
#df_pl_tpl_stable = df_merge.loc[passed_idxs,:].reset_index()
df_pl_tpl_stable = df_merge.loc[passed_idxs,:]

# find indices where alpha matchese to PL/TPL fit
match_idxs = df_pl_tpl_stable.index[np.abs(df_pl_tpl_stable.loc[:,'alpha'] + 1 - df_pl_tpl_stable.loc[:,'alpha_upper']) < 0.2]

# full distribution best fitted by stable dist and tails best fitted by tpl
#df_merge[(df_merge.loc[:,'bf1_upper']=='power_law') & (df_merge.loc[:,'ks pvalue stable']>=0.05)].head(5)

#l, wmat_idx = 96, 260
#net = 'efficientnet_v2_m'

# featured weight matrix
feat_i = match_idxs[2]  # best
# #feat_i = match_idxs[8]  # second best
# #feat_i = match_idxs[0]  # cur
l = df_pl_tpl_stable.loc[feat_i,'idx']
wmat_idx = df_pl_tpl_stable.loc[feat_i,'wmat_idx']
net = df_pl_tpl_stable.loc[feat_i,'model_name']

# original
# feat_i = 6053
# l = df_tail.loc[feat_i,'idx']
# wmat_idx = df_tail.loc[feat_i,'wmat_idx']
# net = df_tail.loc[feat_i,'model_name']

w_type = f"{net}_layer_{l}_{wmat_idx}"
param_type = f"{net}_allfit_{l}_{wmat_idx}"
w_path = f"{prepath}/weights_all/{w_type}"

print('----------------------------')
print(f'Featured networks: {net} \n')
print(f'{param_type} \n')
print('----------------------------')


#allfit_path = f"{prepath}/{allfit_path_dir}/{net}/{param_type}.csv"

for allfit_path_dir in allfit_paths:
    allfit_path = f"{allfit_path_dir}/{net}/{param_type}.csv"
    print(allfit_path + "\n")
    if isfile(allfit_path):        
        break

print(f'pass_ad_test = {pass_ad_test} \n')

print(f'pass_ks_test = {pass_ks_test} \n')

print(f"Featured weight matrix: {net}_{l}_{wmat_idx} \n")

# tail fit paths
param_type_tail = f"{net}_plfit_{l}_{wmat_idx}"
tailfit_path = f"{prepath}/ww_plfit_all/{net}/{param_type_tail}.csv"

params_stable = pd.read_csv(allfit_path).iloc[0,3:7]
params_normal = pd.read_csv(allfit_path).iloc[0,11:13]
params_tstudent = pd.read_csv(allfit_path).iloc[0,19:22]
params_lognorm = pd.read_csv(allfit_path).iloc[0,26:29]
sample_w = torch.load(w_path).detach().numpy()

"""
#2. Filter weight matrices/tensors with very few entries
#cond1_idxs = np.where(metrics_all['weight_num'] >= min_weight_num)
#cond1_idxs = [ ind for ind in range(len(metrics_all['weight_num'])) if metrics_all['weight_num'][ind] >= min_weight_num]
#cond1_idxs = df_full.index[df_full.loc[:,'weight_num'] >= min_weight_num]
#for metric_name in metric_names + ['top-1', 'top-5']:
#    if metric_name != 'sample_w':
#        metrics_all[metric_name] = [metrics_all[metric_name][ind] for ind in cond1_idxs]
df_full = df_full[df_full.loc[:,'weight_num'] >= min_weight_num].reset_index(drop=True)
df_tail = df_tail[df_tail.loc[:,'weight_num'] >= min_weight_num].reset_index(drop=True)

# 1. filter out None and nan values
df_full = df_full.replace(to_replace='None', value=np.nan).dropna().reset_index(drop=True)
df_tail = df_tail.replace(to_replace='None', value=np.nan).dropna().reset_index(drop=True)

#metrics_all = df_full.to_dict() 
#metrics_tail = df_tail.to_dict()
"""

#print(f"{len(metrics_all[metric_name])}/{total_wmat} of tensors are taken into account!")


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

# xlim_ls = [[-0.3, 0.3], [0.5,2], [0,1267035], [1e-2, 1], [0,15], [0,5]]
# ylim_ls = [[0,12.5], [0,5], [0,3e-3], [1e-4,500], [0,0.3], [0,2]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0,3e-3], [1e-4,500], [0,15], [0,5]]
ylim_ls = [[0,12.5], [0,5], [0,1267035], [1e-2, 1], [0,0.3], [0,2]]

#axs_1 = [ax1, ax2, ax3, ax4, ax5, ax6]
#metric_names_plot = ["sample_w", "alpha", "nu", "sample_w", "alpha_tail_tpl", "alpha_tail_pl"]
#metric_names_plot = ["sample_w", "alpha", "nu", "sample_w", "alpha_tail_tpl", "xlogrange_pl"]
metric_names_plot = ["sample_w", "alpha", "nu", "sample_w", "alpha_tail_pl", "xlogrange_pl"]
metrics_all_plot = {}
top_n = 'top-5'     # choose Top-1 or Top-5 accuracy

for i, metric_name in enumerate(metric_names_plot):

    fig = plt.figure(figsize=(24, 21))
    axis = plt.subplot(111)
    # set up individual figures
    if i != 3:
        plt.rcParams.update(param_sizes_2)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)             
    else:
        plt.rcParams.update(param_sizes_1)    

    # full distribution
    if metric_name in ['alpha','nu']:
        # filter
        best_ks = "ks pvalue stable" if metric_name == 'alpha' else "ks pvalue tstudent"
        best_ad = "ad sig level stable" if metric_name == 'alpha' else "ad sig level tstudent"
        #cond = (df_merge.loc[:,'weight_num'] > min_weight_num) & (df_merge.loc[:,'ks_best'] == best_ks)
        #cond = (df_merge.loc[:,'weight_num'] > min_weight_num)
        #cond = (df_merge.loc[:,'fit_size'] > min_weight_num)
        cond = (df_merge.loc[:,'removed_perc'] <= max_removed_perc)
        # if param_selection_base == 'ad':
        #     cond = cond & (df_merge.loc[:,'ad_best'] == best_ad)
        # elif param_selection_base == 'ks':
        #     cond = cond & (df_merge.loc[:,'ks_best'] == best_ks)            
        if pass_ks_test:
            cond = cond & (df_merge.loc[:,best_ks] >= 0.05)        
        if pass_ad_test:
            cond = cond & (df_merge.loc[:,best_ad] >= 0.05)

        selected_idxs = df_merge[cond].index

        metric = df_merge.loc[selected_idxs,metric_name]
        #print(f'plotted {metric_name}: {len(metric)} \n')

        if metric_name == 'nu':
            metric = metric[metric < 18]  # remove outliers

        metrics_all_plot[metric_name] = metric
    
    elif metric_name in ['alpha_tail_tpl', 'alpha_tail_pl']:        

        best_tailfit = "truncated_power_law" if metric_name == 'alpha_tail_tpl' else 'power_law'        

        # filter
        #cond = (df_merge.loc[:,'fit_size'] > min_weight_num) & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        cond = (df_merge.loc[:,'removed_perc'] <= max_removed_perc) & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        alpha_lower = df_merge.loc[selected_idxs,'alpha_lower']

        #cond = (df_merge.loc[:,'fit_size'] > min_weight_num) & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        cond = (df_merge.loc[:,'removed_perc'] <= max_removed_perc) & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        alpha_upper = df_merge.loc[cond,'alpha_upper']

        metric = pd.concat([alpha_lower, alpha_upper])

        metrics_all_plot[metric_name] = metric

    elif metric_name in ['xlogrange_pl']:

        #best_tailfit = "truncated_power_law"
        best_tailfit = 'power_law'

        # filter
        #cond = (df_merge.loc[:,'fit_size'] > min_weight_num) & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        cond = (df_merge.loc[:,'removed_perc'] <= max_removed_perc) & (df_merge.loc[:,'bf1_lower'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        xlogrange_lower = df_merge.loc[selected_idxs,'xlogrange_lower']

        #cond = (df_merge.loc[:,'fit_size'] > min_weight_num) & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        cond = (df_merge.loc[:,'removed_perc'] <= max_removed_perc) & (df_merge.loc[:,'bf1_upper'] == best_tailfit)
        selected_idxs = df_merge[cond].index        
        xlogrange_upper = df_merge.loc[cond,'xlogrange_upper']

        metric = pd.concat([xlogrange_lower, xlogrange_upper])       

        metrics_all_plot[metric_name] = metric 

    # plotting the histogram
    if i == 0:
        axis.hist(sample_w[np.abs(sample_w) > 1e-5], bin_ls[i], color=c_ls_2[i], density=True)       
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, *params_normal)
        #y_tstudent = sst.t.pdf(x, *params_tstudent)
        #y_lognorm = lognorm.pdf(x, *params_lognorm)
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dotted', label = 'Stable')
        #axis.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[2], linestyle='dashdot', label = 'Student t')
        #axis.plot(x, y_lognorm, linewidth=lwidth, c=c_ls_1[3], linestyle='dotted', label = 'Log-normal')  

        #xlim_ls[0][0], xlim_ls[0][1] = sample_w[np.abs(sample_w) > 1e-5].min(), sample_w[np.abs(sample_w) > 1e-5].max()
        ylim_ls[0][1] = max(y_stable.max(), y_normal.max()) + 2

    elif i == 1:
        # KDE
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,1000) 
        #axis.plot(x, density(x))   

        # HIST
        axis.hist(metric, bin_ls[i], color=c_ls_2[i], density=True)
        #sns.distplot(metric, hist=False, color=c_ls_2[i], ax=axis)  # kde_kws={'linestyle':'--', 'linewidth':'width'}

        #axis.set_xlim(np.percentile(metric, 1), np.percentile(metric, 99))

    elif i == 2:
        # KDE
        # density = sst.gaussian_kde(metric)
        # x = np.linspace(0,160,1000) 
        # #axis.plot(x, density(x))    

        # HIST   
        axis.hist(metric, bin_ls[i], color=c_ls_2[i], density=True)
        #sns.distplot(metric, hist=False, color=c_ls_2[i], ax=axis)  # kde_kws={'linestyle':'--', 'linewidth':'width'}

        # BOXPLOT
        # bp = axis.boxplot(metric, showfliers=False)
        # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     plt.setp(bp[element], color=c_ls_2[i], linewidth=lwidth)                

    elif i == 3 and isfile(tailfit_path):
        # distribution tail
        df_tailfit = pd.read_csv(tailfit_path)
        layer_idx, wmat_idx, total_entries, fit_entries, \
        k, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, \
        best_fit_1, best_fit_2, Rs_all, ps_all = df_tailfit.iloc[1,:].values

        print(f"Plaw to upper tail: k = {k}, xmin = {xmin}, xmax = {xmax}.")

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
        from weightwatcher.RMT_Util import ax_plot_loghist       
        num_bins = 500
        counts, binEdges = ax_plot_loghist(axis, evals_to_plot, bins=num_bins, xmin=None)  

        # ----- method 4 (kde) -----
        density = sst.gaussian_kde(metric)
        x = np.linspace(min_evals_to_plot,evals.max(),1000)  
        axis.plot(x, density(x))                    

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
        axis.plot(xs, ys, c='g', linestyle='dotted', label=rf'PL fit ($k$ = {round(k,2)})')

        # --- eye guide 2 --- (from stable)   
        # k_stable = params_stable[0] + 1               
        # b_stable = (np.log(yc) + k_stable * np.log(xc))/np.log(10) + 1
        # ys_stable = 10**b_stable * xs**(-k_stable)
        # axis.plot(xs, ys_stable, c='b', linestyle='dotted', label=rf'Stable fit ($k$ = {round(params_stable[0] + 1,2)})')

        # mark xmin
        #axis.plot([], [], c='r', linestyle='dashed', label=rf'$x_{{\min}}$ = {round(xmin,2)}') 
        axis.axvline(xmin, color='r', linestyle='dashed')  # label=rf'$x_{{\min}}$ = {round(xmin,2)}'

        # kde
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,500) 
        #axis.plot(x, density(x))

        axis.set_xlim(evals_to_plot.min(), evals_to_plot.max())        
        axis.set_xscale('log'); axis.set_yscale('log')     

        #ax1_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

        #axis.minorticks_off()
        #axis.tick_params(left=False, right=False , labelleft=False,
        #                 labelbottom=False, bottom=False)       

    elif i == 4:
        # KDE
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,500) 
        #axis.plot(x, density(x))  

        # HIST
        axis.hist(metric, bin_ls[i], color=c_ls_2[3], density=True) 
        #sns.distplot(metric, hist=False, color=c_ls_2[3], ax=axis)  # kde_kws={'linestyle':'--', 'linewidth':'width'}
        

        # bp = axis.boxplot(metric)
        # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #     plt.setp(bp[element], color=c_ls_2[i-1], linewidth=lwidth)           
         
    elif i == 5:
        axis.hist(metric, bin_ls[i], color=c_ls_2[4], density=True)        
        #sns.distplot(metric, hist=False, color=c_ls_2[4], ax=axis)  # kde_kws={'linestyle':'--', 'linewidth':'width'}

    #if i == 0 or i == 3:
    if i == 0:
        axis.legend(loc = 'upper left', fontsize = legend_size_1, frameon=False)

    if i not in [0,3]:
        plotted_metrics[metric_name] = metric
        print(f"{metric_name} size: {len(metric)} \n")
        if i == 1:
            print(print(f"{metric_name} smaller than 1.9: {len(metric[metric <= 1.9])} \n"))

    # set axis limit
    #if i != 3:  
    if i not in [1,2,3,4]:
        axis.set_xlim(xlim_ls[i])
        axis.set_ylim(ylim_ls[i])

    # tick labels
    if i != 3:
        axis.tick_params(axis='x', labelsize=axis_size_1)
        axis.tick_params(axis='y', labelsize=axis_size_1)
    else:
        axis.tick_params(axis='x', labelsize=axis_size_2)
        axis.tick_params(axis='y', labelsize=axis_size_2)  
        axis.set_yticklabels([])      

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
    elif i < 3:
        fig_file = f'fig1_{alphabet[i]}.pdf'
    elif i == 3:
        fig_file = 'fig1_a_inset.pdf'
    else:
        fig_file = f'fig1_{alphabet[i-1]}.pdf'

    fig1_path = "/project/PDLAI/project2_data/figure_ms/pretrained_fitting"
    if not isdir(fig1_path): os.makedirs(fig1_path)
    #axis.subplots_adjust(hspace=0.2)
    plt.tight_layout()    
    plt.savefig(join(fig1_path, fig_file), bbox_inches='tight')    

print("---------------------------- \n")

# -------------------- Statistics summary --------------------

print(f"Total number of weight matrices: {df_merge.shape[0]} \n")

print("Full fit \n")

pval_thresh = 0.05
#cond = (df_merge.loc[:,'fit_done'] == True) & (df_merge.loc[:,'fit_size'] >= min_weight_num)
cond = (df_merge.loc[:,'fit_done'] == True) & (df_merge.loc[:,'removed_perc'] <= max_removed_perc)
total_legit_wmat = df_merge[cond].shape[0]
#print(f"Total entries greater than {1e-5} and with fit_size at least {min_weight_num}: {total_legit_wmat} \n")
print(f"Proportion of entries greater than {1e-5} removed at most {max_removed_perc*100}%: {total_legit_wmat} \n")

# AD p-value > 0.05
cond_ad = ((df_merge.loc[:,'ad sig level stable'] >= pval_thresh) | (df_merge.loc[:,'ad sig level normal'] >= pval_thresh) | \
            (df_merge.loc[:,'ad sig level tstudent'] >= pval_thresh) | (df_merge.loc[:,'ad sig level lognorm'] >= pval_thresh))
if pass_ad_test:
    cond = cond & cond_ad

# KS p-value > 0.05
cond_ks = ((df_merge.loc[:,'ks pvalue stable'] >= pval_thresh) | (df_merge.loc[:,'ks pvalue normal'] >= pval_thresh) | \
            (df_merge.loc[:,'ks pvalue tstudent'] >= pval_thresh) | (df_merge.loc[:,'ks pvalue lognorm'] >= pval_thresh))
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

print("---------------------------- \n")
print("Tail fit \n")

#cond = (df_merge.loc[:,'fit_done_tail'] == True) & (df_merge.loc[:,'fit_size'] >= min_weight_num)
cond = (df_merge.loc[:,'fit_done_tail'] == True) & (df_merge.loc[:,'removed_perc'] <= max_removed_perc)
total_legit_wmat = df_merge[cond].shape[0]
#print(f"Total wmat that with most entries greater than {1e-5} and with entries at least {min_weight_num}: {total_legit_wmat} \n")
print(f"Proportion of entries greater than {1e-5} removed at most {max_removed_perc*100}%: {total_legit_wmat} \n")

criterion_pl = ['bf1', 'bf2']
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


