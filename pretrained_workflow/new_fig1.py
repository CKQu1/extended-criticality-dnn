import argparse
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
from os.path import join, isfile
from scipy.stats import levy_stable, norm, distributions, lognorm
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name

pub_font = {'family' : 'sans-serif'}
plt.rc('font', **pub_font)

t0 = time.time()

# ---------------------------

# 3 by 3 template

# colour schemes
cm_type_1 = 'coolwarm'
cm_type_2 = 'RdGy'
#c_ls_1 = ["forestgreen", "coral",]
c_ls_1 = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
c_ls_2 = ["peru", "dodgerblue", "limegreen"]
c_ls_3 = ["red", "blue"]
#c_hist_1 = "tab:blue"
#c_hist_1 = "dimgrey"
#c_hist_2 = "dimgrey"
c_hist_1 = "tab:blue"
c_hist_2 = "dimgrey"

tick_size = 18.5 * 1.5
label_size = 18.5 * 1.5
axis_size = 18.5 * 1.5
legend_size = 14 * 1.5
lwidth = 4.2
text_size = 14 * 1.5
marker_size = 20

label_ls = [f"({letter})" for letter in list(string.ascii_lowercase)]
linestyle_ls = ["solid", "dashed","dashdot"]

params = {'legend.fontsize': legend_size,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size}
plt.rcParams.update(params)

fig = plt.figure(figsize=(24, 16))
gs = mpl.gridspec.GridSpec(490, 740, wspace=0, hspace=0)   
# main plots 
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[290:490, 0:200])
ax5 = fig.add_subplot(gs[290:490, 260:460])
ax6 = fig.add_subplot(gs[290:490, 520:720])

# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"
grouped_stats_path = join(prepath, "grouped_stats")
if not os.path.isdir(grouped_stats_path): os.makedirs(grouped_stats_path)

df_full_file = "full_dist_grouped.csv"
df_tail_file = "tail_dist_grouped.csv"

"""
Featured CNN convolution tensor
"""

l = 1
wmat_idx = 2
net = "vgg11"
w_type = f"{net}_layer_{l}_{wmat_idx}"
param_type = f"{net}_allfit_{l}_{wmat_idx}"
w_path = f"{prepath}/weights_all/{w_type}"
allfit_path = f"{prepath}/allfit_all/{net}/{param_type}.csv"

params_stable = pd.read_csv(allfit_path).iloc[0,3:7]
params_normal = pd.read_csv(allfit_path).iloc[0,11:13]
params_tstudent = pd.read_csv(allfit_path).iloc[0,19:22]
params_lognorm = pd.read_csv(allfit_path).iloc[0,26:29]
sample_w = torch.load(w_path).detach().numpy()

# tail fit paths
param_type_tail = f"{net}_plfit_{l}_{wmat_idx}"
tailfit_path = f"{prepath}/ww_plfit_all/{net}/{param_type_tail}.csv"

metric_names = ['alpha', 'sigma',    # full distribution fit  
                'shap pvalue', 'ks pvalue stable', 'ks pvalue normal',
                'ks pvalue tstudent', 'ks pvalue lognorm',
                'nu', 'sigma_t', 'mu_t',      
                'logl_stable', 'logl_norm', 'logl_t', 'logl_lognorm',          
                'model_name', 'param_shape', 'weight_num', 'wmat_idx', 'idx', 'weight_file',
                'fit_done', 'dirname']

metric_names_tail = ['alpha_lower', 'alpha_upper', 
                    #'total_entries', 'fit_entries', 'xmin', 'xmax', 
                    #'best_fit_1', 'best_fit_2',
                    'bf1_lower', 'bf1_upper', 'bf2_lower', 'bf2_upper',
                    'model_name', 'param_shape', 'weight_num', 'wmat_idx', 'idx', 'weight_file',
                    'fit_done', 'dirname']

# replace even if created
replace = False
if not (isfile(join(grouped_stats_path, df_full_file)) and isfile(join(grouped_stats_path, df_tail_file))) or replace:
    print("At least one file does not exist, creating now!")

    metrics_all = {}
    #metrics_all['sample_w'] = torch.load(w_path).detach().numpy()    

    """
    Distribution of fitted stable distribution parameters
    """

    # load two files of fitted params from Pytorch and Tensorflow
    files = []
    #for mpath in [f"{prepath}/allfit_all", f"{prepath}/allfit_all_tf"]:
    allfit_paths = [f"{prepath}/allfit_all", f"{prepath}/allfit_all_tf", f"{prepath}/nan_allfit_all", f"{prepath}/nan_allfit_all_tf"]
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
                metrics_all['dirname'].append(None)

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
                metrics_tail['dirname'].append(None)

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
            # log-likelihood
            logl_stable, logl_norm, logl_t, logl_lognorm = df.loc[0,["logl_stable","logl_norm","logl_t", "logl_lognorm"]]
        else:          
            alpha, sigma, nu, sigma_t, mu_t, shape_stat, ks_pvalue, ks_pvalue_normal = None, None, None, None, None, None, None, None
            ks_pvalue_tstudent, ks_pvalue_lognorm = None, None
            files_failed.append( weight_foldername )
        metrics_all['alpha'].append( alpha )
        metrics_all['sigma'].append( sigma )     
        metrics_all['nu'].append( nu )
        metrics_all['sigma_t'].append( sigma_t )    
        metrics_all['mu_t'].append( mu_t )      

        metrics_all['shap pvalue'].append(shap_stat)    
        metrics_all['ks pvalue stable'].append(ks_pvalue_stable)
        metrics_all['ks pvalue normal'].append(ks_pvalue_normal)   
        metrics_all['ks pvalue lognorm'].append(ks_pvalue_lognorm)
        metrics_all['ks pvalue tstudent'].append(ks_pvalue_tstudent)      

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
            alpha_lower = df_pl.loc[0,'alpha'].item()
            alpha_upper = df_pl.loc[1,'alpha'].item()
            bf1_lower = df_pl.loc[0,'best_fit_1']
            bf1_upper = df_pl.loc[1,'best_fit_1']
            bf2_lower = df_pl.loc[0,'best_fit_2']
            bf2_upper = df_pl.loc[1,'best_fit_2']       
                         
        else:          
            alpha_lower, alpha_upper = None, None
            bf1_lower, bf1_upper, bf2_lower, bf2_upper = None, None, None, None
            files_failed.append( weight_foldername )
        metrics_tail['alpha_lower'].append( alpha_lower )
        metrics_tail['alpha_upper'].append( alpha_upper )
        metrics_tail['bf1_lower'].append(bf1_lower)
        metrics_tail['bf1_upper'].append(bf1_upper)
        metrics_tail['bf2_lower'].append(bf1_lower)
        metrics_tail['bf2_upper'].append(bf1_upper)

    df_full = pd.DataFrame(data=metrics_all)
    df_full.to_csv(join(grouped_stats_path, df_full_file)) 

    df_tail = pd.DataFrame(data=metrics_tail)
    df_tail.to_csv(join(grouped_stats_path, df_tail_file)) 
    print("Summary csv saved!")

else:
    print("Files already created!")
    df_full = pd.read_csv(join(grouped_stats_path, df_full_file))
    df_tail = pd.read_csv(join(grouped_stats_path, df_tail_file))


#2. Filter weight matrices/tensors with very few entries
min_weight_num = 1000
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

print("\n")
#print(f"{len(metrics_all[metric_name])}/{total_wmat} of tensors are taken into account!")


# -------------------- Plot --------------------
print("Start plotting")

good = 0
#bin_ls = [1000,50,2500]
#bin_ls = [1000,1000,1000,100,100]
bin_ls = [1000] * 6
#xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\alpha$"] 
#ylabel_ls = ["Probability density", "Probability density", r"$\sigma_w$"]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})"] \
            + [r"$\alpha$ (Stable)", r"$\nu$ (Student t)"] \
            + [r"$\mathbf{{W}}^{{{}}}$ upper tail".format(l + 1), r"$k$ (lower)", r"$k$ (upper)"]
            
ylabel_ls = ["Probability density", "Probability density", "Probability density", 
             "Probability density", "Probability density", "Probability density"]

xlim_ls = [[-0.3, 0.3], [0.5,2], [0,1e4], [1e-2, 1], [0,25], [0,25]]
ylim_ls = [[0,12.5], [0,5], [0,1e-2], [1e-4,500], [0,0.5], [0,0.5]]

axs_1 = [ax1, ax2, ax3, ax4, ax5, ax6]
#metric_names_plot = ["sample_w", "alpha", "sigma_scaled"]
metric_names_plot = ["sample_w", "alpha", "nu", "sample_w", "alpha_lower", "alpha_upper"]
top_n = 'top-5'     # choose Top-1 or Top-5 accuracy
#for i in range(3):
for i, metric_name in enumerate(metric_names_plot):

    if metric_name != '' and metric_name != 'sample_w':
        if 1 <= i <= 2 or i == 3:
            metric = df_full.loc[:,metric_name]
        else:
            metric = df_tail.loc[:,metric_name]
        # remove None values
        """
        while None in metric:
            metric.remove(None)
        print(f"{metric_names_plot[i]} has total values of {len(metric)}")
        """   

    axis = axs_1[i]

    # figure labels
    label = label_ls[i] 
    #axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
    #     fontsize=label_size, va='top', ha='right')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # plotting the histogram
    if i == 0:
        axis.hist(sample_w, bin_ls[i], color=c_hist_1, density=True)       
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, *params_normal)
        #y_tstudent = sst.t.pdf(x, *params_tstudent)
        #y_lognorm = lognorm.pdf(x, *params_lognorm)
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[2], linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')
        #axis.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[2], linestyle='dashdot', label = 'Student t')
        #axis.plot(x, y_lognorm, linewidth=lwidth, c=c_ls_1[3], linestyle='dotted', label = 'Log-normal')  

    elif i == 1:
        axis.hist(metric[pval_idxss['ks pvalue stable']], bin_ls[i], color=c_hist_2, density=True)
        #axis.set_xlim(np.percentile(metric, 1), np.percentile(metric, 99))

    elif i == 2:
        density = sst.gaussian_kde(metric[pval_idxss['ks pvalue tstudent']])
        x = np.linspace(0,2,500) 
        axis.plot(x, density(x))        

    elif i == 3:
        # distribution tail
        df_tailfit = pd.read_csv(tailfit_path)
        layer_idx, wmat_idx, total_entries, fit_entries, \
        alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, \
        best_fit_1, best_fit_2, Rs_all, ps_all = df_tailfit.iloc[1,:].values

        print(f"Plaw to upper tail: k = {alpha}, xmin = {xmin}, xmax = {xmax}.")

        # upper tail
        min_evals_to_plot = (xmin/100)
        evals = np.abs(sample_w[sample_w>=0])
        evals_to_plot = evals[evals>min_evals_to_plot]

        # adapted from weightwatcher/WW_powerlaw.py (fit_powerlaw()) 
        from weightwatcher.RMT_Util import ax_plot_loghist       
        num_bins = 500
        counts, binEdges = ax_plot_loghist(axis, evals_to_plot, bins=num_bins, xmin=xmin)       

        """
        counts, binEdges = np.histogram(evals_to_plot, bins=num_bins, density=True)
        logbins = np.logspace(np.log10(binEdges[0]),np.log10(binEdges[-1]),len(binEdges))
        axis.hist(x, bins=logbins, density=True)
        axis.axvline(xmin, color='r', label=r"$x_{{{}}}$".format("min"))
        """

        # eye guide
        binEdges = binEdges[1:]
        xc = binEdges[binEdges>=xmin].min()
        yc = counts[binEdges>=xmin].max()
        #b = yc + alpha * xc
        b = (np.log(yc) + alpha * np.log(xc))/np.log(10)
        #xs = np.linspace(np.exp(xmin), np.exp(xmax), 500)
        xs = np.linspace(xmin, xmax, 500)
        #ys = -alpha * xs + b
        #axs[0,0].plot(np.exp(xs), np.exp(ys), c='g')
        ys = 10**b * xs**(-alpha)
        axis.plot(xs, ys, c='g', label='powerlaw fit')           

        # kde
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,500) 
        #axis.plot(x, density(x))

        axis.set_xlim(evals_to_plot.min(), evals_to_plot.max())
        
        # log axis
        axis.set_xscale('log'); axis.set_yscale('log')     

        #ax1_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

        #axis.minorticks_off()
        #axis.tick_params(left=False, right=False , labelleft=False,
        #                 labelbottom=False, bottom=False)       

    elif i == 4:
        axis.hist(metric[lower1_idxss['truncated_power_law']], bin_ls[i], color=c_hist_2, density=True)
        #axis.set_xlim(np.percentile(metric, 1), np.percentile(metric, 99))
        # get kde
        #density = sst.gaussian_kde(metric)
        #x = np.linspace(0,2,500) 
        #axis.plot(x, density(x))   
         
    elif i == 5:
        axis.hist(metric[upper1_idxss['truncated_power_law']], bin_ls[i], color=c_hist_2, density=True)
      
    if i == 0 or i == 3:
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

    # set axis limit
    axis.set_xlim(xlim_ls[i])
    axis.set_ylim(ylim_ls[i])

    # tick labels
    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    #axis.xaxis.set_minor_locator(AutoMinorLocator())
    #axis.yaxis.set_minor_locator(AutoMinorLocator())

    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    #if i == 0 or i == 1:
    #axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
    #axis.set_title(f"{ylabel_ls[i]}", fontsize=axis_size)


# -------------------- Statistics summary --------------------




# -------------------- Save fig --------------------

print(f"Time: {time.time() - t0}")

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
plt.savefig(f"{fig1_path}/new_fig1.pdf", bbox_inches='tight')
#plt.show()

