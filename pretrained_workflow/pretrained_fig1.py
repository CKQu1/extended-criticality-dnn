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
from os.path import join
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
c_hist_1 = "dimgrey"
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

"""
Featured CNN convolution tensor
"""

l = 1
wmat_idx = 2
net = "vgg11"
w_type = f"{net}_layer_{l}_{wmat_idx}"
param_type = f"{net}_allfit_{l}_{wmat_idx}"
w_path = f"{prepath}/weights_all/{w_type}"
param_path = f"{prepath}/allfit_all/{net}/{param_type}.csv"

metrics_all = {}
metrics_all['sample_w'] = torch.load(w_path).detach().numpy()
params_stable = pd.read_csv(param_path).iloc[0,3:7]
params_normal = pd.read_csv(param_path).iloc[0,11:13]
params_tstudent = pd.read_csv(param_path).iloc[0,19:22]
params_lognorm = pd.read_csv(param_path).iloc[0,26:29]

"""
Distribution of fitted stable distribution parameters
"""

# load two files of fitted params from Pytorch and Tensorflow
files = []
#for mpath in [f"{prepath}/allfit_all", f"{prepath}/allfit_all_tf"]:
allfit_paths = [f"{prepath}/allfit_all", f"{prepath}/allfit_all_tf", f"{prepath}/nan_allfit_all", f"{prepath}/nan_allfit_all_tf"]
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
metric_names = ['alpha', 'sigma',  
                'shap pvalue', 'ks pvalue stable', 'ks pvalue normal',
                'nu', 'sigma_t', 'mu_t',                
                'model_name', 'param_shape', 'weight_num', 'wmat_idx', 'idx', 'weight_file',
                'fit_done', 'dirname', 'max_weight']
net_names = []
for metric_name in metric_names:
    metrics_all[metric_name] = []

# weight tensor information 
#for weight_info_name in ["weight_info.csv"]:
for weight_info_name in ["weight_info.csv", "weight_info_tf.csv"]:
    weight_info = pd.read_csv( join(prepath,weight_info_name) )    
    for metric_idx in range(metric_names.index('model_name'),len(metric_names)-3):
        metric_name = metric_names[metric_idx]
        metrics_all[metric_name] += list(weight_info.loc[:,metric_name])
    for ii in tqdm(range(weight_info.shape[0])):
        fit_done = False
        weight_foldername = replace_name(weight_info.loc[ii,'weight_file'], "allfit") + ".csv"    
        model_name = weight_info.loc[ii,'model_name']        
        for dirname in ["allfit_all", "nan_allfit_all", "allfit_all_tf", "nan_allfit_all_tf"]:        
            fit_done = fit_done or os.path.isfile(join(prepath, dirname, model_name, weight_foldername))
            if fit_done:
                break     
        metrics_all['fit_done'].append(fit_done)
        if fit_done:
            metrics_all['dirname'].append(dirname)
        else:
            metrics_all['dirname'].append(None)
        wmat_file = replace_name(weight_info.loc[ii,'weight_file'], "layer") + '.npy'
        if "tf" not in weight_info_name:
            metrics_all['max_weight'].append( np.max(np.load(join(prepath, "np_weights_all", wmat_file))) )
        else:
            metrics_all['max_weight'].append( np.max(np.load(join(prepath, "np_weights_all_tf", wmat_file))) )
        
total_wmat = len(metrics_all['model_name'])
print(f"{len(files)} out of {total_wmat} have been analyzed!")

# top-1 and top-5 acc
#net_names_all = [pd.read_csv(join(prepath,fname)) for fname in ["net_names_all.csv", "net_names_all_tf.csv"]]
#metrics_all['top-1'], metrics_all['top-5'] = [], []

files_failed = []
# load stablefit alpha and sigma
for ii in tqdm(range(len(metrics_all['param_shape']))):
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
        shap_stat, ks_pvalue_stable, ks_pvalue_normal = df.loc[0,["shap pvalue","ks pvalue stable","ks pvalue normal"]]
    else:          
        alpha, sigma, nu, sigma_t, mu_t, shape_stat, ks_pvalue, ks_pvalue_normal = None, None, None, None, None, None, None, None
        files_failed.append( weight_foldername )
    metrics_all['alpha'].append( alpha )
    metrics_all['sigma'].append( sigma )     
    metrics_all['nu'].append( nu )
    metrics_all['sigma_t'].append( sigma_t )    
    metrics_all['mu_t'].append( mu_t )      
    metrics_all['shap pvalue'].append(shap_stat)
    metrics_all['ks pvalue stable'].append(ks_pvalue_stable)
    metrics_all['ks pvalue normal'].append(ks_pvalue_normal)     

# filter weight matrices/tensors with very few entries
min_weight_num = 1000
#indices = np.where(metrics_all['weight_num'] >= min_weight_num)
indices = [ ind for ind in range(len(metrics_all['weight_num'])) if metrics_all['weight_num'][ind] >= min_weight_num]
#for metric_name in metric_names + ['top-1', 'top-5']:
#    if metric_name != 'sample_w':
#        metrics_all[metric_name] = [metrics_all[metric_name][ind] for ind in indices]

print(f"{len(metrics_all[metric_name])}/{total_wmat} of tensors are taken into account!")

# histogram and fit plot 
print("Start plotting")

good = 0
#bin_ls = [1000,50,2500]
bin_ls = [1000,1000,1000,100,100]
#xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\alpha$"] 
#ylabel_ls = ["Probability density", "Probability density", r"$\sigma_w$"]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})"]*2 +  ["", r"$\alpha$", r"$\nu$", ""] 
ylabel_ls = ["Probability density", "Right tail", "Extreme weights", "Probability density", "Probability density", ""]

#xlim_ls = [[-0.3, 0.3], [0.5,2], [0,0.3]]
#ylim_ls = [[0,12.5], [0,3], [0,40]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0.45,2.05]]
#ylim_ls = [[0,12.5], [0,3], [-1,1000]]
ylim_ls = [[0,12.5], [], [-.5,20]]

axs_1 = [ax1, ax2, ax3, ax4, ax5, ax6]
#metric_names_plot = ["sample_w", "alpha", "sigma_scaled"]
metric_names_plot = ["sample_w", "sample_w", "max_weight", "alpha", "nu", ""]
top_n = 'top-5'     # choose Top-1 or Top-5 accuracy
#for i in range(3):
for i in range(6):

    if metric_names_plot[i] != '':
        metric = metrics_all[metric_names_plot[i]] 
        # remove None values
        while None in metric:
            metric.remove(None)
        print(f"{metric_names_plot[i]} has total values of {len(metric)}")

    axis = axs_1[i]

    # figure labels
    label = label_ls[i] 
    #axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
    #     fontsize=label_size, va='top', ha='right')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # plotting the histogram
    if i == 0:
        axis.hist(metric, bin_ls[i], color=c_hist_1, density=True)    
    elif i == 1:
        # distribution tail
        percs = [5e-6, 30, 70, 99.999995]
        percentiles = [np.percentile(metric, per) for per in percs]
        lb, ub = percentiles[2:]
        axis.hist(metric, bin_ls[i], color=c_hist_1, density=True)

        # log axis
        axis.set_xscale('log')
        axis.set_yscale('log')

        axis.set_xlim(lb,ub)
        #ax1_inset.set_ylim(1e-1,1e1)
        #axis.set_ylim(0.5e-1,1e1)

        #ax1_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

        axis.minorticks_off()
        axis.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
    
    elif i == 2:
        axis.hist(metric, bin_ls[i], color=c_hist_1, density=True)
        axis.set_xscale('log')
        axis.set_yscale('log')        

    elif i == 3 or i == 4:
        axis.hist(metric, bin_ls[i], color=c_hist_1, density=True)
        #axis.set_xlim(np.percentile(metric, 1), np.percentile(metric, 99))
        # get kde
        density = sst.gaussian_kde(metric)
        x = np.linspace(0,2,500) 
        axis.plot(x, density(x))

    if i == 0 or i == 1:
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, *params_normal)
        y_tstudent = sst.t.pdf(x, *params_tstudent)
        y_lognorm = lognorm.pdf(x, *params_lognorm)
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')
        axis.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[2], linestyle='dashdot', label = 'Student t')
        axis.plot(x, y_lognorm, linewidth=lwidth, c=c_ls_1[3], linestyle='dotted', label = 'Log-normal')        
    if i == 0:
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

    # set axis limit
    #axis.set_xlim(xlim_ls[i])
    #axis.set_ylim(ylim_ls[i])

    # tick labels
    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    #axis.xaxis.set_minor_locator(AutoMinorLocator())
    #axis.yaxis.set_minor_locator(AutoMinorLocator())

    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    #if i == 0 or i == 1:
    #axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
    axis.set_title(f"{ylabel_ls[i]}", fontsize=axis_size)

# -------------------- Save fig --------------------

print(f"Time: {time.time() - t0}")

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_fig1.pdf", bbox_inches='tight')
#plt.show()

