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
import string

from ast import literal_eval
from os.path import join
from scipy.stats import levy_stable, norm, distributions
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name

pub_font = {'family' : 'serif'}
plt.rc('font', **pub_font)

t0 = time.time()

# ---------------------------

# 3 by 3 template

# colour schemes
cm_type_1 = 'hot'
cm_type_2 = 'RdGy'

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

fig, axis = plt.subplots(1, 1,figsize=(12.5 + 4.5, 9.5/2*3 + 0.5))

# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"
main_path = f"{prepath}/stablefit_all"

"""
Distribution of fitted stable distribution parameters
"""

# load two files of fitted params from Pytorch and Tensorflow
files = []
for mpath in [f"{prepath}/stablefit_all", f"{prepath}/stablefit_all_tf"]:
    for f1 in os.listdir(mpath):
        f1_path = os.path.join(mpath, f1)
        for f2 in os.listdir(f1_path):
            #if 'plfit' in f2:
            if 'stablefit' in f2 and '.csv' in f2:
                plaw_data_path = os.path.join(f1_path, f2)
                files.append(plaw_data_path)

print(f"Total number of weight matrices and convolution tensors: {len(files)}")

# load stable fit params, and weight tensor size and dimension
# sigma_scaled corresponds to D_w^(1/alpha)
metric_names = ['alpha', 'sigma', 'sigma_scaled', 'shap pvalue', 'ks pvalue stable', 'ks pvalue normal',
                'model_name', 'param_shape', 'weight_num', 'wmat_idx', 'idx', 'weight_file', 'dirname']
metrics_all = {}
net_names = []
for metric_name in metric_names:
    metrics_all[metric_name] = []

# weight tensor information 
# files for each weight tensor (method 2)
#for weight_info_name in ["weight_info.csv"]:
for weight_info_name in ["weight_info.csv", "weight_info_tf.csv"]:
    dirname = "stablefit_all_tf" if "_tf" in weight_info_name else "stablefit_all"
    weight_info = pd.read_csv( join(prepath,weight_info_name) )
    for metric_idx in range(6,len(metric_names)-1):
        metric_name = metric_names[metric_idx]
        metrics_all[metric_name] += list(weight_info.loc[:,metric_name])

    metrics_all['dirname'] += [dirname] * len(weight_info.loc[:,'model_name'])

#assert len(files) == len(metrics_all['model_name']), "Mismatch between files and imported data."

# top-1 and top-5 acc
net_names_all = [pd.read_csv(join(prepath,fname)) for fname in ["net_names_all.csv", "net_names_all_tf.csv"]]

metrics_all['top-1'], metrics_all['top-5'] = [], []
# load stablefit alpha and sigma
for ii in tqdm(range(len(metrics_all['param_shape']))):
#for ii in tqdm(range(1)):
    weight_foldername = replace_name(metrics_all['weight_file'][ii], "stablefit") + ".csv"
    dirname = metrics_all['dirname'][ii]
    model_name = metrics_all['model_name'][ii]
    df = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )
    # stablefit params
    alpha, sigma = df.loc[0,'alpha'].item(), df.loc[0,'sigma'].item()
    metrics_all['alpha'].append( alpha )
    metrics_all['sigma'].append( sigma )
    # fitting stats
    shap_stat, ks_pvalue_stable, ks_pvalue_normal = df.loc[0,["shap pvalue","ks pvalue stable","ks pvalue normal"]]
    metrics_all['shap pvalue'].append(shap_stat)
    metrics_all['ks pvalue stable'].append(ks_pvalue_stable)
    metrics_all['ks pvalue normal'].append(ks_pvalue_normal)

    fidx = 0 if "_tf" not in dirname else 1
    database = net_names_all[fidx]
    row_num = database[database.loc[:,"model_name"] == model_name].index.item()
    top_1, top_5 = database[database.loc[:,"model_name"] == model_name].loc[row_num, ['top-1', 'top-5']]
    metrics_all['top-1'].append(top_1)
    metrics_all['top-5'].append(top_5)
    
    param_shape = literal_eval(metrics_all['param_shape'][ii])
    metrics_all['param_shape'][ii] = param_shape
    if len(param_shape)==2:     # fully-connected 
        sigma_scaled = (2*np.sqrt(np.prod(param_shape)))**(1/alpha) * sigma
    elif len(param_shape) == 4:
        if "_tf" in dirname:
            c_in, c_out = param_shape[2:]
            ksize1, ksize2 = param_shape[0:2]
        else:
            c_out, c_in = param_shape[0:2]
            ksize1, ksize2 = param_shape[2:]
        sigma_scaled = (2 * c_out * ksize1 * ksize2 )**(1/alpha) * sigma   # theoretically the most correct
        #sigma_scaled = (2 * c_out * np.sqrt( ksize1 * ksize2 ) )**(1/alpha) * sigma
        #sigma_scaled = (2 * c_in * ksize1 * ksize2 )**(1/alpha) * sigma
    metrics_all['sigma_scaled'].append( sigma_scaled )

print("alpha and sigma loaded!")

for metric_name in metric_names + ['top-1', 'top-5']:
    metrics_all[metric_name] = np.array(metrics_all[metric_name])

# filter weight matrices/tensors with very few entries
min_weight_num = 1000
indices = np.where(metrics_all['weight_num'] >= min_weight_num)
for metric_name in metric_names + ['top-1', 'top-5']:
    if metric_name != 'sample_w':
        metrics_all[metric_name] = metrics_all[metric_name][indices]

print(f"{len(metrics_all[metric_name])}/{len(files)} of tensors are taken into account!")

# -------------------- Plot row 1 --------------------

# histogram and fit plot 
print("Start plotting")

good = 0

xlim_ls = [0.45,2.05]
ylim_ls = [-.5,10] 

axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)

pretrained_acc = metrics_all['top-5']
cmap_bd = [round(np.percentile(pretrained_acc,5)), round(np.percentile(pretrained_acc,95))]
im = axis.scatter(metrics_all['alpha'], metrics_all['sigma_scaled'], 
             c=pretrained_acc, vmin=cmap_bd[0], vmax=cmap_bd[1],
             marker='.', s=marker_size, alpha=0.6, cmap=plt.cm.get_cmap(cm_type_1))

#axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

# colour bar
cbar_ticks = list(range(cmap_bd[0],cmap_bd[1]+1,2))
cbar = fig.colorbar(im, ax=axis, 
                    orientation="vertical",
                    ticks=cbar_ticks)
                    #panchor=False)

cbar.ax.set_yticklabels(cbar_ticks,size=tick_size-3)

# set axis limit
axis.set_xlim(xlim_ls)
axis.set_ylim(ylim_ls)

# tick labels
axis.tick_params(axis='x', labelsize=axis_size - 1)
axis.tick_params(axis='y', labelsize=axis_size - 1)
# minor ticks
axis.xaxis.set_minor_locator(AutoMinorLocator())
axis.yaxis.set_minor_locator(AutoMinorLocator())

axis.set_xlabel(r"$\alpha$", fontsize=axis_size)
axis.set_ylabel(r"$\sigma_w$", fontsize=axis_size)
axis.set_title(f"Top-1 test accuracy", fontsize=axis_size)

good += 1

plt.show()
