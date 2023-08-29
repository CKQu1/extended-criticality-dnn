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

pub_font = {'family' : 'sans-serif'}
plt.rc('font', **pub_font)

t0 = time.time()

# ---------------------------

# 3 by 3 template

# colour schemes
cm_type_1 = 'coolwarm'
cm_type_2 = 'RdGy'
c_ls_1 = ["forestgreen", "coral"]
#c_ls_2 = list(mcl.TABLEAU_COLORS.keys())
#c_ls_2 = ["black", "dimgray", "darkgray"]
#c_ls_2 = ["indianred", "limegreen", "dodgerblue"]
c_ls_2 = ["peru", "dodgerblue", "limegreen"]
c_ls_3 = ["red", "blue"]
#c_ls_3 = ["black", "darkgray"]
#c_ls_3 = ["indianred", "dodgerblue"]
c_hist_1 = "tab:blue"
#c_hist_1 = "dimgrey"
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

fig = plt.figure(figsize=(24, 25))
gs = mpl.gridspec.GridSpec(850, 740, wspace=0, hspace=0)   
# main plots 
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[290:440, 100:620])
ax5 = fig.add_subplot(gs[440:590,100:620])
ax6 = fig.add_subplot(gs[650:, 0:200])
ax7 = fig.add_subplot(gs[650:, 260:460])
ax8 = fig.add_subplot(gs[650:, 520:720])
# colorbars
ax1_cbar = fig.add_subplot(gs[0:200, 730:740])
#ax2_cbar = fig.add_subplot(gs[650:, 210:220])
#ax3_cbar = fig.add_subplot(gs[650:, 470:480])
ax4_cbar = fig.add_subplot(gs[650:, 730:740])
# inset
ax1_inset = fig.add_subplot(gs[5:81, 130:206])

# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"

"""
Featured CNN convolution tensor
"""

#l = 2
#wmat_idx = 4
#net = "alexnet"
l = 1
wmat_idx = 2
net = "vgg11"
w_type = f"{net}_layer_{l}_{wmat_idx}"
param_type = f"{net}_tstudentfit_{l}_{wmat_idx}"
w_path = f"{prepath}/weights_all/{w_type}"
param_path = f"{prepath}/tstudentfit_all/{net}/{param_type}.csv"

metrics_all = {}
metrics_all['sample_w'] = torch.load(w_path).detach().numpy()
params_tstudent = pd.read_csv(param_path).iloc[0,3:7]
params_normal = pd.read_csv(param_path).iloc[0,7:9]

"""
Distribution of fitted tstudent distribution parameters
"""

# load two files of fitted params from Pytorch and Tensorflow
files = []
for mpath in [f"{prepath}/tstudentfit_all", f"{prepath}/tstudentfit_all_tf"]:
    for f1 in os.listdir(mpath):
        f1_path = os.path.join(mpath, f1)
        for f2 in os.listdir(f1_path):
            #if 'plfit' in f2:
            if 'tstudentfit' in f2 and '.csv' in f2:
                plaw_data_path = os.path.join(f1_path, f2)
                files.append(plaw_data_path)

print(f"Total number of weight matrices and convolution tensors: {len(files)}")

# load tstudent fit params, and weight tensor size and dimension

metric_names = ['dgfr', 'scale,', 'loc', 'ks pvalue tstudent',
                'model_name', 'param_shape', 'weight_num', 'wmat_idx', 'idx', 'weight_file', 'dirname']

net_names = []
for metric_name in metric_names:
    metrics_all[metric_name] = []

# weight tensor information 
# files for each weight tensor (method 2)
#for weight_info_name in ["weight_info.csv"]:
for weight_info_name in ["weight_info.csv", "weight_info_tf.csv"]:
    dirname = "tstudentfit_all_tf" if "_tf" in weight_info_name else "tstudentfit_all"
    weight_info = pd.read_csv( join(prepath,weight_info_name) )
    for metric_idx in range(4,len(metric_names)-1):
        metric_name = metric_names[metric_idx]
        metrics_all[metric_name] += list(weight_info.loc[:,metric_name])

    metrics_all['dirname'] += [dirname] * len(weight_info.loc[:,'model_name'])

#assert len(files) == len(metrics_all['model_name']), "Mismatch between files and imported data."

# top-1 and top-5 acc
net_names_all = [pd.read_csv(join(prepath,fname)) for fname in ["net_names_all.csv", "net_names_all_tf.csv"]]

metrics_all['top-1'], metrics_all['top-5'] = [], []
# load tstudentfit alpha and sigma
for ii in tqdm(range(len(metrics_all['param_shape']))):
    weight_foldername = replace_name(metrics_all['weight_file'][ii], "tstudentfit") + ".csv"
    dirname = metrics_all['dirname'][ii]
    model_name = metrics_all['model_name'][ii]
    df = pd.read_csv( join(prepath, dirname, model_name, weight_foldername) )
    # tstudentfit params
    if 'scale,' in df.columns:
        dgfr, scale = df.loc[0,'dgfr'].item(), df.loc[0,'scale,'].item()
    else:
        dgfr, scale = df.loc[0,'dgfr'].item(), df.loc[0,'scale'].item()
    metrics_all['dgfr'].append( dgfr )
    metrics_all['scale,'].append( scale )
    # fitting stats
    ks_pvalue_tstudent = df.loc[0,"ks pvalue tstudent"]
    metrics_all['ks pvalue tstudent'].append(ks_pvalue_tstudent)

    fidx = 0 if "_tf" not in dirname else 1
    database = net_names_all[fidx]
    row_num = database[database.loc[:,"model_name"] == model_name].index.item()
    top_1, top_5 = database[database.loc[:,"model_name"] == model_name].loc[row_num, ['top-1', 'top-5']]
    metrics_all['top-1'].append(top_1)
    metrics_all['top-5'].append(top_5)
    
    # only for Levy-alpha stable
    """
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
    """

print("dgfr and scale loaded!")

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
#bin_ls = [1000,250,2500]
bin_ls = [1000,50,2500]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\nu$", r"$\nu$"] 
ylabel_ls = ["Probability density", "Probability density", r"$\tau$"]
#ylabel_ls = ["Probability Density", r"$\sigma$"]

#xlim_ls = [[-0.3, 0.3], [0.5,2], [0,0.3]]
#ylim_ls = [[0,12.5], [0,3], [0,40]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0.45,2.05]]
#ylim_ls = [[0,12.5], [0,3], [-1,1000]]
ylim_ls = [[0,12.5], [0,3], [-.5,20]]

axs_1 = [ax1, ax2, ax3]
metric_names_plot = ["sample_w", "dgfr", "scale"]
top_n = 'top-5'     # choose Top-1 or Top-5 accuracy
for i in range(3):

    metric = metrics_all[metric_names_plot[i]]   

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
        axis.hist(metric, bin_ls[i], color=c_hist_2, density=True)
    else:
        pretrained_acc = metrics_all['top-5']
        cmap_bd = [round(np.percentile(pretrained_acc,5)), round(np.percentile(pretrained_acc,95))]
        im = axis.scatter(metrics_all['dgfr'], metrics_all['scale'], 
                     c=pretrained_acc, vmin=cmap_bd[0], vmax=cmap_bd[1],
                     marker='.', s=marker_size, alpha=0.6, cmap=plt.cm.get_cmap(cm_type_1))
                     #marker='.', s=45, alpha=0.6, cmap=plt.cm.get_cmap(cm_type_1))

        #axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

        # colour bar
        ax1_cbar.xaxis.set_ticks_position('bottom')
        cbar_ticks = list(range(cmap_bd[0],cmap_bd[1]+1,2))
        cbar = fig.colorbar(im, ax=axis, 
                            cax=ax1_cbar, orientation="vertical",
                            ticks=cbar_ticks)
                            #panchor=False)
        #ax1_cbar.xaxis.set_label_position('top')
        #ax1_cbar.xaxis.set_ticks_position('top')
        cbar.ax.set_yticklabels(cbar_ticks,size=tick_size-3)

    if i == 0:
        print(f"tstudent params: {params_tstudent}")
        x = np.linspace(-1, 1, 1000)
        y_tstudent = levy_stable.pdf(x, *params_tstudent)
        y_normal = norm.pdf(x, params_normal[0], params_normal[1])
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'tstudent')
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

        # inset plot for the tail
        #lb, ub = 0.05, 0.1
        lb, ub = 0.06, 0.23
        ax1_inset.hist(metric, bin_ls[i], color=c_hist_1, density=True)
        ax1_inset.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        ax1_inset.plot(x, y_tstudent, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'tstudent')

        # log axis
        ax1_inset.set_xscale('log')
        ax1_inset.set_yscale('log')

        ax1_inset.set_xlim(lb,ub)
        #ax1_inset.set_ylim(1e-1,1e1)
        ax1_inset.set_ylim(0.5e-1,1e1)

        #ax1_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

        ax1_inset.minorticks_off()
        ax1_inset.set_xticks([])
        ax1_inset.set_yticks([])
        ax1_inset.set_xticklabels([])
        ax1_inset.set_yticklabels([])
        ax1_inset.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)

    # set axis limit
    axis.set_xlim(xlim_ls[i])
    axis.set_ylim(ylim_ls[i])

    # tick labels
    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())

    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    if i == 0 or i == 1:
        #axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
        axis.set_title(f"{ylabel_ls[i]}", fontsize=axis_size)
    if i == 2:
        axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
        axis.set_title(f"{top_n[0].upper() + top_n[1:]} test accuracy", fontsize=axis_size)

    good += 1

# -------------------- Save fig --------------------

print(f"Time: {time.time() - t0}")

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
#fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_tstudentfit_grid.pdf", bbox_inches='tight')    