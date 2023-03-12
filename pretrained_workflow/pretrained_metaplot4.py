import argparse
import scipy.io as sio
import math
import matplotlib
# load latex
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
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
from matplotlib.gridspec import GridSpec
#from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name

plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.rcParams["font.family"] = "Helvetica"

t0 = time.time()

# ---------------------------

# 3 by 3 template

# colour schemes
#cm_type = 'CMRmap'
#cm_type = 'jet'
cm_type = 'hot'
c_hist_1 = "dimgrey"
c_ls_1 = ["forestgreen", "coral"]
c_ls_2 = ['tab:blue','tab:orange','tab:green']
#c_ls_3 = ['dodgerblue', 'darkorange']
#c_ls_3 = ['dodgerblue', 'lightcoral']
c_ls_3 = ["red", "blue"]

inset_width, inset_height = 0.9/9, 1.4/10

tick_size = 18.5
label_size = 18.5
axis_size = 18.5
legend_size = 14
linewidth = 0.8
text_size = 14
mark = 1/3
width = 0.26
#height = 0.28
height = 0.33
#height = 1
#label_ls = ['(a)', '(b)', '(c)']
label_ls = [f"({letter})" for letter in list(string.ascii_lowercase)]
linestyle_ls = ["solid", "dashed","dashdot"]

#figure(figsize=(12.5,3*3.5)) #3 by 3
#figure(figsize=(12.5,1*3.5 - 0.3)) #1 by 3
fig = figure(figsize=(12.5,2*3.5 + 2.5)) #1 by 3

axes_list = []
axes_inset_list = []

# ---------------------------

prepath = "/project/PDLAI/project2_data/pretrained_workflow"
main_path = f"{prepath}/stablefit_all"

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
param_type = f"{net}_stablefit_{l}_{wmat_idx}"
w_path = f"{prepath}/weights_all/{w_type}"
param_path = f"{prepath}/stablefit_all/{net}/{param_type}.csv"

metrics_all = {}
metrics_all['sample_w'] = torch.load(w_path).detach().numpy()
params_stable = pd.read_csv(param_path).iloc[0,3:7]
params_normal = pd.read_csv(param_path).iloc[0,7:9]

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
net_names = []
for metric_name in metric_names:
    metrics_all[metric_name] = []

# files for each weight tensor (method 1)
"""
for file_idx in range(len(files)):
    f = files[file_idx]
    df = pd.read_csv(f"{f}")
    for metric_idx in range(len(metric_names)):
        metric_name = metric_names[metric_idx]
        metrics_all[metric_name].append(df.loc[:,metric_name].item())
"""

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

# only keep fully-connected weight matrices
"""
indices = [i for i in range(len(metrics_all['param_shape'])) if len(metrics_all['param_shape'][i]) == 2]
for metric_name in metric_names:
    if metric_name != 'sample_w':
        metrics_all[metric_name] = metrics_all[metric_name][indices]
"""

print(f"{len(metrics_all[metric_name])}/{len(files)} of tensors are taken into account!")

# histogram and fit plot ---------------------------

print("Start plotting")

good = 0
#bin_ls = [2000,250,2500]
bin_ls = [1000,250,2500]
#bin_ls = [750, 100,200]
#title_ls = [r"Pareto index $\alpha$", "Powerlaw/lognormal \n log likelihood ratio", "Powerlaw/exponential ratio \n log likelihood ratio"]
#title_ls = [r"Stability parameter $\alpha$", "KS test p-value (stable)", "KS test p-value (normal)"]
#title_ls = [r"Stability parameter $\alpha$", "KS test p-value ratio \n (stable/normal)", "Alexnet weight entries"]
#xlabel_ls = [r"$\mathbf{W}^4$ entries (Alexnet)", r"$\alpha$", r"$D_w^{1/\alpha}$"]
#xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\sigma$"]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\alpha$"] 
#ylabel_ls = ["Probability Density", r"$D_w^{1/\alpha}$"]
ylabel_ls = ["Probability Density", r"$\sigma_w$"]
#ylabel_ls = ["Probability Density", r"$\sigma$"]

#xlim_ls = [[-0.3, 0.3], [0.5,2], [0,0.3]]
#ylim_ls = [[0,12.5], [0,3], [0,40]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0.45,2.05]]
#ylim_ls = [[0,12.5], [0,3], [-1,1000]]
ylim_ls = [[0,12.5], [0,3], [0,20]]

metric_names_plot = ["sample_w", "alpha", "sigma_scaled"]
for i in range(3):

    metric = metrics_all[metric_names_plot[i]]   

    #axis = plt.axes([mark*(i%3), 0, width, height])
    axis = plt.axes([mark*(i%3), 0.5, width, height])

    # figure labels
    label = label_ls[i] 
    axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
         fontsize=label_size, va='top', ha='right')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # plotting the histogram
    if i == 0:
        axis.hist(metric, bin_ls[i], color=c_hist_1, density=True)
    elif i == 1:
        axis.hist(metric, bin_ls[i], density=True)
    else:
        #torch_indices = [i for i in range(len(metrics_all['dirname'])) if "_tf" not in metrics_all['dirname'][i]]
        #tf_indices = [i for i in range(len(metrics_all['dirname'])) if "_tf" in metrics_all['dirname'][i]]
        #conv_indices = [i for i in range(len(metrics_all['dirname'])) if len(metrics_all['param_shape'][i]) == 4]
        #fc_indices = [i for i in range(len(metrics_all['dirname'])) if len(metrics_all['param_shape'][i]) == 2]
        #axis.plot(metrics_all['alpha'][conv_indices], metrics_all['sigma_scaled'][conv_indices], '.', markersize=3.5, label="Conv2d")
        #axis.plot(metrics_all['alpha'][fc_indices], metrics_all['sigma_scaled'][fc_indices], '.', markersize=3.5, alpha=0.65, label="Linear")

        # im = ax.scatter(x , y , z, c=thetas, vmin=cmap_bd[0], vmax=cmap_bd[1], marker='.', s=4, alpha=1, cmap=cm)
        pretrained_acc = metrics_all['top-5']
        cmap_bd = [round(np.percentile(pretrained_acc,5)), round(np.percentile(pretrained_acc,95))]
        im = axis.scatter(metrics_all['alpha'], metrics_all['sigma_scaled'], 
                     c=pretrained_acc, vmin=cmap_bd[0], vmax=cmap_bd[1],
                     marker='.', s=12, alpha=0.6, cmap=plt.cm.get_cmap(cm_type))
                     #marker='.', s=45, alpha=0.6, cmap=plt.cm.get_cmap(cm_type))

        #axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

        # colour bar
        cbar_ax = plt.axes([0.93, 0.5, 0.012, height])
        cbar_ticks = list(range(cmap_bd[0],cmap_bd[1]+1,2))
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks,size=tick_size-3)

    if i == 0:
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, params_normal[0], params_normal[1])
        axis.plot(x, y_normal, linewidth=1.8*1.8, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=1.8*1.8, c=c_ls_1[1], linestyle='dashed', label = 'Stable')
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)
        # inset plot for the tail
        #axis_inset = plt.axes([1 - 1.25/10, 1 - 0.85/3, 1.1/9, 1.6/10], xscale='log', yscale='log')
        axis_inset = plt.axes([1.75/10, 1 - 0.85/3, inset_width, inset_height], xscale='log', yscale='log')
        #lb, ub = 0.05, 0.1
        lb, ub = 0.06, 0.23
        axis_inset.hist(metric, bin_ls[i], color=c_hist_1, density=True)
        axis_inset.plot(x, y_normal, linewidth=1.5*1.8, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis_inset.plot(x, y_stable, linewidth=1.5*1.8, c=c_ls_1[1], linestyle='dashed', label = 'Stable')

        axis_inset.set_xlim(lb,ub)
        #axis_inset.set_ylim(1e-1,1e1)
        axis_inset.set_ylim(0.5e-1,1e1)

        axis_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

        axis_inset.minorticks_off()

        #axis_inset.set_xticks([lb, ub])
        #axis_inset.set_yticks([0])
        #axis_inset.set_xticklabels([0.05,0.1])
        axis_inset.set_xticklabels([])
        axis_inset.set_yticklabels([])

    # set axis limit

    axis.set_xlim(xlim_ls[i])
    axis.set_ylim(ylim_ls[i])

    # tick labels
    #axis.set_xticks([-0.75, 0, 0.75])
    #axis.set_xticks(ytick_ls[i])
    #axis.set_xticklabels([-0.75, 0, 0.75], fontsize=axis_size - 1)
    #axis.set_xticklabels(ytick_ls[i], fontsize=axis_size - 1)
    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)

    # minor ticks
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())

    # ax title
    #axis.set_title(f"{title_ls[i]}", fontsize=axis_size)
    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    if i == 0:
        axis.set_ylabel(f"{ylabel_ls[0]}", fontsize=axis_size)
    if i == 2:
        axis.set_ylabel(f"{ylabel_ls[1]}", fontsize=axis_size)

    # xaxis label
    #axis.set_xlabel(f'{net_names[i]}', fontsize=axis_size)

    # figure labels
    """
    if i // 3 == 0:
        label = label_ls[i] 
        axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,
             fontsize=label_size, fontweight='bold', va='top', ha='right')
    """

    good += 1

# Plotting for FCN -----------------------------------------------------------------------------------

mlp_widx = 2    # the index for the weight matrix
# choose epochs
epoch = 210
#epoch = 500
epoch_list = [epoch]
#epoch_list = [209, 210, 500]
#epoch_list = [200, 250, 500]
weight_list = []
xlabel_ls = [r'$\mathbf{{W}}^{{{}}}$'.format(mlp_widx + 1) + " entries (FC5)", "Epoch", "Epoch"]
#ylabel_ls = ["Probability Density", r"$\alpha$", r"$D_w^{1/\alpha}$"]
ylabel_ls = ["Probability Density", r"$\alpha$", "Accuracy"]
#w_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/trained_nets"
w_path = "/project/phys_DL/project2_data/trained_nets"
net = "fc5_mnist_tanh_epoch550_lr=0.1_bs=128_data_mnist"

# load weight data
for i in range(len(epoch_list)):
    data = sio.loadmat(f"{w_path}/{net}/weights/model_" + str(epoch_list[i]) + "_sub_loss_w.mat")
    weight_list.append(data['sub_weights'][0])
#weights = data['sub_weights']

# load stable fitting params
stable_params = sio.loadmat(f"{w_path}/{net}/W_stable_params_1-550.mat")
#acc_loss = sio.loadmat("fc5_mnist_tanh_epoch550_lr=0.1_bs=128_data_mnist/fc5_mnist_tanh_loss_log.mat")
acc_loss = sio.loadmat(f"{w_path}/{net}/fc5_mnist_tanh_loss_log.mat")
alphas = stable_params['alphas']
betas = stable_params['betas']
sigmas = stable_params['sigmas']
deltas = stable_params['deltas']

# limit
xlim_ls = [[-0.01, 0.01], [0,550], [0,550]]
#ylim_ls = [[0,2.05], [0,0.0032], [0,250]]
#ylim_ls = [[0,250], [0,2.05], [-0.2,8.0]]
ylim_ls = [[0,250], [0,2.05], [0,105]]

fcn_axs = [] 
for i in range(3):

    axis = plt.axes([mark*(i%3), 0, width, height])

    # figure labels
    label = label_ls[i + 3] 
    axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
         fontsize=label_size, va='top', ha='right')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)  

    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)
    # minor ticks
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    # ax title
    #axis.set_title(f"{title_ls[i]}", fontsize=axis_size)
    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    #if i != 1 or i != 2:
    if i == 0:
        axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)
    
    #if i != 1:
    axis.set_xlim(xlim_ls[i])
    axis.set_ylim(ylim_ls[i])

    # scientific notation
    if i == 0:
        #axis.xaxis.get_major_formatter().set_scientific(True)
        axis.ticklabel_format(style='sci',axis='x', scilimits=(0,0))

    fcn_axs.append(axis)  

# fcn_axs[1] double-labeled y-axis
fcn_add = fcn_axs[1].twinx()
#fcn_add.set_ylabel(r"$D_w^{1/\alpha}$", fontsize=axis_size)
fcn_add.tick_params(axis='x', labelsize=axis_size - 1)
fcn_add.tick_params(axis='y', labelsize=axis_size - 1)
# minor ticks
fcn_add.xaxis.set_minor_locator(AutoMinorLocator())
fcn_add.yaxis.set_minor_locator(AutoMinorLocator())

for idx in range(3):
    fcn_axs[1].plot(alphas[idx][:], linewidth=1.5*1.8, c=c_ls_2[idx], label=r'$\mathbf{{W}}^{{{}}}$'.format(idx + 1))
    #fcn_axs[1].plot(sigmas[idx][:], label=r'$\mathbf{W}^{{{}}}$'.format(idx + 1))
    # metrics_all['sigma'] = metrics_all['sigma']/(1/(2*784))**(1/metrics_all['alpha'])
    #fcn_axs[1].plot(sigmas[idx][:], linewidth=1.5*1.5, label=r'$W^{{{}}}$'.format(idx + 1))
    #fcn_axs[2].plot(sigmas[idx][:]/(1/(2*784))**(1/alphas[idx][:]), linewidth=1.5*1.8, label=r'$\mathbf{{W}}^{{{}}}$'.format(idx + 1))
    fcn_add.plot(sigmas[idx][:]/(1/(2*784))**(1/alphas[idx][:]), linewidth=1.5*1.8, linestyle="--", c=c_ls_2[idx])
# add full and dotted line labels
fcn_axs[1].plot([],[], c='k', linewidth=1.5*1.8, linestyle="-", label=r"$\alpha$")
fcn_axs[1].plot([],[], c='k', linewidth=1.5*1.8, linestyle="--", label=r"$\sigma_w$")

# plot accuracy
fcn_axs[2].plot(acc_loss['training_history'][:,1], c=c_ls_3[0], linewidth=1.5*1.8, label="Train Acc.")
fcn_axs[2].plot(acc_loss['testing_history'][:,1], c=c_ls_3[1], linewidth=1.5*1.8, linestyle="--", label="Test Acc.")
fcn_axs[2].legend(bbox_to_anchor=(1.1,0.9), fontsize = legend_size, frameon=False)

# stable fit
#binsize = 500
binsize = 750
#epoch = 210
weights = weight_list[0][mlp_widx].flatten()

# featured alpha
fcn_axs[1].plot(epoch, alphas[mlp_widx][epoch - 1], 'r.', markersize=12)
# featured sigma
#fcn_axs[1].plot(epoch, sigmas[mlp_widx][epoch - 1]*(2*784)**(1/alphas[mlp_widx][epoch - 1]), 'r.', markersize=12, label='__nolegend__')
fcn_axs[1].legend(loc = 'upper left', fontsize = legend_size, frameon=False)

fcn_axs[0].hist(weight_list[0][mlp_widx].flatten(), binsize, color=c_hist_1, density=True)

#alpha,beta,loc,scale = alphas[i//3][epoch-1],betas[i//3][epoch-1], deltas[i//3][epoch-1],sigmas[i//3][epoch-1]
alpha,beta,loc,scale = alphas[2][epoch-1],betas[2][epoch-1], deltas[2][epoch-1],sigmas[2][epoch-1]
limit = max(abs(max(weights)),abs(min(weights)))
x = np.linspace(-limit, limit, num=2000)
y_stable = levy_stable(alpha,beta,loc,scale).pdf(x)
# normal fit
mu, sigma_norm = distributions.norm.fit(weights)
y_normal = norm.pdf(x, mu, sigma_norm)

fcn_axs[0].plot(x, y_normal, linewidth=1.8*1.8, c=c_ls_1[0], linestyle='solid', label = 'Normal')
fcn_axs[0].plot(x, y_stable, linewidth=1.8*1.8, c=c_ls_1[1], linestyle='dashed', label = 'Stable')

# scientific notation
#fcn_axs[0].xaxis.get_major_formatter().set_scientific(True)

# inset plot 2
#axis_inset = plt.axes([1 - 1.25/10, 1.25/6, 1.1/9, 1.6/10], xscale='log', yscale='log')
axis_inset = plt.axes([1.75/10, 1.25/6, inset_width, inset_height], xscale='log', yscale='log')
#lb, ub = 0.003, 0.006
lb, ub = 0.003, 0.0065
axis_inset.hist(weights, binsize, color=c_hist_1, density=True)
axis_inset.plot(x, y_normal, linewidth=1.5*1.8, c=c_ls_1[0], linestyle='solid', label = 'Normal')
axis_inset.plot(x, y_stable, linewidth=1.5*1.8, c=c_ls_1[1], linestyle='dashed', label = 'Stable')

axis_inset.set_xlim(lb,ub)
#axis_inset.set_ylim(5e-1,1e2)
axis_inset.set_ylim(8e-1,3e2)

axis_inset.minorticks_off()

#axis_inset.set_xticks([lb, ub])

#axis_inset.set_xticklabels([0.05,0.1])
axis_inset.set_xticklabels([])
axis_inset.set_yticklabels([])

axis_inset.tick_params(axis='both', which='major', labelsize=tick_size - 4)

# print the necessary statistics

metric_name = 'alpha'
thresholds = 1.9
count = len(metrics_all[metric_name])
percentage = sum(metrics_all[metric_name] < thresholds)
print(f"{metric_name} smaller than {thresholds} percentage: {percentage/count}")

thresholds = 2.5e-2
percentage = sum(metrics_all["shap pvalue"] < thresholds)
print(f"Shapiro Wilk test p-value smaller than {thresholds} percentage: {percentage/count}")

thresholds = 1
percentage = sum(metrics_all["ks pvalue stable"]/metrics_all["ks pvalue normal"] > thresholds)
print(f"stable/normal Kolmogorov-Smirnov test pvalue ratio greater than {thresholds} percentage: {percentage/count}")

print(f"Time: {time.time() - t0}")

plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
#fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_stablefit_torch7.pdf", bbox_inches='tight')

