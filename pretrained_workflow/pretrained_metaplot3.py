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

from scipy.stats import levy_stable, norm, distributions

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplot, title, axis, xlim, ylim, gca, xticks, yticks, xlabel, ylabel, plot, legend, gcf, cm # colorbar
from matplotlib.ticker import AutoMinorLocator

plt.rcParams["font.family"] = "serif"     # set plot font globally
#plt.rcParams["font.family"] = "Helvetica"

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import axes3d

# colorbar scheme
from matplotlib.cm import coolwarm

from multiprocessing import Pool
from functools import partial

t0 = time.time()

# ---------------------------

# 3 by 3 template

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
figure(figsize=(12.5,2*3.5 + 2.5)) #1 by 3

axes_list = []
axes_inset_list = []

# ---------------------------

#main_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow/plfit_all"
#prepath = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/pretrained_workflow"
prepath = "/project/PDLAI/project2_data/pretrained_workflow"
main_path = f"{prepath}/stablefit_all"

# sample weight matrix
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

#net_ls = [net[0] for net in os.walk(main_path)][1:]
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

metric_names = ['alpha', 'sigma']
net_names = []
for metric_name in metric_names:
    metrics_all[metric_name] = []

for file_idx in range(len(files)):

    f = files[file_idx]
    df = pd.read_csv(f"{f}")
    for metric_idx in range(len(metric_names)):
        metric_name = metric_names[metric_idx]
        metrics_all[metric_name].append(df.loc[:,metric_name].item())

for metric_name in metric_names:
    metrics_all[metric_name] = np.array(metrics_all[metric_name])

# convert to D_w^(1/alpha)
#metrics_all['sigma'] = metrics_all['sigma']/(1/(2*784))**(1/metrics_all['alpha'])    # THIS IS ACTUALLY INCORRECT SINCE THE SIZES ARE DIFFERENT
#metrics_all['sigma'] = metrics_all['sigma'][metrics_all['sigma'] <= 20]

metric_names = list(metrics_all.keys())

# histogram and fit plot ---------------------------

print("Start plotting")

good = 0
#bin_ls = [2000,250,2500]
bin_ls = [1000,250,2500]
#bin_ls = [750, 100,200]
#title_ls = [r"Pareto index $\alpha$", "Powerlaw/lognormal \n log likelihood ratio", "Powerlaw/exponential ratio \n log likelihood ratio"]
#title_ls = [r"Stability parameter $\alpha$", "KS test p-value (stable)", "KS test p-value (normal)"]
#title_ls = [r"Stability parameter $\alpha$", "KS test p-value ratio \n (stable/normal)", "Alexnet weight entries"]
#title_ls = [r"Stability parameter $\alpha$", r"$D_w^{1/\alpha}$", "Alexnet weight entries"]
#xlabel_ls = [r"$\mathbf{W}^4$ entries (Alexnet)", r"$\alpha$", r"$D_w^{1/\alpha}$"]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\sigma$"] 
#xlabel_ls = [r"$\alpha$", r"$D_w^{1/\alpha}$", r"$\mathbf{W}$ (Alexnet)"]
ylabel_ls = ["Probability Density"]
#xlim_ls = [[0.5,2], [0.5,2.5], [-0.25, 0.25]]
#xlim_ls = [[0.5,2], [0.5,2.5], [-0.2, 0.2]]
#xlim_ls = [[-0.2, 0.2], [0.5,2], [0,20]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0,0.3]]
#ylim_ls = [[0,15], [0,3], [0,1.2]]
ylim_ls = [[0,12.5], [0,3], [0,40]]
#xlim_ls = [[0,2], [0,7e2], [0, 7e2]]
#ylim_ls = [[0, 0.7], [0, 7e-4], [0, 5e-4]]
#ylim_ls = [[0, 2.7], [0, 8e-1], [0, 9e-2]]
#ytick_ls = [np.arange(2,12 + 1,2), np.arange(-1.5e4,0 + 1,0.5e4), np.arange(-2e4,2e4 + 1,1e4)]

for i in range(len(metric_names)):

    metric = metrics_all[metric_names[i]]   

    #axis = plt.axes([mark*(i%3), 0, width, height])
    axis = plt.axes([mark*(i%3), 0.5, width, height])

    # figure labels
    label = label_ls[i] 
    axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
         fontsize=label_size, va='top', ha='right')

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # plotting the histogram
    #if i == 0 or i == 2:
    axis.hist(metric, bin_ls[i], density=True)
    """
    else:
        #count, bins_count = np.histogram(metric, bins=bin_ls[i])
        #pdf = count / sum(count)
        #cdf = np.cumsum(pdf)
        #axis.plot(bins_count[1:], cdf)
        axis.hist(metric, bin_ls[i], density=True)
    """

    if i == 0:
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, params_normal[0], params_normal[1])
        axis.plot(x, y_normal, linewidth=1.8*1.8, c="tab:green", linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=1.8*1.8, c="tab:orange", linestyle='dashed', label = 'Stable')
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)
        # inset plot for the tail
        #axis_inset = plt.axes([1 - 1.25/10, 1 - 0.85/3, 1.1/9, 1.6/10], xscale='log', yscale='log')
        axis_inset = plt.axes([1.75/10, 1 - 0.85/3, inset_width, inset_height], xscale='log', yscale='log')
        #lb, ub = 0.05, 0.1
        lb, ub = 0.06, 0.23
        axis_inset.hist(metric, bin_ls[i], density=True)
        axis_inset.plot(x, y_normal, linewidth=1.5*1.8, c="tab:green", linestyle='solid', label = 'Normal')
        axis_inset.plot(x, y_stable, linewidth=1.5*1.8, c="tab:orange", linestyle='dashed', label = 'Stable')

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
        axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)

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
    if i != 2:
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
    fcn_axs[1].plot(alphas[idx][:], linewidth=1.5*1.8, label=r'$\mathbf{{W}}^{{{}}}$'.format(idx + 1))
    #fcn_axs[1].plot(sigmas[idx][:], label=r'$\mathbf{W}^{{{}}}$'.format(idx + 1))
    # metrics_all['sigma'] = metrics_all['sigma']/(1/(2*784))**(1/metrics_all['alpha'])
    #fcn_axs[1].plot(sigmas[idx][:], linewidth=1.5*1.5, label=r'$W^{{{}}}$'.format(idx + 1))
    #fcn_axs[2].plot(sigmas[idx][:]/(1/(2*784))**(1/alphas[idx][:]), linewidth=1.5*1.8, label=r'$\mathbf{{W}}^{{{}}}$'.format(idx + 1))
    fcn_add.plot(sigmas[idx][:]/(1/(2*784))**(1/alphas[idx][:]), linewidth=1.5*1.8, linestyle="--")
fcn_axs[1].legend(loc = 'upper left', fontsize = legend_size, frameon=False)

# plot accuracy
fcn_axs[2].plot(acc_loss['training_history'][:,1], linewidth=1.5*1.8, label="Train Acc.")
fcn_axs[2].plot(acc_loss['testing_history'][:,1], linewidth=1.5*1.8, linestyle="--", label="Test Acc.")
fcn_axs[2].legend(loc = 'best', fontsize = legend_size, frameon=False)

# stable fit
#binsize = 500
binsize = 750
#epoch = 210
weights = weight_list[0][mlp_widx].flatten()

# feature
fcn_axs[1].plot(epoch, alphas[mlp_widx][epoch - 1], 'r.', markersize=12)
fcn_axs[1].plot(epoch, sigmas[mlp_widx][epoch - 1]*(2*784)**(1/alphas[mlp_widx][epoch - 1]), 'r.', markersize=12, label='__nolegend__')
fcn_axs[1].legend(loc = 'upper left', fontsize = legend_size, frameon=False)

fcn_axs[0].hist(weight_list[0][mlp_widx].flatten(), binsize, density=True)

#alpha,beta,loc,scale = alphas[i//3][epoch-1],betas[i//3][epoch-1], deltas[i//3][epoch-1],sigmas[i//3][epoch-1]
alpha,beta,loc,scale = alphas[2][epoch-1],betas[2][epoch-1], deltas[2][epoch-1],sigmas[2][epoch-1]
limit = max(abs(max(weights)),abs(min(weights)))
x = np.linspace(-limit, limit, num=2000)
y_stable = levy_stable(alpha,beta,loc,scale).pdf(x)
# normal fit
mu, sigma_norm = distributions.norm.fit(weights)
y_normal = norm.pdf(x, mu, sigma_norm)

fcn_axs[0].plot(x, y_normal, linewidth=1.8*1.8, c="tab:green", linestyle='solid', label = 'Normal')
fcn_axs[0].plot(x, y_stable, linewidth=1.8*1.8, c="tab:orange", linestyle='dashed', label = 'Stable')

# scientific notation
#fcn_axs[0].xaxis.get_major_formatter().set_scientific(True)

# inset plot 2
#axis_inset = plt.axes([1 - 1.25/10, 1.25/6, 1.1/9, 1.6/10], xscale='log', yscale='log')
axis_inset = plt.axes([1.75/10, 1.25/6, inset_width, inset_height], xscale='log', yscale='log')
#lb, ub = 0.003, 0.006
lb, ub = 0.003, 0.0065
axis_inset.hist(weights, binsize, density=True)
axis_inset.plot(x, y_normal, linewidth=1.5*1.8, c="tab:green", linestyle='solid', label = 'Normal')
axis_inset.plot(x, y_stable, linewidth=1.5*1.8, c="tab:orange", linestyle='dashed', label = 'Stable')

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

#thresholds = [2, 0.05, 0.05]
#thresholds = [1.95, 1]
#thresholds = [1.90, 0.025]
thresholds = 1.90
#for idx in range(len(metric_names)):
idx = 1
metric_name = metric_names[idx]
print(metric_name)
print(sum(metrics_all[metric_name] <= thresholds)/len(metrics_all[metric_name]))
#print(sum(metrics_all[metric_name] >= thresholds[idx])/len(metrics_all[metric_name]))


print(f"Time: {time.time() - t0}")

plt.tight_layout()
plt.show()

fig1_path = "/project/phys_DL/project2_data/figure_ms"
#fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
#plt.savefig(f"{fig1_path}/pretrained_stablefit_torch4.pdf", bbox_inches='tight')

