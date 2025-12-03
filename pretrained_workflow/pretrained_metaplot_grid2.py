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
#cm_type = 'CMRmap'
#cm_type = 'jet'
cm_type_1 = 'hot'
#cm_type_2 = 'Spectral'
#cm_type_2 = 'bwr'
cm_type_2 = 'RdGy'
c_hist_1 = "dimgrey"
c_ls_1 = ["forestgreen", "coral"]
c_ls_2 = list(mcl.TABLEAU_COLORS.keys())
c_ls_3 = ["red", "blue"]
markers = ["o", "x", "^"]

suptitle_size = 18.5 * 2
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

fig = plt.figure(figsize=(25, 28))
gs = mpl.gridspec.GridSpec(850, 740, wspace=0, hspace=0)   
# main plots 
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[320:470, 100:620])
ax6 = fig.add_subplot(gs[500:700, 0:200])
ax7 = fig.add_subplot(gs[500:700, 260:460])
ax8 = fig.add_subplot(gs[500:700, 520:720])
ax5 = fig.add_subplot(gs[730:850,100:620])
# colorbars
ax1_cbar = fig.add_subplot(gs[0:200, 730:740])
#ax2_cbar = fig.add_subplot(gs[500:700, 210:220])
#ax3_cbar = fig.add_subplot(gs[500:700, 470:480])
ax4_cbar = fig.add_subplot(gs[500:700, 730:740])
# inset
ax1_inset = fig.add_subplot(gs[5:81, 130:206])

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

print(f"{len(metrics_all[metric_name])}/{len(files)} of tensors are taken into account!")

# -------------------- Plot row 1 --------------------

# histogram and fit plot 
print("Start plotting")

good = 0
bin_ls = [1000,250,2500]
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(l + 1) + f"entries ({net.upper()})", r"$\alpha$", r"$\alpha$"] 
ylabel_ls = ["Probability density", "Probability density", r"$\sigma_w$"]
#ylabel_ls = ["Probability Density", r"$\sigma$"]

#xlim_ls = [[-0.3, 0.3], [0.5,2], [0,0.3]]
#ylim_ls = [[0,12.5], [0,3], [0,40]]
xlim_ls = [[-0.3, 0.3], [0.5,2], [0.45,2.05]]
#ylim_ls = [[0,12.5], [0,3], [-1,1000]]
ylim_ls = [[0,12.5], [0,3], [-.5,20]]

axs_1 = [ax1, ax2, ax3]
metric_names_plot = ["sample_w", "alpha", "sigma_scaled"]
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
        axis.hist(metric, bin_ls[i], density=True)
    else:
        pretrained_acc = metrics_all['top-5']
        cmap_bd = [round(np.percentile(pretrained_acc,5)), round(np.percentile(pretrained_acc,95))]
        im = axis.scatter(metrics_all['alpha'], metrics_all['sigma_scaled'], 
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
        print(f"Stable params: {params_stable}")
        x = np.linspace(-1, 1, 1000)
        y_stable = levy_stable.pdf(x, *params_stable)
        y_normal = norm.pdf(x, params_normal[0], params_normal[1])
        axis.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        axis.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')
        axis.legend(loc = 'upper left', fontsize = legend_size, frameon=False)

        # inset plot for the tail
        #lb, ub = 0.05, 0.1
        lb, ub = 0.06, 0.23
        ax1_inset.hist(metric, bin_ls[i], color=c_hist_1, density=True)
        ax1_inset.plot(x, y_normal, linewidth=lwidth, c=c_ls_1[0], linestyle='solid', label = 'Normal')
        ax1_inset.plot(x, y_stable, linewidth=lwidth, c=c_ls_1[1], linestyle='dashed', label = 'Stable')

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

# -------------------- Plot row 2 --------------------

# Plotting for FCN
xlabel_ls = ["Epoch", "Epoch"]
ylabel_ls = [r"$\alpha$", "Accuracy"]
w_path = "/project/phys_DL/project2_data/trained_nets"
net = "fc5_mnist_tanh_epoch550_lr=0.1_bs=128_data_mnist"

# load stable fitting params
# fc5
"""
stable_params = sio.loadmat(f"{w_path}/{net}/W_stable_params_1-550.mat")
#acc_loss = sio.loadmat("fc5_mnist_tanh_epoch550_lr=0.1_bs=128_data_mnist/fc5_mnist_tanh_loss_log.mat")
acc_loss = sio.loadmat(f"{w_path}/{net}/fc5_mnist_tanh_loss_log.mat")
alphas = stable_params['alphas']
betas = stable_params['betas']
sigmas = stable_params['sigmas']
deltas = stable_params['deltas']
"""

# FC4

axs_2 = [ax4, ax5]
fcn_add1 = ax4.twinx()

# selected epochs
init_epochs = [0,5,10]
selected_epochs = list(range(0,11))
# network path
from constants import root_data
fc_net_path = join(root_data,"trained_mlps/debug/fc4_None_None_b14f28b8-b6a0-11ed-a119-b083feccfbe3_mnist_sgd_lr=0.5_bs=256_epochs=200")
print(f"Fig third part path: {fc_net_path}")
net_log = pd.read_csv(join(fc_net_path, "net_log.csv"))
L = net_log.loc[0,"depth"]
model_dims = literal_eval(net_log.loc[0,"model_dims"])
for widx in range(3):
    stable_params = pd.read_csv(join(fc_net_path, f"stablefit_epoch_widx={widx}"))
    alphas = stable_params.loc[:,'alpha']
    sigmas = stable_params.loc[:,'sigma']
    print(f"W {widx} statistics")
    print(alphas)
    axs_2[0].plot(alphas[selected_epochs], linewidth=lwidth, marker=markers[widx], markersize=8.5, c=c_ls_2[widx]) 
    N_eff = np.sqrt(model_dims[widx] * model_dims[widx+1])
    fcn_add1.plot(sigmas[selected_epochs]/(1/(2*N_eff))**(1/alphas[selected_epochs]), 
                 linewidth=lwidth , marker=markers[widx], markersize=8.5, linestyle="--", c=c_ls_2[widx])

    # legend
    axs_2[0].plot([],[], marker=markers[widx], markersize=8.5, c=c_ls_2[widx], label=r'$\mathbf{{W}}^{{{}}}$'.format(widx + 1))

# limit
xlim_ls = [[min(selected_epochs),max(selected_epochs)], [min(selected_epochs),max(selected_epochs)]]
ylim_ls = [[0.9,2.1], [0,115]]

# featured epochs corresponding to hidden layers
for init_epoch in init_epochs:
    axs_2[0].axvline(x=init_epoch, ymin=-0.2, ymax=1,
                     c='grey', linestyle=":", linewidth=lwidth-2,
                     clip_on=False)

    axs_2[1].axvline(x=init_epoch, ymin=0, ymax=1.3,
                     c='grey', linestyle=":", linewidth=lwidth-2,
                     clip_on=False)

for i in range(2):

    axis = axs_2[i]
    # figure labels
    label = label_ls[i + 3] 
    #axis.text(-0.1, 1.2, '%s'%label, transform=axis.transAxes,      # fontweight='bold'
    #     fontsize=label_size, va='top', ha='right')

    #axis.spines['top'].set_visible(False)
    #axis.spines['right'].set_visible(False)  

    axis.tick_params(axis='x', labelsize=axis_size - 1)
    axis.tick_params(axis='y', labelsize=axis_size - 1)
    # minor ticks
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())

    #if i == 1:
    #    axis.set_xlabel(f"{xlabel_ls[i]}", fontsize=axis_size)
    axis.set_ylabel(f"{ylabel_ls[i]}", fontsize=axis_size)

    #if i != 1:
    axis.set_xlim(xlim_ls[i])
    #axis.set_ylim(ylim_ls[i])

    # scientific notation
    if i == 0:
        #axis.xaxis.get_major_formatter().set_scientific(True)
        axis.ticklabel_format(style='sci',axis='x', scilimits=(0,0))

axs_2[0].set_xticklabels([])
yticks = np.round(np.arange(1.0,2.1,0.25),2)
axs_2[0].set_yticks(yticks)
axs_2[0].set_yticklabels(yticks)

axs_2[1].set_xticks(list(range(11)))
axs_2[1].set_xticklabels(list(range(11)))
axs_2[1].set_xlabel("Epoch")

# add full and dotted line labels
axs_2[0].plot([],[], c='k', linewidth=lwidth, linestyle="-", label=r"$\alpha$")
axs_2[0].plot([],[], c='k', linewidth=lwidth, linestyle="--", label=r"$\sigma_w$")
# legend
axs_2[0].legend(loc='upper left', bbox_to_anchor=(1.1, 1),
                ncol=1, fontsize=legend_size, frameon=False)

# plot accuracy and loss
fcn_add2 = ax5.twinx()
acc_loss = pd.read_csv(join(fc_net_path, f"acc_loss"))
print("Train acc")
print(acc_loss.iloc[selected_epochs,1]*100)
print("Test acc")
print(acc_loss.iloc[selected_epochs,3]*100)

# loss
axs_2[1].plot(acc_loss.iloc[selected_epochs,0], c=c_ls_3[0], linewidth=lwidth, label="Train")
axs_2[1].plot(acc_loss.iloc[selected_epochs,2], c=c_ls_3[1], linewidth=lwidth, linestyle="--", label="Test")
#axs_2[1].legend(bbox_to_anchor=(0.9,0.9), fontsize = legend_size, frameon=False)
axs_2[1].legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1,
                fontsize=legend_size, frameon=False)
axs_2[1].set_ylabel("Loss")

# accuracy
fcn_add2.plot(acc_loss.iloc[selected_epochs,1]*100, c=c_ls_3[0], linewidth=lwidth)
fcn_add2.plot(acc_loss.iloc[selected_epochs,3]*100, c=c_ls_3[1], linestyle="--", linewidth=lwidth)
fcn_add2.set_ylabel("Accuracy")
fcn_add2.set_ylim(ylim_ls[1])

# fcn_add1 setting
fcn_add1.set_ylabel(r"$\sigma_w$", fontsize=axis_size)
fcn_add1.set_ylim(0,16)
#fcn_add1.tick_params(axis='x', labelsize=axis_size - 1)
#fcn_add1.tick_params(axis='y', labelsize=axis_size - 1)
fcn_add1.xaxis.set_minor_locator(AutoMinorLocator())
fcn_add1.yaxis.set_minor_locator(AutoMinorLocator())

# print the necessary statistics (plot row 1)
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

# -------------------- Plot row 3 --------------------

from path_names import root_data
from NetPortal.models import ModelFactory
from train_supervised import get_data, set_data
from sklearn.decomposition import PCA
from UTILS.utils_dnn import compute_dq

axs_3 = [ax6, ax7, ax8]
#axs_colbar = [ax2_cbar, ax3_cbar, ax4_cbar]
# load MNIST
image_type = 'mnist'
batch_size = 10000
train_ds, valid_ds = set_data(image_type ,True)
train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
images, labels = next(iter(train_dl))
#image = images[0,:,:].float()/255
image = images[0,:][None,:]
# load one image (leave out zeros for aesthetics)
#image[image==0] = np.NaN

# load network
kwargs = {}
kwargs["architecture"] = "fc"
kwargs["activation"] = net_log.loc[0,"activation"]
kwargs["dims"] = literal_eval(net_log.loc[0,"model_dims"])
kwargs["alpha"] = None
kwargs["g"] = None
kwargs["init_path"] = fc_net_path
kwargs["with_bias"] = False

l = 1
ims = []
hidden_pixels = np.array([])
# getting bounds for cmap_bd
for init_idx, init_epoch in enumerate(init_epochs):
    kwargs["init_epoch"] = init_epoch
    model = ModelFactory(**kwargs)
    model.eval()
    with torch.no_grad():
        preact_layers = model.preact_layer(image)
    hidden_layer = preact_layers[l].detach().numpy().flatten()
    #normalize
    hidden_layer = (hidden_layer - hidden_layer.mean()) / hidden_layer.std()
    hidden_pixels = np.hstack([hidden_pixels, hidden_layer])

    print(init_epoch)   # delete

#cmap_bd = [np.percentile(hidden_pixels,5), np.percentile(hidden_pixels,95)]
#cbar_ticks_2 = np.round(cmap_bd,1)
print("Plotting hidden layers!")

plot_layer, post = False, False
for init_idx, init_epoch in enumerate(init_epochs):
    kwargs["init_epoch"] = init_epoch
    model = ModelFactory(**kwargs)
    model.eval()
    with torch.no_grad():
        if plot_layer:
            if not post:
                hidden_layers = model.preact_layer(image)
            else:
                hidden_layers, output = model.postact_layer(image)
        else:
            if not post:
                hidden_layers = model.preact_layer(images)
            else:
                hidden_layers, output = model.postact_layer(images)
    # selected hidden layer
    hidden_layer = hidden_layers[l].detach().numpy()
    if plot_layer:
        quantity = hidden_layer
        d2s = [compute_dq(quantity.flatten(), 2)]
        d2s = np.array(d2s)
    else:
        # center hidden layer
        hidden_layer = hidden_layer - hidden_layer.mean(0)
        pca = PCA()
        pca.fit(hidden_layer)
        eigvals = pca.explained_variance_
        eigvecs = pca.components_

        #print(eigvals[:2])
        top_pc = eigvecs[0,:]
        quantity = top_pc
        d2s = [compute_dq(eigvecs[eidx,:], 2) for eidx in range(eigvecs.shape[0])]
        d2s = np.array(d2s)

    # normalize
    quantity = (quantity - quantity.mean()) / quantity.std()
    cmap_bd = [np.percentile(quantity.flatten(), 5), np.percentile(quantity.flatten(), 95)]
    cmap_bd = [np.min(quantity.flatten()), np.max(quantity.flatten())]

    # remove pixels with small acitivity
    #hidden_layer[np.abs(hidden_layer) < 5e-1] = np.NaN
    quantity = quantity.reshape(28,28)
    
    colmap = plt.cm.get_cmap(cm_type_2)
    # need to mention the threshold
    #hidden_layer[np.abs(hidden_layer) < 1e-2] = np.NaN
    #colmap.set_bad(color="k")
    im = axs_3[init_idx].imshow(quantity, 
                                vmin=-3, vmax=3,
                                aspect="auto", cmap=colmap) 
              
    #axs_3[init_idx].set_title(f"Epoch {init_epoch}")

    # colorbar
    #cbar_ticks = [np.round(cmap_bd[0],1), 0, np.round(cmap_bd[1],1)]
    cbar_ticks = [-3,0,3]
    if init_idx == 2:
        cbar_image = fig.colorbar(im, ax=ax4_cbar, 
                                  cax=ax4_cbar, ticks=cbar_ticks,
                                  orientation="vertical")

        cbar_image.ax.tick_params(labelsize=tick_size-3)
        #cbar.ax.set_yticklabels(cbar_ticks,size=tick_size-3)

    #axs_3[init_idx].set_ylabel(f"Epoch {init_epoch}")

axs_3[0].set_ylabel(f"Layer {l+1} (Top PC)")

#im1 = axs_3[0].imshow(image, aspect="auto", cmap=colmap)
#axs_3[0].set_title("Input image")

# plot settings

xyticks = np.array([0,27])
for col in range(3):
    axs_3[col].set_xticks(xyticks)
    axs_3[col].set_yticks(xyticks)
    axs_3[col].set_xticklabels(xyticks+1, fontsize=tick_size)
    if col == 0:
        axs_3[col].set_yticklabels(xyticks[::-1]+1, fontsize=tick_size)
    else:
        axs_3[col].set_yticklabels([])

# suptitles
ax2.text(.95, 1.3, "Pretrained CNNs", transform=ax2.transAxes, fontweight='bold',
         fontsize=suptitle_size, va='top', ha='right')
ax4.text(.735, 1.3, "Fully-connected DNNs", transform=ax4.transAxes, fontweight='bold',
         fontsize=suptitle_size, va='top', ha='right')

# -------------------- Save fig --------------------

print(f"Time: {time.time() - t0}")

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
#plt.show()

fig1_path = "/project/PDLAI/project2_data/figure_ms"
#fig1_path = "/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD/figure_ms"
plt.savefig(f"{fig1_path}/pretrained_stablefit_grid2.pdf", bbox_inches='tight')

