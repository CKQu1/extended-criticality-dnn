import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcl
import numpy as np
import time
import torch
import os
import pandas as pd
import string
import sys

from ast import literal_eval
from os.path import join
from scipy.stats import levy_stable, norm, distributions
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator

from pretrained_wfit import replace_name, get_int_power, convert_sigfig
lib_path = os.getcwd()
sys.path.append(f'{lib_path}')
from constants import root_data

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
c_ls_2 = ["peru", "dodgerblue", "limegreen"]
c_ls_3 = ["red", "blue"]
c_ls_4 = ["blue", "red"]
#c_ls_4 = ["darkblue", "darkred"]
c_ls_5 = ['tab:blue', 'tab:orange', 'tab:green']
#c_ls_3 = ["black", "darkgray"]
#c_ls_3 = ["indianred", "dodgerblue"]
c_hist_1 = "tab:blue"
#c_hist_1 = "dimgrey"
c_hist_2 = "dimgrey"

tick_size = 18.5 * 1.5
label_size = 18.5 * 1.5
axis_size = 18.5 * 1.5
legend_size = 14 * 1.5
lwidth = 3.9
text_size = 14 * 1.5
marker_size = 20

label_ls = [f"({letter})" for letter in list(string.ascii_lowercase)]
linestyles = ["solid", "dashed","dashdot"]

params = {'legend.fontsize': legend_size,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size}
plt.rcParams.update(params)


""" 
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
#ax1_inset = fig.add_subplot(gs[5:81, 130:206])
"""

fig = plt.figure(figsize=(24, 16))
gs = mpl.gridspec.GridSpec(490, 740, wspace=0, hspace=0)  
ax1 = fig.add_subplot(gs[0:200, 0:200])
ax2 = fig.add_subplot(gs[0:200, 260:460])
ax3 = fig.add_subplot(gs[0:200, 520:720])
ax4 = fig.add_subplot(gs[290:490, 0:200])
ax5 = fig.add_subplot(gs[290:490, 260:460])
ax6 = fig.add_subplot(gs[290:490, 520:720])
# colorbars
ax4_cbar = fig.add_subplot(gs[290:490, 730:740])

nets_dir = join(root_data, "trained_mlps", "fc3_sgd_fig1")
# net_path1 = join(nets_dir, "fc3_100_100_1fcaad72-5f4c-11ee-b091-246e969a9ce0_mnist_sgd_lr=0.01_bs=1024_epochs=200")
# net_path2 = join(nets_dir, "fc3_200_100_9355aa9e-5f4c-11ee-afb5-246e969a9ce0_mnist_sgd_lr=0.01_bs=1024_epochs=200")
# net_paths = [net_path1, net_path2]

net_alpha100s = [120, 200]
net_alphas = [alpha100/100 for alpha100 in net_alpha100s]

net_paths1 = []
net_paths2 = []
for subdir in next(os.walk(nets_dir))[1]:
    if f'fc3_{net_alpha100s[0]}_100' in subdir and '_mnist_sgd_lr=0.01_bs=1024_epochs=200' in subdir:
        net_paths1.append(join(nets_dir, subdir))
    elif f'fc3_{net_alpha100s[1]}_100' in subdir and '_mnist_sgd_lr=0.01_bs=1024_epochs=200' in subdir:
        net_paths2.append(join(nets_dir, subdir))

# total ensembles taken
ensembles = 5
net_paths1 = net_paths1[:ensembles]
net_paths2 = net_paths2[:ensembles]
assert len(net_paths1) == len(net_paths2)

#quit()

#selected_epoch = 50
total_epoch = 200
selected_epoch = 200
wmat_idx = 0
wmats = []
for net_path in [net_paths1[0], net_paths2[0]]:
    # network info
    net_log = pd.read_csv(join(net_path, "log"))
    model_dims = literal_eval(net_log.loc[0,"model_dims"])

    # load weight matrix
    weights_all = np.load(join(net_path, f"epoch_{selected_epoch}", "weights.npy"))
    start = 0
    for idx in range(wmat_idx + 1):
        start += model_dims[idx] * model_dims[idx+1]
    wmat_size = model_dims[idx+1] * model_dims[idx+2]
    end = start + wmat_size
    wmat = weights_all[start:end]

    wmats.append(wmat)
    del weights_all

# relevant labels
xlabel_ls = [r"$\mathbf{{W}}^{{{}}}$ ".format(wmat_idx + 1), "Epoch", "Epoch"]
#title_ls = ["Right tail", r"$k$", "Test Acc."]
title_ls = ["Right tail", "Test Acc.", r"$\vert\vert \Delta \Theta \vert\vert$"]

for ii, axis in enumerate([ax1, ax2, ax3]):
    axis.set_xlabel(f"{xlabel_ls[ii]}", fontsize=axis_size)
    axis.set_title(f"{title_ls[ii]}", fontsize=axis_size)

    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)    

# -------------------- Figure A --------------------
print("Figure A, B \n")

for ii, net_path in tqdm(enumerate([net_paths1[0], net_paths2[0]])):
    #plaw_fit_file = join(net_path, "wmat-fit-tail", f"wfit-epoch={selected_epoch}-wmat_idx={wmat_idx}.csv")
    #if os.path.isfile(plaw_fit_file):
    #    df = pd.read_csv(plaw_fit_file)            

    wmat = wmats[ii]
    net_log = pd.read_csv(join(net_path, "log"))
    alpha100 = int(net_log.loc[0, "alpha100"])
    alpha = alpha100/100
    net_alphas.append(alpha)

    if alpha100 != 200:        
        import powerlaw as plaw
        from weightwatcher.WW_powerlaw import fit_powerlaw

        # method 1        
        scinote = "{:e}".format(wmat.max())
        e_idx = scinote.find("e")
        integer = float(scinote[:e_idx])
        power = int(scinote[e_idx+1:])   
        #xmin, xmax = integer*10**(power-4), integer*10**(power-1)    
        xmin, xmax = integer*10**(power-4), wmat.max()
        plaw_fit = plaw.Fit(wmat, xmin=xmin, xmax=xmax, fit_method="KS", verbose=False)

        # method 2        
        # plfit = fit_powerlaw(wmat, total_is=len(wmat), plot_id=None, savedir=None,
        #                         plot=False)       
        # alpha, Lambda, xmin, xmax, D, sigma, num_pl_spikes, num_fingers, raw_alpha, status, warning, best_fit_1, Rs_all, ps_all = plfit
        # plaw_fit = plaw.Fit(wmat, xmin=xmin, xmax=xmax, fit_method="KS", verbose=False)

        #plaw_fit.plot_ccdf(ax=ax1, c=c_ls_4[0], linewidth=lwidth, label=rf'$\alpha$ = {round(alpha, 2)}')
        plaw_fit.plot_ccdf(ax=ax1, c=c_ls_4[0], linewidth=lwidth)
        #plaw_fit.power_law.plot_ccdf(ax=ax1, color='k', linestyle='--', label='Power law fit')  
        plaw_fit.power_law.plot_ccdf(ax=ax1, color='k', linestyle='dotted', label='Power law fit')
    else:
        #sns.distplot(wmat, hist=False, color=c_ls_4[1], label=rf'$\alpha$ = {round(alpha, 2)}', ax=ax1)
        sns.distplot(wmat, hist=False, color=c_ls_4[1], ax=ax1, kde_kws={'linestyle':'--', 'linewidth':'width'})

#ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, fontsize=legend_size, frameon=False)

# # -------------------- Figure B (old) --------------------

save_epoch = 10
epochs = [0] + list(range(save_epoch, total_epoch+1, save_epoch))
#epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
net_path = net_paths1[0]
#epochs = [0] + list(range(10, 51, 10))
#epochs = [0, 101, 200]
wmat_idxs = list(range(len(model_dims) - 1))
fitted_alphas = np.zeros([ensembles, len(wmat_idxs), len(epochs)])

# replace = False
# for net_idx in range(ensembles):
#     for eidx, epoch in tqdm(enumerate(epochs)):
#         # load weight matrix
#         weights_all = np.load(join(net_path, f"epoch_{epoch}", "weights.npy"))     
#         start = 0   
#         if os.path.isfile(join(net_path, f"epoch_{epoch}", 'plfit.csv')) and not replace:
#             df_plfit = pd.read_csv(join(net_path, f"epoch_{epoch}", 'plfit.csv'))
#             fitted_alphas[net_idx, :, eidx] = df_plfit.loc[:,'plfit_alpha'] - 1
#         else:
#             dict_df = {'plfit_alpha': [], 'xmin': [], 'xmax': []}
#             for wmat_idx in wmat_idxs:
#                 end = start + model_dims[wmat_idx] * model_dims[wmat_idx+1]
#                 wmat = weights_all[start:end]
#                 start = end      

#                 #xmin = np.quantile(wmat, 0.75)
#                 #plaw_fit = plaw.Fit(wmat, xmin=xmin, verbose=False)

#                 # PL fit
#                 scinote = "{:e}".format(wmat.max())
#                 e_idx = scinote.find("e")
#                 integer = float(scinote[:e_idx])
#                 power = int(scinote[e_idx+1:])   
#                 xmin, xmax = integer*10**(power-3), integer*10**(power-1)
#                 #plaw_fit = plaw.Fit(wmat, xmin=xmin, verbose=False)
#                 plaw_fit = plaw.Fit(wmat, xmin=xmin, xmax=xmax, verbose=False)                
#                 fitted_alphas[net_idx, wmat_idx, eidx] = plaw_fit.alpha - 1   

#                 # save df
#                 dict_df['plfit_alpha'].append(plaw_fit.alpha)
#                 dict_df['xmin'].append(xmin)
#                 dict_df['xmax'].append(xmax) 

#             df_plfit = pd.DataFrame.from_dict(dict_df)              
#             df_plfit.to_csv(join(net_path, f"epoch_{epoch}", 'plfit.csv'))
         

# fitted_alphas_mean = fitted_alphas.mean(0)
# fitted_alphas_std = fitted_alphas.std(0)
# for wmat_idx in wmat_idxs:  
#     ax2.plot(epochs, fitted_alphas_mean[wmat_idx,:], c=c_ls_5[wmat_idx], linewidth=lwidth, label=r"$\mathbf{{W}}^{{{}}}$ ".format(wmat_idx + 1))

#     # ax2.fill_between(epochs, fitted_alphas_mean[wmat_idx,:] - fitted_alphas_std[wmat_idx,:], 
#     #                  fitted_alphas_mean[wmat_idx,:] + fitted_alphas_std[wmat_idx,:],
#     #                  color=c_ls_5[wmat_idx], alpha=0.5)

# ax2.legend(loc='upper left', ncol=1, fontsize=legend_size, frameon=False)

# -------------------- Figure B --------------------
print("Figure B \n")

acc_loss = pd.read_csv(join(net_paths1[0], "acc_loss"))
for ii, net_paths in enumerate([net_paths1, net_paths2]):
    accs = np.zeros([ensembles, acc_loss.shape[0]])
    for net_idx, net_path in enumerate(net_paths):
        acc_loss = pd.read_csv(join(net_path, "acc_loss"))
        accs[net_idx, :] = acc_loss.iloc[:,3]*100

    if ii == 0:
        accs_ht = accs

    net_log = pd.read_csv(join(net_path, "log"))
    alpha100 = int(net_log.loc[0, "alpha100"])
    alpha = alpha100/100

    accs_mean = accs.mean(0)
    accs_std = accs.std(0)
    ax2.plot(list(range(epochs[-1]+1)), accs_mean, 
             c=c_ls_4[ii], linewidth=lwidth, linestyle=linestyles[ii], label=rf'$\alpha$ = {round(alpha, 2)}')

    ax2.fill_between(list(range(epochs[-1]+1)), accs_mean - accs_std, 
                     accs_mean + accs_std,
                     color=c_ls_4[ii], alpha=0.5)   

ax2.set_xlim([-5,max(epochs) + 5])                      

# -------------------- Figure C, D, E, F --------------------
print("Figure C, D, E, F \n")

from path_names import root_data
from NetPortal.models import ModelFactory
from train_supervised import get_data, set_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from UTILS.utils_dnn import compute_dq

#title_ls = [r"$\vert\vert \Delta \Theta \vert\vert$", rf"$\alpha$ = 1", rf"$\alpha$ = 2"]
title_ls = ['Input', rf"$\alpha$ = 1", rf"$\alpha$ = 2"]
axs_3 = [ax4, ax5, ax6]
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
#init_epochs = [0,25,50]
#init_epochs = [50]
#init_epochs = [101]
init_epochs = [200]
l = 1

# -------------------- Figure C -------------------- (plot change of weights)
for ii, net_paths in enumerate([net_paths1, net_paths2]):
    weight_diffs = np.zeros([len(net_paths), len(epochs)])
    for ensemble, net_path in enumerate(net_paths):
        for idx in range(len(epochs)):
            # load weight matrix
            if idx > 0:
                weights_all_cur = np.load(join(net_path, f"epoch_{epochs[idx]}", "weights.npy"))
                weights_all_prev = np.load(join(net_path, f"epoch_{epochs[idx-1]}", "weights.npy"))
                weight_diffs[ensemble,idx] = np.sqrt(((weights_all_cur - weights_all_prev)**2).sum())  

    alpha = net_alphas[ii]
    ax3.plot(epochs[1:], weight_diffs.mean(0)[1:], 
             c=c_ls_4[ii], linestyle=linestyles[ii], linewidth=lwidth, label=rf"$\alpha$={alpha}")                              

    ax3.fill_between(epochs[1:], weight_diffs.mean(0)[1:] - weight_diffs.std(0)[1:], 
                     weight_diffs.mean(0)[1:] + weight_diffs.std(0)[1:],
                     color=c_ls_4[ii], alpha=0.5)

ax3.set_xlim([-5,max(epochs) + 5])

# -------------------- Figure D - F -------------------- (input image + neural representation)
                              

plot_layer, post = True, False
for net_idx, net_path in enumerate([net_paths1[0], net_paths2[0]]):
    # load network
    kwargs = {"architecture": "fc", "activation": net_log.loc[0,"activation"], "dims": model_dims,
              "alpha": None, "g": None, "init_path": net_path, "with_bias": False}

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

    ymax_ls = [1.2, 1.1, 1.2]    
    #for init_idx, init_epoch in enumerate(init_epochs):
    init_epoch = init_epochs[0]
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
        #hidden_layer = StandardScaler().fit_transform(hidden_layer)
        hidden_layer = hidden_layer - hidden_layer.mean(0)
        pca = PCA()
        #pca.fit(hidden_layer)
        X = pca.fit_transform(hidden_layer) # PCs
        eigvals = pca.explained_variance_   # principal axes
        eigvecs = pca.components_

        #quantity = X[0,:]   # top PC
        quantity = eigvecs[0,:]   # top PA

        #d2s = [compute_dq(eigvecs[eidx,:], 2) for eidx in range(eigvecs.shape[0])]
        d2s = [compute_dq(X[eidx,:], 2) for eidx in range(X.shape[0])]
        d2s = np.array(d2s)

    alpha = net_alphas[net_idx]
    
    # eigenvalues (initial version)    
    # eigvals = eigvals/eigvals.sum()
    # eigvals = sorted(eigvals, reverse=True)
    # axs_3[0].plot(list(range(1, len(eigvals)+1)), eigvals, c=c_ls_4[net_idx], linewidth=lwidth, label=rf"$\alpha$={alpha}")


    # principal directions
    # normalize
    quantity = (quantity - quantity.mean()) / quantity.std()
    cmap_bd = [np.percentile(quantity.flatten(), 5), np.percentile(quantity.flatten(), 95)]
    cmap_bd = [np.min(quantity.flatten()), np.max(quantity.flatten())]

    # remove pixels with small acitivity
    #hidden_layer[np.abs(hidden_layer) < 5e-1] = np.NaN
    quantity = quantity.reshape(28,28)    

    # -------------------- Figure D --------------------
    if net_idx == 0:
        colmap = plt.cm.get_cmap(cm_type_2)
        # need to mention the threshold
        #hidden_layer[np.abs(hidden_layer) < 1e-2] = np.NaN
        #colmap.set_bad(color="k")        
        im = axs_3[0].imshow(image.reshape(28,28),   # init_idx
                            vmin=-3, vmax=3,
                            aspect="auto", cmap=colmap)     

    im = axs_3[net_idx+1].imshow(quantity,   # init_idx
                                 vmin=-3, vmax=3,
                                 aspect="auto", cmap=colmap) 

    """
    axs_3[init_idx].axvline(x=13.5, ymin=1, ymax=ymax_ls[init_idx],
                    c='grey', linestyle=":", linewidth=lwidth-1,
                    clip_on=False)
    """

    #axs_3[init_idx].set_title(f"Epoch {init_epoch}")

    # colorbar
    #cbar_ticks = [np.round(cmap_bd[0],1), 0, np.round(cmap_bd[1],1)]
    cbar_ticks = [-3,0,3]
    #if init_idx == 2:
    if net_idx == 1:
        cbar_image = fig.colorbar(im, ax=ax4_cbar, 
                                  cax=ax4_cbar, ticks=cbar_ticks,
                                  orientation="vertical")

        cbar_image.ax.tick_params(labelsize=tick_size-3)
        #cbar.ax.set_yticklabels(cbar_ticks,size=tick_size-3)

    #axs_3[init_idx].set_ylabel(f"Epoch {init_epoch}")

    #im1 = axs_3[0].imshow(image, aspect="auto", cmap=colmap)
    #axs_3[0].set_title("Input image")

    # plot settings
    #axs_3[0].set_ylabel(f"Layer {l + 1}")

    xyticks = np.array([0,27])
    #for col in range(3):
    for col in [0,1,2]:
        axs_3[col].set_xticks(xyticks)
        axs_3[col].set_yticks(xyticks)
        axs_3[col].set_xticklabels(xyticks+1, fontsize=tick_size)
        if col == 0:
            axs_3[col].set_yticklabels(xyticks[::-1]+1, fontsize=tick_size)
        else:
            axs_3[col].set_yticklabels([])

# axs_3[0].spines['top'].set_visible(False)
# axs_3[0].spines['right'].set_visible(False)

#axs_3[0].set_xscale("log"); axs_3[0].set_yscale("log")
#axs_3[0].legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, fontsize=legend_size, frameon=False)
# axs_3[0].set_ylabel(f"Layer {l+1}, Epoch {init_epochs[0]}")
# axs_3[1].set_ylabel(f"Top PD")
# axs_3[2].set_ylabel(f"Top PD")
# axs_3[0].set_xlabel('Epoch')
for ii in range(len(axs_3)):
    axs_3[ii].set_title(title_ls[ii], fontsize=axis_size)

# save figure
print(f"Time: {time.time() - t0}")
fig1_path = "/project/PDLAI/project2_data/figure_ms"
if not os.path.isdir(fig1_path): os.makedirs(fig1_path)
plt.savefig(f"{fig1_path}/mlp_trained.pdf", bbox_inches='tight')

#plt.show()
