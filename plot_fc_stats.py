import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from ast import literal_eval
from os.path import join
from sklearn.decomposition import PCA

from path_names import root_data
from NetPortal.models import ModelFactory
from train_supervised import get_data, set_data
from utils_dnn import compute_dq

#cm_type_2 = 'seismic'
#cm_type_2 = "jet"
#cm_type_2 = "Spectral"
cm_type_2 = "bwr"
colmap = plt.cm.get_cmap(cm_type_2).copy()

def plot_aa(nidx, post=False, plot_layer=False, root_data="/project/PDLAI/project2_data/trained_mlps/debug"):
    global stablefit, l, depth, net_ls, acc_loss, postact_layers, image
    global hidden_layers, hidden_layer
    global eigvecs, eigvals, top_pc

    post = literal_eval(post) if isinstance(post, str) else post
    nidx = int(nidx)
    net_ls = next(os.walk(root_data))[1]
    net_ls = [join(root_data, net_ls[idx]) for idx in range(len(net_ls)) if "_None_None_" in net_ls[idx]]
    for net_path in net_ls:
        print(net_path)
    print(f"Total networks {len(net_ls)}")

    # selected network info
    net_log = pd.read_csv(join(net_ls[nidx], "net_log.csv"))
    L = net_log.loc[0,"depth"]

    fig, axs = plt.subplots(L, 3,sharex = False,sharey=False,figsize=(9.5/2*L,7.142/2*3))
    #net_ls = next(os.walk(root_data))[1]
    print(net_ls[nidx])

    acc_loss = pd.read_csv(join(net_ls[nidx], "acc_loss"))
    #selected_epochs = list(range(acc_loss.shape[0]))
    selected_epochs = list(range(26))

    # init epochs
    #init_epochs = [0,5,100]
    init_epochs = [0,5,10]

    # ---------- Row 1 ----------

    axs[0,0].plot(acc_loss.iloc[selected_epochs,1], label="train acc")
    axs[0,0].plot(acc_loss.iloc[selected_epochs,3], label="test acc")

    print(f"Max test accuracy: {acc_loss.iloc[selected_epochs,3].max()}")
    log = pd.read_csv(join(net_ls[nidx], "log"))
    depth = log.loc[0,"depth"]
    for l in range(depth):
        stablefit = pd.read_csv(join(net_ls[nidx], f"stablefit_epoch_widx={l}"))
        #print(f"l = {l}, stablefit params {stablefit.iloc[-1,:]}")    
        axs[0,1].plot(stablefit.loc[selected_epochs,'alpha'], label=f"{l+1}")
    for init_epoch in init_epochs:
        axs[0,1].axvline(x = init_epoch, c='k')

    axs[0,0].legend()
    axs[0,1].legend()


    # ---------- Row 2 ----------

    # load MNIST
    image_type = 'mnist'
    batch_size = 10000
    train_ds, valid_ds = set_data(image_type ,True)
    train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)#, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    images, labels = next(iter(train_dl))
    image = images[0,:][None,:]
    cmap_bd = [image.min().item(), image.max().item()]
    
    im = axs[0,2].imshow(image.reshape(28,28), vmin=cmap_bd[0], vmax=cmap_bd[1],
                                               aspect="auto", cmap=colmap)
    cbar = fig.colorbar(im, ax=axs[0,2], 
                        orientation="vertical")

    # load network
    kwargs = {}
    kwargs["architecture"] = "fc"
    kwargs["activation"] = net_log.loc[0,"activation"]
    kwargs["dims"] = literal_eval(net_log.loc[0,"model_dims"])
    kwargs["alpha"] = None
    kwargs["g"] = None
    kwargs["init_path"] = net_ls[nidx]
    kwargs["with_bias"] = False

    #cbar_ticks_2 = np.round(cmap_bd,1)
    print("Plotting hidden layers!")

    ims = []
    for init_idx, init_epoch in enumerate(init_epochs):
        kwargs["init_epoch"] = init_epoch
        model = ModelFactory(**kwargs)
        model.eval()
        with torch.no_grad():
            if plot_layer:
                if not post:
                    preact_layers = model.preact_layer(image)
                else:
                    postact_layers, output = model.postact_layer(image)
            else:
                if not post:
                    preact_layers = model.preact_layer(images)
                else:
                    postact_layers, output = model.postact_layer(images)
        for lidx, l in enumerate(range(L - 1)):
            # selected hidden layer
            if not post:
                hidden_layer = preact_layers[l].detach().numpy()
            else:
                hidden_layer = postact_layers[l].detach().numpy()

            # center hidden layer
            if plot_layer:
                quantity = hidden_layer
                d2s = [compute_dq(quantity.flatten(), 2)]
                d2s = np.array(d2s)
            else:
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
            #hidden_layer = np.abs(hidden_layer)
            cmap_bd = [np.percentile(quantity.flatten(), 20), np.percentile(quantity.flatten(), 80)]
            cmap_bd = [np.min(quantity.flatten()), np.max(quantity.flatten())]

            # remove pixels with small acitivity
            #hidden_layer[np.abs(hidden_layer) < 5e-1] = np.NaN
            quantity = quantity.reshape(28,28)
            
            #colmap = plt.cm.get_cmap(cm_type_2)
            # need to mention the threshold
            #hidden_layer[np.abs(hidden_layer) < 1e-2] = np.NaN
            colmap.set_bad(color="white")
            im = axs[lidx + 1,init_idx].imshow(quantity, 
                                               vmin=cmap_bd[0], vmax=cmap_bd[1],
                                               aspect="auto", cmap=colmap)

            stablefit = pd.read_csv(join(net_ls[nidx], f"stablefit_epoch_widx={l}"))
            alphas = round(stablefit.loc[init_epoch,'alpha'],2)
            axs[lidx + 1,init_idx].text(-0.1, 1.2, (alphas,round(d2s[0],2)), transform=axs[lidx + 1,init_idx].transAxes,      # fontweight='bold'
                                        va='top', ha='right')

            cbar = fig.colorbar(im, ax=axs[lidx + 1,init_idx], 
                                orientation="vertical")

            if init_idx == 0:
                axs[lidx + 1,init_idx].set_ylabel(f"Layer {l + 1}")
        axs[1,init_idx].set_title(f"Epoch {init_epoch}")                

    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
