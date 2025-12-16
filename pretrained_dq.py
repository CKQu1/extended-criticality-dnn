import json, os, random, sys
import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from numpy import linalg as la
from os.path import join
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset 

sys.path.append(os.getcwd())
from constants import DROOT
from UTILS.data_utils import set_data
from UTILS.utils_dnn import IPR, compute_dq
from UTILS.mutils import njoin
from nporch.input_loader import get_data_normalized

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global POST_DICT, REIG_DICT
POST_DICT = {0:'pre', 1:'post'}
REIG_DICT = {0:'l', 1:'r'}
DEVICE = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_net(net_path, post, epoch):
    """    
    computes and saves all the D^l W^l's matrix multiplication, i.e. pre- or post-activation jacobians
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian
    - reig = 1: right eigenvectors
           = 0: left eigenvectors

    - input_idx: the index of the image
    """    
    from NetPortal.models import ModelFactory

    post = int(post)
    assert post in [0,1], "No such option!" 

    # load nets and weights
    f = open(njoin(net_path,'model_config.json'))
    model_config = json.load(f)
    f.close()
    kwargs = {"dims": model_config['dims'], "alpha": None, "g": None,
              "init_path": net_path, "init_epoch": epoch,
              "activation": 'tanh', "with_bias": False,
              "architecture": 'fc'}
    return ModelFactory(**kwargs)

# -------------------- Jacobian-related computations --------------------

def jac_dq(net_path, navg, epoch, post, reig):
    """
    Computing the eigenvectors of the associated jacobians from jac_layerwise(), 
    saving one set of eigenvectors for a randomly selected (image) input (without saving the rest)   
    and then saving the Dq's averaged across different inputs.
    """

    global DW_all, eigvals, eigvecs, qs, D_qs, DW, dq_means, dq_stds, L, model, image, trainloader, net_log

    # number of images to take average over
    navg = int(navg)
    post, reig = int(post), int(reig)
    assert post in [0,1], "No such option!"; assert reig in [0,1], "No such option!"

    # load model config
    f = open(njoin(net_path,'model_config.json'))
    model_config = json.load(f)
    f.close()
    net_log = pd.read_csv(njoin(net_path,'log'))  # , index_col=0

    # set seed
    seed = int(net_log.loc[0,'model_id'])
    seed_everything(seed)

    # load dataset
    image_type = net_log.loc[0,'name']
    # trainloader , _, _ = get_data_normalized(image_type,1)  # batch size 1 
    train_ds, _ = set_data('mnist', True)
    train_ds = TensorDataset(torch.stack([e[0] for e in train_ds]).to(DEVICE),
                                torch.tensor(train_ds.targets).to(DEVICE))
    trainloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # sub-sample dataset
    input_idxs = random.sample(range(len(trainloader)), navg)

    # load model
    L = len(model_config['dims']) - 1
    model = get_net(net_path, post, epoch)
    model.eval()

    qs = np.linspace(0,2,50)
    for idx, input_idx in enumerate(tqdm(input_idxs)):

        # compute DW's
        image = trainloader.dataset[input_idx][0]
        DW_all = model.layerwise_jacob_ls(image[None], post)

        if idx == 0:
            # ----- initialize first time only -----
            dq_shape = [navg, L, len(qs)]
            dq_means = np.zeros(dq_shape) 
            dq_stds = np.zeros(dq_shape)      
            # ----- save full Jacobian for idx == 0 (once) -----
            DW_save = {}
            for ii, DW in enumerate(DW_all):
                DW_save[f'DW_l={ii}_input={input_idx}'] = DW.detach().numpy()
            # np.savez(njoin(net_path, f'jac_input={input_idx}.npz'), **DW_save)
            # del DW_save
            # ----- save corresponding D_qs -----
            Dq_save = {}

        for l in range(L):
            D_qss = []

            DW = DW_all[l].detach().numpy()
            if reig == 0:  # left eigenvector
                DW = DW.T

            if idx == navg - 1:
                print(f"layer {l}: {DW.shape}")
            # SVD if not square matrix
            if DW.shape[0] != DW.shape[1]:
                # old method
                #DW = DW[:10,:]
                #DW = DW.T @ DW  # final DW is 10 x 784
                _, eigvals, eigvecs = la.svd(DW)
            else:
                eigvals, eigvecs = la.eig(DW)

            for i in range(len(eigvals)):  
                eigvec = eigvecs[:,i]
                IPRs = np.array([IPR(eigvec ,q) for q in qs])
                D_qs = np.log(IPRs) / (1-qs) / np.log(len(eigvec))
                
                D_qss.append(D_qs)
            # save D_qss for idx == 0
            if idx == 0:
                Dq_save[f'Dq_l={l}_input={input_idx}'] = np.array(D_qss)

            # means over eigenvalue index for single input and layer
            dq_means[idx,l,:] = np.mean(D_qss, axis=0)
            dq_stds[idx,l,:] = np.std(D_qss, axis=0)

    # save D_q 
    np.savez(njoin(net_path, f'jac_epoch={epoch}_post={post}_reig={reig}.npz'), 
             qs=qs, dq_means=dq_means, dq_stds=dq_stds, input_idx=input_idxs[0],
             **Dq_save, **DW_save)

    print(f"Conversion to Dq complete for {net_path} at epoch {epoch}!")


# -------------------- Neural representations-related computations --------------------

def npc_compute(net_path, epoch, post, batch_size):
    """
    - batch_size (int): only used for computing effective dimension (ED)
    """

    from sklearn.decomposition import PCA

    epoch = int(epoch); post = int(post); batch_size = int(batch_size)
    assert post in [0,1], "No such option!"

    # load model config
    f = open(njoin(net_path,'model_config.json'))
    model_config = json.load(f)
    f.close()
    net_log = pd.read_csv(njoin(net_path,'log'))  # , index_col=0

    # set seed
    seed = int(net_log.loc[0,'model_id'])
    seed_everything(seed)

    # load dataset
    image_type = net_log.loc[0,'name']
    # trainloader , _, _ = get_data_normalized(image_type,1)  # batch size 1 
    train_ds, _ = set_data(image_type, True)
    targets = train_ds.targets.unique()

    train_dataset = TensorDataset(torch.stack([e[0] for e in train_ds]).to(DEVICE),
                                  torch.tensor(train_ds.targets).to(DEVICE))
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # load model
    model = get_net(net_path, post, epoch)
    model.eval()

    model_depth = len(model.sequential)
    C_dims = np.zeros([model_depth])   # record dimension of correlation matrix
    if post == 0 or post == 1:
        l_remainder = post
    else:
        l_remainder = None
    l_start = l_remainder
    if post == 0 or post == 1:
        depth_idxs = list(range(l_start,model_depth,2))
        L = len(depth_idxs)
    else:
        L = model_depth

    # ---------- Part I: Effective dimension ----------
    ED_means = np.zeros([1,L]); ED_stds = np.zeros([1,L])
    # create batches and compute and mean/std over the batches
    for batch_idx, (hidden_layer, yb) in tqdm(enumerate(trainloader)):
        l = 0
        for lidx in range(model_depth):
            hidden_layer = model.sequential[lidx](hidden_layer)
            hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
            if lidx % 2 == l_remainder or l_remainder == None:     # pre or post activation
                # covariance matrix (without cirectly using PCA form sklearn)
                C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer)\
                      - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
                ED = torch.trace(C)**2/torch.trace(C@C)
                ED = ED.item()

                ED_means[0,l] += ED
                ED_stds[0,l] += ED**2
                if batch_idx == 0:
                    C_dims[lidx] = hidden_layer.shape[1]

                l += 1

    print(f"Batches: {batch_idx + 1}")
    batches = batch_idx + 1
    for l in range(ED_means.shape[1]):
        ED_means[0,l] /= batches
        ED_stds[0,l] = np.sqrt( ED_stds[0,l]/batches - ED_means[0,l]**2 )

    # ---------- Part II: Neural representations ----------
    # trainloader , _, _ = get_data_normalized(image_type, full_batch_size)  # full batch
    trainloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)  # full batch
    n_components = 3

    target_indices = np.empty([len(targets), len(train_ds)])
    # group in classes
    _, labels = next(iter(trainloader))
    for cl_idx, cls in enumerate(targets):
        target_indices[cl_idx] = labels == cls

    npc_data = {}
    # compute the associated D2 quantities over the full batch
    hidden_layer = torch.squeeze(torch.stack([e[0] for e in trainloader]).to(DEVICE))        
    for lidx in tqdm(range(len(model.sequential))):
        hidden_layer = model.sequential[lidx](hidden_layer)
        #hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
        if lidx % 2 == l_remainder or l_remainder == None:     # pre or post activation
            # directly use PCA
            hidden_layer_centered = hidden_layer - hidden_layer.mean(0)
            if "cpu" in DEVICE.type:
                hidden_layer_centered = hidden_layer_centered.detach().numpy()
            else:
                hidden_layer_centered = hidden_layer_centered.cpu().detach().numpy()
            #hidden_layer_centered = StandardScaler().fit_transform(hidden_layer.detach().numpy())
            pca = PCA()
            #pca.fit(hidden_layer_centered)
            PCs = pca.fit_transform(hidden_layer_centered)
            eigvals = pca.explained_variance_
            eigvecs = pca.components_
            #ED = np.sum(eigvals)**2/np.sum(eigvals**2)

            dqs_npc = np.zeros(PCs.shape[0])
            dqs_npd = np.zeros(len(eigvals))
            for pcidx in range(PCs.shape[0]):
                dqs_npc[pcidx] = compute_dq(PCs[pcidx,:],2)
            for eidx in range(len(eigvals)):
                dqs_npd[eidx] = compute_dq(eigvecs[eidx,:],2)

            # pass in data to npc_data
            npc_data[f'D2_npc_{l}'] = dqs_npc
            npc_data[f'D2_npd_{l}'] = dqs_npd

            npc_data[f'npc_{l}'] = PCs[:, n_components]
            npc_data[f'npc_eigvals_{l}'] = eigvals

            l += 1

    # save all data
    np.savez(njoin(net_path, f"npc_epoch={epoch}_post={post}.npz"), 
             batch_size=batch_size, ED_means=ED_means, ED_stds=ED_stds, 
             targets=targets, target_indices=target_indices, **npc_data)

    print(f"ED via method batches is saved for epochs {epoch}!")




if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])