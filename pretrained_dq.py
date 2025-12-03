import json, os, random, sys
import numpy as np
import pandas as pd
from ast import literal_eval
from numpy import linalg as la
from os.path import join
from tqdm import tqdm

sys.path.append(os.getcwd())
from constants import DROOT
from UTILS.utils_dnn import IPR
from UTILS.mutils import njoin

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global POST_DICT, REIG_DICT
POST_DICT = {0:'pre', 1:'post'}
REIG_DICT = {0:'l', 1:'r'}


def get_net(net_path, post, epoch, *args):
    """    
    computes and saves all the D^l W^l's matrix multiplication, i.e. pre- or post-activation jacobians
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian

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

# ---------- Jacobian-related computations ----------

def jac_dq(net_path, navg, epoch, post, reig, *args):
    """

    Computing the eigenvectors of the associated jacobians from jac_layerwise() (without saving), 
    and then saving the Dq's averaged across different inputs

    """
    from nporch.input_loader import get_data_normalized

    global DW_all, eigvals, eigvecs, qs, D_qs, DW, dq_means, dq_stds, L, model, image, trainloader, net_log

    # number of images to take average over
    navg = int(navg)
    # post: pre/post activation for Jacobians, reig: right or left eigenvectors
    post, reig = int(post), int(reig)
    assert post in [0,1], "No such option!"
    assert reig in [0,1], "No such option!"

    # load model config
    f = open(njoin(net_path,'model_config.json'))
    model_config = json.load(f)
    f.close()
    net_log = pd.read_csv(njoin(net_path,'log'))  # , index_col=0
     
    # load dataset
    image_type = net_log.loc[0,'name']
    trainloader , _, _ = get_data_normalized(image_type,1)  # batch size 1 

    # sub-sample dataset
    seed = net_log.loc[0,'model_id']
    random.seed(int(seed))
    input_idxs = random.sample(range(len(trainloader)), navg)

    # load model
    L = len(model_config['dims']) - 1
    model = get_net(net_path, post, epoch)
    model.eval()

    qs = np.linspace(0,2,50)
    for idx, input_idx in enumerate(tqdm(input_idxs)):

        # compute DW's
        image = trainloader.dataset[input_idx][0]
        DW_all = model.layerwise_jacob_ls(image, post)

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


# ---------- Neural representations-related computations ----------


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])