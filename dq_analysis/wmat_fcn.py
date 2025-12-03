import matplotlib.pyplot as plt
import numpy as np
import os, sys
from ast import literal_eval
from os.path import join
from tqdm import tqdm

#plt.switch_backend('agg')

sys.path.append(os.getcwd())
from constants import root_data
from UTILS.utils_dnn import IPR

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global reig_dict
reig_dict = {0:'l', 1:'r'}

# ---------- Computing D_q without saving the Jacobian ----------

def get_wmats(alpha100, g100, epoch):
    global net
    from NetPortal.models import ModelFactory

    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.  

    # load nets and weights
    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
    hidden_N = [784]*L + [10]
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "with_bias": False,
              "architecture": 'fc'}
    net = ModelFactory(**kwargs)    
    wmats = [p for p in net.parameters()]

    return wmats

# averages can only be taken across the dimension (i.e. number of eigenvalues)
def wmat_dq(alpha100, g100, epoch, reig, is_eig=True, save_fig=False, *args):
    """

    Computing the eigenvectors of the weight matrices

    """

    global wmats, eigvals, eigvecs, qs, D_qs, D_qss, nan_count, wmat
    global dq_means, dq_stds

    from numpy import linalg as la

    # reig: right or left eigenvectors
    reig = int(reig)
    is_eig = literal_eval(is_eig) if isinstance(is_eig, str) else is_eig
    assert reig == 1 or reig == 0, "No such option!"

    main_path = join(root_data, "geometry_data")
    save_path = f"{main_path}/wmat_dq_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   

    qs = np.linspace(0,2,50)
    wmats = get_wmats(alpha100, g100, epoch)
    L = len(wmats)  
    dq_shape = [L + 1, len(qs)]    # extra for storing qs
    dq_means = np.zeros(dq_shape)
    dq_stds = np.zeros(dq_shape)         

    #for l in [0]:
    for l in range(L):
        D_qss = []

        wmat = wmats[l].detach().numpy()
        # left eigenvector
        if reig == 0:
            wmat = wmat.T

        print(f"layer {l}: {wmat.shape}")
        # SVD if not square matrix
        if wmat.shape[0] != wmat.shape[1] or not is_eig:
            # old method
            #DW = DW[:10,:]
            #DW = DW.T @ DW  # final DW is 10 x 784
            _, eigvals, eigvecs = la.svd(wmat)
        else:
            eigvals, eigvecs = la.eig(wmat)

        nan_count = 0
        for i in range(len(eigvals)):  
            eigvec = eigvecs[:,i]
            IPRs = np.array([IPR(eigvec ,q) for q in qs])
            D_qs = np.log(IPRs) / (1-qs) / np.log(len(eigvec))
            
            if not any(np.isnan(D_qs)): D_qss.append(D_qs)
            else: nan_count += 1  

        if nan_count > 0:
            print(f"Layer {l} has nan_count: {nan_count}")

        # save D_q values individually for layers
        """
        result = np.transpose([qs, np.mean(D_qss, axis=0), np.std(D_qss, axis=0)])
        outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.txt'
        outpath = f"{save_path}/{outfname}"
        print(f'Saving {outpath}')
        np.savetxt(outpath, result)
        """

        if l == 0:
            dq_means = np.transpose([qs, np.mean(D_qss, axis=0)])
            dq_stds = np.transpose([qs, np.std(D_qss, axis=0)])
        else:
            dq_mean = np.mean(D_qss, axis=0)
            dq_std = np.std(D_qss, axis=0)
            dq_mean, dq_std = np.expand_dims(dq_mean,axis=1), np.expand_dims(dq_std,axis=1)
            dq_means = np.concatenate((dq_means, dq_mean),axis=1)
            dq_stds = np.concatenate((dq_stds, dq_std),axis=1)
        
        print(dq_means.shape)
        print(dq_stds.shape)
        print("\n")

    # save D_q values
    vec_type = "eig" if is_eig else "sing"
    # mean    
    outfname = f'wmat_{vec_type}vec_dqmean_alpha{alpha100}_g{g100}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving means: {outpath}')
    np.savetxt(outpath, dq_means)
    outfname = f'wmat_{vec_type}vec_dqstd_alpha{alpha100}_g{g100}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving stds: {outpath}')
    np.savetxt(outpath, dq_stds)
    print(f"Conversion to weight matrix Dq complete for ({alpha100}, {g100}) at epoch {epoch}!")

# ---------- Computing D_2 (without saving the weight matrix) ----------

def wmat_d2(alpha100, g100, epoch, reig, is_eig=True, save_fig=False, *args):
    """

    Computing the eigenvectors of the associated jacobians from jac_layerwise() (without saving), 
    and then saving the Dq's averaged across different inputs

    """

    global wmats, eigvals, eigvecs, qs, D_qs, D_qss, wmat
    global d2_means, d2_stds

    import numpy as np    
    import random
    import torch
    from numpy import linalg as la
    from tqdm import tqdm

    # number of images to take average over
    # reig: right or left eigenvectors
    reig = int(reig)
    is_eig = literal_eval(is_eig) if isinstance(is_eig, str) else is_eig
    assert reig == 1 or reig == 0, "No such option!"

    main_path = join(root_data, "geometry_data")
    save_path = f"{main_path}/wmat_d2_layerwise_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   

    q = 2
    wmats = get_wmats(alpha100, g100, epoch)  # just for getting L
    L = len(wmats)  
    dq_shape = [L]    # multiple inputs
    d2_means = np.zeros(dq_shape)    

    #for l in [0]:
    for l in range(L):
        D_qss = []

        wmat = wmats[l].detach().numpy()
        # left eigenvector
        if reig == 0:
            wmat = wmat.T

        print(f"layer {l}: {wmat.shape}")
        # SVD if not square matrix
        if wmat.shape[0] != wmat.shape[1] or not is_eig:
            # old method
            #DW = DW[:10,:]
            #DW = DW.T @ DW  # final DW is 10 x 784
            _, eigvals, eigvecs = la.svd(wmat)
        else:
            eigvals, eigvecs = la.eig(wmat)   

        d2_mean = 0
        for i in range(len(eigvals)):  
            eigvec = eigvecs[:,i]
            d2_mean += np.log(IPR(eigvec ,q)) / (1-q) / np.log(len(eigvec))
        d2_means[l] = d2_mean/len(eigvals) 

    # save D_q values
    vec_type = "eig" if is_eig else "sing"
    # mean
    outfname = f'wmat-{vec_type}vec-d2means-alpha{alpha100}-g{g100}-ep{epoch}'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving means: {outpath}')
    np.save(outpath, d2_means)
    print(f"Conversion to D2 complete for ({alpha100}, {g100}) at epoch {epoch}!")


def submit(*args):
    from qsub import qsub, job_divider, project_ls
    pbs_array_data = [(alpha100, g100, epoch)
                      for alpha100 in range(100, 201, 10)
                      for g100 in range(25, 301, 25)
                      #for alpha100 in [100, 200]
                      #for g100 in [100]
                      for epoch in [0,1] + list(range(50,651,50))
                      #for epoch in [0, 100, 650]
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data/wmat_dq_l"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

def nosave_submit(*args):
    from qsub import qsub, job_divider, project_ls
    is_eigs = [True, False]
    #reig = 0
    reigs = [0,1]
    pbs_array_data = [(alpha100, g100, epoch, reig, is_eig)
                      #for alpha100 in range(100, 201, 10)
                      #for g100 in range(25, 301, 25)
                      for alpha100 in [120, 200]
                      for g100 in [25, 100, 300]
                      for epoch in [0,1,50,150,200,250,300,650]
                      for reig in reigs
                      #for epoch in [0,1] + list(range(50,651,50))
                      #for epoch in [0, 100, 650]
                      #for epoch in [0]
                      for is_eig in is_eigs
                      ]

    #pbs_array_data = pbs_array_data[:2]  # delete
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data/wmat_dq_r"),
             P=project_ls[pidx],
             #source="virt-test-qu/bin/activate",
             source="newtorch/bin/activate",
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
