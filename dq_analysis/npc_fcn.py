import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from ast import literal_eval
from os.path import join
from tqdm import tqdm

plt.switch_backend('agg')

sys.path.append(os.getcwd())
from path_names import root_data
from utils import IPR

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global post_dict, reig_dict
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

def npc_layerwise(post, alpha100, g100, input_idx, epoch, *args):
    """

    computes the PCs of the layerwise neural representations
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian

    - input_idx: the index of the image
    
    """

    import scipy.io as sio
    import sys
    import torch
    import train_DNN_code.model_loader as model_loader
    from nporch.input_loader import get_data_normalized    
    from NetPortal.models import ModelFactory

    post = int(post)
    assert post == 1 or post == 0, "No such option!"

    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    input_idx = int(input_idx)
    # load MNIST
    image_type = 'mnist'
    trainloader , _, _ = get_data_normalized(image_type,1)      

    # load nets and weights
    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
    hidden_N = [784]*L + [10]
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # push the image through
    image = trainloader.dataset[input_idx][0]
    num = trainloader.dataset[input_idx][1]
    print(torch.sum(image))
    print(num)

    return net.layerwise_jacob_ls(image, post)

def submit(*args):
    from qsub import qsub, job_divider, project_ls
    input_idxs = str( list(range(1,50)) ).replace(" ", "")
    post = 0
    pbs_array_data = [(post, alpha100, g100, input_idxs, epoch)
                      for alpha100 in range(100, 201, 10)
                      for g100 in [25,100,300]
                      #for alpha100 in [100, 200]
                      #for g100 in [100]
                      #for epoch in [0,1] + list(range(50,651,50))
                      for epoch in [0, 1, 50, 100, 150, 200, 650]
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

# ----- preplot -----

def npc_to_dq(alpha100, g100, input_idxs, epoch, post, reig, *args):

    """

    Computing the eigenvectors of the associated jacobians from jac_save(), and then saving the Dq's

    """

    import numpy as np    
    import torch
    from numpy import linalg as la
    from tqdm import tqdm

    # post: pre/post activation for Jacobians, reig: right or left eigenvectors
    post, reig = int(post), int(reig)
    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"
    input_idxs = literal_eval(input_idxs)

    join(root_data, "geometry_data")
    data_path = f"{main_path}/{post_dict[post]}jac_layerwise"
    save_path = f"{main_path}/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"
    fig_path = f"{main_path}/dq_layerplot_{post_dict[post]}_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   
    if not os.path.exists(f'{fig_path}'):
        os.makedirs(f'{fig_path}')  

    qs = np.linspace(0,2,50)

    for idx, input_idx in enumerate(tqdm(input_idxs)):
        # load DW's
        DW_all = torch.load(f"{data_path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")
        DWs_shape = DW_all.shape

        for l in tqdm(range(DWs_shape[0])):
            D_qss = []

            DW = DW_all[l].numpy()
            if l == DWs_shape[0]:
                DW = DW[:10,:]
                DW = DW.T @ DW  # final DW is 10 x 784

            # left eigenvector
            if reig == 0:
                DW = DW.T

            print(f"layer {l}: {DW.shape}")
            eigvals, eigvecs = la.eig(DW)
            nan_count = 0
            for i in range(len(eigvals)):  
                eigvec = eigvecs[:,i]
                IPRs = np.array([IPR(eigvec ,q) for q in qs])
                D_qs = np.log(IPRs) / (1-qs) / np.log(len(eigvec))
                
                if not any(np.isnan(D_qs)): D_qss.append(D_qs)
                else: nan_count += 1  

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

            # plot D_q image
            plt.errorbar(qs, np.mean(D_qss, axis=0), yerr=np.std(D_qss, axis=0))
            plt.ylim([0,1.1])
            plt.xlim([0, 2])
            plt.ylabel('$D_q$')
            plt.xlabel('$q$')
            #plt.title(path.split('/')[-2:]+[str(nan_count)])
            outfname = f'dq_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}_l{l}.txt'
            plt.title([outfname]+[str(nan_count)])
            #plt.show()

            outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.png'
            outpath = f"{fig_path}/{outfname}"
            print(f'Saving {outpath}')
            plt.savefig(outpath)
            plt.clf()

            print("\n")
            print(dq_means.shape)
            print(dq_stds.shape)

        # save D_q values
        # mean
        outfname = f'dqmean_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}.txt'
        outpath = f"{save_path}/{outfname}"
        #print(f'Saving means: {outpath}')
        np.savetxt(outpath, dq_means)
        outfname = f'dqstd_alpha{alpha100}_g{g100}_ipidx{input_idx}_ep{epoch}.txt'
        outpath = f"{save_path}/{outfname}"
        #print(f'Saving stds: {outpath}')
        np.savetxt(outpath, dq_stds)
    print(f"Conversion to Dq complete for ({alpha100}, {g100}) at epoch {epoch}!")


def submit_preplot(*args,post=1,reig=0):
#def submit_preplot(path):

    post = int(post)
    assert post == 1 or post == 0, "No such option!"
    reig = int(reig)    # whether to compute for right eigenvector or not
    assert reig == 1 or reig == 0, "No such option!"
    input_idxs = str( list(range(1,50)) ).replace(" ", "")

    # change the path accordingly
    data_path = join(root_data, f"geometry_data/{post_dict[post]}jac_layerwise")
    # find the `alpha100`s and `g100`s of the files in the folder
    # dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}
    pbs_array_data = set([tuple(re.findall('\d+', fname)[:4]) + (int(post),int(reig)) for fname in os.listdir(data_path)
                      if all(s in fname for s in ('dw', 'alpha', 'g', 'ipidx', 'epoch'))])

    #pbs_array_data = set([tuple(re.findall('\d+', fname)[:4]) for fname in os.listdir(data_path) if all(s in fname for s in ('dw', 'alpha', 'g', 'ipidx', 'epoch'))])

    # test
    #pbs_array_data = { ('100', '100', '0', '650'), ('200', '100', '0', '650') }
    print(list(pbs_array_data)[0])
    print(f"Total subjobs: {len(pbs_array_data)}")

    # rerunning missing data
    """
    dq_path = join(root_data, "geometry_data")
    missing_data = np.loadtxt(f"{dq_path}/missing_data.txt")
    pbs_array_data = []
    for m in missing_data:
        pbs_array_data.append(tuple(m.astype('int64')))
    pbs_array_data = set(pbs_array_data)
    """

    from qsub import qsub, job_divider, project_ls
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
