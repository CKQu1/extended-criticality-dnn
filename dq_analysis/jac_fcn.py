import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from ast import literal_eval
from os.path import join
from tqdm import tqdm

#plt.switch_backend('agg')

sys.path.append(os.getcwd())
from constants import DROOT
from UTILS.utils_dnn import IPR

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global post_dict, reig_dict
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

def jac_layerwise(post, alpha100, g100, input_idx, epoch, *args):
    """    
    computes and saves all the D^l W^l's matrix multiplication, i.e. pre- or post-activation jacobians
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian

    - input_idx: the index of the image
    """

    global utils, train_loader

    from nporch.input_loader import get_data_normalized    
    from NetPortal.models import ModelFactory

    post = int(post)
    assert post == 1 or post == 0, "No such option!"

    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    #data_path = join(DROOT, f"trained_mlps/fcn_grid/{fcn}_grid")

    seed = 0
    data_path = join(DROOT, f'{fcn}_sgd_mnist', f'{fcn}_sgd_mnist_seed={seed}')

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    input_idx = int(input_idx)
    # load MNIST
    image_type = 'mnist'
    trainloader , _, _ = get_data_normalized(image_type,1)      

    # load nets and weights
    #net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"      
    net_folder = f'{fcn}_{alpha100}_{g100}_{seed}_mnist_sgd_lr=0.005_bs=1024_epochs=100'
    hidden_N = [784]*L + [10]
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "with_bias": False,
              "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # push the image through
    image = trainloader.dataset[input_idx][0]
    #num = trainloader.dataset[input_idx][1]
    #print(torch.sum(image))
    #print(num)

    return net.layerwise_jacob_ls(image, post)

# ---------- Computing D_q without saving the Jacobian ----------

def jac_dq(alpha100, g100, navg, epoch, post, reig, save_fig=False, *args):
    """

    Computing the eigenvectors of the associated jacobians from jac_layerwise() (without saving), 
    and then saving the Dq's averaged across different inputs

    """

    global DW_all, eigvals, eigvecs, qs, D_qs, DW

    import numpy as np    
    import random
    import torch
    from numpy import linalg as la
    from tqdm import tqdm

    # number of images to take average over
    navg = int(navg)
    # post: pre/post activation for Jacobians, reig: right or left eigenvectors
    post, reig = int(post), int(reig)
    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    main_path = join(DROOT, "geometry_data")
    save_path = f"{main_path}/dq_layerwise_navg={navg}_{post_dict[post]}_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   

    qs = np.linspace(0,2,50)
    input_idxs = random.sample(range(60000), navg)
    DW_all = jac_layerwise(post, alpha100, g100, 0, epoch)  # just for getting L
    L = len(DW_all)  
    dq_shape = [L + 1, DW_all.shape[0], len(qs)]    # extra for storing qs
    dq_means = np.zeros(dq_shape)
    dq_stds = np.zeros(dq_shape)
    for idx, input_idx in enumerate(tqdm(input_idxs)):
        # compute DW's
        DW_all = jac_layerwise(post, alpha100, g100, input_idx, epoch)
        L = len(DW_all)        

        #for l in [0]:
        for l in range(L):
            D_qss = []

            DW = DW_all[l].detach().numpy()
            # left eigenvector
            if reig == 0:
                DW = DW.T

            if idx == len(input_idxs) - 1:
                print(f"layer {l}: {DW.shape}")
            # SVD if not square matrix
            if DW.shape[0] != DW.shape[1]:
                # old method
                #DW = DW[:10,:]
                #DW = DW.T @ DW  # final DW is 10 x 784
                _, eigvals, eigvecs = la.svd(DW)
            else:
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

            print("\n")
            print(dq_means.shape)
            print(dq_stds.shape)

    # save D_q values
    # mean
    outfname = f'dqmean_alpha{alpha100}_g{g100}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving means: {outpath}')
    np.savetxt(outpath, dq_means)
    outfname = f'dqstd_alpha{alpha100}_g{g100}_ep{epoch}.txt'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving stds: {outpath}')
    np.savetxt(outpath, dq_stds)
    print(f"Conversion to Dq complete for ({alpha100}, {g100}) at epoch {epoch}!")

# ---------- Computing D_2 for multiple inputs (without saving the Jacobian) ----------

def jac_d2(alpha100, g100, navg, epoch, post, reig, save_fig=False, *args):
    """

    Computing the eigenvectors of the associated jacobians from jac_layerwise() (without saving), 
    and then saving the Dq's averaged across different inputs

    """

    global DW_all, eigvals, eigvecs, qs, D_qs, DW

    import numpy as np    
    import random
    import torch
    from numpy import linalg as la
    from tqdm import tqdm

    # number of images to take average over
    navg = int(navg)
    # post: pre/post activation for Jacobians, reig: right or left eigenvectors
    post, reig = int(post), int(reig)
    assert post == 1 or post == 0, "No such option!"
    assert reig == 1 or reig == 0, "No such option!"

    main_path = join(DROOT, "geometry_data")
    save_path = f"{main_path}/d2_layerwise_navg={navg}_{post_dict[post]}_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   

    q = 2
    input_idxs = random.sample(range(60000), navg)
    DW_all = jac_layerwise(post, alpha100, g100, 0, epoch)  # just for getting L
    L = len(DW_all)  
    dq_shape = [L, navg]    # multiple inputs
    d2_means = np.zeros(dq_shape)
    for idx, input_idx in enumerate(tqdm(input_idxs)):
        # compute DW's
        DW_all = jac_layerwise(post, alpha100, g100, input_idx, epoch)
        L = len(DW_all)        

        #for l in [0]:
        for l in range(L):
            D_qss = []

            DW = DW_all[l].detach().numpy()
            # left eigenvector
            if reig == 0:
                DW = DW.T

            if idx == len(input_idxs) - 1:
                print(f"layer {l}: {DW.shape}")
            # SVD if not square matrix
            if DW.shape[0] != DW.shape[1]:
                # old method
                #DW = DW[:10,:]
                #DW = DW.T @ DW  # final DW is 10 x 784
                _, eigvals, eigvecs = la.svd(DW)
            else:
                eigvals, eigvecs = la.eig(DW)
            d2_mean = 0
            for i in range(len(eigvals)):  
                eigvec = eigvecs[:,i]
                d2_mean += np.log(IPR(eigvec ,q)) / (1-q) / np.log(len(eigvec))
            d2_means[l,idx] = d2_mean/len(eigvals) 

    # save D_q values
    # mean
    outfname = f'd2means-navg={navg}-alpha{alpha100}-g{g100}-ep{epoch}'
    outpath = f"{save_path}/{outfname}"
    #print(f'Saving means: {outpath}')
    np.save(outpath, d2_means)
    print(f"Conversion to D2 complete for ({alpha100}, {g100}) at epoch {epoch}!")

# ---------- Saving Jacobian ----------

# this actually eats up a lot of storage, don't recommend saving for too many input_idxs
def jac_save(post, alpha100, g100, input_idxs='[0,1]', epoch=100, *args):
    import torch
    post = int(post)
    print((post, alpha100, g100, input_idxs, epoch))
    input_idxs = literal_eval(input_idxs)
    print(input_idxs)
    print(type(input_idxs))
    assert post == 1 or post == 0, "No such option!"
    if post == 1:
        path = join(DROOT, "geometry_data/postjac_layerwise")
    elif post == 0:
        path = join(DROOT, "geometry_data/prejac_layerwise")
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')

    if post == 1:
        print("Computing post-activation layerwise Jacobian")
    elif post == 0:
        print("Computing pre-activation layerwise Jacobian")
    for idx, input_idx in enumerate(tqdm(input_idxs)):
        DW_ls = jac_layerwise(post, alpha100, g100, input_idx, epoch, *args)
        DW_all = torch.ones((len(DW_ls), DW_ls[0].shape[0], DW_ls[0].shape[1]))
        for idx in range(0, len(DW_ls)):
            DW = DW_ls[idx]
            DW_all[idx,:DW.shape[0],:DW.shape[1]] = DW.detach().clone()

        torch.save(DW_all, f"{path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")
    print("DW_all shape: {DW_all.shape}")
    print(f"A total of {len(input_idxs)} images computed for (alpha, g) = ({alpha100}, {g100}) at epoch {epoch}!")
    print(f"Saved as: {path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}!")

# func: jac_save
def submit(*args):
    from qsub import qsub, job_divider, project_ls
    input_idxs = str( list(range(1,50)) ).replace(" ", "")
    post = 0
    pbs_array_data = [(post, alpha100, g100, input_idxs, epoch)
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
             path=join(DROOT, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

# func: jac_d2, jac_dq
def nosave_submit(*args):
    from qsub import qsub, job_divider, project_ls
    post = 0
    navg = 1000
    reig = 1
    pbs_array_data = [(alpha100, g100, navg, epoch, post, reig)
                      #for alpha100 in range(100, 201, 10)
                      #for g100 in range(25, 301, 25)
                      for alpha100 in [120, 200]
                      for g100 in [25, 100, 300]
                      #for epoch in [0,1,50,150,200,250,300,650]
                      for epoch in [0,100]
                      #for epoch in [0,1] + list(range(50,651,50))
                      #for epoch in [0, 100, 650]
                      #for epoch in [0]
                      ]

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    quit()
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(DROOT, "geometry_data"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='1GB') 

# ----- preplot -----

# only use this when you have saved values in files
def jac_to_dq(alpha100, g100, input_idx, epoch, post, reig, save_fig=False, *args):

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
    #input_idxs = literal_eval(input_idxs)

    main_path = join(DROOT, "geometry_data")
    data_path = f"{main_path}/{post_dict[post]}jac_layerwise"
    save_path = f"{main_path}/dq_layerwise_{post_dict[post]}_{reig_dict[reig]}"
    print(save_path)
    fig_path = f"{main_path}/dq_layerplot_{post_dict[post]}_{reig_dict[reig]}"
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')   
    if not os.path.exists(f'{fig_path}'):
        os.makedirs(f'{fig_path}')  

    qs = np.linspace(0,2,50)
    # load DW's
    DW_all = torch.load(f"{data_path}/dw_alpha{alpha100}_g{g100}_ipidx{input_idx}_epoch{epoch}")
    DWs_shape = DW_all.shape

    for l in tqdm(range(DWs_shape[0])):
        D_qss = []

        DW = DW_all[l].numpy()
        # left eigenvector
        if reig == 0:
            DW = DW.T
        #if idx == len(input_idxs) - 1:
        print(f"layer {l}: {DW.shape}")
        # SVD if not square matrix
        if l == DWs_shape[0]:
            # old method
            #DW = DW[:10,:]
            #DW = DW.T @ DW  # final DW is 10 x 784
            _, eigvals, eigvecs = la.svd(DW)
        else:
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

        if save_fig:
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

            #outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.png'
            outfname = f'dq_alpha{alpha100}_g{g100}_ep{epoch}_l{l}.pdf'
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

# func: jac_to_dq
def preplot_submit(*args,post=0,reig=1):
#def submit_preplot(path):
    global pbs_array_data, pbs_array_data_epochs

    post = int(post)
    assert post == 1 or post == 0, "No such option!"
    reig = int(reig)    # whether to compute for right eigenvector or not
    assert reig == 1 or reig == 0, "No such option!"
    #input_idxs = str( list(range(1,50)) ).replace(" ", "")

    # change the path accordingly
    data_path = join(DROOT, f"geometry_data/{post_dict[post]}jac_layerwise")
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
    dq_path = join(DROOT, "geometry_data")
    missing_data = np.loadtxt(f"{dq_path}/missing_data.txt")
    pbs_array_data = []
    for m in missing_data:
        pbs_array_data.append(tuple(m.astype('int64')))
    pbs_array_data = set(pbs_array_data)
    """

    from qsub import qsub, job_divider, project_ls
    pbs_array_data = list(pbs_array_data)

    # analysis on selected epochs
    #selected_epochs = ['0','1','50','100','150','200','250','650']
    selected_epochs = ['0', '100']
    pbs_array_data_epochs = [pbs_array_data[idx] for idx in range(len(pbs_array_data)) if pbs_array_data[idx][3] in selected_epochs]
    #pbs_array_data_epochs = pbs_array_data_epochs[0:2]    # test
    perm, pbss = job_divider(pbs_array_data_epochs, len(project_ls))
    
    print(len(pbs_array_data_epochs))
    print(len(pbss))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(DROOT, "geometry_data"),
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
