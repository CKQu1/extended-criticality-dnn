import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import re
import sys
import torch
from ast import literal_eval
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm

#plt.switch_backend('agg')

sys.path.append(os.getcwd())
from path_names import root_data
from utils_dnn import IPR, compute_dq, effective_dimension

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global post_dict, reig_dict
post_dict = {0:'pre', 1:'post'}
reig_dict = {0:'l', 1:'r'}

def npc_layerwise(post, alpha100, g100, epochs, reig=1):
    """

    computes the PCs of the layerwise neural representations
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian

    - input_idx: the index of the image
    
    """
    global C_ls, C_dims, dqs, eigvals, eigvecs, ii, hidden_layer, ED, ED_means, net, trainloader

    import scipy.io as sio
    import sys
    import torch
    from NetPortal.models import ModelFactory
    from train_supervised import get_data, set_data

    post = int(post)
    assert post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs

    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    # load MNIST
    image_type = 'mnist'
    trainloader, _ = set_data(image_type, rshape=True)
    trainloader = DataLoader(trainloader, batch_size=512, shuffle=False)

    # load nets and weights
    epoch = 0
    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
    hidden_N = [784]*L + [10]
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # compute effective dimension for a single input manifold
    C_ls = []    # list of correlation matrices
    C_dims = np.zeros([len(net.sequential)])   # record dimension of correlation matrix
    l_remainder = 0 if post == 0 else 1
    l_start = l_remainder
    depth_idxs = list(range(l_start,len(net.sequential),2))
    L = len(depth_idxs)
    save_path = join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_{reig_dict[reig]}")
    if not os.path.isdir(save_path): os.makedirs(save_path)
    with torch.no_grad():
        for epoch in tqdm(epochs):
            ED_means = np.zeros([1,L])
            ED_stds = np.zeros([1,L])
            D2_means = np.zeros([1,L])
            D2_stds = np.zeros([1,L])
            # load nets and weights
            net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
            hidden_N = [784]*L + [10]
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch,
                      "activation": 'tanh', "architecture": 'fc'}
            net = ModelFactory(**kwargs)
            # create batches and compute and mean/std over the batches
            for batch_idx, (hidden_layer, yb) in tqdm(enumerate(trainloader)):
                l = 0
                for lidx in range(len(net.sequential)):
                    hidden_layer = net.sequential[lidx](hidden_layer)
                    hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
                    if lidx % 2 == l_remainder:     # pre or post activation
                        C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
                        ED = torch.trace(C)**2/torch.trace(C@C)
                        ED = ED.item()
                        ED_means[0,l] += ED
                        ED_stds[0,l] += ED**2
                        if batch_idx == 0:
                            C_dims[lidx] = C.shape[0]
                        
                        eigvals, eigvecs = la.eigh(C)
                        ii = np.argsort(np.abs(eigvals))
                        dqs = np.zeros(len(ii))
                        for eidx, eigval in enumerate(eigvals):
                            dqs[eidx] = compute_dq(eigvecs[:,eidx],2)
                        D2_means[0,l] += dqs.mean()
                        D2_stds[0,l] += dqs.std()
                        
                        l += 1

            print(f"Batches: {batch_idx}")
            batches = batch_idx + 1
            for l in range(ED_means.shape[1]):
                ED_means[0,l] /= batches
                ED_stds[0,l] = ED_stds[0,l]/batches - ED_means[0,l]**2

            D2_means /= batches
            D2_stds /= batches

            # save data
            np.save(join(save_path, f"ED_means_{epoch}"), ED_means)
            np.save(join(save_path, f"ED_stds_{epoch}"), ED_stds)
            np.save(join(save_path, f"D2_means_{epoch}"), D2_means)
            np.save(join(save_path, f"D2_stds_{epoch}"), D2_stds)

    print(f"ED and D2 via method batches is saved for epochs {epochs}!")


"""
def npc_layerwise_2(post, alpha100, g100, total_images, epochs, reig=1):
    
    # computes the PCs of the layerwise neural representations
    # - post = 1: post-activation jacobian
    #        = 0: pre-activation jacobian

    # - input_idx: the index of the image
    

    global alpha, g
    global C_ls, C_dims, dqs, eigvals, eigvecs, ii, hidden_layer, hidden_layer_mean, ED, ED_means, net, C
    global hidden_layer, trainloader

    import scipy.io as sio
    import sys
    import torch    
    from NetPortal.models import ModelFactory
    from train_supervised import get_data, set_data

    post = int(post)
    total_images = int(total_images)
    assert post == 1 or post == 0, "No such option!"

    epochs = literal_eval(epochs) if isinstance(epochs,str) else epochs

    # storage of trained nets
    L = 10
    total_epoch = 650
    fcn = f"fc{L}"
    net_type = f"{fcn}_mnist_tanh"
    data_path = join(root_data, f"trained_mlps/fcn_grid/{fcn}_grid")

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    # load MNIST
    image_type = 'mnist'
    trainloader, _ = set_data(image_type, rshape=True)
    trainloader = DataLoader(trainloader, batch_size=len(trainloader), shuffle=False)

    # load nets and weights
    epoch = 0
    net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
    hidden_N = [784]*L + [10]
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # compute effective dimension for a single input manifold
    C_ls = []    # list of correlation matrices
    C_dims = np.zeros([len(net.sequential)])   # record dimension of correlation matrix
    l_remainder = 0 if post == 0 else 1
    l_start = l_remainder
    depth_idxs = list(range(l_start,len(net.sequential),2))
    L = len(depth_idxs)
    save_path = join(data_path, net_folder, f"ed-dq-method2_{post_dict[post]}_{reig_dict[reig]}")
    if not os.path.isdir(save_path): os.makedirs(save_path)
    with torch.no_grad():
        for epoch in epochs:
            l = 0
            ED_means = np.zeros([1,L])
            ED_stds = np.zeros([1,L])
            D2_means = np.zeros([1,L])
            D2_stds = np.zeros([1,L])
            # load nets and weights
            net_folder = f"{net_type}_id_stable{round(alpha,1)}_{round(g,2)}_epoch{total_epoch}_algosgd_lr=0.001_bs=1024_data_mnist"  
            hidden_N = [784]*L + [10]
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch,
                      "activation": 'tanh', "architecture": 'fc'}
            net = ModelFactory(**kwargs)
            # push the image through
            hidden_layer = torch.stack([trainloader.dataset[idx][0].reshape(784) for idx in range(total_images)])
            #hidden_layer = trainloader.dataset[:total_images][0].reshape((total_images,784))
            print(hidden_layer.shape)
            for lidx in range(len(net.sequential)):
                hidden_layer = net.sequential[lidx](hidden_layer)
                hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
                if lidx % 2 == l_remainder:     # pre or post activation
                    C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
                    ED = torch.trace(C)**2/torch.trace(C@C)
                    ED = ED.item()
                    ED_means[0,l] = ED
                    ED_stds[0,l] = ED**2
                    C_dims[lidx] = C.shape[0]
                    
                    eigvals, eigvecs = la.eigh(C)
                    ii = np.argsort(np.abs(eigvals))
                    dqs = np.zeros(len(ii))
                    for eidx, eigval in enumerate(eigvals):
                        dqs[eidx] = compute_dq(eigvecs[:,eidx],2)
                    D2_means[0,l] = dqs.mean()
                    D2_stds[0,l] = dqs.std()
                    
                    l += 1

            D2_means /= total_images
            D2_stds /= total_images

            # save data
            np.save(join(save_path, f"ED_means_{epoch}"), ED_means)
            np.save(join(save_path, f"ED_stds_{epoch}"), ED_stds)
            np.save(join(save_path, f"D2_means_{epoch}"), D2_means)
            np.save(join(save_path, f"D2_stds_{epoch}"), D2_stds)

    print(f"ED and D2 via method 2 is saved for {epochs}!")
    #plt.plot(range(1,L+1), ED_means[0,:])
    #plt.show()
"""


def submit(*args):
    from qsub import qsub, job_divider, project_ls
    input_idxs = str( list(range(1,50)) ).replace(" ", "")
    post = 0
    #epochs = [0,1] + list(range(50,651,50))
    epochs = [0,1,50,100,150,200,250,300,650]
    #epochs = [0,1]
    epochs_ls = [f"[{epoch}]" for epoch in epochs]
    pbs_array_data = [(post, alpha100, g100, epoch_ls)
                      for alpha100 in range(100, 201, 10)
                      for g100 in range(25,301,25)
                      #for alpha100 in [200]
                      #for g100 in [25,100,300]
                      #for epoch in [0,1] + list(range(50,651,50))
                      #for epoch in [0, 1, 50, 100, 150, 200, 650]
                      for epoch_ls in epochs_ls
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
