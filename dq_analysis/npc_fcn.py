import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import pandas as pd
import re
import sys
import torch
from ast import literal_eval
from os.path import join
from torch.utils.data import DataLoader
from tqdm import tqdm

#plt.switch_backend('agg')

sys.path.append(os.getcwd())
from constants import root_data
from utils_dnn import IPR, compute_dq, effective_dimension, setting_from_path
from train_supervised import get_data, set_data

dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                   if torch.cuda.is_available() else "cpu")

"""

This is for computing and saving the multifractal data of the (layerwise) Jacobian eigenvectors.

"""

global post_dict, reig_dict
post_dict = {0:'pre', 1:'post', 2:'all'}
reig_dict = {0:'l', 1:'r'}

# computes the angle between two vectors
def vec_angle(vec1, vec2):
    return np.arccos( np.dot(vec1,vec2)/( np.sqrt(np.dot(vec1,vec1) * np.dot(vec2,vec2)) ) )

# return the **TRAINING** dataset, i.e. mnist or gaussian_data
def get_dataset(image_type, batch_size,  **kwargs):
    if image_type.lower() == "mnist":
        trainloader, testloader = set_data(image_type, rshape=True)
        #trainloader = DataLoader(trainloader, batch_size=batch_size, shuffle=False)
        trainloader = DataLoader(trainloader, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
        return trainloader, testloader
    elif image_type.lower() == "gaussian":
        from UTILS.generate_gaussian_data import delayed_mixed_gaussian
        """
        num_train, num_test = kwargs.get("num_train"), kwargs.get("num_test")
        X_dim = kwargs.get("X_dim")
        Y_classes, X_clusters = kwargs.get("Y_classes"), kwargs.get("X_clusters")
        n_hold, final_time_point = kwargs.get("n_hold"), kwargs.get("final_time_point")    # not needed for feedforwards nets, always set to 0 for MLPs
        noise_sigma = kwargs.get("noise_sigma")
        cluster_seed, assignment_and_noise_seed = kwargs.get("cluster_seed"), kwargs.get("assignment_and_noise_seed")
        avg_magn = kwargs.get("avg_magn")
        min_sep = kwargs.get("min_sep")
        freeze_input = kwargs.get("freeze_input")

        class_datasets, centers, cluster_class_label = delayed_mixed_gaussian(num_train, num_test, X_dim, Y_classes, X_clusters, n_hold,
                                                                              final_time_point, noise_sigma,
                                                                              cluster_seed, assignment_and_noise_seed, avg_magn,                                        
                                                                              min_sep, freeze_input)
        """
        class_datasets, centers, cluster_class_label = delayed_mixed_gaussian(**kwargs)
        trainloader = class_datasets['train']
        trainloader = DataLoader(trainloader, batch_size=batch_size, shuffle=True)
        return trainloader, centers, cluster_class_label
    else:
        raise Exception("Only mnist or gaussian is allowed for MLPs!")
    
def npc_layerwise_ed(data_path, post, alpha100, g100, epochs):
    """

    computes the effective dimension of PCs of the layerwise neural representations,
    means and stds are taken across batches of images for each layer
    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian

    - input_idx: the index of the image
    
    """
    #global C_ls, C_dims, dqs, eigvals, eigvecs, ii, hidden_layer, ED, ED_means, ED_stds, net, trainloader,C
    #global input_mean

    import scipy.io as sio
    import sys
    import torch
    from sklearn.decomposition import PCA
    from NetPortal.models import ModelFactory

    post = int(post)
    alpha100, g100 = int(alpha100), int(g100)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs

    # getting network setting from path
    fcn, activation, epoch_last, net_folder = setting_from_path(data_path, alpha100, g100)
    L = int(fcn[2:])
    batch_size = 1000
    #batch_size = 500
    if "fcn_grid" in data_path or "gaussian_data" not in data_path:
        # load MNIST
        image_type = 'mnist'
        trainloader, _ = get_dataset(image_type, batch_size)
    else:
        image_type = "gaussian"
        gaussian_data_setting = pd.read_csv(join(data_path, "gaussian_data_setting.csv"))
        gaussian_data_kwargs = {}
        for param_name in gaussian_data_setting.columns:
            gaussian_data_kwargs[param_name] = gaussian_data_setting.loc[0,param_name]
        trainloader, _, _ = get_dataset(image_type, batch_size, **gaussian_data_kwargs)
             
    # load nets and weights
    epoch = 0
    hidden_N = [784]*L + [10]
    with_bias = False
    #is_weight_share = False
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "with_bias": with_bias,
              "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # compute effective dimension for a single input manifold
    C_dims = np.zeros([len(net.sequential)])   # record dimension of correlation matrix
    if post == 0 or post == 1:
        l_remainder = post
    else:
        l_remainder = None
    l_start = l_remainder
    if post == 0 or post == 1:
        depth_idxs = list(range(l_start,len(net.sequential),2))
        L = len(depth_idxs)
    else:
        L = len(net.sequential)
    #save_path = join(data_path, net_folder, f"ed-dq-batches_{post_dict[post]}_{reig_dict[reig]}")
    save_path = join(data_path, net_folder, f"ed-batches_{post_dict[post]}")
    if not os.path.isdir(save_path): os.makedirs(save_path)
    print("Computation starting.")
    with torch.no_grad():
        for epoch in tqdm(epochs):
            ED_means = np.zeros([1,L])
            ED_stds = np.zeros([1,L])
            # these record the average D2's of the top PC (i.e. the PC that corresponds to the largest eigenvalue of the covariance matrix)
            #D2_means = np.zeros([1,L])
            #D2_stds = np.zeros([1,L])
            # load nets and weights
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch,
                      "activation": 'tanh', "with_bias": with_bias,
                      "architecture": 'fc'}

            net = ModelFactory(**kwargs)
            # create batches and compute and mean/std over the batches
            for batch_idx, (hidden_layer, yb) in tqdm(enumerate(trainloader)):
                l = 0
                for lidx in range(len(net.sequential)):
                    hidden_layer = net.sequential[lidx](hidden_layer)
                    hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
                    if lidx % 2 == l_remainder or l_remainder == None:     # pre or post activation
                        # covariance matrix
                        C = (1/hidden_layer.shape[0])*torch.matmul(hidden_layer.T,hidden_layer) - (1/hidden_layer.shape[0]**2)*torch.matmul(hidden_layer_mean.T, hidden_layer_mean)
                        ED = torch.trace(C)**2/torch.trace(C@C)
                        ED = ED.item()

                        # directly use PCA
                        #hidden_layer_centered = hidden_layer - hidden_layer.mean(0)
                        #pca = PCA()
                        #pca.fit(hidden_layer_centered.detach().numpy())
                        #eigvals = pca.explained_variance_
                        #eigvecs = pca.components_

                        # SVD
                        #_, Rs, _ = np.linalg.svd(hidden_layer_centered.detach().numpy())
                        #eigvals = np.diag(Rs)

                        #ED = np.sum(eigvals)**2/np.sum(eigvals**2)

                        ED_means[0,l] += ED
                        ED_stds[0,l] += ED**2
                        if batch_idx == 0:
                            #C_dims[lidx] = C.shape[0]
                            C_dims[lidx] = hidden_layer.shape[1]
                        
                        # computation moved to npc_layerwise_d2()
                        # method 1
                        #eigvals, eigvecs = la.eigh(C)
                        #ii = np.argsort(np.abs(eigvals))[::-1]
                        #eigvals = eigvals[ii]   # reorder the eigenvalues based on its magnitudes
                        #eigvecs = eigvecs[:,ii] 
                        # dqs = np.zeros(len(ii))
                        
                        # method 2
                        #dqs = np.zeros(hidden_layer.shape[1])
                        #for eidx, eigval in enumerate(eigvals):
                        #    dqs[eidx] = compute_dq(eigvecs[eidx,:],2)
                        #D2_means[0,l] += dqs.mean()
                        #D2_stds[0,l] += dqs.std()
                        
                        l += 1

                        #plt.plot(np.log(np.abs(eigvals)), dqs, ".-"); plt.show()
                        #break

            print(f"Batches: {batch_idx}")
            batches = batch_idx + 1
            for l in range(ED_means.shape[1]):
                ED_means[0,l] /= batches
                ED_stds[0,l] = np.sqrt( ED_stds[0,l]/batches - ED_means[0,l]**2 )

            # save data
            np.save(join(save_path, f"ED_means_{epoch}"), ED_means)
            np.save(join(save_path, f"ED_stds_{epoch}"), ED_stds)

    print(f"ED via method batches is saved for epochs {epochs}!")


def hidden_layerwise_d2(data_path, post, alpha100, g100, epochs):
    """
    computes 1. the correlation dimension D2 of the layerwise neural representations corresponding to input images
        - data_path: /project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid

        - post = 1: post-activation jacobian
               = 0: pre-activation jacobian
               = 2: all layers including both pre- and post- activation
    """
    global C_ls, C_dims, dqs, hidden_layer, net, trainloader, C, dqs_hidden

    import scipy.io as sio
    import sys
    import torch
    from torch.utils.data import TensorDataset

    from NetPortal.models import ModelFactory
    from train_supervised import get_data, set_data
    from utils_dnn import D_q_all    

    post = int(post)
    alpha100, g100 = int(alpha100), int(g100)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    # load MNIST
    image_type = 'mnist'
    trainloader, _ = set_data(image_type, rshape=True)
    trainloader = DataLoader(trainloader, batch_size=len(trainloader), shuffle=False)
    #assert len(trainloader) == 1, "The computation of D_2 in npc_layerwise_d2() is not over the full-batch images!"

    # getting network setting from path
    fcn, activation, epoch_last, net_folder = setting_from_path(data_path, alpha100, g100)
    L = int(fcn[2:])
    batch_size = 60000
    if "fcn_grid" in data_path or "gaussian_data" not in data_path:
        # load MNIST
        image_type = 'mnist'
        trainloader, _ = get_dataset(image_type, batch_size)
    else:
        image_type = "gaussian"
        gaussian_data_setting = pd.read_csv(join(data_path, "gaussian_data_setting.csv"))
        gaussian_data_kwargs = {}
        for param_name in gaussian_data_setting.columns:
            gaussian_data_kwargs[param_name] = gaussian_data_setting.loc[0,param_name]
        trainloader, _, _ = get_dataset(image_type, batch_size, **gaussian_data_kwargs)

    # load nets and weights
    epoch = 0
    hidden_N = [784]*L + [10]
    with_bias = False
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "with_bias": with_bias,
              "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # compute effective dimension for a single input manifold
    C_dims = np.zeros([len(net.sequential)])   # record dimension of correlation matrix
    if post == 0 or post == 1:
        l_remainder = post
    else:
        l_remainder = None
    l_start = l_remainder
    if post == 0 or post == 1:
        depth_idxs = list(range(l_start,len(net.sequential),2))
        L = len(depth_idxs)
    else:
        L = len(net.sequential)
    dq_hidden_save_path = join(data_path, net_folder, f"dq_hidden-fullbatch_{post_dict[post]}")
    if not os.path.isdir(dq_hidden_save_path): os.makedirs(dq_hidden_save_path)
    print("Computation starting.")    
    with torch.no_grad():
        for epoch in tqdm(epochs):
            l = 0
            # load nets and weights
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch,
                      "activation": 'tanh', "with_bias": with_bias,
                      "architecture": 'fc'}
            net = ModelFactory(**kwargs)
            # compute the associated D2 quantities over the full batch
            hidden_layer = torch.squeeze(torch.stack([e[0] for e in trainloader]).to(dev))        
            for lidx in range(len(net.sequential)):
                hidden_layer = net.sequential[lidx](hidden_layer)                                
                if lidx % 2 == l_remainder or l_remainder == None:     # pre or post activation
                    # save data
                    dqs_hidden = D_q_all(hidden_layer.detach().numpy().T,2)
                    #quit()  # delete
                    np.save(join(dq_hidden_save_path, f"D2_{l}_{epoch}"), dqs_hidden)
                    l += 1

    print(f"D2 for hidden layers from full batch of images is saved for epochs {epochs}!")


def npc_layerwise_d2(data_path, post, alpha100, g100, epochs):
    """
    computes 1. the correlation dimension D2 of the layerwise neural representations' top eigenvectors
    2. saves the variance explained by each PC (eigvals)

    - post = 1: post-activation jacobian
           = 0: pre-activation jacobian
           = 2: all layers including both pre- and post- activation
    """
    from torch.utils.data import TensorDataset
    from sklearn.decomposition import PCA
    #from sklearn.preprocessing import StandardScaler

    global C_ls, C_dims, dqs, eigvals, eigvecs, hidden_layer, hidden_layer_centered, net, trainloader,C
    global PCs, dqs_npc, dqs_npd

    import scipy.io as sio
    import sys
    import torch
    from NetPortal.models import ModelFactory
    from train_supervised import get_data, set_data

    post = int(post)
    alpha100, g100 = int(alpha100), int(g100)
    assert post == 2 or post == 1 or post == 0, "No such option!"
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs

    # Extract numeric arguments.
    alpha, g = int(alpha100)/100., int(g100)/100.
    # load MNIST
    image_type = 'mnist'
    trainloader, _ = set_data(image_type, rshape=True)
    trainloader = DataLoader(trainloader, batch_size=len(trainloader), shuffle=False)
    #assert len(trainloader) == 1, "The computation of D_2 in npc_layerwise_d2() is not over the full-batch images!"

    # getting network setting from path
    fcn, activation, epoch_last, net_folder = setting_from_path(data_path, alpha100, g100)
    L = int(fcn[2:])
    batch_size = 60000
    if "fcn_grid" in data_path or "gaussian_data" not in data_path:
        # load MNIST
        image_type = 'mnist'
        trainloader, _ = get_dataset(image_type, batch_size)
    else:
        image_type = "gaussian"
        gaussian_data_setting = pd.read_csv(join(data_path, "gaussian_data_setting.csv"))
        gaussian_data_kwargs = {}
        for param_name in gaussian_data_setting.columns:
            gaussian_data_kwargs[param_name] = gaussian_data_setting.loc[0,param_name]
        trainloader, _, _ = get_dataset(image_type, batch_size, **gaussian_data_kwargs)

    # load nets and weights
    epoch = 0
    hidden_N = [784]*L + [10]
    with_bias = False
    kwargs = {"dims": hidden_N, "alpha": None, "g": None,
              "init_path": join(data_path, net_folder), "init_epoch": epoch,
              "activation": 'tanh', "with_bias": with_bias,
              "architecture": 'fc'}
    net = ModelFactory(**kwargs)

    # compute effective dimension for a single input manifold
    C_dims = np.zeros([len(net.sequential)])   # record dimension of correlation matrix
    if post == 0 or post == 1:
        l_remainder = post
    else:
        l_remainder = None
    l_start = l_remainder
    if post == 0 or post == 1:
        depth_idxs = list(range(l_start,len(net.sequential),2))
        L = len(depth_idxs)
    else:
        L = len(net.sequential)
    dq_npc_save_path = join(data_path, net_folder, f"dq_npc-fullbatch_{post_dict[post]}")
    dq_npd_save_path = join(data_path, net_folder, f"dq_npd-fullbatch_{post_dict[post]}")
    eigvals_save_path = join(data_path, net_folder, f"eigvals-fullbatch_{post_dict[post]}")
    if not os.path.isdir(dq_npc_save_path): os.makedirs(dq_npc_save_path)
    if not os.path.isdir(dq_npd_save_path): os.makedirs(dq_npd_save_path)
    if not os.path.isdir(eigvals_save_path): os.makedirs(eigvals_save_path)
    print("Computation starting.")    
    with torch.no_grad():
        for epoch in tqdm(epochs):
            l = 0
            # load nets and weights
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch,
                      "activation": 'tanh', "with_bias": with_bias,
                      "architecture": 'fc'}
            net = ModelFactory(**kwargs)
            # compute the associated D2 quantities over the full batch
            hidden_layer = torch.squeeze(torch.stack([e[0] for e in trainloader]).to(dev))        
            for lidx in range(len(net.sequential)):
                hidden_layer = net.sequential[lidx](hidden_layer)
                #hidden_layer_mean = torch.mean(hidden_layer,0)    # center the hidden representations
                if lidx % 2 == l_remainder or l_remainder == None:     # pre or post activation
                    # directly use PCA
                    hidden_layer_centered = hidden_layer - hidden_layer.mean(0)
                    if dev == "cpu":
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

                    # save data
                    np.save(join(dq_npc_save_path, f"D2_{l}_{epoch}"), dqs_npc)
                    np.save(join(dq_npd_save_path, f"D2_{l}_{epoch}"), dqs_npd)
                    np.save(join(eigvals_save_path, f"npc-eigvals_{l}_{epoch}"), eigvals)
                    l += 1

    print(f"D2 for the full batch of images is saved for epochs {epochs}!")


def class_separation_pca(data_path, alpha100, g100, epochs):
    #global trainloader, gaussian_data_kwargs, eigvecs, hidden_layer, targets_indices, X_pca
    #global centers, cluster_class_label
    #global net, hidden_N, targets
    global testloader, net

    from NetPortal.models import ModelFactory
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    alpha100, g100 = int(alpha100), int(g100)
    epochs = literal_eval(epochs) if isinstance(epochs, str) else epochs

    if "fcn_grid" in data_path or "gaussian_data" not in data_path:
        # load MNIST
        image_type = 'mnist'
        batch_size = 10000
        _, testloader = get_dataset(image_type, batch_size)
    else:
        image_type = "gaussian"
        gaussian_data_setting = pd.read_csv(join(data_path, "gaussian_data_setting.csv"))
        gaussian_data_kwargs = {}
        for param_name in gaussian_data_setting.columns:
            gaussian_data_kwargs[param_name] = gaussian_data_setting.loc[0,param_name]
        batch_size = int(gaussian_data_kwargs["num_train"])  # full batch
        trainloader, centers, cluster_class_label = get_dataset(image_type, batch_size, **gaussian_data_kwargs)

    # Network setting
    alpha, g = int(alpha100)/100., int(g100)/100.
    # load nets and weights
    fcn, activation, total_epoch, net_folder = setting_from_path(data_path, alpha100, g100)
    L = int(fcn[2:])    
    save_path = join(data_path, net_folder, "X_pca_test")
    if not os.path.exists(save_path): os.makedirs(save_path)  

    testloader = next(iter(testloader))
    targets = testloader[1].unique()

    hidden_N = [784]*L + [len(targets)]
    targets_indices = []
    # group in classes
    for cl_idx, cl in enumerate(targets):
        targets_indices.append(list(testloader[1] == cl))

    # number of PCs
    n_components = 3   
    print("Starting PCA!")         
    with torch.no_grad():
        for epoch_plot in epochs:        
            # load nets and weights
            with_bias = False
            kwargs = {"dims": hidden_N, "alpha": None, "g": None,
                      "init_path": join(data_path, net_folder), "init_epoch": epoch_plot,
                      "activation": 'tanh', "with_bias": with_bias,
                      "architecture": 'fc'}
            net = ModelFactory(**kwargs)
            #quit()  # delete

            hidden_layer = testloader[0]
            # create batches and compute and mean/std over the batches
            total_depth = len(net.sequential)
            for lidx in range(total_depth):
                hidden_layer = net.sequential[lidx](hidden_layer)                        
                # center data
                hidden_layer_centered = hidden_layer - hidden_layer.mean(0)
                pca = PCA(n_components)
                X_pca = pca.fit_transform(hidden_layer_centered.detach().numpy())
                # save data
                np.save(join(save_path, f"npc-depth={lidx}-epoch={epoch_plot}"), X_pca)   

    # save target indices
    np.save(join(save_path, "target_indices"), np.array(targets_indices))
    np.save(join(save_path, "target"), targets.detach().numpy())                


# functions: npc_layerwise_ed(), hidden_layerwise_d2(), npc_layerwise_d2()
def submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    input_idxs = str( list(range(1,50)) ).replace(" ", "")
    #post = 1
    #epochs = [0,1] + list(range(50,651,50))
    #epochs = [0,1,50,100,150,200,250,300,650]
    #epochs = [0,650]
    epochs = [0,1,50,100,650]
    post_ls = [0,1,2]
    perm_epoch, pbss_epoch = job_divider(epochs, 3)
    epochs_ls = [str(epoch_ls).replace(" ", "") for epoch_ls in pbss_epoch]
    #data_path = "/project/PDLAI/project2_data/trained_mlps/fc10_tanh_gaussian_data_Y_classes=10_X_clusters=120/"
    data_path = "/project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid/"
    pbs_array_data = [(data_path, post, alpha100, g100, epoch_ls)
                      for post in post_ls
                      #for alpha100 in range(100, 201, 10)
                      #for g100 in range(25,301,25)
                      for alpha100 in [120,200]
                      for g100 in [25,100,300]
                      for epoch_ls in epochs_ls
                      ]
    print(f"Total jobs: {len(pbs_array_data)}")
    print(epochs_ls)
    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    ncpus, ngpus, select = 1, 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)    
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             #path=join(root_data, "geometry_data", "jobs_all", "hidden_layerwise_d2"),
             path=join(root_data, "geometry_data", "jobs_all", args[0]),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             #mem='1GB')
             mem='2GB') 

# functions: class_separation_pca()
def pca_submit(*args):
    from qsub import qsub, job_divider, project_ls, command_setup
    #epochs = [0,650]
    epochs = [0,1] + list(range(50,651,50))
    perm_epoch, pbss_epoch = job_divider(epochs, 1)
    epochs_ls = [str(epoch_ls).replace(" ", "") for epoch_ls in pbss_epoch]
    data_path = "/project/PDLAI/project2_data/trained_mlps/fcn_grid/fc10_grid/"
    pbs_array_data = [(data_path, alpha100, g100, epoch_ls)
                      for alpha100 in [120,200]
                      for g100 in [25,100,300]
                      for epoch_ls in epochs_ls
                      ]
    print(f"Total jobs: {len(pbs_array_data)}")
    print(epochs_ls)
    perm, pbss = job_divider(pbs_array_data, len(project_ls))

    ncpus, ngpus, select = 1, 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)       
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join(root_data, "trained_mlps/PCA_jobs"),
             P=project_ls[pidx],
             ncpus=1,
             walltime='23:59:59',
             mem='3GB') 

# ----- preplot ----- (useless from here onwards)

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

    from qsub import qsub, job_divider, project_ls, command_setup
    ncpus, ngpus, select = 1, 1, 0
    singularity_path = "/project/phys_DL/built_containers/FaContainer_v2.sif"
    bind_path = "/project"
    command = command_setup(singularity_path, bind_path=bind_path)

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
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
